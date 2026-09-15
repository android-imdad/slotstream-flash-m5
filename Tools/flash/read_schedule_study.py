#!/usr/bin/env python3
"""Bounded exact read-scheduling screen; no inference speedup claim."""
import argparse
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import EvidenceError, atomic_json, fresh_output, harness_hashes, read_json, sha256
from receipts import require_terminal_sampling, validate_receipt_file
import benchmark
from engine_bench import wait_for_nominal


def decision(report, receipt):
    if report.get("format") != "slotstream-read-scheduling-component-v1" or report.get("check", {}).get("passed") is not True:
        raise EvidenceError("component checks incomplete")
    if type(report.get("sourceRegionBytes")) is not int or not 0 < report["sourceRegionBytes"] <= 256 << 20:
        raise EvidenceError("source region bound missing")
    if not report.get("readControls") or any(row.get("noCacheReturnCode") != 0 or row.get("readAheadReturnCode") != 0
                                             for row in report["readControls"]):
        raise EvidenceError("descriptor controls incomplete")
    if receipt.get("memory", {}).get("qualified") is not True or receipt["memory"].get("peak_bytes", math.inf) > 512_000_000:
        raise EvidenceError("component physical footprint exceeded or unqualified")
    eligible = all(report.get(key) == {"thermalState": "nominal", "lowPowerModeEnabled": False}
                   for key in ("conditionsBefore", "conditionsAfter"))
    for key in ("swapins", "swapouts"):
        before = receipt.get("vm", {}).get("before", {}).get(key)
        after = receipt.get("vm", {}).get("after", {}).get(key)
        eligible &= type(before) is int and type(after) is int and before >= 0 and before == after
    rows = []
    seen = set()
    pooled = []
    for case in report.get("cases", []):
        identity = (case.get("layer"), case.get("queueDepth"))
        if identity not in {(layer, depth) for layer in (0, 5, 22) for depth in (12, 32)} or identity in seen:
            raise EvidenceError("unknown or duplicate component case")
        seen.add(identity)
        if case.get("experts") != [0, 7, 19, 43, 79, 131, 211, 307, 401, 511]:
            raise EvidenceError("component expert coverage changed")
        if case.get("orders") != [["strided", "balanced"] if pair % 2 == 0 else ["balanced", "strided"] for pair in range(8)]:
            raise EvidenceError("component pair order/count changed")
        before, after = case.get("stridedSeconds", []), case.get("balancedSeconds", [])
        if len(before) != 8 or len(after) != 8 or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in before + after):
            raise EvidenceError("invalid component timings")
        digest = case.get("outputSHA256", "")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise EvidenceError("full output digest missing")
        improvements = [1 - b / a for a, b in zip(before, after)]
        rows.append({"layer": identity[0], "queue_depth": identity[1],
                     "strided_median_seconds": statistics.median(before),
                     "balanced_median_seconds": statistics.median(after),
                     "median_paired_reduction": statistics.median(improvements)})
        if identity[1] == 32: pooled.extend(improvements)
    if len(seen) != 6: raise EvidenceError("missing component cases")
    reduction = statistics.median(pooled)
    passes = reduction >= .10 and all(r["median_paired_reduction"] >= -.05 for r in rows if r["queue_depth"] == 32)
    return {"timing_eligible": eligible, "admit_engine_benchmark": eligible and passes,
            "disposition": "admitted_for_engine_benchmark" if eligible and passes else "rejected" if eligible else "timing_ineligible",
            "default_depth_median_paired_reduction": reduction if eligible else None,
            "cases": rows, "inference_speedup_qualified": False,
            "limits": "Bounded reader experiment; timings include expansion and staging, not isolated SSD latency."}


def run(binary, model, output):
    output = fresh_output(output)
    try:
        if benchmark.archive(binary, output / "archive"): raise EvidenceError("archive failed")
        frozen = harness_hashes()
        atomic_json(output / "protocol.json", {"harness_hashes": frozen, "pairs": 8,
            "minimum_default_depth_paired_reduction": .10, "maximum_per_layer_regression": .05,
            "native_binary_sha256": sha256(output / "archive/bin/slotstream")})
        atomic_json(output / "readiness.json", wait_for_nominal())
        destination = output / "launch"
        command = [str((output / "archive/bin/slotstream").resolve()), "read-scheduling-check",
                   "--model", str(model.resolve()), "--output", str((destination / "native").resolve())]
        if benchmark.launch(destination, .512, 180, command): raise EvidenceError("component launcher failed")
        receipt = validate_receipt_file(destination / "receipt.json")
        require_terminal_sampling(receipt)
        native = destination / "native/report.json"
        if read_json(destination / "native/completion.json") != {"complete": True, "report_sha256": sha256(native)}:
            raise EvidenceError("native completion mismatch")
        if harness_hashes() != frozen: raise EvidenceError("harness changed during component run")
        result = decision(read_json(native), receipt)
        result.update(native_report_sha256=sha256(native), receipt_sha256=sha256(destination / "receipt.json"),
                      protocol_sha256=sha256(output / "protocol.json"))
        atomic_json(output / "report.json", result)
        atomic_json(output / "completion.json", {"complete": True, "report_sha256": sha256(output / "report.json")})
        print(result, flush=True)
        return 0
    except Exception as error:
        atomic_json(output / "failure.json", {"error": str(error)})
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("binary", "model", "output"): parser.add_argument("--" + field, type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.binary, args.model, args.output))
