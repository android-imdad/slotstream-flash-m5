#!/usr/bin/env python3
"""Screen the existing expanded whole-expert layout on bounded JANG regions."""
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

LAYERS = [0, 5, 22]
EXPERTS = [0, 7, 19, 43, 79, 131, 211, 307, 401, 511]
SIZES = {0: 3_174_400, 5: 3_584_000, 22: 3_993_600}
RECORD = 4_300_800


def decision(report, receipt, *, representation="expanded"):
    if representation not in ("expanded", "source-native"):
        raise EvidenceError("unknown component representation")
    source_native = representation == "source-native"
    expected_format = "slotstream-source-native-component-v1" if source_native else "slotstream-whole-expert-component-v1"
    if (report.get("format") != expected_format
            or report.get("check", {}).get("passed") is not True
            or report.get("layers") != LAYERS or report.get("experts") != EXPERTS
            or report.get("queueDepth") != 32):
        raise EvidenceError("component geometry or checks incomplete")
    expected_artifact = sum(SIZES.values()) * 10 if source_native else RECORD * 30
    if report.get("originalRegionBytes") != sum(SIZES.values()) * 10 or report.get("artifactBytes") != expected_artifact:
        raise EvidenceError("source/artifact byte accounting changed")
    if not report.get("rawReadControls") or any(row.get("noCacheReturnCode") != 0 or row.get("readAheadReturnCode") != 0
                                              for row in report["rawReadControls"]):
        raise EvidenceError("raw descriptor controls incomplete")
    if receipt.get("memory", {}).get("qualified") is not True or receipt["memory"].get("peak_bytes", math.inf) > 512_000_000:
        raise EvidenceError("unqualified process footprint")
    eligible = all(report.get(key) == {"thermalState": "nominal", "lowPowerModeEnabled": False}
                   for key in ("conditionsBefore", "conditionsAfter"))
    for key in ("swapins", "swapouts"):
        before = receipt.get("vm", {}).get("before", {}).get(key)
        after = receipt.get("vm", {}).get("after", {}).get(key)
        eligible &= type(before) is int and type(after) is int and before >= 0 and before == after
    seen, rows = set(), []
    totals = {label: [0.] * 8 for label in ("raw", "whole")}
    for case in report.get("cases", []):
        identity = case.get("layer"), case.get("count")
        if identity not in {(layer, count) for layer in LAYERS for count in (1, 4, 10)} or identity in seen:
            raise EvidenceError("unknown or duplicate component case")
        seen.add(identity)
        layer, count = identity
        whole_record = SIZES[layer] if source_native else RECORD
        if case.get("rawSourceBytesPerCall") != SIZES[layer] * count or case.get("wholeSourceBytesPerCall") != whole_record * count:
            raise EvidenceError("per-call read bytes changed")
        if case.get("orders") != [["raw", "whole"] if (pair + LAYERS.index(layer)) % 2 == 0 else ["whole", "raw"] for pair in range(8)]:
            raise EvidenceError("pair count/order changed")
        raw, whole = case.get("rawSeconds", []), case.get("wholeSeconds", [])
        if len(raw) != 8 or len(whole) != 8 or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in raw + whole):
            raise EvidenceError("component timing is incomplete or invalid")
        digest = case.get("outputSHA256", "")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise EvidenceError("output byte digest missing")
        for pair in range(8):
            totals["raw"][pair] += raw[pair]
            totals["whole"][pair] += whole[pair]
        rows.append({"layer": layer, "count": count, "raw_median_seconds": statistics.median(raw),
                     "whole_median_seconds": statistics.median(whole),
                     "median_paired_reduction": statistics.median(1 - b / a for a, b in zip(raw, whole)),
                     "read_byte_increase_fraction": whole_record / SIZES[layer] - 1})
    if len(seen) != 9: raise EvidenceError("missing component cases")
    reduction = statistics.median(1 - b / a for a, b in zip(totals["raw"], totals["whole"]))
    passes = reduction >= .10 and all(row["median_paired_reduction"] >= -.05 for row in rows)
    return {"timing_eligible": eligible, "admit_engine_benchmark": eligible and passes,
            "disposition": "admitted_for_engine_benchmark" if eligible and passes else "rejected" if eligible else "timing_ineligible",
            "median_paired_total_reader_time_reduction": reduction if eligible else None,
            "cases": rows, "inference_speedup_qualified": False,
            "limits": "Repeated bounded sample; all reader work timed. No cold SSD or end-to-end inference claim."}


def run(binary, model, output, *, representation="expanded"):
    output = fresh_output(output)
    try:
        if representation not in ("expanded", "source-native"): raise EvidenceError("unknown component representation")
        if benchmark.archive(binary, output / "archive"): raise EvidenceError("archive failed")
        frozen = harness_hashes()
        atomic_json(output / "protocol.json", {"harness_hashes": frozen, "pairs_per_case": 8,
            "minimum_paired_total_reduction": .10, "maximum_per_case_regression": .05,
            "native_binary_sha256": sha256(output / "archive/bin/slotstream"),
            "layers": LAYERS, "experts": EXPERTS, "batch_sizes": [1, 4, 10], "queue_depth": 32, "representation": representation})
        atomic_json(output / "readiness.json", wait_for_nominal())
        destination = output / "launch"
        subcommand = "source-native-check" if representation == "source-native" else "whole-expert-check"
        command = [str((output / "archive/bin/slotstream").resolve()), subcommand,
                   "--model", str(model.resolve()), "--output", str((destination / "native").resolve())]
        if benchmark.launch(destination, .512, 180, command): raise EvidenceError("component launcher failed")
        receipt = validate_receipt_file(destination / "receipt.json")
        require_terminal_sampling(receipt)
        native = destination / "native/report.json"
        if read_json(destination / "native/completion.json") != {"complete": True, "report_sha256": sha256(native)}:
            raise EvidenceError("native completion mismatch")
        if harness_hashes() != frozen: raise EvidenceError("harness changed during component run")
        result = decision(read_json(native), receipt, representation=representation)
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
    parser.add_argument("--representation", choices=("expanded", "source-native"), default="expanded")
    args = parser.parse_args()
    raise SystemExit(run(args.binary, args.model, args.output, representation=args.representation))
