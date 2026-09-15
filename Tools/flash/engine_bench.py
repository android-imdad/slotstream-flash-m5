#!/usr/bin/env python3
"""Matched JANG before/after benchmark using the saved upstream workload target.

Both JANG arms use fresh CLI processes, prefix cache off, 24 GB, greedy seed 42,
32K context and the saved prompts. Upstream uses another quant and a persistent
HTTP server with MTP/lookahead; it is a target, not the causal control arm.
"""
import argparse
import os
from pathlib import Path
import statistics
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import EvidenceError, atomic_json, fresh_output, harness_hashes, read_json, sha256, validate_build_identity
from receipts import require_terminal_sampling, validate_receipt_file
import benchmark
import cache_study
import widen_study
import thermal_readiness


def wait_for_nominal(*, stable_seconds=30, max_seconds=900, interval=5,
                     reader=thermal_readiness.observe, clock=time.monotonic, sleep=time.sleep):
    """Require a quiet window of nominal observations, outside model timing."""
    if not 0 < stable_seconds <= max_seconds or not 0 < interval <= 30:
        raise EvidenceError("invalid benchmark cooldown policy")
    started = clock()
    nominal_since = None
    observations = []
    previous = None
    while True:
        observation = reader()
        now = clock()
        observations.append({"elapsed_seconds": now - started, **observation})
        conditions = observation.get("conditions", {})
        ready = conditions.get("thermalState") == "nominal" and conditions.get("lowPowerModeEnabled") is False
        if conditions != previous:
            print("BENCH READINESS", conditions, flush=True)
            previous = dict(conditions)
        if ready:
            if nominal_since is None: nominal_since = now
            if now - nominal_since >= stable_seconds:
                return {"stable_seconds": stable_seconds, "interval_seconds": interval,
                        "elapsed_seconds": now - started, "observations": observations}
        else:
            nominal_since = None
        if now - started >= max_seconds:
            raise EvidenceError("benchmark cooldown did not reach nominal conditions")
        sleep(min(interval, max_seconds - (now - started)))


def load_baseline(path):
    document = read_json(path)
    hashes = read_json(path.parent / "SHA256SUMS.json")
    if hashes[path.name] != {"bytes": path.stat().st_size, "sha256": sha256(path)}:
        raise EvidenceError("saved baseline bytes changed")
    if (document["memory_target_decimal_gb"] != 24 or document["max_context_tokens"] != 32768
            or document["max_output_tokens"] != 128 or document["prefix_cache_enabled"] is not False
            or set(document["requests"]) != {"explanation", "coding", "reasoning"}):
        raise EvidenceError("unsupported baseline configuration")
    for request in document["requests"].values():
        if (len(request["messages"]) != 1 or request["messages"][0]["role"] != "user"
                or request["think"] is not False
                or request["options"] != {"temperature": 0, "seed": 42, "num_predict": 128}):
            raise EvidenceError("baseline prompt or decoding settings differ")
    return document


def arm(binary, model, text, policy, output):
    command = [str(binary), "run", "--model", str(model), "--memory-gb", "24", "--max-context", "32768",
        "--mtp", "off", "--vision", "off", "--prompt", text, "--max-tokens", "128", "--greedy",
        "--seed", "42", "--sample-footprint", "--stats-json", str(output / "stats.json")]
    if policy == "packed4-to6":
        command += ["--expert-widening", policy]
    readiness = wait_for_nominal()
    readiness_path = output.parent / (output.name + "-readiness.json")
    atomic_json(readiness_path, readiness)
    settling = benchmark.settle_before_model_launch()
    previous = os.environ.get("SLOTSTREAM_PREFIX_CACHE")
    os.environ["SLOTSTREAM_PREFIX_CACHE"] = "0"
    try:
        code = benchmark.launch(output, 24, 900, command)
    finally:
        if previous is None: os.environ.pop("SLOTSTREAM_PREFIX_CACHE", None)
        else: os.environ["SLOTSTREAM_PREFIX_CACHE"] = previous
    atomic_json(output / "settling.json", settling)
    receipt = validate_receipt_file(output / "receipt.json")
    require_terminal_sampling(receipt)
    if code or not receipt["result"]["functional_success"]:
        raise EvidenceError("matched benchmark arm did not finish")
    stats = read_json(output / "stats.json")
    if (stats["plan"]["target_gb"] != 24 or stats["plan"]["max_context_tokens"] != 32768
            or stats["plan"]["runtime_prefix_cache_enabled"] or stats["effective_mtp"]
            or stats["sampling"] != {"greedy": True, "seed": "42", "requested_max_tokens": "128"}
            or (policy == "packed4-to6" and stats.get("effective_expert_widening") != policy)
            or receipt["environment"].get("SLOTSTREAM_PREFIX_CACHE") != "0"):
        raise EvidenceError("benchmark effective controls differ from saved workload")
    eligible, reason = True, None
    try: widen_study.validate_timing_eligibility(stats, receipt)
    except EvidenceError as error: eligible, reason = False, str(error)
    numbers = stats["stats"]
    times = {key: widen_study._number(numbers.get(key), key) for key in
             ("decodeSeconds", "firstTokenSeconds", "firstTextSeconds", "prefillSeconds", "requestSeconds")}
    if times["decodeSeconds"] <= 0 or numbers["decodeTokens"] <= 0:
        raise EvidenceError("benchmark has no completed decode work")
    return stats, receipt, {"timing_eligible": eligible, "ineligible_reason": reason,
        "readiness_sha256": sha256(readiness_path), "cooldown_seconds": readiness["elapsed_seconds"],
        "decode_tokens_per_second": numbers["decodeTokens"] / times["decodeSeconds"],
        "decode_tokens": numbers["decodeTokens"], "timings": times,
        "peak_bytes": receipt["memory"]["peak_bytes"], "load_seconds": stats["load_seconds"]}


def run(reference, candidate, model, baseline, output):
    output = fresh_output(output)
    try:
        target = load_baseline(baseline)
        if any(key.startswith("SLOTSTREAM_") for key in os.environ):
            raise EvidenceError("ambient Slotstream controls are unsupported")
        model = model.resolve(strict=True)
        verified = cache_study.verified_model_revision(model)
        geometry = cache_study.derive_source_geometry(model, verification=verified)
        reference = reference.resolve(strict=True)
        identity, _ = validate_build_identity(reference, historical=True)
        if identity["binary_sha256"] != widen_study.REFERENCE_BINARY_SHA256:
            raise EvidenceError("benchmark requires the immutable pre-widening scalar reference")
        if benchmark.archive(candidate, output / "archive"):
            raise EvidenceError("benchmark candidate archive failed")
        candidate = output / "archive/bin/slotstream"
        frozen_build, _ = validate_build_identity(candidate)
        frozen_harness = harness_hashes()
        protocol = {"format": "slotstream-stage-benchmark-protocol-v1", "pairs": 3,
            "workloads": target["requests"], "baseline_sha256": sha256(baseline),
            "reference_binary_sha256": identity["binary_sha256"], "candidate_binary_sha256": sha256(candidate),
            "model_verification": verified, "memory_gb": 24, "context": 32768, "prefix_cache": False,
            "before": "immutable JANG scalar", "after": "current JANG packed4-to6; neuron oracle disabled",
            "readiness_policy": {"nominal_observation_seconds": 30, "poll_seconds": 5, "maximum_wait_seconds": 900,
                "stop_cohort_on_ineligible_arm": True},
            "harness_hashes": frozen_harness, "qualification": False}
        atomic_json(output / "protocol.json", protocol)
        rows = []
        for prompt_index, (name, request) in enumerate(target["requests"].items()):
            for pair in range(3):
                order = ["before", "after"] if (pair + prompt_index) % 2 == 0 else ["after", "before"]
                docs, receipts, results = {}, {}, {}
                for label in order:
                    path = output / name / f"pair-{pair:02d}" / label
                    print(f"BENCH START {name} pair={pair + 1} {label}", flush=True)
                    docs[label], receipts[label], results[label] = arm(reference if label == "before" else candidate,
                        model, request["messages"][0]["content"], "scalar" if label == "before" else "packed4-to6", path)
                    results[label].update(path=str(path.relative_to(output)), receipt_sha256=sha256(path / "receipt.json"), stats_sha256=sha256(path / "stats.json"))
                    print(f"BENCH PASS {name} {label} tok/s={results[label]['decode_tokens_per_second']:.3f}", flush=True)
                    if not results[label]["timing_eligible"]:
                        raise EvidenceError("timing-ineligible arm preserved; complete benchmark cohort invalidated")
                    if validate_build_identity(candidate)[0] != frozen_build or harness_hashes() != frozen_harness:
                        raise EvidenceError("benchmark source or harness changed")
                widen_study._exact_work(docs["before"], docs["after"])
                widen_study.compare_runtime_controls(docs["before"], receipts["before"], docs["after"], receipts["after"], left_known_scalar_reference=True)
                rows.append({"workload": name, "pair": pair, "order": order, "arms": results,
                             "exact_output_ids": docs["after"]["output_ids"]})
                atomic_json(output / "progress.json", {"rows": rows})
        if cache_study.derive_source_geometry(model, verification=verified) != geometry:
            raise EvidenceError("benchmark checkpoint changed")
        workloads = {}
        for name in target["requests"]:
            selected = [r for r in rows if r["workload"] == name]
            summary = {label: {"median_decode_tokens_per_second": statistics.median(r["arms"][label]["decode_tokens_per_second"] for r in selected),
                "median_first_text_seconds": statistics.median(r["arms"][label]["timings"]["firstTextSeconds"] for r in selected),
                "maximum_peak_bytes": max(r["arms"][label]["peak_bytes"] for r in selected)} for label in ("before", "after")}
            eligible = all(r["arms"][label]["timing_eligible"] for r in selected for label in ("before", "after"))
            summary.update(timing_eligible=eligible, baseline_target_tok_s=target["targets"][name]["median_decode_tokens_per_second"],
                median_paired_decode_time_reduction=statistics.median(1-r["arms"]["after"]["timings"]["decodeSeconds"]/r["arms"]["before"]["timings"]["decodeSeconds"] for r in selected) if eligible else None)
            workloads[name] = summary
        report = {"format": "slotstream-stage-benchmark-v1", "workloads": workloads, "rows": rows,
            "protocol_sha256": sha256(output / "protocol.json"), "qualification": False,
            "scope": "Three paired fresh-process CLI runs per saved workload; identical JANG output IDs. No neuron-skipping speed claim.",
            "baseline_limits": "Upstream is a different quant on a persistent HTTP server with MTP/lookahead. Its client-visible latency includes different boundaries."}
        atomic_json(output / "report.json", report)
        atomic_json(output / "completion.json", {"report_sha256": sha256(output / "report.json"), "complete": True})
        print(workloads, flush=True)
        return 0
    except Exception as error:
        atomic_json(output / "failure.json", {"error": f"{type(error).__name__}: {error}"})
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "candidate", "model", "baseline", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.reference, args.candidate, args.model, args.baseline, args.output))
