#!/usr/bin/env python3
"""Empirical causal-prefetch admission screen, not native prefetch execution."""
import argparse
from collections import deque
import copy
import math
import os
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import EvidenceError, atomic_json, fresh_output, harness_hashes, read_json, sha256, validate_build_identity
from receipts import validate_receipt_file, require_terminal_sampling
from replay import ClockCache, CACHE_RECORD_BYTES, parse_trace, replay, validate_runtime_stats
import benchmark
import cache_study
import prefetch_study
import widen_study
from engine_bench import wait_for_nominal

METHODS = ("previous-token", "recent8-frequency")
SIZES = {0: 3_174_400, 5: 3_584_000, 22: 3_993_600}
EXPERTS = [0, 7, 19, 43, 79, 131, 211, 307, 401, 511]


def positive(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise EvidenceError("invalid positive " + name)
    return value


def curves(report, receipt):
    if (report.get("format") != "slotstream-prefetch-reader-costs-v1" or report.get("policy") != "packed4-to6"
            or report.get("queueDepth") != 32 or report.get("experts") != EXPERTS
            or report.get("uniqueSourceBytes") != sum(SIZES.values()) * 10
            or report.get("check", {}).get("passed") is not True):
        raise EvidenceError("reader cost identity/coverage mismatch")
    if (receipt.get("memory", {}).get("qualified") is not True
            or receipt["memory"].get("peak_bytes", math.inf) > 512_000_000):
        raise EvidenceError("reader cost memory unqualified")
    if not report.get("readControls") or any(c.get("noCacheReturnCode") != 0 or c.get("readAheadReturnCode") != 0 for c in report["readControls"]):
        raise EvidenceError("reader descriptor controls incomplete")
    for key in ("conditionsBefore", "conditionsAfter"):
        if report.get(key) != {"thermalState": "nominal", "lowPowerModeEnabled": False}:
            raise EvidenceError("reader costs have ineligible thermal conditions")
    for key in ("swapins", "swapouts"):
        a = receipt.get("vm", {}).get("before", {}).get(key)
        b = receipt.get("vm", {}).get("after", {}).get(key)
        if type(a) is not int or type(b) is not int or a < 0 or a != b:
            raise EvidenceError("reader costs have ineligible paging observations")
    result = {size: {0: {"median": 0., "min": 0., "max": 0.}} for size in SIZES.values()}
    seen = set()
    for case in report.get("cases", []):
        layer, count = case.get("layer"), case.get("count")
        if layer not in SIZES or type(count) is not int or not 1 <= count <= 10 or (layer, count) in seen:
            raise EvidenceError("unknown or duplicate reader-cost case")
        seen.add((layer, count))
        if case.get("sourceBytes") != SIZES[layer] * count: raise EvidenceError("reader byte count mismatch")
        values = case.get("seconds", [])
        if len(values) != 8: raise EvidenceError("reader case lacks eight repetitions")
        for value in values: positive(value, "reader cost")
        digest = case.get("outputSHA256", "")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise EvidenceError("reader output digest missing")
        result[SIZES[layer]][count] = {"median": statistics.median(values), "min": min(values), "max": max(values)}
    if len(seen) != 30: raise EvidenceError("reader cost grid incomplete")
    return result


def cost_replay(groups, capacity, sizes, costs, method):
    if method not in METHODS or not groups or groups[0].tokens <= 1 or any(g.tokens != 1 for g in groups[1:]):
        raise EvidenceError("invalid causal replay input")
    reserve = 10 * (max(sizes) + CACHE_RECORD_BYTES)
    reserved_slots = (reserve + CACHE_RECORD_BYTES - 1) // CACHE_RECORD_BYTES
    candidate_slots = capacity - reserved_slots
    if candidate_slots < 10: raise EvidenceError("staging leaves no demand capacity")
    control, candidate = ClockCache(capacity, sizes), ClockCache(candidate_slots, sizes)
    for layer, rows in enumerate(groups[0].layers):
        control.request(layer, rows, 0)
        candidate.request(layer, rows, 0)
    histories = [deque(maxlen=8) for _ in sizes]
    pending = set()
    totals = {name: 0. for name in ("control_reader_cost", "charged_demand_cost", "remaining_demand_cost",
                                  "optimistic_remaining_cost", "prefetch_reader_cost")}
    useful_bytes = unused_bytes = miss_bytes = predicted_bytes = 0
    position = groups[0].tokens
    for group_index, group in enumerate(groups[1:]):
        for layer, rows in enumerate(group.layers):
            table = costs[sizes[layer]]
            demand = {(layer, expert) for expert in rows[0]}
            misses = demand - set(candidate.slot_of)
            useful = pending & misses
            count, saved = len(misses), len(useful)
            base = control.request(layer, rows, position)
            current = candidate.request(layer, rows, position)
            if current.misses != count: raise EvidenceError("demand path changed")
            totals["control_reader_cost"] += table[base.misses]["median"]
            full = table[count]["median"]
            remaining = table[count - saved]
            totals["charged_demand_cost"] += full
            # With no useful prediction, cost is unchanged. Useful predictions
            # may be ignored if a smaller batch has a noisier/higher cost.
            totals["remaining_demand_cost"] += min(full, remaining["median"]) if saved else full
            totals["optimistic_remaining_cost"] += (full * min(1., remaining["min"] / table[count]["max"])) if saved else full
            miss_bytes += current.miss_bytes
            useful_bytes += saved * sizes[layer]
            unused_bytes += len(pending - useful) * sizes[layer]
            histories[layer].append(tuple(rows[0]))
            target = (layer + 1) % len(sizes)
            final = group_index == len(groups) - 2 and layer == len(sizes) - 1
            prediction = [] if final else prefetch_study.forecast(histories[target], method)
            pending = {(target, expert) for expert in prediction if (target, expert) not in candidate.slot_of}
            predicted_bytes += len(pending) * sizes[target]
            totals["prefetch_reader_cost"] += costs[sizes[target]][len(pending)]["median"]
        position += 1
    if predicted_bytes != useful_bytes + unused_bytes: raise EvidenceError("prefetch accounting does not close")
    old = prefetch_study.screen(groups, capacity, sizes, method)
    if (old["demand_miss_bytes"], old["useful_bytes"], old["unused_bytes"], old["prefetched_bytes"]) != (miss_bytes, useful_bytes, unused_bytes, predicted_bytes):
        raise EvidenceError("cost replay differs from the frozen causal byte policy")
    return {**totals, "byte_screen": old, "candidate_slots": candidate_slots, "reserved_bytes": reserve}


def timing_screen(costs, decode_seconds, io_seconds):
    total = positive(decode_seconds, "decode seconds")
    io = positive(io_seconds, "reader seconds")
    if io > total: raise EvidenceError("reader scope exceeds decode; overlap controls unsupported")
    for key in ("remaining_demand_cost", "optimistic_remaining_cost", "prefetch_reader_cost"):
        value = costs[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise EvidenceError("invalid modeled cost")
    scale = io / positive(costs["control_reader_cost"], "baseline cost sum")
    non_io = total - io
    remaining = costs["remaining_demand_cost"] * scale
    optimistic = costs["optimistic_remaining_cost"] * scale
    prefetch = costs["prefetch_reader_cost"] * scale
    return {"decode_seconds": total, "measured_reader_seconds": io,
        "reader_calibration_factor": scale, "non_reader_overlap_budget_seconds": non_io,
        "estimated_prefetch_work_seconds": prefetch,
        "estimated_exposed_prefetch_seconds": max(0., prefetch - non_io),
        "free_prefetch_improvement_fraction": 1 - (non_io + remaining) / total,
        "optimistic_sample_range_fraction": 1 - (non_io + optimistic) / total,
        "aggregate_overlap_improvement_fraction": 1 - (remaining + max(non_io, prefetch)) / total}


def disposition(rows):
    if len(rows) != 3 or len({r["prompt_id"] for r in rows}) != 3: raise EvidenceError("three complete workloads required")
    result = {}
    for method in METHODS:
        values = [r["methods"][method]["timing"] for r in rows]
        for value in values:
            for key in ("aggregate_overlap_improvement_fraction", "optimistic_sample_range_fraction"):
                if type(value.get(key)) not in (int, float) or not math.isfinite(value[key]):
                    raise EvidenceError("invalid admission metric")
        if all(v["aggregate_overlap_improvement_fraction"] >= .10 for v in values):
            state = "promising_for_native_overlap_experiment"
        elif any(v["optimistic_sample_range_fraction"] < .10 for v in values):
            state = "rejected_by_empirical_screen"
        else:
            state = "inconclusive_cost_spread"
        result[method] = {"disposition": state, "native_prefetch_implemented": False,
                          "inference_speedup_qualified": False}
    return result


def same_geometry(left, right):
    # The checkout moved; only these two evidence-file aliases may resolve to
    # the same existing file. All identities, hashes and model fields still match.
    left, right = copy.deepcopy(left), copy.deepcopy(right)
    for value in (left, right):
        for key in ("verification_output", "summary_path"):
            value["verification"][key] = str(Path(value["verification"][key]).resolve(strict=True))
    return left == right


def historical_traces(collection, model):
    manifest = read_json(collection / "collection.json")
    complete = read_json(collection / "completion.json")
    if (manifest.get("format") != "slotstream-cache-collection-v1" or manifest.get("schema_version") != 1
            or manifest.get("complete") is not True or manifest.get("qualification") is not False
            or complete.get("complete") is not True or complete.get("collection_sha256") != sha256(collection / "collection.json")):
        raise EvidenceError("trace collection incomplete or changed")
    fixture = cache_study.load_fixture()
    cache_study.validate_cohort_shape(manifest["pairs"], fixture)
    if manifest["fixture_sha256"] != sha256(cache_study.FIXTURE): raise EvidenceError("trace fixture changed")
    identity, _ = validate_build_identity(Path(manifest["binary"]["path"]), historical=True)
    archive_path = Path(manifest["binary"]["archive_receipt"])
    archive_receipt = validate_receipt_file(archive_path, require_qualified=True)
    if (sha256(archive_path) != manifest["binary"]["archive_receipt_sha256"]
            or archive_receipt["identities"]["build_identity"] != identity):
        raise EvidenceError("historical archive receipt changed")
    if identity != manifest["binary"]["build_identity"] or identity["binary_sha256"] != manifest["binary"]["sha256"]:
        raise EvidenceError("historical trace binary identity changed")
    geometry = cache_study.derive_source_geometry(model, verification=cache_study.verified_model_revision(model))
    if not same_geometry(geometry, manifest["model"]): raise EvidenceError("original trace model identity changed")
    result = {}
    for pair in manifest["pairs"]:
        entry = pair["arms"]["traced"]
        root = (collection / entry["path"]).resolve(strict=True)
        if not root.is_relative_to(collection.resolve()): raise EvidenceError("trace path escaped collection")
        cache_study.validate_artifact_hashes(root, entry["artifacts"])
        receipt = validate_receipt_file(root / "receipt.json"); require_terminal_sampling(receipt)
        if not receipt["result"]["functional_success"] or not {"stats.json", "router-trace.bin"}.issubset(receipt["artifacts"]):
            raise EvidenceError("trace receipt lacks completed native artifacts")
        stats = read_json(root / "stats.json")
        groups = parse_trace(root / "router-trace.bin")
        validate_runtime_stats(stats, groups, replay(groups, stats["effective_pool_slots"], geometry["layer_source_record_bytes"]))
        result[pair["prompt_id"]] = (stats, receipt, groups)
    return fixture, geometry, result


def run(binary, model, collection, output):
    output = fresh_output(output)
    try:
        if any(key.startswith("SLOTSTREAM_") for key in os.environ): raise EvidenceError("ambient runtime controls unsupported")
        fixture, geometry, traces = historical_traces(collection, model.resolve(strict=True))
        if benchmark.archive(binary, output / "archive"): raise EvidenceError("candidate archive failed")
        binary = (output / "archive/bin/slotstream").resolve()
        frozen_build = validate_build_identity(binary)[0]
        frozen = harness_hashes()
        atomic_json(output / "protocol.json", {"format": "slotstream-prefetch-cost-screen-v1", "harness_hashes": frozen,
            "binary_sha256": sha256(binary), "collection_sha256": sha256(collection / "collection.json"),
            "model": geometry, "fixture_sha256": sha256(cache_study.FIXTURE), "fresh_control_repetitions": 3,
            "memory_gb": 14, "minimum_each_workload_improvement": .10,
            "optimism": "Free instantaneous forecasts/insertion; all non-reader decode time usable as one global overlap window. Observed cost extrema are sensitivity only, not physical bounds.",
            "scope": "Historical exact routes joined to fresh matching untraced packed controls; no 24 GB qualification."})
        atomic_json(output / "cost-readiness.json", wait_for_nominal())
        launch = output / "cost-launch"
        if benchmark.launch(launch, .512, 180, [str(binary), "prefetch-costs-check", "--model", str(model.resolve()),
                           "--output", str((launch / "native").resolve())]): raise EvidenceError("reader calibration failed")
        receipt = validate_receipt_file(launch / "receipt.json"); require_terminal_sampling(receipt)
        native = launch / "native/report.json"
        if read_json(launch / "native/completion.json") != {"complete": True, "report_sha256": sha256(native)}:
            raise EvidenceError("cost completion mismatch")
        cost_table = curves(read_json(native), receipt)
        observations = {p["id"]: [] for p in fixture["prompts"]}
        for round in range(3):
            ordered = fixture["prompts"][round:] + fixture["prompts"][:round]
            for prompt in ordered:
                label = prompt["id"]
                print(f"CONTROL {label} round={round + 1}", flush=True)
                atomic_json(output / f"{label}-{round}-readiness.json", wait_for_nominal())
                path = output / "controls" / label / f"round-{round}"
                stats, receipt, _ = widen_study._run_arm(binary, model.resolve(), prompt, "packed4-to6", path, "prefetch-cost-screen-v1")
                widen_study.validate_stats(stats, prompt, "packed4-to6")
                widen_study.validate_timing_eligibility(stats, receipt)
                prior, prior_receipt, groups = traces[label]
                widen_study._exact_work(prior, stats)
                original_receipt = copy.deepcopy(prior_receipt)
                original_receipt["environment"].pop("SLOTSTREAM_ROUTER_TRACE", None)
                widen_study.compare_runtime_controls(prior, original_receipt, stats, receipt, left_known_scalar_reference=True)
                if stats["optimizations"]["overlapResidentExperts"] or stats["optimizations"]["overlapSharedExpert"]:
                    raise EvidenceError("reader scopes overlap existing GPU work")
                validate_runtime_stats(stats, groups, replay(groups, stats["effective_pool_slots"], geometry["layer_source_record_bytes"]))
                observations[label].append({"path": str(path.relative_to(output)), "stats_sha256": sha256(path / "stats.json"),
                    "receipt_sha256": sha256(path / "receipt.json"), "decode_seconds": stats["stats"]["decodeSeconds"],
                    "io_seconds": stats["stats"]["decodeIOSeconds"]})
                atomic_json(output / "progress.json", observations)
                print(f"PASS {label} decode={stats['stats']['decodeSeconds']:.3f}s", flush=True)
                if harness_hashes() != frozen or validate_build_identity(binary)[0] != frozen_build:
                    raise EvidenceError("source/harness changed during screen")
        rows = []
        for prompt in fixture["prompts"]:
            label = prompt["id"]
            prior, _, groups = traces[label]
            methods = {}
            for method in METHODS:
                costs = cost_replay(groups, prior["effective_pool_slots"], geometry["layer_source_record_bytes"], cost_table, method)
                timing_rows = [timing_screen(costs, o["decode_seconds"], o["io_seconds"]) for o in observations[label]]
                timing = {key: statistics.median(row[key] for row in timing_rows) for key in timing_rows[0]}
                methods[method] = {"costs": costs, "timing": timing, "per_run_timing": timing_rows}
            rows.append({"prompt_id": label, "methods": methods})
        if cache_study.derive_source_geometry(model.resolve(), verification=cache_study.verified_model_revision(model.resolve())) != geometry:
            raise EvidenceError("model changed during study")
        report = {"format": "slotstream-prefetch-cost-screen-v1", "rows": rows, "decision": disposition(rows),
            "controls": observations, "cost_report_sha256": sha256(native), "protocol_sha256": sha256(output / "protocol.json"),
            "limits": "Empirical 14 GB admission screen. Generous overlap and cost sample sensitivity do not prove a physical bound or an actual native speedup. No fresh 24 GB claim."}
        atomic_json(output / "report.json", report)
        atomic_json(output / "completion.json", {"complete": True, "report_sha256": sha256(output / "report.json")})
        print(report["decision"], flush=True)
        return 0
    except Exception as error:
        atomic_json(output / "failure.json", {"error": f"{type(error).__name__}: {error}"})
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("binary", "model", "collection", "output"): parser.add_argument("--" + field, type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.binary, args.model, args.collection, args.output))
