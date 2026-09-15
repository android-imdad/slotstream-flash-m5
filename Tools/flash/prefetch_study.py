#!/usr/bin/env python3
"""Causal SSD-prefetch screening against exact CLOCK demand replay.

One next-layer CPU staging batch; predictions never evict or publish GPU slots.
No simulated timing can establish an inference gain or accelerator execution.
"""
import argparse
from collections import Counter, deque
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import EvidenceError, atomic_json, fresh_output, read_json, sha256
from receipts import validate_receipt_file
from replay import CACHE_RECORD_BYTES, ClockCache, parse_trace, replay, validate_runtime_stats


def forecast(history, method):
    if not history:
        return []
    if method == "previous-token":
        return list(history[-1])
    if method != "recent8-frequency":
        raise EvidenceError("unknown prefetch method")
    counts = Counter(expert for row in history for expert in row)
    return sorted(counts, key=lambda expert: (-counts[expert], expert))[:10]


def screen(groups, capacity, source_bytes, method):
    if not groups or groups[0].tokens <= 1 or any(group.tokens != 1 for group in groups[1:]):
        raise EvidenceError("prefetch needs initial prefill and unit decode groups")
    if method not in ("previous-token", "recent8-frequency"):
        raise EvidenceError("unknown prefetch method")
    # Charge both original CPU staging and expanded insertion scratch, with no
    # credit for freed buffers. The GPU pool loses the corresponding slots.
    staging_bytes = 10 * (max(source_bytes) + CACHE_RECORD_BYTES)
    staging_slots = (staging_bytes + CACHE_RECORD_BYTES - 1) // CACHE_RECORD_BYTES
    candidate_capacity = capacity - staging_slots
    if candidate_capacity < 10:
        raise EvidenceError("prefetch staging leaves fewer than one layer's demand slots")
    cache = ClockCache(candidate_capacity, source_bytes)
    for layer, rows in enumerate(groups[0].layers):
        cache.request(layer, rows, 0)
    histories = [deque(maxlen=8) for _ in range(48)]
    pending = set()
    result = {"demand_miss_bytes": 0, "prefetched_bytes": 0, "useful_bytes": 0,
              "unused_bytes": 0, "peak_staging_source_bytes": 0,
              "predicted_batches": 0, "decode_groups": len(groups) - 1}
    position = groups[0].tokens
    for group_index, group in enumerate(groups[1:]):
        for layer, rows in enumerate(group.layers):
            demand = {(layer, expert) for expert in rows[0]}
            misses = demand - set(cache.slot_of)
            useful = pending & misses
            result["useful_bytes"] += len(useful) * source_bytes[layer]
            result["unused_bytes"] += len(pending - useful) * source_bytes[layer]
            actual = cache.request(layer, rows, position)
            result["demand_miss_bytes"] += actual.miss_bytes
            histories[layer].append(tuple(rows[0]))
            # Only past routed IDs are visible at this point. At layer 47 the
            # history for layer 0 includes this token, predicting the next one.
            target = (layer + 1) % 48
            final = group_index == len(groups) - 2 and layer == 47
            prediction = [] if final else forecast(histories[target], method)
            pending = {(target, expert) for expert in prediction if (target, expert) not in cache.slot_of}
            amount = len(pending) * source_bytes[target]
            result["prefetched_bytes"] += amount
            result["peak_staging_source_bytes"] = max(result["peak_staging_source_bytes"], amount)
            result["predicted_batches"] += bool(pending)
        position += 1
    if result["prefetched_bytes"] != result["useful_bytes"] + result["unused_bytes"]:
        raise EvidenceError("prefetch byte accounting does not close")
    matched = replay(groups, candidate_capacity, source_bytes)
    control = replay(groups, capacity, source_bytes)
    if result["demand_miss_bytes"] != matched["decode"]["miss_bytes"]:
        raise EvidenceError("prefetch changed the authoritative demand/cache path")
    denominator = control["decode"]["miss_bytes"]
    result.update({"potential_demand_coverage": result["useful_bytes"] / denominator if denominator else 0,
                   "extra_io_fraction": result["unused_bytes"] / denominator if denominator else 0,
                   "hypothetical_total_read_bytes": result["demand_miss_bytes"] + result["unused_bytes"],
                   "control_demand_miss_bytes": denominator,
                   "cache_budget_miss_delta_bytes": result["demand_miss_bytes"] - denominator,
                   "baseline_slots": capacity, "candidate_slots": candidate_capacity,
                   "reserved_staging_and_insertion_bytes": staging_bytes,
                   "native_prefetch_implemented": False, "speedup_qualified": False,
                   "timing_admission": "requires measured read/expansion/staging overlap; byte coverage alone cannot admit"})
    return result


def analyze(collection, output):
    output = fresh_output(output)
    try:
        manifest = read_json(collection / "collection.json")
        completion = read_json(collection / "completion.json")
        if (completion.get("collection_sha256") != sha256(collection / "collection.json")
                or completion.get("complete") is not True):
            raise EvidenceError("trace collection is incomplete or changed")
        sizes = manifest["model"]["layer_source_record_bytes"]
        results = []
        for pair in manifest["pairs"]:
            arm = pair["arms"]["traced"]
            root = collection / arm["path"]
            if not root.resolve().is_relative_to(collection.resolve()):
                raise EvidenceError("trace arm escapes collection")
            receipt = validate_receipt_file(root / "receipt.json")
            if (sha256(root / "receipt.json") != arm["artifacts"]["receipt.json"]
                    or not receipt["result"]["functional_success"]
                    or not {"stats.json", "router-trace.bin"}.issubset(receipt["artifacts"])):
                raise EvidenceError("trace/stats are not bound by a complete launcher receipt")
            stats = read_json(root / "stats.json")
            groups = parse_trace(root / "router-trace.bin")
            slots = stats["effective_pool_slots"]
            validate_runtime_stats(stats, groups, replay(groups, slots, sizes))
            reserved_slots = (10 * (max(sizes) + CACHE_RECORD_BYTES) + CACHE_RECORD_BYTES - 1) // CACHE_RECORD_BYTES
            if slots - reserved_slots < 640:
                raise EvidenceError("prefetch reservation cannot preserve the JANG 640-slot floor")
            results.append({"prompt_id": pair["prompt_id"], "receipt_sha256": sha256(root / "receipt.json"),
                "methods": {method: screen(groups, slots, sizes, method)
                            for method in ("previous-token", "recent8-frequency")}})
        report = {"format": "slotstream-causal-prefetch-screen-v1", "results": results,
                  "collection_sha256": sha256(collection / "collection.json"),
                  "source_sha256": sha256(Path(__file__)), "qualification": False,
                  "forecast_boundary": "after current layer demand, before next layer router; first decode token has no forecast",
                  "limits": "Historical trace replay only; no native asynchronous I/O, predictor, time saving or Neural Engine execution."}
        atomic_json(output / "report.json", report)
        atomic_json(output / "completion.json", {"report_sha256": sha256(output / "report.json"), "qualification": False})
        return 0
    except Exception as error:
        atomic_json(output / "failure.json", {"error": f"{type(error).__name__}: {error}"})
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(analyze(args.collection, args.output))
