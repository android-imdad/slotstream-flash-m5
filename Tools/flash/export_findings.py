#!/usr/bin/env python3
"""Export portable, hash-bound findings without publishing local machine paths.

This reads existing completed evidence; it launches no model and creates no new
performance measurement. Full original archives remain unchanged under .build.
"""
import argparse
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import EvidenceError, atomic_json, read_json, sha256

ROOT = Path(__file__).resolve().parents[2]


def export(runs, output):
    if output.exists(): raise EvidenceError("public export already exists")
    sources = []
    def source(path):
        sources.append({"path": str(path.resolve().relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)})
        return read_json(path)
    def completed(name):
        root = runs / name
        report = source(root / "report.json")
        marker = source(root / "completion.json")
        if marker.get("report_sha256") != sha256(root / "report.json"):
            raise EvidenceError("completion mismatch: " + name)
        if "complete" in marker and marker["complete"] is not True: raise EvidenceError("incomplete study")
        return report

    baseline = source(runs / "baseline-summary-final.json")
    if baseline["qualification"]["baseline_prerequisite_complete"] is not True: raise EvidenceError("baseline incomplete")
    verified = baseline["model_verification"]
    if verified["verified"] is not True or verified["observed_exit_code"] != 0:
        raise EvidenceError("original checkpoint verification failed")
    verify_output = runs / verified["stdout"]
    if sha256(verify_output) != verified["stdout_sha256"]: raise EvidenceError("checkpoint verification output changed")
    sources.append({"path": str(verify_output.resolve().relative_to(ROOT)), "bytes": verify_output.stat().st_size, "sha256": sha256(verify_output)})

    m5 = source(runs / "m5-summary-final.json")
    private = m5["private_dispatch"]
    private_report = runs / private["path"]
    if sha256(private_report) != private["sha256"]: raise EvidenceError("M5 dispatch report changed")
    source(private_report)
    dispatch = [{k: row[k] for k in ("case", "kernel", "gpu_eval_and_numerical_report_passed")}
                for row in private["cases"]]
    if not all(row["gpu_eval_and_numerical_report_passed"] for row in dispatch): raise EvidenceError("dispatch incomplete")

    widening = completed("engine-benchmark-24gb-cooled-20260915")
    widening_protocol = source(runs / "engine-benchmark-24gb-cooled-20260915/protocol.json")
    if len(widening["rows"]) != 9 or not all(v["timing_eligible"] for v in widening["workloads"].values()):
        raise EvidenceError("widening cohort incomplete")
    oracle = []
    for retained in (10, 8, 6):
        r = completed(f"oracle-campaign-v2-20260915/retained-{retained:02d}")
        if not r["complete"]: raise EvidenceError("oracle cohort incomplete")
        oracle.append({k: r[k] for k in ("retained_blocks", "positions", "pooled", "teacher_forced_pass", "exact_dense_parity")})
    stop = source(runs / "oracle-campaign-v2-20260915/operator-decision.json")
    window = completed("window-screen-terminal")
    scheduling = completed("read-scheduling-20260915")
    whole = completed("whole-expert-20260915")
    native = completed("source-native-20260915")
    prefetch = completed("prefetch-cost-20260915")
    prefetch_verification = source(runs / "prefetch-cost-20260915/verification.json")
    if prefetch_verification["report_sha256"] != sha256(runs / "prefetch-cost-20260915/report.json"):
        raise EvidenceError("prefetch verification mismatch")
    controls = {}
    for label, rows in prefetch["controls"].items():
        if len(rows) != 3: raise EvidenceError("control cohort incomplete")
        docs = []
        for row in rows:
            p = runs / "prefetch-cost-20260915" / row["path"] / "stats.json"
            if sha256(p) != row["stats_sha256"]: raise EvidenceError("control stats changed")
            docs.append(source(p))
        controls[label] = {"repetitions": 3, "median_decode_tokens_per_second": statistics.median(d["stats"]["decodeTokens"] / d["stats"]["decodeSeconds"] for d in docs)}
    public = {"format": "slotstream-public-flash-findings-v1", "date": "2026-09-15",
        "scope": "Portable extract from completed local reports, not a new inference run. Source hashes refer to unchanged full local evidence; binaries and full captures are not included in Git.",
        "machine_record": "records/machines/local-m5-max-48gb",
        "checkpoint": {"name": "JANG_6S", "revision": verified["revision"], "full_download_verified": True},
        "m5": {"conclusion": m5["conclusion"], "dispatch": dispatch, "production_equivalent_build": private["production_equivalent_build"]},
        "widening": {"workloads": widening["workloads"], "pairs_per_workload": 3, "total_arms": 18,
            "memory_target_gb": 24, "context_tokens": 32768, "greedy_seed": 42, "max_output_tokens": 128,
            "prefix_cache_enabled": False, "mtp_enabled": False, "vision_enabled": False,
            "candidate_binary_sha256": widening_protocol["candidate_binary_sha256"], "qualification": False},
        "oracle": {"cohorts": oracle, "incomplete_retained": stop["incomplete_retained"], "untested_retained": stop["untested_retained"],
                   "partial_mask_enabled": stop["partial_mask_enabled"], "predictor_advancement": stop["predictor_advancement"]},
        "window": window["decision"],
        "loading_screens": {"balanced_scheduling": {"disposition": scheduling["disposition"], "reader_time_reduction_fraction": scheduling["default_depth_median_paired_reduction"]},
            "expanded_layout": {"disposition": whole["disposition"], "reader_time_reduction_fraction": whole["median_paired_total_reader_time_reduction"]},
            "source_native_layout": {"disposition": native["disposition"], "reader_time_reduction_fraction": native["median_paired_total_reader_time_reduction"]}},
        "prefetch": {"memory_target_gb": 14, "decision": prefetch["decision"],
            "estimates": [{"prompt_id": row["prompt_id"], "methods": {name: value["timing"] for name, value in row["methods"].items()}} for row in prefetch["rows"]],
            "control_measurements": controls, "maximum_control_peak_bytes": prefetch_verification["maximum_control_peak_bytes"],
            "cost_peak_bytes": prefetch_verification["cost_peak_bytes"], "model_limit": "Empirical estimates with generous overlap; neither native speedup nor physical bound."},
        "source_artifacts": sources}
    # Whitelisting above is deliberate. Never export full receipts/commands or
    # source dictionaries that can carry local usernames, paths or host IDs.
    import json
    if "/Users/" in json.dumps(public) or "/Volumes/" in json.dumps(public):
        raise EvidenceError("private filesystem path in portable export")
    atomic_json(output, public)
    print(str(output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT / ".build/flash/runs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.runs, args.output)
