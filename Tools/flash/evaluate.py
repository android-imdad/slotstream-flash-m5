#!/usr/bin/env python3
"""Freeze and run complete, serial, development-only neuron oracle cohorts."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from common import EvidenceError, atomic_json, fresh_output, harness_hashes, read_json, sha256, validate_build_identity
from receipts import validate_receipt_file
import benchmark
import calibrate
import capture
import cache_study
import oracle


def validate_freeze(root):
    frozen = read_json(root / "frozen.json")
    if read_json(root / "freeze-completion.json") != {"frozen_sha256": sha256(root / "frozen.json")}:
        raise EvidenceError("oracle freeze completion differs")
    if (frozen["format"] != "slotstream-oracle-campaign-v1" or frozen["split"] != "development"
            or frozen["retained_order"] != list(calibrate.RETAINED)
            or frozen["harness_hashes"] != harness_hashes()):
        raise EvidenceError("oracle frozen policy or harness changed")
    for path, digest in frozen["inputs"].items():
        if sha256(Path(path)) != digest:
            raise EvidenceError("oracle frozen input changed: " + path)
    if cache_study.derive_source_geometry(Path(frozen["model_geometry"]["model_path"]),
            verification=frozen["model_verification"]) != frozen["model_geometry"]:
        raise EvidenceError("oracle checkpoint source identity changed")
    binary = root / "archive/bin/slotstream"
    build, _ = validate_build_identity(binary)
    if build != frozen["build_identity"]:
        raise EvidenceError("oracle candidate source or executable changed")
    for entries in frozen["configs"].values():
        for item in entries:
            if sha256(root / item["path"]) != item["sha256"]:
                raise EvidenceError("oracle predeclared configuration changed")
    return frozen


def freeze(suite, reference, study, norms, binary, output):
    import metrics
    output = fresh_output(output)
    try:
        if any(key.startswith("SLOTSTREAM_") for key in os.environ):
            raise EvidenceError("ambient Slotstream controls are unsupported")
        reference = reference.resolve(strict=True)
        index, rows = metrics.validate_cohort(reference)
        if (index["kind"] != "full-study" or index["split"] != "development"
                or index["suite"]["sha256"] != sha256(suite)):
            raise EvidenceError("oracle requires the complete frozen development reference")
        score = read_json(study)
        if (read_json(study.parent / "completion.json") != {"study_sha256": sha256(study), "qualification": False}
                or score.get("cohort_sha256") != sha256(reference / "index.json")
                or score.get("suite_sha256") != sha256(suite)
                or score.get("positions") != len(rows)
                or score.get("retained_order") != list(calibrate.RETAINED)):
            raise EvidenceError("oracle study does not bind the development reference")
        norm_manifest, norm_files = calibrate.validate_norms(norms)
        if score["norm_manifest_sha256"] != sha256(norms / "manifest.json"):
            raise EvidenceError("oracle study binds another norm asset")
        export = norms.parent.parent
        export_completion = read_json(export / "completion.json")
        receipt = validate_receipt_file(export / "run/receipt.json")
        if (export_completion.get("manifest_sha256") != sha256(norms / "manifest.json")
                or export_completion.get("receipt_sha256") != sha256(export / "run/receipt.json")
                or not receipt["result"]["functional_success"] or receipt["memory"]["target_gb_decimal"] != 1
                or not receipt["memory"]["qualified"] or export_completion.get("scope") != "full"):
            raise EvidenceError("norm asset lacks a complete bounded export receipt")
        norm_build, paths = validate_build_identity(export / "archive/bin/slotstream", historical=True)
        if norm_manifest["provenance"] != {"slotstream": norm_build["binary_sha256"],
                "mlx.metallib": norm_build["metallib_sha256"], "build-identity.json": sha256(paths["identity"]),
                "build-source.tar.gz": norm_build["source_archive_sha256"]}:
            raise EvidenceError("norm exporter provenance differs")
        if benchmark.archive(binary, output / "archive"):
            raise EvidenceError("oracle archive failed")
        identity, _ = validate_build_identity(output / "archive/bin/slotstream")
        model_paths = {str(Path(shard["capture"]["model"]).resolve()) for shard in index["shards"]}
        if len(model_paths) != 1:
            raise EvidenceError("oracle reference crosses checkpoints")
        model = Path(next(iter(model_paths)))
        verification = cache_study.verified_model_revision(model)
        geometry = cache_study.derive_source_geometry(model, verification=verification)
        inputs = {str(path.resolve()): sha256(path) for path in
                  [suite, reference / "index.json", study, norms / "manifest.json", *norm_files.values()]}
        configs = {}
        tokenized = ROOT / index["tokenized"]["path"]
        for retained in calibrate.RETAINED:
            entries = []
            for ordinal, shard in enumerate(index["shards"]):
                manifest = tokenized / "corpus" / shard["path"]
                path = output / "configs" / f"retain-{retained:02d}-shard-{ordinal:03d}.json"
                atomic_json(path, oracle.configuration(norms, study, suite, manifest, retained))
                oracle.validate_configuration(path, manifest)
                entries.append({"path": str(path.relative_to(output)), "sha256": sha256(path),
                                "manifest": str(manifest.resolve()), "positions": shard["positions"]})
            configs[str(retained)] = entries
        frozen = {"format": "slotstream-oracle-campaign-v1", "split": "development",
            "retained_order": list(calibrate.RETAINED), "reference": str(reference), "positions": len(rows),
            "build_identity": identity, "inputs": inputs, "configs": configs, "harness_hashes": harness_hashes(),
            "model_verification": verification, "model_geometry": geometry,
            "qualification": False, "predictor_advancement": False}
        atomic_json(output / "frozen.json", frozen)
        atomic_json(output / "freeze-completion.json", {"frozen_sha256": sha256(output / "frozen.json")})
        print("ORACLE FROZEN", len(rows), "positions", len(index["shards"]), "shards", flush=True)
        return 0
    except Exception as error:
        atomic_json(output / "failure.json", {"error": f"{type(error).__name__}: {error}"})
        print(str(error), file=sys.stderr)
        return 1


def quality_pass(pooled, categories, thresholds):
    def passed(value):
        return (value["meanKL"] <= thresholds["meanKL"] and value["p99KL"] <= thresholds["p99KL"]
                and value["top1Agreement"] >= thresholds["top1Agreement"] and value["pplRatio"] <= thresholds["pplRatio"])
    return passed(pooled) and all(passed(value) for value in categories.values())


def completed_report(campaign, retained, frozen, *, verify_artifacts=False):
    import metrics
    root = campaign / f"retained-{retained:02d}"
    report = read_json(root / "report.json")
    if (read_json(root / "completion.json") != {"report_sha256": sha256(root / "report.json"), "complete": True}
            or report.get("format") != "slotstream-oracle-quality-v1"
            or report.get("frozen_sha256") != sha256(campaign / "frozen.json")
            or report.get("retained_blocks") != retained or report.get("split") != "development"
            or report.get("complete") is not True or report.get("positions") != frozen["positions"]
            or report.get("thresholds") != metrics.APPROX
            or sha256(root / "rows.json") != report.get("rows_sha256")
            or len(report.get("evidence", [])) != len(frozen["configs"][str(retained)])):
        raise EvidenceError("oracle completed cohort is missing, changed or belongs to another campaign")
    rows = read_json(root / "rows.json")["rows"]
    if len(rows) != frozen["positions"]:
        raise EvidenceError("oracle metric row coverage changed")
    pooled = metrics.aggregate(rows)
    categories = {name: metrics.aggregate([r for r in rows if r["category"] == name])
                  for name in sorted({r["category"] for r in rows})}
    if (report["pooled"] != pooled or report["categories"] != categories
            or report["teacher_forced_pass"] != quality_pass(pooled, categories, metrics.APPROX)
            or report["exact_dense_parity"] is not (retained == 10)):
        raise EvidenceError("oracle decision or metrics disagree with completed rows")
    if verify_artifacts:
        for ordinal, (entry, config) in enumerate(zip(report["evidence"], frozen["configs"][str(retained)])):
            if entry["ordinal"] != ordinal or entry["path"] != f"shard-{ordinal:03d}":
                raise EvidenceError("oracle evidence order changed")
            path = root / entry["path"]
            receipt = validate_receipt_file(path / "receipt.json")
            if (sha256(path / "receipt.json") != entry["receipt_sha256"]
                    or sha256(path / "native/report.json") != entry["report_sha256"]
                    or entry["config_sha256"] != config["sha256"]
                    or receipt["identities"]["executable"]["sha256"] != frozen["build_identity"]["binary_sha256"]
                    or not receipt["result"]["functional_success"] or not receipt["memory"]["qualified"]):
                raise EvidenceError("oracle recorded payload or executable changed")
    return report


def run(campaign, retained):
    import metrics
    import numpy as np
    campaign = campaign.resolve(strict=True)
    output = None
    try:
        if type(retained) is not int or retained not in calibrate.RETAINED:
            raise EvidenceError("unsupported retained count")
        frozen = validate_freeze(campaign)
        for previous in calibrate.RETAINED[:calibrate.RETAINED.index(retained)]:
            prior_report = completed_report(campaign, previous, frozen)
            if previous == 10 and prior_report.get("exact_dense_parity") is not True:
                raise EvidenceError("dense control did not preserve exact reference output")
        output = fresh_output(campaign / f"retained-{retained:02d}")
        reference, refs = metrics.validate_cohort(Path(frozen["reference"]))
        candidate = campaign / "archive/bin/slotstream"
        all_rows, evidence, seen = [], [], set()
        for ordinal, (shard, config) in enumerate(zip(reference["shards"], frozen["configs"][str(retained)])):
            # Read and check frozen inputs before every model launch. A failed
            # shard invalidates this entire predeclared cohort; no replacements.
            if harness_hashes() != frozen["harness_hashes"]:
                raise EvidenceError("oracle harness changed during cohort")
            config_path = campaign / config["path"]
            if sha256(config_path) != config["sha256"]:
                raise EvidenceError("oracle configuration changed during cohort")
            model = Path(shard["capture"]["model"]).resolve(strict=True)
            if cache_study.derive_source_geometry(model, verification=frozen["model_verification"]) != frozen["model_geometry"]:
                raise EvidenceError("oracle checkpoint changed before capture")
            original_root = ROOT / shard["capture"]["capturePath"]
            original = read_json(original_root / "native/report.json")
            if sha256(original_root / "native/report.json") != shard["capture"]["reportSHA256"]:
                raise EvidenceError("reference capture report changed")
            print(f"ORACLE START retained={retained} shard={ordinal + 1}/{len(reference['shards'])}", flush=True)
            destination = output / f"shard-{ordinal:03d}"
            report, receipt = oracle.run_native(candidate, model, Path(config["manifest"]), config_path, destination)
            oracle.compare_controls(original, report)
            original_receipt = read_json(original_root / "receipt.json")
            if original_receipt["environment"] != receipt["environment"]:
                raise EvidenceError("oracle launcher environment changed")
            if retained == 10 and capture._capture_identity(original) != capture._capture_identity(report):
                raise EvidenceError("dense-ten logits, ordered routes, state or token IDs differ")
            old_positions = {(d["id"], p["position"]): p for d in original["documents"] for p in d["positions"]}
            for doc in report["documents"]:
                for pos in doc["positions"]:
                    old = old_positions[(doc["id"], pos["position"])]
                    if oracle.state_structure(old["state"]) != oracle.state_structure(pos["state"]):
                        raise EvidenceError("oracle retained-state structure or positions differ")
            for artifact in report["files"]:
                if artifact["category"] != "logits":
                    continue
                key = (ordinal, artifact["document_id"], artifact["token_position"])
                if key in seen or key not in refs:
                    raise EvidenceError("duplicate or unexpected oracle position")
                seen.add(key)
                ref = refs[key]
                if ref["inputID"] != artifact["input_id"]:
                    raise EvidenceError("oracle input position mismatch")
                result = metrics.compare_row(np.fromfile(ref["path"], dtype="<f4"),
                    np.fromfile(destination / "native" / artifact["name"], dtype="<f4"), ref["nextTokenID"])
                result.update({"documentID": ref["documentID"], "position": ref["position"], "category": ref["category"],
                               "inputID": ref["inputID"], "nextTokenID": ref["nextTokenID"]})
                all_rows.append(result)
            evidence.append({"ordinal": ordinal, "path": str(destination.relative_to(output)),
                "receipt_sha256": sha256(destination / "receipt.json"),
                "report_sha256": sha256(destination / "native/report.json"),
                "config_sha256": sha256(config_path), "peak_bytes": receipt["memory"]["peak_bytes"]})
            atomic_json(output / "progress.json", {"completed_shards": len(evidence), "positions": len(all_rows), "evidence": evidence})
            print(f"ORACLE PASS retained={retained} shard={ordinal + 1} positions={len(all_rows)} peak={receipt['memory']['peak_bytes']}", flush=True)
        if seen != set(refs) or len(all_rows) != frozen["positions"]:
            raise EvidenceError("oracle cohort does not cover the complete frozen reference")
        validate_freeze(campaign)
        pooled = metrics.aggregate(all_rows)
        categories = {name: metrics.aggregate([r for r in all_rows if r["category"] == name]) for name in reference["expectedCategories"]}
        passed = quality_pass(pooled, categories, metrics.APPROX)
        report = {"format": "slotstream-oracle-quality-v1", "retained_blocks": retained,
            "split": "development", "positions": len(all_rows), "complete": True,
            "frozen_sha256": sha256(campaign / "frozen.json"), "evidence": evidence,
            "exact_dense_parity": retained == 10, "pooled": pooled, "categories": categories,
            "bootstrap": metrics.bootstrap_documents(all_rows), "thresholds": metrics.APPROX,
            "teacher_forced_pass": passed, "predictor_advancement": False, "qualification": False,
            "limits": "Dense-load diagnostic. Free generation, learned predictor, sparse SSD reads and Neural Engine execution remain unqualified."}
        atomic_json(output / "rows.json", {"rows": all_rows})
        report["rows_sha256"] = sha256(output / "rows.json")
        atomic_json(output / "report.json", report)
        atomic_json(output / "completion.json", {"report_sha256": sha256(output / "report.json"), "complete": True})
        print("ORACLE COMPLETE", retained, "teacher_forced_pass", passed, json.dumps(pooled), flush=True)
        return 0
    except Exception as error:
        if output is not None:
            atomic_json(output / "failure.json", {"error": f"{type(error).__name__}: {error}", "complete": False})
        print(f"{type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1


def summarize(campaign):
    frozen = validate_freeze(campaign)
    results = []
    for retained in frozen["retained_order"]:
        results.append(completed_report(campaign, retained, frozen, verify_artifacts=True))
    passing = [r["retained_blocks"] for r in results if r["retained_blocks"] < 10 and r["teacher_forced_pass"]]
    summary = {"format": "slotstream-oracle-decision-v1", "frozen_sha256": sha256(campaign / "frozen.json"),
        "disposition": "teacher-forced-candidate" if passing else "rejected",
        "selected_retained_blocks": min(passing) if passing else None,
        "predictor_advancement": False, "qualification": False,
        "cohorts": [{"retained": r["retained_blocks"], "teacher_forced_pass": r["teacher_forced_pass"], "pooled": r["pooled"]} for r in results]}
    atomic_json(campaign / "decision.json", summary)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("freeze")
    for name in ("suite", "reference", "study", "norms", "binary", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p = sub.add_parser("run")
    p.add_argument("--campaign", type=Path, required=True)
    p.add_argument("--retained", type=int, choices=list(calibrate.RETAINED), required=True)
    p = sub.add_parser("summarize")
    p.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        return freeze(args.suite, args.reference, args.study, args.norms, args.binary, args.output)
    if args.command == "run":
        return run(args.campaign, args.retained)
    summarize(args.campaign)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
