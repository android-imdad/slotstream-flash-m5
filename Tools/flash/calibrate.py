#!/usr/bin/env python3
"""Export and validate original norms; freeze development-only neuron-block scoring.

This is a dense diagnostic. Estimated source bytes and contribution scores are
not measured SSD savings, full-model quality, or permission to train a predictor.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import struct
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import EvidenceError, atomic_json, fresh_output, harness_hashes, read_json, sha256, validate_build_identity
import benchmark
import cache_study
from receipts import require_terminal_sampling, validate_receipt_file

ALGORITHM = "fp32-values-fp64-sequential-column-l2-block64-v1"
RETAINED = (10, 8, 6, 4, 2)


def column_norms(values, rows, columns):
    if rows <= 0 or columns <= 0 or len(values) != rows * columns:
        raise EvidenceError("invalid column-norm shape")
    sums = [0.0] * columns
    for row in range(rows):
        for column in range(columns):
            value = float(values[row * columns + column])
            if not math.isfinite(value):
                raise EvidenceError("nonfinite down-projection value")
            sums[column] += value * value
    return [struct.unpack("<f", struct.pack("<f", math.sqrt(value)))[0] for value in sums]


def block_scores(hidden, norms):
    if len(hidden) != 640 or len(norms) != 640:
        raise EvidenceError("oracle requires 640 neurons")
    scores = [0.0] * 10
    for i, (hidden_value, norm) in enumerate(zip(hidden, norms)):
        value, norm = float(hidden_value), float(norm)
        if not math.isfinite(value) or not math.isfinite(norm) or norm < 0:
            raise EvidenceError("invalid oracle activation or norm")
        scores[i // 64] += abs(value) * norm
    if not all(math.isfinite(value) for value in scores):
        raise EvidenceError("nonfinite block score")
    return scores


def selected_blocks(scores, retained):
    if (type(retained) is not int or retained not in RETAINED or len(scores) != 10
            or any(not math.isfinite(float(x)) or x < 0 for x in scores)):
        raise EvidenceError("invalid oracle block selection")
    order = sorted(range(10), key=lambda i: (-scores[i], i))
    chosen = set(order[:retained])
    return [i in chosen for i in range(10)]


def validate_norms(root: Path, *, require_full=True):
    root = root.resolve(strict=True)
    manifest = read_json(root / "manifest.json")
    completion = read_json(root / "completion.json")
    if completion != {"format": "slotstream-column-norms-completion-v1",
                      "manifest_sha256": sha256(root / "manifest.json")}:
        raise EvidenceError("norm completion does not bind its manifest")
    required = {"format", "schema_version", "algorithm", "scope", "dtype", "shape", "expert_count",
                "model_path", "model_metadata", "source_identities", "source_layouts", "provenance",
                "artifacts", "model_loaded", "qualification"}
    if (set(manifest) != required or manifest["format"] != "slotstream-column-norms-v1"
            or manifest["schema_version"] != 1 or manifest["algorithm"] != ALGORITHM
            or manifest["scope"] not in ("sample", "full")
            or (require_full and manifest["scope"] != "full")
            or manifest["dtype"] != "float32-le" or manifest["shape"] != [48, 512, 640]
            or manifest["model_loaded"] is not False or manifest["qualification"] is not False):
        raise EvidenceError("unsupported or incomplete norm asset")
    files = {}
    count = 0
    for artifact in manifest["artifacts"]:
        if set(artifact) != {"path", "layer", "experts", "shape", "bytes", "sha256"}:
            raise EvidenceError("malformed norm artifact")
        layer, experts = artifact["layer"], artifact["experts"]
        if (type(layer) is not int or not 0 <= layer < 48 or layer in files
                or not isinstance(experts, list) or not experts
                or any(type(expert) is not int or not 0 <= expert < 512 for expert in experts)
                or sorted(set(experts)) != experts
                or artifact["path"] != f"layer-{layer:02d}.f32"
                or artifact["shape"] != [len(experts), 640]
                or artifact["bytes"] != len(experts) * 640 * 4):
            raise EvidenceError("norm layer/expert coverage or geometry differs")
        path = root / artifact["path"]
        if (path.is_symlink() or not path.is_file() or path.stat().st_size != artifact["bytes"]
                or sha256(path) != artifact["sha256"]):
            raise EvidenceError("norm bytes changed or are missing")
        if any(not math.isfinite(x[0]) or x[0] < 0 for x in struct.iter_unpack("<f", path.read_bytes())):
            raise EvidenceError("nonfinite or negative norm")
        if manifest["scope"] == "full" and experts != list(range(512)):
            raise EvidenceError("full norm layer is incomplete")
        files[layer] = path
        count += len(experts)
    if count != manifest["expert_count"] or (manifest["scope"] == "full" and (len(files) != 48 or count != 24576)):
        raise EvidenceError("norm expert count differs")
    layouts = manifest["source_layouts"]
    if (not isinstance(layouts, list) or len(layouts) != 48
            or [x.get("layer") for x in layouts] != list(range(48))
            or any(x.get("group_size") != 64 or x.get("bits") not in (4, 6)
                   or type(x.get("source_record_bytes")) is not int or x["source_record_bytes"] <= 0 for x in layouts)):
        raise EvidenceError("source layouts are incomplete")
    expected_files = {"manifest.json", "completion.json", *(x["path"] for x in manifest["artifacts"])}
    if {p.name for p in root.iterdir()} != expected_files:
        raise EvidenceError("norm directory has unexpected or incomplete artifacts")
    return manifest, files


def export_norms(binary: Path, model: Path, output: Path, *, sample=False):
    output = fresh_output(output)
    try:
        verification = cache_study.verified_model_revision(model)
        geometry = cache_study.derive_source_geometry(model, verification=verification)
        if benchmark.archive(binary, output / "archive"):
            raise EvidenceError("norm binary archive failed")
        candidate = output / "archive/bin/slotstream"
        command = [str(candidate), "flash-column-norms", "--model", str(model.resolve()),
                   "--output", str(output / "run/native")]
        if sample:
            command.append("--sample")
        code = benchmark.launch(output / "run", 1.0, 3600, command,
                                run_set_id="column-norms-v1", model_hash=sha256(model / "config.json"))
        receipt = validate_receipt_file(output / "run/receipt.json")
        require_terminal_sampling(receipt)
        if code or not receipt["result"]["functional_success"] or not receipt["memory"]["qualified"]:
            raise EvidenceError("bounded norm export failed")
        manifest, _ = validate_norms(output / "run/native", require_full=not sample)
        build, paths = validate_build_identity(candidate)
        expected = {"slotstream": build["binary_sha256"], "mlx.metallib": build["metallib_sha256"],
                    "build-identity.json": sha256(paths["identity"]), "build-source.tar.gz": build["source_archive_sha256"]}
        if manifest["provenance"] != expected:
            raise EvidenceError("native norm provenance differs from archived binary")
        if cache_study.derive_source_geometry(model, verification=verification) != geometry:
            raise EvidenceError("norm checkpoint changed during export")
        atomic_json(output / "completion.json", {"format": "slotstream-norm-export-v1",
            "manifest_sha256": sha256(output / "run/native/manifest.json"),
            "receipt_sha256": sha256(output / "run/receipt.json"), "model_verification": verification,
            "scope": manifest["scope"], "qualification": False})
        return 0
    except Exception as error:
        atomic_json(output / "failure.json", {"error": f"{type(error).__name__}: {error}"})
        return 1


def study(suite: Path, cohort: Path, norms: Path, output: Path):
    # Import the pinned evaluation environment only for actual capture work.
    import numpy as np
    import metrics
    output = fresh_output(output)
    try:
        index, positions = metrics.validate_cohort(cohort)
        if (index["kind"] != "full-study" or index["split"] != "development"
                or index["suite"]["sha256"] != sha256(suite)):
            raise EvidenceError("calibration requires the complete frozen development cohort")
        manifest, files = validate_norms(norms)
        source = index["shards"][0]["capture"]["sourceIdentity"]
        if manifest["model_metadata"] != {"config.json": source["model_config_sha256"],
                                          "model.safetensors.index.json": source["model_index_sha256"]}:
            raise EvidenceError("norms and activations bind different checkpoints")
        # A read-only mapping is the single norm representation; input validation
        # and scoring retain only one layer/activation file at a time.
        tables = {layer: np.memmap(path, dtype="<f4", mode="r", shape=(512, 640)) for layer, path in files.items()}
        aggregates = {}
        neuron_zeros = neuron_count = 0
        observed = 0
        with (output / "scores.jsonl").open("x") as stream:
            for ordinal, shard in enumerate(index["shards"]):
                native = metrics.ROOT / shard["capture"]["capturePath"] / "native"
                report = read_json(native / "report.json")
                if shard["capture"]["mode"] != "reference-on":
                    raise EvidenceError("calibration requires dense activation capture")
                seen = set()
                for item in report["files"]:
                    if item["category"] != "hidden":
                        continue
                    path = native / item["name"]
                    if sha256(path) != item["sha256"]:
                        raise EvidenceError("activation bytes changed")
                    if item["dtype"] == "float32":
                        values = np.fromfile(path, dtype="<f4")
                    elif item["dtype"] == "bfloat16":
                        values = (np.fromfile(path, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
                    elif item["dtype"] == "float16":
                        values = np.fromfile(path, dtype="<f2").astype(np.float32)
                    else:
                        raise EvidenceError("unsupported hidden precision")
                    if values.size != len(item["expert_ids"]) * 640:
                        raise EvidenceError("hidden geometry differs")
                    values = values.reshape(-1, 640)
                    position_key = (ordinal, item["document_id"], item["token_position"])
                    category = positions[position_key]["category"]
                    layer = item["layer"]
                    for rank, expert, hidden in zip(item["router_ranks"], item["expert_ids"], values):
                        key = (item["document_id"], item["token_position"], layer, rank)
                        if key in seen:
                            raise EvidenceError("duplicate hidden router rank")
                        seen.add(key)
                        column = tables[layer][expert]
                        scores = block_scores(hidden, column)
                        masks = {str(k): selected_blocks(scores, k) for k in RETAINED}
                        row = {"split": "development", "document": item["document_id"],
                               "position": item["token_position"], "layer": layer, "rank": rank,
                               "expert": expert, "category": category, "scores": scores, "masks": masks}
                        stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
                        neuron_zeros += int(np.count_nonzero(hidden == 0))
                        neuron_count += 640
                        observed += 1
                        total = sum(scores)
                        individual = sorted(range(640), key=lambda i: (-abs(float(hidden[i])) * float(column[i]), i))
                        for k in RETAINED:
                            group = aggregates.setdefault((category, layer, k), {"examples": 0,
                                "omitted_contribution_fraction_sum": 0.0, "individual_selection_block_union_sum": 0,
                                "hypothetical_dense_source_bytes": 0, "hypothetical_selected_source_bytes": 0})
                            group["examples"] += 1
                            group["omitted_contribution_fraction_sum"] += sum(x for x, keep in zip(scores, masks[str(k)]) if not keep) / total if total else 0
                            group["individual_selection_block_union_sum"] += len({i // 64 for i in individual[:k * 64]})
                            dense = manifest["source_layouts"][layer]["source_record_bytes"]
                            # 64-neuron-aligned gate/up rows and original down
                            # columns plus their scale/bias groups scale together.
                            group["hypothetical_dense_source_bytes"] += dense
                            group["hypothetical_selected_source_bytes"] += dense * k // 10
                if len(seen) != shard["positions"] * 48 * 10:
                    raise EvidenceError("hidden capture misses a position/layer/router rank")
        if observed != len(positions) * 48 * 10:
            raise EvidenceError("incomplete development hidden coverage")
        rows = [{"category": category, "layer": layer, "retained": k, **value}
                for (category, layer, k), value in sorted(aggregates.items())]
        report = {"format": "slotstream-neuron-study-v1", "algorithm": ALGORITHM,
            "retained_order": list(RETAINED), "split": "development", "positions": len(positions),
            "routed_examples": observed, "neuron_count": neuron_count, "exact_zero_count": neuron_zeros,
            "suite_sha256": sha256(suite), "cohort_sha256": sha256(cohort / "index.json"),
            "norm_manifest_sha256": sha256(norms / "manifest.json"), "scores_sha256": sha256(output / "scores.jsonl"),
            "groups": rows, "harness_hashes": harness_hashes(), "qualification": False,
            "predictor_advancement": False, "measured_ssd_bytes_saved": 0,
            "limits": "Contribution scores and hypothetical bundled bytes only. Cache hits, fallback, prefetch, predictor costs and full-model quality are not evaluated."}
        atomic_json(output / "study.json", report)
        atomic_json(output / "completion.json", {"study_sha256": sha256(output / "study.json"), "qualification": False})
        return 0
    except Exception as error:
        atomic_json(output / "failure.json", {"error": f"{type(error).__name__}: {error}"})
        return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("norms")
    for name in ("binary", "model", "output"):
        export.add_argument("--" + name, type=Path, required=True)
    export.add_argument("--sample", action="store_true")
    calibrate = sub.add_parser("study")
    for name in ("suite", "capture-index", "norms", "output"):
        calibrate.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.command == "norms":
        return export_norms(args.binary, args.model, args.output, sample=args.sample)
    return study(args.suite, args.capture_index, args.norms, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
