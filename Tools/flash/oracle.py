#!/usr/bin/env python3
"""Strict, separately versioned neuron-oracle configuration and capture evidence."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import EvidenceError, atomic_json, read_json, sha256
from receipts import require_terminal_sampling, validate_receipt_file
import benchmark
import calibrate
import capture

RESERVATION = 208 << 20
EXTRA = 80 << 20
RECORD_BYTES = 4_300_800
CONFIG_FIELDS = {"format", "schema_version", "algorithm", "retained_blocks", "split",
    "norm_manifest", "norm_manifest_sha256", "study", "study_sha256", "suite", "suite_sha256",
    "capture_manifest_sha256", "diagnostic_reserved_bytes"}


def configuration(norms: Path, study: Path, suite: Path, manifest: Path, retained: int):
    if type(retained) is not int or retained not in calibrate.RETAINED:
        raise EvidenceError("unfrozen retained count")
    return {"format": "slotstream-oracle-config-v1", "schema_version": 1,
        "algorithm": calibrate.ALGORITHM, "retained_blocks": retained, "split": "development",
        "norm_manifest": str((norms / "manifest.json").resolve()),
        "norm_manifest_sha256": sha256(norms / "manifest.json"),
        "study": str(study.resolve()), "study_sha256": sha256(study),
        "suite": str(suite.resolve()), "suite_sha256": sha256(suite),
        "capture_manifest_sha256": sha256(manifest), "diagnostic_reserved_bytes": RESERVATION}


def validate_configuration(path: Path, manifest: Path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 8192:
        raise EvidenceError("unsafe or excessive oracle configuration")
    value = read_json(path)
    if (set(value) != CONFIG_FIELDS or value["format"] != "slotstream-oracle-config-v1"
            or type(value["schema_version"]) is not int or value["schema_version"] != 1
            or value["algorithm"] != calibrate.ALGORITHM or value["split"] != "development"
            or type(value["retained_blocks"]) is not int or value["retained_blocks"] not in calibrate.RETAINED
            or type(value["diagnostic_reserved_bytes"]) is not int or value["diagnostic_reserved_bytes"] != RESERVATION
            or value["capture_manifest_sha256"] != sha256(manifest)):
        raise EvidenceError("unsupported or mismatched oracle configuration")
    for name in ("norm_manifest", "study", "suite"):
        asset = Path(value[name])
        if not asset.is_absolute() or asset.is_symlink() or sha256(asset) != value[name + "_sha256"]:
            raise EvidenceError("oracle bound asset changed")
    study = read_json(Path(value["study"]))
    if (study.get("format") != "slotstream-neuron-study-v1"
            or study.get("algorithm") != value["algorithm"] or study.get("split") != "development"
            or study.get("retained_order") != list(calibrate.RETAINED)
            or study.get("norm_manifest_sha256") != value["norm_manifest_sha256"]
            or study.get("suite_sha256") != value["suite_sha256"]):
        raise EvidenceError("oracle study bindings differ")
    return value


def validate_masks(report, root: Path, config, *, verify_scores=True):
    evidence = report.get("oracle", {})
    expected = {"format": "slotstream-oracle-evidence-v1", "study_sha256": config["study_sha256"],
        "suite_sha256": config["suite_sha256"], "norm_manifest_sha256": config["norm_manifest_sha256"],
        "algorithm": config["algorithm"], "retained_blocks": config["retained_blocks"],
        "diagnostic_reserved_bytes": RESERVATION, "norm_payload_bytes": 48 * 512 * 640 * 4,
        "warmup_policy": "dense-whole-document", "mask_failure_state_reuse_refused": True}
    if (set(evidence) != set(expected) | {"mask_files", "config_sha256"}
            or evidence.get("mask_failure_state_reuse_refused") is not True
            or any(evidence.get(k) != v for k, v in expected.items())):
        raise EvidenceError("oracle execution or reservation evidence differs")
    positions = [(doc["id"], pos) for doc in report["documents"] for pos in doc["positions"]]
    files = evidence["mask_files"]
    if not isinstance(files, list) or len(files) != len(positions):
        raise EvidenceError("oracle mask position coverage differs")
    tables = {}
    if verify_scores and config["retained_blocks"] < 10:
        import numpy as np
        norm_root = Path(config["norm_manifest"]).parent
        _, norm_files = calibrate.validate_norms(norm_root)
        tables = {layer: np.memmap(path, dtype="<f4", mode="r", shape=(512, 640)) for layer, path in norm_files.items()}
    for ordinal, (item, (document, pos)) in enumerate(zip(files, positions)):
        if (set(item) != {"name", "document_id", "token_position", "bytes", "sha256"}
                or item["name"] != f"oracle-mask-{ordinal:03d}.json"
                or type(item["token_position"]) is not int
                or item["document_id"] != document or item["token_position"] != pos["position"]
                or type(item["bytes"]) is not int or not 0 < item["bytes"] <= 128 << 10):
            raise EvidenceError("oracle mask identity or size differs")
        path = root / item["name"]
        if path.is_symlink() or not path.is_file() or path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise EvidenceError("oracle mask bytes changed")
        rows = json.loads(path.read_text())
        if not isinstance(rows, list) or len(rows) != 48:
            raise EvidenceError("oracle mask layer coverage differs")
        for layer, row in enumerate(rows):
            if (set(row) != {"layer", "router_ranks", "expert_ids", "blocks"}
                    or type(row["layer"]) is not int or row["layer"] != layer
                    or row["router_ranks"] != list(range(10))
                    or any(type(x) is not int for x in row["router_ranks"] + row["expert_ids"])
                    or row["expert_ids"] != pos["routes"][layer]["ids"]
                    or len(set(row["expert_ids"])) != 10 or len(row["blocks"]) != 10):
                raise EvidenceError("oracle mask router mapping differs")
            for mask in row["blocks"]:
                if (not isinstance(mask, list) or len(mask) != 10
                        or any(type(x) is not bool for x in mask)
                        or sum(mask) != config["retained_blocks"]):
                    raise EvidenceError("oracle mask cardinality or Boolean format differs")
            if tables:
                hidden = [f for f in report["files"] if f["category"] == "hidden"
                    and f["document_id"] == document and f["token_position"] == pos["position"] and f["layer"] == layer]
                if len(hidden) != 1 or hidden[0]["router_ranks"] != list(range(10)):
                    raise EvidenceError("oracle requires one unsplit original hidden tensor")
                values = np.fromfile(root / hidden[0]["name"], dtype="<f4").reshape(10, 640)
                for rank, expert in enumerate(row["expert_ids"]):
                    expected_mask = calibrate.selected_blocks(calibrate.block_scores(values[rank], tables[layer][expert]), config["retained_blocks"])
                    if expected_mask != row["blocks"][rank]:
                        raise EvidenceError("native mask differs from independent host scoring")


def validate_output(root: Path, manifest: Path, config_path: Path, binary: Path, model: Path):
    config = validate_configuration(config_path, manifest)
    request = capture.validate_manifest(manifest)
    if request.get("split") != "development":
        raise EvidenceError("oracle capture requires frozen development tokens")
    report = read_json(root / "report.json")
    if read_json(root / "completion.json") != {"format": "slotstream-flash-oracle-completion-v1",
            "report_sha256": sha256(root / "report.json"), "qualification": False}:
        raise EvidenceError("oracle completion does not bind its report")
    capture._validate_native_report(report, root, manifest, "oracle", binary, model, request, oracle=True)
    if report.get("oracle", {}).get("config_sha256") != sha256(config_path):
        raise EvidenceError("oracle report binds another configuration")
    validate_masks(report, root, config)
    expected_files = {"report.json", "completion.json", *(x["name"] for x in report["files"]),
                      *(x["name"] for x in report["oracle"]["mask_files"])}
    if {p.name for p in root.iterdir()} != expected_files:
        raise EvidenceError("oracle native inventory has unexpected or missing files")
    return report


def compare_controls(reference, candidate):
    if reference["optimizations"] != candidate["optimizations"] or reference["numerical_environment"] != candidate["numerical_environment"]:
        raise EvidenceError("oracle changed ordinary arithmetic controls")
    for name in ("model_config_sha256", "model_index_sha256", "metallib_sha256"):
        if reference["source_identity"][name] != candidate["source_identity"][name]:
            raise EvidenceError("oracle changed checkpoint or kernel library")
    a, b = copy.deepcopy(reference["plan"]), copy.deepcopy(candidate["plan"])
    la, lb = a.pop("memory_ledger"), b.pop("memory_ledger")
    if la != reference["memory_ledger"] or lb != candidate["memory_ledger"]:
        raise EvidenceError("oracle plan/ledger disagreement")
    if (la["diagnostic_reserved_bytes"] != 128 << 20 or lb["diagnostic_reserved_bytes"] != RESERVATION
            or b["target_gb"] != 14 or b["runtime_prefix_cache_enabled"] or b["mtp"] or b["vision"]
            or b["pool_slots"] < 640 or lb["pool_bytes"] != b["pool_slots"] * RECORD_BYTES):
        raise EvidenceError("oracle reservation, pool or disabled feature contract differs")
    available = 13_000_000_000 - la["expected_peak_bytes"] + la["pool_bytes"] - EXTRA
    if b["pool_slots"] != available // RECORD_BYTES:
        raise EvidenceError("oracle pool was not sized from the charged reservation")
    if lb["expected_peak_bytes"] != la["expected_peak_bytes"] + EXTRA + lb["pool_bytes"] - la["pool_bytes"]:
        raise EvidenceError("oracle peak does not account for norm ownership")
    for key in ("diagnostic_reserved_bytes", "pool_bytes", "expected_peak_bytes"):
        la.pop(key); lb.pop(key)
    if la != lb:
        raise EvidenceError("oracle changed an unrelated memory ledger field")
    for key in ("device_available_gb", "pool_slots", "pool_gb", "experts_per_layer_cached", "expected_peak_gb"):
        a.pop(key); b.pop(key)
    if a != b:
        raise EvidenceError("oracle changed an unrelated plan control")


def state_structure(state):
    value = copy.deepcopy(state)
    for field in value["fields"]:
        field.pop("sha256", None)
    return value


def run_native(binary, model, manifest, config, output):
    validate_configuration(config, manifest)
    command = [str(binary), "flash-capture", "--model", str(model), "--manifest", str(manifest),
        "--split", "development", "--mode", "oracle", "--oracle-config", str(config),
        "--memory-gb", "14", "--max-context", "2048", "--live-limit-mb", "64", "--output", str(output / "native")]
    settling = benchmark.settle_before_model_launch()
    code = benchmark.launch(output, 14, 900, command)
    atomic_json(output / "settling.json", settling)
    receipt = validate_receipt_file(output / "receipt.json")
    require_terminal_sampling(receipt)
    if code or not receipt["result"]["functional_success"] or not receipt["memory"]["qualified"]:
        raise EvidenceError("monitored oracle capture failed")
    return validate_output(output / "native", manifest, config, binary, model), receipt
