#!/usr/bin/env python3
"""Strict validators for Flash receipt and selection documents."""
from __future__ import annotations

import math
import json
from pathlib import Path
from typing import Any

from common import EvidenceError, RECEIPT_FORMAT, SELECTION_FORMAT, read_json, sha256


def _exact(value: dict[str, Any], required: set[str], label: str) -> None:
    if set(value) != required:
        raise EvidenceError(f"{label} fields differ: expected {sorted(required)}, got {sorted(value)}")


def _artifact_path(evidence_dir: Path, name: str) -> Path:
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise EvidenceError(f"unsafe artifact path: {name}")
    try:
        resolved = (evidence_dir / path).resolve(strict=True)
    except OSError as error:
        raise EvidenceError(f"missing artifact: {name}") from error
    if not resolved.is_relative_to(evidence_dir.resolve()) or not resolved.is_file():
        raise EvidenceError(f"unsafe artifact path: {name}")
    return resolved


def validate_receipt(value: dict[str, Any], evidence_dir: Path | None = None, *, require_qualified: bool = False) -> None:
    required = {"format", "schema_version", "kind", "qualified", "qualification_reasons", "command",
                "environment", "started_at_unix", "ended_at_unix", "duration_seconds", "process",
                "memory", "vm", "result", "artifacts", "identities"}
    _exact(value, required, "receipt")
    if value["format"] != RECEIPT_FORMAT or value["schema_version"] != 1:
        raise EvidenceError("unsupported receipt format")
    if value["kind"] not in ("launch", "archive", "stage0"):
        raise EvidenceError("unknown receipt kind")
    if (type(value["qualified"]) is not bool or not isinstance(value["qualification_reasons"], list)
            or not all(isinstance(reason, str) and reason for reason in value["qualification_reasons"])):
        raise EvidenceError("invalid qualification fields")
    if require_qualified and not value["qualified"]:
        raise EvidenceError("receipt is not qualified")
    for key in ("started_at_unix", "ended_at_unix", "duration_seconds"):
        if type(value[key]) not in (int, float) or not math.isfinite(value[key]) or value[key] < 0:
            raise EvidenceError(f"invalid {key}")
    if value["ended_at_unix"] < value["started_at_unix"]:
        raise EvidenceError("receipt ends before it starts")
    if not isinstance(value["command"], list) or not value["command"] or not all(isinstance(x, str) for x in value["command"]):
        raise EvidenceError("invalid command")
    if not isinstance(value["environment"], dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in value["environment"].items()):
        raise EvidenceError("invalid environment")
    if not isinstance(value["process"], dict) or not isinstance(value["memory"], dict) or not isinstance(value["result"], dict):
        raise EvidenceError("invalid nested receipt object")
    if type(value["result"].get("exit_code")) not in (int, type(None)):
        raise EvidenceError("invalid exit code")
    if not isinstance(value["artifacts"], dict):
        raise EvidenceError("invalid artifacts")
    if evidence_dir is not None:
        for name, artifact in value["artifacts"].items():
            if not isinstance(artifact, dict) or set(artifact) != {"bytes", "sha256"}:
                raise EvidenceError(f"invalid artifact entry: {name}")
            path = _artifact_path(evidence_dir, name)
            if path.stat().st_size != artifact["bytes"] or sha256(path) != artifact["sha256"]:
                raise EvidenceError(f"artifact hash mismatch: {name}")
    if value["kind"] == "launch":
        memory = value["memory"]
        expected_memory = {"target_gb_decimal", "sample_interval_seconds", "qualified", "peak_bytes",
                           "sample_count", "samples_artifact", "sampler_error"}
        _exact(memory, expected_memory, "launch memory")
        target = memory["target_gb_decimal"]
        count = memory["sample_count"]
        peak = memory["peak_bytes"]
        if type(target) not in (int, float) or not math.isfinite(target) or target <= 0:
            raise EvidenceError("invalid memory target")
        if type(memory["qualified"]) is not bool or type(count) is not int or count < 0 or (count == 0) != (peak is None):
            raise EvidenceError("invalid memory sample count")
        if peak is not None and (type(peak) is not int or peak <= 0):
            raise EvidenceError("invalid memory peak")
        if evidence_dir is not None:
            sample_name = memory["samples_artifact"]
            if not isinstance(sample_name, str) or sample_name not in value["artifacts"]:
                raise EvidenceError("memory samples artifact is not declared and hashed")
            sample_path = _artifact_path(evidence_dir, sample_name)
            observed_count = 0
            observed_peak = None
            previous_elapsed = -1.0
            try:
                with sample_path.open() as stream:
                    for line in stream:
                        try: sample = json.loads(line)
                        except json.JSONDecodeError as error: raise EvidenceError(f"malformed memory sample: {error}") from error
                        required_sample = {"start_abstime", "physical_footprint_bytes", "lifetime_peak_bytes",
                                           "resident_bytes", "elapsed_seconds"}
                        _exact(sample, required_sample, "memory sample")
                        for key in ("start_abstime", "physical_footprint_bytes", "lifetime_peak_bytes", "resident_bytes"):
                            if type(sample[key]) is not int or sample[key] <= 0:
                                raise EvidenceError(f"invalid memory sample field: {key}")
                        if sample["start_abstime"] != value["process"].get("start_identity"):
                            raise EvidenceError("memory sample process identity mismatch")
                        elapsed = sample["elapsed_seconds"]
                        if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0 or elapsed < previous_elapsed:
                            raise EvidenceError("invalid or nonmonotonic memory sample elapsed time")
                        previous_elapsed = elapsed
                        item_peak = max(sample["physical_footprint_bytes"], sample["lifetime_peak_bytes"])
                        observed_peak = item_peak if observed_peak is None else max(observed_peak, item_peak)
                        observed_count += 1
            except OSError as error:
                raise EvidenceError(f"missing memory samples: {error}") from error
            if observed_count != count or observed_peak != peak:
                raise EvidenceError("memory sample summary mismatch")
        if memory["qualified"] and (count <= 0 or peak > target * 1e9):
            raise EvidenceError("memory evidence exceeds target or is empty")
        result = value["result"]
        _exact(result, {"exit_code", "functional_success", "timed_out", "interrupted", "budget_exceeded", "evidence_failure"}, "launch result")
        for key in ("functional_success", "timed_out", "interrupted", "budget_exceeded", "evidence_failure"):
            if type(result[key]) is not bool:
                raise EvidenceError(f"invalid launch result Boolean: {key}")
        expected_success = (result["exit_code"] == 0 and not result["timed_out"] and not result["interrupted"]
                            and not result["budget_exceeded"] and memory["qualified"]
                            and memory["sampler_error"] is None and not result["evidence_failure"])
        if result["functional_success"] != expected_success:
            raise EvidenceError("functional success contradicts exit, memory, or interruption state")
    if value["qualified"]:
        if value["qualification_reasons"]:
            raise EvidenceError("qualified receipt has failure reasons")
        if value["result"].get("exit_code") != 0:
            raise EvidenceError("qualified receipt has nonzero result")
        if value["kind"] == "launch":
            memory = value["memory"]
            if not memory.get("qualified") or memory.get("sample_count", 0) <= 0:
                raise EvidenceError("qualified receipt lacks memory evidence")
            if any(value["identities"].get(key) is None for key in ("run_set_id", "model_hash")):
                raise EvidenceError("qualified receipt lacks required identities")


def validate_receipt_file(path: Path, *, require_qualified: bool = False) -> dict[str, Any]:
    value = read_json(path)
    validate_receipt(value, path.parent, require_qualified=require_qualified)
    completion_path = path.parent / "completion.json"
    functional = bool(value.get("result", {}).get("functional_success", value["qualified"]))
    if functional:
        completion = read_json(completion_path)
        _exact(completion, {"format", "receipt_sha256", "qualified", "functional_success"}, "completion")
        if (completion["format"] != "slotstream-flash-completion-v1"
                or type(completion["qualified"]) is not bool or completion["qualified"] != value["qualified"]
                or completion["functional_success"] is not True
                or completion["receipt_sha256"] != sha256(path)):
            raise EvidenceError("completion does not bind the receipt")
    elif completion_path.exists():
        raise EvidenceError("failed receipt must not have completion metadata")
    return value


def validate_selection(value: dict[str, Any]) -> None:
    _exact(value, {"format", "schema_version", "selected_stage_ids", "selected_arm_ids", "stages"}, "selection")
    if value["format"] != SELECTION_FORMAT or value["schema_version"] != 1:
        raise EvidenceError("unsupported selection format")
    if not isinstance(value["stages"], list) or len(value["stages"]) != 8:
        raise EvidenceError("selection must contain stages 0 through 7")
    for index, stage in enumerate(value["stages"]):
        _exact(stage, {"stage_id", "disposition", "reason", "evidence"}, f"stage {index}")
        if stage["stage_id"] != index:
            raise EvidenceError("selection stage IDs are not exact")
        if stage["disposition"] not in ("accepted", "rejected", "blocked", "already_used", "not_selected", "pending"):
            raise EvidenceError("invalid stage disposition")
