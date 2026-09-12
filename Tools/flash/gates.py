#!/usr/bin/env python3
"""Strict stage gates for Flash experiments."""
from __future__ import annotations

import argparse
import io
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from common import (EvidenceError, RECEIPT_FORMAT, SELECTION_FORMAT, atomic_json,
                    fresh_output, harness_hashes, sha256, validate_build_identity)
from receipts import validate_selection

REQUIRED_CHECKS = {"jang-formats", "jang-numerics"}
M5_REQUIRED_CHECKS = {"m5-eligibility", "m5-dispatch"}
FLASH_REQUIRED_CHECKS = {"flash-identity", "flash-observation"}


def validate_checks(
    document: dict[str, Any], *, required: set[str] = REQUIRED_CHECKS,
    exact_names: bool = False
) -> None:
    if set(document) != {"checks", "passed", "failed", "skipped"}:
        raise EvidenceError("check report has unexpected fields")
    checks = document["checks"]
    if not isinstance(checks, list) or not checks:
        raise EvidenceError("check report has no checks")
    names = [item.get("name") for item in checks if isinstance(item, dict)]
    if len(names) != len(checks) or len(set(names)) != len(names):
        raise EvidenceError("check names are malformed or duplicated")
    if not required.issubset(set(names)):
        raise EvidenceError(f"required checks absent: {sorted(required - set(names))}")
    if exact_names and set(names) != required:
        raise EvidenceError(f"unexpected filtered checks: {sorted(set(names) - required)}")
    if document["failed"] != 0 or document["skipped"] != 0 or document["passed"] != len(checks):
        raise EvidenceError("check totals contain failures, skips, or inconsistencies")
    for check in checks:
        if check.get("skipped") is not None:
            raise EvidenceError(f"check skipped: {check['name']}")
        items = check.get("items")
        if not isinstance(items, list) or not items:
            raise EvidenceError(f"check has zero assertions: {check['name']}")
        if any(type(item.get("passed")) is not bool or not item["passed"] for item in items if isinstance(item, dict)):
            raise EvidenceError(f"check has failed or malformed assertions: {check['name']}")
        if any(not isinstance(item, dict) for item in items):
            raise EvidenceError(f"check has malformed assertions: {check['name']}")


def validate_python_test_result(result: dict[str, Any], required_classes: set[str]) -> int:
    required = {"test_ids", "test_classes", "tests_run", "failures", "errors", "skipped", "successful"}
    if set(result) != required:
        raise EvidenceError("Python test result fields are malformed")
    if not result["successful"] or result["failures"] or result["errors"]:
        raise EvidenceError("Python gate tests failed")
    if type(result["tests_run"]) is not int or result["tests_run"] <= 0 or result["tests_run"] != len(result["test_ids"]):
        raise EvidenceError("Python gate discovered zero tests")
    missing = required_classes - set(result["test_classes"])
    if missing:
        raise EvidenceError(f"named Python test classes absent: {sorted(missing)}")
    if result["skipped"]:
        raise EvidenceError("Python gate contains skipped tests")
    return result["tests_run"]


def _flatten(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from _flatten(item)
        else:
            yield item


def run_python_tests() -> dict[str, Any]:
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for name in ("test_launcher", "test_receipts", "test_observe", "test_m5_probe", "test_cache_study",
                 "test_replay", "test_capture", "test_tokenize"):
        suite.addTests(loader.loadTestsFromModule(importlib.import_module(name)))
    tests = list(_flatten(suite))
    ids = [test.id() for test in tests]
    classes = sorted({test.__class__.__name__ for test in tests})
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=0).run(suite)
    return {"test_ids": ids, "test_classes": classes, "tests_run": result.testsRun,
            "failures": [test.id() for test, _ in result.failures],
            "errors": [test.id() for test, _ in result.errors],
            "skipped": [{"id": test.id(), "reason": reason} for test, reason in result.skipped],
            "successful": result.wasSuccessful()}


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=ROOT, text=True, capture_output=True)


def stage_zero(output: Path, *, runner: Callable[[list[str]], subprocess.CompletedProcess[str]] = _run) -> int:
    output = fresh_output(output)
    started = time.time()
    commands: list[dict[str, Any]] = []
    try:
        build_command = ["make", "build", "SLOTSTREAM_BUILD_JOBS=2"]
        built = runner(build_command)
        commands.append({"argv": build_command, "exit_code": built.returncode,
                         "stdout": built.stdout, "stderr": built.stderr})
        if built.returncode != 0:
            raise EvidenceError("two-job release build failed")
        binary = ROOT / ".build" / "release" / "slotstream"
        identity, _ = validate_build_identity(binary)

        check_command = [str(ROOT / ".build" / "release" / "slotstream-checks"), "--tier", "t0", "--tier", "t1", "--json"]
        checked = runner(check_command)
        commands.append({"argv": check_command, "exit_code": checked.returncode,
                         "stdout": checked.stdout, "stderr": checked.stderr})
        if checked.returncode != 0:
            raise EvidenceError("native check catalogue failed")
        try:
            checks = json.loads(checked.stdout)
        except json.JSONDecodeError as error:
            raise EvidenceError(f"malformed native check JSON: {error}") from error
        if not isinstance(checks, dict):
            raise EvidenceError("native check JSON is not an object")
        validate_checks(checks)
        atomic_json(output / "checks.json", checks)

        test_command = [sys.executable, "-m", "unittest", "Tools/flash/test_launcher.py", "Tools/flash/test_receipts.py"]
        test_result = run_python_tests()
        test_count = validate_python_test_result(test_result, {"LauncherTests", "ReceiptTests", "ObserveTests"})
        commands.append({"argv": test_command, "exit_code": 0, "structured_result": test_result})
        atomic_json(output / "commands.json", {"commands": commands})

        selection = {
            "format": SELECTION_FORMAT, "schema_version": 1,
            "selected_stage_ids": [0], "selected_arm_ids": ["baseline"],
            "stages": [{"stage_id": stage, "disposition": "pending",
                        "reason": "host foundation passed; model verification and smoke pending" if stage == 0 else "not yet evaluated",
                        "evidence": [f"{output.name}/checks.json", f"{output.name}/receipt.json"] if stage == 0 else []}
                       for stage in range(8)],
        }
        validate_selection(selection)
        atomic_json(output.parent / "selection.json", selection)
        ended = time.time()
        artifact_names = ["checks.json", "commands.json"]
        receipt = {
            "format": RECEIPT_FORMAT, "schema_version": 1, "kind": "stage0", "qualified": True,
            "qualification_reasons": [], "command": ["gates.py", "--stage", "0", "--output", str(output)],
            "environment": {}, "started_at_unix": started, "ended_at_unix": ended,
            "duration_seconds": ended - started, "process": {"pid": os.getpid()},
            "memory": {"qualified": False, "reason": "host checks do not launch a full model"}, "vm": {},
            "result": {"exit_code": 0, "qualification_scope": "stage0-host-foundation",
                       "overall_stage_disposition": "pending", "native_checks": len(checks["checks"]), "python_tests": test_count,
                       "catalogue": checks["checks"]},
            "artifacts": {name: {"bytes": (output / name).stat().st_size, "sha256": sha256(output / name)}
                          for name in artifact_names},
            "identities": {"build_identity": identity, "harness_hashes": harness_hashes()},
        }
        atomic_json(output / "receipt.json", receipt)
        atomic_json(output / "completion.json", {"format": "slotstream-flash-completion-v1",
                    "receipt_sha256": sha256(output / "receipt.json"), "qualified": True,
                    "functional_success": True})
        return 0
    except Exception as error:
        atomic_json(output / "commands.json", {"commands": commands})
        atomic_json(output / "failure.json", {"format": "slotstream-flash-failure-v1",
                    "error": f"{type(error).__name__}: {error}"})
        return 1


def stage_m5(output: Path, *, runner: Callable[[list[str]], subprocess.CompletedProcess[str]] = _run,
             label: str = "m5", required_checks: set[str] = M5_REQUIRED_CHECKS) -> int:
    output = fresh_output(output)
    started = time.time()
    commands: list[dict[str, Any]] = []
    try:
        build_command = ["make", "build", "SLOTSTREAM_BUILD_JOBS=2"]
        built = runner(build_command)
        commands.append({"argv": build_command, "exit_code": built.returncode,
                         "stdout": built.stdout, "stderr": built.stderr})
        if built.returncode != 0:
            raise EvidenceError("two-job release build failed")
        binary = ROOT / ".build" / "release" / "slotstream"
        identity, _ = validate_build_identity(binary)
        check_command = [str(ROOT / ".build" / "release" / "slotstream-checks"),
                         "--tier", "t0", "--tier", "t1", "--filter", label, "--json"]
        checked = runner(check_command)
        commands.append({"argv": check_command, "exit_code": checked.returncode,
                         "stdout": checked.stdout, "stderr": checked.stderr})
        if checked.returncode != 0:
            raise EvidenceError("M5 native check catalogue failed")
        try: checks = json.loads(checked.stdout)
        except json.JSONDecodeError as error: raise EvidenceError(f"malformed M5 check JSON: {error}") from error
        validate_checks(checks, required=required_checks, exact_names=True)
        atomic_json(output / "checks.json", checks)
        test_result = run_python_tests()
        test_count = validate_python_test_result(test_result,
            {"LauncherTests", "ReceiptTests", "ObserveTests", "M5ProbeTests", "CacheStudyTests", "ReplayTests", "CaptureTests", "TokenizeTests"})
        commands.append({"argv": [sys.executable, "-m", "unittest", "discover", "-s", "Tools/flash"],
                         "exit_code": 0, "structured_result": test_result})
        atomic_json(output / "commands.json", {"commands": commands})
        ended = time.time()
        receipt = {
            "format": RECEIPT_FORMAT, "schema_version": 1, "kind": label, "qualified": True,
            "qualification_reasons": [], "command": ["gates.py", "--stage", label, "--output", str(output)],
            "environment": {}, "started_at_unix": started, "ended_at_unix": ended,
            "duration_seconds": ended - started, "process": {"pid": os.getpid()},
            "memory": {"qualified": False, "reason": "per-case MLX peak is recorded in checks.json"}, "vm": {},
            "result": {"exit_code": 0, "functional_success": True,
                       "qualification_scope": f"{label}-component-diagnostic",
                       "native_checks": len(checks["checks"]), "python_tests": test_count,
                       "observation_status": "unverified" if label == "m5" else "reference-unaccepted",
                       "catalogue": checks["checks"]},
            "artifacts": {name: {"bytes": (output / name).stat().st_size, "sha256": sha256(output / name)}
                          for name in ("checks.json", "commands.json")},
            "identities": {"build_identity": identity, "harness_hashes": harness_hashes()},
        }
        atomic_json(output / "receipt.json", receipt)
        atomic_json(output / "completion.json", {"format": "slotstream-flash-completion-v1",
                    "receipt_sha256": sha256(output / "receipt.json"), "qualified": True,
                    "functional_success": True})
        return 0
    except Exception as error:
        atomic_json(output / "commands.json", {"commands": commands})
        atomic_json(output / "failure.json", {"format": "slotstream-flash-failure-v1",
                    "error": f"{type(error).__name__}: {error}"})
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("0", "m5", "flash"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.stage == "0": return stage_zero(args.output)
    if args.stage == "m5": return stage_m5(args.output)
    return stage_m5(args.output, label="flash", required_checks=FLASH_REQUIRED_CHECKS)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except EvidenceError as error:
        raise SystemExit(str(error))
