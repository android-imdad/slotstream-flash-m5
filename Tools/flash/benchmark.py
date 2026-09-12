#!/usr/bin/env python3
"""Launch monitored Flash experiments or archive an exact Slotstream build."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "Tools"))

from common import (EvidenceError, RECEIPT_FORMAT, atomic_json, copy_verified, fresh_output,
                    harness_hashes, sha256, validate_build_identity)
from observe import DarwinSampler, SamplingError
from prefill_bench import preflight, terminate_child_tree, vm_snapshot

ALLOWED_ENVIRONMENT = ("LANG", "LC_ALL", "MLX_ENABLE_TF32", "SLOTSTREAM_FLASH_MODE")


def _environment() -> dict[str, str]:
    return {key: os.environ[key] for key in ALLOWED_ENVIRONMENT if key in os.environ}


def _artifacts(output: Path, names: list[str]) -> dict[str, dict[str, Any]]:
    result = {}
    for name in names:
        path = output / name
        result[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    return result


def _all_artifacts(output: Path, failures: list[str]) -> dict[str, dict[str, Any]]:
    result = {}
    for path in sorted(output.rglob("*")):
        if path.is_symlink():
            failures.append(f"child produced a symlink artifact: {path.relative_to(output)}")
            continue
        if path.is_file() and path.name not in ("receipt.json", "completion.json", "failure.json"):
            name = str(path.relative_to(output))
            result[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    return result


def _vm_safe(provider: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        return provider()
    except Exception as error:
        return {"unavailable": f"{type(error).__name__}: {error}"}


def launch(output: Path, memory_gb: float, max_seconds: float, command: list[str], *,
           run_set_id: str | None = None, model_hash: str | None = None,
           sampler_factory: Callable[[int], Any] = DarwinSampler,
           preflight_func: Callable[[float], dict[str, Any]] = preflight,
           vm_provider: Callable[[], dict[str, Any]] = vm_snapshot,
           clock: Callable[[], float] = time.monotonic,
           wall_clock: Callable[[], float] = time.time,
           sleep: Callable[[float], None] = time.sleep,
           cancelled: Callable[[], bool] = lambda: False) -> int:
    if type(memory_gb) not in (int, float) or not (0 < memory_gb < 1024):
        raise EvidenceError("memory-gb must be finite and between 0 and 1024")
    if type(max_seconds) not in (int, float) or not (0 < max_seconds <= 86400):
        raise EvidenceError("max-seconds must be between 0 and 86400")
    if not command or not all(isinstance(item, str) and item for item in command):
        raise EvidenceError("a nonempty argv is required after --")
    output = fresh_output(output)
    stdout_path, stderr_path = output / "stdout.txt", output / "stderr.txt"
    started_wall, started = wall_clock(), clock()
    failures: list[str] = []
    sample_count = 0
    peak: int | None = None
    child: subprocess.Popen[bytes] | None = None
    process: dict[str, Any] = {"pid": None, "process_group": None, "start_identity": None}
    before = _vm_safe(vm_provider)
    after: dict[str, Any] = {}
    exit_code: int | None = None
    timed_out = interrupted = budget_exceeded = False
    sampler_error: str | None = None
    executable = Path(command[0])
    if not executable.is_absolute():
        found = shutil.which(command[0])
        executable = Path(found) if found else ROOT / command[0]
    executable_identity = None
    try:
        executable = executable.resolve(strict=True)
        executable_identity = {"path": str(executable), "bytes": executable.stat().st_size, "sha256": sha256(executable)}
    except OSError:
        pass
    memory_path = output / "memory.jsonl"
    try:
        # The shared preflight owns the lock check and measures target + 3 GB.
        before = preflight_func(memory_gb + 3.0)
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr, memory_path.open("w") as memory_stream:
            child = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr, start_new_session=True)
            process.update(pid=child.pid, process_group=child.pid)
            try:
                sampler = sampler_factory(child.pid)
                process["start_identity"] = sampler.start_identity
                next_sample = started
                while True:
                    now = clock()
                    if cancelled():
                        interrupted = True
                        failures.append("cancelled")
                        break
                    if now - started > max_seconds:
                        timed_out = True
                        failures.append("timeout")
                        break
                    if now >= next_sample:
                        try:
                            observed = dict(sampler.sample())
                            if observed.get("start_abstime") != sampler.start_identity:
                                raise SamplingError("PID identity changed while sampling")
                            observed["elapsed_seconds"] = now - started
                            memory_stream.write(json.dumps(observed, sort_keys=True) + "\n")
                            memory_stream.flush()
                            sample_count += 1
                            observed_peak = max(observed["physical_footprint_bytes"], observed["lifetime_peak_bytes"])
                            peak = observed_peak if peak is None else max(peak, observed_peak)
                            if max(observed["physical_footprint_bytes"], observed["lifetime_peak_bytes"]) > memory_gb * 1e9:
                                budget_exceeded = True
                                failures.append("memory budget exceeded")
                                break
                        except Exception as error:
                            sampler_error = f"{type(error).__name__}: {error}"
                            failures.append("memory sampler failed")
                            break
                        next_sample = now + 0.5
                    exit_code = child.poll()
                    if exit_code is not None:
                        if not sample_count:
                            failures.append("child exited before first valid memory sample")
                        if exit_code != 0:
                            failures.append(f"child exited {exit_code}")
                        break
                    sleep(min(0.05, max(0.0, next_sample - clock())))
            except Exception as error:
                sampler_error = f"{type(error).__name__}: {error}"
                failures.append("memory sampler unavailable")
            finally:
                if child.poll() is None:
                    try:
                        terminate_child_tree(child)
                    except Exception as error:
                        failures.append(f"cleanup unverified: {type(error).__name__}: {error}")
                exit_code = child.poll()
    except KeyboardInterrupt:
        interrupted = True
        failures.append("SIGINT")
        if child is not None and child.poll() is None:
            terminate_child_tree(child)
        exit_code = child.poll() if child is not None else None
    except Exception as error:
        failures.append(f"launch failed: {type(error).__name__}: {error}")
        if child is not None and child.poll() is None:
            try:
                terminate_child_tree(child)
            except Exception as cleanup:
                failures.append(f"cleanup unverified: {type(cleanup).__name__}: {cleanup}")
        if not stdout_path.exists(): stdout_path.touch()
        if not stderr_path.exists(): stderr_path.touch()
        if not memory_path.exists(): memory_path.touch()
    finally:
        after = _vm_safe(vm_provider)

    identity_reasons = []
    if run_set_id is None: identity_reasons.append("run-set identity unavailable")
    else: identity_reasons.append("run-set identity is caller-supplied and unverified")
    if model_hash is None: identity_reasons.append("model hash unavailable")
    else: identity_reasons.append("model hash is caller-supplied and unverified")
    ended_wall = wall_clock()
    evidence_failure_start = len(failures)
    if executable_identity is not None:
        try:
            if sha256(Path(executable_identity["path"])) != executable_identity["sha256"]:
                failures.append("executable changed during launch")
        except OSError as error:
            failures.append(f"executable unavailable after launch: {type(error).__name__}: {error}")
    artifacts = _all_artifacts(output, failures)
    evidence_failure = len(failures) > evidence_failure_start
    functional_success = not failures and exit_code == 0 and sample_count > 0
    qualified = False
    receipt = {
        "format": RECEIPT_FORMAT, "schema_version": 1, "kind": "launch",
        "qualified": qualified, "qualification_reasons": failures + identity_reasons,
        "command": command, "environment": _environment(),
        "started_at_unix": started_wall, "ended_at_unix": ended_wall,
        "duration_seconds": max(0.0, ended_wall - started_wall), "process": process,
        "memory": {"target_gb_decimal": memory_gb, "sample_interval_seconds": 0.5,
                   "qualified": sample_count > 0 and sampler_error is None and not budget_exceeded,
                   "peak_bytes": peak, "sample_count": sample_count, "samples_artifact": "memory.jsonl",
                   "sampler_error": sampler_error},
        "vm": {"before": before, "after": after},
        "result": {"exit_code": exit_code, "functional_success": functional_success,
                   "timed_out": timed_out, "interrupted": interrupted,
                   "budget_exceeded": budget_exceeded, "evidence_failure": evidence_failure},
        "artifacts": artifacts,
        "identities": {"run_set_id": run_set_id, "model_hash": model_hash,
                       "identity_status": "unverified", "executable": executable_identity,
                       "harness_hashes": harness_hashes(),
                       "missing_reason": None if run_set_id and model_hash else "caller did not supply both identities"},
    }
    atomic_json(output / "receipt.json", receipt)
    if functional_success:
        atomic_json(output / "completion.json", {"format": "slotstream-flash-completion-v1",
                    "receipt_sha256": sha256(output / "receipt.json"), "qualified": qualified,
                    "functional_success": True})
    return 0 if functional_success else 1


def archive(binary: Path, output: Path, *, repo_root: Path = ROOT) -> int:
    output = fresh_output(output)
    started = time.time()
    try:
        identity, paths = validate_build_identity(binary, root=repo_root)
        destination = output / "bin"
        destination.mkdir()
        copied = {}
        for key, source in paths.items():
            target_name = source.name
            copied[f"bin/{target_name}"] = copy_verified(source, destination / target_name, within=source.parent)
        archived_identity, _ = validate_build_identity(destination / binary.name, root=repo_root)
        if archived_identity != identity:
            raise EvidenceError("archived build identity changed")
        artifacts = {name: {"bytes": entry["bytes"], "sha256": entry["sha256"]} for name, entry in copied.items()}
        receipt = {
            "format": RECEIPT_FORMAT, "schema_version": 1, "kind": "archive", "qualified": True,
            "qualification_reasons": [], "command": ["archive", "--binary", str(binary), "--output", str(output)],
            "environment": _environment(), "started_at_unix": started, "ended_at_unix": time.time(),
            "duration_seconds": time.time() - started, "process": {"pid": os.getpid()},
            "memory": {"qualified": False, "reason": "archive does not execute the model"}, "vm": {},
            "result": {"exit_code": 0}, "artifacts": artifacts,
            "identities": {"build_identity": identity, "harness_hashes": harness_hashes()},
        }
        atomic_json(output / "receipt.json", receipt)
        atomic_json(output / "completion.json", {"format": "slotstream-flash-completion-v1",
                    "receipt_sha256": sha256(output / "receipt.json"), "qualified": True,
                    "functional_success": True})
        return 0
    except Exception as error:
        atomic_json(output / "failure.json", {"format": "slotstream-flash-failure-v1", "error": f"{type(error).__name__}: {error}"})
        return 1


def self_test() -> int:
    import unittest
    suite = unittest.TestSuite()
    for module in ("test_launcher", "test_receipts"):
        suite.addTests(unittest.defaultTestLoader.loadTestsFromName(module))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if result.testsRun <= 0 or result.failures or result.errors or result.skipped:
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    subparsers = parser.add_subparsers(dest="action")
    launch_parser = subparsers.add_parser("launch")
    launch_parser.add_argument("--output", type=Path, required=True)
    launch_parser.add_argument("--memory-gb", type=float, required=True)
    launch_parser.add_argument("--max-seconds", type=float, required=True)
    launch_parser.add_argument("--run-set-id")
    launch_parser.add_argument("--model-hash")
    launch_parser.add_argument("command", nargs=argparse.REMAINDER)
    archive_parser = subparsers.add_parser("archive")
    archive_parser.add_argument("--binary", type=Path, required=True)
    archive_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.self_test:
        if args.action is not None: parser.error("--self-test cannot be combined with a command")
        return self_test()
    if args.action == "launch":
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        return launch(args.output, args.memory_gb, args.max_seconds, command,
                      run_set_id=args.run_set_id, model_hash=args.model_hash)
    if args.action == "archive":
        return archive(args.binary, args.output)
    parser.error("choose --self-test, launch, or archive")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except EvidenceError as error:
        raise SystemExit(str(error))
