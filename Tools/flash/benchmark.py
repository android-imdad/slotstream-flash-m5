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
from observe import (DarwinExitObserver, DarwinSampler, ProcessExitedDuringSample,
                     SamplingError, TERMINAL_SAMPLING_POLICY)
from prefill_bench import preflight, terminate_child_tree, vm_snapshot

ALLOWED_ENVIRONMENT = ("LANG", "LC_ALL", "MLX_ENABLE_TF32", "SLOTSTREAM_FLASH_MODE",
                       "SLOTSTREAM_M5_DISPATCH_LOG", "SLOTSTREAM_M5_TRACE_CASE",
                       "SLOTSTREAM_ROUTER_TRACE", "SLOTSTREAM_OPT_RESIDENT_OVERLAP", "SLOTSTREAM_PREFIX_CACHE")
MODEL_SETTLE_SECONDS = 2.0
TERMINAL_CONFIRM_SECONDS = 0.1
TERMINAL_CONFIRM_INTERVAL_SECONDS = 0.005


def settle_before_model_launch(*, seconds: float = MODEL_SETTLE_SECONDS,
                               sleep: Callable[[float], None] = time.sleep,
                               clock: Callable[[], float] = time.monotonic) -> dict[str, Any]:
    """Apply the fixed post-process VM-stat settling window outside model timing."""
    if type(seconds) not in (int, float) or not (0 < seconds <= 60):
        raise EvidenceError("model settle seconds must be finite and between 0 and 60")
    started = clock()
    sleep(float(seconds))
    ended = clock()
    return {"policy": "fixed-before-full-model-launch", "requested_seconds": float(seconds),
            "observed_elapsed_seconds": max(0.0, ended - started)}


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


def _cleanup_owned_child(child: subprocess.Popen[bytes], exit_observer: Any | None) -> int:
    """Reap an already-exited owned child, otherwise drain its owned tree."""
    if exit_observer is not None:
        try:
            exited = exit_observer.observe()
        except Exception:
            exited = None
        if exited is not None:
            try:
                return child.wait(timeout=5)
            except Exception:
                pass
    try:
        terminate_child_tree(child)
    except PermissionError:
        # The root can become an unreaped zombie between WNOHANG and TERM.
        return child.wait(timeout=5)
    if child.returncode is None:
        raise RuntimeError("owned child cleanup did not reap the process")
    return child.returncode


def launch(output: Path, memory_gb: float, max_seconds: float, command: list[str], *,
           run_set_id: str | None = None, model_hash: str | None = None,
           sampler_factory: Callable[[int], Any] = DarwinSampler,
           exit_observer_factory: Callable[[int], Any] = DarwinExitObserver,
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
    live_sample_count = 0
    terminal_sample_count = 0
    terminal_exit_abstime: int | None = None
    terminal_lifetime_peak: int | None = None
    peak: int | None = None
    child: subprocess.Popen[bytes] | None = None
    process: dict[str, Any] = {"pid": None, "process_group": None, "start_identity": None}
    before = _vm_safe(vm_provider)
    after: dict[str, Any] = {}
    exit_code: int | None = None
    child_reaped = False
    exit_observer: Any | None = None
    lifecycle_evidence_failure = False
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
                exit_observer = exit_observer_factory(child.pid)
                sampler = sampler_factory(child.pid)
                process["start_identity"] = sampler.start_identity
                next_sample = started

                def record_sample(observed: dict[str, Any], kind: str, now: float) -> None:
                    nonlocal sample_count, live_sample_count, terminal_sample_count
                    nonlocal peak, terminal_exit_abstime, terminal_lifetime_peak, budget_exceeded
                    if observed.get("start_abstime") != sampler.start_identity:
                        raise SamplingError("PID identity changed while sampling")
                    observed["sample_kind"] = kind
                    observed["elapsed_seconds"] = now - started
                    memory_stream.write(json.dumps(observed, sort_keys=True) + "\n")
                    memory_stream.flush()
                    sample_count += 1
                    if kind == "live":
                        live_sample_count += 1
                    else:
                        terminal_sample_count += 1
                        terminal_exit_abstime = observed["exit_abstime"]
                        terminal_lifetime_peak = observed["lifetime_peak_bytes"]
                    observed_peak = max(observed["physical_footprint_bytes"],
                                        observed["lifetime_peak_bytes"])
                    peak = observed_peak if peak is None else max(peak, observed_peak)
                    if observed_peak > memory_gb * 1e9:
                        budget_exceeded = True
                        failures.append("memory budget exceeded")

                def finish_exited(exited: dict[str, int], now: float) -> None:
                    nonlocal exit_code, child_reaped, sampler_error, lifecycle_evidence_failure
                    terminal = dict(sampler.terminal_sample(exited))
                    try:
                        reaped = child.wait(timeout=5)
                        child_reaped = True
                    except Exception as error:
                        lifecycle_evidence_failure = True
                        atomic_json(output / "rejected-terminal-sample.json", {
                            "reason": f"reap failed: {type(error).__name__}: {error}",
                            "non_reaping_observation": exited,
                            "terminal_rusage": terminal,
                        })
                        raise SamplingError("terminal child reap failed") from error
                    exit_code = reaped
                    if reaped != exited["exit_code"]:
                        lifecycle_evidence_failure = True
                        sampler_error = "SamplingError: reaped exit status differs from non-reaping observation"
                        failures.append("reaped exit status differs from non-reaping observation")
                        atomic_json(output / "rejected-terminal-sample.json", {
                            "reason": "reaped exit status differs from non-reaping observation",
                            "actual_reaped_exit_code": reaped,
                            "non_reaping_observation": exited,
                            "terminal_rusage": terminal,
                        })
                        return
                    record_sample(terminal, "terminal", now)
                    if exit_code != 0:
                        failures.append(f"child exited {exit_code}")

                def confirm_terminal_exit(now: float) -> dict[str, int] | None:
                    nonlocal timed_out, interrupted
                    deadline = min(started + max_seconds, now + TERMINAL_CONFIRM_SECONDS)
                    while True:
                        if cancelled():
                            interrupted = True
                            failures.append("cancelled during terminal exit confirmation")
                            return None
                        current = clock()
                        if current - started > max_seconds:
                            timed_out = True
                            failures.append("timeout during terminal exit confirmation")
                            return None
                        exited = exit_observer.observe()
                        if exited is not None:
                            return exited
                        if current >= deadline:
                            raise SamplingError(
                                "process exit was not confirmed after terminal rusage appeared")
                        sleep(min(TERMINAL_CONFIRM_INTERVAL_SECONDS, deadline - current))

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
                    exited = exit_observer.observe()
                    if exited is not None:
                        finish_exited(exited, now)
                        break
                    if now >= next_sample:
                        try:
                            observed = dict(sampler.sample())
                            record_sample(observed, "live", now)
                        except ProcessExitedDuringSample:
                            exited = confirm_terminal_exit(now)
                            if exited is None:
                                break
                            finish_exited(exited, now)
                            break
                        except Exception as error:
                            sampler_error = f"{type(error).__name__}: {error}"
                            failures.append("memory sampler failed")
                            break
                        if budget_exceeded:
                            break
                        next_sample = now + 0.5
                    sleep(min(0.05, max(0.0, next_sample - clock())))
            except Exception as error:
                sampler_error = f"{type(error).__name__}: {error}"
                failures.append("memory sampler unavailable")
            finally:
                if not child_reaped:
                    try:
                        exit_code = _cleanup_owned_child(child, exit_observer)
                        child_reaped = True
                    except Exception as error:
                        failures.append(f"cleanup unverified: {type(error).__name__}: {error}")
    except KeyboardInterrupt:
        interrupted = True
        failures.append("SIGINT")
        if child is not None and not child_reaped:
            try:
                exit_code = _cleanup_owned_child(child, exit_observer)
                child_reaped = True
            except Exception as cleanup:
                failures.append(f"cleanup unverified: {type(cleanup).__name__}: {cleanup}")
    except Exception as error:
        failures.append(f"launch failed: {type(error).__name__}: {error}")
        if child is not None and not child_reaped:
            try:
                exit_code = _cleanup_owned_child(child, exit_observer)
                child_reaped = True
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
    evidence_failure = lifecycle_evidence_failure or len(failures) > evidence_failure_start
    functional_success = (not failures and exit_code == 0 and sample_count > 0
                          and terminal_sample_count == 1)
    qualified = False
    receipt = {
        "format": RECEIPT_FORMAT, "schema_version": 1, "kind": "launch",
        "qualified": qualified, "qualification_reasons": failures + identity_reasons,
        "command": command, "environment": _environment(),
        "started_at_unix": started_wall, "ended_at_unix": ended_wall,
        "duration_seconds": max(0.0, ended_wall - started_wall), "process": process,
        "memory": {"target_gb_decimal": memory_gb, "sample_interval_seconds": 0.5,
                   "sampling_policy": TERMINAL_SAMPLING_POLICY,
                   "qualified": sample_count > 0 and terminal_sample_count == 1
                                and sampler_error is None and not budget_exceeded,
                   "peak_bytes": peak, "sample_count": sample_count, "samples_artifact": "memory.jsonl",
                   "live_sample_count": live_sample_count,
                   "terminal_sample_count": terminal_sample_count,
                   "terminal_exit_abstime": terminal_exit_abstime,
                   "terminal_lifetime_peak_bytes": terminal_lifetime_peak,
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
    for module in ("test_launcher", "test_receipts", "test_observe"):
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
