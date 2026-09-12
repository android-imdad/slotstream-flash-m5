#!/usr/bin/env python3
"""Build an isolated pinned-MLX copy with dispatch-site logging."""
from __future__ import annotations

import argparse
import difflib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import benchmark
from common import EvidenceError, atomic_json, fresh_output, harness_hashes, sha256, validate_build_identity
from m5_probe import bounded_child_succeeded, run_bounded_export, validate_m5_report
from receipts import validate_receipt_file

PINNED_MLX_REVISION = "0bb916c67f4b9e5c682cbe02a42c701c93ab5021"


def validate_inventory_delta(original: dict[str, str], instrumented: dict[str, str]) -> list[str]:
    if set(original) != set(instrumented):
        raise EvidenceError("private MLX inventory changed paths")
    changed = [path for path in original if original[path] != instrumented[path]]
    expected = ["Source/Cmlx/mlx/mlx/backend/metal/quantized.cpp"]
    if changed != expected:
        raise EvidenceError(f"private MLX delta is not exactly quantized.cpp: {changed}")
    return changed


def validate_bound_hash(actual: str, expected: str, label: str) -> None:
    if actual != expected: raise EvidenceError(f"changed {label}")


def validate_private_completion(value: dict[str, Any], result_hash: str) -> None:
    if (set(value) != {"format", "result_sha256", "functional_success"}
            or value["format"] != "slotstream-private-mlx-dispatch-completion-v1"
            or value["functional_success"] is not True or value["result_sha256"] != result_hash):
        raise EvidenceError("private dispatch completion mismatch")


def instrument_quantized(source: str) -> str:
    if "SLOTSTREAM_M5_DISPATCH_LOG" in source:
        raise EvidenceError("MLX source is already instrumented")
    source = source.replace('#include "mlx/utils.h"\n',
        '#include "mlx/utils.h"\n#include <cstdio>\n#include <cstdlib>\n#include <unistd.h>\n', 1)
    qmv_anchor = "  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);\n}\n\nvoid gather_qvm("
    qmv_replacement = '''  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
  if (std::getenv("SLOTSTREAM_M5_DISPATCH_LOG")) {
    std::fprintf(stderr,
        "SLOTSTREAM_M5_DISPATCH case=%s pid=%d path=gather_qmv kernel=%s M=%d N=%d K=%d B=%d bits=%d submitted=1\\n",
        std::getenv("SLOTSTREAM_M5_TRACE_CASE") ?: "unknown", getpid(), kname.c_str(), M, N, K, B, bits);
    std::fflush(stderr);
  }
}

void gather_qvm('''
    if source.count(qmv_anchor) != 1:
        raise EvidenceError("gather_qmv dispatch anchor changed")
    source = source.replace(qmv_anchor, qmv_replacement, 1)
    nax_anchor = "  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);\n}\n\nvoid gather_qmm_rhs("
    nax_replacement = '''  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
  if (std::getenv("SLOTSTREAM_M5_DISPATCH_LOG")) {
    std::fprintf(stderr,
        "SLOTSTREAM_M5_DISPATCH case=%s pid=%d path=gather_qmm_rhs_nax kernel=%s M=%d N=%d K=%d B=%d bits=%d submitted=1\\n",
        std::getenv("SLOTSTREAM_M5_TRACE_CASE") ?: "unknown", getpid(), hash_name.c_str(), M, N, K, M, bits);
    std::fflush(stderr);
  }
}

void gather_qmm_rhs('''
    if source.count(nax_anchor) != 1:
        raise EvidenceError("gather_qmm_rhs_nax dispatch anchor changed")
    return source.replace(nax_anchor, nax_replacement, 1)


def private_package(original: str, mlx: Path) -> str:
    replacements = {
        '.package(url: "https://github.com/ml-explore/mlx-swift.git", .upToNextMinor(from: "0.31.6"))':
            f'.package(path: "{mlx}")',
        '.package(url: "https://github.com/huggingface/swift-transformers.git", from: "1.3.0")':
            f'.package(path: "{ROOT / ".build/checkouts/swift-transformers"}")',
        '.package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.3.0")':
            f'.package(path: "{ROOT / ".build/checkouts/swift-argument-parser"}")',
    }
    for old, new in replacements.items():
        if original.count(old) != 1: raise EvidenceError(f"Package dependency anchor changed: {old}")
        original = original.replace(old, new)
    return original


def source_inventory(root: Path) -> dict[str, str]:
    result = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if not path.is_file() or ".git" in relative.parts or ".build" in relative.parts:
            continue
        if path.is_symlink(): raise EvidenceError(f"dependency source symlink is not accepted: {path}")
        result[str(relative)] = sha256(path)
    if not result: raise EvidenceError(f"dependency source inventory is empty: {root}")
    return result


def git_identity(path: Path) -> dict[str, Any]:
    revision = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain",
                                      "--untracked-files=no"], text=True)
    if status.strip(): raise EvidenceError(f"dependency checkout is dirty: {path}: {status}")
    return {"path": str(path), "revision": revision, "inventory": source_inventory(path)}


def copy_sources(output: Path) -> tuple[Path, Path, Path, Path]:
    app = output / "app"
    mlx = output / "deps" / "mlx-swift"
    shutil.copytree(ROOT / "Sources", app / "Sources", symlinks=False)
    (app / "Tools").mkdir(parents=True)
    for name in ("build_identity.py", "fetch_metallib.sh"):
        shutil.copy2(ROOT / "Tools" / name, app / "Tools" / name)
    for name in ("Package.resolved", "Makefile"):
        shutil.copy2(ROOT / name, app / name)
    original_mlx = ROOT / ".build/checkouts/mlx-swift"
    original_identity = git_identity(original_mlx)
    submodules = subprocess.check_output(["git", "-C", str(original_mlx), "submodule", "status", "--recursive"], text=True).splitlines()
    if any(not line.startswith(" ") for line in submodules):
        raise EvidenceError(f"MLX submodule is missing or changed: {submodules}")
    shutil.copytree(original_mlx, mlx,
                    ignore=shutil.ignore_patterns(".git", ".build"), symlinks=False)
    package = private_package((ROOT / "Package.swift").read_text(), mlx)
    (app / "Package.swift").write_text(package)
    quantized = mlx / "Source/Cmlx/mlx/mlx/backend/metal/quantized.cpp"
    quantized.chmod(quantized.stat().st_mode | stat.S_IWUSR)
    original = quantized.read_text()
    modified = instrument_quantized(original)
    quantized.write_text(modified)
    patch = "".join(difflib.unified_diff(original.splitlines(True), modified.splitlines(True),
                                         fromfile="pinned/quantized.cpp", tofile="instrumented/quantized.cpp"))
    patch_path = output / "quantized-dispatch.patch"
    patch_path.write_text(patch)
    instrumented_inventory = source_inventory(mlx)
    original_inventory = original_identity.pop("inventory")
    changed = validate_inventory_delta(original_inventory, instrumented_inventory)
    dependencies = {
        "mlx-swift": {**original_identity, "submodules": submodules,
                      "original_inventory": original_inventory,
                      "instrumented_inventory": instrumented_inventory,
                      "changed_paths": changed},
        "swift-transformers": git_identity(ROOT / ".build/checkouts/swift-transformers"),
        "swift-argument-parser": git_identity(ROOT / ".build/checkouts/swift-argument-parser"),
    }
    dependency_path = output / "dependency-identities.json"
    atomic_json(dependency_path, dependencies)
    return app, mlx, patch_path, dependency_path


def _build(app: Path, output: Path) -> tuple[Path, dict[str, Any]]:
    scratch = output / "scratch"
    resolved = run_bounded_export(["swift", "package", "--package-path", str(app),
                                   "--scratch-path", str(scratch), "resolve"],
                                  output / "resolve.stdout.txt", output / "resolve.stderr.txt",
                                  timeout_seconds=300, max_bytes=32 << 20, memory_gb=6.0)
    if not bounded_child_succeeded(resolved):
        raise EvidenceError(f"private dependency resolution failed: {resolved}")
    show = subprocess.run(["swift", "build", "--package-path", str(app), "--scratch-path", str(scratch),
                           "-c", "release", "--show-bin-path"], text=True, capture_output=True, timeout=30)
    if show.returncode != 0: raise EvidenceError(f"private Swift bin path failed: {show.stderr}")
    binary_dir = Path(show.stdout.strip())
    before = subprocess.run([sys.executable, str(app / "Tools/build_identity.py"), "before", str(binary_dir)],
                            cwd=app, text=True, capture_output=True, timeout=30)
    if before.returncode != 0: raise EvidenceError(f"private pre-build identity failed: {before.stderr}")
    build = run_bounded_export(["swift", "build", "--package-path", str(app), "--scratch-path", str(scratch),
                                "-c", "release", "-j", "2"],
                               output / "build.stdout.txt", output / "build.stderr.txt",
                               timeout_seconds=900, max_bytes=64 << 20, memory_gb=6.0)
    if not bounded_child_succeeded(build):
        raise EvidenceError(f"private two-job build failed: {build}")
    shutil.copy2(ROOT / "Tools/lib/mlx-0.31.1.metallib", binary_dir / "mlx.metallib")
    after = subprocess.run([sys.executable, str(app / "Tools/build_identity.py"), "after", str(binary_dir)],
                           cwd=app, text=True, capture_output=True, timeout=60)
    if after.returncode != 0: raise EvidenceError(f"private post-build identity failed: {after.stderr}")
    binary = binary_dir / "slotstream"
    validate_build_identity(binary, root=app)
    return binary, {"resolve": resolved, "compile": build}


LOG = re.compile(r"^SLOTSTREAM_M5_DISPATCH case=(\w+) pid=(\d+) path=(\w+) kernel=(\S+) "
                 r"M=(\d+) N=(\d+) K=(\d+) B=(\d+) bits=(\d+) submitted=1$", re.M)


def validate_dispatch_claim(case: str, matches: list[tuple[str, ...]], report_pid: int) -> tuple[str, ...]:
    expected_path = "gather_qmm_rhs_nax" if case == "grouped6" else "gather_qmv"
    expected_batch = 16 if case == "grouped6" else 1
    accepted = [groups for groups in matches if groups[0] == case and int(groups[1]) == report_pid
                and groups[2] == expected_path and int(groups[4]) == expected_batch
                and int(groups[5]) == 640 and int(groups[6]) == 2560
                and int(groups[7]) == expected_batch and int(groups[8]) == 6]
    if not accepted: raise EvidenceError(f"instrumented {case} did not log its actual expected dispatch")
    unexpected = [groups for groups in matches if groups[0] == case and int(groups[1]) == report_pid
                  and groups[2] != expected_path]
    if unexpected: raise EvidenceError(f"instrumented {case} logged an unexpected competing dispatch: {unexpected}")
    kernel = accepted[0][3]
    if case == "grouped6" and not kernel.startswith("affine_gather_qmm_rhs_nax_"):
        raise EvidenceError("grouped selected case kernel family is not exact")
    if case == "decode6" and not kernel.startswith("affine_gather_qmv"):
        raise EvidenceError("decode selected case kernel family is not exact")
    return accepted[0]


def run_case(binary: Path, app: Path, output: Path, case: str, *, suffix: str = "") -> dict[str, Any]:
    evidence = output / f"{case}-evidence{suffix}"
    diagnostic = evidence / "diagnostic"
    old_log, old_case = os.environ.get("SLOTSTREAM_M5_DISPATCH_LOG"), os.environ.get("SLOTSTREAM_M5_TRACE_CASE")
    os.environ["SLOTSTREAM_M5_DISPATCH_LOG"] = "1"
    os.environ["SLOTSTREAM_M5_TRACE_CASE"] = case
    try:
        code = benchmark.launch(evidence, 2.0, 120.0,
            [str(binary), "m5-check", "--synthetic", "--trace-case", case, "--output", str(diagnostic)])
    finally:
        if old_log is None: os.environ.pop("SLOTSTREAM_M5_DISPATCH_LOG", None)
        else: os.environ["SLOTSTREAM_M5_DISPATCH_LOG"] = old_log
        if old_case is None: os.environ.pop("SLOTSTREAM_M5_TRACE_CASE", None)
        else: os.environ["SLOTSTREAM_M5_TRACE_CASE"] = old_case
    receipt = validate_receipt_file(evidence / "receipt.json")
    if code != 0 or not receipt["result"]["functional_success"]:
        raise EvidenceError(f"instrumented {case} launcher failed")
    report = validate_m5_report(diagnostic / "report.json", binary, repo_root=app)
    matches = [match.groups() for match in LOG.finditer((evidence / "stderr.txt").read_text())]
    accepted = validate_dispatch_claim(case, matches, report["processID"])
    return {"case": case, "path": accepted[2], "kernel": accepted[3],
            "pid": int(accepted[1]), "M": int(accepted[4]), "N": int(accepted[5]),
            "K": int(accepted[6]), "B": int(accepted[7]), "bits": int(accepted[8]),
            "gpu_eval_and_numerical_report_passed": report["diagnostic"]["check"]["passed"],
            "launcher_receipt": str((evidence / "receipt.json").relative_to(output)),
            "launcher_receipt_sha256": sha256(evidence / "receipt.json")}


def validate_result(output: Path, *, require_completion: bool = True) -> dict[str, Any]:
    result_path = output / "result.json"
    result = json.loads(result_path.read_text())
    required = {"format", "schema_version", "functional_success", "production_equivalent_build",
                "instrumentation_scope", "pinned_mlx_revision", "source_patch",
                "dependency_identities",
                "original_quantized_sha256", "instrumented_quantized_sha256", "private_binary",
                "production_binary_sha256", "private_build_identity_sha256",
                "private_source_archive_sha256", "private_metallib_sha256", "build", "cases",
                "claim_boundary"}
    if (set(result) != required or result["format"] != "slotstream-private-mlx-dispatch-v1"
            or result["schema_version"] != 1 or result["functional_success"] is not True
            or result["production_equivalent_build"] is not False
            or result["pinned_mlx_revision"] != PINNED_MLX_REVISION):
        raise EvidenceError("invalid private dispatch result envelope")
    patch = output / result["source_patch"]["path"]
    dependency_path = output / result["dependency_identities"]["path"]
    binary = output / result["private_binary"]["path"]
    if (sha256(patch) != result["source_patch"]["sha256"]
            or sha256(dependency_path) != result["dependency_identities"]["sha256"]
            or sha256(binary) != result["private_binary"]["sha256"]
            or sha256(output / "deps/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/quantized.cpp")
                != result["instrumented_quantized_sha256"]
            or sha256(ROOT / ".build/release/slotstream") != result["production_binary_sha256"]
            or sha256(ROOT / ".build/checkouts/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/quantized.cpp")
                != result["original_quantized_sha256"]):
        raise EvidenceError("private dispatch source or binary identity mismatch")
    dependencies = json.loads(dependency_path.read_text())
    if set(dependencies) != {"mlx-swift", "swift-transformers", "swift-argument-parser"}:
        raise EvidenceError("private local dependency identities are incomplete")
    mlx = dependencies["mlx-swift"]
    if (mlx.get("revision") != PINNED_MLX_REVISION
            or mlx.get("changed_paths") != ["Source/Cmlx/mlx/mlx/backend/metal/quantized.cpp"]
            or set(mlx.get("original_inventory", {})) != set(mlx.get("instrumented_inventory", {}))
            or any(not line.startswith(" ") for line in mlx.get("submodules", []))):
        raise EvidenceError("private MLX inventory or submodule identity is invalid")
    changed = [path for path in mlx["original_inventory"]
               if mlx["original_inventory"][path] != mlx["instrumented_inventory"][path]]
    if changed != mlx["changed_paths"]:
        raise EvidenceError("private MLX inventory contains an undeclared delta")
    if source_inventory(ROOT / ".build/checkouts/mlx-swift") != mlx["original_inventory"]:
        raise EvidenceError("pinned MLX source no longer matches the recorded original inventory")
    if source_inventory(output / "deps/mlx-swift") != mlx["instrumented_inventory"]:
        raise EvidenceError("instrumented MLX copy no longer matches its full inventory")
    for name in ("swift-transformers", "swift-argument-parser"):
        dependency = dependencies[name]
        current = Path(dependency["path"])
        if (subprocess.check_output(["git", "-C", str(current), "rev-parse", "HEAD"], text=True).strip()
                != dependency["revision"] or source_inventory(current) != dependency["inventory"]):
            raise EvidenceError(f"private local dependency identity changed: {name}")
    private_identity, private_paths = validate_build_identity(binary, root=output / "app")
    if (sha256(private_paths["identity"]) != result["private_build_identity_sha256"]
            or private_identity["source_archive_sha256"] != result["private_source_archive_sha256"]
            or sha256(private_paths["metallib"]) != result["private_metallib_sha256"]):
        raise EvidenceError("private build identity, source archive, or metallib mismatch")
    expected = {"grouped6": ("gather_qmm_rhs_nax", 16), "decode6": ("gather_qmv", 1)}
    if {case.get("case") for case in result["cases"]} != set(expected):
        raise EvidenceError("private dispatch cases are incomplete")
    for case in result["cases"]:
        path, batch = expected[case["case"]]
        if (case["path"] != path or case["M"] != batch or case["B"] != batch
                or case["N"] != 640 or case["K"] != 2560 or case["bits"] != 6
                or case["gpu_eval_and_numerical_report_passed"] is not True):
            raise EvidenceError(f"invalid private dispatch case: {case['case']}")
        receipt_path = (output / case["launcher_receipt"]).resolve(strict=True)
        if not receipt_path.is_relative_to(output.resolve()):
            raise EvidenceError(f"changed private case receipt: {case['case']}")
        validate_bound_hash(sha256(receipt_path), case["launcher_receipt_sha256"],
                            f"private case receipt: {case['case']}")
        receipt = validate_receipt_file(receipt_path)
        if (not receipt["result"]["functional_success"]
                or receipt["identities"].get("harness_hashes") != harness_hashes()):
            raise EvidenceError(f"private case receipt is stale or failed: {case['case']}")
        report = validate_m5_report(receipt_path.parent / "diagnostic/report.json", binary,
                                    repo_root=output / "app")
        matches = [match.groups() for match in LOG.finditer((receipt_path.parent / "stderr.txt").read_text())]
        accepted = validate_dispatch_claim(case["case"], matches, report["processID"])
        actual = {"path": accepted[2], "kernel": accepted[3], "pid": int(accepted[1]),
                  "M": int(accepted[4]), "N": int(accepted[5]), "K": int(accepted[6]),
                  "B": int(accepted[7]), "bits": int(accepted[8])}
        if any(case[key] != value for key, value in actual.items()):
            raise EvidenceError(f"private case claim differs from its log: {case['case']}")
    if require_completion:
        completion = json.loads((output / "completion.json").read_text())
        validate_private_completion(completion, sha256(result_path))
    return result


def run(output_path: Path) -> int:
    output = fresh_output(output_path)
    try:
        original_revision = subprocess.check_output(
            ["git", "-C", str(ROOT / ".build/checkouts/mlx-swift"), "rev-parse", "HEAD"], text=True).strip()
        if original_revision != PINNED_MLX_REVISION:
            raise EvidenceError(f"unexpected pinned MLX revision: {original_revision}")
        app, mlx, patch_path, dependency_path = copy_sources(output)
        binary, build = _build(app, output)
        cases = [run_case(binary, app, output, case) for case in ("grouped6", "decode6")]
        identity, paths = validate_build_identity(binary, root=app)
        result = {"format": "slotstream-private-mlx-dispatch-v1", "schema_version": 1,
                  "functional_success": True, "production_equivalent_build": False,
                  "instrumentation_scope": "private pinned MLX copy; dispatch-site stderr logging only",
                  "pinned_mlx_revision": original_revision,
                  "source_patch": {"path": patch_path.name, "sha256": sha256(patch_path)},
                  "dependency_identities": {"path": dependency_path.name,
                                            "sha256": sha256(dependency_path)},
                  "original_quantized_sha256": sha256(ROOT / ".build/checkouts/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/quantized.cpp"),
                  "instrumented_quantized_sha256": sha256(mlx / "Source/Cmlx/mlx/mlx/backend/metal/quantized.cpp"),
                  "private_binary": {"path": str(binary.relative_to(output)), "sha256": sha256(binary)},
                  "production_binary_sha256": sha256(ROOT / ".build/release/slotstream"),
                  "private_build_identity_sha256": sha256(paths["identity"]),
                  "private_source_archive_sha256": identity["source_archive_sha256"],
                  "private_metallib_sha256": sha256(paths["metallib"]),
                  "build": build, "cases": cases,
                  "claim_boundary": "instrumented diagnostic proves the pinned source dispatch choice; it is not production-binary utilization or an LLM speedup"}
        atomic_json(output / "result.json", result)
        validate_result(output, require_completion=False)
        atomic_json(output / "completion.json", {"format": "slotstream-private-mlx-dispatch-completion-v1",
                    "result_sha256": sha256(output / "result.json"), "functional_success": True})
        validate_result(output)
        return 0
    except Exception as error:
        atomic_json(output / "failure.json", {"format": "slotstream-private-mlx-dispatch-failure-v1",
                    "error": f"{type(error).__name__}: {error}"})
        return 1


def resume(output: Path) -> int:
    output = output.resolve(strict=True)
    try:
        previous = json.loads((output / "result.json").read_text())
        binary = output / previous["private_binary"]["path"]
        app = output / "app"
        validate_build_identity(binary, root=app)
        shutil.copy2(output / "result.json", output / "result-superseded.json")
        if (output / "completion.json").exists():
            shutil.copy2(output / "completion.json", output / "completion-superseded.json")
        version = 2
        while any((output / f"{case}-evidence-v{version}").exists()
                  for case in ("grouped6", "decode6")):
            version += 1
        previous["cases"] = [run_case(binary, app, output, case, suffix=f"-v{version}")
                             for case in ("grouped6", "decode6")]
        previous["production_binary_sha256"] = sha256(ROOT / ".build/release/slotstream")
        atomic_json(output / "result.json", previous)
        validate_result(output, require_completion=False)
        atomic_json(output / "completion.json", {"format": "slotstream-private-mlx-dispatch-completion-v1",
                    "result_sha256": sha256(output / "result.json"), "functional_success": True})
        validate_result(output)
        return 0
    except Exception as error:
        atomic_json(output / "resume-failure.json", {"format": "slotstream-private-mlx-dispatch-failure-v1",
                    "error": f"{type(error).__name__}: {error}"})
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--resume", type=Path)
    args = parser.parse_args(argv)
    return run(args.output) if args.output else resume(args.resume)


if __name__ == "__main__":
    raise SystemExit(main())
