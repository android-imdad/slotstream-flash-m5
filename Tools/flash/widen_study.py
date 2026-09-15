#!/usr/bin/env python3
"""Validate, measure, and qualify exact expert-code widening without changing defaults."""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import random
import statistics
import sys
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import benchmark
import capture
import cache_study
from common import (EvidenceError, atomic_json, fresh_output, harness_hashes, read_json,
                    sha256, validate_build_identity)
from receipts import require_terminal_sampling, validate_receipt_file

FIXTURE = ROOT / "Tools/fixtures/flash/cache-study.json"
REFERENCE_BINARY_SHA256 = "d3b0e7ffbcaf89186a8097a4dcd65493c595cf506adf7ef5098d787fc4a9c095"
MODEL_MEMORY_GB = 14.0
MODEL_SECONDS = 900.0
COMPONENT_MEMORY_GB = 0.512
COMPONENT_SECONDS = 120.0


def _number(value: Any, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise EvidenceError(f"invalid completed timing: {name}")
    return float(value)


def _median(values: list[float]) -> float:
    if not values:
        raise EvidenceError("paired cohort is empty")
    return float(statistics.median(values))


def paired_bootstrap(control: list[float], candidate: list[float], *,
                     resamples: int = 10_000, seed: int = 17) -> dict[str, Any]:
    if (len(control) != len(candidate) or not control or type(resamples) is not int
            or resamples < 1 or any(type(value) not in (int, float) or not math.isfinite(value)
                                    or value <= 0 for value in control + candidate)):
        raise EvidenceError("paired bootstrap inputs are invalid")
    improvements = [1.0 - new / old for old, new in zip(control, candidate)]
    generator = random.Random(seed)
    draws = []
    for _ in range(resamples):
        sample = [improvements[generator.randrange(len(improvements))]
                  for _ in improvements]
        draws.append(_median(sample))
    draws.sort()
    lower = draws[int(0.025 * (resamples - 1))]
    upper = draws[int(0.975 * (resamples - 1))]
    return {"seed": seed, "resamples": resamples, "pairs": len(improvements),
            "statistic": "median paired decode-time improvement fraction",
            "estimate": _median(improvements), "lower_95": lower, "upper_95": upper}


def qualification_decision(rows: list[dict[str, Any]], *, expected_pairs: int = 10,
                           resamples: int = 10_000) -> dict[str, Any]:
    prompts = {prompt["id"] for prompt in cache_study.load_fixture()["prompts"]}
    grouped = {prompt: [] for prompt in prompts}
    for row in rows:
        if not isinstance(row, dict) or row.get("prompt_id") not in grouped:
            raise EvidenceError("qualification row has an unknown prompt")
        grouped[row["prompt_id"]].append(row)
    workloads = {}
    admitted = True
    reasons = []
    for prompt in sorted(prompts):
        values = grouped[prompt]
        pair_indices = [row.get("pair_index") for row in values]
        if (len(values) != expected_pairs
                or any(type(index) is not int for index in pair_indices)
                or sorted(pair_indices) != list(range(expected_pairs))):
            raise EvidenceError(f"{prompt} is missing a predeclared pair")
        controls = [_number(row.get("scalar_decode_seconds"), "scalar decode") for row in values]
        candidates = [_number(row.get("packed_decode_seconds"), "packed decode") for row in values]
        first_control = [_number(row.get("scalar_first_token_seconds"), "scalar first token")
                         for row in values]
        first_candidate = [_number(row.get("packed_first_token_seconds"), "packed first token")
                           for row in values]
        if any(value <= 0 for value in first_control + first_candidate):
            raise EvidenceError("completed generation has no positive first-token timing")
        bootstrap = paired_bootstrap(controls, candidates, resamples=resamples, seed=17)
        first_regressions = [new / old - 1 for old, new in zip(first_control, first_candidate)]
        first_median = _median(first_regressions)
        passed = (bootstrap["estimate"] >= 0.10 and bootstrap["lower_95"] > 0
                  and first_median <= 0.05)
        if not passed:
            admitted = False
            reasons.append(f"{prompt} failed decode improvement/CI or first-token guardrail")
        workloads[prompt] = {"decode": bootstrap,
                             "median_first_token_regression_fraction": first_median,
                             "passed": passed}
    return {"disposition": "qualified" if admitted else "rejected",
            "qualified": admitted, "reasons": reasons, "workloads": workloads,
            "thresholds": {"minimum_median_decode_improvement_fraction": 0.10,
                           "bootstrap_interval_must_exclude_zero": True,
                           "maximum_median_first_token_regression_fraction": 0.05}}


def validate_stats(document: dict[str, Any], prompt: dict[str, Any], policy: str, *,
                   known_scalar_reference: bool = False) -> dict[str, float]:
    cache_study.validate_stats(document, prompt)
    effective = document.get("effective_expert_widening")
    if policy == "packed4-to6":
        if effective != policy:
            raise EvidenceError("packed run does not report the effective widening policy")
    elif effective != "scalar" and not (known_scalar_reference and effective is None):
        raise EvidenceError("scalar run has no verified scalar policy identity")
    stats = document["stats"]
    decode = _number(stats.get("decodeSeconds"), "decodeSeconds")
    first = _number(stats.get("firstTokenSeconds"), "firstTokenSeconds")
    prefill = _number(stats.get("prefillSeconds"), "prefillSeconds")
    request = _number(stats.get("requestSeconds"), "requestSeconds")
    load = _number(document.get("load_seconds"), "load_seconds")
    if decode <= 0 or first <= 0:
        raise EvidenceError("completed generation has no positive decode/first-token timing")
    return {"decode_seconds": decode, "first_token_seconds": first,
            "prefill_seconds": prefill, "request_seconds": request,
            "load_seconds": load,
            "decode_tokens_per_second": stats["decodeTokens"] / decode}


def validate_timing_eligibility(document: dict[str, Any], receipt: dict[str, Any]) -> None:
    """Exclude contaminated timings without changing functional acceptance.

    Global paging is a diagnostic for correctness, but this frozen performance
    study requires clean measured intervals. Endpoint thermal observations are
    the available evidence; they do not prove continuous nominal temperature.
    The generic launch receipt never qualifies caller-supplied model identities;
    this study binds those separately and checks the receipt's memory evidence.
    """
    memory = receipt.get("memory", {})
    if memory.get("qualified") is not True:
        raise EvidenceError("timing is ineligible: process memory was not qualified")
    stats = document.get("stats", {})
    for key in ("generatorSystemBefore", "generatorSystemAfter"):
        conditions = stats.get(key)
        if (not isinstance(conditions, dict)
                or conditions.get("thermalState") != "nominal"
                or conditions.get("lowPowerModeEnabled") is not False):
            raise EvidenceError("timing is ineligible: non-nominal or unknown operating conditions")
    vm = receipt.get("vm", {})
    for before, after in ((vm.get("before"), vm.get("after")),
                          (stats.get("generatorVMBefore"), stats.get("generatorVMAfter"))):
        if not isinstance(before, dict) or not isinstance(after, dict):
            raise EvidenceError("timing is ineligible: paging observations are missing")
        for key in ("swapins", "swapouts"):
            old, new = before.get(key), after.get(key)
            if (type(old) is not int or type(new) is not int or old < 0 or new < 0
                    or old != new):
                raise EvidenceError("timing is ineligible: paging counters changed or are unknown")


def _stable_plan(plan: Any) -> dict[str, Any]:
    if not isinstance(plan, dict):
        raise EvidenceError("runtime plan is missing")
    result = copy.deepcopy(plan)
    result.pop("device_available_gb", None)
    return result


def runtime_controls(document: dict[str, Any], receipt: dict[str, Any], *,
                     known_scalar_reference: bool = False) -> dict[str, Any]:
    plan = _stable_plan(document.get("plan"))
    numerical = document.get("numerical_environment")
    if numerical is None and known_scalar_reference:
        raw = receipt.get("environment", {}).get("MLX_ENABLE_TF32")
        numerical = {"mlx_enable_tf32_raw": raw, "effective_tf32": raw != "0"}
    expected_numerical = {"mlx_enable_tf32_raw", "effective_tf32"}
    if not isinstance(numerical, dict) or set(numerical) != expected_numerical:
        raise EvidenceError("runtime numerical environment is missing")
    effective_vision = document.get("effective_vision")
    if effective_vision is None and known_scalar_reference:
        effective_vision = plan.get("vision")
    controls = {"plan": plan, "memory_ledger": plan.get("memory_ledger"),
                "optimizations": document.get("optimizations"),
                "numerical_environment": numerical,
                "effective_pool_slots": document.get("effective_pool_slots"),
                "effective_prefill_chunk": document.get("effective_prefill_chunk"),
                "effective_mtp": document.get("effective_mtp"),
                "effective_vision": effective_vision,
                "launcher_environment": receipt.get("environment")}
    if (not isinstance(controls["memory_ledger"], dict)
            or not isinstance(controls["optimizations"], dict)
            or type(controls["effective_pool_slots"]) is not int
            or type(controls["effective_prefill_chunk"]) is not int
            or type(controls["effective_mtp"]) is not bool
            or type(controls["effective_vision"]) is not bool
            or not isinstance(controls["launcher_environment"], dict)):
        raise EvidenceError("runtime controls are incomplete")
    return controls


def compare_runtime_controls(left_document: dict[str, Any], left_receipt: dict[str, Any],
                             right_document: dict[str, Any], right_receipt: dict[str, Any], *,
                             left_known_scalar_reference: bool = False) -> None:
    left = runtime_controls(left_document, left_receipt,
                            known_scalar_reference=left_known_scalar_reference)
    right = runtime_controls(right_document, right_receipt)
    if left != right:
        changed = sorted(key for key in left if left.get(key) != right.get(key))
        raise EvidenceError(f"paired runtime controls changed: {changed}")


def compare_capture_runtime_controls(left: dict[str, Any], left_receipt: dict[str, Any],
                                     right: dict[str, Any], right_receipt: dict[str, Any]) -> None:
    def controls(report: dict[str, Any], receipt: dict[str, Any]) -> dict[str, Any]:
        return {"plan": _stable_plan(report.get("plan")),
                "memory_ledger": report.get("memory_ledger"),
                "optimizations": report.get("optimizations"),
                "numerical_environment": report.get("numerical_environment"),
                "launcher_environment": receipt.get("environment")}
    a, b = controls(left, left_receipt), controls(right, right_receipt)
    if (not isinstance(a["memory_ledger"], dict) or not isinstance(b["memory_ledger"], dict)
            or not isinstance(a["optimizations"], dict) or not isinstance(b["optimizations"], dict)
            or not isinstance(a["numerical_environment"], dict)
            or not isinstance(b["numerical_environment"], dict)):
        raise EvidenceError("capture runtime controls are incomplete")
    if a != b:
        changed = sorted(key for key in a if a.get(key) != b.get(key))
        raise EvidenceError(f"capture runtime controls changed: {changed}")


def _build_binding(binary: Path) -> dict[str, str]:
    identity, paths = validate_build_identity(binary)
    return {"binary_sha256": identity["binary_sha256"],
            "metallib_sha256": identity["metallib_sha256"],
            "source_archive_sha256": identity["source_archive_sha256"],
            "build_identity_sha256": sha256(paths["identity"])}


def _model_binding(model: Path) -> dict[str, str]:
    return {"path": str(model.resolve()), "config_sha256": sha256(model / "config.json"),
            "index_sha256": sha256(model / "model.safetensors.index.json")}


def _corpus_binding(corpus: Path) -> dict[str, str]:
    return {"path": str(corpus.resolve()),
            "index_sha256": sha256(corpus.resolve() / "corpus/corpus.json"),
            "completion_sha256": sha256(corpus.resolve() / "corpus/completion.json")}


def validate_parity_evidence(root: Path, *, candidate: dict[str, str],
                             model: dict[str, str], corpus: dict[str, str]) -> dict[str, Any]:
    root = root.resolve(strict=True)
    report_path = root / "report.json"
    report = read_json(report_path)
    completion = read_json(root / "completion.json")
    if completion != {"format": "slotstream-widening-study-completion-v1",
                      "report_sha256": sha256(report_path), "qualification": False}:
        raise EvidenceError("parity completion does not bind its report")
    required = {"format", "schema_version", "qualification", "exact_capture_parity",
                "ordinary_cli_id_parity", "reference_binary_sha256", "candidate",
                "model", "corpus", "captures", "ordinary", "harness_hashes"}
    if (set(report) != required or report.get("format") != "slotstream-widening-parity-v1"
            or report.get("schema_version") != 1 or report.get("qualification") is not False
            or report.get("exact_capture_parity") is not True
            or report.get("ordinary_cli_id_parity") is not True
            or report.get("reference_binary_sha256") != REFERENCE_BINARY_SHA256
            or report.get("candidate") != candidate or report.get("model") != model
            or report.get("corpus") != corpus or not report.get("captures")
            or not report.get("ordinary") or not isinstance(report.get("harness_hashes"), dict)):
        raise EvidenceError("parity evidence is missing, stale, or bound to another candidate")
    return report


def _exact_work(left: dict[str, Any], right: dict[str, Any]) -> None:
    if left.get("prompt_ids") != right.get("prompt_ids") or left.get("output_ids") != right.get("output_ids"):
        raise EvidenceError("paired widening arms changed exact prompt or output IDs")
    fields = ("decodeTokens", "promptTokens", "prefillTokens", "decodeForwardPasses",
              "prefillRecords", "decodeRecords", "prefillReadBytes", "decodeReadBytes",
              "finishReason")
    if any(left["stats"].get(key) != right["stats"].get(key) for key in fields):
        raise EvidenceError("paired widening arms did not perform identical model work")


def _run_arm(binary: Path, model: Path, prompt: dict[str, Any], policy: str,
             evidence: Path, run_set_id: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    stats_path = evidence / "stats.json"
    command = [str(binary), "run", "--model", str(model), "--memory-gb", "14",
               "--max-context", "2048", "--mtp", "off", "--vision", "off",
               "--prompt", prompt["text"], "--max-tokens", "128", "--greedy",
               "--seed", "7", "--sample-footprint", "--stats-json", str(stats_path)]
    if policy != "scalar":
        command.extend(["--expert-widening", policy])
    settled = benchmark.settle_before_model_launch()
    code = benchmark.launch(evidence, MODEL_MEMORY_GB, MODEL_SECONDS, command,
                            run_set_id=run_set_id,
                            model_hash=sha256(model / "config.json"))
    atomic_json(evidence / "settling.json", settled)
    receipt = validate_receipt_file(evidence / "receipt.json")
    require_terminal_sampling(receipt)
    if code or not receipt["result"]["functional_success"]:
        raise EvidenceError(f"{prompt['id']} {policy} monitored generation failed")
    stats = read_json(stats_path)
    return stats, receipt, settled


def _archive_candidate(binary: Path, output: Path) -> tuple[Path, dict[str, Any]]:
    archive = output / "candidate-archive"
    if benchmark.archive(binary, archive):
        raise EvidenceError("candidate archive failed")
    receipt = validate_receipt_file(archive / "receipt.json", require_qualified=True)
    candidate = archive / "bin/slotstream"
    validate_build_identity(candidate)
    return candidate, receipt


def _validate_check(check: Any, name: str, expected_items: set[str]) -> None:
    if (not isinstance(check, dict) or check.get("name") != name or check.get("passed") is not True
            or not isinstance(check.get("items"), list) or not check["items"]):
        raise EvidenceError(f"{name} exact check is missing")
    actual = {item.get("name") for item in check["items"] if isinstance(item, dict)}
    if not expected_items.issubset(actual) or any(item.get("passed") is not True for item in check["items"]):
        raise EvidenceError(f"{name} expected exact checks are incomplete")


def validate_component_native(native: Path, launch: Path, receipt: dict[str, Any],
                              binary: Path, *, synthetic: bool,
                              model: Path | None) -> dict[str, Any]:
    report_path = native / "report.json"
    report = read_json(report_path)
    completion = read_json(native / "completion.json")
    if completion != {"format": "slotstream-widening-completion-v1", "passed": True,
                      "report_sha256": sha256(report_path), "schema_version": 1}:
        raise EvidenceError("widening component completion does not bind its report")
    selected_key = "synthetic" if synthetic else "model"
    if (set(report) != {"format", "schemaVersion", "processID", "mode", "provenance",
                        selected_key}
            or report.get("format") != "slotstream-widening-command-v1"
            or report.get("schemaVersion") != 1
            or report.get("processID") != receipt.get("process", {}).get("pid")
            or report.get("mode") != ("synthetic" if synthetic else "bounded-model-reader")
            or not isinstance(report.get(selected_key), dict)):
        raise EvidenceError("widening component command identity is malformed")
    binary = binary.resolve(strict=True)
    build, paths = validate_build_identity(binary)
    provenance = report.get("provenance")
    expected_provenance = {"executablePath": str(binary),
                           "executableSHA256": build["binary_sha256"],
                           "metallibSHA256": build["metallib_sha256"],
                           "buildIdentitySHA256": sha256(paths["identity"]),
                           "sourceArchiveSHA256": build["source_archive_sha256"]}
    if (not isinstance(provenance, dict)
            or set(provenance) != set(expected_provenance) | {"compilerVersion"}
            or any(provenance.get(key) != value for key, value in expected_provenance.items())
            or not isinstance(provenance.get("compilerVersion"), str)
            or not provenance["compilerVersion"].strip()):
        raise EvidenceError("widening component native provenance differs from the binary")
    executable = receipt.get("identities", {}).get("executable")
    if (not isinstance(executable, dict) or Path(executable.get("path", "")).resolve() != binary
            or executable.get("sha256") != build["binary_sha256"]
            or receipt.get("command", [])[:2] != [str(binary), "widening-check"]):
        raise EvidenceError("widening component launcher does not bind the native executable")
    artifact = receipt.get("artifacts", {}).get("native/report.json")
    if (not isinstance(artifact, dict) or artifact.get("sha256") != sha256(report_path)
            or artifact.get("bytes") != report_path.stat().st_size):
        raise EvidenceError("widening component native report is outside the launcher receipt")
    exact_items = {"all 65536 two-byte inputs match an independent bit oracle",
                   "overlap retains scalar clear-then-expand behavior",
                   "real 1638400-code projection is byte-identical",
                   "omitted policy remains scalar"}
    selected = report[selected_key]
    if synthetic:
        if (selected.get("format") != "slotstream-widening-synthetic-v1"
                or selected.get("schemaVersion") != 1 or selected.get("pairCount") != 8
                or selected.get("codeCountPerCall") != 1_638_400
                or selected.get("allocatedBufferBytes") != 32_768_000
                or type(selected.get("processFootprintEndBytes")) is not int
                or selected["processFootprintEndBytes"] <= 0):
            raise EvidenceError("synthetic widening component metadata is malformed")
        _validate_check(selected.get("exactCheck"), "exact-widening", exact_items)
        results = selected.get("results")
        expected = {(workers, policy) for workers in (1, 8, 16)
                    for policy in ("scalar", "packed4-to6")}
        if (not isinstance(results, list) or len(results) != 6
                or {(item.get("workers"), item.get("policy")) for item in results} != expected):
            raise EvidenceError("synthetic widening worker/policy coverage is incomplete")
        checksums = {}
        for item in results:
            durations = item.get("secondsPerCall")
            if (not isinstance(durations, list) or len(durations) != 8
                    or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                           for value in durations)
                    or type(item.get("medianSecondsPerCall")) not in (int, float)
                    or not math.isfinite(item["medianSecondsPerCall"])
                    or item["medianSecondsPerCall"] != _median(durations)
                    or item["medianSecondsPerCall"] <= 0 or type(item.get("checksum")) is not int):
                raise EvidenceError("synthetic widening timing/checksum evidence is malformed")
            checksums.setdefault(item["workers"], set()).add(item["checksum"])
        if any(len(values) != 1 for values in checksums.values()):
            raise EvidenceError("synthetic scalar and packed checksums differ")
    else:
        if model is None or Path(selected.get("modelPath", "")).resolve() != model.resolve():
            raise EvidenceError("model component is bound to another checkpoint")
        if (selected.get("format") != "slotstream-widening-model-component-v1"
                or selected.get("schemaVersion") != 1
                or selected.get("checkpointFormat") != "jang6S"
                or selected.get("policies") != ["scalar", "packed4-to6"]
                or selected.get("sourceIdentityBefore") != selected.get("sourceIdentityAfter")
                or not 0 < selected.get("uniqueOriginalRegionBytes", 0) < 256 << 20
                or selected.get("uniqueOriginalRegionLimitBytes") != 256 << 20
                or selected.get("physicalFootprintLimitBytes") != 512_000_000
                or selected.get("logicalSourceBytesRead") != 1_720_320_000):
            raise EvidenceError("model widening component identity or bounds are malformed")
        model_check_items = {"checkpoint identity is unchanged",
                             "all descriptor controls are observed successful",
                             "packed widening rejects a bypassing packed expert layout"}
        model_check_items |= {f"{api} layer {layer} exact full tensor bytes"
                              for api in ("readBatchChecked", "readRunsChecked")
                              for layer in (0, 5, 22)}
        _validate_check(selected.get("check"), "widening-model-component", model_check_items)
        cases = selected.get("cases")
        expected = {(api, layer) for api in ("readBatchChecked", "readRunsChecked")
                    for layer in (0, 5, 22)}
        if (not isinstance(cases, list) or len(cases) != 6
                or {(case.get("api"), case.get("layer")) for case in cases} != expected):
            raise EvidenceError("model widening API/layer coverage is incomplete")
        classes = {0: 2, 5: 1, 22: 0}
        depths = {"readBatchChecked": 32, "readRunsChecked": 12}
        for case in cases:
            if (case.get("queueDepth") != depths[case["api"]]
                    or case.get("expectedWideningProjections") != classes[case["layer"]]
                    or case.get("observedWideningProjections") != classes[case["layer"]]
                    or case.get("experts") != list(range(10)) or case.get("exactBytes") is not True
                    or case.get("comparedBytes") != 43_008_000
                    or not isinstance(case.get("outputSHA256"), str)
                    or len(case["outputSHA256"]) != 64
                    or len(case.get("pairOrders", [])) != 4
                    or case["pairOrders"] != [["scalar", "packed4-to6"],
                                               ["packed4-to6", "scalar"],
                                               ["scalar", "packed4-to6"],
                                               ["packed4-to6", "scalar"]]
                        and case["pairOrders"] != [["packed4-to6", "scalar"],
                                                   ["scalar", "packed4-to6"],
                                                   ["packed4-to6", "scalar"],
                                                   ["scalar", "packed4-to6"]]):
                raise EvidenceError("model widening case metadata is malformed")
            timings = case.get("scalarSeconds", []) + case.get("packedSeconds", [])
            if (len(timings) != 8 or any(type(value) not in (int, float)
                    or not math.isfinite(value) or value <= 0 for value in timings)):
                raise EvidenceError("model widening case timings are incomplete")
            if (case.get("scalarMedianSeconds") != _median(case["scalarSeconds"])
                    or case.get("packedMedianSeconds") != _median(case["packedSeconds"])):
                raise EvidenceError("model widening case medians are stale")
        controls = selected.get("readControls")
        if (not isinstance(controls, list) or not controls
                or any(item.get("noCacheReturnCode") != 0 or item.get("readAheadReturnCode") != 0
                       for item in controls)):
            raise EvidenceError("model widening read controls are incomplete")
        for key in ("diskBytesReadBefore", "diskBytesReadAfter"):
            if type(selected.get(key)) is not int or selected[key] < 0:
                raise EvidenceError("model widening disk-byte observation is unavailable")
        if (selected["diskBytesReadAfter"] < selected["diskBytesReadBefore"]
                or not isinstance(selected.get("diskObservationScope"), str)
                or not selected["diskObservationScope"]):
            raise EvidenceError("model widening disk-byte observation is malformed")
    return report


def component(binary: Path, output: Path, *, synthetic: bool, model: Path | None) -> int:
    output = fresh_output(output)
    try:
        binary = binary.resolve(strict=True)
        validate_build_identity(binary)
        if synthetic == (model is not None):
            raise EvidenceError("choose exactly one component mode")
        if any(key.startswith("SLOTSTREAM_") for key in __import__("os").environ):
            raise EvidenceError("ambient Slotstream controls are unsupported")
        launch_dir = output / "launch"
        native = launch_dir / "native"
        command = [str(binary), "widening-check",
                   "--synthetic" if synthetic else "--model",
                   str(model.resolve(strict=True)) if model is not None else "",
                   "--output", str(native)]
        if synthetic:
            command = [item for item in command if item]
        code = benchmark.launch(launch_dir, COMPONENT_MEMORY_GB, COMPONENT_SECONDS, command,
                                run_set_id="widening-component-v1",
                                model_hash=None if model is None else sha256(model / "config.json"))
        receipt = validate_receipt_file(launch_dir / "receipt.json")
        require_terminal_sampling(receipt)
        if code or not receipt["result"]["functional_success"]:
            raise EvidenceError("monitored widening component failed")
        if receipt["memory"]["peak_bytes"] > int(COMPONENT_MEMORY_GB * 1e9):
            raise EvidenceError("widening component exceeded 512 MB")
        validate_component_native(native, launch_dir, receipt, binary,
                                  synthetic=synthetic, model=model)
        result = {"format": "slotstream-widening-component-evidence-v1", "schema_version": 1,
                  "mode": "synthetic" if synthetic else "model", "qualification": False,
                  "binary_sha256": sha256(binary), "report_sha256": sha256(native / "report.json"),
                  "receipt_sha256": sha256(launch_dir / "receipt.json"),
                  "terminal_before_reap": True, "peak_bytes": receipt["memory"]["peak_bytes"],
                  "memory_limit_bytes": 512_000_000, "harness_hashes": harness_hashes()}
        atomic_json(output / "report.json", result)
        atomic_json(output / "completion.json", {"format": "slotstream-widening-study-completion-v1",
                    "report_sha256": sha256(output / "report.json"), "qualification": False})
        return 0
    except Exception as error:
        atomic_json(output / "failure.json", {"format": "slotstream-widening-study-failure-v1",
                    "error": f"{type(error).__name__}: {error}"})
        return 1


def performance(mode: str, model: Path, candidate_binary: Path, corpus: Path,
                parity_evidence: Path, output: Path, *, pairs: int) -> int:
    output = fresh_output(output)
    try:
        if mode == "screen" and pairs != 1:
            raise EvidenceError("screen requires exactly one pair per prompt")
        if mode == "qualify" and pairs != 10:
            raise EvidenceError("qualification requires exactly 10 pairs per prompt")
        if any(key.startswith("SLOTSTREAM_") for key in __import__("os").environ):
            raise EvidenceError("ambient Slotstream controls are unsupported")
        fixture = cache_study.load_fixture(FIXTURE)
        frozen_harness = harness_hashes()
        model = model.resolve(strict=True)
        verification = cache_study.verified_model_revision(model)
        source_geometry = cache_study.derive_source_geometry(model, verification=verification)
        binary, archive_receipt = _archive_candidate(candidate_binary, output)
        frozen_identity, _ = validate_build_identity(binary)
        candidate_binding = _build_binding(binary)
        model_binding = _model_binding(model)
        corpus_binding = _corpus_binding(corpus)
        parity_report = validate_parity_evidence(parity_evidence,
            candidate=candidate_binding, model=model_binding, corpus=corpus_binding)
        parity_report_path = parity_evidence.resolve() / "report.json"
        rows = []
        run_set = f"widening-{mode}-v1"
        for prompt_index, prompt in enumerate(fixture["prompts"]):
            for pair_index in range(pairs):
                order = (["scalar", "packed4-to6"]
                         if (prompt_index + pair_index) % 2 == 0
                         else ["packed4-to6", "scalar"])
                documents = {}
                receipts = {}
                arm_evidence = {}
                for policy in order:
                    evidence = output / "prompts" / prompt["id"] / f"pair-{pair_index:02d}" / policy
                    stats, receipt, settled = _run_arm(binary, model, prompt, policy,
                                                       evidence, run_set)
                    timings = validate_stats(stats, prompt, policy)
                    validate_timing_eligibility(stats, receipt)
                    documents[policy] = stats
                    receipts[policy] = receipt
                    arm_evidence[policy] = {"path": str(evidence.relative_to(output)),
                                            "receipt_sha256": sha256(evidence / "receipt.json"),
                                            "stats_sha256": sha256(evidence / "stats.json"),
                                            "settling_sha256": sha256(evidence / "settling.json"),
                                            "timings": timings}
                    if validate_build_identity(binary)[0] != frozen_identity:
                        raise EvidenceError("candidate build identity changed during cohort")
                    if harness_hashes() != frozen_harness:
                        raise EvidenceError("widening harness changed during cohort")
                _exact_work(documents["scalar"], documents["packed4-to6"])
                compare_runtime_controls(documents["scalar"], receipts["scalar"],
                                         documents["packed4-to6"], receipts["packed4-to6"])
                scalar = arm_evidence["scalar"]["timings"]
                packed = arm_evidence["packed4-to6"]["timings"]
                rows.append({"prompt_id": prompt["id"], "pair_index": pair_index,
                             "order": order, "prompt_ids": documents["scalar"]["prompt_ids"],
                             "output_ids": documents["scalar"]["output_ids"],
                             "scalar_decode_seconds": scalar["decode_seconds"],
                             "packed_decode_seconds": packed["decode_seconds"],
                             "scalar_first_token_seconds": scalar["first_token_seconds"],
                             "packed_first_token_seconds": packed["first_token_seconds"],
                             "arms": arm_evidence})
        if cache_study.derive_source_geometry(model, verification=verification) != source_geometry:
            raise EvidenceError("model source identity changed during cohort")
        if mode == "qualify":
            decision = qualification_decision(rows, expected_pairs=10, resamples=10_000)
        else:
            improvements = [1 - row["packed_decode_seconds"] / row["scalar_decode_seconds"]
                            for row in rows]
            promising = all(value > 0 for value in improvements)
            decision = {"disposition": "promising" if promising else "not-promising",
                        "promising": promising,
                        "criterion": "packed decode is faster in each of the three frozen workloads",
                        "paired_decode_improvement_fractions": improvements}
        report = {"format": f"slotstream-widening-{mode}-v1", "schema_version": 1,
                  "qualification": mode == "qualify" and decision.get("qualified") is True,
                  "scope": "three frozen short-prompt workloads on this device/checkpoint/budget only",
                  "fixture_sha256": sha256(FIXTURE), "model_verification": verification,
                  "parity_evidence_path": str(parity_evidence.resolve()),
                  "parity_evidence_sha256": sha256(parity_report_path),
                  "candidate_archive_receipt_sha256": sha256(output / "candidate-archive/receipt.json"),
                  "candidate_binary_sha256": sha256(binary), "pairs_per_prompt": pairs,
                  "rows": rows, "decision": decision, "harness_hashes": frozen_harness}
        atomic_json(output / "report.json", report)
        atomic_json(output / "completion.json", {"format": "slotstream-widening-study-completion-v1",
                    "report_sha256": sha256(output / "report.json"),
                    "qualification": report["qualification"]})
        return 0
    except Exception as error:
        atomic_json(output / "failure.json", {"format": "slotstream-widening-study-failure-v1",
                    "error": f"{type(error).__name__}: {error}"})
        return 1


def parity(model: Path, reference: Path, candidate_binary: Path, corpus: Path,
           output: Path) -> int:
    output = fresh_output(output)
    try:
        model = model.resolve(strict=True)
        reference_binary = (reference / "bin/slotstream").resolve(strict=True)
        reference_identity, _ = validate_build_identity(reference_binary, historical=True)
        if (sha256(reference_binary) != REFERENCE_BINARY_SHA256
                or reference_identity.get("binary_sha256") != REFERENCE_BINARY_SHA256):
            raise EvidenceError("immutable scalar reference identity changed")
        candidate, _ = _archive_candidate(candidate_binary, output)
        candidate_binding = _build_binding(candidate)
        frozen_candidate_identity, _ = validate_build_identity(candidate)
        model_binding = _model_binding(model)
        corpus_binding = _corpus_binding(corpus)
        frozen_harness = harness_hashes()
        if any(key.startswith("SLOTSTREAM_") for key in __import__("os").environ):
            raise EvidenceError("ambient Slotstream controls are unsupported")
        import tokenize_corpus
        validated = tokenize_corpus.validate_corpus(corpus, model=model)
        shards = [entry for entry in validated["index"]["shards"]
                  if entry.get("split") == "development"]
        if not shards:
            raise EvidenceError("accepted natural corpus has no development shard")
        captures = []
        for ordinal, entry in enumerate(shards):
            manifest = corpus.resolve() / "corpus" / entry["path"]
            reference_report, reference_receipt, reference_settle = capture._native(
                reference_binary, model, manifest, "development", "reference-off",
                output / "captures" / f"{ordinal:03d}-scalar")
            if (validate_build_identity(candidate)[0] != frozen_candidate_identity
                    or harness_hashes() != frozen_harness):
                raise EvidenceError("source or harness changed after parity scalar capture arm")
            candidate_report, candidate_receipt, candidate_settle = capture._native(
                candidate, model, manifest, "development", "reference-off",
                output / "captures" / f"{ordinal:03d}-packed", widening_policy="packed4-to6")
            if capture._capture_identity(reference_report) != capture._capture_identity(candidate_report):
                raise EvidenceError(f"capture parity failed for {entry['path']}")
            if (validate_build_identity(candidate)[0] != frozen_candidate_identity
                    or harness_hashes() != frozen_harness):
                raise EvidenceError("source or harness changed after parity packed capture arm")
            compare_capture_runtime_controls(reference_report, reference_receipt,
                                             candidate_report, candidate_receipt)
            if harness_hashes() != frozen_harness:
                raise EvidenceError("widening harness changed during parity")
            captures.append({"shard": entry["path"],
                             "scalar_report_sha256": sha256(output / "captures" / f"{ordinal:03d}-scalar/native/report.json"),
                             "packed_report_sha256": sha256(output / "captures" / f"{ordinal:03d}-packed/native/report.json"),
                             "scalar_receipt_sha256": sha256(output / "captures" / f"{ordinal:03d}-scalar/receipt.json"),
                             "packed_receipt_sha256": sha256(output / "captures" / f"{ordinal:03d}-packed/receipt.json")})
        ordinary = []
        for prompt in cache_study.load_fixture(FIXTURE)["prompts"]:
            scalar, scalar_receipt, _ = _run_arm(reference_binary, model, prompt, "scalar",
                                    output / "ordinary" / prompt["id"] / "scalar",
                                    "widening-parity-v1")
            if (validate_build_identity(candidate)[0] != frozen_candidate_identity
                    or harness_hashes() != frozen_harness):
                raise EvidenceError("source or harness changed after parity scalar arm")
            packed, packed_receipt, _ = _run_arm(candidate, model, prompt, "packed4-to6",
                                    output / "ordinary" / prompt["id"] / "packed",
                                    "widening-parity-v1")
            if (validate_build_identity(candidate)[0] != frozen_candidate_identity
                    or harness_hashes() != frozen_harness):
                raise EvidenceError("source or harness changed after parity packed arm")
            validate_stats(scalar, prompt, "scalar", known_scalar_reference=True)
            validate_stats(packed, prompt, "packed4-to6")
            _exact_work(scalar, packed)
            compare_runtime_controls(scalar, scalar_receipt, packed, packed_receipt,
                                     left_known_scalar_reference=True)
            ordinary.append({"prompt_id": prompt["id"], "prompt_ids": scalar["prompt_ids"],
                             "output_ids": scalar["output_ids"]})
        report = {"format": "slotstream-widening-parity-v1", "schema_version": 1,
                  "qualification": False, "exact_capture_parity": True,
                  "ordinary_cli_id_parity": True, "reference_binary_sha256": REFERENCE_BINARY_SHA256,
                  "candidate": candidate_binding, "model": model_binding, "corpus": corpus_binding,
                  "captures": captures, "ordinary": ordinary,
                  "harness_hashes": frozen_harness}
        atomic_json(output / "report.json", report)
        atomic_json(output / "completion.json", {"format": "slotstream-widening-study-completion-v1",
                    "report_sha256": sha256(output / "report.json"), "qualification": False})
        return 0
    except Exception as error:
        atomic_json(output / "failure.json", {"format": "slotstream-widening-study-failure-v1",
                    "error": f"{type(error).__name__}: {error}"})
        return 1


def self_test() -> int:
    import unittest
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromName("test_widen_study"))
    return 0 if result.testsRun and not result.failures and not result.errors and not result.skipped else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    commands = parser.add_subparsers(dest="command")
    component_parser = commands.add_parser("component")
    component_parser.add_argument("--binary", type=Path, required=True)
    group = component_parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--synthetic", action="store_true")
    group.add_argument("--model", type=Path)
    component_parser.add_argument("--output", type=Path, required=True)
    parity_parser = commands.add_parser("parity")
    for name in ("model", "reference", "candidate", "corpus", "output"):
        parity_parser.add_argument(f"--{name}", type=Path, required=True)
    for name, default_pairs in (("screen", 1), ("qualify", 10)):
        sub = commands.add_parser(name)
        sub.add_argument("--model", type=Path, required=True)
        sub.add_argument("--candidate", type=Path, required=True)
        sub.add_argument("--corpus", type=Path, required=True)
        sub.add_argument("--parity", type=Path, required=True)
        sub.add_argument("--pairs", type=int, default=default_pairs)
        sub.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.self_test:
        if args.command is not None:
            parser.error("--self-test cannot be combined with a command")
        return self_test()
    if args.command == "component":
        return component(args.binary, args.output, synthetic=args.synthetic, model=args.model)
    if args.command == "parity":
        return parity(args.model, args.reference, args.candidate, args.corpus, args.output)
    if args.command in ("screen", "qualify"):
        return performance(args.command, args.model, args.candidate, args.corpus,
                           args.parity, args.output, pairs=args.pairs)
    parser.error("choose --self-test, component, parity, screen, or qualify")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except EvidenceError as error:
        raise SystemExit(str(error))
