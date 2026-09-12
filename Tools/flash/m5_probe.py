#!/usr/bin/env python3
"""Run the bounded M5 diagnostic, optionally under an owned xctrace child."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import selectors
import subprocess
import sys
import time
from typing import Any
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import benchmark
from common import EvidenceError, atomic_json, fresh_output, read_json, sha256, validate_build_identity
from observe import DarwinSampler
from prefill_bench import preflight, terminate_child_tree
from receipts import validate_receipt_file


def assess_actual_dispatch(records: list[Any]) -> dict[str, Any]:
    matched = []
    decode = []
    for record in records:
        if not isinstance(record, dict):
            continue
        if (record.get("source") == "xctrace-export-row"
                and record.get("record_type") == "metal-compute-encoder"
                and record.get("case") == "grouped6" and record.get("target_pid_match") is True
                and record.get("process")
                and record.get("encoder")
                and isinstance(record.get("kernel_name"), str)
                and "gather_qmm_rhs_nax" in record["kernel_name"]):
            matched.append(record["kernel_name"])
        if (isinstance(record, dict) and record.get("source") == "xctrace-export-row"
                and record.get("record_type") == "metal-compute-encoder"
                and record.get("case") == "decode6" and record.get("target_pid_match") is True
                and record.get("process") and record.get("encoder")
                and isinstance(record.get("kernel_name"), str)
                and "qmv" in record["kernel_name"] and "nax" not in record["kernel_name"]):
            decode.append(record["kernel_name"])
    if matched and decode:
        return {"status": "observed", "path": "nax-grouped-rhs-with-decode-control",
                "record_count": len(matched) + len(decode),
                "grouped_records": len(matched), "decode_control_records": len(decode),
                "reason": "xctrace rows bind both NAX grouped RHS and non-NAX decode control kernels"}
    return {"status": "unverified", "path": None, "record_count": 0,
            "grouped_records": len(matched), "decode_control_records": len(decode),
            "reason": "exported Metal rows did not bind both NAX grouped RHS and decode control kernels"}


def metal_schemas(toc_text: str) -> list[str]:
    try:
        root = ET.fromstring(toc_text)
    except ET.ParseError as error:
        raise EvidenceError(f"invalid xctrace TOC XML: {error}") from error
    relevant = {"metal-gpu-intervals", "metal-application-encoders-list",
                "metal-object-label", "metal-shader-profiler-shader-list"}
    result = set()
    for table in root.iter("table"):
        schema = table.attrib.get("schema", "")
        if schema in relevant: result.add(schema)
    return sorted(result)


def _element_value(element: ET.Element, ids: dict[str, ET.Element]) -> str:
    target = ids.get(element.attrib.get("ref", ""), element)
    values = [target.attrib.get(key, "") for key in ("fmt", "name", "label", "value")]
    values += [text.strip() for text in target.itertext() if text.strip()]
    return " ".join(dict.fromkeys(value for value in values if value))


def _pid_value(element: ET.Element | None, ids: dict[str, ET.Element]) -> int | None:
    if element is None: return None
    target = ids.get(element.attrib.get("ref", ""), element)
    pid_node = next(iter(target.iter("pid")), None)
    if pid_node is None: return None
    digits = re.sub(r"[^0-9]", "", _element_value(pid_node, ids))
    return int(digits) if digits else None


def trace_table_context(xml_text: str, schema: str, *, target_pid: int) -> dict[str, Any]:
    try: root = ET.fromstring(xml_text)
    except ET.ParseError as error: raise EvidenceError(f"invalid xctrace table XML for {schema}: {error}") from error
    ids = {element.attrib["id"]: element for element in root.iter() if "id" in element.attrib}
    schema_node = next((item for item in root.iter("schema") if item.attrib.get("name") == schema), None)
    if schema_node is None: raise EvidenceError(f"xctrace table lacks the declared {schema} schema")
    mnemonics = [(column.find("mnemonic").text if column.find("mnemonic") is not None else "")
                 for column in schema_node.findall("col")]
    compiled, encoders = set(), set()
    for row in root.iter("row"):
        fields = {mnemonic: (child, _element_value(child, ids))
                  for mnemonic, child in zip(mnemonics, list(row))}
        if _pid_value(fields.get("process", (None, ""))[0], ids) != target_pid: continue
        if schema == "metal-shader-profiler-shader-list":
            name = fields.get("name", (None, ""))[1]
            if "qmm" in name.lower() or "qmv" in name.lower(): compiled.add(name)
        if schema == "metal-gpu-intervals":
            encoder = fields.get("encoder-id", (None, ""))[1]
            if encoder: encoders.add(encoder)
    return {"schema": schema, "target_pid": target_pid,
            "compiled_kernel_candidates": sorted(compiled),
            "executed_encoder_ids": sorted(encoders)}


def parse_metal_rows(xml_text: str, schema: str, *, case: str, target_pid: int) -> list[dict[str, Any]]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as error:
        raise EvidenceError(f"invalid xctrace table XML for {schema}: {error}") from error
    if schema != "metal-gpu-intervals":
        return []  # Creation/compilation tables are context, never execution proof.
    ids = {element.attrib["id"]: element for element in root.iter() if "id" in element.attrib}
    schema_node = next((item for item in root.iter("schema") if item.attrib.get("name") == schema), None)
    if schema_node is None:
        raise EvidenceError(f"xctrace table lacks the declared {schema} schema")
    mnemonics = []
    for column in schema_node.findall("col"):
        mnemonic = column.find("mnemonic")
        mnemonics.append(mnemonic.text if mnemonic is not None else "")
    records = []
    for row in root.iter("row"):
        children = list(row)
        fields = {mnemonic: (child, _element_value(child, ids))
                  for mnemonic, child in zip(mnemonics, children)}
        process_element, process = fields.get("process", (None, ""))
        row_pid = _pid_value(process_element, ids)
        _, encoder = fields.get("encoder-id", (None, ""))
        _, kernel = fields.get("event-label", (None, ""))
        if kernel and ("qmm" in kernel.lower() or "qmv" in kernel.lower()):
            records.append({"source": "xctrace-export-row", "record_type": "metal-compute-encoder",
                            "schema": schema, "process": process, "encoder": encoder,
                            "kernel_name": kernel, "case": case,
                            "row_pid": row_pid, "target_pid_match": row_pid == target_pid})
    return records


def capture_status(*, requested: bool, tool: str | None, exit_code: int | None,
                   trace_exists: bool, architecture: str) -> dict[str, Any]:
    if not requested:
        return {"status": "not-requested", "reason": "capture was not requested"}
    if tool is None:
        return {"status": "unavailable", "reason": "xctrace is unavailable"}
    if exit_code != 0:
        return {"status": "failed", "reason": f"owned capture child exited {exit_code}"}
    if not trace_exists:
        return {"status": "failed", "reason": "capture child exited zero without a trace package"}
    if architecture.lower() == "unknown":
        return {"status": "unsupported", "reason": "MLX reports an unknown Metal architecture"}
    return {"status": "captured", "reason": "trace package exists; dispatch still requires encoder rows"}


def validate_hash_binding(provenance: dict[str, Any], expected: dict[str, str]) -> None:
    if set(provenance) != set(expected):
        raise EvidenceError("missing or unexpected provenance hashes")
    if any(provenance[key] != value for key, value in expected.items()):
        raise EvidenceError("stale provenance hash")


def validate_case_numbers(case: dict[str, Any]) -> None:
    for key in ("maxAbsoluteError", "maxRelativeError", "scalarSpotError", "tolerance"):
        if type(case.get(key)) not in (int, float) or not math.isfinite(case[key]) or case[key] < 0:
            raise EvidenceError(f"invalid numerical evidence in {case.get('id', '<unknown>')}")
    if case["maxAbsoluteError"] > case["tolerance"] or case["scalarSpotError"] > case["tolerance"]:
        raise EvidenceError(f"recorded tolerance exceeded in {case.get('id', '<unknown>')}")


def expected_case_combinations(mode: str) -> set[tuple[str, int, int, bool]]:
    if mode == "trace-grouped6": return {("gate", 6, 16, True)}
    if mode == "trace-decode6": return {("gate", 6, 1, False)}
    return {(projection, bits, batch, sorted_indices)
            for projection in ("gate", "up", "down") for bits in (4, 6)
            for batch in (1, 4, 16, 64, 256) for sorted_indices in (False, True)}


def validate_m5_report(path: Path, binary: Path, *, repo_root: Path = ROOT) -> dict[str, Any]:
    report = read_json(path)
    required = {"format", "schemaVersion", "mode", "processID", "provenance", "diagnostic"}
    optional = {"boundedModelCheck", "modelPath", "modelScope"}
    if (not required.issubset(report) or not set(report).issubset(required | optional)
            or report["format"] != "slotstream-m5-command-v1" or report["schemaVersion"] != 1):
        raise EvidenceError("invalid M5 command report envelope")
    if type(report["processID"]) is not int or report["processID"] <= 0:
        raise EvidenceError("invalid M5 diagnostic process identity")
    provenance = report["provenance"]
    expected_provenance_fields = {"executablePath", "executableSHA256", "metallibSHA256",
                                  "buildIdentitySHA256", "sourceArchiveSHA256"}
    if set(provenance) != expected_provenance_fields:
        raise EvidenceError("invalid M5 provenance")
    binary = binary.resolve(strict=True)
    if Path(provenance["executablePath"]).resolve(strict=True) != binary or provenance["executableSHA256"] != sha256(binary):
        raise EvidenceError("M5 executable identity mismatch")
    _, paths = validate_build_identity(binary, root=repo_root)
    expected = {"metallibSHA256": sha256(paths["metallib"]),
                "buildIdentitySHA256": sha256(paths["identity"]),
                "sourceArchiveSHA256": sha256(paths["source_archive"])}
    validate_hash_binding({key: provenance[key] for key in expected}, expected)
    diagnostic = report["diagnostic"]
    diag_required = {"format", "schemaVersion", "mode", "deviceArchitecture", "operatingSystem",
                     "effectiveTF32", "mlxMetalNoNAX", "internalArchitectureGeneration",
                     "observationStatus", "observationReason", "allocationLimitBytes", "mlxPeakBytes",
                     "processFootprintEndBytes", "cases", "check"}
    if set(diagnostic) != diag_required or diagnostic["format"] != "slotstream-m5-diagnostic-v1":
        raise EvidenceError("invalid M5 diagnostic report")
    if diagnostic["mode"] != report["mode"] or diagnostic["observationStatus"] not in ("unverified", "observed"):
        raise EvidenceError("inconsistent M5 mode or observation status")
    if type(diagnostic["effectiveTF32"]) is not bool:
        raise EvidenceError("effective TF32 must be Boolean")
    if (type(diagnostic["mlxPeakBytes"]) is not int or diagnostic["mlxPeakBytes"] <= 0
            or diagnostic["mlxPeakBytes"] > diagnostic["allocationLimitBytes"]):
        raise EvidenceError("M5 allocation bound is missing or exceeded")
    cases = diagnostic["cases"]
    if not isinstance(cases, list) or not cases:
        raise EvidenceError("M5 report has no cases")
    ids = set()
    combinations = set()
    for case in cases:
        case_required = {"id", "projection", "bits", "sourceEncoding", "batch", "expertCount",
                         "sortedIndices", "contiguousInput", "inputShape", "inputStrides", "weightShape",
                         "outputShape", "inputDType", "metadataDType", "expectedDispatch",
                         "observedDispatch", "observationStatus", "observationReason", "maxAbsoluteError",
                         "maxRelativeError", "scalarSpotError", "scalarSpotCoverage", "tolerance", "passed"}
        optional_case = {"observedDispatch"}
        if (not isinstance(case, dict) or not (case_required - optional_case).issubset(case)
                or not set(case).issubset(case_required)):
            raise EvidenceError("malformed M5 case")
        if case["id"] in ids or case["projection"] not in ("gate", "up", "down") or case["bits"] not in (4, 6):
            raise EvidenceError("duplicate or unknown M5 case")
        ids.add(case["id"])
        combinations.add((case["projection"], case["bits"], case["batch"], case["sortedIndices"]))
        expected_source = "native-4-bit" if case["bits"] == 4 else "mixed-native-6-and-exact-widened-4"
        if (case["batch"] not in (1, 4, 16, 64, 256) or case["expertCount"] != 4
                or case["metadataDType"] != "float32" or case["sourceEncoding"] != expected_source
                or case["passed"] is not True):
            raise EvidenceError(f"M5 case failed its contract: {case['id']}")
        validate_case_numbers(case)
        output_width = 2560 if case["projection"] == "down" else 640
        input_width = 640 if case["projection"] == "down" else 2560
        if case["inputShape"] != [case["batch"], 1, input_width] or case["outputShape"] != [case["batch"], 1, output_width]:
            raise EvidenceError(f"shape contract failed in {case['id']}")
        if (case["weightShape"] != [4, output_width, input_width * case["bits"] // 32]
                or not isinstance(case["inputStrides"], list)
                or not all(type(value) is int and value > 0 for value in case["inputStrides"])
                or not isinstance(case["scalarSpotCoverage"], list)):
            raise EvidenceError(f"layout or scalar coverage failed in {case['id']}")
        if case["observationStatus"] == "observed" and not case["observedDispatch"]:
            raise EvidenceError("observed M5 dispatch lacks a pipeline name")
    expected_combinations = expected_case_combinations(report["mode"])
    if combinations != expected_combinations:
        raise EvidenceError("M5 report does not contain exact full case coverage")
    check = diagnostic["check"]
    if (check.get("name") != "m5-dispatch" or check.get("passed") is not True or not check.get("items")
            or any(item.get("passed") is not True for item in check["items"] if isinstance(item, dict))
            or any(not isinstance(item, dict) for item in check["items"])):
        raise EvidenceError("M5 diagnostic check failed")
    if not isinstance(check.get("measurements"), dict) or any(
            type(value) not in (int, float) or not math.isfinite(value)
            for value in check["measurements"].values()):
        raise EvidenceError("M5 check measurements are malformed")
    if report["mode"] == "bounded-model-rows":
        bounded = report.get("boundedModelCheck")
        if (not isinstance(bounded, dict) or bounded.get("passed") is not True
                or not bounded.get("items") or not report.get("modelPath")
                or "bounded original rows" not in report.get("modelScope", "")):
            raise EvidenceError("bounded model mode lacks original-row evidence")
    completion = read_json(path.parent / "completion.json")
    if (set(completion) != {"format", "schema_version", "report_sha256", "passed", "observation_status"}
            or completion["format"] != "slotstream-m5-completion-v1" or completion["passed"] is not True
            or completion["report_sha256"] != sha256(path)
            or completion["observation_status"] != diagnostic["observationStatus"]):
        raise EvidenceError("M5 completion does not bind its report")
    return report


def _xctrace() -> str | None:
    environment = dict(os.environ)
    environment["DEVELOPER_DIR"] = "/Applications/Xcode.app/Contents/Developer"
    try:
        result = subprocess.run(["/usr/bin/xcrun", "--find", "xctrace"], env=environment,
                                text=True, capture_output=True, timeout=5)
    except subprocess.TimeoutExpired:
        return None
    return result.stdout.strip() if result.returncode == 0 and result.stdout.strip() else None


def run_bounded_export(command: list[str], stdout_path: Path, stderr_path: Path,
                       *, timeout_seconds: float = 20, max_bytes: int = 16 << 20,
                       memory_gb: float = 2.0,
                       environment: dict[str, str] | None = None,
                       preflight_func=preflight, sampler_factory=DarwinSampler) -> dict[str, Any]:
    if timeout_seconds <= 0 or max_bytes <= 0 or memory_gb <= 0:
        raise EvidenceError("bounded child limits must be positive")
    admission = preflight_func(memory_gb + 3.0)
    process = None
    sampler_error = cleanup_error = None
    peak_bytes = None
    interrupted = memory_exceeded = False
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        timed_out = False
        output_exceeded = False
        selector = selectors.DefaultSelector()
        written = 0
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=environment,
                                       start_new_session=True)
            assert process.stdout is not None and process.stderr is not None
            selector.register(process.stdout, selectors.EVENT_READ, stdout)
            selector.register(process.stderr, selectors.EVENT_READ, stderr)
            sampler = sampler_factory(process.pid)
            deadline = time.monotonic() + timeout_seconds
            next_sample = 0.0
            while process.poll() is None or selector.get_map():
                now = time.monotonic()
                if process.poll() is None and now >= next_sample:
                    try:
                        sample = sampler.sample()
                        observed = max(sample["physical_footprint_bytes"], sample["lifetime_peak_bytes"])
                        peak_bytes = observed if peak_bytes is None else max(peak_bytes, observed)
                        if observed > memory_gb * 1e9:
                            memory_exceeded = True
                            break
                    except KeyboardInterrupt:
                        raise
                    except Exception as error:
                        sampler_error = f"{type(error).__name__}: {error}"
                        break
                    next_sample = now + 0.25
                for key, _ in selector.select(timeout=0.05):
                    data = os.read(key.fileobj.fileno(), 64 << 10)
                    if not data:
                        selector.unregister(key.fileobj)
                        continue
                    remaining = max_bytes - written
                    if remaining > 0:
                        key.data.write(data[:remaining])
                        written += min(len(data), remaining)
                    if len(data) > remaining:
                        output_exceeded = True
                        break
                if output_exceeded: break
                if process.poll() is None and now >= deadline:
                    timed_out = True
                    break
        except KeyboardInterrupt:
            interrupted = True
        except Exception as error:
            sampler_error = f"{type(error).__name__}: {error}"
        finally:
            if process is not None and process.poll() is None:
                try: terminate_child_tree(process)
                except Exception as error: cleanup_error = f"{type(error).__name__}: {error}"
            selector.close()
            if process is not None:
                if process.stdout is not None: process.stdout.close()
                if process.stderr is not None: process.stderr.close()
            code = process.poll() if process is not None else None
            stdout.flush(); stderr.flush()
            if stdout_path.stat().st_size + stderr_path.stat().st_size > max_bytes:
                output_exceeded = True
    sizes = {"stdout_bytes": stdout_path.stat().st_size, "stderr_bytes": stderr_path.stat().st_size}
    return {"pid": process.pid if process is not None else None,
            "exit_code": code, "timed_out": timed_out,
            "output_exceeded": output_exceeded, "interrupted": interrupted,
            "memory_gb_decimal": memory_gb, "memory_peak_bytes": peak_bytes,
            "memory_exceeded": memory_exceeded,
            "memory_qualified": peak_bytes is not None and not memory_exceeded and sampler_error is None,
            "sampler_error": sampler_error, "cleanup_error": cleanup_error,
            "preflight_reclaimable_bytes": admission.get("reclaimable_bytes"), **sizes}


def bounded_child_succeeded(result: dict[str, Any]) -> bool:
    return (result.get("exit_code") == 0 and result.get("timed_out") is False
            and result.get("output_exceeded") is False and result.get("interrupted") is False
            and result.get("memory_exceeded") is False and result.get("memory_qualified") is True
            and result.get("sampler_error") is None and result.get("cleanup_error") is None)


def trace_manifest(path: Path, *, max_total_bytes: int = 192 << 20) -> dict[str, Any]:
    files = [item for item in sorted(path.rglob("*")) if item.is_file()]
    total = sum(item.stat().st_size for item in files)
    if total > max_total_bytes:
        raise EvidenceError(f"trace package exceeds {max_total_bytes} byte bound: {total}")
    return {"total_bytes": total, "files": {str(item.relative_to(path)): sha256(item) for item in files}}


def run_probe(binary: Path, output: Path, *, synthetic: bool, capture: bool) -> int:
    if capture and not synthetic:
        raise EvidenceError("--capture requires --synthetic selected cases; model mode is bounded row validation only")
    binary = binary.resolve(strict=True)
    validate_build_identity(binary)
    output = fresh_output(output)
    tool = _xctrace() if capture else None
    records: list[Any] = []
    exported_tables = []
    table_contexts = []
    capture_results = []
    report_paths = []
    if capture and tool is None:
        capture_results.append({"case": "grouped6", "status": capture_status(
            requested=True, tool=None, exit_code=None, trace_exists=False, architecture="unknown")})
        capture_results.append({"case": "decode6", "status": capture_status(
            requested=True, tool=None, exit_code=None, trace_exists=False, architecture="unknown")})
    if capture and tool and synthetic:
        for case in ("grouped6", "decode6"):
            evidence = output / f"{case}-capture-evidence"
            diagnostic = evidence / "diagnostic"
            trace = output / f"{case}-capture.trace"
            target = [str(binary), "m5-check", "--synthetic", "--trace-case", case,
                      "--output", str(diagnostic)]
            command = [tool, "record", "--template", "Metal System Trace", "--output", str(trace),
                       "--time-limit", "5s", "--no-prompt", "--target-stdout", "-",
                       "--env", "MTL_CAPTURE_ENABLED=1",
                       "--launch", "--", *target]
            code = benchmark.launch(evidence, 6.0, 45.0, command)
            launcher = validate_receipt_file(evidence / "receipt.json")
            launch_ok = code == 0 and launcher["result"]["functional_success"]
            report_path = diagnostic / "report.json"
            report = validate_m5_report(report_path, binary) if launch_ok and report_path.exists() else None
            architecture = report["diagnostic"]["deviceArchitecture"] if report else "unknown"
            status = capture_status(requested=True, tool=tool, exit_code=code,
                                    trace_exists=trace.exists(), architecture=architecture)
            entry = {"case": case, "status": status, "launcher_receipt": str((evidence / 'receipt.json').relative_to(output)),
                     "launcher_receipt_sha256": sha256(evidence / "receipt.json"),
                     "launcher_result": launcher["result"]}
            if report:
                report_paths.append(report_path)
                entry["diagnostic_report"] = str(report_path.relative_to(output))
                entry["target_pid"] = report["processID"]
            if status["status"] == "captured" and report:
                try:
                    entry["trace_manifest"] = trace_manifest(trace)
                except EvidenceError as error:
                    entry["status"] = {"status": "failed", "reason": str(error)}
                    capture_results.append(entry)
                    continue
                environment = dict(os.environ)
                environment["DEVELOPER_DIR"] = "/Applications/Xcode.app/Contents/Developer"
                toc_path = output / f"{case}-capture-toc.xml"
                toc_error = output / f"{case}-capture-toc.stderr.txt"
                toc = run_bounded_export([tool, "export", "--input", str(trace), "--toc"],
                                         toc_path, toc_error, environment=environment)
                entry["toc_exit_code"] = toc["exit_code"]
                entry["toc_timed_out"] = toc["timed_out"]
                entry["toc_output_exceeded"] = toc["output_exceeded"]
                entry["toc_sha256"] = sha256(toc_path)
                if not bounded_child_succeeded(toc):
                    entry["status"] = {"status": "failed", "reason":
                        f"bounded xctrace TOC export exited {toc['exit_code']}; timed_out={toc['timed_out']}"}
                else:
                    for index, schema in enumerate(metal_schemas(toc_path.read_text())):
                        table_path = output / f"{case}-capture-table-{index}.xml"
                        table_error = output / f"{case}-capture-table-{index}.stderr.txt"
                        table = run_bounded_export([tool, "export", "--input", str(trace), "--xpath",
                            f"/trace-toc/run[@number='1']/data/table[@schema='{schema}']"],
                            table_path, table_error, max_bytes=32 << 20, environment=environment)
                        exported_tables.append({"case": case, "schema": schema, "path": table_path.name,
                                                "exit_code": table["exit_code"], "timed_out": table["timed_out"],
                                                "output_exceeded": table["output_exceeded"],
                                                "sha256": sha256(table_path)})
                        if bounded_child_succeeded(table) and table_path.stat().st_size:
                            table_text = table_path.read_text()
                            records.extend(parse_metal_rows(table_text, schema, case=case,
                                                           target_pid=report["processID"]))
                            context = trace_table_context(table_text, schema, target_pid=report["processID"])
                            context["case"] = case
                            table_contexts.append(context)
            capture_results.append(entry)

    functional = bool(report_paths) and all(entry["status"]["status"] == "captured" for entry in capture_results)
    if not functional:
        fallback = output / "fallback-evidence"
        fallback_diagnostic = fallback / "diagnostic"
        fallback_target = [str(binary), "m5-check", "--synthetic" if synthetic else "--model",
                           "models/jang-6s" if not synthetic else "", "--output", str(fallback_diagnostic)]
        fallback_code = benchmark.launch(fallback, 2.0, 240.0, [item for item in fallback_target if item])
        fallback_receipt = validate_receipt_file(fallback / "receipt.json")
        functional = fallback_code == 0 and fallback_receipt["result"]["functional_success"]
        if functional:
            report_path = fallback_diagnostic / "report.json"
            validate_m5_report(report_path, binary)
            report_paths.append(report_path)
        else:
            atomic_json(output / "result.json", {"format": "slotstream-m5-probe-v1",
                        "schema_version": 1, "functional_success": False,
                        "captures": capture_results, "observation": assess_actual_dispatch(records)})
            return 1
    observation = assess_actual_dispatch(records)
    result = {"format": "slotstream-m5-probe-v1", "schema_version": 1,
              "functional_success": True, "captures": capture_results, "observation": observation,
              "diagnostic_reports": [{"path": str(path.relative_to(output)), "sha256": sha256(path)}
                                     for path in report_paths],
              "binary_sha256": sha256(binary),
              "production_equivalent_build": True,
              "exported_tables": exported_tables,
              "trace_context": table_contexts,
              "limitation": None if observation["status"] == "observed" else
                  "xctrace exposed target-bound compiled kernel names and executed encoder IDs in separate tables, but no relation maps a specific kernel to a completed encoder"}
    atomic_json(output / "result.json", result)
    atomic_json(output / "completion.json", {"format": "slotstream-m5-probe-completion-v1",
                "result_sha256": sha256(output / "result.json"), "functional_success": True,
                "observation_status": observation["status"]})
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--synthetic", action="store_true")
    mode.add_argument("--model", action="store_true")
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    return run_probe(args.binary, args.output, synthetic=args.synthetic, capture=args.capture)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (EvidenceError, OSError) as error:
        raise SystemExit(str(error))
