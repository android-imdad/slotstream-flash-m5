#!/usr/bin/env python3
"""Collect and analyze a bounded recent-token expert-cache development screen."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import sys
import time
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import benchmark
from common import (EvidenceError, atomic_json, fresh_output, harness_hashes, read_json,
                    sha256, validate_build_identity)
from receipts import validate_receipt_file
from replay import (AGE_METADATA_BYTES_PER_SLOT, CACHE_RECORD_BYTES, LAYERS, parse_trace,
                    replay, validate_runtime_stats)

FIXTURE = ROOT / "Tools/fixtures/flash/cache-study.json"
PINNED_REVISION = "3781190c6bbdf0a7637beda49ba179822612058a"
MAX_HEADER_BYTES = 64 << 20
BASELINE_SUMMARY = ROOT / ".build/flash/runs/baseline-summary-final.json"
ARCHIVE = ROOT / ".build/flash/runs/reviewer-m5-archive"
JANG6S_SMALL_PINS = {
    "LICENSE": (3235, "a0dc422560841fd68e06d974907f8b4c709bca44a67daad2b528437bdf676c08"),
    "chat_template.jinja": (8952, "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"),
    "config.json": (113950, "06d4ba9c9604e901ed5d8a7c2e8d236d32adbbedd1c3b5d1a621a525bffc594a"),
    "generation_config.json": (177, "621cacfcde04fdcc44c946eaad16d19091cfe4b50ae6a4e788f0710e28bf759b"),
    "merges.txt": (3353259, "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d"),
    "model.safetensors.index.json": (312785, "b9ee9e7becdfa76f1be4eb13127a07e615b419b9bce9a3a459224587c6d1492f"),
    "preprocessor_config.json": (390, "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516"),
    "tokenizer.json": (12809320, "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"),
    "tokenizer_config.json": (17928, "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27"),
    "video_preprocessor_config.json": (385, "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13"),
    "vocab.json": (6722759, "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003"),
}


def jang6s_manifest_pins() -> dict[str, tuple[int, str]]:
    source = (ROOT / "Sources/Slotstream/JANGManifests.swift").read_text()
    match = re.search(r"public static let jang6S = .*?revision: \"([0-9a-f]+)\", files: \[(.*?)\n    \]\)",
                      source, re.S)
    if not match or match.group(1) != PINNED_REVISION:
        raise EvidenceError("cannot parse pinned JANG_6S manifest")
    files = {name: (int(size), digest) for name, size, digest in re.findall(
        r'\.init\(path: "([^"]+)", size: (\d+), sha256: "([0-9a-f]{64})"\)', match.group(2))}
    if len(files) != 37 or any(files.get(name) != value for name, value in JANG6S_SMALL_PINS.items()):
        raise EvidenceError("pinned JANG_6S manifest inventory changed")
    return files


def canonical_hash(value: dict[str, Any], omitted: str = "content_sha256") -> str:
    copy = dict(value); copy.pop(omitted, None)
    return hashlib.sha256(json.dumps(copy, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def load_fixture(path: Path = FIXTURE) -> dict[str, Any]:
    value = read_json(path)
    required = {"format", "schema_version", "content_sha256", "development_only", "options", "prompts"}
    if set(value) != required or value["format"] != "slotstream-cache-study-fixture-v1" or value["schema_version"] != 1:
        raise EvidenceError("invalid cache-study fixture")
    if value["content_sha256"] != canonical_hash(value): raise EvidenceError("cache-study fixture hash mismatch")
    options = value["options"]
    expected = {"greedy": True, "max_context": 2048, "max_tokens": 128,
                "memory_gb_decimal": 14, "mtp": "off", "seed": 7, "vision": "off"}
    if options != expected: raise EvidenceError("cache-study options changed")
    prompts = value["prompts"]
    if not isinstance(prompts, list) or len(prompts) != 3 or len({p.get("id") for p in prompts}) != 3:
        raise EvidenceError("cache-study requires three unique prompts")
    for prompt in prompts:
        if set(prompt) != {"id", "text", "max_ascii_bytes", "max_output_utf8_bytes"}:
            raise EvidenceError("malformed cache-study prompt")
        try: encoded = prompt["text"].encode("ascii")
        except UnicodeEncodeError as error: raise EvidenceError("development prompts must be exact ASCII") from error
        if not encoded or len(encoded) > prompt["max_ascii_bytes"]:
            raise EvidenceError("development prompt exceeds its frozen ASCII bound")
    return value


def _quantization(config: dict[str, Any], name: str) -> dict[str, int]:
    bit_map = config.get("jang_config", {}).get("bit_map")
    if not isinstance(bit_map, dict): raise EvidenceError("JANG quantization map is missing")
    matches = [(len(prefix), value) for prefix, value in bit_map.items()
               if prefix == "default" or name == prefix or name.startswith(prefix + ".")]
    if not matches: raise EvidenceError(f"quantization is missing for {name}")
    value = max(matches, key=lambda item: item[0])[1]
    if not isinstance(value, dict) or set(value) != {"bits", "group_size"}:
        raise EvidenceError(f"malformed quantization for {name}")
    return value


def _header(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    with path.open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8: raise EvidenceError(f"truncated safetensors header length: {path.name}")
        length = struct.unpack("<Q", raw)[0]
        if length <= 0 or length > MAX_HEADER_BYTES or 8 + length > path.stat().st_size:
            raise EvidenceError(f"unsafe safetensors header length: {path.name}")
        payload = stream.read(length)
    try: header = json.loads(payload)
    except json.JSONDecodeError as error: raise EvidenceError(f"malformed safetensors header: {path.name}") from error
    if not isinstance(header, dict): raise EvidenceError(f"invalid safetensors header: {path.name}")
    identity = {"path": path.name, "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns, "header_bytes": length,
                "header_sha256": hashlib.sha256(payload).hexdigest()}
    return header, identity


def verified_model_revision(model: Path | None = None) -> dict[str, Any]:
    summary = read_json(BASELINE_SUMMARY)
    verification = summary.get("model_verification")
    if not isinstance(verification, dict) or verification.get("revision") != PINNED_REVISION or verification.get("verified") is not True:
        raise EvidenceError("baseline does not bind the required verified JANG revision")
    output = ROOT / ".build/flash/runs" / verification["stdout"]
    expected = f"VERIFY PASS: qwen3.8-flash-next:jang-6s matches {PINNED_REVISION}"
    if output.read_text().strip() != expected or sha256(output) != verification["stdout_sha256"]:
        raise EvidenceError("model verification output changed")
    if model is not None:
        command = verification.get("command")
        if not isinstance(command, list) or command.count("--dir") != 1:
            raise EvidenceError("historical verification command does not bind a model directory")
        recorded = Path(command[command.index("--dir") + 1])
        recorded = (ROOT / recorded).resolve() if not recorded.is_absolute() else recorded.resolve()
        if recorded != model.resolve(): raise EvidenceError("model path differs from historical verification")
    return {"revision": PINNED_REVISION, "summary_path": str(BASELINE_SUMMARY),
            "summary_sha256": sha256(BASELINE_SUMMARY), "verification_output": str(output),
            "verification_output_sha256": sha256(output),
            "payload_verification_scope": "historical full-payload verification; current run rechecks bounded headers and small metadata only"}


def derive_source_geometry(model: Path, verification: dict[str, Any] | None = None) -> dict[str, Any]:
    model = model.resolve(strict=True)
    if verification is not None:
        if verification.get("revision") != PINNED_REVISION:
            raise EvidenceError("source geometry verification is not pinned JANG_6S")
        pins = jang6s_manifest_pins()
        for name, (size, digest) in JANG6S_SMALL_PINS.items():
            path = model / name
            if not path.is_file() or path.stat().st_size != size or sha256(path) != digest:
                raise EvidenceError(f"pinned JANG_6S metadata changed: {name}")
    config_path, index_path = model / "config.json", model / "model.safetensors.index.json"
    config, index = read_json(config_path), read_json(index_path)
    text = config.get("text_config", {})
    if (text.get("hidden_size"), text.get("moe_intermediate_size"), text.get("num_hidden_layers"),
            text.get("num_experts"), text.get("num_experts_per_tok")) != (2560, 640, 48, 512, 10):
        raise EvidenceError("unsupported model geometry")
    if config.get("jang_config", {}).get("format") != "jang_v2": raise EvidenceError("unsupported JANG format")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict): raise EvidenceError("safetensors index weight_map is missing")
    names = []
    for layer in range(LAYERS):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            for member in ("weight", "scales", "biases"):
                names.append(f"language_model.layers.{layer}.mlp.switch_mlp.{projection}.{member}")
    mapped_files = [weight_map.get(name) for name in names]
    if any(filename is None for filename in mapped_files):
        raise EvidenceError("expert tensor is absent from the safetensors index")
    files = sorted(set(mapped_files))
    manifest_pins = jang6s_manifest_pins() if verification is not None else None
    headers, identities = {}, {}
    for filename in files:
        path = model / filename
        if manifest_pins is not None and (filename not in manifest_pins or path.stat().st_size != manifest_pins[filename][0]):
            raise EvidenceError(f"safetensors file size differs from pinned manifest: {filename}")
        header, identity = _header(path)
        if manifest_pins is not None:
            identity["historical_payload_sha256"] = manifest_pins[filename][1]
            identity["size_matches_pin"] = True
        headers[filename], identities[filename] = header, identity
    layer_bytes = []
    tensor_rows = {}
    for layer in range(LAYERS):
        total = 0
        for projection in ("gate_proj", "up_proj", "down_proj"):
            base = f"language_model.layers.{layer}.mlp.switch_mlp.{projection}"
            quant = _quantization(config, base)
            bits, group = quant["bits"], quant["group_size"]
            if bits not in (4, 6) or group != 64: raise EvidenceError(f"unsupported expert quantization: {base}")
            rows, columns = (640, 2560) if projection != "down_proj" else (2560, 640)
            for member in ("weight", "scales", "biases"):
                name = f"{base}.{member}"; filename = weight_map[name]
                entry = headers[filename].get(name)
                expected_shape = [512, rows, columns * bits // 32] if member == "weight" else [512, rows, columns // group]
                expected_dtype = "U32" if member == "weight" else "F16"
                if (not isinstance(entry, dict) or entry.get("dtype") != expected_dtype
                        or entry.get("shape") != expected_shape):
                    raise EvidenceError(f"unsupported tensor geometry or dtype: {name}")
                offsets = entry.get("data_offsets")
                if (not isinstance(offsets, list) or len(offsets) != 2
                        or not all(type(value) is int for value in offsets)
                        or offsets[0] < 0 or offsets[1] <= offsets[0]):
                    raise EvidenceError(f"invalid tensor offsets: {name}")
                span = offsets[1] - offsets[0]
                payload_bytes = (model / filename).stat().st_size - 8 - identities[filename]["header_bytes"]
                if offsets[1] > payload_bytes:
                    raise EvidenceError(f"tensor offsets exceed file payload: {name}")
                dtype_bytes = 4 if member == "weight" else 2
                if span != math.prod(expected_shape) * dtype_bytes or span % 512:
                    raise EvidenceError(f"tensor byte size disagrees with header: {name}")
                row_bytes = span // 512
                tensor_rows[name] = {"file": filename, "row_bytes": row_bytes,
                                     "bits": bits, "dtype": expected_dtype, "shape": expected_shape}
                total += row_bytes
        layer_bytes.append(total)
    if any(value <= 0 or value > CACHE_RECORD_BYTES for value in layer_bytes):
        raise EvidenceError("source expert bytes are outside the cache-record bound")
    small_files = {name: {"size": (model / name).stat().st_size, "sha256": sha256(model / name)}
                   for name in JANG6S_SMALL_PINS if (model / name).is_file()}
    return {"format": "slotstream-expert-source-geometry-v1", "model_path": str(model),
            "revision": verification["revision"] if verification else None,
            "verification": verification, "config_sha256": sha256(config_path),
            "index_sha256": sha256(index_path), "file_headers": identities,
            "small_file_identities": small_files,
            "layer_source_record_bytes": layer_bytes, "tensor_rows": tensor_rows,
            "cache_record_bytes": CACHE_RECORD_BYTES}


def validate_stats(document: dict[str, Any], prompt: dict[str, Any]) -> None:
    if document.get("schema_version") != 1 or not isinstance(document.get("stats"), dict):
        raise EvidenceError("stats JSON is partial or malformed")
    stats = document["stats"]
    if stats.get("runtimeError") is not None or stats.get("finishReason") not in ("stop", "length"):
        raise EvidenceError("generation did not finish successfully")
    if not document.get("output_ids") or not document.get("prompt_ids") or not document.get("text"):
        raise EvidenceError("generation output or token identities are empty")
    if len(document["prompt_ids"]) >= 64: raise EvidenceError("rendered prompt enters an unsupported sweep path")
    if len(document["text"].encode("utf-8")) > prompt["max_output_utf8_bytes"]:
        raise EvidenceError("generation text exceeds the frozen development bound")
    sampling = document.get("sampling", {})
    if sampling != {"greedy": True, "requested_max_tokens": "128", "seed": "7"}:
        raise EvidenceError("generation sampling options differ from the fixture")
    plan = document.get("plan", {})
    if (document.get("effective_mtp") is not False or document.get("experimental_memory_family") is not False
            or plan.get("checkpoint_format") != "jang6S" or plan.get("target_gb") != 14
            or plan.get("max_context_tokens") != 2048 or plan.get("mtp") is not False
            or plan.get("vision") is not False or type(document.get("effective_pool_slots")) is not int
            or document["effective_pool_slots"] <= 0):
        raise EvidenceError("effective generation plan differs from the frozen study")
    for key in ("prefillRecords", "decodeRecords", "prefillReadBytes", "decodeReadBytes",
                "decodeForwardPasses", "decodeTokens"):
        if type(stats.get(key)) is not int or stats[key] < 0: raise EvidenceError(f"invalid stats field: {key}")
    if (stats["decodeTokens"] != len(document["output_ids"])
            or stats.get("promptTokens") != len(document["prompt_ids"])
            or stats["decodeTokens"] > 128):
        raise EvidenceError("completed stats token counts disagree")
    for key in ("load_seconds", "launch_seconds"):
        if type(document.get(key)) not in (int, float) or not math.isfinite(document[key]) or document[key] < 0:
            raise EvidenceError(f"invalid completed timing: {key}")
    sampled = stats.get("sampledFootprint")
    if (not isinstance(sampled, dict) or type(sampled.get("peakBytes")) is not int
            or sampled["peakBytes"] <= 0 or type(sampled.get("samples")) is not int or sampled["samples"] <= 0):
        raise EvidenceError("completed stats lack sampled footprint")


def validate_pair_documents(untraced: dict[str, Any], traced: dict[str, Any]) -> None:
    if (untraced.get("prompt_ids") != traced.get("prompt_ids")
            or untraced.get("output_ids") != traced.get("output_ids")):
        raise EvidenceError("traced/untraced token identities changed")
    for key in ("effective_pool_slots", "effective_mtp", "effective_prefill_chunk", "optimizations"):
        if untraced.get(key) != traced.get(key): raise EvidenceError(f"traced arm changed effective configuration: {key}")
    plan_fields = ("checkpoint_format", "target_gb", "max_context_tokens", "mtp", "vision",
                   "pool_slots", "pool_gb", "experts_per_layer_cached", "prefill_chunk",
                   "expected_peak_gb", "memory_ledger", "source", "prefix_cache_max_tokens",
                   "runtime_prefix_cache_enabled", "context_qualification")
    untraced_plan, traced_plan = untraced.get("plan", {}), traced.get("plan", {})
    if ({key: untraced_plan.get(key) for key in plan_fields}
            != {key: traced_plan.get(key) for key in plan_fields}):
        raise EvidenceError("traced arm changed resolved allocation or model configuration")
    for key in ("prefillRecords", "decodeRecords", "prefillReadBytes", "decodeReadBytes",
                "decodeForwardPasses", "finishReason"):
        if untraced.get("stats", {}).get(key) != traced.get("stats", {}).get(key):
            raise EvidenceError(f"traced arm changed native cache observation: {key}")


def _command_value(command: list[str], flag: str) -> str:
    if command.count(flag) != 1: raise EvidenceError(f"generation command must contain {flag} exactly once")
    try: index = command.index(flag)
    except ValueError as error: raise EvidenceError(f"generation command omits {flag}") from error
    if index + 1 >= len(command): raise EvidenceError(f"generation command truncates {flag}")
    return command[index + 1]


def validate_cohort_shape(pairs: Any, fixture: dict[str, Any]) -> None:
    expected = {prompt["id"] for prompt in fixture["prompts"]}
    if (not isinstance(pairs, list) or len(pairs) != 3
            or {pair.get("prompt_id") for pair in pairs if isinstance(pair, dict)} != expected):
        raise EvidenceError("collection does not contain each frozen prompt exactly once")
    for pair in pairs:
        if (set(pair) != {"prompt_id", "arms", "prompt_tokens", "output_tokens"}
                or not isinstance(pair["arms"], dict) or set(pair["arms"]) != {"untraced", "traced"}):
            raise EvidenceError("collection pair or arms are malformed")
        for arm in ("untraced", "traced"):
            info = pair["arms"][arm]
            expected_artifacts = {"receipt.json", "stats.json", "environment.json"}
            if arm == "traced": expected_artifacts.add("router-trace.bin")
            if (not isinstance(info, dict) or set(info) != {"path", "artifacts"}
                    or not isinstance(info["artifacts"], dict)
                    or set(info["artifacts"]) != expected_artifacts):
                raise EvidenceError(f"collection arm artifacts are incomplete: {pair['prompt_id']} {arm}")


def validate_artifact_hashes(root: Path, artifacts: dict[str, str]) -> None:
    for name, digest in artifacts.items():
        path = root / name
        try: actual = sha256(path)
        except OSError as error: raise EvidenceError(f"collection artifact missing: {name}") from error
        if actual != digest: raise EvidenceError(f"collection artifact changed: {name}")


def validate_collection(
    collection: Path, manifest: dict[str, Any], *, require_completion: bool = True,
    verification_provider=verified_model_revision, geometry_provider=derive_source_geometry
) -> dict[str, Any]:
    collection = collection.resolve(strict=True)
    required = {"format", "schema_version", "complete", "qualification", "fixture_sha256",
                "binary", "model", "pairs", "harness_hashes"}
    if (set(manifest) != required or manifest.get("format") != "slotstream-cache-collection-v1"
            or manifest.get("schema_version") != 1 or manifest.get("complete") is not True
            or manifest.get("qualification") is not False):
        raise EvidenceError("collection manifest is incomplete or malformed")
    fixture = load_fixture()
    if manifest["fixture_sha256"] != sha256(FIXTURE) or manifest["harness_hashes"] != harness_hashes():
        raise EvidenceError("collection fixture or harness changed")
    binary = Path(manifest["binary"].get("path", ""))
    archive_receipt_path = Path(manifest["binary"].get("archive_receipt", ""))
    archive_receipt = validate_receipt_file(archive_receipt_path, require_qualified=True)
    identity, _ = validate_build_identity(binary, historical=True)
    if (manifest["binary"].get("sha256") != sha256(binary)
            or manifest["binary"].get("build_identity") != identity
            or manifest["binary"].get("archive_receipt_sha256") != sha256(archive_receipt_path)
            or archive_receipt["identities"]["build_identity"] != identity):
        raise EvidenceError("collection archived executable identity mismatch")
    binary_artifact = str(binary.resolve().relative_to(archive_receipt_path.parent.resolve()))
    if (binary_artifact not in archive_receipt["artifacts"]
            or archive_receipt["artifacts"][binary_artifact]["sha256"] != sha256(binary)):
        raise EvidenceError("archive receipt does not bind the collection executable")
    model_path = Path(manifest["model"]["model_path"])
    verification = verification_provider(model_path)
    current_geometry = geometry_provider(model_path, verification=verification)
    if current_geometry != manifest["model"]:
        raise EvidenceError("collection model headers or bounded identities changed")
    expected_prompts = {prompt["id"]: prompt for prompt in fixture["prompts"]}
    pairs = manifest["pairs"]
    validate_cohort_shape(pairs, fixture)
    validated = {}
    for pair in pairs:
        prompt = expected_prompts[pair["prompt_id"]]
        documents, environments = {}, {}
        for arm in ("untraced", "traced"):
            info = pair["arms"][arm]
            try: root = (collection / info["path"]).resolve(strict=True)
            except OSError as error: raise EvidenceError(f"collection arm path is missing: {info['path']}") from error
            if not root.is_relative_to(collection.resolve()): raise EvidenceError("collection arm escapes its root")
            validate_artifact_hashes(root, info["artifacts"])
            receipt = validate_receipt_file(root / "receipt.json")
            if (not receipt["result"]["functional_success"]
                    or receipt["identities"].get("executable", {}).get("sha256") != sha256(binary)
                    or "stats.json" not in receipt["artifacts"]
                    or (arm == "traced") != ("router-trace.bin" in receipt["artifacts"])):
                raise EvidenceError(f"collection launcher receipt is incomplete: {pair['prompt_id']} {arm}")
            if (receipt["memory"].get("target_gb_decimal") != 14
                    or receipt["memory"].get("qualified") is not True):
                raise EvidenceError("collection launcher memory target is not qualified at 14 GB")
            command = receipt["command"]
            model_argument = Path(_command_value(command, "--model"))
            model_argument = (ROOT / model_argument).resolve() if not model_argument.is_absolute() else model_argument.resolve()
            if (Path(command[0]).resolve() != binary.resolve()
                    or _command_value(command, "--prompt") != prompt["text"]
                    or model_argument != Path(manifest["model"]["model_path"])
                    or [_command_value(command, flag) for flag in ("--memory-gb", "--max-context", "--mtp", "--vision", "--max-tokens", "--seed")]
                        != ["14", "2048", "off", "off", "128", "7"]
                    or command.count("--greedy") != 1 or command.count("--sample-footprint") != 1
                    or Path(_command_value(command, "--stats-json")).resolve() != (root / "stats.json").resolve()):
                raise EvidenceError(f"collection command differs from frozen options: {pair['prompt_id']} {arm}")
            documents[arm] = read_json(root / "stats.json"); validate_stats(documents[arm], prompt)
            environments[arm] = read_json(root / "environment.json")
            if environments[arm] != receipt["environment"]:
                raise EvidenceError("collection environment does not match launcher receipt")
        trace_value = environments["traced"].pop("SLOTSTREAM_ROUTER_TRACE", None)
        if trace_value != str((collection / pair["arms"]["traced"]["path"] / "router-trace.bin")):
            raise EvidenceError("traced environment does not bind its trace path")
        if "SLOTSTREAM_ROUTER_TRACE" in environments["untraced"] or environments["traced"] != environments["untraced"]:
            raise EvidenceError("trace environment is not the only arm difference")
        validate_pair_documents(documents["untraced"], documents["traced"])
        if (pair["prompt_tokens"] != len(documents["traced"]["prompt_ids"])
                or pair["output_tokens"] != len(documents["traced"]["output_ids"])):
            raise EvidenceError("collection pair token counts are stale")
        trace_path = collection / pair["arms"]["traced"]["path"] / "router-trace.bin"
        groups = parse_trace(trace_path)
        native = replay(groups, documents["traced"]["effective_pool_slots"],
                        current_geometry["layer_source_record_bytes"])
        validate_runtime_stats(documents["traced"], groups, native)
        validate_runtime_stats(documents["untraced"], groups, native)
        validated[pair["prompt_id"]] = {"pair": pair, "documents": documents,
                                        "groups": groups, "native": native}
    if require_completion:
        completion = read_json(collection / "completion.json")
        if (set(completion) != {"format", "collection_sha256", "complete", "qualification"}
                or completion["format"] != "slotstream-cache-collection-completion-v1"
                or completion["collection_sha256"] != sha256(collection / "collection.json")
                or completion["complete"] is not True or completion["qualification"] is not False):
            raise EvidenceError("collection completion does not bind its manifest")
    return {"fixture": fixture, "geometry": current_geometry, "binary": binary,
            "pairs": validated}


def _set_trace(path: str | None):
    prior = os.environ.get("SLOTSTREAM_ROUTER_TRACE")
    if path is None: os.environ.pop("SLOTSTREAM_ROUTER_TRACE", None)
    else: os.environ["SLOTSTREAM_ROUTER_TRACE"] = path
    return prior


def collect(model: Path, output: Path) -> int:
    fixture = load_fixture()
    output = fresh_output(output)
    binary = ROOT / ".build/flash/runs/reviewer-m5-archive/bin/slotstream"
    try:
        identity, _ = validate_build_identity(binary, historical=True)
        archive_receipt_path = ARCHIVE / "receipt.json"
        archive_receipt = validate_receipt_file(archive_receipt_path, require_qualified=True)
        verification = verified_model_revision(model)
        source = derive_source_geometry(model, verification=verification)
        ambient = sorted(key for key in os.environ if key.startswith("SLOTSTREAM_") and key != "SLOTSTREAM_ROUTER_TRACE")
        if ambient: raise EvidenceError(f"unsupported ambient Slotstream controls: {ambient}")
        pairs = []
        for prompt in fixture["prompts"]:
            arms = {}
            for traced in (False, True):
                arm = "traced" if traced else "untraced"
                evidence = output / "prompts" / prompt["id"] / arm
                stats_path = evidence / "stats.json"
                trace_path = evidence / "router-trace.bin"
                command = [str(binary), "run", "--model", str(model), "--memory-gb", "14",
                           "--max-context", "2048", "--mtp", "off", "--vision", "off",
                           "--prompt", prompt["text"], "--max-tokens", "128", "--greedy",
                           "--seed", "7", "--sample-footprint", "--stats-json", str(stats_path)]
                prior = _set_trace(str(trace_path) if traced else None)
                try: code = benchmark.launch(evidence, 14.0, 900.0, command)
                finally: _set_trace(prior)
                receipt = validate_receipt_file(evidence / "receipt.json")
                if code != 0 or not receipt["result"]["functional_success"]:
                    raise EvidenceError(f"{prompt['id']} {arm} monitored generation failed")
                stats = read_json(stats_path); validate_stats(stats, prompt)
                environment = receipt["environment"]
                if traced != (environment.get("SLOTSTREAM_ROUTER_TRACE") == str(trace_path)):
                    raise EvidenceError("traced and untraced environments are not explicit")
                if traced:
                    groups = parse_trace(trace_path)
                    native = replay(groups, stats["effective_pool_slots"], source["layer_source_record_bytes"])
                    validate_runtime_stats(stats, groups, native)
                atomic_json(evidence / "environment.json", environment)
                names = ["receipt.json", "stats.json", "environment.json"] + (["router-trace.bin"] if traced else [])
                arms[arm] = {"path": str(evidence.relative_to(output)),
                             "artifacts": {name: sha256(evidence / name) for name in names}}
            untraced = read_json(output / arms["untraced"]["path"] / "stats.json")
            traced = read_json(output / arms["traced"]["path"] / "stats.json")
            validate_pair_documents(untraced, traced)
            untraced_env = read_json(output / arms["untraced"]["path"] / "environment.json")
            traced_env = read_json(output / arms["traced"]["path"] / "environment.json")
            traced_env.pop("SLOTSTREAM_ROUTER_TRACE", None)
            if traced_env != untraced_env:
                raise EvidenceError(f"{prompt['id']} traced arm changed more than the router-trace environment")
            pairs.append({"prompt_id": prompt["id"], "arms": arms,
                          "prompt_tokens": len(traced["prompt_ids"]), "output_tokens": len(traced["output_ids"])})
        manifest = {"format": "slotstream-cache-collection-v1", "schema_version": 1,
                    "complete": True, "qualification": False, "fixture_sha256": sha256(FIXTURE),
                    "binary": {"path": str(binary), "sha256": identity["binary_sha256"],
                               "build_identity": identity, "archive_receipt": str(archive_receipt_path),
                               "archive_receipt_sha256": sha256(archive_receipt_path)},
                    "model": source, "pairs": pairs, "harness_hashes": harness_hashes()}
        atomic_json(output / "collection.json", manifest)
        validate_collection(output, manifest, require_completion=False)
        atomic_json(output / "completion.json", {"format": "slotstream-cache-collection-completion-v1",
                    "collection_sha256": sha256(output / "collection.json"), "complete": True,
                    "qualification": False})
        return 0
    except Exception as error:
        atomic_json(output / "failure.json", {"format": "slotstream-cache-collection-failure-v1",
                    "error": f"{type(error).__name__}: {error}"})
        return 1


def decision(windows: dict[int, dict[str, Any]]) -> dict[str, Any]:
    eligible = []
    for window, result in windows.items():
        if result["pooled_decode_savings_fraction"] >= 0.10 and result["max_prompt_decode_increase_fraction"] <= 0.05:
            eligible.append((result["pooled_decode_savings_fraction"], window))
    if not eligible:
        return {"disposition": "rejected", "selected_window": None,
                "reason": "no recent window saved at least 10% pooled decode bytes without a prompt exceeding 5% increase"}
    _, selected = max(eligible)
    return {"disposition": "candidate", "selected_window": selected,
            "reason": "development threshold passed; a separate runtime implementation plan is still required"}


def byte_comparison(control: int, candidate: int) -> tuple[float, float]:
    if control < 0 or candidate < 0: raise EvidenceError("byte totals must be nonnegative")
    if control == 0:
        return (0.0, 0.0) if candidate == 0 else (-1.0, 1.0)
    return 1 - candidate / control, (candidate - control) / control


def analyze(collection: Path, output: Path, *, verification_provider=verified_model_revision,
            geometry_provider=derive_source_geometry) -> int:
    output = fresh_output(output)
    try:
        manifest_path = collection / "collection.json"; manifest = read_json(manifest_path)
        validated = validate_collection(collection, manifest,
            verification_provider=verification_provider, geometry_provider=geometry_provider)
        source_bytes = validated["geometry"]["layer_source_record_bytes"]
        per_prompt, aggregates = [], {window: {"control": 0, "candidate": 0, "increases": []}
                                      for window in (1, 2, 4, 8)}
        for prompt_id, entry in validated["pairs"].items():
            arms, groups, native = entry["documents"], entry["groups"], entry["native"]
            slots = arms["traced"]["effective_pool_slots"]
            total_budget = slots * CACHE_RECORD_BYTES
            candidate_slots = total_budget // (CACHE_RECORD_BYTES + AGE_METADATA_BYTES_PER_SLOT)
            matched = replay(groups, candidate_slots, source_bytes)
            windows = {}
            for window in (1, 2, 4, 8):
                candidate = replay(groups, candidate_slots, source_bytes, window=window)
                control_bytes, candidate_bytes = native["decode"]["miss_bytes"], candidate["decode"]["miss_bytes"]
                savings, increase = byte_comparison(control_bytes, candidate_bytes)
                windows[str(window)] = {"matched_clock": matched, "candidate": candidate,
                                        "native_control": native, "decode_savings_fraction": savings,
                                        "decode_increase_fraction": increase}
                aggregates[window]["control"] += control_bytes
                aggregates[window]["candidate"] += candidate_bytes
                aggregates[window]["increases"].append(increase)
            per_prompt.append({"prompt_id": prompt_id, "native_reconciliation": native,
                               "cache_slots": slots, "candidate_slots": candidate_slots,
                               "total_cache_budget_bytes": total_budget,
                               "age_metadata_bytes": candidate_slots * AGE_METADATA_BYTES_PER_SLOT,
                               "windows": windows})
        window_summary = {}
        for window, values in aggregates.items():
            pooled_savings, _ = byte_comparison(values["control"], values["candidate"])
            window_summary[window] = {"pooled_decode_savings_fraction": pooled_savings,
                "max_prompt_decode_increase_fraction": max(values["increases"]),
                "pooled_control_decode_bytes": values["control"],
                "pooled_candidate_decode_bytes": values["candidate"]}
        report = {"format": "slotstream-cache-window-screen-v1", "schema_version": 1,
                  "qualification": False, "runtime_policy_changed": False,
                  "collection": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
                  "fixture_sha256": sha256(FIXTURE), "per_prompt": per_prompt,
                  "windows": {str(key): value for key, value in window_summary.items()},
                  "decision": decision(window_summary),
                  "limitations": ["three short public development prompts only",
                                  "offline demand-byte simulation; no tokens-per-second claim",
                                  "candidate requires a separate runtime implementation and qualification plan"],
                  "provenance": {"cache_study_sha256": sha256(Path(__file__)),
                                 "replay_sha256": sha256(HERE / "replay.py"),
                                 "harness_hashes": harness_hashes()}}
        atomic_json(output / "report.json", report)
        atomic_json(output / "completion.json", {"format": "slotstream-cache-window-completion-v1",
                    "report_sha256": sha256(output / "report.json"), "qualification": False,
                    "disposition": report["decision"]["disposition"]})
        return 0
    except Exception as error:
        atomic_json(output / "failure.json", {"format": "slotstream-cache-analysis-failure-v1",
                    "error": f"{type(error).__name__}: {error}"})
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collect_parser = commands.add_parser("collect")
    collect_parser.add_argument("--model", type=Path, required=True)
    collect_parser.add_argument("--output", type=Path, required=True)
    analyze_parser = commands.add_parser("analyze")
    analyze_parser.add_argument("--collection", type=Path, required=True)
    analyze_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    return collect(args.model, args.output) if args.command == "collect" else analyze(args.collection, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
