#!/usr/bin/env python3
"""Freeze and validate complete tokenizer-only task prompts."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import benchmark
from common import EvidenceError, atomic_json, fresh_output, regular_file, sha256, validate_build_identity
from receipts import require_terminal_sampling, validate_receipt_file

TRANSFORMERS_REVISION = "2fa33e1f5e7131a7fc64c28e6d161dcec0d24820"
PINNED_REVISION = "3781190c6bbdf0a7637beda49ba179822612058a"
TOKENIZER_PINS = {
    "tokenizer.json": (12809320, "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"),
    "tokenizer_config.json": (17928, "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27"),
    "config.json": (113950, "06d4ba9c9604e901ed5d8a7c2e8d236d32adbbedd1c3b5d1a621a525bffc594a"),
}
SPLITS = ("training", "development", "qualification")
SOURCE_LIMIT = 8 << 20
TEXT_LIMIT = 32 << 10
TOKEN_LIMIT = 8192
AGGREGATE_TOKEN_LIMIT = 4_000_000
OUTPUT_LIMIT = 128 << 20
VOCAB_SIZE = 248_320


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise EvidenceError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _constant(value: str) -> None:
    raise EvidenceError(f"non-finite JSON number: {value}")


def read_strict_json(path: Path, *, limit: int | None = None) -> dict[str, Any]:
    if path.is_symlink():
        raise EvidenceError(f"symlink is not accepted: {path}")
    try:
        before = path.stat()
        if not path.is_file() or (limit is not None and before.st_size > limit):
            raise EvidenceError(f"regular bounded JSON file required: {path}")
        with path.open("rb") as stream:
            data = stream.read((limit + 1) if limit is not None else -1)
            opened = os.fstat(stream.fileno())
        after = path.stat()
    except OSError as error:
        raise EvidenceError(f"cannot read JSON at {path}: {error}") from error
    if (limit is not None and len(data) > limit) or len(data) != before.st_size:
        raise EvidenceError(f"JSON file exceeds its bound or changed: {path}")
    identity = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    if identity(before) != identity(opened) or identity(opened) != identity(after):
        raise EvidenceError(f"JSON file changed while being read: {path}")
    try:
        value = json.loads(data, object_pairs_hook=_pairs, parse_constant=_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvidenceError(f"invalid JSON at {path}: {error}") from error
    if not isinstance(value, dict):
        raise EvidenceError(f"JSON object required at {path}")
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_hash(value: dict[str, Any], omitted: str) -> str:
    item = copy.deepcopy(value)
    item.pop(omitted, None)
    return hashlib.sha256(canonical_json(item).encode()).hexdigest()


def chat_content_hash(document: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json({
        "system": document["system"], "user": document["user"],
    }).encode()).hexdigest()


def validate_source(path: Path) -> dict[str, Any]:
    source = read_strict_json(path, limit=SOURCE_LIMIT)
    if (set(source) != {"format", "schemaVersion", "manifestSHA256", "documents"}
            or source["format"] != "slotstream-prompt-source-v1"
            or type(source["schemaVersion"]) is not int or source["schemaVersion"] != 1
            or not isinstance(source["manifestSHA256"], str)
            or source["manifestSHA256"] != canonical_hash(source, "manifestSHA256")
            or not isinstance(source["documents"], list)
            or not 0 < len(source["documents"]) <= 2048):
        raise EvidenceError("prompt source identity or shape is invalid")
    identifiers: set[str] = set()
    partitions: dict[str, str] = {}
    common = {"id", "kind", "category", "split", "sourceID", "license", "partitionKey"}
    for document in source["documents"]:
        if not isinstance(document, dict):
            raise EvidenceError("prompt document must be an object")
        kind = document.get("kind")
        expected = (common | {"text", "textSHA256"} if kind == "raw"
                    else common | {"system", "user", "contentSHA256"} if kind == "chat"
                    else set())
        strings = [document.get(key) for key in ("sourceID", "license", "partitionKey")]
        if (not expected or set(document) != expected
                or not isinstance(document.get("id"), str)
                or re.fullmatch(r"[A-Za-z0-9_-]{1,64}", document["id"]) is None
                or document["id"] in identifiers
                or not isinstance(document.get("category"), str)
                or re.fullmatch(r"[A-Za-z0-9_-]{1,64}", document["category"]) is None
                or document.get("split") not in SPLITS
                or not all(isinstance(value, str) and 0 < len(value.encode()) <= 256
                           for value in strings)
                or any(ord(character) < 32 or ord(character) == 127
                       for character in document["sourceID"])
                or (document["partitionKey"] in partitions
                    and partitions[document["partitionKey"]] != document["split"])):
            raise EvidenceError("prompt document fields, identity, or split are invalid")
        identifiers.add(document["id"])
        partitions[document["partitionKey"]] = document["split"]
        if kind == "raw":
            text = document["text"]
            if (not isinstance(text, str) or not text or len(text.encode()) > TEXT_LIMIT
                    or not isinstance(document["textSHA256"], str)
                    or document["textSHA256"] != hashlib.sha256(text.encode()).hexdigest()):
                raise EvidenceError("raw prompt document is invalid")
        else:
            system, user = document["system"], document["user"]
            if (not isinstance(system, str) or not isinstance(user, str) or not user
                    or len(system.encode()) > TEXT_LIMIT or len(user.encode()) > TEXT_LIMIT
                    or not isinstance(document["contentSHA256"], str)
                    or document["contentSHA256"] != chat_content_hash(document)):
                raise EvidenceError("chat prompt document is invalid")
    return source


def source_metadata(document: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(document)
    if document["kind"] == "raw":
        result.pop("text")
    else:
        result.pop("system")
        result.pop("user")
    return result


def expected_tokenizer_identity(model: Path) -> dict[str, Any]:
    files = []
    for name in ("tokenizer.json", "tokenizer_config.json", "config.json"):
        size, digest = TOKENIZER_PINS[name]
        path = regular_file(model / name, within=model)
        if path.stat().st_size != size or sha256(path) != digest:
            raise EvidenceError(f"pinned tokenizer file changed: {name}")
        files.append({"path": name, "bytes": size, "sha256": digest})
    config = read_strict_json(model / "tokenizer_config.json", limit=1 << 20)
    template = config.get("chat_template")
    if not isinstance(template, str) or not template:
        raise EvidenceError("embedded tokenizer template is missing")
    return {
        "model_revision": PINNED_REVISION,
        "swift_transformers_revision": TRANSFORMERS_REVISION,
        "files": files,
        "embedded_chat_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
        "text_add_special_tokens": False,
        "thinking": False,
    }


def _exact(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise EvidenceError(f"{label} fields differ")
    return value


def _safe_artifact(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise EvidenceError(f"unsafe prompt artifact path: {relative}")
    return regular_file(root / candidate, within=root)


def _validate_executable(index: dict[str, Any], binary: Path) -> None:
    executable = _exact(index["executable_identity"], {
        "path", "binary_sha256", "metallib_sha256", "build_identity_sha256",
        "source_archive_sha256", "historical",
    } | ({"archive_receipt_path", "archive_receipt_sha256"}
         if index["executable_identity"].get("historical") is True else set()),
        "executable identity")
    # A moved checkout may retain its original location as a directory alias.
    # Require the same actual file; all byte/archive checks below remain intact.
    if (type(executable["historical"]) is not bool
            or Path(executable["path"]).resolve(strict=True) != binary.resolve(strict=True)):
        raise EvidenceError("executable identity path or historical flag differs")
    build, paths = validate_build_identity(binary, historical=executable["historical"])
    if (executable["binary_sha256"] != build["binary_sha256"]
            or executable["metallib_sha256"] != build["metallib_sha256"]
            or executable["build_identity_sha256"] != sha256(paths["identity"])
            or executable["source_archive_sha256"] != sha256(paths["source_archive"])):
        raise EvidenceError("prompt tokenizer executable identity changed")
    if executable["historical"]:
        archive_path = Path(executable["archive_receipt_path"])
        archive = validate_receipt_file(archive_path, require_qualified=True)
        try:
            relative_binary = str(binary.resolve().relative_to(archive_path.parent.resolve()))
        except ValueError as error:
            raise EvidenceError("archived binary escapes its receipt") from error
        if (executable["archive_receipt_sha256"] != sha256(archive_path)
                or relative_binary not in archive["artifacts"]
                or archive["artifacts"][relative_binary]["sha256"] != sha256(binary)):
            raise EvidenceError("prompt tokenizer archive receipt does not bind its binary")


def validate_corpus(root: Path) -> dict[str, Any]:
    if root.is_symlink():
        raise EvidenceError("symlinked corpus root is not accepted")
    root = root.resolve(strict=True)
    if (root / "failure.json").exists() or (root / "corpus/failure.json").exists():
        raise EvidenceError("failed output cannot be accepted as a prompt corpus")
    native = root / "corpus"
    expected_root = {"corpus", "stdout.txt", "stderr.txt", "memory.jsonl",
                     "receipt.json", "completion.json"}
    if {path.name for path in root.iterdir()} != expected_root:
        raise EvidenceError("prompt freeze root has missing or extra artifacts")
    expected_native = {"prompts", "prompts.json", "completion.json"}
    if not native.is_dir() or {path.name for path in native.iterdir()} != expected_native:
        raise EvidenceError("native prompt corpus has missing or extra artifacts")
    regular_file(native / "prompts.json", within=native)
    index_path = native / "prompts.json"
    index = read_strict_json(index_path, limit=32 << 20)
    completion = read_strict_json(native / "completion.json", limit=4096)
    if completion != {
        "format": "slotstream-prompt-corpus-completion-v1",
        "prompts_sha256": sha256(index_path),
        "qualification": False,
    }:
        raise EvidenceError("prompt corpus completion does not bind the index")
    required = {
        "format", "schema_version", "qualification", "model_loaded", "tokenizer_only",
        "source_manifest_path", "source_manifest_sha256", "source_provenance_scope",
        "tokenizer_identity", "executable_identity", "canonical_test_vector", "context",
        "documents", "aggregate", "artifacts",
    }
    if (set(index) != required or index["format"] != "slotstream-prompt-corpus-v1"
            or type(index["schema_version"]) is not int or index["schema_version"] != 1
            or index["qualification"] is not False or index["model_loaded"] is not False
            or index["tokenizer_only"] is not True
            or not isinstance(index["source_provenance_scope"], str)
            or not index["source_provenance_scope"]):
        raise EvidenceError("prompt corpus index is malformed")
    context = _exact(index["context"], {
        "artifact_token_limit", "inference_context_limit",
        "artifact_limit_is_inference_permission",
    }, "prompt context")
    if (type(context["artifact_token_limit"]) is not int
            or type(context["inference_context_limit"]) is not int
            or type(context["artifact_limit_is_inference_permission"]) is not bool
            or context != {
        "artifact_token_limit": 8192,
        "inference_context_limit": 2048,
        "artifact_limit_is_inference_permission": False,
    }):
        raise EvidenceError("tokenizer artifact and inference context distinction changed")
    vector = _exact(index["canonical_test_vector"], {"value", "canonical_json", "sha256"},
                    "canonical test vector")
    expected_canonical = canonical_json(vector["value"])
    if (vector["canonical_json"] != expected_canonical
            or vector["sha256"] != hashlib.sha256(expected_canonical.encode()).hexdigest()):
        raise EvidenceError("Swift and Python canonical JSON encodings differ")

    source_path = Path(index["source_manifest_path"])
    source = validate_source(source_path)
    if index["source_manifest_sha256"] != sha256(source_path):
        raise EvidenceError("prompt source changed after tokenization")

    receipt = validate_receipt_file(root / "receipt.json")
    require_terminal_sampling(receipt)
    command = receipt["command"]
    if (len(command) != 8 or command[1] != "flash-tokenize-prompts"
            or command[2] != "--model" or command[4] != "--source"
            or command[6] != "--output"):
        raise EvidenceError("launcher command is not the exact prompt tokenizer invocation")
    binary = Path(command[0]).resolve(strict=True)
    model = Path(command[3]).resolve(strict=True)
    command_source = Path(command[5]).resolve(strict=True)
    command_output = Path(command[7]).resolve(strict=True)
    if (command_source != source_path.resolve(strict=True) or command_output != native
            or receipt["qualified"] is not False
            or receipt["result"]["functional_success"] is not True
            or receipt["memory"]["target_gb_decimal"] != 1
            or receipt["memory"]["peak_bytes"] > 1_000_000_000):
        raise EvidenceError("launcher receipt does not bind the bounded prompt freeze")
    launcher_executable = _exact(
        receipt.get("identities", {}).get("executable"), {"path", "bytes", "sha256"},
        "launcher executable identity")
    if (not isinstance(launcher_executable["path"], str)
            or type(launcher_executable["bytes"]) is not int
            or launcher_executable["bytes"] <= 0
            or not isinstance(launcher_executable["sha256"], str)
            or Path(launcher_executable["path"]).resolve(strict=True) != binary
            or launcher_executable["bytes"] != binary.stat().st_size
            or launcher_executable["sha256"] != sha256(binary)
            or launcher_executable["sha256"]
            != index["executable_identity"].get("binary_sha256")):
        raise EvidenceError("launcher executable identity does not bind the native index")
    identity = expected_tokenizer_identity(model)
    if index["tokenizer_identity"] != identity:
        raise EvidenceError("prompt tokenizer identity changed")
    _validate_executable(index, binary)

    documents = index["documents"]
    if (not isinstance(documents, list) or len(documents) != len(source["documents"])
            or len(documents) > 2048):
        raise EvidenceError("prompt document coverage differs from source")
    expected_paths = [f"prompts/prompt-{ordinal:04d}.json"
                      for ordinal in range(len(source["documents"]))]
    prompt_directory = native / "prompts"
    if prompt_directory.is_symlink() or not prompt_directory.is_dir():
        raise EvidenceError("prompt artifact directory is missing or symlinked")
    actual_paths = sorted(str(path.relative_to(native)) for path in prompt_directory.iterdir())
    if actual_paths != expected_paths:
        raise EvidenceError("prompt artifacts are missing, extra, or reordered")
    expected_launcher_artifacts = {
        "stdout.txt", "stderr.txt", "memory.jsonl", "corpus/prompts.json",
        *(f"corpus/{path}" for path in expected_paths),
    }
    if set(receipt["artifacts"]) != expected_launcher_artifacts:
        raise EvidenceError("launcher artifact inventory has missing or extra records")
    artifacts = index["artifacts"]
    if not isinstance(artifacts, dict) or set(artifacts) != set(expected_paths):
        raise EvidenceError("prompt artifact inventory differs")

    aggregate_tokens = 0
    aggregate_bytes = 0
    seen_ids: set[str] = set()
    for ordinal, (authored, entry) in enumerate(zip(source["documents"], documents)):
        path_name = expected_paths[ordinal]
        entry = _exact(entry, {"id", "path", "prompt_count", "prompt_ids_sha256",
                               "source_hash", "artifact_bytes", "artifact_sha256"},
                       "prompt index entry")
        source_hash = authored.get("textSHA256", authored.get("contentSHA256"))
        if (entry["id"] != authored["id"] or entry["id"] in seen_ids
                or entry["path"] != path_name or entry["source_hash"] != source_hash
                or type(entry["prompt_count"]) is not int
                or type(entry["artifact_bytes"]) is not int):
            raise EvidenceError("prompt index order or source binding differs")
        seen_ids.add(entry["id"])
        path = _safe_artifact(native, path_name)
        artifact = _exact(artifacts[path_name], {"bytes", "sha256"}, "artifact receipt")
        if (type(artifact["bytes"]) is not int or artifact["bytes"] <= 0
                or artifact["bytes"] != path.stat().st_size
                or artifact["sha256"] != sha256(path)
                or entry["artifact_bytes"] != artifact["bytes"]
                or entry["artifact_sha256"] != artifact["sha256"]):
            raise EvidenceError("prompt artifact bytes or hash changed")
        prompt = read_strict_json(path, limit=OUTPUT_LIMIT)
        _exact(prompt, {"format", "schemaVersion", "sourceDocument", "promptIDs",
                        "promptCount", "promptIDsSHA256"}, "prompt artifact")
        prompt_ids = prompt["promptIDs"]
        if (prompt["format"] != "slotstream-tokenized-prompt-v1"
                or type(prompt["schemaVersion"]) is not int or prompt["schemaVersion"] != 1
                or prompt["sourceDocument"] != source_metadata(authored)
                or type(prompt["promptCount"]) is not int
                or not isinstance(prompt_ids, list) or not 0 < len(prompt_ids) <= TOKEN_LIMIT
                or any(type(token) is not int or not 0 <= token < VOCAB_SIZE for token in prompt_ids)):
            raise EvidenceError("tokenized prompt is malformed or differs from its source")
        ids_hash = hashlib.sha256(canonical_json(prompt_ids).encode()).hexdigest()
        if (prompt["promptCount"] != len(prompt_ids) or prompt["promptIDsSHA256"] != ids_hash
                or entry["prompt_count"] != len(prompt_ids)
                or entry["prompt_ids_sha256"] != ids_hash):
            raise EvidenceError("prompt token count or ID hash differs")
        aggregate_tokens += len(prompt_ids)
        aggregate_bytes += path.stat().st_size
    if aggregate_tokens > AGGREGATE_TOKEN_LIMIT:
        raise EvidenceError("prompt corpus exceeds the aggregate token quota")
    aggregate = _exact(index["aggregate"], {
        "document_count", "prompt_token_count", "prompt_artifact_bytes",
    }, "prompt aggregate")
    if (any(type(aggregate[key]) is not int for key in (
            "document_count", "prompt_token_count", "prompt_artifact_bytes"))
            or aggregate != {
        "document_count": len(documents),
        "prompt_token_count": aggregate_tokens,
        "prompt_artifact_bytes": aggregate_bytes,
    }):
        raise EvidenceError("prompt aggregate counts or bytes differ")
    native_files = [path for path in native.rglob("*") if path.is_file()]
    if any(path.is_symlink() for path in native.rglob("*")):
        raise EvidenceError("prompt corpus contains a symlink")
    if sum(path.stat().st_size for path in native_files) > OUTPUT_LIMIT:
        raise EvidenceError("prompt corpus exceeds the aggregate output quota")
    return {"index": index, "source": source, "prompt_token_count": aggregate_tokens}


def freeze(binary: Path, model: Path, source: Path, output: Path) -> int:
    if output.exists() or output.is_symlink():
        return 1
    try:
        validate_source(source)
        binary = binary.resolve(strict=True)
        archive_receipt = binary.parent.parent / "receipt.json"
        historical = archive_receipt.is_file()
        validate_build_identity(binary, historical=historical)
        if historical:
            validate_receipt_file(archive_receipt, require_qualified=True)
        model = model.resolve(strict=True)
        expected_tokenizer_identity(model)
        command = [
            str(binary), "flash-tokenize-prompts", "--model", str(model),
            "--source", str(source.resolve()), "--output", str(output / "corpus"),
        ]
        code = benchmark.launch(output, 1, 120, command)
        if code:
            raise EvidenceError("tokenizer-only monitored child failed")
        validate_corpus(output)
        return 0
    except Exception as error:
        if not output.exists():
            try:
                output = fresh_output(output)
            except EvidenceError:
                return 1
        if not (output / "failure.json").exists():
            atomic_json(output / "failure.json", {
                "format": "slotstream-prompt-freeze-failure-v1",
                "error": f"{type(error).__name__}: {error}",
            })
        return 1


def self_test() -> int:
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "Tools/flash/test_prompt_corpus.py"], cwd=ROOT)
    return result.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    subparsers = parser.add_subparsers(dest="command")
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--binary", type=Path, required=True)
    freeze_parser.add_argument("--model", type=Path, required=True)
    freeze_parser.add_argument("--source", type=Path, required=True)
    freeze_parser.add_argument("--output", type=Path, required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--corpus", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    if args.command == "freeze":
        return freeze(args.binary, args.model, args.source, args.output)
    if args.command == "validate":
        try:
            validate_corpus(args.corpus)
            return 0
        except Exception as error:
            print(f"{type(error).__name__}: {error}", file=sys.stderr)
            return 1
    parser.error("choose --self-test, freeze, or validate")


if __name__ == "__main__":
    raise SystemExit(main())
