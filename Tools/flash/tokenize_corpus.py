#!/usr/bin/env python3
"""Freeze and validate tokenizer-only natural-text capture shards."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import benchmark
from common import EvidenceError, atomic_json, fresh_output, read_json, sha256, validate_build_identity
from receipts import require_terminal_sampling, validate_receipt_file

TRANSFORMERS_REVISION = "2fa33e1f5e7131a7fc64c28e6d161dcec0d24820"
PINNED_REVISION = "3781190c6bbdf0a7637beda49ba179822612058a"
TOKENIZER_PINS = {
    "tokenizer.json": (12809320, "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"),
    "tokenizer_config.json": (17928, "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27"),
    "config.json": (113950, "06d4ba9c9604e901ed5d8a7c2e8d236d32adbbedd1c3b5d1a621a525bffc594a"),
}
SPLITS = ("training", "development", "qualification")


def canonical_hash(value: dict[str, Any], omitted: str) -> str:
    item = copy.deepcopy(value)
    item.pop(omitted, None)
    data = json.dumps(item, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(data).hexdigest()


def foundation_canonical_hash(value: dict[str, Any], omitted: str) -> str:
    return canonical_hash(value, omitted)


def chat_content_hash(document: dict[str, Any]) -> str:
    value = {key: document.get(key, "") for key in ("reference", "system", "user")}
    return canonical_hash(value, "")


def validate_source(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 8 << 20:
        raise EvidenceError("authored source must be a regular file no larger than 8 MiB")
    source = read_json(path)
    if (set(source) != {"format", "schemaVersion", "manifestSHA256", "documents"}
            or source["format"] != "slotstream-authored-corpus-v1"
            or source["schemaVersion"] != 1
            or source["manifestSHA256"] != canonical_hash(source, "manifestSHA256")
            or not isinstance(source["documents"], list)
            or not 0 < len(source["documents"]) <= 2048):
        raise EvidenceError("authored source identity or shape is invalid")
    identifiers: set[str] = set()
    partitions: dict[str, str] = {}
    for document in source["documents"]:
        common = {"id", "kind", "category", "split", "sourceID", "license",
                  "partitionKey", "scoreTokens"}
        expected = (common | {"text", "textSHA256", "warmupTokens"}
                    if document.get("kind") == "raw"
                    else common | {"system", "user", "reference", "contentSHA256"})
        strings = [document.get(key) for key in ("license", "partitionKey")]
        if (set(document) != expected or not isinstance(document.get("id"), str)
                or re.fullmatch(r"[A-Za-z0-9_-]{1,64}", document["id"]) is None
                or document["id"] in identifiers or document.get("split") not in SPLITS
                or re.fullmatch(r"[A-Za-z0-9_-]{1,64}", document.get("category", "")) is None
                or not isinstance(document.get("sourceID"), str)
                or not 0 < len(document["sourceID"].encode()) <= 256
                or any(ord(character) < 32 or ord(character) == 127 for character in document["sourceID"])
                or not all(isinstance(value, str) and 0 < len(value.encode()) <= 256 for value in strings)
                or type(document.get("scoreTokens")) is not int
                or not 1 <= document["scoreTokens"] <= 64
                or (document["partitionKey"] in partitions
                    and partitions[document["partitionKey"]] != document["split"])):
            raise EvidenceError("authored document fields, identity or split are invalid")
        identifiers.add(document["id"])
        partitions[document["partitionKey"]] = document["split"]
        if document["kind"] == "raw":
            text = document.get("text")
            if (not isinstance(text, str) or not text or len(text.encode()) > 32 << 10
                    or document.get("textSHA256") != hashlib.sha256(text.encode()).hexdigest()
                    or type(document.get("warmupTokens")) is not int
                    or not 0 <= document["warmupTokens"] <= 256):
                raise EvidenceError("raw authored document is invalid")
        elif document["kind"] == "chat":
            values = [document.get(key) for key in ("system", "user", "reference")]
            if (not all(isinstance(value, str) and len(value.encode()) <= 32 << 10 for value in values)
                    or not document["user"] or not document["reference"]
                    or document.get("contentSHA256") != chat_content_hash(document)):
                raise EvidenceError("chat authored document is invalid")
        else:
            raise EvidenceError("authored document kind is unsupported")
    return source


def build_capture_window(full_ids: list[int], *, warmup_count: int,
                         score_tokens: int) -> dict[str, Any]:
    if (not isinstance(full_ids, list) or not all(type(value) is int and 0 <= value < 248320 for value in full_ids)
            or not 0 <= warmup_count <= 256 or not 1 <= score_tokens <= 64
            or len(full_ids) > 2048 or warmup_count + score_tokens >= len(full_ids)):
        raise EvidenceError("token IDs cannot form a bounded capture window")
    positions = []
    for offset in range(score_tokens):
        index = warmup_count + offset
        positions.append({"position": index, "inputID": full_ids[index],
                          "nextTokenID": full_ids[index + 1]})
    return {"warmupIDs": full_ids[:warmup_count], "positions": positions,
            "scored_range": [warmup_count + 1, warmup_count + 1 + score_tokens],
            "unscored_range": [warmup_count + 1 + score_tokens, len(full_ids)]}


def build_chat_capture_window(prompt_ids: list[int], full_ids: list[int],
                              score_tokens: int) -> dict[str, Any]:
    if not prompt_ids or full_ids[:len(prompt_ids)] != prompt_ids:
        raise EvidenceError("completed chat tokens do not preserve the prompt prefix")
    return build_capture_window(full_ids, warmup_count=len(prompt_ids) - 1,
                                score_tokens=score_tokens)


def expected_tokenizer_identity(model: Path) -> dict[str, Any]:
    files = []
    for name in ("tokenizer.json", "tokenizer_config.json", "config.json"):
        size, digest = TOKENIZER_PINS[name]
        path = model / name
        if not path.is_file() or path.stat().st_size != size or sha256(path) != digest:
            raise EvidenceError(f"pinned tokenizer file changed: {name}")
        files.append({"path": name, "bytes": size, "sha256": digest})
    config = read_json(model / "tokenizer_config.json")
    template = config.get("chat_template")
    if not isinstance(template, str) or not template:
        raise EvidenceError("embedded tokenizer template is missing")
    return {"model_revision": PINNED_REVISION,
            "swift_transformers_revision": TRANSFORMERS_REVISION,
            "files": files,
            "embedded_chat_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
            "text_add_special_tokens": False, "thinking": False}


def _validate_shard(path: Path, expected_identity: dict[str, Any],
                    source_sha256: str) -> tuple[dict[str, Any], int]:
    shard = read_json(path)
    required = {"format", "schemaVersion", "manifestSHA256", "tokenSource", "split",
                "sourceCorpusSHA256", "tokenizerIdentity", "shardID", "documents"}
    if (set(shard) != required or shard["format"] != "slotstream-tokenized-capture-shard-v2"
            or shard["schemaVersion"] != 2
            or shard["manifestSHA256"] != foundation_canonical_hash(shard, "manifestSHA256")
            or shard["tokenSource"] != "slotstream-auto-tokenizer-chat-template-v1"
            or shard["split"] not in SPLITS or shard["sourceCorpusSHA256"] != source_sha256
            or shard["tokenizerIdentity"] != expected_identity
            or re.fullmatch(r"shard-[0-9]{3,4}", shard["shardID"]) is None
            or not isinstance(shard["documents"], list) or not shard["documents"]):
        raise EvidenceError("tokenized capture shard is malformed")
    count = 0
    for document in shard["documents"]:
        if (set(document) != {"id", "category", "sourceID", "sourceHash", "split",
                              "warmupIDs", "positions"}
                or document["split"] != shard["split"]
                or re.fullmatch(r"[A-Za-z0-9_-]{1,64}", document["id"]) is None
                or re.fullmatch(r"[A-Za-z0-9_-]{1,64}", document["category"]) is None
                or not document["sourceID"] or len(document["sourceID"].encode()) > 256
                or any(ord(character) < 32 or ord(character) == 127 for character in document["sourceID"])
                or re.fullmatch(r"[0-9a-f]{64}", document["sourceHash"]) is None
                or len(document["warmupIDs"]) > 256 or not document["positions"]):
            raise EvidenceError("tokenized shard document is malformed")
        if any(type(token) is not int or not 0 <= token < 248320 for token in document["warmupIDs"]):
            raise EvidenceError("tokenized shard warmup IDs are invalid")
        expected_position = len(document["warmupIDs"])
        previous_target = None
        for position in document["positions"]:
            if (set(position) != {"position", "inputID", "nextTokenID"}
                    or type(position["position"]) is not int
                    or position["position"] != expected_position
                    or any(type(position[key]) is not int or not 0 <= position[key] < 248320
                           for key in ("inputID", "nextTokenID"))
                    or (previous_target is not None and position["inputID"] != previous_target)):
                raise EvidenceError("tokenized shard position chain is invalid")
            previous_target = position["nextTokenID"]
            expected_position += 1
            count += 1
    if not 0 < count <= 64:
        raise EvidenceError("tokenized shard position count is invalid")
    return shard, count


def validate_corpus(root: Path, *, model: Path | None = None) -> dict[str, Any]:
    root = root.resolve(strict=True)
    native = root / "corpus"
    index_path = native / "corpus.json"
    index = read_json(index_path)
    completion = read_json(native / "completion.json")
    if (completion != {"format": "slotstream-tokenized-corpus-completion-v1",
                       "corpus_sha256": sha256(index_path), "qualification": False}):
        raise EvidenceError("tokenized corpus completion does not bind the index")
    required = {"format", "schema_version", "qualification", "model_loaded", "tokenizer_only",
                "source_manifest_path", "source_manifest_sha256", "source_provenance_scope",
                "tokenizer_identity", "executable_identity", "canonical_test_vector",
                "documents", "shards", "artifacts"}
    if (set(index) != required or index["format"] != "slotstream-tokenized-corpus-v1"
            or index["schema_version"] != 1 or index["qualification"] is not False
            or index["model_loaded"] is not False or index["tokenizer_only"] is not True):
        raise EvidenceError("tokenized corpus index is malformed")
    vector = index["canonical_test_vector"]
    if set(vector) != {"value", "canonical_json", "sha256"}:
        raise EvidenceError("canonical JSON test vector is malformed")
    expected_canonical = json.dumps(vector["value"], sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    if (vector["canonical_json"] != expected_canonical
            or vector["sha256"] != hashlib.sha256(expected_canonical.encode()).hexdigest()):
        raise EvidenceError("Swift and Python canonical JSON encodings differ")
    source_path = Path(index["source_manifest_path"])
    source = validate_source(source_path)
    if index["source_manifest_sha256"] != sha256(source_path):
        raise EvidenceError("tokenized corpus source changed")
    if model is None:
        model = ROOT / "models/jang-6s"
    model = model.resolve()
    identity = expected_tokenizer_identity(model)
    if index["tokenizer_identity"] != identity:
        raise EvidenceError("tokenized corpus tokenizer identity changed")
    binary = Path(index["executable_identity"]["path"])
    executable = index["executable_identity"]
    historical = executable.get("historical")
    expected_executable = {"path", "binary_sha256", "metallib_sha256",
                           "build_identity_sha256", "source_archive_sha256", "historical"}
    if historical is True:
        expected_executable |= {"archive_receipt_path", "archive_receipt_sha256"}
    if set(executable) != expected_executable or type(historical) is not bool:
        raise EvidenceError("tokenizer executable identity fields are malformed")
    build, paths = validate_build_identity(binary, historical=historical)
    if (index["executable_identity"].get("binary_sha256") != build["binary_sha256"]
            or index["executable_identity"].get("metallib_sha256") != build["metallib_sha256"]
            or index["executable_identity"].get("build_identity_sha256") != sha256(paths["identity"])
            or index["executable_identity"].get("source_archive_sha256") != sha256(paths["source_archive"])):
        raise EvidenceError("tokenizer executable identity changed")
    if historical:
        archive_path = Path(executable["archive_receipt_path"])
        archive = validate_receipt_file(archive_path, require_qualified=True)
        relative_binary = str(binary.resolve().relative_to(archive_path.parent.resolve()))
        if (executable["archive_receipt_sha256"] != sha256(archive_path)
                or relative_binary not in archive["artifacts"]
                or archive["artifacts"][relative_binary]["sha256"] != sha256(binary)):
            raise EvidenceError("tokenizer archive receipt does not bind its binary")
    if [item.get("id") for item in index["documents"]] != [item["id"] for item in source["documents"]]:
        raise EvidenceError("tokenized corpus document coverage differs")
    documents = {item["id"]: item for item in index["documents"]}
    for authored in source["documents"]:
        item = documents[authored["id"]]
        required = {"id", "kind", "category", "split", "sourceID", "license", "partitionKey",
                    "source_hash", "full_ids", "full_count", "warmup_count", "scored_range",
                    "unscored_range", "capture"}
        if authored["kind"] == "chat":
            required.add("prompt_count")
        if (set(item) != required or item["kind"] != authored["kind"]
                or any(item[key] != authored[key] for key in
                       ("id", "category", "split", "sourceID", "license", "partitionKey"))
                or item["source_hash"] != authored.get("textSHA256", authored.get("contentSHA256"))
                or item["full_count"] != len(item["full_ids"])):
            raise EvidenceError("tokenized document provenance differs from authored source")
        warmup = (item.get("prompt_count", 0) - 1 if authored["kind"] == "chat"
                  else authored["warmupTokens"])
        expected_window = build_capture_window(item["full_ids"], warmup_count=warmup,
                                               score_tokens=authored["scoreTokens"])
        expected_capture = {"id": item["id"], "category": item["category"],
                            "sourceID": item["sourceID"], "sourceHash": item["source_hash"],
                            "split": item["split"], "warmupIDs": expected_window["warmupIDs"],
                            "positions": expected_window["positions"]}
        if (item["warmup_count"] != warmup or item["scored_range"] != expected_window["scored_range"]
                or item["unscored_range"] != expected_window["unscored_range"]
                or item["capture"] != expected_capture):
            raise EvidenceError("tokenized document capture window is inconsistent")
    seen: list[str] = []
    shard_documents: dict[str, dict[str, Any]] = {}
    total_positions = 0
    for ordinal, entry in enumerate(index["shards"]):
        if set(entry) != {"path", "split", "positions", "sha256"}:
            raise EvidenceError("tokenized shard index entry is malformed")
        expected_name = f"shards/shard-{ordinal:03d}.json"
        if entry["path"] != expected_name or entry["sha256"] != sha256(native / expected_name):
            raise EvidenceError("tokenized shard order or hash changed")
        shard, count = _validate_shard(native / expected_name, identity, index["source_manifest_sha256"])
        if entry["split"] != shard["split"] or entry["positions"] != count:
            raise EvidenceError("tokenized shard summary differs")
        seen.extend(item["id"] for item in shard["documents"])
        for document in shard["documents"]:
            if document["id"] in shard_documents:
                raise EvidenceError("tokenized document appears in more than one shard")
            shard_documents[document["id"]] = document
        total_positions += count
    if seen != [item["id"] for item in index["documents"]]:
        # Producer orders shards by split, so compare the explicit deterministic grouping.
        expected = [item["id"] for split in SPLITS for item in index["documents"] if item["split"] == split]
        if seen != expected:
            raise EvidenceError("tokenized shard document order or coverage changed")
    if set(shard_documents) != set(documents) or any(
            shard_documents[name] != documents[name]["capture"] for name in documents):
        raise EvidenceError("tokenized shard documents differ from the corpus index")
    for name, artifact in index["artifacts"].items():
        path = native / name
        if (set(artifact) != {"bytes", "sha256"} or not path.is_file()
                or path.stat().st_size != artifact["bytes"] or sha256(path) != artifact["sha256"]):
            raise EvidenceError("tokenized corpus artifact changed")
    receipt = validate_receipt_file(root / "receipt.json")
    require_terminal_sampling(receipt)
    if (not receipt["result"]["functional_success"] or receipt["memory"]["target_gb_decimal"] != 1
            or receipt["command"][:4] != [str(binary), "flash-tokenize", "--model", str(model)]):
        raise EvidenceError("tokenizer launcher receipt differs from the frozen command")
    return {"index": index, "source": source, "positions": total_positions}


def freeze(binary: Path, model: Path, source: Path, output: Path) -> int:
    if output.exists():
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
        command = [str(binary), "flash-tokenize", "--model", str(model),
                   "--source", str(source.resolve()), "--output", str(output / "corpus")]
        code = benchmark.launch(output, 1, 120, command)
        if code:
            raise EvidenceError("tokenizer-only monitored child failed")
        validate_corpus(output, model=model)
        return 0
    except Exception as error:
        if not output.exists():
            try:
                output = fresh_output(output)
            except EvidenceError:
                return 1
        if not (output / "failure.json").exists():
            atomic_json(output / "failure.json", {"format": "slotstream-tokenize-failure-v1",
                                                   "error": f"{type(error).__name__}: {error}"})
        return 1


def self_test() -> int:
    import subprocess
    result = subprocess.run([sys.executable, "-m", "unittest", "Tools/flash/test_tokenize.py"],
                            cwd=ROOT)
    return result.returncode


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    sub = parser.add_subparsers(dest="command")
    freeze_parser = sub.add_parser("freeze")
    freeze_parser.add_argument("--binary", type=Path, required=True)
    freeze_parser.add_argument("--model", type=Path, required=True)
    freeze_parser.add_argument("--source", type=Path, required=True)
    freeze_parser.add_argument("--output", type=Path, required=True)
    validate_parser = sub.add_parser("validate")
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
    parser.error("choose --self-test, freeze or validate")


if __name__ == "__main__":
    raise SystemExit(main())
