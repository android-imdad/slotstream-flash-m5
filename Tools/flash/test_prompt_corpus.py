#!/usr/bin/env python3
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import EvidenceError, FLASH_ROOT, atomic_json, sha256

SPEC = importlib.util.spec_from_file_location("prompt_corpus_tool", HERE / "prompt_corpus.py")
prompt_tool = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(prompt_tool)


class PromptCorpusTests(unittest.TestCase):
    def setUp(self) -> None:
        FLASH_ROOT.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="prompt-corpus-test-", dir=FLASH_ROOT)
        self.root = Path(self.temporary.name)
        self.fixture = HERE.parent / "fixtures/flash/prompt-synthetic-v1.json"
        self.fake_ordinal = 0

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_executable_directory_alias_preserves_identity_but_other_file_fails(self):
        directory = self.root / "moved"
        directory.mkdir()
        binary = directory / "slotstream"
        binary.write_bytes(b"fixture")
        alias = self.root / "original"
        alias.symlink_to(directory, target_is_directory=True)
        index = {"executable_identity": {"path": str(alias / "slotstream"),
            "binary_sha256": "a", "metallib_sha256": "b", "build_identity_sha256": "c",
            "source_archive_sha256": "d", "historical": False}}
        build = {"binary_sha256": "a", "metallib_sha256": "b"}
        with mock.patch.object(prompt_tool, "validate_build_identity", return_value=(build, {
                "identity": "identity", "source_archive": "archive"})), \
                mock.patch.object(prompt_tool, "sha256", side_effect=lambda path: {"identity": "c", "archive": "d"}[path]):
            prompt_tool._validate_executable(index, binary)
            other = self.root / "another-binary"
            other.write_bytes(b"fixture")
            with self.assertRaisesRegex(EvidenceError, "identity path"):
                prompt_tool._validate_executable(index, other)

    def write_source(self, documents: list[dict], name: str = "source.json") -> Path:
        source = {
            "format": "slotstream-prompt-source-v1",
            "schemaVersion": 1,
            "manifestSHA256": "",
            "documents": documents,
        }
        source["manifestSHA256"] = prompt_tool.canonical_hash(source, "manifestSHA256")
        path = self.root / name
        atomic_json(path, source)
        return path

    @staticmethod
    def raw_document(identifier: str = "doc", text: str = "hello / \"world\"\nélan e\u0301") -> dict:
        return {
            "id": identifier,
            "kind": "raw",
            "category": "prose",
            "split": "development",
            "sourceID": "source/fixture",
            "license": "CC0-1.0",
            "partitionKey": f"{identifier}-partition",
            "text": text,
            "textSHA256": hashlib.sha256(text.encode()).hexdigest(),
        }

    @staticmethod
    def chat_document(identifier: str = "chat") -> dict:
        document = {
            "id": identifier,
            "kind": "chat",
            "category": "instruction",
            "split": "training",
            "sourceID": "source/chat",
            "license": "CC0-1.0",
            "partitionKey": f"{identifier}-partition",
            "system": "System / \"quoted\"\nélan e\u0301",
            "user": "今天怎么样？",
            "contentSHA256": "",
        }
        document["contentSHA256"] = prompt_tool.chat_content_hash(document)
        return document

    def test_synthetic_fixture_is_strict_and_explicitly_marks_boundaries(self) -> None:
        source = prompt_tool.validate_source(self.fixture)
        self.assertEqual({document["kind"] for document in source["documents"]}, {"raw", "chat"})
        self.assertEqual({document["split"] for document in source["documents"]},
                         {"training", "development", "qualification"})
        long = next(document for document in source["documents"]
                    if document["id"] == "synthetic-long-boundary")
        self.assertIn("quota-repetition", long["sourceID"])
        self.assertEqual(next(document for document in source["documents"]
                              if document["id"] == "synthetic-no-target")["text"], "Hi")

    def test_raw_and_chat_hashes_cover_canonical_unicode_and_escaping(self) -> None:
        path = self.write_source([self.raw_document(), self.chat_document()])
        source = prompt_tool.validate_source(path)
        self.assertEqual(source["documents"][0]["text"], "hello / \"world\"\nélan e\u0301")
        self.assertEqual(source["documents"][1]["user"], "今天怎么样？")
        self.assertNotEqual("é", "e\u0301")

    def mutated_source(self, mutation) -> Path:
        source = json.loads(self.fixture.read_text())
        mutation(source)
        source["manifestSHA256"] = prompt_tool.canonical_hash(source, "manifestSHA256")
        path = self.root / "mutated.json"
        atomic_json(path, source)
        return path

    def test_source_rejects_unknown_null_roles_bad_ids_and_hashes(self) -> None:
        mutations = [
            lambda value: value["documents"][0].update(extra=True),
            lambda value: value["documents"][0].update(text=None),
            lambda value: value["documents"][0].update(id="unsafe id"),
            lambda value: value["documents"][0].update(sourceID="bad\nsource"),
            lambda value: value["documents"][0].update(textSHA256="0" * 64),
            lambda value: value["documents"][4].update(reference="not-allowed"),
            lambda value: value["documents"][4].update(system=None),
        ]
        for index, mutation in enumerate(mutations):
            with self.subTest(index=index), self.assertRaises(EvidenceError):
                prompt_tool.validate_source(self.mutated_source(mutation))

    def test_source_rejects_duplicate_ids_cross_split_partitions_and_bad_manifest(self) -> None:
        with self.assertRaises(EvidenceError):
            prompt_tool.validate_source(self.mutated_source(
                lambda value: value["documents"][1].update(id=value["documents"][0]["id"])))
        with self.assertRaises(EvidenceError):
            prompt_tool.validate_source(self.mutated_source(
                lambda value: value["documents"][1].update(
                    partitionKey=value["documents"][0]["partitionKey"])))
        source = json.loads(self.fixture.read_text())
        source["manifestSHA256"] = "0" * 64
        path = self.root / "manifest.json"
        atomic_json(path, source)
        with self.assertRaises(EvidenceError):
            prompt_tool.validate_source(path)

    def test_duplicate_json_keys_and_oversized_source_fail(self) -> None:
        duplicate = self.root / "duplicate.json"
        duplicate.write_text('{"format":"a","format":"b"}')
        with self.assertRaisesRegex(EvidenceError, "duplicate"):
            prompt_tool.read_strict_json(duplicate)
        oversized = self.root / "oversized.json"
        oversized.write_bytes(b" " * ((8 << 20) + 1))
        with self.assertRaises(EvidenceError):
            prompt_tool.validate_source(oversized)

    def test_tokenizer_identity_rejects_wrong_template_or_pin(self) -> None:
        model = self.root / "tiny-model"
        model.mkdir()
        files = {
            "tokenizer.json": b"tokens",
            "tokenizer_config.json": json.dumps({"chat_template": "template-v1"}).encode(),
            "config.json": b"{}",
        }
        pins = {}
        for name, data in files.items():
            (model / name).write_bytes(data)
            pins[name] = (len(data), hashlib.sha256(data).hexdigest())
        with mock.patch.object(prompt_tool, "TOKENIZER_PINS", pins):
            prompt_tool.expected_tokenizer_identity(model)
            (model / "tokenizer_config.json").write_text(json.dumps({"chat_template": "wrong"}))
            with self.assertRaisesRegex(EvidenceError, "pinned tokenizer file changed"):
                prompt_tool.expected_tokenizer_identity(model)

    def fake_corpus(self, prompt_ids: list[int] | None = None):
        self.fake_ordinal += 1
        tag = str(self.fake_ordinal)
        prompt_ids = prompt_ids or [11, 12, 13]
        authored = self.raw_document(text="hello world")
        source_path = self.write_source([authored], f"source-{tag}.json")
        source = prompt_tool.validate_source(source_path)
        evidence = self.root / f"evidence-{tag}"
        native = evidence / "corpus"
        prompts = native / "prompts"
        prompts.mkdir(parents=True)
        ids_hash = hashlib.sha256(prompt_tool.canonical_json(prompt_ids).encode()).hexdigest()
        prompt = {
            "format": "slotstream-tokenized-prompt-v1",
            "schemaVersion": 1,
            "sourceDocument": prompt_tool.source_metadata(authored),
            "promptIDs": prompt_ids,
            "promptCount": len(prompt_ids),
            "promptIDsSHA256": ids_hash,
        }
        prompt_path = prompts / "prompt-0000.json"
        atomic_json(prompt_path, prompt)
        binary = self.root / f"bin-{tag}/slotstream"
        binary.parent.mkdir()
        binary.write_bytes(b"binary")
        binary_sha256 = sha256(binary)
        model = self.root / f"model-{tag}"
        model.mkdir()
        identity = {"model_revision": "test-revision"}
        entry = {
            "id": authored["id"], "path": "prompts/prompt-0000.json",
            "prompt_count": len(prompt_ids), "prompt_ids_sha256": ids_hash,
            "source_hash": authored["textSHA256"], "artifact_bytes": prompt_path.stat().st_size,
            "artifact_sha256": sha256(prompt_path),
        }
        index = {
            "format": "slotstream-prompt-corpus-v1", "schema_version": 1,
            "qualification": False, "model_loaded": False, "tokenizer_only": True,
            "source_manifest_path": str(source_path), "source_manifest_sha256": sha256(source_path),
            "source_provenance_scope": "unverified fixture", "tokenizer_identity": identity,
            "executable_identity": {
                "path": str(binary), "binary_sha256": binary_sha256,
                "metallib_sha256": "b" * 64, "build_identity_sha256": "c" * 64,
                "source_archive_sha256": "d" * 64, "historical": False,
            },
            "canonical_test_vector": {
                "value": {"slash": "/", "unicode": "é e\u0301"},
                "canonical_json": prompt_tool.canonical_json({"slash": "/", "unicode": "é e\u0301"}),
                "sha256": hashlib.sha256(prompt_tool.canonical_json(
                    {"slash": "/", "unicode": "é e\u0301"}).encode()).hexdigest(),
            },
            "context": {"artifact_token_limit": 8192, "inference_context_limit": 2048,
                        "artifact_limit_is_inference_permission": False},
            "documents": [entry],
            "aggregate": {"document_count": 1, "prompt_token_count": len(prompt_ids),
                          "prompt_artifact_bytes": prompt_path.stat().st_size},
            "artifacts": {"prompts/prompt-0000.json": {
                "bytes": prompt_path.stat().st_size, "sha256": sha256(prompt_path)}},
        }
        index_path = native / "prompts.json"
        atomic_json(index_path, index)
        atomic_json(native / "completion.json", {
            "format": "slotstream-prompt-corpus-completion-v1",
            "prompts_sha256": sha256(index_path), "qualification": False,
        })
        for name in ("stdout.txt", "stderr.txt", "memory.jsonl", "completion.json", "receipt.json"):
            (evidence / name).write_text("fixture")
        receipt = {
            "qualified": False,
            "result": {"functional_success": True},
            "memory": {"target_gb_decimal": 1, "peak_bytes": 12345},
            "artifacts": {
                "stdout.txt": {}, "stderr.txt": {}, "memory.jsonl": {},
                "corpus/prompts.json": {},
                "corpus/prompts/prompt-0000.json": {},
            },
            "identities": {"executable": {
                "path": str(binary), "bytes": binary.stat().st_size,
                "sha256": binary_sha256,
            }},
            "command": [str(binary), "flash-tokenize-prompts", "--model", str(model),
                        "--source", str(source_path), "--output", str(native)],
        }
        return evidence, index_path, prompt_path, identity, receipt, source

    def validate_fake(self, evidence: Path, identity: dict, receipt: dict):
        with mock.patch.object(prompt_tool, "validate_receipt_file", return_value=receipt), \
             mock.patch.object(prompt_tool, "require_terminal_sampling"), \
             mock.patch.object(prompt_tool, "expected_tokenizer_identity", return_value=identity), \
             mock.patch.object(prompt_tool, "_validate_executable"):
            return prompt_tool.validate_corpus(evidence)

    def rebind_index(self, index_path: Path) -> None:
        atomic_json(index_path.parent / "completion.json", {
            "format": "slotstream-prompt-corpus-completion-v1",
            "prompts_sha256": sha256(index_path), "qualification": False,
        })

    def test_validate_joins_source_order_ids_hashes_context_and_launcher(self) -> None:
        evidence, _, _, identity, receipt, _ = self.fake_corpus()
        result = self.validate_fake(evidence, identity, receipt)
        self.assertEqual(result["prompt_token_count"], 3)
        self.assertFalse(result["index"]["model_loaded"])
        self.assertTrue(result["index"]["tokenizer_only"])

    def test_validate_rejects_forged_integer_and_boolean_types(self) -> None:
        index_mutations = [
            lambda value: value["documents"][0].update(prompt_count=True),
            lambda value: value["documents"][0].update(
                artifact_bytes=float(value["documents"][0]["artifact_bytes"])),
            lambda value: value["aggregate"].update(document_count=True),
            lambda value: value["aggregate"].update(prompt_token_count=True),
            lambda value: value["aggregate"].update(
                prompt_artifact_bytes=float(value["aggregate"]["prompt_artifact_bytes"])),
            lambda value: value["context"].update(artifact_token_limit=8192.0),
            lambda value: value["context"].update(inference_context_limit=2048.0),
            lambda value: value["context"].update(artifact_limit_is_inference_permission=0),
        ]
        for ordinal, mutation in enumerate(index_mutations):
            with self.subTest(index_mutation=ordinal):
                evidence, index_path, _, identity, receipt, _ = self.fake_corpus([7])
                index = json.loads(index_path.read_text())
                mutation(index)
                atomic_json(index_path, index)
                self.rebind_index(index_path)
                with self.assertRaises(EvidenceError):
                    self.validate_fake(evidence, identity, receipt)

        evidence, index_path, prompt_path, identity, receipt, _ = self.fake_corpus([7])
        prompt = json.loads(prompt_path.read_text())
        prompt["promptCount"] = True
        atomic_json(prompt_path, prompt)
        index = json.loads(index_path.read_text())
        size, digest = prompt_path.stat().st_size, sha256(prompt_path)
        index["documents"][0].update(artifact_bytes=size, artifact_sha256=digest)
        index["aggregate"]["prompt_artifact_bytes"] = size
        index["artifacts"]["prompts/prompt-0000.json"] = {"bytes": size, "sha256": digest}
        atomic_json(index_path, index)
        self.rebind_index(index_path)
        with self.assertRaises(EvidenceError):
            self.validate_fake(evidence, identity, receipt)

    def test_validate_rejects_wrong_launcher_executable_hash(self) -> None:
        evidence, _, _, identity, receipt, _ = self.fake_corpus()
        receipt["identities"]["executable"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(EvidenceError, "launcher executable identity"):
            self.validate_fake(evidence, identity, receipt)

    def test_validate_rejects_source_tokenizer_completion_and_one_token_mutations(self) -> None:
        evidence, index_path, prompt_path, identity, receipt, source = self.fake_corpus()
        index = json.loads(index_path.read_text())
        index["tokenizer_identity"] = {"foreign": True}
        atomic_json(index_path, index)
        self.rebind_index(index_path)
        with self.assertRaisesRegex(EvidenceError, "tokenizer identity"):
            self.validate_fake(evidence, identity, receipt)

        index["tokenizer_identity"] = identity
        atomic_json(index_path, index)
        self.rebind_index(index_path)
        completion = json.loads((index_path.parent / "completion.json").read_text())
        completion["prompts_sha256"] = "0" * 64
        atomic_json(index_path.parent / "completion.json", completion)
        with self.assertRaisesRegex(EvidenceError, "completion"):
            self.validate_fake(evidence, identity, receipt)

        self.rebind_index(index_path)
        prompt = json.loads(prompt_path.read_text())
        prompt["promptIDs"][0] += 1
        atomic_json(prompt_path, prompt)
        with self.assertRaises(EvidenceError):
            self.validate_fake(evidence, identity, receipt)

        authored = source["documents"][0]
        authored["text"] += " changed"
        source["manifestSHA256"] = prompt_tool.canonical_hash(source, "manifestSHA256")
        atomic_json(Path(index["source_manifest_path"]), source)
        with self.assertRaises(EvidenceError):
            self.validate_fake(evidence, identity, receipt)

    def test_validate_rejects_missing_extra_and_reordered_prompt_files(self) -> None:
        evidence, _, prompt_path, identity, receipt, _ = self.fake_corpus()
        completion_path = evidence / "corpus/completion.json"
        completion = completion_path.read_bytes()
        completion_path.unlink()
        with self.assertRaises(EvidenceError):
            self.validate_fake(evidence, identity, receipt)
        completion_path.write_bytes(completion)
        original = prompt_path.read_bytes()
        prompt_path.unlink()
        with self.assertRaises(EvidenceError):
            self.validate_fake(evidence, identity, receipt)
        prompt_path.write_bytes(original)
        (prompt_path.parent / "extra.json").write_text("{}")
        with self.assertRaisesRegex(EvidenceError, "missing, extra, or reordered"):
            self.validate_fake(evidence, identity, receipt)
        (prompt_path.parent / "extra.json").unlink()
        prompt_path.rename(prompt_path.parent / "prompt-0001.json")
        with self.assertRaisesRegex(EvidenceError, "missing, extra, or reordered"):
            self.validate_fake(evidence, identity, receipt)

    def test_validate_rejects_token_and_aggregate_quotas(self) -> None:
        evidence, _, _, identity, receipt, _ = self.fake_corpus(list(range(8193)))
        with self.assertRaisesRegex(EvidenceError, "malformed"):
            self.validate_fake(evidence, identity, receipt)
        evidence, _, _, identity, receipt, _ = self.fake_corpus([1, 2, 3])
        with mock.patch.object(prompt_tool, "AGGREGATE_TOKEN_LIMIT", 2), \
             self.assertRaisesRegex(EvidenceError, "aggregate token quota"):
            self.validate_fake(evidence, identity, receipt)

    def test_freeze_refuses_reused_and_outside_outputs_without_mutation(self) -> None:
        existing = self.root / "existing"
        existing.mkdir()
        marker = existing / "marker"
        marker.write_text("keep")
        self.assertEqual(prompt_tool.freeze(Path("/missing"), Path("/missing"), self.fixture, existing), 1)
        self.assertEqual(marker.read_text(), "keep")
        outside = Path(tempfile.gettempdir()) / f"prompt-corpus-outside-{id(self)}"
        if outside.exists():
            self.fail(f"unexpected existing test path: {outside}")
        self.assertEqual(prompt_tool.freeze(Path("/missing"), Path("/missing"), self.fixture, outside), 1)
        self.assertFalse(outside.exists())

    def test_archive_identity_requires_exact_receipt_and_artifact_hash(self) -> None:
        binary = self.root / "archive/bin/slotstream"
        binary.parent.mkdir(parents=True)
        binary.write_bytes(b"binary")
        receipt_path = binary.parent.parent / "receipt.json"
        receipt_path.write_text("{}")
        identity = {
            "path": str(binary), "binary_sha256": "a" * 64,
            "metallib_sha256": "b" * 64, "build_identity_sha256": "c" * 64,
            "source_archive_sha256": "d" * 64, "historical": True,
            "archive_receipt_path": str(receipt_path),
            "archive_receipt_sha256": sha256(receipt_path),
        }
        index = {"executable_identity": identity}
        build = {"binary_sha256": "a" * 64, "metallib_sha256": "b" * 64}
        identity_file = self.root / "identity"
        archive_file = self.root / "source.tar"
        identity_file.write_bytes(b"identity")
        archive_file.write_bytes(b"source")
        identity["build_identity_sha256"] = sha256(identity_file)
        identity["source_archive_sha256"] = sha256(archive_file)
        archive_receipt = {"artifacts": {"bin/slotstream": {"sha256": sha256(binary)}}}
        with mock.patch.object(prompt_tool, "validate_build_identity",
                               return_value=(build, {"identity": identity_file,
                                                    "source_archive": archive_file})), \
             mock.patch.object(prompt_tool, "validate_receipt_file",
                               return_value=archive_receipt):
            prompt_tool._validate_executable(index, binary)
            archive_receipt["artifacts"]["bin/slotstream"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(EvidenceError, "does not bind"):
                prompt_tool._validate_executable(index, binary)

    def test_output_schemas_cover_all_required_properties(self) -> None:
        for name in ("prompt-source-v1.json", "prompt-artifact-v1.json", "prompt-corpus-v1.json"):
            schema = json.loads((HERE / "schemas" / name).read_text())
            self.assertLessEqual(set(schema["required"]), set(schema["properties"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
