#!/usr/bin/env python3
from __future__ import annotations

import json
import io
from pathlib import Path
import struct
import sys
import tarfile
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from cache_study import (PINNED_REVISION, byte_comparison, decision, derive_source_geometry, load_fixture,
                         analyze, validate_artifact_hashes, validate_cohort_shape, validate_collection,
                         validate_pair_documents, validate_stats)
from common import (EvidenceError, FLASH_ROOT, RECEIPT_FORMAT, atomic_json, harness_hashes,
                    sha256)
from replay import LAYERS, TOP_K, TraceGroup, encode_trace, replay


class CacheStudyTests(unittest.TestCase):
    def setUp(self):
        FLASH_ROOT.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="cache-study-test-", dir=FLASH_ROOT)
        self.root = Path(self.temp.name)

    def tearDown(self): self.temp.cleanup()

    def fake_model(self, bad_dtype=False):
        model = self.root / "model"; model.mkdir()
        bit_map = {"default": {"bits": 6, "group_size": 64},
                   "language_model.layers.0.mlp.switch_mlp": {"bits": 4, "group_size": 64}}
        config = {"text_config": {"hidden_size": 2560, "moe_intermediate_size": 640,
                                  "num_hidden_layers": 48, "num_experts": 512,
                                  "num_experts_per_tok": 10},
                  "jang_config": {"format": "jang_v2", "bit_map": bit_map}}
        (model / "config.json").write_text(json.dumps(config))
        header, weight_map, offset = {}, {}, 0
        filename = "model-00001-of-00001.safetensors"
        for layer in range(48):
            bits = 4 if layer == 0 else 6
            for projection in ("gate_proj", "up_proj", "down_proj"):
                rows, columns = (640, 2560) if projection != "down_proj" else (2560, 640)
                for member in ("weight", "scales", "biases"):
                    name = f"language_model.layers.{layer}.mlp.switch_mlp.{projection}.{member}"
                    shape = [512, rows, columns * bits // 32] if member == "weight" else [512, rows, columns // 64]
                    dtype = "U32" if member == "weight" else "F16"
                    if bad_dtype and layer == 0 and projection == "gate_proj" and member == "scales": dtype = "BF16"
                    size = 1
                    for value in shape: size *= value
                    size *= 4 if member == "weight" else 2
                    header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
                    weight_map[name] = filename; offset += size
        payload = json.dumps(header, separators=(",", ":")).encode()
        path = model / filename
        with path.open("wb") as stream:
            stream.write(struct.pack("<Q", len(payload))); stream.write(payload)
            stream.seek(8 + len(payload) + offset - 1); stream.write(b"\0")
        (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
        return model

    def test_frozen_fixture_hash_and_prompt_bounds(self):
        fixture = load_fixture()
        self.assertEqual(len(fixture["prompts"]), 3)
        self.assertTrue(fixture["development_only"])

    def test_header_geometry_derives_varying_source_bytes(self):
        geometry = derive_source_geometry(self.fake_model())
        self.assertEqual(geometry["layer_source_record_bytes"][0], 2_764_800)
        self.assertEqual(geometry["layer_source_record_bytes"][1], 3_993_600)
        self.assertEqual(len(geometry["tensor_rows"]), 48 * 9)

    def test_unsupported_header_dtype_is_rejected(self):
        with self.assertRaisesRegex(EvidenceError, "dtype"):
            derive_source_geometry(self.fake_model(bad_dtype=True))

    def test_geometry_compatible_fake_cannot_claim_pinned_verification(self):
        with self.assertRaises(EvidenceError):
            derive_source_geometry(self.fake_model(), verification={"revision": PINNED_REVISION})

    def test_partial_stats_and_omitted_trace_are_hard_failures(self):
        prompt = load_fixture()["prompts"][0]
        with self.assertRaises(EvidenceError): validate_stats({"schema_version": 1, "stats": {}}, prompt)
        from replay import parse_trace
        with self.assertRaisesRegex(EvidenceError, "missing"):
            parse_trace(self.root / "omitted.bin")

    def test_changed_pair_output_ids_are_rejected(self):
        with self.assertRaises(EvidenceError):
            validate_pair_documents({"prompt_ids": [1], "output_ids": [2]},
                                    {"prompt_ids": [1], "output_ids": [3]})

    def test_mutated_untraced_cache_policy_is_rejected(self):
        base = {"prompt_ids": [1], "output_ids": [2], "effective_pool_slots": 3,
                "effective_mtp": False, "effective_prefill_chunk": 256, "plan": {},
                "optimizations": {"sparsePoolPins": False}, "stats": {"prefillRecords": 1,
                    "decodeRecords": 1, "prefillReadBytes": 2, "decodeReadBytes": 2,
                    "decodeForwardPasses": 1, "finishReason": "stop"}}
        changed = json.loads(json.dumps(base)); changed["optimizations"]["sparsePoolPins"] = True
        with self.assertRaisesRegex(EvidenceError, "effective configuration"):
            validate_pair_documents(changed, base)

    def test_pair_allows_availability_drift_but_rejects_resolved_plan_change(self):
        common = {"prompt_ids": [1], "output_ids": [2], "effective_pool_slots": 793,
                  "effective_mtp": False, "effective_prefill_chunk": 256, "optimizations": {},
                  "stats": {key: value for key, value in (("prefillRecords", 1), ("decodeRecords", 2),
                      ("prefillReadBytes", 3), ("decodeReadBytes", 4), ("decodeForwardPasses", 1),
                      ("finishReason", "stop"))}}
        left = json.loads(json.dumps(common)); right = json.loads(json.dumps(common))
        left["plan"] = {"target_gb": 14, "pool_slots": 793, "checkpoint_format": "jang6S",
                        "memory_ledger": {"pool_bytes": 1}, "device_available_gb": 25.6}
        right["plan"] = {"target_gb": 14, "pool_slots": 793, "checkpoint_format": "jang6S",
                         "memory_ledger": {"pool_bytes": 1}, "device_available_gb": 22.1}
        validate_pair_documents(left, right)
        right["plan"]["memory_ledger"]["pool_bytes"] = 2
        with self.assertRaisesRegex(EvidenceError, "resolved allocation"):
            validate_pair_documents(left, right)

    def test_decision_requires_pooled_and_per_prompt_thresholds(self):
        rejected = {1: {"pooled_decode_savings_fraction": 0.09,
                        "max_prompt_decode_increase_fraction": 0.0}}
        self.assertEqual(decision(rejected)["disposition"], "rejected")
        guarded = {2: {"pooled_decode_savings_fraction": 0.12,
                       "max_prompt_decode_increase_fraction": 0.051}}
        self.assertEqual(decision(guarded)["disposition"], "rejected")
        accepted = {4: {"pooled_decode_savings_fraction": 0.10,
                        "max_prompt_decode_increase_fraction": 0.05}}
        self.assertEqual(decision(accepted), {"disposition": "candidate", "selected_window": 4,
            "reason": "development threshold passed; a separate runtime implementation plan is still required"})

    def test_missing_cohort_and_receipt_artifact_are_rejected(self):
        fixture = load_fixture()
        pairs = [{"prompt_id": prompt["id"], "prompt_tokens": 2, "output_tokens": 1,
                  "arms": {arm: {"path": f"{prompt['id']}/{arm}",
                                  "artifacts": {name: "x" for name in
                                      ({"receipt.json", "stats.json", "environment.json", "settling.json", "router-trace.bin"}
                                       if arm == "traced" else {"receipt.json", "stats.json", "environment.json", "settling.json"})}}
                           for arm in ("untraced", "traced")}} for prompt in fixture["prompts"]]
        validate_cohort_shape(pairs, fixture)
        with self.assertRaises(EvidenceError): validate_cohort_shape(pairs[:-1], fixture)
        del pairs[0]["arms"]["traced"]["artifacts"]["receipt.json"]
        with self.assertRaises(EvidenceError): validate_cohort_shape(pairs, fixture)

    def test_missing_physical_artifact_is_rejected(self):
        with self.assertRaisesRegex(EvidenceError, "missing"):
            validate_artifact_hashes(self.root, {"receipt.json": "0" * 64})

    def test_zero_control_positive_candidate_is_regression(self):
        self.assertEqual(byte_comparison(0, 0), (0.0, 0.0))
        self.assertEqual(byte_comparison(0, 1), (-1.0, 1.0))

    def _completion(self, directory, receipt):
        atomic_json(directory / "completion.json", {"format": "slotstream-flash-completion-v1",
                    "receipt_sha256": sha256(directory / "receipt.json"),
                    "qualified": receipt["qualified"], "functional_success": True})

    def _archive(self, collection):
        archive = collection / "archive"; binary_dir = archive / "bin"; binary_dir.mkdir(parents=True)
        source_data = b"package"
        source_hash = __import__("hashlib").sha256(source_data).hexdigest()
        (binary_dir / "slotstream").write_bytes(b"binary")
        (binary_dir / "mlx.metallib").write_bytes(b"metal")
        atomic_json(binary_dir / "build-source-before.json", {"Package.swift": source_hash})
        with tarfile.open(binary_dir / "build-source.tar.gz", "w:gz") as archive_file:
            info = tarfile.TarInfo("Package.swift"); info.size = len(source_data)
            archive_file.addfile(info, io.BytesIO(source_data))
        identity = {"source": {"Package.swift": source_hash},
                    "source_archive_sha256": sha256(binary_dir / "build-source.tar.gz"),
                    "binary_sha256": sha256(binary_dir / "slotstream"),
                    "metallib_sha256": sha256(binary_dir / "mlx.metallib")}
        atomic_json(binary_dir / "build-identity.json", identity)
        artifacts = {str(path.relative_to(archive)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
                     for path in binary_dir.iterdir()}
        receipt = {"format": RECEIPT_FORMAT, "schema_version": 1, "kind": "archive",
                   "qualified": True, "qualification_reasons": [], "command": ["archive"],
                   "environment": {}, "started_at_unix": 1.0, "ended_at_unix": 2.0,
                   "duration_seconds": 1.0, "process": {"pid": 1},
                   "memory": {"qualified": False}, "vm": {}, "result": {"exit_code": 0},
                   "artifacts": artifacts, "identities": {"build_identity": identity}}
        atomic_json(archive / "receipt.json", receipt); self._completion(archive, receipt)
        return archive, binary_dir / "slotstream", identity

    def _launch_receipt(self, directory, binary, stats, trace_path, environment):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "stdout.txt").write_text("ok")
        (directory / "stderr.txt").write_text("")
        sample = {"start_abstime": 1, "physical_footprint_bytes": 1,
                  "lifetime_peak_bytes": 1, "resident_bytes": 1, "elapsed_seconds": 0.0}
        (directory / "memory.jsonl").write_text(json.dumps(sample) + "\n")
        atomic_json(directory / "stats.json", stats)
        atomic_json(directory / "environment.json", environment)
        atomic_json(directory / "settling.json", {"policy": "fixed-before-full-model-launch",
                    "requested_seconds": 2.0, "observed_elapsed_seconds": 2.0})
        names = ["stdout.txt", "stderr.txt", "memory.jsonl", "stats.json"]
        if trace_path is not None: names.append("router-trace.bin")
        prompt = stats["fixture_prompt"]
        command = [str(binary), "run", "--model", stats["fixture_model"], "--memory-gb", "14",
                   "--max-context", "2048", "--mtp", "off", "--vision", "off", "--prompt", prompt,
                   "--max-tokens", "128", "--greedy", "--seed", "7", "--sample-footprint",
                   "--stats-json", str(directory / "stats.json")]
        stored = dict(stats); stored.pop("fixture_prompt"); stored.pop("fixture_model")
        atomic_json(directory / "stats.json", stored)
        receipt = {"format": RECEIPT_FORMAT, "schema_version": 1, "kind": "launch",
                   "qualified": False, "qualification_reasons": ["identities unavailable"],
                   "command": command, "environment": environment, "started_at_unix": 1.0,
                   "ended_at_unix": 2.0, "duration_seconds": 1.0,
                   "process": {"pid": 1, "process_group": 1, "start_identity": 1},
                   "memory": {"target_gb_decimal": 14, "sample_interval_seconds": 0.5,
                              "qualified": True, "peak_bytes": 1, "sample_count": 1,
                              "samples_artifact": "memory.jsonl", "sampler_error": None},
                   "vm": {}, "result": {"exit_code": 0, "functional_success": True,
                                         "timed_out": False, "interrupted": False,
                                         "budget_exceeded": False, "evidence_failure": False},
                   "artifacts": {name: {"bytes": (directory / name).stat().st_size,
                                        "sha256": sha256(directory / name)} for name in names},
                   "identities": {"run_set_id": None, "model_hash": None,
                                  "harness_hashes": harness_hashes(),
                                  "identity_status": "unverified",
                                  "executable": {"path": str(binary), "bytes": binary.stat().st_size,
                                                 "sha256": sha256(binary)},
                                  "missing_reason": "test"}}
        atomic_json(directory / "receipt.json", receipt); self._completion(directory, receipt)

    def _complete_collection(self):
        collection = self.root / "collection"; collection.mkdir()
        archive, binary, identity = self._archive(collection)
        fixture = load_fixture(); model = self.root / "synthetic-model"; model.mkdir()
        source = {"format": "slotstream-expert-source-geometry-v1", "model_path": str(model),
                  "revision": "test", "verification": {"revision": "test"},
                  "config_sha256": "c", "index_sha256": "i", "file_headers": {},
                  "small_file_identities": {}, "layer_source_record_bytes": [100] * LAYERS,
                  "tensor_rows": {}, "cache_record_bytes": 4_300_800}
        rows = tuple(tuple([0] * TOP_K) for _ in range(2))
        prefill = TraceGroup(2, tuple(rows for _ in range(LAYERS)))
        decode_rows = tuple(([tuple([0] * TOP_K)]) for _ in range(LAYERS))
        decode = TraceGroup(1, tuple(tuple(value) for value in decode_rows))
        groups = [prefill, decode]; native = replay(groups, 48, source["layer_source_record_bytes"])
        pairs = []
        for prompt in fixture["prompts"]:
            arms = {}
            for arm in ("untraced", "traced"):
                directory = collection / "prompts" / prompt["id"] / arm
                directory.mkdir(parents=True)
                trace = directory / "router-trace.bin" if arm == "traced" else None
                if trace: trace.write_bytes(encode_trace(groups))
                environment = {"LANG": "C.UTF-8"}
                if trace: environment["SLOTSTREAM_ROUTER_TRACE"] = str(trace)
                stats = {"schema_version": 1, "fixture_prompt": prompt["text"], "fixture_model": str(model),
                         "effective_mtp": False, "effective_pool_slots": 48, "effective_prefill_chunk": 256,
                         "experimental_memory_family": False, "prompt_ids": [1, 2], "output_ids": [3],
                         "text": "ok", "load_seconds": 1.0, "launch_seconds": 2.0,
                         "sampling": {"greedy": True, "requested_max_tokens": "128", "seed": "7"},
                         "plan": {"checkpoint_format": "jang6S", "target_gb": 14,
                                  "max_context_tokens": 2048, "mtp": False, "vision": False},
                         "optimizations": {"readScopeTokens": 0, "layerLocalFloorCache": False,
                                           "layerExpertWorkspace": False, "sparsePoolPins": False,
                                           "skipUnusedFinalForward": True},
                         "stats": {"prefillRecords": native["prefill"]["misses"],
                                   "decodeRecords": native["decode"]["misses"],
                                   "prefillReadBytes": native["prefill"]["miss_bytes"],
                                   "decodeReadBytes": native["decode"]["miss_bytes"],
                                   "decodeForwardPasses": 1, "decodeTokens": 1, "finishReason": "stop",
                                   "promptTokens": 2, "prefillTokens": 2, "smallPrefillSweeps": 0,
                                   "abortedReadScopes": 0, "reusedPrefixTokens": 0,
                                   "contextArithmetic": "standard", "verifyPasses": 0,
                                   "memoryPressureCancelled": False,
                                   "sampledFootprint": {"peakBytes": 1, "samples": 1}}}
                self._launch_receipt(directory, binary, stats, trace, environment)
                names = {"receipt.json", "stats.json", "environment.json", "settling.json"}
                if trace: names.add("router-trace.bin")
                arms[arm] = {"path": str(directory.relative_to(collection)),
                             "artifacts": {name: sha256(directory / name) for name in names}}
            pairs.append({"prompt_id": prompt["id"], "arms": arms, "prompt_tokens": 2, "output_tokens": 1})
        manifest = {"format": "slotstream-cache-collection-v1", "schema_version": 1,
                    "complete": True, "qualification": False,
                    "fixture_sha256": sha256(Path(__file__).resolve().parents[1] / "fixtures/flash/cache-study.json"),
                    "binary": {"path": str(binary), "sha256": sha256(binary),
                               "build_identity": identity, "archive_receipt": str(archive / "receipt.json"),
                               "archive_receipt_sha256": sha256(archive / "receipt.json")},
                    "model": source, "pairs": pairs, "harness_hashes": harness_hashes(),
                    "model_settling_policy": {"kind": "fixed-before-full-model-launch", "seconds": 2.0}}
        atomic_json(collection / "collection.json", manifest)
        atomic_json(collection / "completion.json", {"format": "slotstream-cache-collection-completion-v1",
                    "collection_sha256": sha256(collection / "collection.json"), "complete": True,
                    "qualification": False})
        return collection, manifest, source

    def test_complete_collection_validates_and_analyzes_relative_and_absolute(self):
        collection, manifest, source = self._complete_collection()
        verify = lambda model: {"revision": "test"}
        geometry = lambda model, verification: source
        validate_collection(collection, manifest, verification_provider=verify, geometry_provider=geometry)
        relative = collection.relative_to(Path.cwd())
        self.assertEqual(analyze(relative, self.root / "analysis-relative",
                                 verification_provider=verify, geometry_provider=geometry), 0)
        self.assertEqual(analyze(collection, self.root / "analysis-absolute",
                                 verification_provider=verify, geometry_provider=geometry), 0)
        self.assertEqual(json.loads((self.root / "analysis-relative/report.json").read_text())["decision"]["disposition"], "rejected")

    def test_missing_collection_completion_is_rejected(self):
        collection, manifest, source = self._complete_collection()
        (collection / "completion.json").unlink()
        with self.assertRaises(EvidenceError):
            validate_collection(collection, manifest, verification_provider=lambda model: {"revision": "test"},
                                geometry_provider=lambda model, verification: source)


if __name__ == "__main__": unittest.main(verbosity=2)
