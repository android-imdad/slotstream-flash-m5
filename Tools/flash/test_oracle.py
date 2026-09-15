import copy
import json
from pathlib import Path
import tempfile
import unittest

from common import EvidenceError, atomic_json, sha256
import calibrate
import capture
import oracle
import evaluate


class OracleTests(unittest.TestCase):
    def fixture(self, root, retained=2):
        norm = root / "norms"
        norm.mkdir()
        atomic_json(norm / "manifest.json", {})
        atomic_json(root / "suite.json", {})
        atomic_json(root / "manifest.json", {})
        atomic_json(root / "study.json", {"format": "slotstream-neuron-study-v1", "algorithm": calibrate.ALGORITHM,
            "split": "development", "retained_order": list(calibrate.RETAINED),
            "norm_manifest_sha256": sha256(norm / "manifest.json"), "suite_sha256": sha256(root / "suite.json")})
        config = oracle.configuration(norm, root / "study.json", root / "suite.json", root / "manifest.json", retained)
        atomic_json(root / "config.json", config)
        return config

    def test_config_rejects_unknown_fields_split_unreserved_and_stale_assets(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            config = self.fixture(root)
            oracle.validate_configuration(root / "config.json", root / "manifest.json")
            for key, value in (("retained_blocks", True), ("retained_blocks", 3), ("split", "qualification"),
                               ("diagnostic_reserved_bytes", 128 << 20), ("extra", 1)):
                bad = {**config, key: value}
                atomic_json(root / "config.json", bad)
                with self.assertRaises(EvidenceError):
                    oracle.validate_configuration(root / "config.json", root / "manifest.json")
            atomic_json(root / "config.json", config)
            atomic_json(root / "study.json", {})
            with self.assertRaisesRegex(EvidenceError, "asset changed"):
                oracle.validate_configuration(root / "config.json", root / "manifest.json")

    def mask_fixture(self, root, config):
        rows = [{"layer": layer, "router_ranks": list(range(10)), "expert_ids": list(range(10)),
                 "blocks": [[True, True] + [False] * 8 for _ in range(10)]} for layer in range(48)]
        path = root / "oracle-mask-000.json"
        atomic_json(path, {"placeholder": 1})
        path.write_text(json.dumps(rows))
        metadata = {"name": path.name, "document_id": "doc", "token_position": 0,
                    "bytes": path.stat().st_size, "sha256": sha256(path)}
        report = {"documents": [{"id": "doc", "positions": [{"position": 0,
                    "routes": [{"ids": list(range(10))} for _ in range(48)]}]}],
            "oracle": {"format": "slotstream-oracle-evidence-v1", "config_sha256": "x",
                "study_sha256": config["study_sha256"], "suite_sha256": config["suite_sha256"],
                "norm_manifest_sha256": config["norm_manifest_sha256"], "algorithm": config["algorithm"],
                "retained_blocks": 2, "diagnostic_reserved_bytes": oracle.RESERVATION,
                "norm_payload_bytes": 62914560, "warmup_policy": "dense-whole-document",
                "mask_failure_state_reuse_refused": True, "mask_files": [metadata]}}
        return report, rows, path

    def test_masks_require_complete_router_joins_boolean_counts_and_actual_bytes(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            config = self.fixture(root)
            report, rows, path = self.mask_fixture(root, config)
            oracle.validate_masks(report, root, config, verify_scores=False)
            for change in ("bool", "count", "rank", "rank-bool", "expert", "layer", "missing"):
                bad = copy.deepcopy(rows)
                if change == "bool": bad[0]["blocks"][0][0] = 1
                elif change == "count": bad[0]["blocks"][0][2] = True
                elif change == "rank": bad[0]["router_ranks"][0] = 1
                elif change == "rank-bool": bad[0]["router_ranks"][0] = False
                elif change == "expert": bad[0]["expert_ids"][0] = 11
                elif change == "layer": bad[0]["layer"] = 1
                elif change == "missing": bad.pop()
                path.write_text(json.dumps(bad))
                report["oracle"]["mask_files"][0].update(bytes=path.stat().st_size, sha256=sha256(path))
                with self.assertRaises(EvidenceError):
                    oracle.validate_masks(report, root, config, verify_scores=False)
            path.write_text("[]")
            with self.assertRaisesRegex(EvidenceError, "bytes changed"):
                oracle.validate_masks(report, root, config, verify_scores=False)

    def test_reference_validator_cannot_admit_oracle_payload(self):
        with self.assertRaisesRegex(EvidenceError, "reference validator"):
            capture.validate_native_report({}, None, None, "oracle", None, None, {})

    def controls(self):
        ledger = {"diagnostic_reserved_bytes": 128 << 20, "pool_bytes": 854 * oracle.RECORD_BYTES,
                  "expected_peak_bytes": 12998997528, "fixed_bytes": 8859096600}
        plan = {"memory_ledger": ledger, "target_gb": 14, "runtime_prefix_cache_enabled": False,
                "mtp": False, "vision": False, "pool_slots": 854, "pool_gb": 3.7,
                "experts_per_layer_cached": 18, "expected_peak_gb": 13, "device_available_gb": 23}
        reference = {"plan": plan, "memory_ledger": ledger, "optimizations": {}, "numerical_environment": {},
                     "source_identity": {k: k for k in ("model_config_sha256", "model_index_sha256", "metallib_sha256")}}
        candidate = copy.deepcopy(reference)
        new = candidate["memory_ledger"]
        new["diagnostic_reserved_bytes"] = oracle.RESERVATION
        new["pool_bytes"] = 834 * oracle.RECORD_BYTES
        new["expected_peak_bytes"] += oracle.EXTRA - 20 * oracle.RECORD_BYTES
        candidate["plan"]["pool_slots"] = 834
        candidate["plan"]["device_available_gb"] = 25
        return reference, candidate

    def test_control_comparison_only_allows_charged_pool_difference(self):
        reference, candidate = self.controls()
        oracle.compare_controls(reference, candidate)
        for mutate in ("pool", "ledger", "numerical"):
            bad = copy.deepcopy(candidate)
            if mutate == "pool": bad["plan"]["pool_slots"] += 1
            elif mutate == "ledger": bad["memory_ledger"]["fixed_bytes"] -= 1
            else: bad["numerical_environment"]["effective_tf32"] = False
            with self.assertRaises(EvidenceError):
                oracle.compare_controls(reference, bad)

    def test_quality_requires_each_category_and_original_thresholds(self):
        limit = {"meanKL": .001, "p99KL": .01, "top1Agreement": .99, "pplRatio": 1.01}
        good = {"meanKL": 0, "p99KL": 0, "top1Agreement": 1, "pplRatio": 1}
        self.assertTrue(evaluate.quality_pass(good, {"prose": good}, limit))
        for field, bad_value in (("meanKL", .002), ("p99KL", .02), ("top1Agreement", .98), ("pplRatio", 1.02)):
            self.assertFalse(evaluate.quality_pass(good, {"code": {**good, field: bad_value}}, limit))

    def test_state_structure_allows_numerical_changes_but_not_offsets(self):
        a = {"fields": [{"name": "key.3", "present": True, "shape": [1, 1, 2], "sha256": "a"}], "indexerBases": {"3": 7}}
        b = copy.deepcopy(a)
        b["fields"][0]["sha256"] = "b"
        self.assertEqual(oracle.state_structure(a), oracle.state_structure(b))
        b["indexerBases"]["3"] = 8
        self.assertNotEqual(oracle.state_structure(a), oracle.state_structure(b))


if __name__ == "__main__":
    unittest.main()
