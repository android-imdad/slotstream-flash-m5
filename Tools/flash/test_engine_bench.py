import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from common import EvidenceError, atomic_json, sha256
import engine_bench


class EngineBenchmarkTests(unittest.TestCase):
    def test_baseline_hash_and_exact_workload_controls_are_required(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            path = root / "baseline.json"
            request = {"messages": [{"role": "user", "content": "fixture"}], "think": False,
                       "options": {"temperature": 0, "seed": 42, "num_predict": 128}}
            baseline = {"memory_target_decimal_gb": 24, "max_context_tokens": 32768,
                "max_output_tokens": 128, "prefix_cache_enabled": False,
                "requests": {name: copy.deepcopy(request) for name in ("explanation", "coding", "reasoning")}}
            def save(value):
                atomic_json(path, value)
                atomic_json(root / "SHA256SUMS.json", {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}})
            save(baseline)
            self.assertEqual(engine_bench.load_baseline(path), baseline)
            bad = copy.deepcopy(baseline)
            bad["memory_target_decimal_gb"] = 14
            save(bad)
            with self.assertRaises(EvidenceError): engine_bench.load_baseline(path)
            bad = copy.deepcopy(baseline)
            bad["requests"]["coding"]["options"]["seed"] = 7
            save(bad)
            with self.assertRaises(EvidenceError): engine_bench.load_baseline(path)
            save(baseline)
            path.write_text(path.read_text() + " ")
            with self.assertRaises(EvidenceError): engine_bench.load_baseline(path)

    def test_contaminated_timing_is_retained_without_a_performance_admission(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            stats = {"plan": {"target_gb": 24, "max_context_tokens": 32768, "runtime_prefix_cache_enabled": False},
                "effective_mtp": False, "effective_expert_widening": "packed4-to6", "load_seconds": 1,
                "sampling": {"greedy": True, "seed": "42", "requested_max_tokens": "128"},
                "stats": {"decodeTokens": 128, "decodeSeconds": 16, "firstTokenSeconds": 1,
                          "firstTextSeconds": 1, "prefillSeconds": 1, "requestSeconds": 17}}
            atomic_json(root / "stats.json", stats)
            receipt = {"result": {"functional_success": True}, "memory": {"peak_bytes": 20_000_000_000},
                       "environment": {"SLOTSTREAM_PREFIX_CACHE": "0"}}
            with mock.patch.object(engine_bench.benchmark, "settle_before_model_launch", return_value={}), \
                 mock.patch.object(engine_bench, "wait_for_nominal", return_value={"elapsed_seconds": 30}), \
                 mock.patch.object(engine_bench.benchmark, "launch", return_value=0), \
                 mock.patch.object(engine_bench, "validate_receipt_file", return_value=receipt), \
                 mock.patch.object(engine_bench, "require_terminal_sampling"), \
                 mock.patch.object(engine_bench.widen_study, "validate_timing_eligibility", side_effect=EvidenceError("paging")):
                _, _, result = engine_bench.arm(Path("binary"), Path("model"), "fixture", "packed4-to6", root)
                self.assertFalse(result["timing_eligible"])
                self.assertEqual(result["decode_tokens_per_second"], 8)
                self.assertEqual(result["ineligible_reason"], "paging")

    def test_cooldown_resets_nominal_window_after_a_fair_observation(self):
        clock = [0]
        def sleep(seconds): clock[0] += seconds
        def reader():
            ready = clock[0] >= 2 and clock[0] != 4
            return {"conditions": {"thermalState": "nominal" if ready else "fair", "lowPowerModeEnabled": False}}
        result = engine_bench.wait_for_nominal(stable_seconds=3, max_seconds=15, interval=1,
                    reader=reader, clock=lambda: clock[0], sleep=sleep)
        self.assertEqual(result["elapsed_seconds"], 8)

    def test_cooldown_never_treats_timeout_or_low_power_as_readiness(self):
        clock = [0]
        def sleep(seconds): clock[0] += seconds
        with self.assertRaisesRegex(EvidenceError, "did not reach"):
            engine_bench.wait_for_nominal(stable_seconds=2, max_seconds=3, interval=1,
                reader=lambda: {"conditions": {"thermalState": "nominal", "lowPowerModeEnabled": True}},
                clock=lambda: clock[0], sleep=sleep)


if __name__ == "__main__":
    unittest.main()
