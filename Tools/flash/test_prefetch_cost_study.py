import copy
import unittest
import tempfile
from pathlib import Path

from common import EvidenceError
from replay import TraceGroup
from prefetch_cost_study import curves, cost_replay, timing_screen, disposition, same_geometry, METHODS, SIZES, EXPERTS


class PrefetchCostTests(unittest.TestCase):
    def test_relocated_evidence_paths_allow_only_same_existing_files(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "real").mkdir()
            (root / "alias").symlink_to(root / "real", target_is_directory=True)
            for name in ("summary", "verify"): (root / "real" / name).write_text("fixture")
            left = {"verification": {"summary_path": str(root / "real/summary"),
                                     "verification_output": str(root / "real/verify")}, "hash": "fixed"}
            right = copy.deepcopy(left)
            right["verification"] = {k: v.replace("/real/", "/alias/") for k, v in right["verification"].items()}
            self.assertTrue(same_geometry(left, right))
            right["hash"] = "changed"
            self.assertFalse(same_geometry(left, right))
            right["hash"] = "fixed"
            right["verification"]["summary_path"] = str(root / "real/verify")
            self.assertFalse(same_geometry(left, right))

    def group(self, experts, tokens=1):
        return TraceGroup(tokens, tuple(tuple(tuple(experts) for _ in range(tokens)) for _ in range(48)))

    def table(self):
        return {100: {n: {"median": float(n), "min": n * .9, "max": n * 1.1} for n in range(11)}}

    def test_wrong_predictions_never_gain_from_unrelated_timing_noise(self):
        groups = [self.group(range(10), 2)] + [self.group(range(i * 10, i * 10 + 10)) for i in range(1, 5)]
        costs = cost_replay(groups, 40, [100] * 48, self.table(), "previous-token")
        self.assertEqual(costs["byte_screen"]["useful_bytes"], 0)
        self.assertEqual(costs["remaining_demand_cost"], costs["charged_demand_cost"])
        self.assertEqual(costs["optimistic_remaining_cost"], costs["charged_demand_cost"])
        self.assertGreater(costs["prefetch_reader_cost"], 0)
        self.assertLessEqual(timing_screen(costs, 100, 50)["free_prefetch_improvement_fraction"], 0)

    def test_repeated_causal_predictions_reduce_remaining_demand_and_charge_cache(self):
        groups = [self.group(range(10), 2)] + [self.group(range(10, 20)) for _ in range(4)]
        costs = cost_replay(groups, 40, [100] * 48, self.table(), "previous-token")
        self.assertLess(costs["candidate_slots"], 40)
        self.assertGreater(costs["byte_screen"]["useful_bytes"], 0)
        self.assertLess(costs["remaining_demand_cost"], costs["charged_demand_cost"])
        self.assertLessEqual(costs["optimistic_remaining_cost"], costs["remaining_demand_cost"])

    def test_all_nonreader_time_is_shared_once_not_per_layer(self):
        costs = {"control_reader_cost": 10, "remaining_demand_cost": 8,
                 "optimistic_remaining_cost": 7, "prefetch_reader_cost": 12}
        result = timing_screen(costs, 20, 10)
        self.assertEqual(result["non_reader_overlap_budget_seconds"], 10)
        self.assertEqual(result["estimated_exposed_prefetch_seconds"], 2)
        self.assertAlmostEqual(result["free_prefetch_improvement_fraction"], .1)
        self.assertAlmostEqual(result["aggregate_overlap_improvement_fraction"], 0)
        with self.assertRaises(EvidenceError): timing_screen(costs, 9, 10)
        bad = dict(costs, prefetch_reader_cost=float("nan"))
        with self.assertRaises(EvidenceError): timing_screen(bad, 20, 10)

    def fixture(self):
        report = {"format": "slotstream-prefetch-reader-costs-v1", "policy": "packed4-to6", "queueDepth": 32,
            "experts": EXPERTS, "uniqueSourceBytes": sum(SIZES.values()) * 10, "check": {"passed": True},
            "readControls": [{"noCacheReturnCode": 0, "readAheadReturnCode": 0}],
            "conditionsBefore": {"thermalState": "nominal", "lowPowerModeEnabled": False},
            "conditionsAfter": {"thermalState": "nominal", "lowPowerModeEnabled": False},
            "cases": [{"layer": layer, "count": n, "sourceBytes": size * n,
                       "seconds": [n * .001] * 8, "outputSHA256": "a" * 64}
                      for layer, size in SIZES.items() for n in range(1, 11)]}
        receipt = {"memory": {"qualified": True, "peak_bytes": 100_000_000},
            "vm": {"before": {"swapins": 0, "swapouts": 0}, "after": {"swapins": 0, "swapouts": 0}}}
        return report, receipt

    def test_cost_grid_needs_every_count_and_precision_and_clean_memory_timing(self):
        report, receipt = self.fixture()
        self.assertEqual(len(curves(report, receipt)[SIZES[0]]), 11)
        for mutate in (lambda r: r["cases"].pop(),
                       lambda r: r["cases"].append(copy.deepcopy(r["cases"][0])),
                       lambda r: r["cases"][0]["seconds"].pop(),
                       lambda r: r["cases"][0]["seconds"].__setitem__(0, float("nan")),
                       lambda r: r["conditionsAfter"].__setitem__("thermalState", "fair"),
                       lambda r: r["cases"][0].__setitem__("sourceBytes", 1)):
            bad = copy.deepcopy(report); mutate(bad)
            with self.assertRaises(EvidenceError): curves(bad, receipt)
        receipt["vm"]["after"]["swapins"] = 1
        with self.assertRaises(EvidenceError): curves(report, receipt)

    def test_admission_keeps_inconclusive_separate_and_never_claims_native_speed(self):
        def rows(central, optimistic):
            return [{"prompt_id": str(n), "methods": {m: {"timing": {
                "aggregate_overlap_improvement_fraction": central,
                "optimistic_sample_range_fraction": optimistic}} for m in METHODS}} for n in range(3)]
        for central, optimistic, expected in ((.01, .05, "rejected_by_empirical_screen"),
                                              (.01, .15, "inconclusive_cost_spread"),
                                              (.15, .20, "promising_for_native_overlap_experiment")):
            result = disposition(rows(central, optimistic))
            for item in result.values():
                self.assertEqual(item["disposition"], expected)
                self.assertFalse(item["inference_speedup_qualified"])
                self.assertFalse(item["native_prefetch_implemented"])
        with self.assertRaises(EvidenceError): disposition(rows(.1, .2)[:-1])
