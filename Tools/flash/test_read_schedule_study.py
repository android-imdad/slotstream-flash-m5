import copy
import unittest

from common import EvidenceError
from read_schedule_study import decision


class ReadScheduleTests(unittest.TestCase):
    def fixture(self):
        report = {"format": "slotstream-read-scheduling-component-v1", "check": {"passed": True},
            "sourceRegionBytes": 100_000_000,
            "readControls": [{"noCacheReturnCode": 0, "readAheadReturnCode": 0}],
            "conditionsBefore": {"thermalState": "nominal", "lowPowerModeEnabled": False},
            "conditionsAfter": {"thermalState": "nominal", "lowPowerModeEnabled": False},
            "cases": [{"layer": layer, "queueDepth": depth,
                "experts": [0, 7, 19, 43, 79, 131, 211, 307, 401, 511],
                "orders": [["strided", "balanced"] if pair % 2 == 0 else ["balanced", "strided"] for pair in range(8)],
                "stridedSeconds": [1.] * 8, "balancedSeconds": [.8] * 8, "outputSHA256": "a" * 64}
                for layer in (0, 5, 22) for depth in (12, 32)]}
        receipt = {"memory": {"qualified": True, "peak_bytes": 400_000_000},
            "vm": {"before": {"swapins": 0, "swapouts": 0}, "after": {"swapins": 0, "swapouts": 0}}}
        return report, receipt

    def test_admission_requires_all_cases_and_exact_complete_positive_timings(self):
        report, receipt = self.fixture()
        self.assertTrue(decision(report, receipt)["admit_engine_benchmark"])
        for mutation in (lambda r: r["cases"].pop(),
                         lambda r: r["cases"].append(copy.deepcopy(r["cases"][0])),
                         lambda r: r["cases"][0]["balancedSeconds"].pop(),
                         lambda r: r["cases"][0]["balancedSeconds"].__setitem__(0, float("nan")),
                         lambda r: r["cases"][0].__setitem__("outputSHA256", ""),
                         lambda r: r["readControls"][0].__setitem__("noCacheReturnCode", -1)):
            bad = copy.deepcopy(report)
            mutation(bad)
            with self.assertRaises(EvidenceError): decision(bad, receipt)

    def test_noise_small_win_and_per_layer_regression_do_not_admit(self):
        report, receipt = self.fixture()
        for case in report["cases"]: case["balancedSeconds"] = [.95] * 8
        self.assertFalse(decision(report, receipt)["admit_engine_benchmark"])
        report, receipt = self.fixture()
        report["cases"][1]["balancedSeconds"] = [1.1] * 8
        self.assertFalse(decision(report, receipt)["admit_engine_benchmark"])

    def test_thermal_and_paging_exclude_timing_not_functional_results(self):
        for field in ("thermal", "paging"):
            report, receipt = self.fixture()
            if field == "thermal": report["conditionsAfter"]["thermalState"] = "fair"
            else: receipt["vm"]["after"]["swapins"] = 1
            result = decision(report, receipt)
            self.assertFalse(result["timing_eligible"])
            self.assertFalse(result["admit_engine_benchmark"])
            self.assertIsNone(result["default_depth_median_paired_reduction"])

    def test_peak_is_actual_process_gate(self):
        report, receipt = self.fixture()
        receipt["memory"]["peak_bytes"] = 512_000_001
        with self.assertRaises(EvidenceError): decision(report, receipt)


if __name__ == "__main__": unittest.main()
