import copy
import unittest

from common import EvidenceError
from whole_expert_study import decision, LAYERS, EXPERTS, SIZES, RECORD


class WholeExpertTests(unittest.TestCase):
    def fixture(self):
        report = {"format": "slotstream-whole-expert-component-v1", "check": {"passed": True},
            "layers": LAYERS, "experts": EXPERTS, "queueDepth": 32,
            "originalRegionBytes": sum(SIZES.values()) * 10, "artifactBytes": RECORD * 30,
            "rawReadControls": [{"noCacheReturnCode": 0, "readAheadReturnCode": 0}],
            "conditionsBefore": {"thermalState": "nominal", "lowPowerModeEnabled": False},
            "conditionsAfter": {"thermalState": "nominal", "lowPowerModeEnabled": False},
            "cases": [{"layer": layer, "count": count,
                "orders": [["raw", "whole"] if (pair + LAYERS.index(layer)) % 2 == 0 else ["whole", "raw"] for pair in range(8)],
                "rawSourceBytesPerCall": SIZES[layer] * count, "wholeSourceBytesPerCall": RECORD * count,
                "rawSeconds": [float(count)] * 8, "wholeSeconds": [.8 * count] * 8, "outputSHA256": "a" * 64}
                for layer in LAYERS for count in (1, 4, 10)]}
        receipt = {"memory": {"qualified": True, "peak_bytes": 150_000_000},
            "vm": {"before": {"swapins": 0, "swapouts": 0}, "after": {"swapins": 0, "swapouts": 0}}}
        return report, receipt

    def test_complete_screen_prices_extra_bytes(self):
        report, receipt = self.fixture()
        result = decision(report, receipt)
        self.assertTrue(result["admit_engine_benchmark"])
        self.assertAlmostEqual(result["median_paired_total_reader_time_reduction"], .2)
        self.assertTrue(all(row["read_byte_increase_fraction"] > 0 for row in result["cases"]))
        self.assertFalse(result["inference_speedup_qualified"])

    def test_per_case_regression_and_small_total_gain_fail(self):
        report, receipt = self.fixture()
        report["cases"][0]["wholeSeconds"] = [1.1] * 8
        self.assertFalse(decision(report, receipt)["admit_engine_benchmark"])
        for case in report["cases"]: case["wholeSeconds"] = [.95 * v for v in case["rawSeconds"]]
        self.assertFalse(decision(report, receipt)["admit_engine_benchmark"])

    def test_incomplete_or_rewritten_evidence_is_refused(self):
        report, receipt = self.fixture()
        for mutate in (lambda r: r["cases"].pop(),
                       lambda r: r["cases"].append(copy.deepcopy(r["cases"][0])),
                       lambda r: r["cases"][0]["wholeSeconds"].pop(),
                       lambda r: r["cases"][0]["rawSeconds"].__setitem__(0, float("inf")),
                       lambda r: r["cases"][0].__setitem__("wholeSourceBytesPerCall", 1),
                       lambda r: r.__setitem__("artifactBytes", 1),
                       lambda r: r["cases"][0].__setitem__("outputSHA256", ""),
                       lambda r: r["rawReadControls"][0].__setitem__("noCacheReturnCode", -1)):
            bad = copy.deepcopy(report)
            mutate(bad)
            with self.assertRaises(EvidenceError): decision(bad, receipt)

    def test_environment_and_peak_gates(self):
        for cause in ("thermal", "paging", "unknown"):
            report, receipt = self.fixture()
            if cause == "thermal": report["conditionsAfter"]["thermalState"] = "fair"
            elif cause == "paging": receipt["vm"]["after"]["swapouts"] = 1
            else: del receipt["vm"]["before"]["swapins"]
            result = decision(report, receipt)
            self.assertEqual(result["disposition"], "timing_ineligible")
            self.assertFalse(result["admit_engine_benchmark"])
        report, receipt = self.fixture()
        receipt["memory"]["peak_bytes"] = 512_000_001
        with self.assertRaises(EvidenceError): decision(report, receipt)

    def test_source_native_requires_distinct_identity_and_original_byte_counts(self):
        report, receipt = self.fixture()
        with self.assertRaises(EvidenceError): decision(report, receipt, representation="source-native")
        report["format"] = "slotstream-source-native-component-v1"
        report["artifactBytes"] = report["originalRegionBytes"]
        for case in report["cases"]: case["wholeSourceBytesPerCall"] = case["rawSourceBytesPerCall"]
        result = decision(report, receipt, representation="source-native")
        self.assertTrue(result["admit_engine_benchmark"])
        self.assertTrue(all(row["read_byte_increase_fraction"] == 0 for row in result["cases"]))
        with self.assertRaises(EvidenceError): decision(report, receipt)
        with self.assertRaises(EvidenceError): decision(report, receipt, representation="unknown")
        report["cases"][0]["wholeSourceBytesPerCall"] += 1
        with self.assertRaises(EvidenceError): decision(report, receipt, representation="source-native")


if __name__ == "__main__": unittest.main()
