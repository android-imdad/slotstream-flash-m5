import unittest
from replay import TraceGroup
from common import EvidenceError
import prefetch_study


class PrefetchTests(unittest.TestCase):
    def group(self, experts, tokens=1):
        return TraceGroup(tokens, tuple(tuple(tuple(experts) for _ in range(tokens)) for _ in range(48)))

    def test_first_decode_has_no_prediction_and_hits_are_not_prefetched(self):
        groups = [self.group(range(10), 2), self.group(range(10))]
        for method in ('previous-token', 'recent8-frequency'):
            result = prefetch_study.screen(groups, 640, [100] * 48, method)
            self.assertEqual(result['prefetched_bytes'], 0)
            self.assertEqual(result['useful_bytes'], 0)

    def test_wrong_predictions_add_io_without_changing_clock_demand(self):
        groups = [self.group(range(10), 2)] + [self.group(range(i * 10, i * 10 + 10)) for i in range(1, 5)]
        result = prefetch_study.screen(groups, 40, [100] * 48, 'previous-token')
        self.assertGreater(result['unused_bytes'], 0)
        self.assertEqual(result['useful_bytes'], 0)
        self.assertEqual(result['prefetched_bytes'], result['unused_bytes'])
        self.assertLessEqual(result['peak_staging_source_bytes'], 1000)
        self.assertFalse(result['speedup_qualified'])
        self.assertLess(result['candidate_slots'], result['baseline_slots'])
        self.assertGreater(result['reserved_staging_and_insertion_bytes'], 0)

    def test_repeated_decode_predictions_can_cover_misses_with_bounded_staging(self):
        groups = [self.group(range(10), 2)] + [self.group(range(10, 20)) for _ in range(3)]
        result = prefetch_study.screen(groups, 40, [100] * 48, 'previous-token')
        self.assertGreater(result['useful_bytes'], 0)
        self.assertEqual(result['unused_bytes'], 0)

    def test_forecast_uses_only_supplied_past_and_ties_use_expert_id(self):
        past = [tuple(range(10, 20)), tuple(range(10))]
        self.assertEqual(prefetch_study.forecast(past, 'previous-token'), list(range(10)))
        self.assertEqual(prefetch_study.forecast(past, 'recent8-frequency'), list(range(10)))
        self.assertEqual(prefetch_study.forecast([], 'previous-token'), [])
        with self.assertRaises(EvidenceError):
            prefetch_study.forecast(past, 'future-router')


if __name__ == '__main__':
    unittest.main()
