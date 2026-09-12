#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import struct
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import EvidenceError, FLASH_ROOT
from replay import (ClockCache, LAYERS, TOP_K, TraceGroup, encode_trace, parse_trace,
                    replay, validate_runtime_stats)


def group(tokens, value):
    layers = []
    for layer in range(LAYERS):
        rows = []
        for token in range(tokens):
            expert = value(layer, token) if callable(value) else value
            rows.append(tuple([expert] * TOP_K))
        layers.append(tuple(rows))
    return TraceGroup(tokens=tokens, layers=tuple(layers))


class ReplayTests(unittest.TestCase):
    def setUp(self):
        FLASH_ROOT.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="replay-test-", dir=FLASH_ROOT)
        self.root = Path(self.temp.name)
        self.source = [100 + layer for layer in range(LAYERS)]

    def tearDown(self): self.temp.cleanup()

    def write(self, data):
        path = self.root / "trace.bin"; path.write_bytes(data); return path

    def test_strict_trace_round_trip(self):
        groups = [group(2, lambda layer, token: (layer + token) % 512), group(1, 9)]
        self.assertEqual(parse_trace(self.write(encode_trace(groups))), groups)

    def test_empty_truncated_and_trailing_records_fail(self):
        with self.assertRaises(EvidenceError): parse_trace(self.write(b""))
        with self.assertRaises(EvidenceError): parse_trace(self.write(b"\0" * 11))
        good = encode_trace([group(2, 1)])
        with self.assertRaises(EvidenceError): parse_trace(self.write(good[:-1]))
        with self.assertRaises(EvidenceError): parse_trace(self.write(good + b"x"))

    def test_trace_symlink_and_oversized_file_fail(self):
        target = self.write(encode_trace([group(2, 1)]))
        link = self.root / "link.bin"; link.symlink_to(target)
        with self.assertRaisesRegex(EvidenceError, "symlink"):
            parse_trace(link)
        oversized = self.root / "oversized.bin"; oversized.write_bytes(b"x" * ((1 << 20) + 1))
        with self.assertRaisesRegex(EvidenceError, "exceeds"):
            parse_trace(oversized)

    def test_invalid_id_layer_order_and_topk_fail(self):
        raw = bytearray(encode_trace([group(2, 1)]))
        struct.pack_into("<h", raw, 12, 512)
        with self.assertRaises(EvidenceError): parse_trace(self.write(raw))
        raw = bytearray(encode_trace([group(2, 1)])); struct.pack_into("<i", raw, 0, 1)
        with self.assertRaises(EvidenceError): parse_trace(self.write(raw))
        raw = bytearray(encode_trace([group(2, 1)])); struct.pack_into("<i", raw, 8, 9)
        with self.assertRaises(EvidenceError): parse_trace(self.write(raw))

    def test_duplicate_demands_are_deduplicated_first_use(self):
        result = replay([group(2, 3)], 64, self.source)
        self.assertEqual(result["prefill"]["requests"], 48)
        self.assertEqual(result["prefill"]["misses"], 48)

    def test_all_hits_and_all_misses(self):
        hits = replay([group(2, 3), group(1, 3)], 48, self.source)
        self.assertEqual((hits["decode"]["hits"], hits["decode"]["misses"]), (48, 0))
        misses = replay([group(2, 3), group(1, 4)], 48, self.source)
        self.assertEqual((misses["decode"]["hits"], misses["decode"]["misses"]), (0, 48))

    def test_hits_are_pinned_before_miss_victim_selection(self):
        cache = ClockCache(2, self.source)
        cache.request(0, ((0, 1),), 0)
        cache.request(0, ((0, 2),), 1)
        self.assertIn((0, 0), cache.slot_of)
        self.assertIn((0, 2), cache.slot_of)
        self.assertNotIn((0, 1), cache.slot_of)

    def test_full_clock_wrap_clears_reference_bits(self):
        cache = ClockCache(2, self.source)
        cache.request(0, ((0, 1),), 0)
        self.assertEqual(cache.hand, 0)
        cache.request(0, ((2,),), 1)
        self.assertNotIn((0, 0), cache.slot_of)
        self.assertEqual(cache.hand, 1)

    def test_recent_overflow_falls_back_to_bounded_clock(self):
        cache = ClockCache(2, self.source, window=8)
        cache.request(0, ((0, 1),), 0)
        cache.request(0, ((2,),), 1)
        self.assertEqual(len(cache.slot_of), 2)
        self.assertIn((0, 2), cache.slot_of)

    def test_recent_preference_avoids_unpinned_recent_key(self):
        cache = ClockCache(3, self.source, window=2)
        cache.request(0, ((0, 1, 2),), 0)
        cache.request(0, ((0,),), 9)
        cache.request(0, ((3,),), 10)
        self.assertIn((0, 0), cache.slot_of)
        self.assertNotIn((0, 1), cache.slot_of)

    def test_token_epoch_is_not_layer_visit_epoch(self):
        result = replay([group(3, 0), group(1, 1), group(1, 2)], 48, self.source, window=2)
        self.assertEqual(result["token_positions"], 5)

    def test_cache_smaller_than_recent_union_and_request_reset(self):
        groups = [group(2, 0), group(1, 1), group(1, 2)]
        first = replay(groups, 48, self.source, window=8)
        second = replay(groups, 48, self.source, window=8)
        self.assertEqual(first, second)

    def test_varying_source_bytes_are_charged_by_layer(self):
        source = [100] + [200] * 47
        result = replay([group(2, 0)], 48, source)
        self.assertEqual(result["prefill"]["miss_bytes"], 100 + 47 * 200)

    def test_wrong_flat_replay_totals_fail_reconciliation(self):
        groups = [group(2, 0), group(1, 0)]
        native = replay(groups, 48, self.source)
        stats = {"schema_version": 1, "effective_mtp": False, "effective_pool_slots": 48,
                 "prompt_ids": [1, 2], "output_ids": [3],
                 "optimizations": {"readScopeTokens": 0, "layerLocalFloorCache": False,
                                   "layerExpertWorkspace": False, "sparsePoolPins": False,
                                   "skipUnusedFinalForward": True},
                 "stats": {"prefillRecords": native["prefill"]["misses"],
                           "decodeRecords": native["decode"]["misses"] + 1,
                           "prefillReadBytes": native["prefill"]["miss_bytes"],
                           "decodeReadBytes": native["decode"]["miss_bytes"],
                           "decodeForwardPasses": 1, "decodeTokens": 1, "finishReason": "stop",
                           "promptTokens": 2, "prefillTokens": 2, "smallPrefillSweeps": 0,
                           "abortedReadScopes": 0, "reusedPrefixTokens": 0,
                           "contextArithmetic": "standard", "verifyPasses": 0,
                           "memoryPressureCancelled": False}}
        with self.assertRaisesRegex(EvidenceError, "miss reconciliation"):
            validate_runtime_stats(stats, groups, native)


if __name__ == "__main__": unittest.main(verbosity=2)
