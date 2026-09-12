#!/usr/bin/env python3
"""Strict Slotstream router-trace parsing and expert-cache replay."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import struct
import stat
import sys
from typing import Any, Iterable

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import EvidenceError

LAYERS = 48
TOP_K = 10
EXPERTS = 512
MAX_TRACE_BYTES = 1 << 20
CACHE_RECORD_BYTES = 4_300_800
AGE_METADATA_BYTES_PER_SLOT = 8


@dataclass(frozen=True)
class TraceGroup:
    tokens: int
    layers: tuple[tuple[tuple[int, ...], ...], ...]


@dataclass
class PhaseTotals:
    requests: int = 0
    requested_bytes: int = 0
    hits: int = 0
    misses: int = 0
    miss_bytes: int = 0

    def as_dict(self) -> dict[str, int]:
        return vars(self).copy()


def parse_trace(path: Path) -> list[TraceGroup]:
    if path.is_symlink(): raise EvidenceError("router trace symlinks are not accepted")
    try: before = path.stat()
    except OSError as error: raise EvidenceError(f"router trace missing: {path}") from error
    if not stat.S_ISREG(before.st_mode): raise EvidenceError("router trace must be a regular file")
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        data = stream.read(MAX_TRACE_BYTES + 1)
        after = os.fstat(stream.fileno())
    if len(data) <= 0: raise EvidenceError("router trace is empty")
    if len(data) > MAX_TRACE_BYTES: raise EvidenceError(f"router trace exceeds {MAX_TRACE_BYTES} bytes")
    final = path.stat()
    if ((opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            != (final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns)
            or len(data) != final.st_size):
        raise EvidenceError("router trace changed while being read")
    offset = 0
    records: list[tuple[int, int, tuple[tuple[int, ...], ...]]] = []
    while offset < len(data):
        if len(data) - offset < 12: raise EvidenceError("truncated router trace header")
        layer, tokens, top_k = struct.unpack_from("<iii", data, offset)
        offset += 12
        if layer < 0 or layer >= LAYERS: raise EvidenceError(f"invalid router layer: {layer}")
        if tokens <= 0 or tokens >= 64: raise EvidenceError(f"unsupported router token count: {tokens}")
        if top_k != TOP_K: raise EvidenceError(f"unsupported router topK: {top_k}")
        payload = tokens * top_k * 2
        if payload > len(data) - offset: raise EvidenceError("truncated router trace payload")
        values = struct.unpack_from(f"<{tokens * top_k}h", data, offset)
        offset += payload
        if any(expert < 0 or expert >= EXPERTS for expert in values):
            raise EvidenceError("router trace contains an invalid expert ID")
        rows = tuple(tuple(values[row * top_k:(row + 1) * top_k]) for row in range(tokens))
        records.append((layer, tokens, rows))
    if offset != len(data): raise EvidenceError("router trace has trailing bytes")
    if len(records) % LAYERS: raise EvidenceError("router trace ends with a partial layer group")
    groups = []
    for start in range(0, len(records), LAYERS):
        block = records[start:start + LAYERS]
        tokens = block[0][1]
        if [record[0] for record in block] != list(range(LAYERS)):
            raise EvidenceError("router layer group is not complete and ordered")
        if any(record[1] != tokens for record in block):
            raise EvidenceError("router group token counts differ by layer")
        groups.append(TraceGroup(tokens=tokens, layers=tuple(record[2] for record in block)))
    if not groups or groups[0].tokens <= 1:
        raise EvidenceError("router trace lacks its initial multi-token prefill group")
    if any(group.tokens != 1 for group in groups[1:]):
        raise EvidenceError("router trace contains a non-unit decode group")
    return groups


def encode_trace(groups: Iterable[TraceGroup]) -> bytes:
    out = bytearray()
    for group in groups:
        for layer, rows in enumerate(group.layers):
            out.extend(struct.pack("<iii", layer, group.tokens, TOP_K))
            flat = [expert for row in rows for expert in row]
            out.extend(struct.pack(f"<{len(flat)}h", *flat))
    return bytes(out)


class ClockCache:
    def __init__(self, capacity: int, source_bytes: list[int], window: int | None = None):
        if capacity <= 0: raise EvidenceError("cache capacity must be positive")
        if len(source_bytes) != LAYERS or any(type(value) is not int or value <= 0 for value in source_bytes):
            raise EvidenceError("source bytes must contain 48 positive integer layer sizes")
        if window is not None and window <= 0: raise EvidenceError("recent window must be positive")
        self.capacity = capacity
        self.source_bytes = source_bytes
        self.window = window
        self.key_of: list[tuple[int, int] | None] = [None] * capacity
        self.slot_of: dict[tuple[int, int], int] = {}
        self.ref = [False] * capacity
        self.pinned = [False] * capacity
        self.last_use: list[int | None] = [None] * capacity
        self.hand = 0

    def _native_victim(self, allowed=None) -> int:
        scanned = 0
        while True:
            slot = self.hand
            self.hand = (self.hand + 1) % self.capacity
            if self.pinned[slot] or (allowed is not None and not allowed(slot)):
                scanned += 1
            elif self.ref[slot]:
                self.ref[slot] = False
                scanned += 1
            else:
                return slot
            if scanned >= 3 * self.capacity:
                raise EvidenceError("cache exhausted or eviction scan exceeded native bound")

    def _victim(self, position: int) -> int:
        if self.window is None: return self._native_victim()
        floor = position - self.window + 1
        def nonrecent(slot: int) -> bool:
            return self.key_of[slot] is None or self.last_use[slot] is None or self.last_use[slot] < floor
        if any(not self.pinned[slot] and nonrecent(slot) for slot in range(self.capacity)):
            return self._native_victim(nonrecent)
        return self._native_victim()

    def request(self, layer: int, rows: tuple[tuple[int, ...], ...], base_position: int) -> PhaseTotals:
        ordered: list[tuple[int, int]] = []
        latest: dict[tuple[int, int], int] = {}
        seen = set()
        for token_offset, row in enumerate(rows):
            for expert in row:
                key = (layer, expert)
                latest[key] = base_position + token_offset
                if key not in seen:
                    seen.add(key)
                    ordered.append(key)
        totals = PhaseTotals(requests=len(ordered),
                             requested_bytes=len(ordered) * self.source_bytes[layer])
        hit_slots = {self.slot_of[key] for key in ordered if key in self.slot_of}
        miss_keys = [key for key in ordered if key not in self.slot_of]
        for slot in hit_slots:
            self.ref[slot] = True
            self.pinned[slot] = True
            self.last_use[slot] = latest[self.key_of[slot]]
        totals.hits = len(hit_slots)
        totals.misses = len(miss_keys)
        totals.miss_bytes = len(miss_keys) * self.source_bytes[layer]
        if len(hit_slots) + len(miss_keys) > self.capacity:
            raise EvidenceError("one layer demand exceeds cache capacity")
        victims = []
        position = base_position + len(rows) - 1
        for _ in miss_keys:
            slot = self._victim(position)
            self.pinned[slot] = True
            victims.append(slot)
        for key, slot in zip(miss_keys, victims):
            old = self.key_of[slot]
            if old is not None: self.slot_of.pop(old)
            self.key_of[slot] = key
            self.slot_of[key] = slot
            self.ref[slot] = True
            self.last_use[slot] = latest[key]
        self.pinned = [False] * self.capacity
        return totals


def replay(groups: list[TraceGroup], capacity: int, source_bytes: list[int],
           window: int | None = None) -> dict[str, Any]:
    if not groups: raise EvidenceError("cannot replay an empty trace")
    cache = ClockCache(capacity, source_bytes, window=window)
    phases = {"prefill": PhaseTotals(), "decode": PhaseTotals()}
    position = 0
    for group_index, group in enumerate(groups):
        phase = phases["prefill" if group_index == 0 else "decode"]
        for layer, rows in enumerate(group.layers):
            result = cache.request(layer, rows, position)
            for key, value in vars(result).items(): setattr(phase, key, getattr(phase, key) + value)
        position += group.tokens
    return {"capacity": capacity, "window": window,
            "metadata_bytes": 0 if window is None else capacity * AGE_METADATA_BYTES_PER_SLOT,
            "prefill": phases["prefill"].as_dict(), "decode": phases["decode"].as_dict(),
            "token_positions": position, "final_hand": cache.hand,
            "resident_keys": len(cache.slot_of)}


def validate_runtime_stats(stats_document: dict[str, Any], groups: list[TraceGroup],
                           replay_result: dict[str, Any]) -> None:
    required_top = {"effective_mtp", "effective_pool_slots", "output_ids", "prompt_ids",
                    "optimizations", "stats", "schema_version"}
    if not required_top.issubset(stats_document) or stats_document.get("schema_version") != 1:
        raise EvidenceError("stats document is incomplete")
    stats = stats_document["stats"]
    required_stats = {"prefillRecords", "decodeRecords", "prefillReadBytes", "decodeReadBytes",
                      "decodeForwardPasses", "decodeTokens", "finishReason", "promptTokens",
                      "prefillTokens", "smallPrefillSweeps", "abortedReadScopes", "reusedPrefixTokens",
                      "contextArithmetic", "verifyPasses", "memoryPressureCancelled"}
    if not required_stats.issubset(stats): raise EvidenceError("stats omit cache-replay observations")
    optimizations = stats_document["optimizations"]
    unsupported = {
        "effective_mtp": stats_document["effective_mtp"] is not False,
        "verify_passes": stats["verifyPasses"] != 0,
        "small_prefill_sweep": stats["smallPrefillSweeps"] != 0,
        "read_scope": stats["abortedReadScopes"] != 0 or optimizations.get("readScopeTokens") != 0,
        "prefix_reuse": stats["reusedPrefixTokens"] != 0,
        "layer_local_floor": optimizations.get("layerLocalFloorCache") is not False,
        "sparse_pool_pins": optimizations.get("sparsePoolPins") is not False,
        "workspace": optimizations.get("layerExpertWorkspace") is not False,
        "runtime_error": stats.get("runtimeError") is not None,
        "context_arithmetic": stats["contextArithmetic"] != "standard",
        "memory_cancelled": stats["memoryPressureCancelled"] is not False,
    }
    active = [name for name, enabled in unsupported.items() if enabled]
    if active: raise EvidenceError(f"unsupported cache-policy controls: {active}")
    if stats_document["effective_pool_slots"] != replay_result["capacity"]:
        raise EvidenceError("effective cache capacity differs from replay")
    if groups[0].tokens != stats["promptTokens"] or stats["prefillTokens"] != stats["promptTokens"]:
        raise EvidenceError("trace prefill token count differs from complete stats")
    if len(stats_document["prompt_ids"]) != stats["promptTokens"] or len(stats_document["output_ids"]) != stats["decodeTokens"]:
        raise EvidenceError("token identity/count mismatch")
    finish = stats["finishReason"]
    if finish == "length":
        if optimizations.get("skipUnusedFinalForward") is not True:
            raise EvidenceError("length finish requires the recorded final-forward policy")
        expected_forwards = max(0, stats["decodeTokens"] - 1)
    elif finish == "stop":
        expected_forwards = stats["decodeTokens"]
    else:
        raise EvidenceError(f"unsupported generator finish reason: {finish}")
    trace_forwards = len(groups) - 1
    if trace_forwards != expected_forwards or trace_forwards != stats["decodeForwardPasses"]:
        raise EvidenceError("trace groups do not match finish-derived decode forwards")
    for phase in ("prefill", "decode"):
        replay_phase = replay_result[phase]
        if replay_phase["misses"] != stats[f"{phase}Records"]:
            raise EvidenceError(f"{phase} native CLOCK miss reconciliation failed")
        if replay_phase["miss_bytes"] != stats[f"{phase}ReadBytes"]:
            raise EvidenceError(f"{phase} source-byte reconciliation failed")


def self_test() -> int:
    import unittest
    suite = unittest.defaultTestLoader.loadTestsFromName("test_replay")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.testsRun > 0 and not result.failures and not result.errors and not result.skipped else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if not args.self_test: parser.error("only --self-test is supported; use cache_study.py for collection and analysis")
    return self_test()


if __name__ == "__main__":
    raise SystemExit(main())
