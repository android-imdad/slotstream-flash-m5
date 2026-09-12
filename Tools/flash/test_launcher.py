#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import benchmark
from common import EvidenceError, FLASH_ROOT, read_json, sha256
from observe import SamplingError
from receipts import validate_receipt_file


class FakeSampler:
    footprint = 1024 * 1024

    def __init__(self, pid: int):
        self.pid = pid
        self.start_identity = 12345

    def sample(self):
        return {"start_abstime": self.start_identity, "physical_footprint_bytes": self.footprint,
                "lifetime_peak_bytes": self.footprint, "resident_bytes": self.footprint}


class LauncherTests(unittest.TestCase):
    def setUp(self):
        FLASH_ROOT.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="launcher-test-", dir=FLASH_ROOT)
        self.output = Path(self.temporary.name) / "run"

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def vm():
        return {"reclaimable_bytes": 100_000_000_000, "swapins": 0, "swapouts": 0}

    def run_child(self, source="import time; time.sleep(0.12)", **kwargs):
        max_seconds = kwargs.pop("max_seconds", 2.0)
        return benchmark.launch(self.output, 1.0, max_seconds, [sys.executable, "-c", source],
                                run_set_id="test-run", model_hash="a" * 64,
                                sampler_factory=kwargs.pop("sampler_factory", FakeSampler),
                                preflight_func=lambda needed: self.vm(), vm_provider=self.vm, **kwargs)

    def test_success_writes_hashed_completion(self):
        self.assertEqual(self.run_child(), 0)
        receipt = read_json(self.output / "receipt.json")
        self.assertFalse(receipt["qualified"])
        self.assertTrue(receipt["result"]["functional_success"])
        self.assertGreater(receipt["memory"]["sample_count"], 0)
        self.assertEqual(receipt["artifacts"]["stdout.txt"]["sha256"], sha256(self.output / "stdout.txt"))
        self.assertEqual(read_json(self.output / "completion.json")["receipt_sha256"], sha256(self.output / "receipt.json"))
        validate_receipt_file(self.output / "receipt.json")

    def test_child_stats_are_hashed_without_parent_buffering(self):
        stats = self.output / "stats.json"
        source = f"import pathlib,time; pathlib.Path({str(stats)!r}).write_text('{{\"ok\":true}}'); time.sleep(.05)"
        self.assertEqual(self.run_child(source), 0)
        receipt = read_json(self.output / "receipt.json")
        self.assertEqual(receipt["artifacts"]["stats.json"]["sha256"], sha256(stats))
        self.assertIsNotNone(receipt["identities"]["executable"]["sha256"])

    def test_executable_removed_after_exec_is_recorded_failure(self):
        executable = Path(self.temporary.name) / "self-remove.sh"
        executable.write_text('#!/bin/sh\nrm "$0"\nsleep 0.05\n')
        executable.chmod(0o700)
        code = benchmark.launch(self.output, 1.0, 2.0, [str(executable)], run_set_id="r", model_hash="m",
                                sampler_factory=FakeSampler, preflight_func=lambda needed: self.vm(), vm_provider=self.vm)
        self.assertEqual(code, 1)
        receipt = read_json(self.output / "receipt.json")
        self.assertTrue(any("executable unavailable" in reason for reason in receipt["qualification_reasons"]))
        self.assertFalse((self.output / "completion.json").exists())

    def test_nonzero_exit_is_unqualified(self):
        self.assertEqual(self.run_child("import time,sys; time.sleep(.05); sys.exit(7)"), 1)
        self.assertEqual(read_json(self.output / "receipt.json")["result"]["exit_code"], 7)
        self.assertFalse((self.output / "completion.json").exists())

    def test_timeout_drains_owned_child(self):
        self.assertEqual(self.run_child("import time; time.sleep(30)", max_seconds=0.06), 1)
        receipt = read_json(self.output / "receipt.json")
        self.assertTrue(receipt["result"]["timed_out"])
        with self.assertRaises(ProcessLookupError):
            os.kill(receipt["process"]["pid"], 0)

    def test_cancellation_drains_owned_child(self):
        self.assertEqual(self.run_child("import time; time.sleep(30)", cancelled=lambda: True), 1)
        self.assertTrue(read_json(self.output / "receipt.json")["result"]["interrupted"])

    def test_sigint_preserves_unqualified_receipt(self):
        class InterruptSampler(FakeSampler):
            def sample(self): raise KeyboardInterrupt()
        self.assertEqual(self.run_child("import time; time.sleep(30)", sampler_factory=InterruptSampler), 1)
        receipt = read_json(self.output / "receipt.json")
        self.assertTrue(receipt["result"]["interrupted"])
        self.assertIn("SIGINT", receipt["qualification_reasons"])

    def test_sampler_failure_is_not_zero_memory(self):
        class BrokenSampler(FakeSampler):
            def sample(self): raise SamplingError("injected")
        self.assertEqual(self.run_child(sampler_factory=BrokenSampler), 1)
        memory = read_json(self.output / "receipt.json")["memory"]
        self.assertIsNone(memory["peak_bytes"])
        self.assertFalse(memory["qualified"])

    def test_exit_before_sample_is_unqualified(self):
        def unavailable(_pid):
            time.sleep(0.08)
            raise SamplingError("process already exited")
        self.assertEqual(self.run_child("pass", sampler_factory=unavailable), 1)
        self.assertIn("memory sampler unavailable", read_json(self.output / "receipt.json")["qualification_reasons"])

    def test_budget_exceedance_is_preserved(self):
        class HugeSampler(FakeSampler):
            footprint = 2_000_000_000
        self.assertEqual(self.run_child(sampler_factory=HugeSampler), 1)
        self.assertTrue(read_json(self.output / "receipt.json")["result"]["budget_exceeded"])

    def test_pid_reuse_is_rejected(self):
        class ReusedSampler(FakeSampler):
            def sample(self):
                value = super().sample(); value["start_abstime"] += 1; return value
        self.assertEqual(self.run_child(sampler_factory=ReusedSampler), 1)
        self.assertIn("memory sampler failed", read_json(self.output / "receipt.json")["qualification_reasons"])

    def test_missing_identities_cannot_qualify(self):
        code = benchmark.launch(self.output, 1.0, 2.0, [sys.executable, "-c", "import time; time.sleep(.05)"],
                                sampler_factory=FakeSampler, preflight_func=lambda needed: self.vm(), vm_provider=self.vm)
        self.assertEqual(code, 0)
        self.assertIsNone(read_json(self.output / "receipt.json")["identities"]["model_hash"])

    def test_reused_and_outside_outputs_are_rejected(self):
        self.output.mkdir()
        with self.assertRaises(EvidenceError): self.run_child()
        outside = Path(tempfile.gettempdir()) / "slotstream-forbidden-output"
        with self.assertRaises(EvidenceError):
            benchmark.launch(outside, 1, 1, ["true"], run_set_id="x", model_hash="y",
                             sampler_factory=FakeSampler, preflight_func=lambda _: self.vm(), vm_provider=self.vm)

    def test_traversal_and_symlinked_output_ancestor_are_rejected(self):
        with self.assertRaises(EvidenceError):
            benchmark.launch(FLASH_ROOT / ".." / "escaped", 1, 1, ["true"], run_set_id="x", model_hash="y",
                             sampler_factory=FakeSampler, preflight_func=lambda _: self.vm(), vm_provider=self.vm)
        external = Path(self.temporary.name) / "external"; external.mkdir()
        link = FLASH_ROOT / f"escape-{os.getpid()}-{time.time_ns()}"; link.symlink_to(external)
        try:
            with self.assertRaises(EvidenceError):
                benchmark.launch(link / "run", 1, 1, ["true"], run_set_id="x", model_hash="y",
                                 sampler_factory=FakeSampler, preflight_func=lambda _: self.vm(), vm_provider=self.vm)
        finally:
            link.unlink()

    def test_invalid_budgets_and_empty_argv_are_rejected(self):
        with self.assertRaises(EvidenceError):
            benchmark.launch(self.output, 0, 1, ["true"])
        with self.assertRaises(EvidenceError):
            benchmark.launch(self.output, 1, 0, ["true"])
        with self.assertRaises(EvidenceError):
            benchmark.launch(self.output, 1, 1, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
