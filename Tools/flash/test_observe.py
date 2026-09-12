#!/usr/bin/env python3
from __future__ import annotations

import ctypes
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import FLASH_ROOT
from observe import (CLD_EXITED, DarwinExitObserver, DarwinSampler,
                     ProcessExitedDuringSample, SamplingError, _SigInfo,
                     lifecycle_probe, validate_siginfo_layout)


class ObserveTests(unittest.TestCase):
    def setUp(self):
        FLASH_ROOT.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="observe-test-", dir=FLASH_ROOT)
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_installed_siginfo_layout_matches_ctypes(self):
        validate_siginfo_layout()
        self.assertEqual(ctypes.sizeof(_SigInfo), 104)
        self.assertEqual(_SigInfo.si_pid.offset, 12)

    def test_waitid_observation_validates_pid_code_status_and_errors(self):
        def result(pid=42, code=CLD_EXITED, status=7, error=0):
            def call(_kind, _pid, pointer, _flags):
                info = pointer._obj
                info.si_pid = pid
                info.si_code = code
                info.si_status = status
                ctypes.set_errno(10 if error else 0)
                return error
            return call

        observed = DarwinExitObserver(42, waitid_call=result()).observe()
        self.assertEqual(observed, {"pid": 42, "code": CLD_EXITED,
                                    "status": 7, "exit_code": 7})
        with self.assertRaisesRegex(SamplingError, "foreign"):
            DarwinExitObserver(42, waitid_call=result(pid=43)).observe()
        with self.assertRaisesRegex(SamplingError, "unsupported"):
            DarwinExitObserver(42, waitid_call=result(code=99)).observe()
        with self.assertRaisesRegex(SamplingError, "errno 10"):
            DarwinExitObserver(42, waitid_call=result(error=-1)).observe()

    def sampler(self, values):
        sampler = object.__new__(DarwinSampler)
        sampler.pid = 42
        sampler.start_identity = 100
        iterator = iter(values)
        sampler._read = lambda: next(iterator)
        return sampler

    def test_live_zero_and_exit_between_check_are_distinct_failures(self):
        zero = {"start_abstime": 100, "exit_abstime": 0,
                "physical_footprint_bytes": 0, "lifetime_peak_bytes": 3,
                "resident_bytes": 1}
        with self.assertRaisesRegex(SamplingError, "live"):
            self.sampler([zero]).sample()
        exited = dict(zero, exit_abstime=101, lifetime_peak_bytes=4)
        with self.assertRaises(ProcessExitedDuringSample):
            self.sampler([exited]).sample()

    def test_terminal_sample_requires_identity_exit_and_positive_peak(self):
        observation = {"pid": 42, "code": CLD_EXITED, "status": 0, "exit_code": 0}
        valid = {"start_abstime": 100, "exit_abstime": 101,
                 "physical_footprint_bytes": 0, "lifetime_peak_bytes": 9,
                 "resident_bytes": 0}
        terminal = self.sampler([valid]).terminal_sample(observation)
        self.assertEqual(terminal["lifetime_peak_bytes"], 9)
        for name, mutation in (
            ("wrong start", {"start_abstime": 99}),
            ("missing exit", {"exit_abstime": 0}),
            ("zero peak", {"lifetime_peak_bytes": 0}),
        ):
            value = dict(valid)
            value.update(mutation)
            with self.subTest(name=name), self.assertRaises(SamplingError):
                self.sampler([value]).terminal_sample(observation)
        with self.assertRaisesRegex(SamplingError, "foreign"):
            self.sampler([valid]).terminal_sample(dict(observation, pid=43))

    def test_real_child_remains_waitable_through_terminal_sample(self):
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; x=bytearray(8<<20); time.sleep(.05)"],
            start_new_session=True)
        try:
            sampler = DarwinSampler(child.pid)
            observer = DarwinExitObserver(child.pid)
            deadline = time.monotonic() + 5
            observed = None
            while time.monotonic() < deadline and observed is None:
                observed = observer.observe()
                if observed is None:
                    time.sleep(.01)
            self.assertIsNotNone(observed)
            terminal = sampler.terminal_sample(observed)
            self.assertGreater(terminal["lifetime_peak_bytes"], 0)
            self.assertIsNone(child.returncode)
            self.assertEqual(child.wait(timeout=2), observed["exit_code"])
            with self.assertRaises(SamplingError):
                sampler._read()
        finally:
            if child.returncode is None:
                child.kill()
                child.wait()

    def test_lifecycle_probe_writes_bounded_real_evidence(self):
        output = self.root / "probe"
        self.assertEqual(lifecycle_probe(output), 0)
        self.assertTrue((output / "report.json").is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
