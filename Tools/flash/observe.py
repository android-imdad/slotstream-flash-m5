#!/usr/bin/env python3
"""Darwin child-exit observation and physical-footprint sampling."""
from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable

from common import fresh_output


class SamplingError(RuntimeError):
    pass


class ProcessExitedDuringSample(SamplingError):
    pass


class _RUsageInfoV4(ctypes.Structure):
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64)
        for name in (
            "ri_user_time", "ri_system_time", "ri_pkg_idle_wkups", "ri_interrupt_wkups",
            "ri_pageins", "ri_wired_size", "ri_resident_size", "ri_phys_footprint",
            "ri_proc_start_abstime", "ri_proc_exit_abstime", "ri_child_user_time",
            "ri_child_system_time", "ri_child_pkg_idle_wkups", "ri_child_interrupt_wkups",
            "ri_child_pageins", "ri_child_elapsed_abstime", "ri_diskio_bytesread",
            "ri_diskio_byteswritten", "ri_cpu_time_qos_default", "ri_cpu_time_qos_maintenance",
            "ri_cpu_time_qos_background", "ri_cpu_time_qos_utility", "ri_cpu_time_qos_legacy",
            "ri_cpu_time_qos_user_initiated", "ri_cpu_time_qos_user_interactive",
            "ri_billed_system_time", "ri_serviced_system_time", "ri_logical_writes",
            "ri_lifetime_max_phys_footprint", "ri_instructions", "ri_cycles",
            "ri_billed_energy", "ri_serviced_energy", "ri_interval_max_phys_footprint",
            "ri_runnable_time",
        )
    ]


class _SigInfo(ctypes.Structure):
    """Darwin siginfo_t from the installed SDK's sys/signal.h."""

    _fields_ = [
        ("si_signo", ctypes.c_int),
        ("si_errno", ctypes.c_int),
        ("si_code", ctypes.c_int),
        ("si_pid", ctypes.c_int),
        ("si_uid", ctypes.c_uint),
        ("si_status", ctypes.c_int),
        ("si_addr", ctypes.c_void_p),
        ("si_value", ctypes.c_void_p),
        ("si_band", ctypes.c_long),
        ("_pad", ctypes.c_ulong * 7),
    ]


P_PID = 1
WNOHANG = 0x00000001
WEXITED = 0x00000004
WNOWAIT = 0x00000020
CLD_EXITED = 1
CLD_KILLED = 2
CLD_DUMPED = 3
TERMINAL_SAMPLING_POLICY = "darwin-rusage-v2-terminal-before-reap"


def validate_siginfo_layout() -> None:
    expected = {"size": 104, "si_code": 8, "si_pid": 12, "si_uid": 16, "si_status": 20}
    observed = {
        "size": ctypes.sizeof(_SigInfo),
        "si_code": _SigInfo.si_code.offset,
        "si_pid": _SigInfo.si_pid.offset,
        "si_uid": _SigInfo.si_uid.offset,
        "si_status": _SigInfo.si_status.offset,
    }
    if observed != expected:
        raise SamplingError(f"Darwin siginfo_t ABI mismatch: {observed}")


class DarwinExitObserver:
    """Observe an owned child exit without reaping it."""

    def __init__(self, pid: int, waitid_call: Callable[..., int] | None = None):
        if sys.platform != "darwin":
            raise SamplingError("waitid child observation requires Darwin")
        if type(pid) is not int or pid <= 0:
            raise SamplingError("waitid requires a positive owned child PID")
        validate_siginfo_layout()
        self.pid = pid
        if waitid_call is None:
            libc = ctypes.CDLL(None, use_errno=True)
            libc.waitid.argtypes = [ctypes.c_int, ctypes.c_uint,
                                    ctypes.POINTER(_SigInfo), ctypes.c_int]
            libc.waitid.restype = ctypes.c_int
            self._waitid = libc.waitid
        else:
            self._waitid = waitid_call

    def observe(self) -> dict[str, int] | None:
        info = _SigInfo()
        ctypes.set_errno(0)
        result = self._waitid(P_PID, self.pid, ctypes.byref(info),
                              WEXITED | WNOHANG | WNOWAIT)
        if result != 0:
            raise SamplingError(
                f"waitid({self.pid}, WNOWAIT) failed: errno {ctypes.get_errno()}")
        if info.si_pid == 0:
            return None
        if info.si_pid != self.pid:
            raise SamplingError(f"waitid returned foreign child PID {info.si_pid}")
        if info.si_code not in (CLD_EXITED, CLD_KILLED, CLD_DUMPED):
            raise SamplingError(f"waitid returned unsupported child code {info.si_code}")
        if info.si_status < 0 or (info.si_code != CLD_EXITED and info.si_status == 0):
            raise SamplingError("waitid returned invalid child status")
        exit_code = info.si_status if info.si_code == CLD_EXITED else -info.si_status
        return {"pid": info.si_pid, "code": info.si_code, "status": info.si_status,
                "exit_code": exit_code}


class DarwinSampler:
    def __init__(self, pid: int):
        if sys.platform != "darwin":
            raise SamplingError("proc_pid_rusage sampling requires Darwin")
        self.pid = pid
        self._lib = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        self._lib.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        self._lib.proc_pid_rusage.restype = ctypes.c_int
        first = self._read()
        self.start_identity = first["start_abstime"]

    def _read(self) -> dict[str, int]:
        info = _RUsageInfoV4()
        if self._lib.proc_pid_rusage(self.pid, 4, ctypes.byref(info)) != 0:
            code = ctypes.get_errno()
            raise SamplingError(f"proc_pid_rusage({self.pid}) failed: errno {code}")
        return {
            "start_abstime": int(info.ri_proc_start_abstime),
            "exit_abstime": int(info.ri_proc_exit_abstime),
            "physical_footprint_bytes": int(info.ri_phys_footprint),
            "lifetime_peak_bytes": int(info.ri_lifetime_max_phys_footprint),
            "resident_bytes": int(info.ri_resident_size),
        }

    def sample(self) -> dict[str, Any]:
        value = self._read()
        if value["start_abstime"] != self.start_identity:
            raise SamplingError("PID identity changed while sampling")
        if value["exit_abstime"] > 0:
            raise ProcessExitedDuringSample(
                "process exited between waitid and live sampling")
        if value["physical_footprint_bytes"] <= 0 or value["lifetime_peak_bytes"] <= 0:
            raise SamplingError("live process sampler returned a nonpositive footprint")
        return value

    def terminal_sample(self, exit_observation: dict[str, int]) -> dict[str, Any]:
        value = self._read()
        if exit_observation.get("pid") != self.pid:
            raise SamplingError("terminal observation belongs to a foreign PID")
        if value["start_abstime"] != self.start_identity:
            raise SamplingError("PID identity changed before terminal sampling")
        if value["exit_abstime"] <= 0 or value["exit_abstime"] < value["start_abstime"]:
            raise SamplingError("terminal rusage has an invalid exit identity")
        if value["physical_footprint_bytes"] < 0 or value["resident_bytes"] < 0:
            raise SamplingError("terminal rusage returned a negative current footprint")
        if value["lifetime_peak_bytes"] <= 0:
            raise SamplingError("terminal rusage returned a nonpositive lifetime peak")
        value.update(wait_pid=exit_observation["pid"],
                     wait_code=exit_observation["code"],
                     wait_status=exit_observation["status"],
                     exit_code=exit_observation["exit_code"])
        return value


def lifecycle_probe(output: Path) -> int:
    output = fresh_output(output)
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; payload=bytearray(8<<20); time.sleep(.08)"],
        start_new_session=True)
    samples: list[dict[str, Any]] = []
    try:
        sampler = DarwinSampler(child.pid)
        observer = DarwinExitObserver(child.pid)
        deadline = time.monotonic() + 5
        terminal = None
        while time.monotonic() < deadline:
            exited = observer.observe()
            if exited is not None:
                terminal = sampler.terminal_sample(exited)
                terminal["sample_kind"] = "terminal"
                samples.append(terminal)
                break
            try:
                live = sampler.sample()
                live["sample_kind"] = "live"
                samples.append(live)
            except ProcessExitedDuringSample:
                continue
            time.sleep(.01)
        if terminal is None:
            raise SamplingError("lifecycle probe did not obtain terminal rusage")
        reaped = child.wait(timeout=2)
        if reaped != terminal["exit_code"]:
            raise SamplingError("reaped status differs from non-reaping observation")
        unavailable_after_reap = False
        try:
            sampler._read()
        except SamplingError:
            unavailable_after_reap = True
        if not unavailable_after_reap:
            raise SamplingError("proc_pid_rusage remained available after child reap")
        report = {
            "format": "slotstream-darwin-lifecycle-probe-v1",
            "policy": TERMINAL_SAMPLING_POLICY,
            "pid": child.pid,
            "reaped_exit_code": reaped,
            "samples": samples,
            "terminal_before_reap": True,
            "rusage_unavailable_after_reap": True,
        }
        (output / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n")
        return 0
    except Exception as error:
        if child.poll() is None:
            child.kill()
        child.wait()
        (output / "failure.json").write_text(
            json.dumps({"error": f"{type(error).__name__}: {error}"}) + "\n")
        return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lifecycle-probe", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.lifecycle_probe or args.output is None:
        parser.error("use --lifecycle-probe --output DIR")
    return lifecycle_probe(args.output)


if __name__ == "__main__":
    raise SystemExit(main())
