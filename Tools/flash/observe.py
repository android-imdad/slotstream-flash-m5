#!/usr/bin/env python3
"""Darwin process identity and physical-footprint sampling."""
from __future__ import annotations

import ctypes
import os
import sys
from typing import Any


class SamplingError(RuntimeError):
    pass


class _RUsageInfoV4(ctypes.Structure):
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64) for name in (
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
            "physical_footprint_bytes": int(info.ri_phys_footprint),
            "lifetime_peak_bytes": int(info.ri_lifetime_max_phys_footprint),
            "resident_bytes": int(info.ri_resident_size),
        }

    def sample(self) -> dict[str, Any]:
        value = self._read()
        if value["start_abstime"] != self.start_identity:
            raise SamplingError("PID identity changed while sampling")
        if value["physical_footprint_bytes"] <= 0 or value["lifetime_peak_bytes"] <= 0:
            raise SamplingError("process sampler returned a nonpositive footprint")
        return value
