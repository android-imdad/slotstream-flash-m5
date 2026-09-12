#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import os
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import EvidenceError
from m5_probe import (assess_actual_dispatch, capture_status, parse_metal_rows,
                      bounded_child_succeeded, expected_case_combinations, run_bounded_export,
                      run_probe, trace_table_context,
                      validate_case_numbers, validate_hash_binding)
from mlx_dispatch_probe import (LOG, instrument_quantized, validate_bound_hash,
                                validate_dispatch_claim, validate_inventory_delta,
                                validate_private_completion)


class M5ProbeTests(unittest.TestCase):
    class FakeSampler:
        def __init__(self, pid): self.pid = pid; self.start_identity = 1
        def sample(self):
            return {"start_abstime": 1, "physical_footprint_bytes": 1024,
                    "lifetime_peak_bytes": 1024, "resident_bytes": 1024}

    def bounded(self, command, *, timeout=2, max_bytes=1 << 20, sampler=None):
        root = Path(tempfile.mkdtemp(prefix="m5-bounded-", dir=Path(__file__).resolve().parents[2] / ".build/flash"))
        self.addCleanup(lambda: __import__("shutil").rmtree(root, ignore_errors=True))
        return run_bounded_export(command, root / "stdout", root / "stderr",
            timeout_seconds=timeout, max_bytes=max_bytes,
            preflight_func=lambda needed: {"reclaimable_bytes": 100_000_000_000},
            sampler_factory=sampler or self.FakeSampler)

    def test_bounded_child_success(self):
        result = self.bounded([sys.executable, "-c", "pass"])
        self.assertTrue(bounded_child_succeeded(result))

    def test_bounded_child_timeout_drains_owned_process(self):
        result = self.bounded([sys.executable, "-c", "import time; time.sleep(30)"], timeout=0.05)
        self.assertTrue(result["timed_out"])
        with self.assertRaises(ProcessLookupError): os.kill(result["pid"], 0)

    def test_bounded_child_fast_output_is_caught_after_exit(self):
        result = self.bounded([sys.executable, "-c", "print('x'*10000)"], max_bytes=100)
        self.assertTrue(result["output_exceeded"])
        self.assertLessEqual(result["stdout_bytes"] + result["stderr_bytes"], 100)
        self.assertFalse(bounded_child_succeeded(result))

    def test_bounded_child_interrupt_drains_owned_process(self):
        class InterruptSampler(self.FakeSampler):
            def sample(self): raise KeyboardInterrupt()
        result = self.bounded([sys.executable, "-c", "import time; time.sleep(30)"], sampler=InterruptSampler)
        self.assertTrue(result["interrupted"])
        with self.assertRaises(ProcessLookupError): os.kill(result["pid"], 0)

    def test_zero_records_stays_unverified(self):
        result = assess_actual_dispatch([])
        self.assertEqual(result["status"], "unverified")
        self.assertEqual(result["record_count"], 0)

    def test_nax_string_in_source_blob_is_not_execution_evidence(self):
        records = ["source contains gather_qmm_rhs_nax", {"source": "source-blob",
                   "record_type": "metal-compute-encoder", "kernel_name": "gather_qmm_rhs_nax"}]
        self.assertEqual(assess_actual_dispatch(records)["status"], "unverified")

    def test_grouped_record_without_decode_control_stays_unverified(self):
        records = [{"source": "xctrace-export-row", "record_type": "metal-compute-encoder",
                    "kernel_name": "gather_qmm_rhs_nax", "process": "slotstream 12",
                    "encoder": "encoder", "case": "grouped6", "target_pid_match": True}]
        self.assertEqual(assess_actual_dispatch(records)["status"], "unverified")

    def test_only_xctrace_encoder_row_can_mark_observed(self):
        records = [
            {"source": "xctrace-export-row", "record_type": "metal-compute-encoder",
             "kernel_name": "affine_gather_qmm_rhs_nax_f32", "process": "slotstream 12",
             "encoder": "encoder-1", "case": "grouped6", "target_pid_match": True},
            {"source": "xctrace-export-row", "record_type": "metal-compute-encoder",
             "kernel_name": "affine_gather_qmv_f32", "process": "slotstream 13",
             "encoder": "encoder-2", "case": "decode6", "target_pid_match": True},
        ]
        self.assertEqual(assess_actual_dispatch(records)["status"], "observed")

    def test_export_parser_binds_process_encoder_kernel_and_case(self):
        xml = '''<trace><schema name="metal-gpu-intervals">
                 <col><mnemonic>process</mnemonic></col><col><mnemonic>encoder-id</mnemonic></col>
                 <col><mnemonic>event-label</mnemonic></col></schema>
                 <row><process fmt="slotstream (42)"><pid fmt="42">42</pid></process>
                 <metal-command-buffer-id fmt="encoder-7"/><formatted-label fmt="affine_gather_qmm_rhs_nax_f32"/></row></trace>'''
        records = parse_metal_rows(xml, "metal-gpu-intervals", case="grouped6", target_pid=42)
        self.assertEqual(len(records), 1)
        self.assertTrue(records[0]["target_pid_match"])
        self.assertEqual(records[0]["encoder"], "encoder-7")

    def test_pid_match_is_exact_not_substring(self):
        xml = '''<trace><schema name="metal-gpu-intervals">
                 <col><mnemonic>process</mnemonic></col><col><mnemonic>encoder-id</mnemonic></col>
                 <col><mnemonic>event-label</mnemonic></col></schema>
                 <row><process fmt="slotstream (312)"><pid fmt="312">312</pid></process>
                 <id fmt="encoder"/><formatted-label fmt="gather_qmm_rhs_nax"/></row></trace>'''
        records = parse_metal_rows(xml, "metal-gpu-intervals", case="grouped6", target_pid=12)
        self.assertFalse(records[0]["target_pid_match"])
        self.assertEqual(assess_actual_dispatch(records)["status"], "unverified")

    def test_pipeline_creation_schema_never_counts_as_execution(self):
        xml = '''<trace><schema name="metal-shader-profiler-shader-list">
                 <col><mnemonic>process</mnemonic></col><col><mnemonic>name</mnemonic></col></schema>
                 <row><process fmt="slotstream (42)"><pid fmt="42">42</pid></process>
                 <label fmt="gather_qmm_rhs_nax"/></row></trace>'''
        self.assertEqual(parse_metal_rows(xml, "metal-shader-profiler-shader-list",
                                         case="grouped6", target_pid=42), [])
        context = trace_table_context(xml, "metal-shader-profiler-shader-list", target_pid=42)
        self.assertEqual(context["compiled_kernel_candidates"], ["gather_qmm_rhs_nax"])
        self.assertEqual(context["executed_encoder_ids"], [])

    def test_unrelated_process_context_is_excluded(self):
        xml = '''<trace><schema name="metal-gpu-intervals">
                 <col><mnemonic>process</mnemonic></col><col><mnemonic>encoder-id</mnemonic></col></schema>
                 <row><process fmt="other (312)"><pid fmt="312">312</pid></process><id fmt="encoder-9"/></row></trace>'''
        context = trace_table_context(xml, "metal-gpu-intervals", target_pid=12)
        self.assertEqual(context["executed_encoder_ids"], [])

    def test_capture_unavailable_and_failure_are_explicit(self):
        self.assertEqual(capture_status(requested=True, tool=None, exit_code=None,
                                        trace_exists=False, architecture="Apple M5")["status"], "unavailable")
        self.assertEqual(capture_status(requested=True, tool="xctrace", exit_code=7,
                                        trace_exists=False, architecture="Apple M5")["status"], "failed")

    def test_model_mode_refuses_profiler_capture(self):
        with self.assertRaises(EvidenceError):
            run_probe(Path("missing"), Path("unused"), synthetic=False, capture=True)

    def test_unsupported_device_never_claims_dispatch(self):
        result = capture_status(requested=True, tool="xctrace", exit_code=0,
                                trace_exists=True, architecture="Unknown")
        self.assertEqual(result["status"], "unsupported")

    def test_missing_and_stale_hashes_are_rejected(self):
        expected = {"binary": "a" * 64, "metallib": "b" * 64}
        with self.assertRaises(EvidenceError):
            validate_hash_binding({"binary": "a" * 64}, expected)
        with self.assertRaises(EvidenceError):
            validate_hash_binding({"binary": "0" * 64, "metallib": "b" * 64}, expected)
        validate_hash_binding(dict(expected), expected)

    def test_numeric_evidence_must_be_finite_and_within_tolerance(self):
        case = {"id": "c", "maxAbsoluteError": 0.1, "maxRelativeError": 0.01,
                "scalarSpotError": 0.05, "tolerance": 0.2}
        validate_case_numbers(case)
        case["maxAbsoluteError"] = float("nan")
        with self.assertRaises(EvidenceError): validate_case_numbers(case)
        case["maxAbsoluteError"] = 0.3
        with self.assertRaises(EvidenceError): validate_case_numbers(case)

    def test_full_case_matrix_is_exact(self):
        self.assertEqual(len(expected_case_combinations("synthetic")), 60)
        self.assertEqual(expected_case_combinations("trace-grouped6"), {("gate", 6, 16, True)})
        self.assertEqual(expected_case_combinations("trace-decode6"), {("gate", 6, 1, False)})

    def test_private_instrumentation_requires_both_real_dispatch_anchors(self):
        source = '#include "mlx/utils.h"\n' + (
            '  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);\n}\n\nvoid gather_qvm(') + (
            'x\n  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);\n}\n\nvoid gather_qmm_rhs(')
        patched = instrument_quantized(source)
        self.assertIn("path=gather_qmv", patched)
        self.assertIn("path=gather_qmm_rhs_nax", patched)
        self.assertLess(patched.index("dispatch_threadgroups"), patched.index("path=gather_qmv"))
        with self.assertRaises(EvidenceError): instrument_quantized(patched)

    def test_private_dispatch_log_is_structured(self):
        line = ("SLOTSTREAM_M5_DISPATCH case=grouped6 pid=42 path=gather_qmm_rhs_nax "
                "kernel=affine_gather_qmm_rhs_nax_nt M=16 N=640 K=2560 B=16 bits=6 submitted=1")
        match = LOG.search(line)
        self.assertIsNotNone(match)
        self.assertEqual(match.group(2), "42")
        claim = validate_dispatch_claim("grouped6", [match.groups()], 42)
        self.assertEqual(claim[2], "gather_qmm_rhs_nax")
        changed = list(match.groups()); changed[4] = "15"
        with self.assertRaises(EvidenceError): validate_dispatch_claim("grouped6", [tuple(changed)], 42)
        conflict = list(match.groups()); conflict[2] = "gather_qmv"; conflict[3] = "affine_gather_qmv_fast"
        with self.assertRaises(EvidenceError):
            validate_dispatch_claim("grouped6", [match.groups(), tuple(conflict)], 42)

    def test_private_inventory_allows_only_quantized_delta(self):
        path = "Source/Cmlx/mlx/mlx/backend/metal/quantized.cpp"
        self.assertEqual(validate_inventory_delta({path: "a"}, {path: "b"}), [path])
        with self.assertRaises(EvidenceError): validate_inventory_delta({path: "a"}, {path: "a"})
        with self.assertRaises(EvidenceError): validate_inventory_delta({path: "a"}, {path: "b", "extra": "x"})

    def test_private_case_hash_and_completion_mutations_fail(self):
        validate_bound_hash("a", "a", "case receipt")
        with self.assertRaises(EvidenceError): validate_bound_hash("changed", "a", "case log")
        completion = {"format": "slotstream-private-mlx-dispatch-completion-v1",
                      "result_sha256": "a", "functional_success": True}
        validate_private_completion(completion, "a")
        completion["result_sha256"] = "stale"
        with self.assertRaises(EvidenceError): validate_private_completion(completion, "a")


if __name__ == "__main__":
    unittest.main(verbosity=2)
