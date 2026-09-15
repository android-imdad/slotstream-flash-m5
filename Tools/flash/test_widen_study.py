#!/usr/bin/env python3
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import EvidenceError
import widen_study


class WidenStudyTests(unittest.TestCase):
    def test_cooldown_failure_prevents_model_launch(self):
        with mock.patch.object(widen_study, "wait_for_nominal", side_effect=EvidenceError("cooldown failed")), \
             mock.patch.object(widen_study.benchmark, "launch") as launch:
            with self.assertRaisesRegex(EvidenceError, "cooldown failed"):
                widen_study._run_arm(Path("binary"), Path("model"),
                    {"id": "fixture", "text": "fixture"}, "scalar", Path("output"), "test")
            launch.assert_not_called()

    def test_cooldown_is_saved_and_precedes_settling_and_launch(self):
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw)
            (output / "stats.json").write_text("{}")
            model = output / "model"
            model.mkdir()
            (model / "config.json").write_text("{}")
            events = []
            readiness = {"stable_seconds": 30, "observations": [{"conditions": {"thermalState": "nominal"}}]}
            receipt = {"result": {"functional_success": True}}
            with mock.patch.object(widen_study, "wait_for_nominal", side_effect=lambda: events.append("cooldown") or readiness), \
                 mock.patch.object(widen_study.benchmark, "settle_before_model_launch", side_effect=lambda: events.append("settle") or {"requested_seconds": 2}), \
                 mock.patch.object(widen_study.benchmark, "launch", side_effect=lambda *a, **k: events.append("launch") or 0), \
                 mock.patch.object(widen_study, "validate_receipt_file", return_value=receipt), \
                 mock.patch.object(widen_study, "require_terminal_sampling"):
                _, _, settled = widen_study._run_arm(Path("binary"), model,
                    {"id": "fixture", "text": "fixture"}, "scalar", output, "test")
            self.assertEqual(events, ["cooldown", "settle", "launch"])
            self.assertEqual(settled["requested_seconds"], 2)
            self.assertEqual(json.loads((output / "settling.json").read_text())["cooldown"], readiness)

    def rows(self, packed=0.8, first=1.0):
        result = []
        prompts = [item["id"] for item in widen_study.cache_study.load_fixture()["prompts"]]
        for prompt in prompts:
            for pair in range(10):
                result.append({"prompt_id": prompt, "pair_index": pair,
                               "scalar_decode_seconds": 1.0,
                               "packed_decode_seconds": packed,
                               "scalar_first_token_seconds": 1.0,
                               "packed_first_token_seconds": first})
        return result

    def test_paired_bootstrap_is_deterministic_and_paired(self):
        first = widen_study.paired_bootstrap([1.0] * 10, [0.8] * 10)
        second = widen_study.paired_bootstrap([1.0] * 10, [0.8] * 10)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["estimate"], 0.2)
        self.assertGreater(first["lower_95"], 0)
        with self.assertRaises(EvidenceError):
            widen_study.paired_bootstrap([1.0], [0.8, 0.9])

    def test_qualification_requires_every_predeclared_pair_and_threshold(self):
        accepted = widen_study.qualification_decision(self.rows(), resamples=100)
        self.assertTrue(accepted["qualified"])
        self.assertEqual(accepted["disposition"], "qualified")
        weak = widen_study.qualification_decision(self.rows(packed=0.95), resamples=100)
        self.assertFalse(weak["qualified"])
        with self.assertRaisesRegex(EvidenceError, "missing"):
            widen_study.qualification_decision(self.rows()[:-1], resamples=100)

    def test_qualification_applies_first_token_guardrail_per_workload(self):
        decision = widen_study.qualification_decision(self.rows(first=1.06), resamples=100)
        self.assertFalse(decision["qualified"])
        self.assertTrue(all(not item["passed"] for item in decision["workloads"].values()))

    def test_policy_stats_requires_explicit_packed_and_bounded_scalar_exception(self):
        document = {"stats": {"decodeSeconds": 1.0, "firstTokenSeconds": 1.0,
                              "prefillSeconds": 1.0, "requestSeconds": 2.0,
                              "decodeTokens": 2}, "load_seconds": 1.0}
        with mock.patch.object(widen_study.cache_study, "validate_stats"):
            with self.assertRaisesRegex(EvidenceError, "packed"):
                widen_study.validate_stats(document, {}, "packed4-to6")
            scalar = widen_study.validate_stats(document, {}, "scalar",
                                                 known_scalar_reference=True)
            self.assertEqual(scalar["decode_tokens_per_second"], 2.0)
            document["effective_expert_widening"] = "packed4-to6"
            packed = widen_study.validate_stats(document, {}, "packed4-to6")
            self.assertEqual(packed["decode_seconds"], 1.0)

    def test_qualification_rejects_zero_first_token_and_invalid_pair_indices(self):
        rows = self.rows()
        rows[0]["scalar_first_token_seconds"] = 0
        with self.assertRaisesRegex(EvidenceError, "first-token"):
            widen_study.qualification_decision(rows, resamples=10)
        for index in (False, None, "0", 0.0):
            rows = self.rows()
            rows[0]["pair_index"] = index
            with self.assertRaisesRegex(EvidenceError, "predeclared pair"):
                widen_study.qualification_decision(rows, resamples=10)

    def test_timing_eligibility_separates_paging_from_functional_success(self):
        system = {"thermalState": "nominal", "lowPowerModeEnabled": False}
        vm = {"swapins": 100, "swapouts": 200}
        document = {"stats": {"generatorSystemBefore": dict(system),
                               "generatorSystemAfter": dict(system),
                               "generatorVMBefore": dict(vm), "generatorVMAfter": dict(vm)}}
        receipt = {"memory": {"qualified": True},
                   "result": {"functional_success": True},
                   "vm": {"before": dict(vm), "after": dict(vm)}}
        widen_study.validate_timing_eligibility(document, receipt)
        for counter in ("swapins", "swapouts"):
            changed = copy.deepcopy(receipt)
            changed["vm"]["after"][counter] += 1
            with self.assertRaisesRegex(EvidenceError, "paging"):
                widen_study.validate_timing_eligibility(document, changed)
            self.assertTrue(changed["result"]["functional_success"])
        changed = copy.deepcopy(document)
        changed["stats"]["generatorVMAfter"]["swapins"] += 1
        with self.assertRaisesRegex(EvidenceError, "paging"):
            widen_study.validate_timing_eligibility(changed, receipt)
        for field, value in (("thermalState", "serious"), ("lowPowerModeEnabled", True)):
            changed = copy.deepcopy(document)
            changed["stats"]["generatorSystemAfter"][field] = value
            with self.assertRaisesRegex(EvidenceError, "operating conditions"):
                widen_study.validate_timing_eligibility(changed, receipt)
        changed = copy.deepcopy(receipt)
        changed["vm"]["after"].pop("swapins")
        with self.assertRaisesRegex(EvidenceError, "paging"):
            widen_study.validate_timing_eligibility(document, changed)
        changed["memory"]["qualified"] = False
        with self.assertRaisesRegex(EvidenceError, "memory"):
            widen_study.validate_timing_eligibility(document, changed)

    def test_exact_work_rejects_id_and_work_drift(self):
        fields = {"decodeTokens": 2, "promptTokens": 3, "prefillTokens": 3,
                  "decodeForwardPasses": 2, "prefillRecords": 1, "decodeRecords": 1,
                  "prefillReadBytes": 10, "decodeReadBytes": 10, "finishReason": "stop"}
        left = {"prompt_ids": [1, 2, 3], "output_ids": [4, 5], "stats": fields}
        widen_study._exact_work(left, copy.deepcopy(left))
        changed = copy.deepcopy(left); changed["output_ids"] = [4, 6]
        with self.assertRaisesRegex(EvidenceError, "IDs"):
            widen_study._exact_work(left, changed)
        changed = copy.deepcopy(left); changed["stats"]["decodeRecords"] = 2
        with self.assertRaisesRegex(EvidenceError, "identical"):
            widen_study._exact_work(left, changed)

    def test_schema_covers_every_study_report_format(self):
        schema = json.loads((HERE / "schemas/widening-study-v1.json").read_text())
        formats = {entry["properties"]["format"]["const"] for entry in schema["oneOf"]}
        self.assertEqual(formats, {"slotstream-widening-component-evidence-v1",
                                   "slotstream-widening-parity-v1",
                                   "slotstream-widening-screen-v1",
                                   "slotstream-widening-qualify-v1"})
        for entry in schema["oneOf"]:
            self.assertFalse(entry["additionalProperties"])
            self.assertTrue(set(entry["required"]).issubset(entry["properties"]))

    def test_parity_evidence_rejects_missing_stale_and_wrong_candidate(self):
        candidate = {"binary_sha256": "a" * 64, "metallib_sha256": "b" * 64,
                     "source_archive_sha256": "c" * 64, "build_identity_sha256": "d" * 64}
        model = {"path": "/model", "config_sha256": "e" * 64, "index_sha256": "f" * 64}
        corpus = {"path": "/corpus", "index_sha256": "1" * 64,
                  "completion_sha256": "2" * 64}
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            report = {"format": "slotstream-widening-parity-v1", "schema_version": 1,
                      "qualification": False, "exact_capture_parity": True,
                      "ordinary_cli_id_parity": True,
                      "reference_binary_sha256": widen_study.REFERENCE_BINARY_SHA256,
                      "candidate": candidate, "model": model, "corpus": corpus,
                      "captures": [{}], "ordinary": [{}], "harness_hashes": {"h": "x"}}
            widen_study.atomic_json(root / "report.json", report)
            widen_study.atomic_json(root / "completion.json", {
                "format": "slotstream-widening-study-completion-v1",
                "report_sha256": widen_study.sha256(root / "report.json"),
                "qualification": False})
            widen_study.validate_parity_evidence(root, candidate=candidate, model=model,
                                                 corpus=corpus)
            with self.assertRaisesRegex(EvidenceError, "candidate"):
                widen_study.validate_parity_evidence(root,
                    candidate={**candidate, "binary_sha256": "0" * 64}, model=model, corpus=corpus)
            report["ordinary"].append({})
            widen_study.atomic_json(root / "report.json", report)
            with self.assertRaisesRegex(EvidenceError, "completion"):
                widen_study.validate_parity_evidence(root, candidate=candidate, model=model,
                                                     corpus=corpus)
            (root / "completion.json").unlink()
            with self.assertRaises(EvidenceError):
                widen_study.validate_parity_evidence(root, candidate=candidate, model=model,
                                                     corpus=corpus)

    def test_runtime_controls_allow_only_documented_availability_drift(self):
        plan = {"memory_ledger": {"pool": 1}, "vision": False,
                "device_available_gb": 20, "device_ram_gb": 48,
                "device_working_set_gb": 40, "pool_slots": 100}
        document = {"plan": plan, "optimizations": {"x": False},
                    "numerical_environment": {"mlx_enable_tf32_raw": None,
                                              "effective_tf32": True},
                    "effective_pool_slots": 100, "effective_prefill_chunk": 256,
                    "effective_mtp": False, "effective_vision": False}
        receipt = {"environment": {"LANG": "C"}}
        other = copy.deepcopy(document)
        other["plan"]["device_available_gb"] = 19
        widen_study.compare_runtime_controls(document, receipt, other, copy.deepcopy(receipt))
        mutations = [
            lambda value: value["plan"]["memory_ledger"].update(pool=2),
            lambda value: value["optimizations"].update(x=True),
            lambda value: value["numerical_environment"].update(effective_tf32=False),
            lambda value: value.update(effective_pool_slots=99),
            lambda value: value.update(effective_mtp=True),
            lambda value: value.update(effective_vision=True),
        ]
        for mutate in mutations:
            changed = copy.deepcopy(other); mutate(changed)
            with self.assertRaisesRegex(EvidenceError, "controls"):
                widen_study.compare_runtime_controls(document, receipt, changed,
                                                     copy.deepcopy(receipt))
        changed_receipt = {"environment": {"LANG": "other"}}
        with self.assertRaisesRegex(EvidenceError, "controls"):
            widen_study.compare_runtime_controls(document, receipt, other, changed_receipt)

    def test_capture_runtime_controls_reject_plan_numerical_and_environment_drift(self):
        report = {"plan": {"memory_ledger": {"pool": 1}, "device_available_gb": 20,
                           "device_ram_gb": 48, "device_working_set_gb": 40,
                           "mtp": False, "vision": False, "pool_slots": 100},
                  "memory_ledger": {"pool": 1}, "optimizations": {"x": False},
                  "numerical_environment": {"mlx_enable_tf32_raw": None,
                                            "effective_tf32": True}}
        receipt = {"environment": {"LANG": "C"}}
        other = copy.deepcopy(report); other["plan"]["device_available_gb"] = 19
        widen_study.compare_capture_runtime_controls(report, receipt, other,
                                                     copy.deepcopy(receipt))
        for key, mutate in (
            ("plan", lambda value: value["plan"].update(pool_slots=99)),
            ("ledger", lambda value: value["memory_ledger"].update(pool=2)),
            ("optimization", lambda value: value["optimizations"].update(x=True)),
            ("numerical", lambda value: value["numerical_environment"].update(effective_tf32=False)),
        ):
            changed = copy.deepcopy(other); mutate(changed)
            with self.subTest(key=key), self.assertRaisesRegex(EvidenceError, "controls"):
                widen_study.compare_capture_runtime_controls(report, receipt, changed,
                                                             copy.deepcopy(receipt))
        with self.assertRaisesRegex(EvidenceError, "controls"):
            widen_study.compare_capture_runtime_controls(report, receipt, other,
                {"environment": {"LANG": "other"}})

    def test_component_validator_rejects_forged_missing_and_wrong_provenance(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); launch = root / "launch"; native = launch / "native"
            native.mkdir(parents=True)
            binary = root / "slotstream"; binary.write_bytes(b"binary")
            metallib = root / "mlx.metallib"; metallib.write_bytes(b"metal")
            identity_file = root / "build-identity.json"; identity_file.write_bytes(b"identity")
            source = root / "build-source.tar.gz"; source.write_bytes(b"source")
            build = {"binary_sha256": widen_study.sha256(binary),
                     "metallib_sha256": widen_study.sha256(metallib),
                     "source_archive_sha256": widen_study.sha256(source)}
            paths = {"identity": identity_file, "source_archive": source,
                     "binary": binary, "metallib": metallib}
            exact = {"name": "exact-widening", "passed": True, "measurements": {},
                     "items": [{"name": name, "passed": True} for name in (
                         "all 65536 two-byte inputs match an independent bit oracle",
                         "overlap retains scalar clear-then-expand behavior",
                         "real 1638400-code projection is byte-identical",
                         "omitted policy remains scalar")]}
            results = []
            for workers in (1, 8, 16):
                for policy in ("scalar", "packed4-to6"):
                    results.append({"workers": workers, "policy": policy,
                                    "secondsPerCall": [0.001] * 8,
                                    "medianSecondsPerCall": 0.001,
                                    "checksum": workers})
            report = {"format": "slotstream-widening-command-v1", "schemaVersion": 1,
                      "processID": 42, "mode": "synthetic",
                      "provenance": {"executablePath": str(binary.resolve()),
                          "executableSHA256": build["binary_sha256"],
                          "metallibSHA256": build["metallib_sha256"],
                          "buildIdentitySHA256": widen_study.sha256(identity_file),
                          "sourceArchiveSHA256": build["source_archive_sha256"],
                          "compilerVersion": "Swift test"},
                      "synthetic": {"format": "slotstream-widening-synthetic-v1",
                          "schemaVersion": 1, "method": "test", "codeCountPerCall": 1_638_400,
                          "pairCount": 8, "allocatedBufferBytes": 32_768_000,
                          "results": results, "exactCheck": exact,
                          "processFootprintEndBytes": 1}}
            receipt = {"process": {"pid": 42},
                       "command": [str(binary.resolve()), "widening-check", "--synthetic"],
                       "identities": {"executable": {"path": str(binary.resolve()),
                                                        "sha256": build["binary_sha256"]}},
                       "artifacts": {}}

            def write(value):
                widen_study.atomic_json(native / "report.json", value)
                widen_study.atomic_json(native / "completion.json", {
                    "format": "slotstream-widening-completion-v1", "passed": True,
                    "report_sha256": widen_study.sha256(native / "report.json"),
                    "schema_version": 1})
                receipt["artifacts"]["native/report.json"] = {
                    "sha256": widen_study.sha256(native / "report.json"),
                    "bytes": (native / "report.json").stat().st_size}

            with mock.patch.object(widen_study, "validate_build_identity",
                                   return_value=(build, paths)):
                write(report)
                widen_study.validate_component_native(native, launch, receipt, binary,
                                                       synthetic=True, model=None)
                forged = copy.deepcopy(report); forged["processID"] = 43; write(forged)
                with self.assertRaisesRegex(EvidenceError, "command identity"):
                    widen_study.validate_component_native(native, launch, receipt, binary,
                                                           synthetic=True, model=None)
                missing = copy.deepcopy(report)
                missing["synthetic"]["exactCheck"]["items"].pop(); write(missing)
                with self.assertRaisesRegex(EvidenceError, "exact checks"):
                    widen_study.validate_component_native(native, launch, receipt, binary,
                                                           synthetic=True, model=None)
                wrong = copy.deepcopy(report)
                wrong["provenance"]["executableSHA256"] = "0" * 64; write(wrong)
                with self.assertRaisesRegex(EvidenceError, "provenance"):
                    widen_study.validate_component_native(native, launch, receipt, binary,
                                                           synthetic=True, model=None)
                (native / "report.json").unlink()
                with self.assertRaises(EvidenceError):
                    widen_study.validate_component_native(native, launch, receipt, binary,
                                                           synthetic=True, model=None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
