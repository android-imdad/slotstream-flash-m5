#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import (EvidenceError, FLASH_ROOT, RECEIPT_FORMAT, ROOT, build_source_paths,
                    regular_file, sha256, validate_source_map)
from gates import validate_checks, validate_python_test_result
import benchmark
import build_identity
from receipts import validate_receipt, validate_receipt_file, validate_selection


def good_checks():
    return {"checks": [
        {"name": "jang-formats", "items": [{"name": "format", "passed": True}], "skipped": None},
        {"name": "jang-numerics", "items": [{"name": "numeric", "passed": True}], "skipped": None},
    ], "passed": 2, "failed": 0, "skipped": 0}


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        FLASH_ROOT.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="receipt-test-", dir=FLASH_ROOT)
        self.root = Path(self.temporary.name)
        self.extra_paths = []
        (self.root / "stdout.txt").write_text("ok\n")
        (self.root / "stderr.txt").write_text("")
        sample = {"start_abstime": 2, "physical_footprint_bytes": 1, "lifetime_peak_bytes": 1,
                  "resident_bytes": 1, "elapsed_seconds": 0.0}
        (self.root / "memory.jsonl").write_text(json.dumps(sample) + "\n")
        self.receipt = {
            "format": RECEIPT_FORMAT, "schema_version": 1, "kind": "launch", "qualified": True,
            "qualification_reasons": [], "command": ["true"], "environment": {},
            "started_at_unix": 1.0, "ended_at_unix": 2.0, "duration_seconds": 1.0,
            "process": {"pid": 1, "start_identity": 2},
            "memory": {"target_gb_decimal": 1.0, "sample_interval_seconds": 0.5, "qualified": True,
                       "peak_bytes": 1, "sample_count": 1, "samples_artifact": "memory.jsonl", "sampler_error": None}, "vm": {},
            "result": {"exit_code": 0, "functional_success": True, "timed_out": False,
                       "interrupted": False, "budget_exceeded": False, "evidence_failure": False},
            "artifacts": {name: {"bytes": (self.root / name).stat().st_size, "sha256": sha256(self.root / name)}
                          for name in ("stdout.txt", "stderr.txt", "memory.jsonl")},
            "identities": {"run_set_id": "r", "model_hash": "m"},
        }

    def tearDown(self):
        for path in self.extra_paths:
            if path.is_dir():
                import shutil
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
        self.temporary.cleanup()

    def test_valid_receipt_and_artifact_hashes(self):
        validate_receipt(self.receipt, self.root, require_qualified=True)

    def test_memory_summary_and_sample_types_are_strict(self):
        exceeded = copy.deepcopy(self.receipt); exceeded["memory"]["target_gb_decimal"] = 0.0000000005
        with self.assertRaises(EvidenceError): validate_receipt(exceeded, self.root)
        (self.root / "memory.jsonl").write_text('{"physical_footprint_bytes": "one"}\n')
        malformed = copy.deepcopy(self.receipt)
        malformed["artifacts"]["memory.jsonl"] = {"bytes": (self.root / "memory.jsonl").stat().st_size,
                                                    "sha256": sha256(self.root / "memory.jsonl")}
        with self.assertRaises(EvidenceError): validate_receipt(malformed, self.root)

    def test_memory_artifact_must_be_declared_and_contained(self):
        absent = copy.deepcopy(self.receipt); del absent["artifacts"]["memory.jsonl"]
        with self.assertRaises(EvidenceError): validate_receipt(absent, self.root)
        escaped = self.root.parent / f"{self.root.name}-samples.jsonl"
        escaped.write_text((self.root / "memory.jsonl").read_text()); self.extra_paths.append(escaped)
        escaping = copy.deepcopy(self.receipt)
        escaping["memory"]["samples_artifact"] = "../" + escaped.name
        escaping["artifacts"]["../" + escaped.name] = {"bytes": escaped.stat().st_size, "sha256": sha256(escaped)}
        with self.assertRaises(EvidenceError): validate_receipt(escaping, self.root)

    def test_contradictory_functional_success_is_rejected(self):
        contradictory = copy.deepcopy(self.receipt)
        contradictory["qualified"] = False
        contradictory["qualification_reasons"] = ["failed"]
        contradictory["result"]["exit_code"] = 7
        self.assertTrue(contradictory["result"]["functional_success"])
        with self.assertRaises(EvidenceError): validate_receipt(contradictory, self.root)

    def test_completion_hash_must_bind_receipt(self):
        receipt_path = self.root / "receipt.json"
        receipt_path.write_text(json.dumps(self.receipt))
        (self.root / "completion.json").write_text(json.dumps({"receipt_sha256": "0" * 64, "qualified": True}))
        with self.assertRaises(EvidenceError): validate_receipt_file(receipt_path)

    def test_partial_receipt_rejected(self):
        partial = copy.deepcopy(self.receipt); del partial["result"]
        with self.assertRaises(EvidenceError): validate_receipt(partial, self.root)

    def test_changed_and_missing_artifacts_rejected(self):
        (self.root / "stdout.txt").write_text("changed")
        with self.assertRaises(EvidenceError): validate_receipt(self.receipt, self.root)
        (self.root / "stdout.txt").unlink()
        with self.assertRaises(EvidenceError): validate_receipt(self.receipt, self.root)

    def test_artifact_traversal_and_symlink_ancestor_rejected(self):
        escaped = self.root.parent / f"{self.root.name}-escaped.txt"; escaped.write_text("x"); self.extra_paths.append(escaped)
        traversal = copy.deepcopy(self.receipt)
        traversal["artifacts"] = {"../escaped.txt": {"bytes": 1, "sha256": sha256(escaped)}}
        with self.assertRaises(EvidenceError): validate_receipt(traversal, self.root)
        outside = self.root.parent / f"{self.root.name}-outside"; outside.mkdir(); self.extra_paths.append(outside); (outside / "evidence").write_text("x")
        (self.root / "link").symlink_to(outside, target_is_directory=True)
        linked = copy.deepcopy(self.receipt)
        linked["artifacts"] = {"link/evidence": {"bytes": 1, "sha256": sha256(outside / "evidence")}}
        with self.assertRaises(EvidenceError): validate_receipt(linked, self.root)

    def test_stale_source_map_rejected(self):
        fake = self.make_fake_repo("stale")
        identity = {"source": {str(item.relative_to(fake)): sha256(item) for item in build_source_paths(fake)}}
        validate_source_map(identity, root=fake)
        (fake / "Sources" / "Fixture.swift").write_text("after")
        with self.assertRaises(EvidenceError): validate_source_map(identity, root=fake)

    def test_new_source_file_is_detected(self):
        fake = self.make_fake_repo("added")
        identity = {"source": {str(item.relative_to(fake)): sha256(item) for item in build_source_paths(fake)}}
        (fake / "Sources" / "Added.swift").write_text("new")
        with self.assertRaises(EvidenceError): validate_source_map(identity, root=fake)

    def test_symlink_escape_rejected(self):
        target = self.root / "target"; target.write_text("x")
        link = self.root / "link"; link.symlink_to(target)
        with self.assertRaises(EvidenceError): regular_file(link, within=self.root)

    def test_required_native_checks_are_exact_and_nonempty(self):
        validate_checks(good_checks())
        absent = good_checks(); absent["checks"][0]["name"] = "prefix-jang-formats-extra"
        with self.assertRaises(EvidenceError): validate_checks(absent)
        empty = good_checks(); empty["checks"][1]["items"] = []
        with self.assertRaises(EvidenceError): validate_checks(empty)

    def test_skipped_failed_and_malformed_checks_rejected(self):
        skipped = good_checks(); skipped["checks"][0]["skipped"] = "unavailable"; skipped["skipped"] = 1; skipped["passed"] = 1
        with self.assertRaises(EvidenceError): validate_checks(skipped)
        failed = good_checks(); failed["checks"][0]["items"][0]["passed"] = False
        with self.assertRaises(EvidenceError): validate_checks(failed)
        with self.assertRaises(EvidenceError): validate_checks({"checks": "bad", "passed": 0, "failed": 0, "skipped": 0})

    def test_zero_or_unnamed_python_tests_rejected(self):
        base = {"test_ids": [], "test_classes": [], "tests_run": 0, "failures": [], "errors": [], "skipped": [], "successful": True}
        with self.assertRaises(EvidenceError): validate_python_test_result(base, {"LauncherTests"})
        two = {"test_ids": ["a", "b"], "test_classes": ["ReceiptTests"], "tests_run": 2,
               "failures": [], "errors": [], "skipped": [], "successful": True}
        with self.assertRaises(EvidenceError): validate_python_test_result(two, {"LauncherTests"})
        two["test_classes"].append("LauncherTests")
        self.assertEqual(validate_python_test_result(two, {"LauncherTests", "ReceiptTests"}), 2)

    def make_fake_repo(self, name):
        fake = self.root / name
        (fake / "Sources").mkdir(parents=True)
        (fake / "Tools").mkdir()
        for relative, content in (("Sources/Fixture.swift", "before"), ("Package.swift", "package"),
                                  ("Package.resolved", "resolved"), ("Makefile", "make"),
                                  ("Tools/build_identity.py", "helper"), ("Tools/fetch_metallib.sh", "fetch")):
            (fake / relative).write_text(content)
        return fake

    def make_fake_build(self, name):
        fake = self.make_fake_repo(name)
        release = fake / ".build" / "release"
        build_identity.bind(fake, "before", ".build/release")
        (release / "slotstream").write_bytes(b"binary")
        (release / "mlx.metallib").write_bytes(b"metal")
        build_identity.bind(fake, "after", ".build/release")
        return fake, release

    def test_archive_validates_and_copies_exact_build(self):
        fake, release = self.make_fake_build("archive-valid")
        output = self.root / "archive-output"
        self.assertEqual(benchmark.archive(release / "slotstream", output, repo_root=fake), 0)
        self.assertEqual(sha256(output / "bin" / "build-source.tar.gz"), sha256(release / "build-source.tar.gz"))
        self.assertTrue((output / "completion.json").is_file())
        validate_receipt_file(output / "receipt.json", require_qualified=True)

    def test_archive_rejects_changed_binary_metallib_and_missing_artifact(self):
        for index, target in enumerate(("slotstream", "mlx.metallib", "build-source.tar.gz")):
            fake, release = self.make_fake_build(f"archive-mutation-{index}")
            path = release / target
            if target == "build-source.tar.gz": path.unlink()
            else: path.write_bytes(b"changed")
            output = self.root / f"archive-failure-{index}"
            self.assertEqual(benchmark.archive(release / "slotstream", output, repo_root=fake), 1)
            self.assertFalse((output / "completion.json").exists())

    def test_selection_requires_all_pending_future_stages(self):
        selection = {"format": "slotstream-flash-selection-v1", "schema_version": 1,
                     "selected_stage_ids": [0], "selected_arm_ids": ["baseline"],
                     "stages": [{"stage_id": i, "disposition": "accepted" if i == 0 else "pending",
                                 "reason": "x", "evidence": []} for i in range(8)]}
        validate_selection(selection)
        selection["stages"].pop()
        with self.assertRaises(EvidenceError): validate_selection(selection)


if __name__ == "__main__":
    unittest.main(verbosity=2)
