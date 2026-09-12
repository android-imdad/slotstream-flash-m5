#!/usr/bin/env python3
import copy, hashlib, json, sys, tempfile, unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import corpus
from common import EvidenceError


def rehash(value):
    value["contractSHA256"] = corpus.canonical_hash({k:v for k,v in value.items() if k != "contractSHA256"})
    return value


class CorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prepared = corpus.build_prepared()

    def changed(self):
        return copy.deepcopy(self.prepared)

    def test_normalization_partition_and_message_canonicalization(self):
        self.assertEqual(corpus.normalized("e\u0301  x\n y"), corpus.normalized("é x y"))
        self.assertEqual(corpus.partition("same"), int(hashlib.sha256(b"same").hexdigest(), 16) % 5)
        self.assertNotEqual(corpus.canonical_messages([{"role":"user","content":"ab"}]),
                            corpus.canonical_messages([{"role":"user","content":"a"},{"role":"assistant","content":"b"}]))

    def test_actual_records_and_coverage_are_derived(self):
        result = corpus.validate_prepared_data(self.prepared, reconstruct=True)
        self.assertEqual(result["finalCounts"], {"gsm8k":200,"humaneval":164,"xnli":600,"structured":200})
        self.assertGreaterEqual(sum(x["positions"] for x in result["counts"]["development"].values()), 512)
        self.assertGreaterEqual(sum(x["positions"] for x in result["counts"]["qualification"].values()), 4096)
        first = self.prepared["finalScored"]["xnli"][:3]
        self.assertEqual([x["language"] for x in first], ["en","fr","zh"])
        self.assertEqual(len({x["pairID"] for x in first}), 1)

    def test_source_version_license_and_omitted_source_fail(self):
        for mutation in ("version", "license", "omit"):
            data = self.changed()
            if mutation == "version": data["sources"]["versions"]["gsm8k"]["revision"] = "0" * 40
            elif mutation == "license": data["sources"]["versions"]["xnli"]["license"] = "MIT"
            else: del data["sources"]["files"]["gsm8k-test.jsonl"]
            with self.subTest(mutation=mutation), self.assertRaises(EvidenceError):
                corpus.validate_prepared_data(rehash(data), reconstruct=False)

    def test_duplicate_and_number_only_authored_near_clone_fail(self):
        data = self.changed(); data["teacherForced"]["documents"][1]["text"] = data["teacherForced"]["documents"][0]["text"]
        data["teacherForced"]["documents"][1]["rawSHA256"] = hashlib.sha256(data["teacherForced"]["documents"][1]["text"].encode()).hexdigest()
        data["teacherForced"]["documents"][1]["normalizedSHA256"] = corpus.normalized_hash(data["teacherForced"]["documents"][1]["text"])
        with self.assertRaises(EvidenceError): corpus.validate_prepared_data(rehash(data), reconstruct=False)
        fixture = corpus._fixture()
        altered = copy.deepcopy(fixture); altered["prose"][1]["text"] = altered["prose"][0]["text"].replace("twice", "2")
        # The fixture validator checks skeletons after numeric replacement; directly verify the invariant here.
        skeleton = lambda s: __import__('re').sub(r"\d+", "#", corpus.normalized(s).casefold())
        altered["prose"][0]["text"] = altered["prose"][0]["text"].replace("twice", "1")
        self.assertEqual(skeleton(altered["prose"][0]["text"]), skeleton(altered["prose"][1]["text"]))

    def test_translated_family_leakage_and_forged_partition_fail(self):
        data = self.changed()
        dev = next(x for x in data["teacherForced"]["documents"] if x["split"] == "development" and x["category"] == "multilingual")
        train = next(x for x in data["teacherForced"]["documents"] if x["split"] == "training" and x["category"] == "multilingual")
        train["partitionText"] = dev["partitionText"]; train["partitionNormalizedSHA256"] = dev["partitionNormalizedSHA256"]
        train["partitionBucket"] = dev["partitionBucket"]
        with self.assertRaises(EvidenceError): corpus.validate_prepared_data(rehash(data), reconstruct=False)
        data = self.changed(); doc = next(x for x in data["teacherForced"]["documents"] if x["split"] == "development")
        doc["partitionBucket"] = 1
        with self.assertRaises(EvidenceError): corpus.validate_prepared_data(rehash(data), reconstruct=False)

    def test_selector_order_count_and_category_forgery_fail(self):
        for mutation in ("selector", "count", "category"):
            data = self.changed()
            if mutation == "selector": data["finalScored"]["gsm8k"][0], data["finalScored"]["gsm8k"][1] = data["finalScored"]["gsm8k"][1], data["finalScored"]["gsm8k"][0]
            elif mutation == "count": data["finalScored"]["structured"].pop()
            else:
                data["teacherForced"]["documents"] = [x for x in data["teacherForced"]["documents"]
                                                        if not (x["split"] == "development" and x["category"] == "code")]
            with self.subTest(mutation=mutation), self.assertRaises(EvidenceError):
                corpus.validate_prepared_data(rehash(data), reconstruct=False)

    def test_short_window_offset_prompt_hash_and_qualification_denial(self):
        for mutation in ("short", "offset", "prompt", "qualification"):
            data = self.changed()
            if mutation == "short":
                doc = data["teacherForced"]["documents"][0]; doc["text"] = "x"; doc["rawSHA256"] = hashlib.sha256(b"x").hexdigest(); doc["normalizedSHA256"] = corpus.normalized_hash("x")
            elif mutation == "offset": data["teacherForced"]["documents"][0]["warmupTokens"] = 257
            elif mutation == "prompt": data["finalScored"]["xnli"][0]["prompt"] += " changed"
            else: data["finalScored"]["executionAvailable"] = True
            with self.subTest(mutation=mutation), self.assertRaises(EvidenceError):
                corpus.validate_prepared_data(rehash(data), reconstruct=False)

    def test_plan010_prompt_source_exactly_projects_all_1164_prompts(self):
        source = corpus.prompt_source(self.prepared)
        self.assertEqual(len(source["documents"]), 1164)
        self.assertEqual(source["documents"][0]["user"], self.prepared["finalScored"]["gsm8k"][0]["prompt"])
        self.assertEqual(sum(x["kind"]=="chat" for x in source["documents"]),1000)
        self.assertEqual(sum(x["kind"]=="raw" for x in source["documents"]),164)
        self.assertEqual(source["manifestSHA256"], corpus.canonical_hash({k:v for k,v in source.items() if k != "manifestSHA256"}))

    def test_prepare_refuses_overwrite_and_validate_detects_artifact_drift(self):
        with tempfile.TemporaryDirectory(dir=corpus.ROOT / ".build/flash") as temporary:
            target = Path(temporary) / "existing"; target.mkdir(); marker = target / "keep"; marker.write_text("keep")
            self.assertEqual(corpus.prepare(target), 1); self.assertEqual(marker.read_text(), "keep")
            prepared = Path(temporary) / "prepared"; self.assertEqual(corpus.prepare(prepared), 0)
            corpus.validate_prepared(prepared)
            completion = json.loads((prepared / "completion.json").read_text()); completion["suiteSHA256"] = "0" * 64
            (prepared / "completion.json").write_text(json.dumps(completion))
            with self.assertRaises(EvidenceError): corpus.validate_prepared(prepared)

    def test_freeze_refuses_existing_output_without_validating_dependencies(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "suite.json"; output.write_text("keep")
            self.assertEqual(corpus.freeze(Path("missing"), Path("missing"),Path("missing"),Path("missing"), output), 1)
            self.assertEqual(output.read_text(), "keep")

    def test_tokenizer_shard_mutation_propagates_as_failure(self):
        with mock.patch.object(corpus, "validate_prepared", return_value={"suiteSHA256":"a"*64,
                 "teacherCorpusSHA256":"b"*64,"scoredPromptsSHA256":"c"*64,"contractSHA256":"d"*64,
                 "counts":{},"finalCounts":{}}), mock.patch.object(corpus.tokenize_corpus, "validate_corpus",
                 side_effect=EvidenceError("tokenized shard changed")):
            with tempfile.TemporaryDirectory() as temporary:
                self.assertEqual(corpus.freeze(Path(temporary), Path(temporary),Path(temporary),Path(temporary), Path(temporary)/"new.json"), 1)

    def test_gsm_candidate_numeric_equivalence_and_errors(self):
        self.assertEqual(corpus.parse_gsm_candidate("answer: 2"),corpus.parse_gsm_candidate("answer: 2.0"))
        self.assertEqual(corpus.parse_gsm_candidate("result -1,250.50"),corpus.Decimal("-1250.50"))
        with self.assertRaises(EvidenceError):corpus.parse_gsm_candidate("no answer")

    def test_xnli_first_pair_comes_from_english_order(self):
        def row(language,pair,hyp):return {"promptID":"p","pairID":pair,"language":language,"sentence1":"premise","sentence2":hyp,"gold_label":"neutral"}
        rows=[row("ar","9","foreign"),row("en","4","first"),row("fr","4","f"),row("zh","4","z"),
              row("en","7","second"),row("fr","7","f"),row("zh","7","z"),row("en","9","third"),row("fr","9","f"),row("zh","9","z")]
        self.assertEqual(corpus._xnli_groups(rows)[0]["englishPairOrder"],["4","7","9"])

    def test_archived_producer_survives_current_harness_change_but_not_archive_mutation(self):
        with tempfile.TemporaryDirectory(dir=corpus.ROOT/".build/flash") as raw:
            snapshot=Path(raw)/"snapshot";self.assertEqual(corpus.producer_snapshot(snapshot),0)
            with mock.patch.object(corpus,"harness_hashes",return_value={"unrelated":"changed"}):corpus.validate_producer_snapshot(snapshot)
            target=snapshot/next(iter(json.loads((snapshot/"snapshot.json").read_text())["files"]))
            target.write_bytes(target.read_bytes()+b"x")
            with self.assertRaises(EvidenceError):corpus.validate_producer_snapshot(snapshot)

    def test_producer_snapshot_rejects_missing_extra_symlink_and_metadata_forgery(self):
        for mutation in ("missing","extra","symlink","nested-manifest","omit","bytes","hash"):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory(dir=corpus.ROOT/".build/flash") as raw:
                snapshot=Path(raw)/"snapshot";self.assertEqual(corpus.producer_snapshot(snapshot),0)
                document=json.loads((snapshot/"snapshot.json").read_text());name=next(iter(document["files"]));path=snapshot/name
                if mutation=="missing":path.unlink()
                elif mutation=="extra":(snapshot/"extra").write_text("x")
                elif mutation=="symlink":path.unlink();path.symlink_to(snapshot/"snapshot.json")
                elif mutation=="nested-manifest":nested=snapshot/"nested";nested.mkdir();(nested/"snapshot.json").write_text("extra")
                elif mutation=="omit":del document["files"][name]
                elif mutation=="bytes":document["files"][name]["bytes"]+=1
                else:document["files"][name]["sha256"]="0"*64
                if mutation in {"omit","bytes","hash"}:(snapshot/"snapshot.json").write_text(json.dumps(document))
                with self.assertRaises(EvidenceError):corpus.validate_producer_snapshot(snapshot)


if __name__ == "__main__": unittest.main(verbosity=2)
