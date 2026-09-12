#!/usr/bin/env python3
import copy, hashlib, json, math, sys, tempfile, unittest
from pathlib import Path
from unittest import mock
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import metrics
from common import EvidenceError, atomic_json, sha256


class MetricsTests(unittest.TestCase):
    def setUp(self): self.old_vocab=metrics.VOCAB; metrics.VOCAB=3
    def tearDown(self): metrics.VOCAB=self.old_vocab

    def test_identical_shift_and_three_class_scalar(self):
        a=np.array([1.,2.,3.],np.float32)
        self.assertEqual(metrics.compare_row(a,a.copy(),2)["kl"],0)
        self.assertAlmostEqual(metrics.compare_row(a,a+100,2)["kl"],0,places=6)
        p=np.array([.6,.3,.1]); q=np.array([.2,.5,.3])
        actual=metrics.compare_row(np.log(p).astype(np.float32),np.log(q).astype(np.float32),1)
        self.assertAlmostEqual(actual["kl"],sum(float(x)*math.log(float(x/y)) for x,y in zip(p,q)),places=6)
        self.assertAlmostEqual(actual["referenceNLL"],-math.log(.3),places=6)

    def test_binary_direction_reverse_and_correct_ppl_ratio(self):
        metrics.VOCAB=2
        p=np.array([.75,.25]); q=np.array([.5,.5])
        forward=metrics.compare_row(np.log(p).astype(np.float32),np.log(q).astype(np.float32),0)
        reverse=metrics.compare_row(np.log(q).astype(np.float32),np.log(p).astype(np.float32),0)
        self.assertAlmostEqual(forward["kl"],.13081203594113697,places=6)
        self.assertNotAlmostEqual(forward["kl"],reverse["kl"],places=5)
        self.assertAlmostEqual(math.exp(forward["candidateNLL"]-forward["referenceNLL"]),1.5,places=6)

    def test_token_weighted_aggregate_unequal_documents(self):
        rows=[]
        for doc,values in (("long",[0.,0.,0.]),("short",[4.])):
            rows += [{"documentID":doc,"kl":v,"rawKL":v,"referenceNLL":1.,"candidateNLL":1.2,"top1":1} for v in values]
        result=metrics.aggregate(rows)
        self.assertEqual(result["meanKL"],1.)
        self.assertAlmostEqual(result["pplRatio"],math.exp(.2),places=12)
        self.assertNotEqual(result["meanKL"],2.) # an unweighted document mean

    def test_nearest_rank_ties_and_bounds(self):
        self.assertEqual(metrics.nearest_rank([0,1,1,3],.5),1)
        self.assertEqual(metrics.nearest_rank([0,1,1,3],.99),3)
        with self.assertRaises(EvidenceError): metrics.nearest_rank([])

    def test_cluster_bootstrap_uses_whole_documents_with_known_draws(self):
        rows=[{"documentID":"a","kl":0.},{"documentID":"a","kl":0.},{"documentID":"b","kl":1.}]
        class Fixed:
            def __init__(self): self.draws=iter((np.array([0,0]),np.array([0,1]),np.array([1,0]),np.array([1,1])))
            def integers(self,*args,**kwargs): return next(self.draws)
        with mock.patch.object(metrics.np.random,"default_rng",return_value=Fixed()):
            value=metrics.bootstrap_documents(rows,resamples=4,seed=17)
        self.assertEqual(value["low"],0.)
        self.assertEqual(value["high"],1.)
        self.assertEqual(value["documents"],2)

    def test_nonfinite_overflow_underflow_and_extreme_finite_failures(self):
        with self.assertRaises(EvidenceError): metrics.compare_row(np.array([0,np.nan,1]),np.zeros(3),0)
        extreme=np.array([np.finfo(np.float32).max,-np.finfo(np.float32).max,0],np.float32)
        with self.assertRaises(EvidenceError): metrics.compare_row(extreme,extreme.copy(),0)
        under=metrics.compare_row(np.array([0,-1000,-1000],np.float32),np.array([0,-900,-800],np.float32),0)
        self.assertTrue(math.isfinite(under["kl"]))
        rows=[{"documentID":"a","kl":0.,"rawKL":0.,"referenceNLL":0.,"candidateNLL":1e308,"top1":1}]
        with self.assertRaises(EvidenceError): metrics.aggregate(rows)

    def test_bad_target_vocab_and_negative_roundoff_policy(self):
        with self.assertRaises(EvidenceError): metrics.compare_row(np.zeros(2),np.zeros(2),0)
        with self.assertRaises(EvidenceError): metrics.compare_row(np.zeros(3),np.zeros(3),3)
        tolerated=[{"documentID":"a","kl":0.,"rawKL":-1e-8,"referenceNLL":1.,"candidateNLL":1.,"top1":1}]
        value=metrics.aggregate(tolerated)
        self.assertEqual(value["toleratedNegativeRows"],1); self.assertEqual(value["minimumRawKL"],-1e-8)
        bad=copy.deepcopy(tolerated); bad[0]["rawKL"]=-1e-4
        with self.assertRaises(EvidenceError): metrics.aggregate(bad)

    def _cohort_fixture(self, base: Path):
        tokenized=base/"tokenized"; (tokenized/"corpus/shards").mkdir(parents=True)
        shard_path=tokenized/"corpus/shards/shard-001.json"; shard_path.write_text("{}")
        (tokenized/"corpus/corpus.json").write_text("{}")
        (tokenized/"corpus/completion.json").write_text("{}")
        request={"format":"slotstream-tokenized-capture-shard-v2","split":"development","shardID":"shard-001",
                 "documents":[{"id":"doc","category":"prose","positions":[{"position":2,"inputID":1,"nextTokenID":2}]}]}
        corpus_index={"source_manifest_sha256":"a"*64,"tokenizer_identity":{"pin":"x"},
                      "shards":[{"path":"shards/shard-001.json","split":"development","positions":1,"sha256":sha256(shard_path)}]}
        capture_root=base/"capture"; native=capture_root/"native"; native.mkdir(parents=True)
        cohort=base/"cohort";cohort.mkdir()
        binary=base/"bin"; binary.write_bytes(b"binary")
        siblings={"mlx.metallib":b"metal","build-identity.json":b"identity","build-source.tar.gz":b"source","build-source-before.json":b"before"}
        for name,value in siblings.items(): (base/name).write_bytes(value)
        model=base/"model"; model.mkdir()
        logits=native/"logits-doc-2.bin"; np.array([0.,1.,2.],dtype="<f4").tofile(logits)
        report={"mode":"reference-off","binary":str(binary),"model":str(model),
                "source_identity":{"binary_sha256":sha256(binary),"source_archive_sha256":sha256(base/"build-source.tar.gz"),
                                   "model_config_sha256":"c"*64,"model_index_sha256":"d"*64,
                                   "build_identity_sha256":sha256(base/"build-identity.json"),"metallib_sha256":sha256(base/"mlx.metallib")},
                "documents":[{"id":"doc","positions":[{"position":2,"input_id":1,"next_token_id":2}]}],
                "plan":{"target_gb":14},"optimizations":{},"memory_ledger":{},"numerical_environment":{},
                "files":[{"category":"logits","document_id":"doc","token_position":2,"input_id":1,
                          "name":logits.name,"bytes":12,"sha256":sha256(logits)}]}
        atomic_json(native/"report.json",report); atomic_json(native/"completion.json",{"ok":True})
        atomic_json(capture_root/"receipt.json",{"ok":True}); atomic_json(capture_root/"completion.json",{"ok":True})
        harness_map={"Tools/flash/fake.py":hashlib.sha256(b"fixture").hexdigest()}
        receipt={"command":[str(binary),"flash-capture","--model",str(model),"--manifest",str(shard_path),
                            "--split","development","--mode","reference-off","--output",str(native)],
                 "result":{"functional_success":True},"identities":{"executable":{"sha256":sha256(binary),"bytes":binary.stat().st_size},"harness_hashes":harness_map}}
        hroot=cohort/"harness"/metrics.canonical_hash(harness_map);(hroot/"Tools/flash").mkdir(parents=True);(hroot/"Tools/flash/fake.py").write_bytes(b"fixture")
        atomic_json(hroot/"snapshot.json",{"format":"slotstream-execution-harness-snapshot-v1","launcherHarnessSHA256":metrics.canonical_hash(harness_map),"files":{"Tools/flash/fake.py":{"bytes":7,"sha256":harness_map["Tools/flash/fake.py"]}}})
        harness={"path":metrics._under_root(hroot),"snapshotSHA256":sha256(hroot/"snapshot.json")}
        with mock.patch.object(metrics.capture,"validate_native_output"), \
             mock.patch.object(metrics,"validate_receipt_file",return_value=receipt), \
             mock.patch.object(metrics,"require_terminal_sampling"):
            capture_entry=metrics._capture_entry(capture_root,shard_path,request,None,harness)
        index={"format":"slotstream-quality-cohort-v1","schemaVersion":1,"qualification":False,
               "kind":"diagnostic-control","split":"development","suite":None,
               "tokenized":{"path":metrics._under_root(tokenized),"corpusSHA256":sha256(tokenized/"corpus/corpus.json"),
                            "completionSHA256":sha256(tokenized/"corpus/completion.json"),
                            "sourceManifestSHA256":"a"*64,"tokenizerIdentitySHA256":metrics.canonical_hash({"pin":"x"})},
               "expectedCategories":["prose"],"shards":[{"ordinal":0,"path":"shards/shard-001.json",
               "sha256":sha256(shard_path),"documents":[{"id":"doc","category":"prose","positions":1}],
               "positions":1,"capture":capture_entry}],"positions":1}
        build=cohort/"build"/sha256(binary);build.mkdir(parents=True)
        import shutil
        for name in ("slotstream","mlx.metallib","build-identity.json","build-source.tar.gz","build-source-before.json"):
            shutil.copy2(binary if name=="slotstream" else base/name,build/name)
        capture_entry["immutableBuildPath"]=metrics._under_root(build)
        index["manifestSHA256"]=metrics.canonical_hash(index)
        atomic_json(cohort/"index.json",index)
        atomic_json(cohort/"completion.json",{"format":"slotstream-quality-cohort-completion-v1",
                    "indexSHA256":sha256(cohort/"index.json"),"qualification":False})
        return cohort,index,request,corpus_index,receipt,report

    def _validate_fixture(self, cohort,index,request,corpus_index,receipt):
        with mock.patch.object(metrics.tokenize_corpus,"validate_corpus",return_value={"index":corpus_index}), \
             mock.patch.object(metrics.capture,"validate_manifest",return_value=request), \
             mock.patch.object(metrics.capture,"validate_native_output"), \
             mock.patch.object(metrics,"validate_build_identity"), \
             mock.patch.object(metrics,"validate_receipt_file",return_value=receipt), \
             mock.patch.object(metrics,"require_terminal_sampling"):
            return metrics.validate_cohort(cohort)

    def test_cohort_rejects_duplicate_position_and_report_mutation(self):
        with tempfile.TemporaryDirectory(dir=metrics.ROOT/".build/flash") as raw:
            base=Path(raw); cohort,index,request,corpus_index,receipt,report=self._cohort_fixture(base)
            self._validate_fixture(cohort,index,request,corpus_index,receipt)
            report["documents"][0]["positions"].append(copy.deepcopy(report["documents"][0]["positions"][0]))
            atomic_json(base/"capture/native/report.json",report)
            index["shards"][0]["capture"]["reportSHA256"]=sha256(base/"capture/native/report.json")
            index["manifestSHA256"]=metrics.canonical_hash({k:v for k,v in index.items() if k!="manifestSHA256"})
            atomic_json(cohort/"index.json",index); atomic_json(cohort/"completion.json",{"format":"slotstream-quality-cohort-completion-v1","indexSHA256":sha256(cohort/"index.json"),"qualification":False})
            with self.assertRaises(EvidenceError): self._validate_fixture(cohort,index,request,corpus_index,receipt)

    def test_archived_cohort_survives_old_live_binary_removal_and_rejects_receipt_identity(self):
        with tempfile.TemporaryDirectory(dir=metrics.ROOT/".build/flash") as raw:
            base=Path(raw);cohort,index,request,corpus_index,receipt,report=self._cohort_fixture(base)
            (base/"bin").unlink();self._validate_fixture(cohort,index,request,corpus_index,receipt)
            for field,value in (("sha256","0"*64),("bytes",999)):
                changed=copy.deepcopy(receipt);changed["identities"]["executable"][field]=value
                with self.assertRaises(EvidenceError):self._validate_fixture(cohort,index,request,corpus_index,changed)

    def test_execution_snapshot_rejects_nested_manifest_name(self):
        with tempfile.TemporaryDirectory(dir=metrics.ROOT/".build/flash") as raw:
            base=Path(raw);cohort,index,request,corpus_index,receipt,report=self._cohort_fixture(base)
            hroot=metrics.ROOT/index["shards"][0]["capture"]["harnessSnapshot"]["path"]
            (hroot/"nested").mkdir();(hroot/"nested/snapshot.json").write_text("extra")
            with self.assertRaises(EvidenceError):self._validate_fixture(cohort,index,request,corpus_index,receipt)

    def test_cohort_rejects_receipt_source_archive_shard_and_missing_logits_mutations(self):
        for mutation in ("receipt","archive","shard","missing"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory(dir=metrics.ROOT/".build/flash") as raw:
                base=Path(raw); cohort,index,request,corpus_index,receipt,report=self._cohort_fixture(base)
                if mutation=="receipt": receipt=copy.deepcopy(receipt); receipt["command"][receipt["command"].index("--split")+1]="training"
                elif mutation=="archive": index["shards"][0]["capture"]["sourceIdentity"]["source_archive_sha256"]="0"*64
                elif mutation=="shard": index["shards"][0]["sha256"]="0"*64
                else: report["files"]=[]; atomic_json(base/"capture/native/report.json",report); index["shards"][0]["capture"]["reportSHA256"]=sha256(base/"capture/native/report.json")
                index["manifestSHA256"]=metrics.canonical_hash({k:v for k,v in index.items() if k!="manifestSHA256"})
                atomic_json(cohort/"index.json",index); atomic_json(cohort/"completion.json",{"format":"slotstream-quality-cohort-completion-v1","indexSHA256":sha256(cohort/"index.json"),"qualification":False})
                with self.assertRaises(EvidenceError): self._validate_fixture(cohort,index,request,corpus_index,receipt)

    def test_compare_rejects_cross_split_cross_model_and_unapproved_identity_difference(self):
        base={"kind":"diagnostic-control","split":"development","suite":None,"tokenized":{"corpusSHA256":"a"},
              "expectedCategories":["prose"],"positions":1,"shards":[{"capture":{"mode":"reference-off",
              "sourceIdentity":{"binary_sha256":"b","source_archive_sha256":"c"}}}]}
        row={(0,"doc",2):{"path":Path("unused"),"inputID":1,"nextTokenID":2,"category":"prose","documentID":"doc","position":2}}
        for mutation in ("split","model","mode"):
            other=copy.deepcopy(base)
            if mutation=="split": other["split"]="training"
            elif mutation=="model": other["tokenized"]={"corpusSHA256":"z"}
            else: other["shards"][0]["capture"]["mode"]="reference-on"
            with tempfile.TemporaryDirectory(dir=metrics.ROOT/".build/flash") as raw, \
                 mock.patch.object(metrics,"validate_cohort",side_effect=[(base,row),(other,row)]):
                self.assertEqual(metrics.compare_cohorts(Path("a"),Path("b"),Path(raw)/"out",set(),4),1)

    def test_preexisting_output_is_never_modified(self):
        with tempfile.TemporaryDirectory(dir=metrics.ROOT/".build/flash") as raw:
            output=Path(raw)/"existing";output.mkdir();marker=output/"keep";marker.write_text("keep")
            self.assertEqual(metrics.freeze_cohort(Path("x"),"development","diagnostic-control",[],output),1)
            self.assertEqual(metrics.compare_cohorts(Path("x"),Path("y"),output,set(),4),1)
            self.assertEqual(marker.read_text(),"keep");self.assertEqual({x.name for x in output.iterdir()},{"keep"})

    def test_capture_cohort_denies_qualification_before_preflight(self):
        with mock.patch.object(metrics.benchmark,"preflight") as preflight:
            self.assertEqual(metrics.capture_cohort(Path("x"),Path("y"),"reference-off","qualification",Path("z")),1)
            preflight.assert_not_called()

    def test_capture_cohort_midstream_failure_and_cancellation_are_incomplete(self):
        suite={"tokenized":{"path":"fake"}};index={"shards":[{"split":"development","path":"shards/a"},{"split":"development","path":"shards/b"}]}
        for error,code in ((EvidenceError("failed shard"),1),(KeyboardInterrupt(),130)):
            with self.subTest(code=code),tempfile.TemporaryDirectory(dir=metrics.ROOT/".build/flash") as raw,\
                 mock.patch.object(metrics.corpus,"validate_manifest",return_value=suite),\
                 mock.patch.object(metrics.tokenize_corpus,"validate_corpus",return_value={"index":index}),\
                 mock.patch.object(metrics.benchmark,"preflight"),\
                 mock.patch.object(metrics.capture,"_native",side_effect=[({}, {}, {}),error]):
                output=Path(raw)/"capture";self.assertEqual(metrics.capture_cohort(Path("suite"),Path("binary"),"reference-off","development",output),code)
                self.assertTrue((output/"failure.json").is_file());self.assertFalse((output/"completion.json").exists())


if __name__=="__main__": unittest.main(verbosity=2)
