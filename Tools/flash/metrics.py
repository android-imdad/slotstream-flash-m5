#!/usr/bin/env python3
"""Freeze complete capture cohorts and compare aligned full-vocabulary logits."""
from __future__ import annotations

import argparse, copy, hashlib, json, math, os, shutil, subprocess, sys, tarfile
from pathlib import Path
from typing import Any

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from common import EvidenceError, atomic_json, fresh_output, harness_hashes, read_json, sha256, validate_build_identity
from receipts import require_terminal_sampling, validate_receipt_file
import capture, corpus, tokenize_corpus
import benchmark

VOCAB = 248320
CHUNK_ROWS = 16
NEGATIVE_KL_BAND = -5e-7
APPROX = {"meanKL":1e-3,"p99KL":1e-2,"top1Agreement":.99,"pplRatio":1.01}
NUMERICAL = {"meanKL":1e-4,"p99KL":1e-3,"top1Agreement":.999,"pplRatio":1.01}
IDENTITY_DIFFERENCES = {"mode", "binarySHA256", "sourceArchiveSHA256",
                        "buildIdentitySHA256", "metallibSHA256","executionHarnessSHA256","runtimeControls"}


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                        ensure_ascii=False).encode()).hexdigest()


def log_softmax32(row: np.ndarray) -> np.ndarray:
    value = np.asarray(row, dtype=np.float32)
    if value.ndim != 1 or not np.isfinite(value).all(): raise EvidenceError("nonfinite or non-vector logits")
    maximum = np.max(value).astype(np.float32)
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        shifted = np.subtract(value, maximum, dtype=np.float32)
        exponentials = np.exp(shifted, dtype=np.float32)
        total = np.sum(exponentials, dtype=np.float32)
        logarithm = np.log(total, dtype=np.float32)
        result = np.subtract(shifted, logarithm, dtype=np.float32)
    if not np.isfinite(shifted).all() or not np.isfinite(total) or total <= 0 or not np.isfinite(result).all():
        raise EvidenceError("nonfinite log-softmax intermediate")
    return result


def compare_row(reference: np.ndarray, candidate: np.ndarray, target: int) -> dict[str, Any]:
    reference, candidate = np.asarray(reference), np.asarray(candidate)
    if reference.shape != (VOCAB,) or candidate.shape != (VOCAB,) or type(target) is not int or not 0 <= target < VOCAB:
        raise EvidenceError("wrong vocabulary or target")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all(): raise EvidenceError("nonfinite logits")
    lp, lq = log_softmax32(reference), log_softmax32(candidate)
    with np.errstate(under="ignore", over="ignore", invalid="ignore"):
        probabilities = np.exp(lp, dtype=np.float32).astype(np.float64)
    if not np.isfinite(probabilities).all(): raise EvidenceError("nonfinite reference probabilities")
    active = probabilities > 0
    delta = lp[active].astype(np.float64) - lq[active].astype(np.float64)
    terms = probabilities[active] * delta
    if not np.isfinite(delta).all() or not np.isfinite(terms).all(): raise EvidenceError("nonfinite KL intermediate")
    raw = float(np.sum(terms, dtype=np.float64))
    rnll, cnll = -float(lp[target]), -float(lq[target])
    if not all(math.isfinite(x) for x in (raw, rnll, cnll)): raise EvidenceError("nonfinite metric result")
    if raw < NEGATIVE_KL_BAND: raise EvidenceError("materially negative KL")
    return {"rawKL":raw, "kl":0.0 if raw < 0 else raw, "referenceNLL":rnll,
            "candidateNLL":cnll, "top1":int(np.argmax(reference) == np.argmax(candidate))}


def nearest_rank(values: list[float], quantile: float = .99) -> float:
    if not values or not 0 < quantile <= 1 or any(not math.isfinite(x) for x in values):
        raise EvidenceError("invalid nearest-rank input")
    return sorted(values)[math.ceil(quantile * len(values)) - 1]


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows: raise EvidenceError("empty aligned rows")
    fields = ("kl", "rawKL", "referenceNLL", "candidateNLL")
    if any(any(not math.isfinite(float(row[key])) for key in fields) for row in rows):
        raise EvidenceError("nonfinite aggregate input")
    if any(row["rawKL"] < NEGATIVE_KL_BAND for row in rows):
        raise EvidenceError("materially negative aggregate KL")
    count = len(rows)
    difference = (math.fsum(row["candidateNLL"] for row in rows) -
                  math.fsum(row["referenceNLL"] for row in rows)) / count
    try: ratio = math.exp(difference)
    except OverflowError as error: raise EvidenceError("perplexity ratio overflow") from error
    if not math.isfinite(ratio): raise EvidenceError("nonfinite perplexity ratio")
    return {"tokens":count, "meanKL":math.fsum(row["kl"] for row in rows) / count,
            "p99KL":nearest_rank([row["kl"] for row in rows]),
            "minimumRawKL":min(row["rawKL"] for row in rows),
            "toleratedNegativeRows":sum(row["rawKL"] < 0 for row in rows),
            "top1Agreement":math.fsum(row["top1"] for row in rows) / count,
            "referenceMeanNLL":math.fsum(row["referenceNLL"] for row in rows) / count,
            "candidateMeanNLL":math.fsum(row["candidateNLL"] for row in rows) / count,
            "pplRatio":ratio}


def bootstrap_documents(rows: list[dict[str, Any]], resamples: int = 10000, seed: int = 17):
    if type(resamples) is not int or resamples < 1: raise EvidenceError("bootstrap resample count is invalid")
    groups = {}
    for row in rows:
        name = row.get("documentID")
        if not isinstance(name, str) or not name: raise EvidenceError("bootstrap document identity is absent")
        groups.setdefault(name, []).append(float(row["kl"]))
    names = sorted(groups)
    if not names: raise EvidenceError("bootstrap has no documents")
    rng, samples = np.random.default_rng(seed), []
    for _ in range(resamples):
        chosen = rng.integers(0, len(names), size=len(names))
        values = [value for index in chosen for value in groups[names[int(index)]]]
        samples.append(math.fsum(values) / len(values))
    return {"seed":seed,"resamples":resamples,"documents":len(names),
            "low":nearest_rank(samples,.025),"high":nearest_rank(samples,.975),
            "limits":{"unit":"document cluster","statistic":"token-weighted mean KL within resample"},
            "limitation":"fixed document-cluster interval; no population non-inferiority claim"}


def _under_root(path: Path) -> str:
    value = path.resolve(strict=True)
    if not value.is_relative_to(ROOT.resolve()): raise EvidenceError("cohort path escapes repository")
    return str(value.relative_to(ROOT.resolve()))


def _model_path(model: Path) -> str:
    if model.resolve(strict=True) == (ROOT/"models/jang-6s").resolve(strict=True):
        return "models/jang-6s"
    return _under_root(model)


def _flag(command: list[str], name: str) -> str:
    if command.count(name) != 1: raise EvidenceError(f"capture command lacks one {name}")
    index = command.index(name)
    if index + 1 >= len(command): raise EvidenceError(f"capture command has no value for {name}")
    return command[index + 1]


def _capture_binary(report: dict[str, Any], receipt: dict[str, Any]) -> Path:
    source = report["source_identity"]
    recorded = receipt.get("identities", {}).get("executable", {})
    if recorded.get("sha256") != source["binary_sha256"]:
        raise EvidenceError("launcher receipt executable does not bind native source identity")
    candidates = [Path(report.get("binary", "")),
                  ROOT/".build/flash/runs/diagnostic-reference-corpus-v2/bin/slotstream"]
    for binary in candidates:
        if not binary.is_file() or sha256(binary) != source["binary_sha256"]: continue
        directory = binary.parent
        identities = (("mlx.metallib","metallib_sha256"),("build-identity.json","build_identity_sha256"),
                      ("build-source.tar.gz","source_archive_sha256"))
        if all((directory/name).is_file() and sha256(directory/name) == source[key] for name,key in identities):
            return binary
    raise EvidenceError("matching immutable capture binary and source archive are unavailable")


def _runtime_identity(report):
    plan=copy.deepcopy(report["plan"])
    for key in ("device_available_gb","device_ram_gb","device_working_set_gb"):plan.pop(key,None)
    return {"plan":plan,"optimizations":report["optimizations"],"memoryLedger":report["memory_ledger"],
            "numericalEnvironment":report["numerical_environment"]}


def _capture_entry(capture_root: Path, shard_path: Path, request: dict[str, Any], immutable_binary:Path|None=None,
                   harness_snapshot:dict[str,Any]|None=None) -> dict[str, Any]:
    root = capture_root.resolve(strict=True)
    native = root / "native"
    report = read_json(native / "report.json")
    mode = report.get("mode")
    if mode not in {"reference-off", "reference-on"}: raise EvidenceError("unsupported capture mode")
    receipt = validate_receipt_file(root / "receipt.json"); require_terminal_sampling(receipt)
    binary, model = (immutable_binary or _capture_binary(report,receipt)), Path(report.get("model", ""))
    recorded=receipt.get("identities",{}).get("executable",{})
    if recorded.get("sha256")!=sha256(binary) or recorded.get("bytes")!=binary.stat().st_size or recorded.get("sha256")!=report["source_identity"]["binary_sha256"]:
        raise EvidenceError("launcher executable identity does not bind immutable binary")
    capture.validate_native_output(native, shard_path, mode, binary, model, request)
    command = receipt["command"]
    if len(command) < 2 or command[1] != "flash-capture" or command[0] != report["binary"]:
        raise EvidenceError("launcher command does not bind capture binary")
    if Path(_flag(command,"--model")).resolve() != model.resolve() or Path(_flag(command,"--manifest")).resolve() != shard_path.resolve() or _flag(command,"--split") != request["split"] or _flag(command,"--mode") != mode or Path(_flag(command,"--output")).resolve() != native.resolve():
        raise EvidenceError("launcher command does not bind model, shard, split, mode, and output")
    if receipt["result"]["functional_success"] is not True:
        raise EvidenceError("capture launcher was not functionally successful")
    return {"capturePath":_under_root(root), "mode":mode,
            "reportSHA256":sha256(native/"report.json"), "nativeCompletionSHA256":sha256(native/"completion.json"),
            "receiptSHA256":sha256(root/"receipt.json"), "launcherCompletionSHA256":sha256(root/"completion.json"),
            "launcherHarnessSHA256":canonical_hash(receipt.get("identities",{}).get("harness_hashes",{})),
            "harnessSnapshot":harness_snapshot,"runtimeIdentity":_runtime_identity(report),
            "immutableBuildPath":_under_root(binary.parent),"launcherExecutable":{"bytes":recorded["bytes"],"sha256":recorded["sha256"]},
            "sourceIdentity":report["source_identity"], "model":_model_path(model),
            "documents":len(request["documents"]), "positions":sum(len(x["positions"]) for x in request["documents"])}


def _archive_support(capture_root:Path, output:Path):
    report=read_json(capture_root/"native/report.json");receipt=validate_receipt_file(capture_root/"receipt.json")
    binary=_capture_binary(report,receipt);digest=sha256(binary);build_root=output/"build"/digest
    if not build_root.exists():
        build_root.mkdir(parents=True)
        for name in ("slotstream","mlx.metallib","build-identity.json","build-source.tar.gz","build-source-before.json"):
            shutil.copy2(binary.parent/name,build_root/name)
    validate_build_identity(build_root/"slotstream",historical=True)
    expected=receipt["identities"]["harness_hashes"];hroot=output/"harness"/canonical_hash(expected)
    if not hroot.exists():
        hroot.mkdir(parents=True)
        for name,digest_value in expected.items():
            current=ROOT/name;data=current.read_bytes() if current.is_file() else None
            if data is None or hashlib.sha256(data).hexdigest()!=digest_value:
                historical=subprocess.run(["git","show",f"HEAD:{name}"],cwd=ROOT,capture_output=True,timeout=10)
                data=historical.stdout if historical.returncode==0 else None
            if data is None or hashlib.sha256(data).hexdigest()!=digest_value:raise EvidenceError("execution harness bytes cannot be reconstructed exactly")
            path=hroot/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        identities={name:{"bytes":len((hroot/name).read_bytes()),"sha256":digest} for name,digest in expected.items()}
        atomic_json(hroot/"snapshot.json",{"format":"slotstream-execution-harness-snapshot-v1","launcherHarnessSHA256":canonical_hash(expected),"files":identities})
    snapshot=read_json(hroot/"snapshot.json")
    _validate_snapshot(hroot,snapshot,"slotstream-execution-harness-snapshot-v1",set(expected),canonical_hash(expected))
    return build_root/"slotstream",{"path":_under_root(hroot),"snapshotSHA256":sha256(hroot/"snapshot.json")}


def _validate_snapshot(root:Path,value:dict[str,Any],format_name:str,expected:set[str],owner_hash:str):
    if root.is_symlink() or (root/"snapshot.json").is_symlink():raise EvidenceError("symlinked snapshot root or manifest is not accepted")
    root_manifest=(root/"snapshot.json").resolve(strict=True)
    if set(value)!={"format","launcherHarnessSHA256","files"} or value["format"]!=format_name or value["launcherHarnessSHA256"]!=owner_hash or set(value["files"])!=expected or not 0<len(expected)<=256:
        raise EvidenceError("archived snapshot metadata differs")
    actual=[]
    for path in root.rglob("*"):
        if path==root/"snapshot.json":continue
        if path.is_symlink() or (path.is_file() and not path.resolve().is_relative_to(root.resolve())):raise EvidenceError("unsafe archived snapshot artifact")
        if path.is_file():actual.append(str(path.relative_to(root)))
    if set(actual)!=expected:raise EvidenceError("archived snapshot inventory differs")
    for name,identity in value["files"].items():
        relative=Path(name)
        if relative.is_absolute() or ".." in relative.parts or set(identity)!={"bytes","sha256"}:raise EvidenceError("archived snapshot entry is malformed")
        path=root/relative
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()) or type(identity["bytes"]) is not int or identity["bytes"]<1 or not isinstance(identity["sha256"],str) or len(identity["sha256"])!=64 or path.stat().st_size!=identity["bytes"] or sha256(path)!=identity["sha256"]:raise EvidenceError("archived snapshot bytes differ")


def freeze_cohort(tokenized: Path, split: str, kind: str, captures: list[Path], output: Path,
                  suite: Path | None = None) -> int:
    owned=False
    try:
        if split=="qualification":raise EvidenceError("qualification inference is locked")
        if split not in {"training","development"}:raise EvidenceError("unsupported cohort split")
        output = fresh_output(output);owned=True
        tokenized_result = tokenize_corpus.validate_corpus(tokenized)
        index = tokenized_result["index"]
        expected = [x for x in index["shards"] if x["split"] == split]
        if not expected or len(expected) != len(captures): raise EvidenceError("capture list does not cover every expected split shard")
        if kind == "full-study":
            if suite is None: raise EvidenceError("full-study cohort requires a suite manifest")
            suite_value = corpus.validate_manifest(suite)
            if (ROOT/suite_value["tokenized"]["path"]).resolve() != tokenized.resolve():
                raise EvidenceError("full-study suite points to a different tokenized corpus")
            suite_identity = {"path":_under_root(suite),"sha256":sha256(suite)}
        elif kind == "diagnostic-control":
            if suite is not None: raise EvidenceError("diagnostic control cannot claim a full-study suite")
            suite_identity = None
        else: raise EvidenceError("unknown cohort kind")
        shards = []
        for entry, capture_root in zip(expected, captures):
            shard_path = tokenized.resolve()/"corpus"/entry["path"]
            request = capture.validate_manifest(shard_path)
            immutable,harness=_archive_support(capture_root.resolve(),output)
            shards.append({"ordinal":len(shards),"path":entry["path"],"sha256":entry["sha256"],
                           "documents":[{"id":x["id"],"category":x["category"],
                                         "positions":len(x["positions"])} for x in request["documents"]],
                           "positions":entry["positions"],"capture":_capture_entry(capture_root,shard_path,request,immutable,harness)})
        categories = sorted({x["category"] for shard in shards for x in shard["documents"]})
        capture_identities = {(x["capture"]["mode"], x["capture"]["model"],
                               canonical_hash(x["capture"]["sourceIdentity"]),
                               canonical_hash(x["capture"]["runtimeIdentity"]),
                               x["capture"]["launcherHarnessSHA256"]) for x in shards}
        if len(capture_identities) != 1:
            raise EvidenceError("cohort shards cross capture modes, models, or build identities")
        cohort = {"format":"slotstream-quality-cohort-v1","schemaVersion":1,"qualification":False,
                  "kind":kind,"split":split,"suite":suite_identity,"tokenized":{"path":_under_root(tokenized),
                  "corpusSHA256":sha256(tokenized.resolve()/"corpus/corpus.json"),
                  "completionSHA256":sha256(tokenized.resolve()/"corpus/completion.json"),
                  "sourceManifestSHA256":index["source_manifest_sha256"],
                  "tokenizerIdentitySHA256":canonical_hash(index["tokenizer_identity"])},
                  "expectedCategories":categories,"shards":shards,
                  "positions":sum(x["positions"] for x in shards)}
        cohort["manifestSHA256"] = canonical_hash(cohort)
        atomic_json(output/"index.json",cohort)
        atomic_json(output/"completion.json",{"format":"slotstream-quality-cohort-completion-v1",
                    "indexSHA256":sha256(output/"index.json"),"qualification":False})
        return 0
    except Exception as error:
        if owned:
            atomic_json(output/"failure.json",{"format":"slotstream-quality-cohort-failure-v1","error":f"{type(error).__name__}: {error}"})
        return 1


def validate_cohort(root: Path):
    root = root.resolve(strict=True); index_path = root/"index.json"; value = read_json(index_path)
    required = {"format","schemaVersion","qualification","kind","split","suite","tokenized",
                "expectedCategories","shards","positions","manifestSHA256"}
    if set(value) != required or value.get("format") != "slotstream-quality-cohort-v1" or value.get("schemaVersion") != 1 or value.get("qualification") is not False:
        raise EvidenceError("cohort index is malformed")
    if value["manifestSHA256"] != canonical_hash({k:v for k,v in value.items() if k != "manifestSHA256"}):
        raise EvidenceError("cohort manifest hash differs")
    completion = read_json(root/"completion.json")
    if completion != {"format":"slotstream-quality-cohort-completion-v1","indexSHA256":sha256(index_path),"qualification":False}:
        raise EvidenceError("cohort completion does not bind index")
    tokenized = ROOT/value["tokenized"]["path"]; result = tokenize_corpus.validate_corpus(tokenized)
    corpus_index = result["index"]
    if value["tokenized"] != {"path":value["tokenized"]["path"],"corpusSHA256":sha256(tokenized/"corpus/corpus.json"),
                              "completionSHA256":sha256(tokenized/"corpus/completion.json"),
                              "sourceManifestSHA256":corpus_index["source_manifest_sha256"],
                              "tokenizerIdentitySHA256":canonical_hash(corpus_index["tokenizer_identity"])}:
        raise EvidenceError("cohort tokenized identity differs")
    if value["kind"] == "full-study":
        if not isinstance(value["suite"],dict): raise EvidenceError("full-study suite identity is absent")
        suite = ROOT/value["suite"]["path"]
        if value["suite"] != {"path":value["suite"]["path"],"sha256":sha256(suite)}: raise EvidenceError("suite identity differs")
        suite_value = corpus.validate_manifest(suite)
        if (ROOT/suite_value["tokenized"]["path"]).resolve()!=tokenized.resolve():raise EvidenceError("full-study suite points to another tokenized corpus")
        expected_counts = suite_value["counts"][value["split"]]
        if value["expectedCategories"] != sorted(corpus.CATEGORIES) or value["positions"] != sum(x["positions"] for x in expected_counts.values()):
            raise EvidenceError("full-study cohort does not cover the frozen suite split")
    elif value["kind"] != "diagnostic-control" or value["suite"] is not None:
        raise EvidenceError("cohort kind and suite identity conflict")
    expected = [x for x in corpus_index["shards"] if x["split"] == value["split"]]
    if len(expected) != len(value["shards"]): raise EvidenceError("cohort misses or adds split shards")
    rows, categories = {}, set()
    for ordinal, (expected_shard, item) in enumerate(zip(expected,value["shards"])):
        if item.get("ordinal") != ordinal or item.get("path") != expected_shard["path"] or item.get("sha256") != expected_shard["sha256"] or item.get("positions") != expected_shard["positions"]:
            raise EvidenceError("cohort shard order or identity differs")
        shard_path = tokenized/"corpus"/item["path"]; request = capture.validate_manifest(shard_path)
        expected_docs = [{"id":x["id"],"category":x["category"],"positions":len(x["positions"])} for x in request["documents"]]
        if item.get("documents") != expected_docs: raise EvidenceError("cohort document coverage differs")
        cap_root = ROOT/item["capture"]["capturePath"];harness=item["capture"]["harnessSnapshot"]
        hroot=ROOT/harness["path"];snapshot=read_json(hroot/"snapshot.json")
        if not hroot.resolve().is_relative_to(root.resolve()) or harness["snapshotSHA256"]!=sha256(hroot/"snapshot.json"):raise EvidenceError("archived execution harness owner differs")
        receipt=validate_receipt_file(cap_root/"receipt.json");expected_map=receipt["identities"]["harness_hashes"]
        _validate_snapshot(hroot,snapshot,"slotstream-execution-harness-snapshot-v1",set(expected_map),canonical_hash(expected_map))
        if {name:identity["sha256"] for name,identity in snapshot["files"].items()}!=expected_map:raise EvidenceError("archived harness does not match launcher receipt")
        build_root=ROOT/item["capture"]["immutableBuildPath"]
        if not build_root.resolve().is_relative_to(root.resolve()) or {x.name for x in build_root.iterdir()}!={"slotstream","mlx.metallib","build-identity.json","build-source.tar.gz","build-source-before.json"} or any(x.is_symlink() or not x.is_file() for x in build_root.iterdir()):raise EvidenceError("immutable build inventory differs")
        immutable=build_root/"slotstream"
        validate_build_identity(immutable,historical=True)
        current = _capture_entry(cap_root,shard_path,request,immutable,harness)
        if current != item["capture"]: raise EvidenceError("capture receipt, report, source, or archive identity drifted")
        report = read_json(cap_root/"native/report.json")
        metadata = {x["id"]:x for x in request["documents"]}
        report_docs = {}
        for document in report["documents"]:
            if document["id"] in report_docs: raise EvidenceError("duplicate native document metadata")
            report_docs[document["id"]] = document
        if set(report_docs) != set(metadata): raise EvidenceError("native document coverage differs")
        expected_positions = {}
        for document in request["documents"]:
            categories.add(document["category"])
            for position in document["positions"]:
                key = (document["id"],position["position"])
                if key in expected_positions: raise EvidenceError("duplicate expected position metadata")
                expected_positions[key] = {"inputID":position["inputID"],"nextTokenID":position["nextTokenID"],
                                           "category":document["category"]}
        native_positions = {}
        for document in report["documents"]:
            for position in document["positions"]:
                key = (document["id"],position["position"])
                if key in native_positions: raise EvidenceError("duplicate native position metadata")
                native_positions[key] = {"inputID":position["input_id"],"nextTokenID":position["next_token_id"]}
        if any(native_positions.get(k) != {"inputID":v["inputID"],"nextTokenID":v["nextTokenID"]}
               for k,v in expected_positions.items()) or set(native_positions) != set(expected_positions):
            raise EvidenceError("native position input or target identity differs")
        logits = {}
        for artifact in report["files"]:
            if artifact["category"] != "logits": continue
            key = (artifact["document_id"],artifact["token_position"])
            if key in logits: raise EvidenceError("duplicate logits row")
            if key not in expected_positions or artifact["input_id"] != expected_positions[key]["inputID"]:
                raise EvidenceError("orphan or mismatched logits row")
            logits[key] = cap_root/"native"/artifact["name"]
        if set(logits) != set(expected_positions): raise EvidenceError("missing or extra logits rows")
        for key,path in logits.items():
            global_key = (ordinal,key[0],key[1])
            if global_key in rows: raise EvidenceError("duplicate cohort position")
            rows[global_key] = {**expected_positions[key],"path":path,"documentID":key[0],"position":key[1]}
    identities={(x["capture"]["mode"],x["capture"]["model"],canonical_hash(x["capture"]["sourceIdentity"]),canonical_hash(x["capture"]["runtimeIdentity"]),x["capture"]["launcherHarnessSHA256"]) for x in value["shards"]}
    if len(identities)!=1:raise EvidenceError("cohort shards cross modes, builds, models, controls, or harnesses")
    actual_counts={}
    for row in rows.values():actual_counts.setdefault(row["category"],0);actual_counts[row["category"]]+=1
    if value["kind"]=="full-study" and actual_counts!={name:item["positions"] for name,item in expected_counts.items()}:raise EvidenceError("full-study per-category coverage differs")
    if value["positions"] != len(rows) or value["expectedCategories"] != sorted(categories):
        raise EvidenceError("cohort position or category summary differs")
    return value, rows


def _identity_differences(reference, candidate):
    differences = set()
    if reference["shards"][0]["capture"]["mode"] != candidate["shards"][0]["capture"]["mode"]: differences.add("mode")
    for field, label in (("binary_sha256","binarySHA256"),("source_archive_sha256","sourceArchiveSHA256"),
                         ("build_identity_sha256","buildIdentitySHA256"),("metallib_sha256","metallibSHA256")):
        if any(a["capture"]["sourceIdentity"][field] != b["capture"]["sourceIdentity"][field]
               for a,b in zip(reference["shards"],candidate["shards"])): differences.add(label)
    for field in ("model_config_sha256","model_index_sha256"):
        if any(a["capture"]["sourceIdentity"][field] != b["capture"]["sourceIdentity"][field]
               for a,b in zip(reference["shards"],candidate["shards"])):
            raise EvidenceError("candidate and reference use different model identities")
    if any(a["capture"]["model"] != b["capture"]["model"] for a,b in zip(reference["shards"],candidate["shards"])):
        raise EvidenceError("candidate and reference use different model paths")
    if any(a["capture"]["launcherHarnessSHA256"]!=b["capture"]["launcherHarnessSHA256"] for a,b in zip(reference["shards"],candidate["shards"])):differences.add("executionHarnessSHA256")
    if any(a["capture"]["runtimeIdentity"]!=b["capture"]["runtimeIdentity"] for a,b in zip(reference["shards"],candidate["shards"])):differences.add("runtimeControls")
    return differences


def compare_cohorts(reference_root: Path, candidate_root: Path, output: Path,
                    allowed_differences: set[str], resamples: int = 10000) -> int:
    owned=False
    try:
        if not allowed_differences.issubset(IDENTITY_DIFFERENCES): raise EvidenceError("unknown allowed identity difference")
        output = fresh_output(output);owned=True
        reference, refs = validate_cohort(reference_root); candidate, cands = validate_cohort(candidate_root)
        for key in ("kind","split","suite","tokenized","expectedCategories","positions"):
            if reference[key] != candidate[key]: raise EvidenceError(f"cross-corpus, cross-split, or coverage mismatch: {key}")
        differences = _identity_differences(reference,candidate)
        if differences != allowed_differences: raise EvidenceError("actual identity differences do not equal explicit allowed set")
        if set(refs) != set(cands): raise EvidenceError("capture row coverage differs")
        rows = []
        ordered = sorted(refs)
        for start in range(0,len(ordered),CHUNK_ROWS):
            for key in ordered[start:start+CHUNK_ROWS]:
                left,right = refs[key],cands[key]
                for field in ("inputID","nextTokenID","category","documentID","position"):
                    if left[field] != right[field]: raise EvidenceError("aligned row identity differs")
                result = compare_row(np.fromfile(left["path"],dtype="<f4"),np.fromfile(right["path"],dtype="<f4"),left["nextTokenID"])
                result.update({"documentID":left["documentID"],"position":left["position"],"category":left["category"],
                               "inputID":left["inputID"],"nextTokenID":left["nextTokenID"]}); rows.append(result)
        pooled = aggregate(rows)
        categories = {name:aggregate([x for x in rows if x["category"] == name]) for name in reference["expectedCategories"]}
        def passed(limit):
            def one(value): return value["meanKL"] <= limit["meanKL"] and value["p99KL"] <= limit["p99KL"] and value["top1Agreement"] >= limit["top1Agreement"] and value["pplRatio"] <= limit["pplRatio"]
            return one(pooled) and all(one(x) for x in categories.values())
        applicable = reference["kind"] == "full-study"
        report = {"format":"slotstream-quality-metrics-v2","schemaVersion":2,"qualification":False,
                  "cohortKind":reference["kind"],"diagnosticControl":not applicable,
                  "referenceIndexSHA256":sha256(reference_root.resolve()/"index.json"),
                  "candidateIndexSHA256":sha256(candidate_root.resolve()/"index.json"),
                  "allowedIdentityDifferences":sorted(allowed_differences),"rows":len(rows),
                  "rowIdentitySHA256":canonical_hash([{k:x[k] for k in ("documentID","position","category","inputID","nextTokenID")} for x in rows]),
                  "pooled":pooled,"categories":categories,"bootstrap":bootstrap_documents(rows,resamples,17),
                  "negativeKLRoundoffBand":{"minimumAccepted":NEGATIVE_KL_BAND,"rawValuesReported":True},
                  "thresholds":{"approximate":{"applicable":applicable,"passed":passed(APPROX) if applicable else False,**APPROX},
                                "numerical":{"applicable":applicable,"passed":passed(NUMERICAL) if applicable else False,**NUMERICAL}},
                  "overallQualification":False,
                  "unavailable":["final teacher-forced inference","scored task execution","HumanEval sandbox"],
                  "runtime":{"python":sys.version,"executable":sys.executable,"numpy":np.__version__,"numpyPath":np.__file__,
                             "backend":np.show_config(mode="dicts"),"blasThreads":{"OPENBLAS_NUM_THREADS":os.environ["OPENBLAS_NUM_THREADS"],"OMP_NUM_THREADS":os.environ["OMP_NUM_THREADS"]}}}
        report["qualityHarnessHashes"] = harness_hashes()
        json.dumps(report,allow_nan=False); atomic_json(output/"report.json",report)
        atomic_json(output/"completion.json",{"format":"slotstream-quality-metrics-completion-v2",
                    "reportSHA256":sha256(output/"report.json"),"qualification":False}); return 0
    except Exception as error:
        if owned:
            atomic_json(output/"failure.json",{"format":"slotstream-quality-metrics-failure-v2","error":f"{type(error).__name__}: {error}"})
        return 1


def capture_cohort(suite:Path,binary:Path,mode:str,split:str,output:Path)->int:
    owned=False
    try:
        if split=="qualification":raise EvidenceError("qualification inference is locked")
        if split not in {"training","development"}:raise EvidenceError("unsupported capture split")
        if mode not in {"reference-off","reference-on"}:raise EvidenceError("unsupported capture mode")
        suite_value=corpus.validate_manifest(suite)
        tokenized=ROOT/suite_value["tokenized"]["path"]
        index=tokenize_corpus.validate_corpus(tokenized)["index"]
        shards=[x for x in index["shards"] if x["split"]==split]
        if not shards:raise EvidenceError("frozen split has no shards")
        benchmark.preflight(17);output=fresh_output(output);owned=True;captures=[]
        for ordinal,item in enumerate(shards):
            shard=tokenized/"corpus"/item["path"];run=output/"runs"/f"shard-{ordinal:03d}"
            run.parent.mkdir(parents=True,exist_ok=True)
            capture._native(binary.resolve(),corpus.MODEL.resolve(),shard,split,mode,run)
            captures.append(run)
        code=freeze_cohort(tokenized,split,"full-study",captures,output/"cohort",suite)
        if code:raise EvidenceError("complete cohort indexing failed")
        atomic_json(output/"completion.json",{"format":"slotstream-quality-capture-cohort-completion-v1",
                    "cohortIndexSHA256":sha256(output/"cohort/index.json"),"qualification":False,
                    "shards":len(shards),"serial":True,"headroomGB":17,"targetGB":14,"settlingSeconds":2})
        return 0
    except KeyboardInterrupt:
        if owned:atomic_json(output/"failure.json",{"error":"KeyboardInterrupt: capture cohort cancelled","completed":False})
        return 130
    except Exception as error:
        if owned:atomic_json(output/"failure.json",{"error":f"{type(error).__name__}: {error}","completed":False})
        return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); commands = parser.add_subparsers(dest="command")
    p=commands.add_parser("freeze-cohort"); p.add_argument("--tokenized",type=Path,required=True); p.add_argument("--split",choices=("training","development"),required=True); p.add_argument("--kind",choices=("full-study","diagnostic-control"),required=True); p.add_argument("--capture",type=Path,action="append",required=True); p.add_argument("--suite",type=Path); p.add_argument("--output",type=Path,required=True)
    p=commands.add_parser("validate-cohort"); p.add_argument("--cohort",type=Path,required=True)
    p=commands.add_parser("compare"); p.add_argument("--reference",type=Path,required=True); p.add_argument("--candidate",type=Path,required=True); p.add_argument("--allow-identity-difference",action="append",default=[]); p.add_argument("--bootstrap-resamples",type=int,default=10000); p.add_argument("--output",type=Path,required=True)
    p=commands.add_parser("capture-cohort");p.add_argument("--suite",type=Path,required=True);p.add_argument("--binary",type=Path,required=True);p.add_argument("--mode",choices=("reference-off","reference-on"),required=True);p.add_argument("--split",required=True);p.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(argv)
    try:
        if args.command=="freeze-cohort": return freeze_cohort(args.tokenized,args.split,args.kind,args.capture,args.output,args.suite)
        if args.command=="validate-cohort": validate_cohort(args.cohort); return 0
        if args.command=="compare": return compare_cohorts(args.reference,args.candidate,args.output,set(args.allow_identity_difference),args.bootstrap_resamples)
        if args.command=="capture-cohort":return capture_cohort(args.suite,args.binary,args.mode,args.split,args.output)
        parser.error("choose freeze-cohort, validate-cohort, or compare")
    except Exception as error:
        print(f"{type(error).__name__}: {error}",file=sys.stderr); return 1


if __name__=="__main__": raise SystemExit(main())
