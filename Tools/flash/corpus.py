#!/usr/bin/env python3
"""Prepare, freeze, and reconstruct the Plan 008 quality corpus.

Preparation is host-only. Tokenization is a later, separately reviewed step.
Every evidence-producing command refuses to overwrite an existing path.
"""
from __future__ import annotations

import argparse, copy, csv, gzip, hashlib, io, json, random, re, shutil, sys, unicodedata, zipfile
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from common import EvidenceError, atomic_json, fresh_output, harness_hashes, read_json, sha256
import tokenize_corpus
import prompt_corpus

RECON = ROOT / ".build/flash/corpora/source-recon"
MODEL = ROOT / "models/jang-6s"
ADAPTER = ROOT / ".build/flash/runs/diagnostic-reference-corpus-v2/bin/slotstream"
AUTHORED = ROOT / "Tools/fixtures/flash/authored-quality-v2.json"
CATEGORIES = ("prose", "code", "math", "multilingual", "structured")
PRODUCER_FILES=("Tools/flash/corpus.py","Tools/flash/tokenize_corpus.py","Tools/flash/prompt_corpus.py",
 "Tools/fixtures/flash/authored-quality-v2.json","Tools/flash/schemas/quality-prepared-suite-v2.json",
 "Tools/flash/schemas/quality-suite-v2.json")
SYSTEM_PROMPTS = {
    "gsm8k":"Solve the problem. Return only the final numeric answer.",
    "xnli":"Classify the relationship. Answer with exactly entailment, neutral, or contradiction.",
    "structured":"Return only the requested JSON object. Do not add commentary.",
}
SOURCE_FILES = {
    "gsm8k-train.jsonl": (4166206, "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465"),
    "gsm8k-test.jsonl": (749738, "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"),
    "gsm8k-LICENSE": (1062, "86bbb73e855821d7c401912fd4bf82e34313e6e3b6fd6f909f2b6cc9e209a53b"),
    "humaneval.jsonl.gz": (44877, "b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef"),
    "humaneval-LICENSE": (1083, "bcba3de214851cce46ed5af42d6698044616eeace887c3231bc7a20474ab639e"),
    "xnli.zip": (17865352, "4ba1d5e1afdb7161f0f23c66dc787802ccfa8a25a3ddd3b165a35e50df346ab1"),
    "xnli-LICENSE": (19332, "1d8ca66080a2701508a70be4ddad4be28c509bcc3397a682fb8edf767fdd7a00"),
}
SOURCE_URLS = {
    "gsm8k-train.jsonl":"https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/train.jsonl",
    "gsm8k-test.jsonl":"https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/test.jsonl",
    "gsm8k-LICENSE":"https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/LICENSE",
    "humaneval.jsonl.gz":"https://raw.githubusercontent.com/openai/human-eval/6d43fb980f9fee3c892a914eda09951f772ad10d/data/HumanEval.jsonl.gz",
    "humaneval-LICENSE":"https://raw.githubusercontent.com/openai/human-eval/6d43fb980f9fee3c892a914eda09951f772ad10d/LICENSE",
    "xnli.zip":"https://dl.fbaipublicfiles.com/XNLI/XNLI-1.0.zip",
    "xnli-LICENSE":"https://raw.githubusercontent.com/facebookresearch/XNLI/9624695001060244cf8d5d893b5236093a21a42c/LICENSE",
}
SCORER_FILES = {
    "humaneval-evaluation.py": (3438, "e926d5e9b8f1ec040096cd5952e8cf79f8dc0b7fb80ba6667ce604ffea0397bc"),
    "humaneval-execution.py": (6582, "79901c6f5b59701c465b164aed290fa53abc35abc80a5378fc10f4dac1f9a84c"),
    "humaneval-data.py": (1529, "e1351f0ded89b072490180ce94ce33b88bac4e38b26f08eb26ab0714586585ab"),
}
SCORER_URLS = {
    "humaneval-evaluation.py":"https://raw.githubusercontent.com/openai/human-eval/6d43fb980f9fee3c892a914eda09951f772ad10d/human_eval/evaluation.py",
    "humaneval-execution.py":"https://raw.githubusercontent.com/openai/human-eval/6d43fb980f9fee3c892a914eda09951f772ad10d/human_eval/execution.py",
    "humaneval-data.py":"https://raw.githubusercontent.com/openai/human-eval/6d43fb980f9fee3c892a914eda09951f772ad10d/human_eval/data.py",
}
VERSIONS = {
    "gsm8k": {"revision": "3101c7d5072418e28b9008a6636bde82a006892c", "license": "MIT"},
    "humaneval": {"revision": "6d43fb980f9fee3c892a914eda09951f772ad10d", "license": "MIT"},
    "xnli": {"revision": "9624695001060244cf8d5d893b5236093a21a42c", "license": "CC-BY-NC-4.0"},
    "repository_authored": {"fixture": "Tools/fixtures/flash/authored-quality-v2.json", "license": "CC0-1.0"},
}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def normalized_hash(text: str) -> str:
    return hashlib.sha256(normalized(text).encode()).hexdigest()


def partition(text: str) -> int:
    return int(normalized_hash(text), 16) % 5


def canonical_messages(messages: list[dict[str, str]]) -> str:
    if not messages or any(set(x) != {"role", "content"} or x["role"] not in
                           {"system", "user", "assistant", "tool"} or
                           not isinstance(x["content"], str) for x in messages):
        raise EvidenceError("structured messages are malformed")
    return canonical_bytes(messages).decode()


def chat_content_hash(system: str, user: str) -> str:
    return canonical_hash({"system":system,"user":user})


def parse_gsm_candidate(text:str)->Decimal:
    matches=re.findall(r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)",text)
    if not matches:raise EvidenceError("GSM8K candidate has no numeric answer")
    try:value=Decimal(matches[-1].replace(",",""))
    except InvalidOperation as error:raise EvidenceError("GSM8K candidate number is malformed") from error
    if not value.is_finite():raise EvidenceError("GSM8K candidate number is nonfinite")
    return value


def _verify(path: Path, expected: tuple[int, str]) -> dict[str, Any]:
    size, digest = expected
    if path.is_symlink() or not path.is_file() or path.stat().st_size != size or sha256(path) != digest:
        raise EvidenceError(f"pinned source changed: {path.name}")
    return {"bytes": size, "sha256": digest}


def verify_sources() -> dict[str, Any]:
    files = {name: _verify(RECON / name, identity) for name, identity in SOURCE_FILES.items()}
    scorer_files = {name: {**_verify(RECON / name, identity), "executed": False}
                    for name, identity in SCORER_FILES.items()}
    declared = json.loads((RECON / "sources.json").read_text())
    if not isinstance(declared, list) or [x.get("name") for x in declared] != list(SOURCE_FILES):
        raise EvidenceError("source catalogue set or order differs")
    for item in declared:
        size, digest = SOURCE_FILES[item["name"]]
        if item.get("bytes") != size or item.get("sha256") != digest or item.get("url") != SOURCE_URLS[item["name"]] or item.get("complete") is not True:
            raise EvidenceError("source catalogue identity differs")
    scorers = json.loads((RECON / "humaneval-scorer-sources.json").read_text())
    if not isinstance(scorers, list) or [x.get("name") for x in scorers] != list(SCORER_FILES):
        raise EvidenceError("HumanEval scorer source set or order differs")
    for item in scorers:
        size, digest = SCORER_FILES[item["name"]]
        if item.get("bytes") != size or item.get("sha256") != digest or item.get("url") != SCORER_URLS[item["name"]] or item.get("executed") is not False:
            raise EvidenceError("HumanEval scorer identity differs")
    return {"files": files, "urls":copy.deepcopy(SOURCE_URLS), "scorerFiles": scorer_files,
            "scorerURLs":copy.deepcopy(SCORER_URLS), "versions": copy.deepcopy(VERSIONS),
            "catalogues": {"sourcesSHA256": sha256(RECON / "sources.json"),
                           "humanevalScorersSHA256": sha256(RECON / "humaneval-scorer-sources.json"),
                           "xnliSelectionSHA256":sha256(RECON / "xnli-selection-recon.json"),
                           "splitPreflightSHA256":sha256(RECON / "split-preflight.json")}}


def _load_sources():
    train = [json.loads(x) for x in (RECON / "gsm8k-train.jsonl").read_text().splitlines()]
    test = [json.loads(x) for x in (RECON / "gsm8k-test.jsonl").read_text().splitlines()]
    with gzip.open(RECON / "humaneval.jsonl.gz", "rt", encoding="utf-8") as stream:
        humaneval = [json.loads(x) for x in stream]
    with zipfile.ZipFile(RECON / "xnli.zip") as archive:
        def rows(name):
            with archive.open(name) as raw:
                return list(csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8"), delimiter="\t"))
        dev = rows("XNLI-1.0/xnli.dev.tsv")
        x_test = rows("XNLI-1.0/xnli.test.tsv")
    if (len(train), len(test), len(humaneval)) != (7473, 1319, 164):
        raise EvidenceError("source row counts differ")
    return train, test, humaneval, dev, x_test


def _xnli_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order, ordered_prompts, groups = [], set(), {}
    for ordinal, row in enumerate(rows):
        prompt, pair = row["promptID"], row["pairID"]
        if prompt not in groups:
            groups[prompt] = {"promptID": prompt, "firstOrdinal": ordinal, "pairOrder": [], "pairs": {}}
        if row["language"] == "en" and prompt not in ordered_prompts:
            order.append(prompt); ordered_prompts.add(prompt)
        group = groups[prompt]
        if pair not in group["pairs"]: group["pairs"][pair] = {}
        if row["language"] == "en" and pair not in group["pairOrder"]: group["pairOrder"].append(pair)
        if row["language"] in group["pairs"][pair]:
            raise EvidenceError("duplicate XNLI language row")
        group["pairs"][pair][row["language"]] = row
    if set(order) != set(groups): raise EvidenceError("XNLI promptID lacks an English row")
    result = []
    for prompt in order:
        group = groups[prompt]
        english = [pair for pair in group["pairOrder"] if "en" in group["pairs"][pair]]
        if len(english) != 3:
            raise EvidenceError("XNLI premise lacks three English hypotheses")
        premise = group["pairs"][english[0]]["en"]["sentence1"]
        if any(group["pairs"][pair]["en"]["sentence1"] != premise for pair in english):
            raise EvidenceError("XNLI promptID contains conflicting English premises")
        group["englishPairOrder"], group["englishPremise"] = english, premise
        result.append(group)
    return result


def _document(identifier, category, split, text, partition_text, score, source, license_name, selector):
    return {"id": identifier, "kind": "raw", "category": category, "split": split,
            "sourceID": source, "license": license_name, "partitionText": partition_text,
            "partitionNormalizedSHA256": normalized_hash(partition_text),
            "partitionBucket": partition(partition_text),
            "rawSHA256": hashlib.sha256(text.encode()).hexdigest(),
            "normalizedSHA256": normalized_hash(text), "text": text,
            "warmupTokens": 2, "scoreTokens": score, "selector": selector}


def _tokenizer_document(document):
    return {"id": document["id"], "kind": "raw", "category": document["category"],
            "split": document["split"], "sourceID": document["sourceID"],
            "license": document["license"], "partitionKey": document["partitionNormalizedSHA256"],
            "text": document["text"], "textSHA256": document["rawSHA256"],
            "warmupTokens": document["warmupTokens"], "scoreTokens": document["scoreTokens"]}


def _fixture():
    value = read_json(AUTHORED)
    if set(value) != {"format", "license", "prose", "code"} or value["format"] != "slotstream-authored-quality-fixture-v2" or value["license"] != "CC0-1.0":
        raise EvidenceError("authored fixture shape differs")
    for category in ("prose", "code"):
        rows = value[category]
        if not isinstance(rows, list) or len(rows) < 28:
            raise EvidenceError(f"authored {category} fixture is incomplete")
        skeletons = []
        for row in rows:
            if set(row) != {"id", "text"} or not re.fullmatch(r"[a-z0-9-]+", row["id"]) or len(normalized(row["text"])) < 200:
                raise EvidenceError(f"authored {category} row is malformed or short")
            skeletons.append(re.sub(r"\d+", "#", normalized(row["text"]).casefold()))
        if len(skeletons) != len(set(skeletons)):
            raise EvidenceError(f"authored {category} contains number-only near clones")
    return value


def _structured(index: int, seed: int, prefix: str) -> dict[str, Any]:
    rng = random.Random((seed << 20) + index)
    cities = ["Aster", "Birch", "Cobalt", "Dune", "Ember", "Fjord", "Grove", "Harbor"]
    colors = ["amber", "blue", "coral", "green", "indigo", "silver"]
    mode = index % 5
    if mode == 0:
        expected = {"incident": f"INC-{rng.randrange(1000,9999)}", "city": rng.choice(cities),
                    "severity": rng.randrange(1, 6), "verified": bool(rng.randrange(2))}
        prompt = "Read the incident note and return JSON with incident, city, severity, and verified: " + json.dumps(expected)
        kind = "json_extraction"
    elif mode == 1:
        expected = {"recipient": f"team-{rng.randrange(10,99)}", "priority": rng.choice(colors),
                    "attempts": rng.randrange(1, 6)}
        prompt = "Call queue_delivery with exactly these arguments: " + json.dumps(expected)
        kind = "tool_arguments"
    elif mode == 2:
        expected = {"sensor": {"name": f"probe-{rng.randrange(100,999)}", "units": rng.choice(["C", "kPa", "mm"])},
                    "limits": [rng.randrange(1,20), rng.randrange(21,50)]}
        prompt = "Retain the nested sensor object and limits array in one JSON object: " + json.dumps(expected)
        kind = "nested_json"
    elif mode == 3:
        values = [rng.randrange(2,40) for _ in range(5)]
        expected = {"values": values, "sum": sum(values), "maximum": max(values)}
        prompt = f"For readings {values}, return JSON with values, sum, and maximum."
        kind = "computed_json"
    else:
        expected = {"sku": f"SKU-{rng.randrange(10000,99999)}", "delta": rng.choice([-3,-2,-1,1,2,3]),
                    "location": rng.choice(cities).lower(), "reason": rng.choice(["count","damage","delivery"])}
        prompt = "Prepare inventory_adjust arguments from: " + json.dumps(expected)
        kind = "tool_arguments"
    return {"id": f"{prefix}-{index:03d}", "kind": kind, "prompt": prompt,
            "expected": expected, "expectedCanonicalJSON": canonical_bytes(expected).decode(),
            "seed": seed, "ordinal": index}


def _select_partitioned(rows: Iterable[tuple[str, str, str, dict[str, Any]]], category, source, license_name):
    candidates, documents, decisions, chosen = list(rows), [], [], set()
    for split in ("development", "training"):
        count = 0
        for identifier, text, partition_text, selector in candidates:
            bucket = partition(partition_text)
            eligible = identifier not in chosen and ((bucket == 0) == (split == "development"))
            if eligible and count < 4:
                chosen.add(identifier); count += 1
                documents.append(_document(f"{split[:3]}-{category}-{identifier}", category, split,
                                           text, partition_text, 32, source, license_name, selector))
                decisions.append({"candidateID": identifier, "disposition": "selected", "split": split,
                                  "reason": "first eligible source-order candidate", "partitionBucket": bucket})
            elif identifier not in chosen:
                decisions.append({"candidateID": identifier, "disposition": "rejected", "split": split,
                                  "reason": "partition mismatch" if not eligible else "quota filled",
                                  "partitionBucket": bucket})
            if count == 4: break
        if count != 4: raise EvidenceError(f"insufficient {category} candidates for {split}")
    remaining = [row for row in candidates if row[0] not in chosen]
    if len(remaining) < 20: raise EvidenceError(f"insufficient {category} qualification candidates")
    for identifier, text, partition_text, selector in remaining[:20]:
        documents.append(_document(f"qual-{category}-{identifier}", category, "qualification",
                                   text, partition_text, 42, source, license_name, selector))
        decisions.append({"candidateID": identifier, "disposition": "selected", "split": "qualification",
                          "reason": "first remaining source-order candidate", "partitionBucket": partition(partition_text)})
    return documents, decisions


def _select_multilingual(groups, split, count, excluded):
    documents, decisions, selected = [], [], 0
    for group in groups:
        bucket = partition(group["englishPremise"])
        complete = all(all(language in group["pairs"][pair] for language in ("en","fr","zh"))
                       for pair in group["englishPairOrder"])
        split_ok = split == "qualification" or ((bucket == 0) == (split == "development"))
        eligible = group["promptID"] not in excluded and complete and split_ok
        if not eligible:
            reason = "excluded final premise" if group["promptID"] in excluded else (
                "missing matched translations" if not complete else "partition mismatch")
            decisions.append({"candidateID": group["promptID"], "disposition": "rejected", "split": split,
                              "reason": reason, "partitionBucket": bucket})
            continue
        selected += 1
        decisions.append({"candidateID": group["promptID"], "disposition": "selected", "split": split,
                          "reason": "first eligible English-premise group in file order",
                          "partitionBucket": bucket, "pairIDs": group["englishPairOrder"]})
        for language in ("en", "fr", "zh"):
            first = group["pairs"][group["englishPairOrder"][0]][language]
            lines = ["Premise:", first["sentence1"]]
            for ordinal, pair in enumerate(group["englishPairOrder"], 1):
                row = group["pairs"][pair][language]
                lines.extend([f"Hypothesis {ordinal} ({row['gold_label']}):", row["sentence2"]])
            documents.append(_document(f"{split[:4]}-mult-{group['promptID']}-{language}", "multilingual",
                                       split, "\n".join(lines), group["englishPremise"],
                                       42 if split == "qualification" else 32,
                                       "xnli-test" if split == "qualification" else "xnli-dev",
                                       "CC-BY-NC-4.0", {"promptID": group["promptID"],
                                       "pairIDs": group["englishPairOrder"], "language": language}))
        if selected == count: break
    if selected != count: raise EvidenceError(f"insufficient XNLI groups for {split}")
    return documents, decisions


def build_prepared() -> dict[str, Any]:
    sources = verify_sources()
    gsm_train, gsm_test, humaneval, dev_rows, test_rows = _load_sources()
    authored, x_dev, x_test = _fixture(), _xnli_groups(dev_rows), _xnli_groups(test_rows)
    final_prompt_ids = [group["promptID"] for group in x_test[:200]]
    if len(set(final_prompt_ids)) != 200: raise EvidenceError("final XNLI promptIDs are not unique")
    teacher, selections = [], {}
    for category in ("prose", "code"):
        rows = [(x["id"], x["text"], x["text"], {"fixtureID": x["id"]}) for x in authored[category]]
        docs, decisions = _select_partitioned(rows, category, "repository-authored-plan008-v2", "CC0-1.0")
        teacher.extend(docs); selections[category] = decisions
    structured = [_structured(i, 2, "teacher-structured") for i in range(40)]
    rows = [(x["id"], x["prompt"] + "\nExpected JSON:\n" + x["expectedCanonicalJSON"], x["prompt"],
             {"seed": 2, "ordinal": x["ordinal"], "kind": x["kind"]}) for x in structured]
    docs, decisions = _select_partitioned(rows, "structured", "repository-authored-structured-seed2", "CC0-1.0")
    teacher.extend(docs); selections["structured"] = decisions
    math_rows = [(f"gsm8k-train-{i:04d}", "Question:\n" + x["question"] + "\nWorked answer:\n" + x["answer"],
                  x["question"], {"partition": "train", "row": i}) for i, x in enumerate(gsm_train)]
    math_docs, math_decisions = _select_partitioned(math_rows, "math", "gsm8k-train", "MIT")
    math_docs = [x for x in math_docs if x["split"] != "qualification"]
    math_decisions = [x for x in math_decisions if x["split"] != "qualification"]
    for i, row in list(enumerate(gsm_test))[200:]:
        text = "Question:\n" + row["question"] + "\nWorked answer:\n" + row["answer"]
        if len(normalized(text)) < 300:
            math_decisions.append({"candidateID": f"gsm8k-test-{i:04d}", "disposition": "rejected",
                                   "split": "qualification", "reason": "short pre-tokenization candidate",
                                   "partitionBucket": partition(row["question"])}); continue
        math_docs.append(_document(f"qual-math-gsm8k-{i:04d}", "math", "qualification", text,
                                   row["question"], 42, "gsm8k-test", "MIT", {"partition": "test", "row": i}))
        math_decisions.append({"candidateID": f"gsm8k-test-{i:04d}", "disposition": "selected",
                               "split": "qualification", "reason": "first eligible row after final scored prefix",
                               "partitionBucket": partition(row["question"])})
        if sum(x["split"] == "qualification" for x in math_docs) == 20: break
    teacher.extend(math_docs); selections["math"] = math_decisions
    multi_dev, ddec = _select_multilingual(x_dev, "development", 4, set())
    multi_train, tdec = _select_multilingual(x_dev, "training", 4, set())
    if {x["partitionNormalizedSHA256"] for x in multi_dev} & {x["partitionNormalizedSHA256"] for x in multi_train}:
        raise EvidenceError("translated family leaked across training/development")
    multi_qual, qdec = _select_multilingual(x_test[200:], "qualification", 20, set(final_prompt_ids))
    teacher.extend(multi_dev + multi_train + multi_qual); selections["multilingual"] = ddec + tdec + qdec

    final_gsm = []
    for i, row in enumerate(gsm_test[:200]):
        match = re.search(r"####\s*([^\n]+)\s*$", row["answer"])
        if not match: raise EvidenceError("GSM8K final answer marker is absent")
        final_gsm.append({"id": f"gsm8k-{i:04d}", "sourceRow": i, "prompt": row["question"],
                          "promptRawSHA256": hashlib.sha256(row["question"].encode()).hexdigest(),
                          "promptMode":"chat", "system":SYSTEM_PROMPTS["gsm8k"],
                          "promptContentSHA256":chat_content_hash(SYSTEM_PROMPTS["gsm8k"],row["question"]),
                          "expectedAnswer": row["answer"], "expectedFinal": match.group(1).replace(",", ""),
                          "scorer": "gsm8k-final-number-v1"})
    final_he = []
    for row in sorted(humaneval, key=lambda x: int(x["task_id"].split("/")[1])):
        final_he.append({"id": row["task_id"].replace("/", "-"), "taskID": row["task_id"],
                         "entryPoint": row["entry_point"], "prompt": row["prompt"],
                          "promptRawSHA256": hashlib.sha256(row["prompt"].encode()).hexdigest(),
                         "promptMode":"raw-completion",
                         "canonicalSolution": row["canonical_solution"], "test": row["test"],
                         "scorer": "official-humaneval-unexecuted-v1", "executionAvailable": False})
    final_xnli = []
    for group in x_test[:200]:
        pair = group["englishPairOrder"][0]
        for language in ("en", "fr", "zh"):
            row = group["pairs"][pair].get(language)
            if row is None: raise EvidenceError("final XNLI pair lacks a matched translation")
            prompt = ("Premise: " + row["sentence1"] + "\nHypothesis: " + row["sentence2"] +
                      "\nAnswer with entailment, neutral, or contradiction.")
            final_xnli.append({"id": f"xnli-{group['promptID']}-{pair}-{language}",
                               "promptID": group["promptID"], "pairID": pair, "language": language,
                               "premise": row["sentence1"], "hypothesis": row["sentence2"],
                               "englishPartitionText": group["englishPremise"],
                               "englishPartitionSHA256": normalized_hash(group["englishPremise"]),
                               "prompt": prompt, "promptRawSHA256": hashlib.sha256(prompt.encode()).hexdigest(),
                               "promptMode":"chat", "system":SYSTEM_PROMPTS["xnli"],
                               "promptContentSHA256":chat_content_hash(SYSTEM_PROMPTS["xnli"],prompt),
                               "expectedLabel": row["gold_label"], "scorer": "xnli-exact-label-v1"})
    final_structured = [_structured(i, 1, "structured") for i in range(200)]
    for item in final_structured:
        item["scorer"] = "structured-canonical-json-v1"
        item["promptRawSHA256"] = hashlib.sha256(item["prompt"].encode()).hexdigest()
        item["promptMode"]="chat";item["system"]=SYSTEM_PROMPTS["structured"]
        item["promptContentSHA256"]=chat_content_hash(item["system"],item["prompt"])
    settings = {"decoding": "greedy", "reasoning": False, "seed": None, "eos": "model_eos",
                "benchmarks": {"gsm8k": {"maxNewTokens": 512, "timeoutSeconds": 300,"promptMode":"native-chat","chatThinking":False},
                               "humaneval": {"maxNewTokens": 512, "timeoutSeconds": 300,"promptMode":"raw-completion","chatTemplateApplied":False},
                               "xnli": {"maxNewTokens": 8, "timeoutSeconds": 120,"promptMode":"native-chat","chatThinking":False},
                               "structured": {"maxNewTokens": 256, "timeoutSeconds": 180,"promptMode":"native-chat","chatThinking":False}}}
    scorers = {"gsm8k-final-number-v1": {"expectedSourceParser":"text after final #### marker",
               "candidateParser":"last comma-stripped signed finite decimal; compare numeric values so 2 equals 2.0",
               "errors":"missing, malformed, NaN, and infinity fail"},
               "official-humaneval-unexecuted-v1": {"sourceFiles": list(SCORER_FILES), "executed": False,
                                                     "sandboxVerified": False, "passStubAllowed": False},
               "xnli-exact-label-v1": {"labels": ["entailment","neutral","contradiction"],
                                        "parser": "trim and lowercase"},
               "structured-canonical-json-v1": {"parser": "one JSON value; exact recursively typed equality"}}
    result = {"format": "slotstream-quality-prepared-suite-v2", "schemaVersion": 2,
              "qualification": False, "normalization": "NFKC then Unicode-whitespace collapse; checks only",
              "structuredMessageCanonicalization": "UTF-8 compact canonical JSON object with sorted system/user keys",
              "sources": sources, "authoredFixtureSHA256": sha256(AUTHORED),
              "teacherForced": {"documents": teacher, "selectionDecisions": selections},
              "finalScored": {"executionAvailable": False, "promptTokenIDsAvailable": False,
                              "promptTokenIDBlocker": "awaiting Plan010 prompt-only tokenizer artifact",
                              "settings": settings, "scorerContracts": scorers, "gsm8k": final_gsm,
                              "humaneval": final_he, "xnli": final_xnli, "structured": final_structured},
              "unavailable": ["qualification inference", "scored task execution", "HumanEval sandbox",
                              "final scored prompt token IDs", "population non-inferiority"]}
    result["contractSHA256"] = canonical_hash(result)
    validate_prepared_data(result, reconstruct=False)
    return result


def teacher_counts(documents):
    result = {}
    for item in documents:
        value = result.setdefault(item["split"], {}).setdefault(item["category"], {"documents": 0, "positions": 0})
        value["documents"] += 1; value["positions"] += item["scoreTokens"]
    return result


def prompt_source(prepared: dict[str, Any]) -> dict[str, Any]:
    """Project final cases into the exact Plan 010 prompt-only source contract."""
    documents = []
    licenses = {"gsm8k":"MIT", "humaneval":"MIT", "xnli":"CC-BY-NC-4.0", "structured":"CC0-1.0"}
    for category in ("gsm8k", "humaneval", "xnli", "structured"):
        for item in prepared["finalScored"][category]:
            text = item["prompt"]
            common={"id":item["id"],"category":category,"split":"qualification",
                    "sourceID":f"plan008-final-{category}","license":licenses[category],
                    "partitionKey":normalized_hash(text)}
            if item["promptMode"]=="raw-completion":
                documents.append({**common,"kind":"raw","text":text,
                                  "textSHA256":hashlib.sha256(text.encode()).hexdigest()})
            elif item["promptMode"]=="chat":
                documents.append({**common,"kind":"chat","system":item["system"],"user":text,
                                  "contentSHA256":chat_content_hash(item["system"],text)})
            else: raise EvidenceError("unsupported scored prompt mode")
    result = {"format":"slotstream-prompt-source-v1", "schemaVersion":1,
              "manifestSHA256":"", "documents":documents}
    result["manifestSHA256"] = canonical_hash({k:v for k,v in result.items() if k != "manifestSHA256"})
    return result


def _semantic_source_identity(sources):
    expected = {"files": {name: {"bytes": value[0], "sha256": value[1]} for name, value in SOURCE_FILES.items()},
                "urls":SOURCE_URLS,
                "scorerFiles": {name: {"bytes": value[0], "sha256": value[1], "executed": False}
                                for name, value in SCORER_FILES.items()}, "versions": VERSIONS}
    expected["scorerURLs"]=SCORER_URLS
    for key in expected:
        if sources.get(key) != expected[key]: raise EvidenceError(f"prepared {key} source identity differs")
    if set(sources) != {"files", "urls", "scorerFiles", "scorerURLs", "versions", "catalogues"}:
        raise EvidenceError("prepared source identity has unexpected fields")


def validate_prepared_data(data: dict[str, Any], *, reconstruct: bool):
    required = {"format","schemaVersion","qualification","normalization","structuredMessageCanonicalization",
                "sources","authoredFixtureSHA256","teacherForced","finalScored","unavailable","contractSHA256"}
    if set(data) != required or data.get("format") != "slotstream-quality-prepared-suite-v2" or data.get("schemaVersion") != 2 or data.get("qualification") is not False:
        raise EvidenceError("prepared suite fields are malformed")
    if data["contractSHA256"] != canonical_hash({k:v for k,v in data.items() if k != "contractSHA256"}):
        raise EvidenceError("prepared suite contract hash differs")
    _semantic_source_identity(data["sources"])
    if set(data["teacherForced"]) != {"documents","selectionDecisions"}:
        raise EvidenceError("teacher corpus has unexpected fields")
    documents = data["teacherForced"].get("documents")
    decisions = data["teacherForced"].get("selectionDecisions")
    if not isinstance(documents, list) or not documents or set(decisions or {}) != set(CATEGORIES):
        raise EvidenceError("teacher documents or category selectors are incomplete")
    for category, entries in decisions.items():
        if not isinstance(entries,list) or not entries or any(not isinstance(x,dict) or
                not {"candidateID","disposition","split","reason","partitionBucket"}.issubset(x) or
                x["disposition"] not in {"selected","rejected"} or x["split"] not in {"training","development","qualification"}
                for x in entries): raise EvidenceError(f"{category} selection decisions are malformed")
    scored = data["finalScored"]
    if set(scored) != {"executionAvailable","promptTokenIDsAvailable","promptTokenIDBlocker","settings",
                       "scorerContracts","gsm8k","humaneval","xnli","structured"}:
        raise EvidenceError("final scored contract has unexpected fields")
    final_partition_hashes = {normalized_hash(x["prompt"]) for x in scored["gsm8k"]+scored["structured"]}
    final_partition_hashes.update(x["englishPartitionSHA256"] for x in scored["xnli"])
    ids, texts, partition_splits = set(), set(), {}
    for item in documents:
        keys = {"id","kind","category","split","sourceID","license","partitionText",
                "partitionNormalizedSHA256","partitionBucket","rawSHA256","normalizedSHA256",
                "text","warmupTokens","scoreTokens","selector"}
        if set(item) != keys or item["kind"] != "raw" or item["category"] not in CATEGORIES or item["split"] not in {"training","development","qualification"}:
            raise EvidenceError("teacher document shape differs")
        if item["id"] in ids or item["normalizedSHA256"] in texts: raise EvidenceError("duplicate teacher document")
        ids.add(item["id"]); texts.add(item["normalizedSHA256"])
        if item["rawSHA256"] != hashlib.sha256(item["text"].encode()).hexdigest() or item["normalizedSHA256"] != normalized_hash(item["text"]):
            raise EvidenceError("teacher text identity differs")
        if item["partitionNormalizedSHA256"] != normalized_hash(item["partitionText"]) or item["partitionBucket"] != partition(item["partitionText"]):
            raise EvidenceError("teacher partition identity differs")
        if item["split"] == "development" and item["partitionBucket"] != 0: raise EvidenceError("forged development partition")
        if item["split"] == "training" and item["partitionBucket"] == 0: raise EvidenceError("forged training partition")
        prior = partition_splits.get(item["partitionNormalizedSHA256"])
        if prior is not None and prior != item["split"]: raise EvidenceError("translated family leaked across splits")
        partition_splits[item["partitionNormalizedSHA256"]] = item["split"]
        if item["split"] in {"training","development"} and item["partitionNormalizedSHA256"] in final_partition_hashes:
            raise EvidenceError("final record leaked into training or development")
        if not 0 <= item["warmupTokens"] <= 256 or not 1 <= item["scoreTokens"] <= 64 or len(item["text"].encode()) <= item["scoreTokens"]:
            raise EvidenceError("teacher window is structurally too short")
    counts = teacher_counts(documents)
    for split in ("training", "development"):
        if set(counts.get(split, {})) != set(CATEGORIES) or any(counts[split][c]["documents"] < 4 or counts[split][c]["positions"] < 128 for c in CATEGORIES):
            raise EvidenceError(f"{split} coverage is incomplete")
    if set(counts.get("qualification", {})) != set(CATEGORIES) or any(
            counts["qualification"][c]["documents"] < 20 or counts["qualification"][c]["positions"] < 512
            for c in CATEGORIES) or sum(x["positions"] for x in counts["qualification"].values()) < 4096:
        raise EvidenceError("qualification coverage is incomplete")
    if scored.get("executionAvailable") is not False or scored.get("promptTokenIDsAvailable") is not False or not scored.get("promptTokenIDBlocker"):
        raise EvidenceError("scored execution availability is misstated")
    settings=scored["settings"]
    if set(settings)!={"decoding","reasoning","seed","eos","benchmarks"} or settings["decoding"]!="greedy" or settings["reasoning"] is not False or settings["seed"] is not None or settings["eos"]!="model_eos" or set(settings["benchmarks"])!={"gsm8k","humaneval","xnli","structured"}:
        raise EvidenceError("scored generation settings differ")
    if set(scored["scorerContracts"])!={"gsm8k-final-number-v1","official-humaneval-unexecuted-v1","xnli-exact-label-v1","structured-canonical-json-v1"} or scored["scorerContracts"]["official-humaneval-unexecuted-v1"].get("executed") is not False or scored["scorerContracts"]["official-humaneval-unexecuted-v1"].get("sandboxVerified") is not False or scored["scorerContracts"]["official-humaneval-unexecuted-v1"].get("passStubAllowed") is not False:
        raise EvidenceError("scorer contracts differ or overclaim HumanEval execution")
    if [len(scored.get(name, [])) for name in ("gsm8k","humaneval","xnli","structured")] != [200,164,600,200]:
        raise EvidenceError("final denominators do not derive from actual records")
    gsm_keys={"id","sourceRow","prompt","promptRawSHA256","promptMode","system","promptContentSHA256","expectedAnswer","expectedFinal","scorer"}
    he_keys={"id","taskID","entryPoint","prompt","promptRawSHA256","promptMode","canonicalSolution","test","scorer","executionAvailable"}
    xnli_keys={"id","promptID","pairID","language","premise","hypothesis","englishPartitionText",
               "englishPartitionSHA256","prompt","promptRawSHA256","promptMode","system","promptContentSHA256","expectedLabel","scorer"}
    structured_keys={"id","kind","prompt","expected","expectedCanonicalJSON","seed","ordinal","scorer","promptRawSHA256","promptMode","system","promptContentSHA256"}
    if any(set(x)!=gsm_keys or x["scorer"]!="gsm8k-final-number-v1" or x["promptMode"]!="chat" or x["system"]!=SYSTEM_PROMPTS["gsm8k"] or x["promptContentSHA256"]!=chat_content_hash(x["system"],x["prompt"]) for x in scored["gsm8k"]): raise EvidenceError("GSM8K final record shape differs")
    if any(set(x)!=he_keys or x["scorer"]!="official-humaneval-unexecuted-v1" or x["executionAvailable"] is not False or x["promptMode"]!="raw-completion" for x in scored["humaneval"]): raise EvidenceError("HumanEval final record shape differs")
    if any(set(x)!=xnli_keys or x["scorer"]!="xnli-exact-label-v1" or x["englishPartitionSHA256"]!=normalized_hash(x["englishPartitionText"]) or x["promptMode"]!="chat" or x["system"]!=SYSTEM_PROMPTS["xnli"] or x["promptContentSHA256"]!=chat_content_hash(x["system"],x["prompt"]) for x in scored["xnli"]): raise EvidenceError("XNLI final record shape differs")
    if any(set(x)!=structured_keys or x["scorer"]!="structured-canonical-json-v1" or x["expectedCanonicalJSON"]!=canonical_bytes(x["expected"]).decode() or x["promptMode"]!="chat" or x["system"]!=SYSTEM_PROMPTS["structured"] or x["promptContentSHA256"]!=chat_content_hash(x["system"],x["prompt"]) for x in scored["structured"]): raise EvidenceError("structured final record shape differs")
    if [x["sourceRow"] for x in scored["gsm8k"]] != list(range(200)): raise EvidenceError("GSM8K selector order differs")
    if [x["taskID"] for x in scored["humaneval"]] != [f"HumanEval/{i}" for i in range(164)]: raise EvidenceError("HumanEval selector order differs")
    families = {}
    for item in scored["xnli"]:
        families.setdefault(item["promptID"], []).append(item)
    if len(families) != 200 or any([x["language"] for x in family] != ["en","fr","zh"] or len({x["pairID"] for x in family}) != 1 for family in families.values()):
        raise EvidenceError("XNLI selector or translation coverage differs")
    if [x["ordinal"] for x in scored["structured"]] != list(range(200)) or any(x["seed"] != 1 for x in scored["structured"]):
        raise EvidenceError("structured selector differs")
    for name in ("gsm8k","humaneval","xnli","structured"):
        for item in scored[name]:
            if item["promptRawSHA256"] != hashlib.sha256(item["prompt"].encode()).hexdigest():
                raise EvidenceError(f"{name} prompt identity differs")
    if reconstruct and canonical_bytes(build_prepared()) != canonical_bytes(data):
        raise EvidenceError("prepared suite does not reconstruct from pinned sources")
    return {"counts": counts, "documents": len(documents),
            "finalCounts": {name: len(scored[name]) for name in ("gsm8k","humaneval","xnli","structured")},
            "contractSHA256": data["contractSHA256"]}


def prepare(output: Path) -> int:
    try:
        output = fresh_output(output)
        suite = build_prepared()
        teacher = {"format":"slotstream-authored-corpus-v1","schemaVersion":1,"manifestSHA256":"",
                   "documents":[_tokenizer_document(x) for x in suite["teacherForced"]["documents"]]}
        teacher["manifestSHA256"] = tokenize_corpus.canonical_hash(teacher, "manifestSHA256")
        prompts = prompt_source(suite)
        atomic_json(output / "suite.json", suite); atomic_json(output / "teacher-corpus.json", teacher)
        atomic_json(output / "scored-prompts.json", prompts)
        tokenize_corpus.validate_source(output / "teacher-corpus.json")
        atomic_json(output / "completion.json", {"format":"slotstream-quality-prepared-completion-v2",
                    "qualification":False,"suiteSHA256":sha256(output / "suite.json"),
                    "teacherCorpusSHA256":sha256(output / "teacher-corpus.json"),
                    "scoredPromptsSHA256":sha256(output / "scored-prompts.json"),
                    "hostOnly":True,"tokenizerInvoked":False})
        return 0
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr); return 1


def validate_prepared(path: Path):
    root = path.resolve(strict=True)
    if not root.is_dir() or path.is_symlink(): raise EvidenceError("prepared suite must be a regular directory")
    suite_path, teacher_path = root / "suite.json", root / "teacher-corpus.json"
    result = validate_prepared_data(read_json(suite_path), reconstruct=True)
    teacher = tokenize_corpus.validate_source(teacher_path)
    suite = read_json(suite_path)
    if teacher["documents"] != [_tokenizer_document(x) for x in suite["teacherForced"]["documents"]]:
        raise EvidenceError("teacher corpus differs from prepared suite")
    prompts = read_json(root / "scored-prompts.json")
    if prompts != prompt_source(suite): raise EvidenceError("scored prompt source differs from prepared final cases")
    completion = read_json(root / "completion.json")
    if completion != {"format":"slotstream-quality-prepared-completion-v2","qualification":False,
                      "suiteSHA256":sha256(suite_path),"teacherCorpusSHA256":sha256(teacher_path),
                      "scoredPromptsSHA256":sha256(root / "scored-prompts.json"),
                      "hostOnly":True,"tokenizerInvoked":False}:
        raise EvidenceError("prepared completion does not bind its artifacts")
    result.update({"suiteSHA256":sha256(suite_path),"teacherCorpusSHA256":sha256(teacher_path),
                   "scoredPromptsSHA256":sha256(root / "scored-prompts.json")})
    return result


def _relative(path: Path) -> str:
    value = path.resolve(strict=True)
    if not value.is_relative_to(ROOT.resolve()): raise EvidenceError("suite artifact escapes repository")
    return str(value.relative_to(ROOT.resolve()))


def producer_snapshot(output: Path) -> int:
    owned=False
    try:
        output=fresh_output(output);owned=True
        names=list(PRODUCER_FILES)
        files={}
        for name in names:
            source=ROOT/name;destination=output/name;destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,destination);files[name]={"bytes":destination.stat().st_size,"sha256":sha256(destination)}
        atomic_json(output/"snapshot.json",{"format":"slotstream-quality-producer-snapshot-v1","files":files})
        return 0
    except Exception as error:
        if owned: atomic_json(output/"failure.json",{"error":f"{type(error).__name__}: {error}"})
        return 1


def validate_producer_snapshot(root:Path):
    if root.is_symlink() or (root/"snapshot.json").is_symlink():raise EvidenceError("symlinked producer root or manifest is not accepted")
    root=root.resolve(strict=True);root_manifest=(root/"snapshot.json").resolve(strict=True);value=read_json(root/"snapshot.json")
    if set(value)!={"format","files"} or value["format"]!="slotstream-quality-producer-snapshot-v1" or set(value["files"])!=set(PRODUCER_FILES):
        raise EvidenceError("producer snapshot is malformed")
    actual=[]
    for path in root.rglob("*"):
        if path==root/"snapshot.json":continue
        if path.is_symlink():raise EvidenceError("producer snapshot symlink is not accepted")
        if path.is_file():actual.append(str(path.relative_to(root)))
    if set(actual)!=set(PRODUCER_FILES):raise EvidenceError("producer snapshot inventory differs")
    for name,identity in value["files"].items():
        relative=Path(name)
        if relative.is_absolute() or ".." in relative.parts:raise EvidenceError("producer snapshot path escapes")
        path=root/relative
        if set(identity)!={"bytes","sha256"} or not path.is_file() or path.stat().st_size!=identity["bytes"] or sha256(path)!=identity["sha256"]:raise EvidenceError("producer snapshot artifact changed")
    return {"path":_relative(root),"snapshotSHA256":sha256(root/"snapshot.json")}


def freeze(prepared: Path, tokenized: Path, prompt_tokenized:Path, producers:Path, output: Path) -> int:
    if output.exists() or output.is_symlink():
        print(f"EvidenceError: output already exists: {output}", file=sys.stderr); return 1
    try:
        p = validate_prepared(prepared)
        t = tokenize_corpus.validate_corpus(tokenized, model=MODEL)
        prompts=prompt_corpus.validate_corpus(prompt_tokenized)
        prompt_source_path=prepared.resolve()/"scored-prompts.json"
        if prompts["source"]!=read_json(prompt_source_path) or Path(prompts["index"]["source_manifest_path"]).resolve()!=prompt_source_path or prompts["index"]["source_manifest_sha256"]!=p["scoredPromptsSHA256"]:
            raise EvidenceError("prompt corpus does not bind exact prepared scored prompts")
        expected_prompt_ids=[x["id"] for x in prompts["source"]["documents"]]
        if [x["id"] for x in prompts["index"]["documents"]]!=expected_prompt_ids or len(expected_prompt_ids)!=1164:
            raise EvidenceError("prompt corpus ID order or count differs")
        if Path(t["index"]["source_manifest_path"]).resolve() != prepared.resolve() / "teacher-corpus.json" or t["index"]["source_manifest_sha256"] != p["teacherCorpusSHA256"]:
            raise EvidenceError("tokenized corpus does not bind prepared teacher corpus")
        executable = t["index"]["executable_identity"]
        if Path(executable["path"]).resolve() != ADAPTER.resolve() or executable["binary_sha256"] != "d3b0e7ffbcaf89186a8097a4dcd65493c595cf506adf7ef5098d787fc4a9c095":
            raise EvidenceError("tokenized corpus did not use immutable Plan007 adapter")
        config=read_json(MODEL/"config.json");eos=config.get("text_config",{}).get("eos_token_id")
        if type(eos) is not int or not 0<=eos<248320:raise EvidenceError("pinned EOS token ID is unavailable")
        prepared_data=read_json(prepared.resolve()/"suite.json");limits=[]
        settings=prepared_data["finalScored"]["settings"]["benchmarks"]
        by_id={x["id"]:x for x in prompts["index"]["documents"]}
        for category in ("gsm8k","humaneval","xnli","structured"):
            for case in prepared_data["finalScored"][category]:
                count=by_id[case["id"]]["prompt_count"];maximum=settings[category]["maxNewTokens"]
                if count+maximum>2048:raise EvidenceError(f"scored case exceeds model context: {case['id']}")
                limits.append({"id":case["id"],"promptTokens":count,"maxNewTokens":maximum,"contextLimit":2048,"eosTokenIDs":[eos]})
        manifest = {"format":"slotstream-quality-suite-v2","schemaVersion":2,"qualification":False,
                    "prepared":{"path":_relative(prepared),"suiteSHA256":p["suiteSHA256"],
                                "teacherCorpusSHA256":p["teacherCorpusSHA256"],
                                "scoredPromptsSHA256":p["scoredPromptsSHA256"],"contractSHA256":p["contractSHA256"]},
                    "tokenized":{"path":_relative(tokenized),"corpusSHA256":sha256(tokenized.resolve()/"corpus/corpus.json"),
                                 "completionSHA256":sha256(tokenized.resolve()/"corpus/completion.json"),
                                 "documents":len(t["index"]["documents"]),"positions":t["positions"],
                                 "tokenizerIdentity":t["index"]["tokenizer_identity"]},
                    "promptTokenized":{"path":_relative(prompt_tokenized),"indexSHA256":sha256(prompt_tokenized.resolve()/"corpus/prompts.json"),
                    "documents":1164,"promptTokens":prompts["prompt_token_count"],"sourceSHA256":p["scoredPromptsSHA256"],
                    "tokenizerIdentity":prompts["index"]["tokenizer_identity"],"limitsSHA256":canonical_hash(limits),"limits":limits},
                    "adapter":{"path":_relative(ADAPTER),"binarySHA256":executable["binary_sha256"],
                               "archiveReceiptSHA256":executable.get("archive_receipt_sha256")},
                    "counts":p["counts"],"finalScoredCounts":p["finalCounts"],
                    "producerSnapshot":validate_producer_snapshot(producers),"overallQualification":False,
                    "unavailable":["qualification inference","scored task execution","HumanEval sandbox",
                                   "population non-inferiority"]}
        manifest["manifestSHA256"] = canonical_hash(manifest)
        atomic_json(output, manifest); return 0
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr); return 1


def validate_manifest(path: Path):
    value = read_json(path)
    required = {"format","schemaVersion","qualification","prepared","tokenized","promptTokenized","adapter","counts","producerSnapshot",
                "finalScoredCounts","overallQualification","unavailable","manifestSHA256"}
    if set(value) != required or value.get("format") != "slotstream-quality-suite-v2" or value.get("schemaVersion") != 2 or value.get("qualification") is not False or value.get("overallQualification") is not False:
        raise EvidenceError("suite manifest is malformed")
    if value["manifestSHA256"] != canonical_hash({k:v for k,v in value.items() if k != "manifestSHA256"}):
        raise EvidenceError("suite manifest hash differs")
    def bound(name):
        relative=Path(name)
        if relative.is_absolute() or ".." in relative.parts:raise EvidenceError("suite path escapes repository")
        result=ROOT/relative
        if not result.resolve(strict=True).is_relative_to(ROOT.resolve()):raise EvidenceError("suite path escapes repository")
        return result
    prepared, tokenized, adapter = (bound(value["prepared"]["path"]),bound(value["tokenized"]["path"]),bound(value["adapter"]["path"]))
    p, t = validate_prepared(prepared), tokenize_corpus.validate_corpus(tokenized, model=MODEL)
    if value["prepared"] != {"path":value["prepared"]["path"],"suiteSHA256":p["suiteSHA256"],
                             "teacherCorpusSHA256":p["teacherCorpusSHA256"],
                             "scoredPromptsSHA256":p["scoredPromptsSHA256"],"contractSHA256":p["contractSHA256"]}:
        raise EvidenceError("prepared suite identity differs")
    if value["tokenized"]["corpusSHA256"] != sha256(tokenized/"corpus/corpus.json") or value["tokenized"]["completionSHA256"] != sha256(tokenized/"corpus/completion.json") or value["tokenized"]["positions"] != t["positions"]:
        raise EvidenceError("tokenized suite identity differs")
    if value["counts"]!=p["counts"] or value["finalScoredCounts"]!=p["finalCounts"] or value["tokenized"]["documents"]!=len(t["index"]["documents"]) or value["tokenized"]["tokenizerIdentity"]!=t["index"]["tokenizer_identity"] or Path(t["index"]["source_manifest_path"]).resolve()!=prepared.resolve()/"teacher-corpus.json" or t["index"]["source_manifest_sha256"]!=p["teacherCorpusSHA256"]:
        raise EvidenceError("suite summaries or teacher tokenized join differ")
    prompt_root=bound(value["promptTokenized"]["path"]);prompts=prompt_corpus.validate_corpus(prompt_root)
    if value["promptTokenized"]["indexSHA256"]!=sha256(prompt_root/"corpus/prompts.json") or value["promptTokenized"]["documents"]!=1164 or value["promptTokenized"]["promptTokens"]!=prompts["prompt_token_count"] or value["promptTokenized"]["sourceSHA256"]!=p["scoredPromptsSHA256"] or value["promptTokenized"]["tokenizerIdentity"]!=prompts["index"]["tokenizer_identity"] or prompts["source"]!=read_json(prepared/"scored-prompts.json"):
        raise EvidenceError("prompt-tokenized suite join differs")
    prepared_data=read_json(prepared/"suite.json");by_id={x["id"]:x for x in prompts["index"]["documents"]};eos=read_json(MODEL/"config.json")["text_config"]["eos_token_id"];limits=[]
    for category in ("gsm8k","humaneval","xnli","structured"):
        maximum=prepared_data["finalScored"]["settings"]["benchmarks"][category]["maxNewTokens"]
        for case in prepared_data["finalScored"][category]:limits.append({"id":case["id"],"promptTokens":by_id[case["id"]]["prompt_count"],"maxNewTokens":maximum,"contextLimit":2048,"eosTokenIDs":[eos]})
    if value["promptTokenized"]["limits"]!=limits or value["promptTokenized"]["limitsSHA256"]!=canonical_hash(limits) or any(x["promptTokens"]+x["maxNewTokens"]>2048 for x in limits):raise EvidenceError("scored prompt limits or EOS IDs differ")
    producer=validate_producer_snapshot(bound(value["producerSnapshot"]["path"]))
    if producer!=value["producerSnapshot"]:raise EvidenceError("producer snapshot identity differs")
    if adapter.resolve() != ADAPTER.resolve() or sha256(adapter) != value["adapter"]["binarySHA256"] or value["adapter"]["archiveReceiptSHA256"]!=t["index"]["executable_identity"].get("archive_receipt_sha256"):
        raise EvidenceError("adapter identity differs")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); commands = parser.add_subparsers(dest="command")
    p = commands.add_parser("prepare"); p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("validate-prepared"); p.add_argument("--prepared", type=Path, required=True)
    p = commands.add_parser("snapshot-producers");p.add_argument("--output",type=Path,required=True)
    p = commands.add_parser("freeze"); p.add_argument("--prepared", type=Path, required=True); p.add_argument("--tokenized", type=Path, required=True);p.add_argument("--prompt-tokenized",type=Path,required=True);p.add_argument("--producer-snapshot",type=Path,required=True); p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("validate"); p.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare": return prepare(args.output)
        if args.command == "validate-prepared": validate_prepared(args.prepared); return 0
        if args.command == "snapshot-producers":return producer_snapshot(args.output)
        if args.command == "freeze": return freeze(args.prepared,args.tokenized,args.prompt_tokenized,args.producer_snapshot,args.output)
        if args.command == "validate": validate_manifest(args.manifest); return 0
        parser.error("choose prepare, validate-prepared, freeze, or validate")
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr); return 1


if __name__ == "__main__": raise SystemExit(main())
