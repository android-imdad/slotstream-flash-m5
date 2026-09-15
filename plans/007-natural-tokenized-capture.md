# Plan007: Freeze natural text into bounded capture shards

## Current disposition — September 15, 2026

**DONE.** Natural tokenization/capture bridge accepted at 0b032fa; the reference archive remains immutable.

[Evidence/current findings](README.md) · [Canonical status](../db/records/plan/jang-flash-qualification-status.md). The original execution brief and dated review notes below are retained as history; current work is governed by [the plan index](README.md). Do not restart completed or unselected steps from a historical instruction.

## Status and purpose

DONE at reviewed source commit `0b032fa8b9ded947c16e9b62ddf4e897dcb3f49a`. The review and real-data evidence below qualify only this tokenizer/capture bridge.

P1, effort M, risk MED. Execute in `/Users/imdad/Documents/Codex/2026-09-12/wha/outputs/slotstream-flash-m5`, branch `advisor/001-flash-m5`. Reconciled source baseline `e5107113db58b05b27b4d49b623c049c5ee9351b`; that commit adds reviewed plans only, with implementation unchanged from `7f63f0b`. Dependencies002,005,006 are DONE. The reviewer owns plan edits; the user now explicitly authorized committing plans. No merge/push.

Parent001 next needs natural training/development/final corpus identities. The accepted v1 capture adapter permits arbitrary diagnostic IDs only. Do not relabel natural corpus inputs as arbitrary/non-training data or replace the existing reference. This step creates a tokenizer-only bridge and versioned capture support; full external-suite selection, quality metrics, masking and predictor training follow separately.

## Read-only recon

- Engine.encodeChatWithoutModel loads AutoTokenizer only. Its current overload implicitly requests addGenerationPrompt=true. Preserve that default, but allow an explicit false for a completed assistant turn.
- swift-transformers1.3.3 is pinned at2fa33e1f5e7131a7fc64c28e6d161dcec0d24820. Tokenizer reads tokenizer.json and tokenizer_config.json; the embedded template is in tokenizer_config.json. Bind exact files, extracted template bytes, dependency pin and native tokenizer binary.
- Current FlashCapture/capture.py v1 accepts reference-off/on, development only, <=64 scored positions, <=256 warmup and <=2048 context. It is immutable in `.build/flash/runs/diagnostic-reference/`, binary SHA2564ce7f4e05ba3e374dd790f88d05b15587f6339588796188af84218a1f838638b.
- A separate v2 adapter must retain v1 compatibility and prove computational parity against that anchor. The anchor cannot directly read v2/training inputs. Explicitly identify anchor vs capture adapter; never rebuild/overwrite the anchor.

## Scope

Sources: Engine.swift (small tokenizer facade/defaulted option only), new `Sources/slotstream-cli/Corpus.swift`, registration in main.swift, versioned reader/metadata changes in Flash.swift; small package diagnostics in Diagnostics+Flash.swift/T0Checks if needed. No Model/Layers/ExpertStore/planning/math changes.
Tools: new Tools/flash/tokenize_corpus.py,test_tokenize.py; versioned schemas and small authored fixtures under Tools/flash/schemas and Tools/fixtures/flash; focused capture.py/test_capture.py/gates.py extensions. Keep existing v1 fixtures/readers intact. Ignored artifacts `.build/flash/`. No Python inference dependency, external model download, package pin change, masks, runtime defaults or server changes.

Reconciled prerequisite follow-up: the real cohort `tokenizer-anchor-parity-final-v2/adapter-v2-off` completed native inference with exit0 but exposed terminal rusage before the first WNOWAIT child-exit event. The previous Plan006 sampler correctly rejected the incomplete join. Permit a focused `Tools/flash/benchmark.py` and `test_launcher.py` fix: after the sampler specifically reports process exit, wait at most100ms in5ms bounded sleeps for a non-reaping exit observation, then take the terminal sample and reap once. Preserve cancellation and overall timeout, PID/status checks, failure on no confirmation, no accepting a live-zero footprint, and no pre-confirmation poll/reap. Test delayed confirmation, deadline exhaustion and cancellation independently. Preserve the failed cohort; fresh evidence only after the source and harness freeze.

## Implementation

1. Add `slotstream flash-tokenize`: strict bounded local source manifest, existing pinned JANG model directory and fresh output. It must load only the tokenizer, with no Engine/model/pool or network fallback. Reject missing tokenizer files first. Support raw text documents and text-only supervised chat pairs (optional system, user, reference assistant); no images/tools/reasoning. Bound source file<=8MiB, each text<=32KiB, <=2048 documents, safe unique ASCII IDs<=64 chars. Reject unknown keys, malformed roles, invalid limits before tokenizer work. Inputs include category, split(training/development/qualification), source ID/license/hash and partition key; the bridge preserves these identities but does not claim to validate external dataset provenance.

2. For raw text, freeze exact tokenizer IDs and explicit warmup/scored range. For chat, render prompt with addGenerationPrompt=true and completed prompt+assistant with false, thinking=false; require prompt IDs to be an exact prefix of complete IDs. Do not repair a failed prefix by inventing tokens. Score the first assistant token correctly: warmup IDs are fullIDs[:promptCount-1], first input is fullIDs[promptCount-1], target is fullIDs[promptCount]. Subsequent inputs equal previous targets. For raw text use the declared warmup offset with the same i→i+1 convention. Refuse insufficient text or oversized warmup/context; no silent padding/truncation or dropped documents. Store actual full IDs/counts and selected range so unscored text is explicit.

3. Write versioned tokenized-corpus index and capture shards. Each complete document window is1...64 scored positions, warmup<=256, totalcontext<=2048. Each shard totals<=64 positions and contains one split. Do not reconstruct long contexts by pretending independent shards share state. Bind exact source manifest/file hashes, normalized text/message hashes, source/licensing metadata, all IDs/ranges, tokenizer/config/template/pin, executable/metallib/build/source archive identities, output hashes and deterministic shard order. Fresh contained outputs and atomic completion after all files close. Recompute canonical hashes; never trust caller hash strings alone. Preserve failures. Bound aggregate output/metadata before allocation or stream it.

4. Add an explicit v2 tokenized capture schema alongside v1. Native and host validation must agree on every count/type/path/ID/offset. Verify shard hash and its tokenizer/model binding before allocation. Allow training and development capture; reject qualification capture unless a later locked-runset implementation explicitly enables it (this plan provides no bypass). Keep v1 behavior and old archive untouched. Record category/split/source identities in v2 outputs. No changes to math, model state formulas, ordinary CLI/serving or default numerical controls.

5. Host `tokenize_corpus.py freeze` uses the terminal-qualified launcher at <=1GB process target, separate from the full model, and validates native/tokenizer provenance and every output shard. Add `tokenize_corpus.py validate` and an explicit `--binary` archive input for durable freezes. Corpus validation verifies the archived executable/metallib/source hashes independently of later candidate source changes; it must not require the current checkout to equal the archived tokenizer. Never mutate a pre-existing output when refusing overwrite. Pure tests use injected tokenizers; a real tokenizer run proves tokenization requires no model allocation. The fixtures must be honest small authored texts with a declared license; they are not the parent's final quality suite.

6. Prove the new adapter: a fresh build passes existing native/Python gates; old immutable v1 anchor vs new v1 capture yields exact logits/routes/state/IDs on the existing v1 fixture. New v2 on/off produces exact same computational payload on small real-text fixtures. Record both comparisons explicitly; do not pretend the old binary parsed v2. Archive the new adapter separately as `diagnostic-reference-corpus-v2` only after reviewer approval. Full-model jobs stay serial, fixed2-second settling,14GB target and>=17GB preflight, terminal-before-reap evidence required. Keep source/binary/harness fixed during each cohort.

## Verification

Tests: independent tokenization stub for first-assistant-token/i+1 offset; prefix mismatch; insufficient text; cross-split/duplicate IDs; unknown/nested keys; long names/files/context; bounded metadata; malformed/foreign-tokenizer shards; mutation after freeze; missing or reordered shards/completion; qualification capture refusal; v1 backwards compatibility. Real tokenizer fixture should exercise raw English/code plus a multilingual/chat pair, with exact text/token hashes. No final qualification inference or external generated-code execution.

Implement and run:

```sh
python3 -m unittest discover -s Tools/flash -p 'test_*.py'
make build SLOTSTREAM_BUILD_JOBS=2
python3 Tools/flash/tokenize_corpus.py freeze --binary .build/flash/runs/diagnostic-reference-corpus-v2/bin/slotstream --model models/jang-6s --source Tools/fixtures/flash/tokenize-development.json --output .build/flash/runs/tokenized-development
python3 Tools/flash/tokenize_corpus.py validate --corpus .build/flash/runs/tokenized-development
python3 Tools/flash/capture.py anchor-parity --model models/jang-6s --anchor .build/flash/runs/diagnostic-reference --corpus .build/flash/runs/tokenized-development --output .build/flash/runs/tokenizer-anchor-parity
```

Provide exact required native/host gate counts and current identities. A valid tokenizer result proves IDs/provenance, not model quality. A small v2 parity result does not complete parent001's >=512-position development corpus or >=4096-position final corpus. Those remain next explicit work.

## Done / stop

Done: strict tested tokenizer bridge, validated frozen natural-text shards, versioned reader with qualification gate, full anchor/on-off parity, scoped commit after review. Stop on tokenizer-prefix incompatibility, missing hashes, unsafe resource use, changed arithmetic, or inability to prove v1 parity. Continue independent host checks. Return STATUS/STEPS/STOPPED BECAUSE/FILES CHANGED/NOTES with actual evidence, no model speed/quality claim. No merge/push.

Implementation naming reconciliation: use `tokenize_corpus.py` for the helper. Naming a module `tokenize.py` in Tools/flash shadows Python standard-library tokenize whenever that directory is on sys.path; a reviewer subprocess reproduced linecache.getlines raising AttributeError for missing tokenize.open. Remove the shadowing file rather than retaining an importable compatibility shim. The new command has not shipped. Update imports/tests and keep this native-tokenizer bridge naming distinct from stdlib.

## Final review

Approved after two revision rounds. The fixes align canonical JSON bytes and strict input schemas, preserve completed outputs on retry, bind frozen corpora to independently verified archived tokenizer assets, and handle delayed WNOWAIT visibility after terminal process accounting. No model arithmetic, default inference options, weights or package pins changed.

The reviewer rebuilt and passed the Flash component gate with131 host tests, all50 T0/T1 native groups with27519 assertions and no skips, and the complete static/installer/planner/brain gates. Five independently malformed native inputs were refused before model allocation, with measured peaks below5.5MB. The corrected standard-library-shadow regression inserts Tools/flash before importing tokenize/linecache. The archived corpus remains valid independently of later candidate-source changes; archive mutation fails validation.

Accepted v2 adapter archive: `.build/flash/runs/diagnostic-reference-corpus-v2`, binary SHA256 `d3b0e7ffbcaf89186a8097a4dcd65493c595cf506adf7ef5098d787fc4a9c095`. The immutable v1 diagnostic reference remains untouched. The reviewer preserved the subsequent rebuild's separate source/build receipt bytes in `.build/flash/runs/reviewer-tokenizer-verified-build`; its executable is identical. These two archives have different build/source-archive receipt hashes because the archive wrapper is recreated by each build. Always join the exact recorded asset set, not a moving .build/release path.

Reviewer corpus: `.build/flash/runs/reviewer-tokenized-development-final`, index SHA256 `70d6efa755991d666094f127497f0ce05007b282643af0094a7032b6b9969253`. Five authored documents contain20 selected positions across training/development/qualification; the12-position development shard includes English, multilingual text and a supervised chat. Chat promptCount33 uses32 warmup IDs, so the first scored target is the first assistant token. Qualification capture remains locked. This is not parent001's full quality corpus.

The independent five-process cohort `.build/flash/runs/reviewer-tokenizer-anchor-parity-final` passed v1 anchor/new-adapter parity and v2 observer off/on parity, including complete logits, ordered routes, state identities and continuation/target IDs. Report SHA256 `4da418ca4d8653788f80e523e29f5a8299529154d047315ef901a9e3fda8531f`; maximum observed physical peak10440119864B under the14GB target. Direct full-logit byte and argmax checks also validated the executor cohort, recorded in `reviewer-tokenizer-evidence-check.json`. Every model receipt has the terminal-before-reap memory evidence. No speedup or final model-quality claim follows from these controls.
