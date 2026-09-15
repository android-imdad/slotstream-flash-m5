# Plan010: Freeze complete task prompts without inference

## Current disposition — September 15, 2026

**DONE.** Complete task-prompt tokenization accepted in bc185df, with exact overlap checks.

[Evidence/current findings](README.md) · [Canonical status](../db/records/plan/jang-flash-qualification-status.md). The original execution brief and dated review notes below are retained as history; current work is governed by [the plan index](README.md). Do not restart completed or unselected steps from a historical instruction.

## Status and dependency

P1; effort M; risk MED. DONE at reviewed source commit `bc185df` after one revision round. Core source baseline0b032fa, plan commitf61730d. Depends on DONE007; supplies008's exact scored-prompt IDs and later009 diagnostic generation. Same isolated worktree/branch. Reviewer owns plans. No merge/push.

The gap is confirmed: `flash-tokenize` requires a scored teacher-forced window and <=256 warmup, whereas complete task prompts need their own IDs without an assistant reference or a scoring window. `template-check` only encodes a hard-coded conversation at main.swift:1605–1630. Do not insert dummy references, pretend prompt tokens are scored answers, or relax007's teacher-forcing limits. This is a separate tokenizer-only command, not a generation path.

## Ownership

Prompt executor owns new `Sources/slotstream-cli/Prompts.swift`, its registration in main.swift, `Tools/flash/prompt_corpus.py`, `test_prompt_corpus.py`, prompt-source/output schemas and a small explicitly synthetic fixture. Reuse existing `CorpusSupport` and `SlotstreamTokenizer` without modifying Engine/Model/Layers/ExpertStore/Corpus.swift or their numerical behavior. No weights/pins/default/run/serve changes. The quality executor owns corpus.py/metrics.py/eval_env.py/their tests and gates.py; coordinate any eventual gate registration through that executor, never edit gates.py concurrently. Both work in the same isolated checkout with disjoint source ownership. Preserve all prior reference archives and rejected008 evidence.

## Native interface and format

Add `slotstream flash-tokenize-prompts --model <pinned6S> --source <json> --output <fresh .build/flash directory>`. Load only the existing local pinned tokenizer; no Engine, pool, model, network fallback or generated output. Bind the same tokenizer/config/template/swift-transformers pins and executable/metallib/build/source/archive identities as007. Reuse its proven canonical JSON rules, hash verification and contained fresh-output helpers.

Input format `slotstream-prompt-source-v1`,schemaVersion1,manifestSHA256,documents; reject unknown fields/types. Common document fields: id,kind,category,split,sourceID,license,partitionKey. `id`/category safe ASCII1...64; split is training/development/qualification; other common fields nonempty <=256 UTF8 bytes, with sourceID controls refused as in007. For raw, add only text/textSHA256. For chat, add only system/user/contentSHA256; system is an explicit possibly empty string, user nonempty, and contentSHA256 hashes canonical `{system,user}`. There is no assistant reference, scoreTokens or warmupTokens. External provenance/partition metadata is preserved, not certified by this bridge. Duplicate IDs and cross-split partition keys fail.

Bound source<=8MiB,documents1...2048,each text<=32KiB,each emitted prompt1...8192 valid IDs (<248320),aggregate<=4000000 IDs and<=128MiB output. The8192-token bound is a tokenizer artifact limit, not permission for inference beyond the experiment's2048 context. Record full counts; no padding/truncation/drop. Stream one prompt at a time and retain only bounded index metadata; do not build a corpus-sized boxed-token array before checking quotas.

Raw encoding is addSpecialTokens=false. Chat encoding uses exactly optional nonempty system then user, addGenerationPrompt=true,thinking=false,tools absent, through the pinned template. Do not synthesize/render the template in Python.

Write `prompts/prompt-0000.json` through ordinal2047 in source order, plus `prompts.json` index and atomic completion only after every file closes. Each document file includes its exact source metadata/content hash, all prompt IDs, count and deterministic token-ID hash (canonical compact JSON integer array). Index records exact source path/byte hash, tokenizer/build/archive identity, ordered document IDs/paths/counts/hashes and aggregate counts/bytes. Index flags `model_loaded=false`, `tokenizer_only=true`, `qualification=false`; name the tokenization/context distinction. File and index schemas must cover strict nested fields. Refuse reused/escaped/symlinked outputs without modifying them; preserve only this attempt's partial failures. Bound reads and validate hashes/types before allocation.

## Host interface and validation

`prompt_corpus.py freeze --binary <immutable-or-fresh-binary> --model <model> --source <json> --output <fresh-dir>` runs the native tokenizer with the existing monitored launcher at<=1GB, fixed wall limit120s, and terminal-before-reap evidence. Validate a fresh binary against its source map; archived binaries against their exact archive/asset hashes independently of future candidate source changes. The stable corpus uses an immutable archive after approval. Never alter the v1 or corpus-v2 reference.

`prompt_corpus.py validate --corpus <dir>` validates source canonical/content hashes and complete source set, exact ordered prompt coverage, types/counts/vocabulary/ID hashes, contained regular paths, all artifact hashes, tokenizer/model identity, binary/source/archive receipt, launcher command/output joins, memory/completion and absence of partial success. Do not accept a renamed arbitrary ID list without the bound native tokenizer evidence. No missing/extra/duplicate files/records or caller-supplied success flags. Plain system Python must run pure tests; no NumPy dependency.

## Verification and acceptance

Write meaningful tests for raw/chat source fields and canonical hashing (slashes,quotes,newlines,Unicode/combining text), strict roles/unknown fields/nulls, duplicated IDs/partitions, bad IDs/counts/hashes, changed source/tokenizer/archive, omitted/reordered/extra prompts, missing completion, overwrite/outside-root refusal and long/aggregate quota failure. Tests use small independently constructed fixture output and injected dependencies; no model download or private corpus dependency.

Real tokenizer proof: freeze a small synthetic input containing raw English/code/French/Chinese plus chat, including one complete prompt longer than257 tokens and one short prompt with no possible teacher-forced target. Explicitly label boundary repetitions as synthetic quota tests, never as quality corpus diversity. Prove measured process peak<=1GB and no model allocation.

For the overlapping007 raw/chat source texts, compare new raw full IDs to007 full IDs and new chat IDs to007 full IDs[:promptCount], including its exact first-answer boundary. The old archive need not parse the new format. Preserve both binary identities and all native/tokenizer receipts. A deliberately wrong template/pin or one-token mutation must fail.

Run `python3 -m unittest discover -s Tools/flash -p test_prompt_corpus.py`, a two-job release build, existing required native/Flash gates, and repository static gates. Coordinate the brief final harness freeze with the quality executor so no tool code changes during accepted evidence. No full-model job is required for this tokenizer-only change. Root will re-run the controls, approve/revise, then authorize the separate `.build/flash/runs/prompt-tokenizer-v1` archive and scoped source commit. Provide the exact output contract/example to the quality executor so008 can bind all actual scored-prompt IDs.

Done: strict tested tokenizer-only prompt bridge, stable native/archive provenance, exact007 overlap and long-prompt proof, reviewed commit. Not done: generation, task scoring, oracle masks, sparse I/O or acceleration. Return STATUS/STEPS/STOPPED BECAUSE/FILES CHANGED/NOTES with actual tests/identities and limitations.

## Final review evidence

Root independently passed15 host tests, required native Flash2/2 and complete repository static/installer gates. It repeated archived-native tokenization and checked exact007 overlap:16 English IDs,14 code IDs and33 chat-prefix IDs, including the old first-answer boundary. The synthetic long prompt has300 IDs; the short one has1. All385 IDs were preserved without model allocation.

Accepted immutable archive `.build/flash/runs/prompt-tokenizer-v1` contains binary SHA256a9e21a9aa0be99a74d620bf9270f876e60dfb8a5f4cf3170546666c280fe2609, archive receipt SHA25629b9e385ec37199b6347dd3e75cc5502b87666bf30d65e63b55e6b9dbbd9592e. Root proof `.build/flash/runs/reviewer-prompt-tokenizer-accepted` has index SHA256a2c4bffcde293677a66333aca175c45b254ce025fc96d373bdcc54adfee87956 and terminal-qualified peak171426512B. Review-pending/development copies remain historical artifacts. This is tokenizer compatibility and bounded-resource evidence only, not inference speed or model quality.
