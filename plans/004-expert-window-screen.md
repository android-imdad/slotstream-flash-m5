# Plan 004: Measure whether a recent-token expert window reduces JANG traffic

## Current disposition — September 15, 2026

**COMPLETE — candidate windows rejected.** Native CLOCK replay reconciled exactly; no recent-window policy met admission.

[Evidence/current findings](004-review.md) · [Canonical status](../db/records/plan/jang-flash-qualification-status.md). The original execution brief and dated review notes below are retained as history; current work is governed by [the plan index](README.md). Do not restart completed or unselected steps from a historical instruction.

> Execute this bounded development study after the reviewed M5 diagnostic work. It implements the measurement/replay gate of parent Plan 001 stage 2, not a new runtime eviction policy. Preserve the model and defaults. The reviewer maintains the index; commit only scoped tooling in the isolated worktree, never merge or push.

## Status and context

- DONE at development-study scope: the final complete cohort reconciled exactly; best window saved2.076% decode bytes and failed the10% gate. No runtime candidate selected. See 004-review.md for current evidence; earlier blocked attempts below remain historical.
- Live verification BLOCKED: latest independent preflight observed 8.48 GB reclaimable against the required 17 GB; no model PID started. Offline tooling approved and committed as `d4bb01fac3a7c04dd5e1d427b090542609a6d5e0`. All 84 Python tests and 14 replay self-tests passed independently. See `004-review.md`; actual CLOCK parity and a window decision remain pending.
- Priority P1; effort M; risk LOW for offline replay, MED for monitored model runs.
- Baseline source: reviewed Plan 003 commit `99de033fbc9f037cf459ec6ac8de64f3d1a874dc`, 2026-09-12. Its additions are diagnostic-only and preserve inference math/defaults.
- Worktree: `/Users/imdad/Documents/Codex/2026-09-12/wha/outputs/slotstream-flash-m5`, branch `advisor/001-flash-m5`.
- JANG_6S is complete at `models/jang-6s` (read-only link to the original checkpoint). Pinned revision is `3781190c6bbdf0a7637beda49ba179822612058a`. Plan 002's baseline summary and archive live under `.build/flash/runs/baseline-summary-final.json` and `baseline-v5/`.
- Initial model target remains 14 decimal GB, context 2048, MTP/vision off, fresh preflight >=17 GB, one model/GPU job at a time. Do not stop unrelated apps or simulators.

## Why

The baseline generated 32 tokens while requesting 28,080,742,400 expert bytes: about 0.88 GB per output token at a 793-slot cache. The paper's windowing idea may help at expert granularity without changing model math. Before implementing it, measure whether it beats the actual CLOCK policy at the same cache byte budget. Do not turn simulated byte savings into a tokens/s claim.

## Existing mechanisms and invariants

- `Sources/Slotstream/RouterTrace.swift` already records `SLOTSTREAM_ROUTER_TRACE=<path>`: little-endian int32 layer/tokens/topK followed by tokens*topK int16 expert IDs. The buffer is in memory and writes once at exit; write failure is swallowed. This study bounds the entire trace by a fixed short prompt/output envelope and independently requires a complete file.
- `MoELayer.cached` in `Layers.swift` deduplicates experts in first-use order before `SlotPool.ensureChecked`. `ExpertStore.swift:ensureCore` resolves the whole batch, pins all hits, reserves all victims, reads bounded slices, publishes only complete slots, then `unpinAll` occurs at the next layer. A flat per-expert LRU simulation is not the native policy.
- `ExpertStore.swift:victim` implements CLOCK; read its exact hand/ref-bit/pin behavior. `sourceRecordBytes(layer:)` sums original tensor row bytes, whereas `recordBytes` is the expanded cache representation. JANG_6S cache records are 4,300,800 bytes; original source sizes vary by layer/projection quantization.
- Prefill sweeps bypass the normal pool and can admit selected experts afterwards. To avoid pretending a router-only trace describes those mutations, this first study uses prompts whose **actual rendered token count is below 64**, disabling neither math nor existing controls. Reject a run that enters sweep/read-scope/MTP paths the replay does not support. Do not silently discard prefill or partial steps.
- Existing `Tools/cachesim.py` uses original 2.7648 MB records and flat LRU/LFU simulations; `Tools/trace_convert.py` can drop incomplete steps. Read them as historical context, not an acceptance oracle.

## Scope

Create `Tools/flash/cache_study.py`, `Tools/flash/replay.py`, focused `test_cache_study.py`/`test_replay.py`, and small `Tools/fixtures/flash/cache-study.json`. Extend existing `Tools/flash` receipt/gate helpers only for this study. Artifacts under `.build/flash/`. No Swift, runtime policies, package changes, checkpoint writes/repacking, neuron masking, or defaults in this subtask.

## Steps

1. Freeze three small public development prompts in the fixture before running: the existing sky-blue sentence prompt, a request for a short Python list-deduplication function, and a short explanation of multiplying 37 by 42. Exact UTF-8 strings, seed/options, expected maximum ASCII length, maximum 128 output tokens, and fixture hash belong in the manifest. Actual rendered prompt IDs/counts come from stats and must be <64. These are development samples, not parent Plan 001's final quality corpus.

2. Add `cache_study.py collect --model <path> --output <fresh-dir>`. For each prompt, run an untraced and traced arm in separate monitored processes, using the same binary/configuration/budget, greedy generation and stats JSON. The traced arm sets only `SLOTSTREAM_ROUTER_TRACE` to its own fresh output path. Record that environment explicitly and capture/hash it. Use the foundation launcher; generation failure, missing stats/trace or changed token IDs invalidates the pair. No speed claim from these two runs. Limit trace size to 1 MiB and require complete ordered layer groups. Preserve every failure.

3. Derive original expert bytes per layer by reading only bounded safetensors headers and quantization metadata from the verified checkpoint. Bind file/header/model identities and reject unsupported geometry/dtypes. Price source bytes separately from the uniform six-bit/FP32 cache. Compare trace totals with actual `prefillRecords`, `decodeRecords`, `prefillReadBytes`, `decodeReadBytes`, and effective slot count from the stats JSON; never parse the CLI's legacy human-readable byte estimate.

4. Implement a strict trace parser and a native CLOCK replay. Validate headers before allocating: 48 layers, topK10, expert IDs 0..511, first prefill group followed by one-token decode groups, complete ordering, no trailing/partial bytes, total trace <=1 MiB. Warm the cache by replaying prefill batches. Deduplicate each batch in first-use order, pin every existing demand hit before selecting victims, reserve every miss before publication, and reproduce the native ring/ref-bit rules. Honor stats' actual cache-policy controls; reject unsupported nondefault policies rather than simulate a different one. Derive decode-forward count from the recorded generator finish behavior and trace, and reconcile it explicitly; visible output count alone is not a forward-count oracle.

5. Prove replay fidelity against each live trace: native CLOCK hits/misses and source bytes must agree exactly with the engine totals in both phases. If they do not, fix the replay or report a bounded missing-observation gap; do not benchmark alternative policies on an unvalidated model.

6. Compare recent-token eviction preferences for windows 1, 2, 4 and 8, with identical cache capacity. Last-use ages use actual token positions, not layer visits. For prefill batches, use each expert's latest position within the batch. Recent membership is an eviction preference; only current demand keys are pinned. If all candidate victims are recent, fall back to the native bounded CLOCK choice. No lookahead into future trace rows. Account for the age metadata's bytes separately; if it would displace a slot, use the reduced candidate capacity or include a matched-capacity control. Include an explicitly labeled clairvoyant bound only as optional context, never as a runtime candidate.

7. Write `cache_study.py analyze --collection <dir> --output <fresh-dir>` with per-prompt prefill/decode requested bytes, miss counts, policy overhead estimate, and honest sample limitations. Advance a candidate to a subsequent runtime plan only if it saves >=10% decode demand bytes pooled across the fixed cohort without increasing any prompt's decode bytes >5%, at the same total cache+metadata budget. Otherwise mark the window experiment rejected for this cohort. Do not add a runtime flag or mark parent Plan 001 complete from this offline screen.

## Tests and verification

Use independent small hand-calculated traces: empty/truncated/malformed inputs, invalid IDs/layer order, duplicate demand IDs, all hits, all misses, pin-before-victim behavior, full CLOCK wrap, recent-window overflow, token/layer epoch distinction, cache smaller than window union, request reset, and varying source record sizes. Test that a deliberately wrong flat replay fails native-total reconciliation. Include trace-write omission and partially written stats as hard failures. Never mutate real source files for fixtures.

```sh
python3 -m unittest discover -s Tools/flash -p 'test_*.py'
python3 Tools/flash/replay.py --self-test
python3 Tools/flash/cache_study.py collect --model models/jang-6s --output .build/flash/runs/window-collection
python3 Tools/flash/cache_study.py analyze --collection .build/flash/runs/window-collection --output .build/flash/runs/window-screen
```

Expected: positive test counts with zero required skips; all three complete traced/untraced pairs have identical IDs; source bytes and native CLOCK totals reconcile exactly; reports contain one explicit candidate/reject decision and all provenance. Use current build identities (or an explicitly validated frozen historical binary) and fresh directories. Parent study/model qualification remains false without its final run-set.

## Done / STOP

Done means a tested replay and an evidence-backed development decision, with a scoped worktree commit. A rejected window is a useful result and must not trigger unrequested model pruning. Stop on mismatched native totals after a reasonable correction, unsafe headroom, unsupported execution paths, trace corruption, source drift, or an out-of-scope requirement. Return STATUS, STEPS/commands, STOPPED BECAUSE if needed, FILES CHANGED and NOTES with commit/evidence paths; reviewer owns the index.
