# Plan 005: Establish the bounded diagnostic reference for neuron experiments

## Current disposition — September 15, 2026

**DONE at bounded scope.** The frozen logit/route/state reference and observation parity were accepted; later corpus and oracle work completed separately.

[Evidence/current findings](005-review.md) · [Canonical status](../db/records/plan/jang-flash-qualification-status.md). The original execution brief and dated review notes below are retained as history; current work is governed by [the plan index](README.md). Do not restart completed or unselected steps from a historical instruction.

## Status

- Priority P1; effort L; risk MED. DONE at bounded scope; accepted archive and independent terminal-qualified parity are recorded in 005-review.md.
- Prerequisites: reviewed Plan 002 foundation and Plan 003 diagnostic work. Independent of whether Plan 004 finds a winning cache window.
- Reconciled at `d4bb01fac3a7c04dd5e1d427b090542609a6d5e0`, 2026-09-12. The intervening commit adds reviewed cache-study tooling only; native seams remain unchanged.
- Execution checkout: `/Users/imdad/Documents/Codex/2026-09-12/wha/outputs/slotstream-flash-m5`, branch `advisor/001-flash-m5`. The reviewer owns plans. Do not merge or push.
- This is a bounded part of parent Plan 001 stage 1. It does not complete the shared corpus, all profiling scopes, predictor/oracle, or final qualification by itself.

## Purpose

Before omitting any neurons, preserve an independently validated reference that can expose complete logits, ordered routes, and defined recurrent state. Add bounded activation collection for the later SwiGLU block study. This adapter must leave ordinary inference unchanged when absent, and must prove observation on/off parity on actual model calls before supplying training data.

Memory recovered above the parent's 17 GB full-model preflight threshold on continuation. Recheck immediately before each model process. Reference freezing requires actual model parity; do not mark the prerequisite complete from synthetic tests.

## Existing source and design constraints

- `Model.swift` exposes `lastLogitsChecked`, `makeState`, `routerObserver`, and package `contextNumericsObserver`. Use one consumed token per scored teacher-forced position and normal persistent state. Consuming token i predicts i+1; test this offset with an independent tiny fixture. Reset state between documents.
- `State.diagnosticTensors()` already exposes logical recurrent tensors, written KV prefixes, indexer ranges, n-gram history, token count and optional draft fields. Include `diagnosticIndexerBases()` in the serialized state description. Hash shapes, original dtypes and defined contiguous bytes in stable key order, including explicit absent fields; never hash uninitialized spare capacity or convert state to FP32 before hashing. Release each snapshot before the next model call.
- MLX `asData(access: .copy)` preserves dtype and creates a contiguous copy; `.noCopy` may include noncontiguous strides and has a strict source lifetime. Use bounded slices/copies with a documented lifetime, not an escaping pointer.
- `Layers.swift:cached` computes gate/up, `silu(g)*u`, then down. Resident overlap can call the nested project function on only some router ranks; collection must carry exact rank/expert IDs for those rows. Do not infer ranks from callback order. `Model.swift` already exposes the MoE input through the x2 context stage.
- Existing grouped prefill and terminal pruning can omit irrelevant terminal rows. The first activation adapter supports one-token scored forwards only, after a declared unchanged warmup. Reject unsupported activation shapes before a forward rather than silently omitting rows. Later prefill-kernel qualification remains separate.
- `Engine(modelDir:plan:)` owns normal checkpoint guards, model validation, a 2 GiB MLX allocator cache cap, and fixed JANG policy. Reuse it. Do not construct an unbudgeted bare model. Existing native `ModelProcessGuard` provides the owned model lock.
- `ModelArgs.announcedPlan` supports fixed JANG geometry and `--mtp off --vision off`. Avoid its interactive missing-weight path: the adapter requires a complete existing checkpoint before allocation.

## Scope

Allowed source: new `Sources/Slotstream/Flash/FlashObservation.swift`, new `Sources/SlotstreamDiagnostics/Diagnostics+Flash.swift`; minimal observation wiring in `Model.swift` and `Layers.swift`; explicit diagnostic reservations in `ContextMemory.swift`, `JANGPlanning.swift` and `Plan.swift` if required; CLI additions in `Sources/slotstream-cli/Flash.swift` and registration in `main.swift`; exact catalogue registration in `T0Checks.swift`.

Allowed tools: `Tools/flash/capture.py`, `test_capture.py`, small capture manifest schema/fixtures under `Tools/flash/schemas/` and `Tools/fixtures/flash/`, necessary existing evidence/gate helper extensions. No predictor, masking, changed expert/kernel math, custom Metal, dependency edits, checkpoint writes, runtime/default activation, server changes, or public speed claims.

## Implementation

1. Implement pure versioned request/position/output schemas with unknown-key rejection, integer overflow bounds, fresh contained output paths, fixed maximum context2048, explicit input IDs/next-token IDs and warmup positions, document IDs, split and manifest hash. Initial fixtures are disjoint authored development examples, never a substitute for the parent's frozen final suite. The native command rejects qualification claims and approximate modes.

2. Add a synchronous bounded sink with a maximum64MiB aggregate live buffer, fixed quotas (activation32GiB/logits8GiB defaults, individually lowerable), short-write/EINTR/error handling, cancellation and deterministic close. Validate lengths and quota before copies/materialization. Stream full-vocabulary FP32 logits in at most16-position chunks and activation rows with full IDs. A dry-run computes positions and worst-case bytes without allocating the model. Finish atomically only after all required chunks/hashes/counts are present. Failure retains a partial, unqualified receipt.

3. Add a per-model optional throwing diagnostic observer; absent means no added tensor materialization or retention. Explicit activation mode records x once per layer/token and exact hidden rows for selected routed ranks. Resolve complete event geometry before output. Do not rerun projection math to synthesize a trace. Default functions keep the same arithmetic order/shapes. Drain/close before teardown; a sink failure aborts the diagnostic and the partially advanced state cannot be reused.

4. Price the declared buffer, copies/scratch, and capture state hashing inside the same14GB target. Preserve fixed allowance/margin and dense minimum640slots. If adding a ledger reservation is required, use default-zero optional initializer inputs and checked byte arithmetic; old plans/defaults must decode and behave as before. Actual allocations plus overlapping copies must fit the reservation. Do not treat unused disk quota as allocated memory. Follow every MemoryPlan copy path needed by this diagnostic; a reservation must not silently disappear. Add a minimal factual refusal message listing observed RAM, Metal working set, reclaimable memory and requested target in JANGPlanning; leave all admission inequalities unchanged. This is needed to diagnose two current native refusals while host preflight reported 25–26 GB. Do not infer which observation caused them.

5. Implement `slotstream flash-capture` and the monitored `Tools/flash/capture.py` orchestrator. Required arguments: existing model, immutable manifest, split, mode reference, memory/context, position/disk/live limits and fresh output. All actual inputs/options/source/binary/metallib/model verification IDs appear in native and launcher evidence. Host orchestration validates receipts rather than trusting native status strings. No full model jobs overlap. Fresh preflight>=17GB applies. Add a fixed 2-second pause between model processes in the diagnostic capture and cache-study orchestrators, record it outside model timing, and test it with injected sleep/clock where applicable. Apple's public XNU host.c defines a 1-second randomized rate-limit cache for third-party host_statistics calls, with platform binaries exempt; this explains the observed vm_stat/native discrepancy as a supported hypothesis. Numerical memory guards remain unchanged; require a new complete cohort to test the mitigation. Extending Tools/flash/cache_study.py for this shared diagnostic orchestration policy is in scope.

6. Establish observation correctness with separate serial processes: adapter observer off/on on identical stored IDs and shapes must produce identical full logits, ordered routes, defined retained-state bytes and greedy continuation IDs. State/reference provenance comes from this diagnostic build. Separately compare the old archived CLI and new CLI output IDs/stats under unchanged short generation configuration, and review unchanged arithmetic source. Do not pretend the old CLI supplied full-state dumps. Add bounded original-row/synthetic checks before full model work.

7. After complete actual-model parity, freeze a new immutable `diagnostic-reference` archive and reference outputs. Future candidate builds cannot replace it. If safe headroom or parity is unavailable, retain tested implementation but leave diagnostic-reference unaccepted and downstream numerical experiments blocked. Parent stage1 remains incomplete until shared corpus/metrics and required cost observations also exist.

## Verification

- Native `flash-observation` and `flash-identity` checks: positive independent assertions for bounded sink quotas, malformed dimensions/offsets, short writes/cancellation, no completion on partial output, defined-state bytes, rank mapping with split resident/miss paths and no-op observer. Include old-default ledger equivalence if the ledger changes.
- Python `CaptureTests`: complete position accounting, missing/duplicate/out-of-order chunks, incorrect i+1 target, unknown schema fields, output hash mismatch, foreign reference archive, quota rejection, partial receipt and cancellation, no-live-model dry-run.
- Model observer parity must compare the actual arrays/state, not only readable text or two calls to the same serializer.

Expected commands, to implement before invoking:

```sh
python3 -m unittest discover -s Tools/flash -p 'test_*.py'
python3 Tools/flash/capture.py --self-test
python3 Tools/flash/capture.py dry-run --manifest Tools/fixtures/flash/capture-development.json --split development --output .build/flash/runs/capture-dry-run
make build SLOTSTREAM_BUILD_JOBS=2
.build/release/slotstream-checks --tier t0 --tier t1 --json
python3 Tools/flash/capture.py parity --model models/jang-6s --reference .build/flash/runs/reviewer-m5-archive --manifest Tools/fixtures/flash/capture-development.json --memory-gb 14 --max-context 2048 --output .build/flash/runs/capture-parity
```

Use a new strict bounded gate name rather than pretending all parent stage1 checks exist. Required positive counts, zero required skips, current build identities, preserved failures, and exact parity are mandatory. Archive only after parity and reviewer acceptance; preserve the accepted binary before subsequent builds.

## STOP and handoff

Stop the affected arm on unsafe headroom, geometry or dtype drift, inability to bound copies, changed default math, unmatched counts/state/logits, unavailable source identity, or an out-of-scope prerequisite. Continue independent small tests. Return STATUS / STEPS / STOPPED BECAUSE / FILES CHANGED / NOTES with actual evidence, not inferred passes. Reviewer reruns the gates and owns plan status; a host-tested adapter without actual reference parity is not DONE.
