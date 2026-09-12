# Plan 001: Adapt LLM in a flash to Qwen MoE streaming and qualify M5 acceleration

> **Executor instructions:** Read this entire plan first. Follow the dependency table, execute one bounded stage at a time, and record each gate. This is a staged research and implementation plan: a rejected experiment is a useful result, but is not a shipped speedup. Do not enable an unsuccessful experiment or weaken a gate to finish the plan. Update the status in `plans/README.md` unless a reviewer owns that index.
>
> **Drift check:** `git diff --stat 93fb512..HEAD -- Sources Tools Package.swift Package.resolved Makefile CLAUDE.md docs db/records plans`
> This intentionally covers the execution surface and its dependencies. Also inspect `git status --short`; compare the excerpts below against any changed files. If a prerequisite or invariant changed, reconcile the plan before implementation. Do not revert concurrent work.

## Status

- **Priority:** P1
- **Effort:** L — staged research, native implementation, and hardware qualification
- **Risk:** HIGH — numerical behavior, asynchronous buffer ownership, and process memory
- **Depends on:** no external prerequisite for host infrastructure or synthetic probes; full-model stages require complete verified JANG_6S weights and a successful baseline run
- **Category:** perf / direction
- **Planned at:** `93fb512`, 2026-09-12; upstream base `f60e9a8`
- **Plan revision:** 2, reviewed 2026-09-12; source baseline unchanged.
- **Implementation status:** IN PROGRESS in isolated worktree `../slotstream-flash-m5`, branch `advisor/001-flash-m5`; bounded foundation dispatch is `plans/002-flash-foundation-execution.md`. No full-model qualification claimed yet.
- **Target:** Apple M5 Max, 48 GiB unified memory, macOS 26.5.1 build 25F80. Verify again at execution time.

## Why this matters

Slotstream already runs a model larger than RAM by streaming routed experts and n-gram rows. The opportunity is to reduce bytes needed for each output and reduce the time spent loading or computing them, while retaining the quality of the selected JANG quant. The strongest starting points are exact cache/layout improvements and the M5 matrix kernels already present in MLX. Applying neuron-level sparsity is a separate, higher-risk experiment because this model uses SwiGLU and only 640 intermediate neurons per routed expert.

Success means measured improvement over this checkout's bounded JANG_6S implementation, with a reproducible quality and memory report. It does not mean reproducing the paper's published speedup or enabling a hardware label.

## Research basis and interpretation

**Paper summary:** Apple's [LLM in a flash, version 2](https://arxiv.org/html/2312.11514v2) combines a low-rank activity predictor, a recent-token window of retained neurons, bundled corresponding projection weights, and preallocated memory. Its evaluated models are OPT and a sparsified Falcon; the latter was adapted to ReLU. Predictor errors can change outputs. Its measurements are not a baseline for Slotstream or this Mac. See sections 3.1–3.3 and 4 for those distinctions. The implementation below is our proposed MoE adaptation, not code supplied by the authors.

**Hardware:** M5 has **Neural Accelerators inside its GPU cores**, accessed through Metal TensorOps / Metal Performance Primitives (MPP). Its **Apple Neural Engine (ANE)** is a separate device. MLX already uses the GPU accelerators for suitable matrix operations; Apple's [MLX/M5 report](https://machinelearning.apple.com/research/exploring-llms-mlx-m5) specifies macOS 26.2 or later. The primary hardware track in this plan is the GPU accelerators. ANE is a bounded optional predictor experiment, not the main expert executor.

Apple's [WWDC26 TensorOps session](https://developer.apple.com/videos/play/wwdc2026/330/) distinguishes features available in macOS 26 from new macOS 27 tensor formats and cooperative-tensor inputs. JANG's affine 6-bit weights and additive biases must not be treated as an interchangeable native MXFP4 format. Use supported dequantization followed by TensorOps where necessary; do not change the quantization to fit an API. Consult the [MPP guide](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf) for the SDK actually installed.

### Dependencies and first useful delivery

**Start with stages 0 and 1, then the independent stage-2 exact-loading and stage-5 M5 tracks.** Their combined report is the first useful delivery; it does not wait for predictor training, sparse storage, or ANE work. Stages are implementation units, not a requirement to run heavy jobs in parallel on this shared Mac.

| Stage | Prerequisites | Deliverable | Exit gate |
|---|---|---|---|
| 0 | Download complete for model work; launcher tests can start earlier | Monitored, immutable JANG_6S baseline | Hashes, smoke, component checks, footprint |
| 1 | 0 for full-model traces; synthetic M5 probes need no weights | Frozen diagnostic reference, strict receipt validator, measured cost breakdown | Observer parity, bounded capture, explicit dispatch status |
| 2 | 1 | Independently evaluated storage, expert window and causal heuristic prefetch | Exact parity; measured improvement or rejection per arm |
| 5 | 1; **no dependency on 3–4** for dense M5 work | Verify existing NAX; implement only a justified custom path | Dispatch and numerical/performance evidence, or already-used result |
| 3 | 1; independent of success in 2/5 | Development oracle, compact trained block predictor, frozen artifact | Predictor quality and cost pass on development data |
| 4 | Accepted predictor from 3 | Small verified bundle prototype, then sparse runtime | Bundle identity, sparse math, memory and development speed gates |
| 6 | Useful trained predictor from 3; optional | Separate Core ML/ANE predictor comparison | Complete integration cost beats CPU/MLX or record rejection |
| 7 | Completed evidence for the **selected** tracks | Locked final evaluation and user-facing report | Required gates for each selected claim pass |

Stage 5 may later evaluate sparse shapes after 4, but its dense work is independent. Stage 2 uses causal heuristics only in this plan; learned cross-layer expert prediction is deferred, eliminating a circular dependency on stage 3's different neuron predictor. Stage 7 can first qualify exact/M5 results, then run again under a **new run-set ID** if a sparse candidate becomes ready. Do not mix receipts between those decisions.

Use dispositions `accepted`, `rejected`, `blocked`, `already_used`, and `not_selected`, each with a reason and evidence. `already_used` is valid only for an observed existing hardware path; unknown dispatch is `blocked` for the usage claim. A rejected prerequisite makes its dependent experiment `not_selected`, not an unimplemented test failure. Never mark the overall plan DONE while a required selected track is blocked. Optional ANE can be `not_selected` without installing conversion tooling.

All new Flash execution modes stay **off by default**, including accepted exact/M5 paths. This plan delivers explicit opt-in local CLI/library experiments and evidence. Automatic activation, serving exposure, and default rollout require a later plan.

## Current state and code anchors

Repository root in the planning session: `/Users/imdad/Documents/Codex/2026-09-12/wha/outputs/slotstream`. All paths below are relative to that repository; commands run from its root.

### Runtime and model

- `Sources/Slotstream/Checkpoint.swift:18–31,70` fixes H=2560, 48 layers, 512 experts/layer, top-10 routing, intermediate size 640, and affine group size 64. Configuration checks reject incompatible geometry.
- `Sources/Slotstream/Layers.swift` owns routing, shared experts, cached expert computation, and grouped prefill. `Model.swift` integrates transformer state. `Engine.swift` owns request admission and memory boundaries.
- `ExpertStore.swift` performs bounded reads and maintains a global CLOCK pool. `PackedExpertLayout.swift` already provides verified whole-expert packing. `CacheBookkeeping.swift` contains cache bookkeeping. Do not create a second whole-expert packer without demonstrating a missing capability.
- `NgramStore.swift` / `NgramPrefetch.swift` already have a different, token-derived exact prefetch path. Leave their semantics alone.
- JANG is experimental, text-only, fixed-cache, with MTP and vision disabled. The full checkpoint was still downloading when this plan was written. Component gates are not evidence of complete-model quality or speed.

`Sources/Slotstream/Layers.swift:1091–1101`:

```swift
} else { logits = routerProjection(x) }
contextNumericsObserver?("router", logits)
let idx = RouterSelection.indices(logits, k: cfg.topK, enabled: specializedRouter)
let weights = softmax(takeAlong(logits, idx, axis: -1), axis: -1, precise: true)
let expertIds = idx.asType(.int32).asArray(Int32.self)
```

The real router remains authoritative. Expert prediction can start reads early; it must not replace these IDs, top-K, softmax, or rank order.

`Sources/Slotstream/Layers.swift:1253–1263`:

```swift
let g = gatherQuantizedMM(
    xe, pool.pools[0], scales: pool.pools[1], biases: pool.pools[2],
    rhsIndices: slotIdx, transpose: true, groupSize: cfg.qGroup, bits: cfg.expertBits)
let u = gatherQuantizedMM(
    xe, pool.pools[3], scales: pool.pools[4], biases: pool.pools[5],
    rhsIndices: slotIdx, transpose: true, groupSize: cfg.qGroup, bits: cfg.expertBits)
let hidden = MLXNN.silu(g) * u
return gatherQuantizedMM(
    hidden, pool.pools[6], scales: pool.pools[7], biases: pool.pools[8],
    rhsIndices: slotIdx, transpose: true, groupSize: cfg.qGroup, bits: cfg.expertBits)
    .squeezed(axis: -2)
```

This is SwiGLU. Negative intermediate values are generally nonzero. A magnitude threshold or predictor cannot silently classify them as exactly inactive.

`Sources/Slotstream/CheckpointFormat.swift:14–19`:

```swift
/// JANG_6S uses a uniform six-bit cache. Four-bit source codes are widened
/// exactly on a miss: scales, biases and represented weights are unchanged.
public var expertBits: Int { self == .jang6S ? 6 : 4 }
// JANG metadata is losslessly expanded from FP16 to FP32 in the cache.
// Otherwise MLX promotes the *whole* pool's metadata on every BF16 QMM.
public var expertRecordBytes: Int { 3 * (640 * 2560 * expertBits / 8 + 640 * 2560 / 64 * (isJANG ? 8 : 4)) }
```

`ExpertStore.swift:138–170` expands source metadata and widens 4-bit integer codes on a miss. `readBytes` at lines 946–948 tracks source payload bytes, not the larger cache representation. Preserve that distinction. CPU expansion is a measurement candidate, not an assumed bottleneck.

`ExpertStore.swift:191–195` defines, per projection, gate/up shapes `[640,2560]` and down `[2560,640]` before packing. Down scales and biases have shape `[2560,10]`: each covers 64 intermediate columns. A natural first sparse unit is therefore **64 neurons**, or ten blocks per expert. This avoids changing down-projection quantization groups. It is a design choice for this prototype, not a claim that finer lossless storage is impossible.

`Layers.swift:1390–1410` pads grouped prefill rows with `max(16,4 * group.count)`, repeats the final row/expert, and sets `sortedIndices: true` on all three gathered products. Results return to original router rank. `ExpertStore.swift:632–810` pins required keys, joins readers before slot writes, and publishes complete records. Preserve those ownership and numerical boundaries.

### Existing M5 path

`Package.resolved` pins mlx-swift 0.31.6 at `0bb916c67f4b9e5c682cbe02a42c701c93ab5021`; `Makefile` colocates a matching MLX **0.31.1** metallib. These are distinct version labels, not an inconsistency to fix.

Read-only dependency anchors under `.build/checkouts/mlx-swift/Source/Cmlx/mlx/mlx/`:

- `backend/metal/device.h:268–285`: `is_nax_available()` tests OS availability, GPU architecture, and the compile-time `MLX_METAL_NO_NAX` switch.
- `backend/metal/quantized.cpp:1370–1426`: `GatherQMM::eval_gpu` selects grouped RHS GEMM when `M == 1 && B >= 16 && right_sorted_ && B/E >= 4`; matrix and vector paths are separate.
- `backend/metal/quantized.cpp:1135–1137`: eligible grouped RHS work calls `gather_qmm_rhs_nax`.
- `ops.cpp:5130–5139,5180–5190`: affine gather promotes input and metadata to a common floating type. JANG's FP32 cached metadata matters to dispatch and arithmetic.
- `utils.h:163–165`: `MLX_ENABLE_TF32` defaults to 1 in this dependency. Record the effective value. Do not change it globally to obtain a better benchmark.
- `backend/metal/kernels/quantized_nax.h` handles several packed bit widths, including six. Kernel source availability still does not prove the running binary selected it.

This makes grouped prefill a plausible existing accelerator user. Single-token decode generally reaches the vector path. Confirm with an actual trace before reporting either as measured.

### Conventions and negative evidence

- Swift 5 language mode under Swift tools 6; use `package` for internal diagnostics hooks. Match `ModelError`, checked read APIs, and `CheckBuilder` patterns in `Diagnostics+JANG.swift` and `Diagnostics+PackedLayout.swift`.
- There is no `swift test` target. `slotstream-checks` is the executable catalogue in `Sources/SlotstreamTestKit/T0Checks.swift`; T0 is CPU bookkeeping, T1 small MLX arrays.
- `CLAUDE.md:420–438` requires bounded staging and reports that a previous background cross-layer read-ahead implementation was slower in every paired run. Existing shared/resident compute overlap and the in-layer sweep already cover useful overlap. New prediction must beat those paths with all overhead included.
- `Tools/cachesim.py` hardcodes the original 2.7648 MB record. Do not use its outputs for JANG without supplying actual per-format geometry in the new replay harness.
- `ExpertTransferProfile.swift` deliberately serializes portions of profiling; its timing fields are not additive with `pool.ioSeconds`. `RouterTrace.swift` is a useful trace format but buffers in memory. Long activation captures need a new bounded sink.
- `OptimizationPlatform.swift` qualifies one existing RoPE kernel only for an M5 Pro/OS combination. Do not relabel that as M5 Max or accelerator qualification.
- `CLAUDE.md` says work lands on main. Preserve other edits, make logical local commits only when executing this plan, and do not push or publish a release without a separate instruction. Example existing message: `Add experimental JANG checkpoint streaming support`.
- Canonical records live in `db/records`. `PLAN.md`, `MEASUREMENTS.md`, and `llms-full.txt` are generated. Update canonical records and run generators; do not hand-edit projections.

## Scope

The advisor creates only files under `plans/`. An executor may change only the following implementation surface, and only for the stages selected by their gates:

- Existing runtime: `Sources/Slotstream/{ExpertStore,CacheBookkeeping,PackedExpertLayout,Layers,Model,Engine,Optimizations,OptimizationPlatform,ExpertTransferProfile,Generate,ContextMemory,JANGPlanning,Plan,PrefixCache}.swift`.
- New runtime files under `Sources/Slotstream/Flash/`: `FlashConfiguration.swift`, `FlashObservation.swift`, `ExpertWindow.swift`, `ExpertPrefetch.swift`, `FlashPredictor.swift`, `NeuronBundleLayout.swift`, `NeuronBlockCache.swift`, `SparseExpert.swift`, `M5ExpertKernel.swift`, `CoreMLPredictor.swift`.
- New diagnostics: `Sources/SlotstreamDiagnostics/Diagnostics+Flash.swift`, `Sources/SlotstreamDiagnostics/Diagnostics+M5.swift`; catalogue registration in `Sources/SlotstreamTestKit/T0Checks.swift`.
- CLI: `Sources/slotstream-cli/Flash.swift` (new), registration and options in `Sources/slotstream-cli/main.swift`.
- Tools: `Tools/flash/` (new bounded capture/replay/calibration/benchmark tools and tests); `Tools/fixtures/flash/` (small generated fixtures and licenses). Reuse `Tools/prefill_bench.py` helpers without changing its original acceptance semantics.
- Documentation/evidence: `docs/FLASH.md` (new), `docs/JANG.md`, `plans/README.md`, `db/records/design/flash-moe-m5.md`, `db/records/measurements/flash-moe-m5-*.md`, `db/sources/runs/2026/09/flash-moe-m5-*`, and claims/projections generated by the established brain workflow where required.
- Ignored experiment assets: `.build/flash/` (including all run evidence), model-adjacent sidecars in `models/`. `.gitignore` already ignores `.build/` and `models/`; `bench/flash/` is **not** ignored. Never add model weights or large traces to git. Do not clean `.build` during qualification; archive accepted evidence through the canonical brain workflow first.

**Out of scope:** model downloads/manifests, original checkpoint contents, tokenizer, router arithmetic/top-K, GDN/attention formulas, n-gram algorithm, MTP/vision enablement, HTTP behavior, changing public default quantization, whole-model Core ML conversion, ANE private APIs, retraining or ReLU-converting Qwen, global system settings, dependency upgrades, and modifying checked-out dependencies in place. An isolated debug dependency copy for capture is allowed only inside `.build/flash/`; production package pins and metallib stay unchanged.

## Commands and execution constraints

Existing commands verified by source/CI recon; they are not new qualification results from this planning pass:

```sh
git status --short
git rev-parse HEAD
sw_vers
sysctl -n machdep.cpu.brand_string hw.memsize
make build SLOTSTREAM_BUILD_JOBS=2
.build/release/slotstream-checks --tier t0 --tier t1
PATH="/Users/imdad/Documents/Codex/2026-09-12/wha/work/bin:$PATH" Tools/static_gates.sh
python3 Tools/context_proxy.py --out .build/flash/context-proxy-001
.build/release/slotstream jang-check
```

Expected: build exit 0 and exact build-identity receipt; all selected checks pass; static gates pass; context proxy passes with `hardware_qualified=false`. Use a fresh proxy directory. The local dbmd path above is a planning-session convenience; if absent, use the repository's pinned `Tools/dbmd_install.sh` procedure. `Tools/verify.sh` needs the original PipeNetwork full checkpoint: a missing model is **not** a pass, and a JANG model is not a substitute.

All new commands below are **interfaces to implement**, not tools that already exist. Add `--help`, strict argument validation, nonzero exit on missing prerequisites, and machine-readable results. Tests must verify actual work/counts, not just a PASS string. Only test tools may offer `--self-test`; it must not mark hardware qualification true.

### Configuration and activation contract

Create a version-1 `FlashConfiguration` JSON schema in `Tools/flash/schemas/`. Resolve it once per engine: mode (`off`, `exact`, `numerical`, `approximate`), storage policy, expert window length, heuristic prefetch toggle, predictor hash/path, bundle identity/path, kernel policy, fixed dense crossover, and thresholds. Expose an optional `--flash-config <file>` on **run and diagnostic commands only**, plus an optional default-nil library configuration argument. `serve` rejects an active Flash configuration before allocation in this plan. No environment-only hidden activation; missing config means off. Reject unknown versions/keys, conflicting modes, unsafe paths and unsupported combinations before allocating models. Keep all new defaults off after qualification.

Numerical policy and loading policy are separate: `mode=exact` uses original arithmetic only; `mode=numerical` allows a named full-neuron custom kernel but no masks; `mode=approximate` additionally permits masks. Low confidence/unseen experts use the manifest's fixed dense rule. Missing or mismatched predictor at startup rejects an explicit approximate request with an actionable error; it must not claim approximate mode ran while silently disabling it. An explicitly configured fallback may select off before loading, with the effective mode reported.

`FlashExecutionIdentity` hashes the model revision, numerical mode/version, predictor weight/coverage/normalization hashes, all mask thresholds, dense-crossover/fallback policy, and selected kernel/dtype/TF32 policy. Include it in both ordinary prefix entries and `PromptCheckpointKey` in `PrefixCache.swift`; current fields do not cover these new artifacts. Default identity preserves existing behavior. Loading-only cache changes with proven identical math need not invalidate state. Changing a numerical identity recreates the engine or clears incompatible entries before reuse. Old serialized optimization controls omit the new optional configuration and decode as off; do not add required Codable fields that break saved controls.

Every run's `stats-json` and diagnostic receipt includes requested/effective configuration and identity, including fallback counts. CLI progress labels approximate execution explicitly. No server or model-name alias changes are required in this plan.

### Stage execution and evidence contract

Implement `Tools/flash/gates.py` in stage 0 and extend it as stages are selected. Before **every native verification after Swift changes**, it runs `make build SLOTSTREAM_BUILD_JOBS=2` and validates `build-identity.json` against the current compiled source snapshot, metallib, and executable. An old executable or an after-build source edit fails the gate. Archive before a later build replaces `.build/release`. Do not regenerate a frozen reference from candidate source.

`gates.py --stage <id> --output <fresh-dir>` runs only that stage's required checks, captures the native runner's `--json` output, and validates exact required names, zero required skips/failures, nonempty assertion lists, and receipt hashes. The existing native runner exits 0 on skipped checks; shell status alone is insufficient. Run Python suites through `unittest`'s programmatic loader, require the stage's named `TestCase` classes and a positive discovered test count, and reject required skips. Record discovered IDs/counts. Never create stub PASS tests for an unimplemented optional stage.

All conditional-stage command blocks below run only when that stage/arm is selected for implementation. Maintain `.build/flash/runs/selection.json` from the first stage with stage/arm IDs, selected flag, disposition, config path/hash, and prerequisite/evidence references; the harness initializes it and subsequent decisions append a dated record. `gates.py` reads this file by default (or `--selection <file>`) and derives required checks from the table. A deselected implemented runtime arm still needs its off/fallback tests; changing a selected flag cannot remove checks required by the effective configuration. Stage 5 marked `already_used` validates stage-1 dispatch receipts rather than invoking nonexistent custom-kernel tests; stage 6 `not_selected` executes no exporter.

Required groups (register exact names in `T0Checks.swift`; conditional names become required only when the corresponding arm is implemented):

| Stage | Required native groups | Required Python classes / responsibility |
|---|---|---|
| 0 | Existing `jang-formats`, `jang-numerics` | `LauncherTests`, `ReceiptTests`: owned-process lifecycle, failed sampling, immutable archive |
| 1 | `flash-observation`, `flash-identity`, `m5-dispatch` | `CaptureTests`, `RunSetTests`, `CorpusTests`, `MetricTests`: quotas, position counts, evidence joins and locked evaluation |
| 2 | `flash-window`, `flash-storage`, `flash-prefetch` for each selected arm | `ReplayTests`: actual quant bytes, causal replay, demand priority |
| 3 | `flash-oracle`, `flash-predictor` | `PredictorTests` plus shared corpus/metric tests: labels, training/round-trip, actual-mask quality |
| 4 | `flash-bundles`, `flash-block-cache`, `flash-sparse-math` | `BundleManifestTests`: offsets, hashes, bounds, corruption |
| 5 custom path | `m5-kernel`, `m5-fallback` plus `m5-dispatch` | `RunSetTests` with hardware/shape mismatch fixtures |
| 6 selected | `flash-predictor-backends` | `CoreMLExportTests`: identity and supported fixed shapes |
| 7 | All existing T0/T1 and the union required by selected tracks | All selected classes; `QualificationTests` rejects incomplete/mixed cohorts |

The full-model commands still need their own monitored receipts. Group `m5-dispatch` can pass its capability/fallback bookkeeping tests on unsupported hardware, while the **hardware usage claim remains blocked**. Synthetic checks never prove a live NAX/ANE dispatch.

All tools create fresh outputs and write `completion.json` atomically after outputs are closed and hashed. Required receipt fields: schema version, run-set/run/arm IDs, requested/effective config hashes, reference and candidate source/executable/metallib hashes, model verification identity, corpus/split/scorer hashes, actual prompt/output IDs/counts, device/OS, start/end, completion status, error/exclusion reason, declared output hashes and memory observations. Permit unavailable measurements only as explicit null plus a reason; never zero-fill missing samples.

`Tools/flash/runset.py create` freezes `runset.json`: selected tracks and prerequisites, reference archive, candidate identity per arm, permitted config differences per comparison, workload/case IDs, output lengths, memory/context policy, paired run order/seed, sample count, primary metric, quality thresholds and required receipts. `evaluate.py qualify --runset <file>` recomputes every artifact hash, checks identities and complete coverage, and rejects duplicate, missing, stale, incomplete, cross-model, or cross-split receipts. Expected candidate build/config differences are enumerated per arm; common model/tokenizer/corpus/reference identities must agree. Do not scan an arbitrary directory and combine whatever happens to be present.

Gate/qualify exit codes: 0 = the requested required gates passed, 1 = measured failure/rejected candidate, 2 = invalid or missing prerequisites/evidence. A research-summary command may exit 0 with rejected experiments recorded, but its `qualified` field stays false unless a nonempty selected runtime combination actually passed. A no-change report is useful and cannot count as a performance success.

### Resource policy for every full-model stage

- One model process at a time; honor `/tmp/slotstream-model-<uid>.lock` and current request admission. Never stop unrelated apps or model processes.
- Initial target **14 decimal GB**, context 2048, MTP off, vision off; preflight at least **17 GB reclaimable** immediately before launching, using `Tools.prefill_bench.preflight(17)`. This target prices the JANG layout; do not copy a smaller original-model budget blindly.
- Monitor child physical footprint and lifetime peak at 0.5-second intervals, plus MLX allocations and request-boundary observations. Terminate only the owned child on target exceedance or a 900-second smoke timeout. A sample interval alone is not proof no transient peak occurred.
- Keep the existing 2 GB MLX allocator cache cap, bounded prefill cache, maximum two staging groups **across demand and prefetch together**, and request rollback behavior. Prefetch acquires one of those two credits; it cannot add a third group. Charge predictor weights, activation traces, speculative buffers, JIT workspaces, Core ML residency, and old/new overlapping allocations to the same total budget.
- Add explicit Flash reservations to `ContextMemoryLedger` before sizing the dense pool: block-cache bytes, predictor residency, trace buffer, additional scratch and kernel workspace. Leave the existing conservative fixed allowance and margin intact; do not credit an assumed saving twice. Require `baseWithoutDensePool + densePool + flashReservations + margin <= target`, preserve the 640-slot dense minimum, and reject an unsupported sparse plan before allocation. Roll back reservations on every error path. Integer byte arithmetic uses the repository's checked helpers.
- The launcher also bounds its own buffers and includes them in headroom planning. A very short child that exits before any valid sample is a functional result with **unqualified memory**, not a proven zero peak. Each owned PID is bound to its start identity to avoid PID reuse. Retained state/predictor allocations from auxiliary framework services need explicit accounting or an unqualified memory result.
- Benchmarks need controlled power/thermal observations and no competing model or large download/verification/repack I/O. Wait for the current download and smoke pipeline to finish before timing. No `purge`, memory-pressure generators, swap disabling, or whole-model memory mapping.
- A fresh process empties Slotstream caches, not the OS file cache. Log `F_NOCACHE` / read-ahead configuration results where used; report application-requested bytes and process disk-I/O counters separately. Global swap counters are observations, not attribution to this model or automatic functional failure.

### Shared quality protocol — implement with stage 1

Use three disjoint document-level splits: **training**, **development**, and **final qualification**. Freeze their manifests in stage 1 before numerical-kernel or sparsity experiments. Training fits weights; development chooses ranks, masks, thresholds and crossover; final qualification is untouched until one candidate configuration is frozen. Initial development screening uses at least 512 teacher-forced positions across prose, code, math, multilingual and structured text, covering all 48 layers. Final qualification uses at least 4096 predicted positions across at least 100 documents, with at least 20 documents and 512 positions in each category. Hash normalized text and reject overlap, including translated versions of the same premise. Low-traffic/unseen experts use dense fallback; report their coverage rather than extrapolating predictor quality.

Implement `Tools/flash/corpus.py` to create and validate `Tools/fixtures/flash/suite.json`. Freeze complete revisions, file SHA256s, licenses, case IDs, category, split assignment, exact chat template/prompt, tokenization hash, maximum generation lengths, parser/scorer version, expected answer/tests and denominator. Raw external corpora live in `.build/flash/corpora/`; the checked-in manifest and small synthetic examples identify their immutable bytes. The XNLI code commit alone does not pin its external archive. The following are **final qualification cohorts**, never tuning data:

| Category | Immutable source / selection | Metric |
|---|---|---|
| Math | [GSM8K](https://github.com/openai/grade-school-math/tree/3101c7d5072418e28b9008a6636bde82a006892c), `grade_school_math/data/test.jsonl`, first 200 rows in file order | One greedy completion/case; normalized final numeric answer exact match; no calculator |
| Code | [HumanEval](https://github.com/openai/human-eval/tree/6d43fb980f9fee3c892a914eda09951f772ad10d), all 164 tasks ordered by numeric task ID | One greedy completion/task; fraction passing its official tests |
| Multilingual | [XNLI](https://github.com/facebookresearch/XNLI/tree/9624695001060244cf8d5d893b5236093a21a42c), official test archive, first 200 premise IDs each in English, French and Chinese | Exact entailment/neutral/contradiction label accuracy, reported separately per language |
| Structured output | 200 deterministic repository-authored JSON extraction/tool-argument cases, seed 001, generated and hash-frozen before tuning | Strict schema **and** expected values; malformed/missing outputs fail |

Build training/development cohorts from public training/dev splits and distinct repository-authored examples. Partition source documents by `SHA256(normalizedText) mod 5`: remainder 0 is development, others training; translated premises share one partition key. Exclude all final task IDs/texts. Add distinct authored prose/code documents to meet the final teacher-forced document counts before any candidate results; store them with their licenses/hashes. No final task's reference solution becomes a training example. Final document selection/order is frozen in the manifest; the first-200 task selections above are intentionally a fixed suite, not a random sample of all future user workloads.

The finite-suite scored gate is at most 1 percentage-point observed decline per category/language, with exact case outcomes and denominators reported. **Do not claim population non-inferiority from a degenerate paired bootstrap.** With 164 HumanEval cases, even zero observed regressions cannot tightly bound a sub-1% unseen regression rate. Report counts of baseline-only successes, candidate-only successes, both and neither, plus a one-sided exact binomial upper bound on baseline-only regressions; gains do not hide that risk. Broader non-inferiority is explicitly unproven by this suite. Unscored prose uses KL/perplexity, not a fabricated accuracy metric.

Generated code must run in an isolated test environment with no user-file/network access and hard limits; if unavailable, code qualification is incomplete. Incorrect, malformed and model-length-truncated responses count as failures in their original denominator. An infrastructure timeout, missing case or scorer failure invalidates the cohort instead of being selectively omitted. Freeze case generation lengths/timeouts and model reasoning-template settings, including EOS behavior, before any runs.

**Freeze/validate once in stage 1 (new commands):**

```sh
python3 Tools/flash/corpus.py freeze --output Tools/fixtures/flash/suite.json
python3 Tools/flash/corpus.py validate --manifest Tools/fixtures/flash/suite.json
```

Expected: source/scorer/tokenizer hashes are fixed; required cases and document counts are present; all three splits are disjoint. `freeze` refuses to overwrite a manifest. `validate` fails on changed external bytes, missing cases, or split leakage. Later stages validate and reuse this manifest; they do not recreate it after results.

## Steps

### Step 0 — Establish the complete-checkpoint baseline

1. First implement only the host-side `Tools/flash/benchmark.py` launcher scaffolding, before modifying runtime code. Provide `--self-test` and `launch --output <fresh-directory> --memory-gb 14 --max-seconds 900 -- <argv...>`. Reuse the existing preflight/process-cleanup helpers; add external `proc_pid_rusage` footprint/lifetime-peak observation. Launch failures, failed samplers, timeout, nonzero child exits, and exceeded budgets produce nonzero launcher exit and a preserved result. Self-tests use tiny fake children/injected samplers; they never generate real memory pressure. A sampler unavailable after startup aborts the owned run rather than certifying its memory.
2. Confirm the ongoing download and smoke pipeline have finished before starting another model. The planning-session pipeline status is at `/Users/imdad/Documents/Codex/2026-09-12/wha/work/jang-6s-trial/status.json`; use its completed evidence if still applicable, but do not depend on that private script for future reproducibility.
3. Verify `models/jang-6s` against JANGQ-AI's revision `3781190c6bbdf0a7637beda49ba179822612058a`. Do not accidentally use synthetic sparse fixture files. Preserve a baseline executable, metallib, build-identity receipt, source archive, and model verification receipt in an ignored fresh `.build/flash/runs/baseline/` directory before changing runtime code. Store the executable with its colocated metallib in `baseline/bin/`.
4. Capture a short generation under the resource policy. Preserve output token IDs, timing, stderr, physical footprint, and complete command/environment allowlist. Run native original-row and bounded expert checks as well. A sensible sentence is only a smoke result, not a proof of model parity.
5. If the complete baseline is broken, stop experimental work, report the failure, and fix/qualify the separate JANG integration before continuing. Do not establish goldens from a failing implementation.

Implement the host-only archive path at this stage too: `benchmark.py archive --binary <path> --output <fresh-dir>` copies the executable and colocated metallib plus build/source receipts into `bin/`, recomputes hashes and verifies the compiled source snapshot. `baseline/bin/slotstream` is created before source changes; use the same command in stage 1 to create `diagnostic-reference/bin/slotstream`. It must not run `make clean` or rebuild the reference on its own.

**Verify (existing CLI plus new monitored launcher):**

```sh
.build/release/slotstream pull JANG_6S --dir models/jang-6s --verify
.build/release/slotstream doctor --model models/jang-6s --memory-gb 14 --max-context 2048 --mtp off --vision off --json
python3 Tools/flash/benchmark.py --self-test
python3 Tools/flash/gates.py --stage 0 --output .build/flash/runs/stage-0-checks
python3 Tools/flash/benchmark.py archive --binary .build/release/slotstream --output .build/flash/runs/baseline
python3 Tools/flash/benchmark.py launch --output .build/flash/runs/baseline/expert-check --memory-gb 14 --max-seconds 900 -- .build/release/slotstream jang-check --model models/jang-6s
python3 Tools/flash/benchmark.py launch --output .build/flash/runs/baseline/smoke --memory-gb 14 --max-seconds 900 -- .build/release/slotstream run --model models/jang-6s --memory-gb 14 --max-context 2048 --mtp off --vision off --prompt 'In one sentence, explain why the sky is blue.' --max-tokens 48 --greedy --sample-footprint --stats-json .build/flash/runs/baseline/smoke/stats.json
```

The first two commands already exist; the remaining commands use the newly implemented launcher, which creates its fresh output directory before starting the child. Expected: all file hashes match, doctor accepts the real device, launcher self-tests pass, expert checks pass, generation exits 0 with nonempty finite outputs, and the external peak is within target. Preserve the actual measured values; this plan supplies no expected tokens/s.

### Step 1 — Add bounded measurements and establish M5 dispatch

Implement `FlashObservation` as a per-engine optional observer and extend the step-0 launcher with the benchmark modes below. The default runtime incurs no tensor materialization or retained trace allocations. Use explicit diagnostic modes rather than an always-on activation hook.

For latency presets, freeze prompt IDs from `Tools/fixtures/optimization/{short,prose,code}.txt`: the short fixture as-is and deterministic prose/code prefixes of 256 and 1024 tokens, followed by 128 generated tokens. Reject a fixture shorter than its requested prefix rather than padding silently. Use the same stored IDs and prefill policy in each arm. Benchmark modes default to `.build/release/slotstream` as candidate and require the frozen `.build/flash/runs/diagnostic-reference/bin/slotstream` as reference once that build exists; the baseline mode has only one arm. The preset manifest must list actual IDs/counts, original text/hash, and resolved options. Timed request reuse uses fresh state for its first request and an explicit same-prefix second request, not a second independent process described as warm.

- Record per-layer demand/hit/miss keys, requested source bytes, read call sizes/counts, staging bytes, metadata expansion/widening time, scatter work, compute shape, dtypes, quant bits, and waits. Keep overlapping timing scopes labeled; do not sum them into an invented critical path.
- Activation collection is a separate, intentionally intrusive run. Capture `x` and `silu(g)*u` for a bounded sample of routed experts, after the exact router decision, with token/layer/expert IDs. Use a bounded sink (at most 64 MiB buffered and a declared disk quota), stream chunks, and join/close it on cancellation. Do not retain full layer activations across the model or include user/private prompts.
- Add a small `m5-check` diagnostic using the actual expert shapes and the baseline FP32-metadata contract. Exercise 4-bit, 6-bit, and mixed-source widened records; batches 1, 4, 16, 64, and 256; sorted grouped prefill and unsorted one-token decode. Report effective TF32 setting, device/OS, MLX pin, metallib hash, eligible path, and observed path separately.
- Provide mutually exclusive `m5-check --synthetic` and `--model <path>` modes. Synthetic mode uses bounded random tensors in those shapes with at most 256 MiB live allocation and no checkpoint requirement; it can establish device/kernel capability before the download finishes. It cannot establish real-model quality or end-to-end speed. Model mode reads only bounded original expert samples; the full-model benchmark is a separate monitored command.
- Use `GPU.startCapture` / `stopCapture` where available. Its local Swift documentation requires an MLX debug build, `MTL_CAPTURE_ENABLED=1`, and a fresh destination. An isolated debug dependency copy may enable the capture hooks; never mutate `.build/checkouts` or replace production assets. If this build/toolchain cannot expose kernel names, record dispatch as **unverified**, not inferred from source or speed. A debug trace proves that diagnostic build's dispatch; bind it to a production-equivalent build or explicitly retain that limitation.
- Expected NAX kernel family includes `affine_gather_qmm_rhs_nax_nt_..._b_6_...`; report actual names, not this template as an observation. Source names alone do not measure accelerator utilization percentage. Add a one-layer trace before a full prefill trace to bound capture cost.

Before any optimization, freeze a second **diagnostic reference build** with only these observation adapters and all new execution modes off. `Diagnostics+Flash.swift` exposes a bounded teacher-forced runner using `Qwen4ExpModel.lastLogitsChecked` one position at a time with the normal persistent state; use the same prompt prefill shapes/configuration in both arms. At document position i, consuming token i produces logits predicting token i+1; store both IDs/positions and test the offset with an independent tiny fixture. Reset state between documents, exclude unscored warmup positions explicitly, and never cross the 2048 context cap to reach a corpus-wide count. It streams full-vocabulary FP32 logits in at most 16-position chunks, ordered routes, and hashes of defined live state bytes. The archive from step 0 has no state-dump API: compare its CLI output IDs and existing stats to this build, validate observer on/off logits and state identity within the diagnostic build, and review that its arithmetic source is unchanged. State explicitly that full-state reference evidence comes from this frozen diagnostic build. Future candidates compare against that preserved build and its immutable outputs, never a freshly rebuilt reference that includes candidate changes.

Provide `Tools/flash/capture.py` to orchestrate the monitored child, and `slotstream flash-capture` as its native adapter. Require a fixed corpus manifest, `--mode reference`, model directory, memory/context settings, capture byte/position limits, and output directory. Record each layer input once with routed-expert references, rather than ten duplicate copies of x. Supply finite default quotas (64 MiB live buffer, 32 GiB activation files, 8 GiB logit files); a dry-run reports required bytes/positions before launching. It writes a completion receipt only after all expected positions and hashes exist. Do not use `allLogitsWithMulti` on an unbounded sequence. Any exhausted quota stops capture with a partial/unqualified receipt; a predeclared smaller cohort cannot be substituted after candidate results are seen.

**Verify (new):**

```sh
python3 Tools/flash/gates.py --stage 1 --output .build/flash/runs/stage-1-checks
python3 Tools/flash/benchmark.py archive --binary .build/release/slotstream --output .build/flash/runs/diagnostic-reference
python3 -m unittest discover -s Tools/flash -p 'test_*.py'
.build/release/slotstream-checks --tier t0 --tier t1 --filter flash
.build/release/slotstream m5-check --model models/jang-6s --output .build/flash/runs/m5-baseline
python3 Tools/flash/benchmark.py baseline --model models/jang-6s --memory-gb 14 --max-context 2048 --output .build/flash/runs/baseline-measured
```

Expected: bounded-sink overflow and cancellation tests pass; tracing on/off preserves logits, ordered routes, and retained state bytes; at least one real dispatch receipt or an explicit unverified reason; the full baseline report contains source/binary/model identities and a complete monitored result. Time untraced runs separately from captured/serialized runs.

### Step 2 — Implement only promising exact loading improvements

Create a deterministic replay harness in `Tools/flash/replay.py`. Compare the live CLOCK policy with an expert-level recent-token window using identical JANG cache **bytes**, not equal record counts across formats. Include full 48-layer order, current-demand pins, request reset, prefill pollution, prefix reuse, and cancellation. The window unit is token position per request; layer visits do not advance it 48 times. Evaluate windows 1, 2, 4, and 8, clipped by the actual memory allowance. Recent membership is an eviction preference, not a reason to exceed capacity or pin every historical expert. Preserve the existing per-layer demand lifetime; do not pin the union of all 48 layers until a token ends. Replay the baseline policy against a real short trace and match its hits/misses/bytes before comparing alternatives.

Reviewer bounded-header recon on 2026-09-12 found source expert records of 3,174,400 bytes (40 layers), 3,584,000 (7), and 3,993,600 (1), versus the uniform 4,300,800-byte cache record. Across all 512 experts and 48 layers this is 79,901,491,200 original expert bytes versus 105,696,460,800 expanded packed bytes: 32.28% more payload before considering read-call savings. This is a geometry calculation, not a measured layout-performance result; complete checkpoint size also includes nonexpert weights.

Benchmark the existing `PackedExpertLayout` against raw reads first. It already bundles nine pieces of a whole expert, but stores the expanded pool representation. Price that extra disk traffic and space explicitly. If raw source layout is faster, retain it. Only add a versioned source-native whole-expert representation if profiling shows avoiding repeated widening or nine scattered reads will help after accounting for extra copying. Existing v1 files must remain readable.

Prefetch is a third, independent arm. Use the previous token's routed IDs for the target next layer as the first causal heuristic; the first token has no prediction. Compare to a recent-frequency heuristic with its exact update order frozen before timing. Do not inspect the current token's future router result during replay. A clairvoyant replay can only bound possible benefit. Learned cross-layer expert prediction is deferred; stage 3 trains a different same-layer neuron-block predictor.

If the replay and cost model justify implementation, `ExpertPrefetch` may read at most one next-layer batch into budgeted CPU-owned staging while useful current-layer work runs. Demand reads take priority. Prediction does not mutate the live GPU pool. The owner thread joins I/O, verifies complete records, installs only useful records at a safe boundary, then executes the real router's entire selection. Missing/wrong predictions fall back to ordinary demand reads. No worker constructs MLX graphs or evicts live slots. Track useful-prefetch bytes, unused bytes, duplicate reads, cancellation waste, and induced cache misses.

Use explicit ticket states `reserved -> reading -> ready -> consumed/discarded`. A ticket contains engine/model identity, request generation, token position, target layer, keys, byte reservation and retained file-handle ownership. Cancellation invalidates its generation and drains workers before releasing/reusing buffers; stale completions cannot publish. Join current GPU readers before any slot write. Required demand keys are reserved using `ensureCore`'s existing failure/rollback contract; unused predictions cannot evict those keys. Deduplicate in-flight demand/prefetch reads without waiting on a different request's ticket. Prefetch read failure may be discarded until that key is actually demanded; a demanded original-source failure aborts through the existing error path.

Estimate useful bytes and latency from the actual raw/packed read-size distribution, source metadata, missed blocks, unused prefetch, fallbacks and expansion/scatter costs. Use a bounded I/O probe over at most 256 MiB of existing weight regions, with `F_NOCACHE` results reported, to measure size/queue-depth tradeoffs. Never benchmark against the SSD's advertised peak bandwidth. Measure untraced wall time separately from serialized timing scopes; use an overlap-aware removable-time bound to reject candidates incapable of reaching the target. Report zero-net-byte-savings prefetch honestly: its possible benefit is hidden latency, not reduced storage traffic.

**Admission gates:** replay predicts at least 10% fewer demand bytes, or a measured storage layout reduces read-phase time by at least 10%; a prefetch candidate must have a plausible critical-path win after read/expansion/staging costs. These are proposed screening thresholds. After native implementation, require the end-to-end gates in step 7. A failed arm is disabled and recorded; do not keep stacking experiments onto it.

**Verify (new):**

```sh
python3 Tools/flash/gates.py --stage 2 --output .build/flash/runs/stage-2-checks
python3 Tools/flash/replay.py --self-test
.build/release/slotstream-checks --tier t0 --tier t1 --filter flash
python3 Tools/flash/benchmark.py exact --model models/jang-6s --memory-gb 14 --max-context 2048 --output .build/flash/runs/exact-loading
```

Expected: fake-clock/short-read/pin/cancellation tests pass; each enabled exact arm yields identical complete logits, ordered routes, token IDs, and retained state bytes with the same math shapes, including all-wrong predictions. If a storage change changes arithmetic order or kernel dispatch, it fails this exact arm and must be evaluated as a separately named numerical experiment. Failed prefetch must never publish a partial slot.

### Step 3 — Test whether neuron sparsity can save anything at this quantization

Use the shared quality protocol established in stage 1; it is independent of predictor training. Validate the frozen manifest, then capture training and development splits separately:

```sh
python3 Tools/flash/corpus.py validate --manifest Tools/fixtures/flash/suite.json
python3 Tools/flash/capture.py --mode reference --binary .build/flash/runs/diagnostic-reference/bin/slotstream --model models/jang-6s --manifest Tools/fixtures/flash/suite.json --split training --memory-gb 14 --max-context 2048 --output .build/flash/runs/activation-capture
python3 Tools/flash/capture.py --mode reference --binary .build/flash/runs/diagnostic-reference/bin/slotstream --model models/jang-6s --manifest Tools/fixtures/flash/suite.json --split development --memory-gb 14 --max-context 2048 --output .build/flash/runs/development-capture
```

Expected: nonempty bounded captures with complete position accounting and resource receipts. Missing/changing data blocks the affected arm; never silently substitute data or read the final qualification split during tuning.

In `Tools/flash/calibrate.py`, measure actual zeros, activation magnitude, and error after contribution-aware selection. Define block importance initially as `sum_i(abs(hidden_i) * L2Norm(W_down[:, i]))` over its 64 neurons, with down-column norms computed from original quantized values one expert at a time. This bounds contribution magnitude but is not proof of output quality. Compare individual-neuron selection with its block union. Sweep retained blocks 2, 4, 6, 8, and 10; ten is the dense control. Select the smallest development-passing block count, breaking ties toward more blocks. Oracle timing is a measurement of a dense-load diagnostic; its hypothetical I/O bound must be a separately labeled estimate.

Implement a **diagnostic dense-load oracle** in `Diagnostics+Flash.swift` with a bounded hook in `MoELayer`: compute real g/u and hidden values, select/mask blocks, then complete down projection and every subsequent layer using the altered state. Keep the real router for each arm's current hidden state; earlier approximations may change later routes. One fully dense mask must reproduce the frozen reference. `evaluate.py oracle --split development` launches reference/oracle in separate serial monitored processes, streams logits, and runs free-generation development tasks. It must reject `--split qualification` until the run-set is locked. Local hidden-vector error cannot pass the full-model gate. The oracle changes outputs for quality screening and supplies no SSD speed claim.

Train a small causal **same-layer block predictor** only after a useful oracle result. Specify the initial architecture per layer as `sigmoid(tanh(normalize(x) @ A + E[expert] + b) @ C + d)`, with `A:[2560,r]`, `E:[512,r]`, `C:[r,10]`, and ranks 16/32/64. Expert conditioning occurs before the nonlinearity; a simple additive output bias is not enough to express input/expert interactions. Inputs are only current post-attention `x` and the already selected true expert ID, available before reading its g/u weights. Training labels come from oracle block membership; no hidden activation is a runtime input. Predictor weights, normalization and compiled residency share a **128 MiB cap** and the total process target.

Implement `Tools/flash/train_predictor.py`; freeze `Tools/fixtures/flash/training.json` with optimizer (Adam), seed 17, learning rate 1e-3, batch 128, up to 20 epochs, weighted binary cross-entropy with false-negative weight 4, and one-layer-at-a-time training. The `--study` input supplies column norms, fixed block-count policies and split-tagged oracle labels for both training and development. Calibration consumes both capture manifests to produce those labels; normalization remains training-only. Labels are joined by split/document/position/layer/expert IDs, never by row order alone. Missing development labels are an error, not permission to score training rows. These are initial experiment parameters, not optimality claims. Normalize from training statistics only; early stopping selects the best development score with patience 3. Record every run/config; use no final data. Require at least 64 training and 16 development routed examples for an expert before sparse enablement; others get the deterministic dense rule. Training uses an isolated, hash-pinned Python environment matching MLX 0.31.1; record dependencies in `Tools/flash/requirements.lock`. It does not load the full LLM or add Python to inference. Keep training alone under a 1 GiB monitored footprint with streamed examples; stop if the tiny predictor requires the full-model process budget.

Output `.build/flash/runs/predictor/manifest.json`, `weights.safetensors`, and `training-receipt.json`: schema/architecture, tensor shapes/dtypes, model/tokenizer/quant identity, input normalization, supported expert mask, training/development corpus hashes, thresholds/block counts, confidence/dense rules, seed and dependency hashes. Hash and validate every artifact. Use held-out development data to select one rank/config with the smallest measured end-to-end cost that passes quality, then freeze it. The final manifest is immutable; tuning produces a new artifact ID.

**Actual predictor gate:** `evaluate.py predictor` replaces oracle masks with the trained predictor's outputs in the same dense-load full-model runner. It must pass development KL/perplexity/free-generation gates independently; oracle success cannot certify a learned predictor. Report useful-block recall, weighted false negatives, dense fallback rate, coverage, actual per-layer prediction/synchronization cost, and expected served bytes including fallback experts. Sparse masks derive only from current activations and frozen policy, never cache hits, available RAM or storage timing.

**Gate:** continue to step 4 only if the **trained predictor**, including uncovered-expert fallback, saves at least 20% of estimated expert source bytes after block union/metadata, passes the development quality limits, and predicts at least a 10% request-latency improvement including measured predictor/staging/compute cost. Oracle-only savings cannot open this gate. Otherwise record rejection and continue independent exact/M5 work. Do not ReLU-convert, prune, or requantize Qwen to force a result.

**Verify (new):**

```sh
python3 Tools/flash/gates.py --stage 3 --output .build/flash/runs/stage-3-checks
python3 Tools/flash/calibrate.py --self-test
python3 Tools/flash/calibrate.py --capture .build/flash/runs/activation-capture --development .build/flash/runs/development-capture --output .build/flash/runs/sparsity-study
python3 Tools/flash/evaluate.py oracle --study .build/flash/runs/sparsity-study --model models/jang-6s --reference-binary .build/flash/runs/diagnostic-reference/bin/slotstream --manifest Tools/fixtures/flash/suite.json --split development --memory-gb 14 --max-context 2048 --output .build/flash/runs/oracle-report
python3 Tools/flash/train_predictor.py --config Tools/fixtures/flash/training.json --study .build/flash/runs/sparsity-study --training .build/flash/runs/activation-capture --development .build/flash/runs/development-capture --output .build/flash/runs/predictor
python3 Tools/flash/evaluate.py predictor --artifact .build/flash/runs/predictor --model models/jang-6s --reference-binary .build/flash/runs/diagnostic-reference/bin/slotstream --manifest Tools/fixtures/flash/suite.json --split development --memory-gb 14 --max-context 2048 --output .build/flash/runs/predictor-report
```

Expected: exact dense-mask identity; blocked train/eval document overlap; finite KL/error metrics with explicit sample counts and covered layers/experts; a machine-readable `proceed` or `reject` decision from the fixed gates. An absent capture or missing cohort must fail, not emit an empty successful report.

### Step 4 — Build a small bundle prototype, then a bounded sparse executor

Implement only after step 3 passes. Start with one real expert from each of a few covered layers, not a full checkpoint repack.

`NeuronBundleLayout` stores `(layer, expert, block)` records. For block j, include gate rows `[64j:64j+64,:]`, matching up rows, and down columns `[:,64j:64j+64]`, with original affine codes, scales, biases, bit widths, and orientation metadata. For down, each row's single scale/bias pair belongs to that original 64-column group. Never transpose then requantize. Mixed 4/6-bit projections retain their individual source widths; apply any exact code widening only in the bounded runtime representation.

Use a versioned manifest with checkpoint/predictor-independent identity, per-record offsets, lengths, alignment, and hashes. Reject overflow, overlap, truncation, wrong shapes, unsupported versions and changed files. Write a new sidecar through a temporary file and atomic finalization; original safetensors stay read-only. Follow `PackedExpertLayout`'s verified read/fallback and joined-worker patterns. Include a dry-run byte estimate before a full repack and keep builder memory bounded to a few expert records. A disk-space failure must not damage a completed sidecar.

`NeuronBlockCache` keys include model, layer, expert, block, and representation. Preallocate a hard budget, retain recent-token block membership, admit only complete payloads, and pin all readers until evaluation joins. Dense expert cache, sparse block cache, and prefetch staging share a single allowance; do not allocate a second full cache. If a current selected set cannot fit, execute a bounded fallback with resources reserved in advance rather than briefly holding both representations in full.

For the first `SparseExpert` path, gather selected g/u rows, compute SwiGLU, and multiply the corresponding down columns. Preserve router rank. All-block execution is a storage identity control; a different accumulation tiling may still change floating-point results, so test both byte reconstruction and numerical output independently. Read/amplification telemetry includes padding/metadata. Freeze dense/sparse crossover from development timings; high occupancy and prefill use the existing grouped dense sweep initially. Do not decide masks from whether blocks happen to be cached. On sidecar failure, original-source fallback must reconstruct the **same selected blocks/mask**; loading more bytes cannot silently restore skipped neurons. If reserved memory cannot perform that fallback, abort and roll back the request. Cache pressure must not silently redefine model computation or reuse state under an obsolete identity.

**Verify (new):**

```sh
python3 Tools/flash/gates.py --stage 4 --output .build/flash/runs/stage-4-checks
.build/release/slotstream-checks --tier t0 --tier t1 --filter flash
.build/release/slotstream flash-pack --model models/jang-6s --sample-only --output .build/flash/runs/bundle-sample
.build/release/slotstream flash-check --model models/jang-6s --bundles .build/flash/runs/bundle-sample
python3 Tools/flash/benchmark.py sparse --model models/jang-6s --memory-gb 14 --max-context 2048 --output .build/flash/runs/sparse-prototype
```

Expected: original codes/metadata round-trip byte-for-byte; original-weight dequantization agrees with native MLX; all selected block combinations in synthetic fixtures use correct down columns/scales; read faults never leak partial records; dense controls and approximate quality gates are reported separately. Full repack is allowed only after this prototype passes and the dry-run disk/memory estimate fits.

### Step 5 — Use the M5 GPU accelerators where they improve the workload

First retain a successful existing grouped NAX path from step 1. Do not add a custom kernel just to claim accelerator support. Target measured compute cost: grouped prefill, sufficiently batched selected blocks, or the predictor's matrix operations. One-row SSD-bound decode may be faster with the existing QMV path; duplicating input rows solely to trigger GEMM must be charged as extra work.

If a custom path is justified, prototype `M5ExpertKernel` through the repository's `MLXFast.metalKernel` pattern (see `PartialRotation.swift`). Test MPP header/API availability with a tiny bounded kernel before implementing the real shapes. Use the macOS 26-compatible dequantization-to-tile path when native tensor formats cannot represent JANG. Preserve affine bias, 6-bit packing, dtype/rounding boundaries, padding masks, accumulation policy and output rank. Never dequantize an entire layer or pool into a second persistent FP16 model.

If the JIT wrapper cannot express the required TensorOps integration, stop that custom-kernel arm and write an explicit bridge proposal; do not silently upgrade MLX, alter generated dependency files, or require macOS 27. Current MLX NAX remains available independently of this proposed custom path.

Use `OptimizationPlatform` only for a new, separate qualification record containing actual chip, OS, SDK/compiler, MLX source, metallib and kernel hashes, supported shapes and quant/dtype combinations. Runtime capability and measured qualification are separate checks. Unsupported or unqualified combinations use the established MLX path and report the effective policy. Keep all new Flash defaults off after qualification; activate an accepted numerical kernel only through the explicit configuration contract.

**Verify (new):**

```sh
python3 Tools/flash/gates.py --stage 5 --output .build/flash/runs/stage-5-checks
.build/release/slotstream-checks --tier t0 --tier t1 --filter m5
.build/release/slotstream m5-check --model models/jang-6s --output .build/flash/runs/m5-candidate
python3 Tools/flash/benchmark.py m5 --model models/jang-6s --memory-gb 14 --max-context 2048 --output .build/flash/runs/m5-pairs
```

Expected: finite outputs and stated numerical tolerance against the same quantized reference, explicit unsupported-shape fallback, actual eligible-kernel dispatch evidence, and paired wall-time improvement with compile/capture overhead excluded from steady-state but reported separately. Hardware-usage claims remain unverified if dispatch evidence is unavailable.

### Step 6 — Evaluate the separate Neural Engine only for the predictor

This is optional after a trained predictor proves useful. `CoreMLPredictor` loads only a small fixed-shape predictor and compares CPU, MLX GPU, and Core ML execution on identical inputs. Use public Core ML APIs and compare `.cpuOnly` with [`.cpuAndNeuralEngine`](https://developer.apple.com/documentation/coreml/mlcomputeunits/cpuandneuralengine). That selection permits CPU fallback; it does not force ANE execution. Inspect [MLComputePlan](https://developer.apple.com/documentation/coreml/mlcomputeplan-1w21n) where available and collect runtime profiling evidence before declaring ANE use.

Export all predictor layers/heads through a versioned artifact with a finite shape set; no runtime compilation per token. `export_coreml.py --artifact <predictor-dir> --output <fresh-coreml-dir>` writes the model package and conversion receipt, preserves normalization/mask policy, and validates outputs on stored development vectors. Include model load/compile cost, resident memory, GPU synchronization, input/output copies, prediction time, and overlap with GPU work. Unified physical memory does not remove framework synchronization or ownership costs. Run conversion/training in an isolated tool environment; do not add Python to Slotstream's inference dependencies. Create this exporter only if stage 6 is selected.

Adopt ANE only if at least 10% improvement in total predictor critical-path latency survives end-to-end measurement, the memory cap holds, and masks/quality pass. If it uses CPU or is slower, keep the faster backend and record the result; do not move experts or attention to ANE in this plan.

**Verify (new):**

```sh
python3 Tools/flash/gates.py --stage 6 --output .build/flash/runs/stage-6-checks
python3 Tools/flash/export_coreml.py --self-test
python3 Tools/flash/export_coreml.py --artifact .build/flash/runs/predictor --output .build/flash/runs/predictor-coreml
.build/release/slotstream flash-predictor-check --artifact .build/flash/runs/predictor --coreml-artifact .build/flash/runs/predictor-coreml --backends cpu,mlx,coreml --output .build/flash/runs/predictor-backends
```

Expected: identical artifact identity and input cohort, per-backend output/error checks, footprint and complete timing, explicit observed device or unverified status, and a deterministic keep/reject result. Missing Core ML conversion tooling blocks only this optional stage.

### Step 7 — Qualify the selected combination and document what actually works

Implement aggregate qualification using the explicit run-set and receipt contract above. Create a run-set only for selected implemented tracks; skip unselected optional tools, never their required evidence. Keep all new modes opt-in even after acceptance. Approximation and custom numerical paths must appear in metadata and never masquerade as an unchanged arithmetic path.

First evaluate ablations on development fixtures: baseline, storage, window, heuristic prefetch, predictor, bundles/sparse, custom M5, optional ANE, and selected combination. Then freeze one candidate per intended claim, its mode/artifact hashes and the final run-set. The final suite is run once for that decision. A failure can return to development, but reusing its exposed final set is labeled regression testing, not a fresh holdout; new qualification requires a new disjoint manifest or a report explicitly limited to the exposed fixed suite. Never silently tune ranks/thresholds on final results.

For performance, use 10 complete randomized AB/BA pairs per selected workload as the predeclared default, seed 17; do not stop early when significance first appears. Fresh processes mean cold Slotstream caches, and repeated requests mean a warm engine with a defined exact prefix. Pair at the workload/run level, not by treating correlated tokens as independent samples. An infrastructure-failed pair is preserved and makes that cohort inconclusive; a fresh predeclared cohort is a new run-set, not a selective replacement.

Define timing boundaries in the run-set: startup TTFT is process launch to first token; warm request TTFT is accepted request to first emitted token; prefill is input processing only; decode latency excludes first-token/startup time. Record load/JIT separately. Use both fixed-work teacher-forced replay of 128 reference continuation IDs (for equal-work kernel/I/O comparisons) and natural generation with EOS respected (for user-visible timing). Declare decode eligibility from baseline before candidate timing; fewer than 32 naturally emitted tokens cannot support a steady decode claim. Fixed replay alone cannot establish faster real replies. Report actual token counts, p50/p95 inter-token latencies and bytes per produced/replayed token with correct denominators.

For M5 numerical changes, the final corpus must actually exercise every claimed prefill/quant/dtype path, including 256/1024-token prefill and one-token decode controls; shape/kernel invocation counters and live dispatch receipts prove coverage. A one-token-only numerical test cannot qualify a prefill-only kernel. Capture defined state immediately after prefill and compare downstream continuations; timing runs keep intrusive observers off.

Compute full-vocabulary log-softmax in FP32, then accumulate per-position KL and NLL in FP64: `KL_i=sum_v exp(logp_i[v])*(logp_i[v]-logq_i[v])`; `PPL_ratio=exp(mean(NLL_candidate)-mean(NLL_reference))`. Stream by chunk; no probability truncation or averaging per-document perplexities. Aggregate means by scored token count, p99 by the predeclared nearest-rank rule, and report per-category distributions plus a document-cluster bootstrap interval. Reject nonfinite rows, missing positions, vocabulary mismatch and a materially negative KL; tolerate only a fixed tested roundoff band, not arbitrary clamping. Test known scalar distributions and the next-token offset independently of the new model adapter.

**Proposed acceptance thresholds — conservative experiment targets, not measured results:**

- Exact loading arms: full logits, ordered routes, output IDs and retained state bytes identical at the same shapes. No new quality loss. Identical generated text alone is insufficient.
- New math kernels with all neurons: quantify differences from repeated/rechunked reference runs; require full-vocabulary mean `KL(p_baseline || p_candidate) <= 1e-4` nats, p99 <= 1e-3, top-1 agreement >= 99.9%, no nonfinite values, and no retained-state invariant failures. Apply these gates both pooled and per category with the required coverage. A stricter existing repo gate still applies. Do not enlarge limits on failure.
- Approximate neuron mode: teacher-force the same held-out IDs through **unmodified JANG_6S** and candidate. Require mean KL <= 1e-3 nats, p99 <= 1e-2, top-1 agreement >= 99%, perplexity ratio <= 1.01, both pooled and per category, plus the fixed-suite scored-task gate in the shared quality protocol. Report intervals/coverage; insufficient evaluation remains unqualified. These measure incremental approximation to JANG_6S, **not JANG quantization loss against BF16**. The combined sparse+M5(+ANE) path must pass directly against the original reference; individual error budgets are not added together.
- Approximate execution must also pass free-running structured output/code/math checks and request cancellation/prefix-continuation tests. Teacher-forced quality does not cover autoregressive divergence or persistent state reuse. Prefix entries must be tied to numerical mode and predictor/kernel identity so incompatible states cannot be reused.
- Performance: at least 10% median improvement in the **predeclared** target metric, with a paired bootstrap 95% interval excluding no improvement; other primary metric must not regress > 5%. Use 10,000 resamples of complete workload pairs, seed 17, and report the distribution. Ten pairs do not guarantee power; an inconclusive interval stays inconclusive. Choose no new winning metric after seeing results. Report end-to-end, fixed-work and load/compile measurements separately.
- Memory: external physical peak, allocation ledger, and MLX peak stay within the target. Never trade hidden paging or larger unpriced caches for the speed claim. Concurrent system swap alone does not establish a model violation, but an uncontrolled run is unsuitable for performance qualification.

Extend the report and fixtures to cover original Pipe4 and JANG_4M small-data regressions. Full-model qualification for a quant requires its complete checkpoint and its own paired results. Do not download additional huge models automatically under this plan. The available JANG_6S result must not be advertised as JANG_4M or all-device performance.

**Freeze and run final evidence (new interfaces; select only implemented arms):**

`selection.json` lists selected stage/arm IDs and candidate config paths; construct it from accepted development dispositions. The run-set tool checks prerequisites rather than trusting those labels.

```sh
python3 Tools/flash/gates.py --stage 7 --output .build/flash/runs/prequalification-checks
python3 Tools/flash/runset.py create --selection .build/flash/runs/selection.json --reference .build/flash/runs/diagnostic-reference --suite Tools/fixtures/flash/suite.json --pairs 10 --seed 17 --output .build/flash/runs/final/runset.json
python3 Tools/flash/benchmark.py qualify --runset .build/flash/runs/final/runset.json
python3 Tools/flash/evaluate.py quality --runset .build/flash/runs/final/runset.json
```

The create command makes a fresh run-set directory, verifies current candidate build identities, archives each candidate executable/metallib alongside its receipt, and refuses overwrite. All subsequent runs use those immutable archives, not a moving .build/release path. Subsequent commands write only their own fresh receipt directories under it. `quality` runs only needed modes: exact identity for exact arms; full numerical/held-out gates for numerical or approximate arms; no predictor/coreml imports for an exact-only run-set. Reference and candidate model processes remain serial. Expected: exactly the requested workload/case coverage, complete receipts, and no use of final inputs in training/development artifacts.

**Final checks (new and existing):**

```sh
python3 Tools/flash/gates.py --stage 7 --runset .build/flash/runs/final/runset.json --output .build/flash/runs/final/checks
python3 Tools/flash/evaluate.py qualify --runset .build/flash/runs/final/runset.json --output .build/flash/runs/final/qualification.json
PATH="/Users/imdad/Documents/Codex/2026-09-12/wha/work/bin:$PATH" Tools/static_gates.sh
python3 Tools/context_proxy.py --out .build/flash/context-proxy-final
make docs
PATH="/Users/imdad/Documents/Codex/2026-09-12/wha/work/bin:$PATH" Tools/brain_gates.sh
git diff --check
git status --short
```

Expected: a fresh validated native build; every required check present with zero required skips; report rejects missing/mixed evidence and returns `qualified=true` only for the nonempty passing combination; canonical brain validates; generated docs match; only allowed files changed. If no candidate qualifies, use `evaluate.py summary --selection ... --output ...` to record the research outcome with `qualified=false` instead of manufacturing an empty successful run-set. Run `Tools/verify.sh` if the original full checkpoint is available; otherwise mark it unavailable and preserve original defaults. Document explicit activation, supported shapes/device identity, quality tradeoffs and rejected experiments. Do not put unsupported speed numbers in the README.

## Test plan

Follow `CheckBuilder` in `Diagnostics+JANG.swift`; register new pure bookkeeping checks in T0 and small MLX numerical checks in T1. Model-backed commands are separate and serial. Fixtures must use independent scalar/native-MLX references, including original JANG row samples, rather than comparing two calls to the new packer.

Required cases:

- **Window/prefetch:** token epochs across 48 layers, duplicate IDs, repeated and all-wrong predictions, cache capacity smaller than window union, every required slot pinned, request cancellation, prefix reuse, layer wraparound, delayed read completion after cancellation, demand priority, and deterministic output rank.
- **Read/packing:** short reads, EINTR, EOF, invalid/overflowing offsets, partial manifest, checksum mismatch, file changed after verification, mixed 4/6-bit codes crossing byte/word boundaries, FP16 metadata round-trip, and two concurrent engines with distinct checkpoint identities.
- **Blocks:** first/last 64-neuron blocks, noncontiguous selections, all/none masks, ten-block reconstruction, each down column's original scale group, unsupported alignment, padding masks, dense fallback, and under-budget cache replacement.
- **Math:** negative SwiGLU values, tiny values/subnormals, nonfinite rejection, FP32-metadata promotion, effective TF32 modes as separately labeled reference cohorts, router ties, shape-dependent dispatch, and original rank reduction. Never alter the default TF32 setting mid-process: this MLX version caches it.
- **Predictor/evaluation:** causality checks, train/eval leakage rejection, missing/unknown expert fallback, incorrect model revision, schema mismatch, dense-mask identity, known analytic KL, KL direction, full-vocabulary coverage, empty/missing corpus rejection, and changed numerical mode invalidating prefix state.
- **Resources:** budget arithmetic includes both live representations, predictor loading failure, bounded capture overflow, joined scratch release, sampling failure, timeout, no output published after cancellation, and only the owned process tree stopped by the harness.
- **Backend:** M5 present/absent/unverified, older OS, unavailable TensorOps feature, JIT failure, missing capture evidence, Core ML CPU fallback, and restored default path. A fake hardware report must not pass qualification.

## Done criteria

- [ ] Stage 0 complete-checkpoint baseline and provenance exist.
- [ ] Stages 1–7 have explicit dispositions; selected tracks have evidence, optional/dependent `not_selected` tracks have reasons, and no required selected track is blocked.
- [ ] Every selected exact optimization passes byte/state/routing parity and memory gates.
- [ ] Any selected new kernel or approximation passes its separate quality gates, with real held-out sample counts.
- [ ] M5 usage is proven for the stated build/path, or remains labeled unverified/already-used without a new acceleration claim.
- [ ] Optional ANE result distinguishes requested compute units from observed execution.
- [ ] End-to-end paired performance supports every claimed gain; no paper speedup is inherited.
- [ ] Original default quant, router, MTP/vision boundaries and numerical fallback remain intact.
- [ ] Strict stage gates verify fresh binary identity, required check names/counts, zero required skips, and complete run-set evidence; unavailable full-model gates are recorded as unavailable.
- [ ] Canonical evidence, generated docs, and `plans/README.md` agree about implemented, rejected and unqualified work; all new defaults remain off and activation is documented.

Completion of the research plan may yield no accepted sparse or ANE path. That is a completed investigation, not fulfillment of a promised performance gain. State the distinction explicitly.

## STOP conditions

Stop the affected arm and report; continue independent measured arms where possible:

- Complete JANG_6S weights fail verification, baseline generation fails, or a model reference is known to be numerically wrong.
- Required geometry, metadata dtype, quant groups, package/metallib pin, or ownership boundaries have drifted from this plan.
- An exact arm needs to change the router, discard demanded experts, change math shapes/order, or lower metadata precision.
- The neuron oracle fails quality, 64-neuron block union removes the predicted savings, or predictor/copy overhead erases the win.
- Required TensorOps features exist only in an unavailable OS/SDK, or supporting them requires an out-of-scope bridge/dependency change.
- A claimed GPU/ANE path has no execution evidence. Keep it unverified; never fabricate utilization or use TOPS as a throughput measurement.
- Memory accounting cannot cover overlapping caches/workspaces, footprint exceeds target, ownership of a process is uncertain, or safe headroom is unavailable.
- A verification step fails twice after a reasonable fix, or an implementation requires an out-of-scope edit. Do not improvise a checkpoint conversion or baseline redefinition.

## Maintenance notes

Cache windows, predictor validity, source-native packing, and compute shapes interact. Requalify the combination after any change to quantization, model revision, TensorOps/MLX, OS, cache partition, group alignment or dtype policy. Preserve the previous format readers and ordinary demand path. Reviewers should focus on asynchronous slot lifetime, down-projection group mapping, hidden dtype promotion, predictor causality, and prefix-state identity.

Deferred work includes a finer-than-64-neuron compressed layout, model retraining for sparsity, full ANE model execution, private ANE runtimes, speculative MTP integration, and new device/default rollout. Each needs a separate evidence-based plan.
