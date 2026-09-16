# Plan011: Accelerate exact4-to6-bit expert expansion

## Current disposition — September 16, 2026

**IN PROGRESS — NEON optimization implemented; short full-model comparison completed.** A [14 GB paired benchmark](../db/records/measurements/jang-neon-full-model-2026-09-16.md) now records exploratory throughput results against the previous packed backend. This does not qualify the full suite or update the earlier 24 GB measurements. The existing `packed4-to6` policy now uses a CPU NEON bulk path with exact tails. The release build, focused vector/byte checks and bounded checkpoint-source comparisons pass. A short component comparison measured the conversion improvement; no long qualification campaign was repeated. [SIMD evidence](../db/records/measurements/jang-simd-widening-2026-09-15.md). The earlier packed backend's qualification attempts remain [thermally inconclusive](../db/records/measurements/jang-widening-qualification-attempts-2026-09-15.md); no overall qualification or default activation is claimed.

[Evidence/current findings](009-oracle-execution.md) · [Canonical status](../db/records/plan/jang-flash-qualification-status.md). The original execution brief and dated review notes below are retained as history; current work is governed by [the plan index](README.md). Do not restart completed or unselected steps from a historical instruction.

## Status and purpose

September 15 measurement update: the user requested a performance checkpoint after the neuron-oracle investigation. A separate cooled 24 GB benchmark completed three paired rounds per saved workload, with exact output/work equality. Current packed widening achieved 6.983 / 7.409 / 7.266 tok/s versus scalar 3.199 / 3.331 / 3.272, with all arms timing-eligible and under 24 GB. See `plans/009-oracle-execution.md` and `.build/flash/runs/engine-benchmark-24gb-cooled-20260915`. This is not the predeclared ten-pair 14 GB qualification below and does not enable defaults or serving activation.

P1; effort M; risk MED. IN PROGRESS against `bc185dfe4d5ab8c04f107bef90be47d7952ffc69`; reviewed010 is complete and has released main.swift ownership. This exact-loading experiment takes priority over the conditional neuron oracle. Source/API recon and the010 commit were checked before dispatch. Work in the existing isolated checkout/branch, reviewer owns plans, source commits after review, no merge/push. All new runtime behavior is explicit opt-in; scalar expansion remains the default.

The source opportunity is confirmed. `AffineCodes.widen` in `Sources/Slotstream/AffineRow.swift` clears the destination then visits every code, even for the frequent fixed4→6 conversion. `ExpertStore.readRows` calls it for original4-bit projections entering the6-bit JANG_6S pool. Gate widens in47 layers, up in40, down in0;40 layers need two conversions per missed expert, seven need one, layer22 needs none. Each conversion visits1638400 codes.

Read-only recon `.build/flash/runs/widen-profile/report.json` (SHA256072af2f17fed1fa69d009fd7355ffa1a960270ac95a5e33fd622e28a2b0b03d8) measured the unchanged scalar routine at2.730ms single-worker median;8-/16-worker groups amortized0.518/0.306ms per call. Benchmarks stayed below36MB physical memory. The compiler tree was not memory-qualified. The earlier three-prompt trace contained120444 calls/197335449600 code visits. Decode I/O timing includes read, widening, metadata conversion and staging; multiplying microbenchmark costs by call counts is not removable wall time or a speedup prediction.

## Scope and API

Own AffineRow.swift, narrowly scoped ExpertStore.swift/Model.swift/Engine.swift propagation, Run command in main.swift, bounded Flash.swift/capture.py support for explicit exact-widening evidence, new Diagnostics+Widening.swift/T0 registration, new widening-check CLI and host `Tools/flash/widen_study.py`/tests/schema. Coordinate gates.py changes with the quality executor; do not touch its corpus/metrics/env implementation. No model files, quantization parameters, cache policy, math kernels, package pins, serving defaults, masks or prefetch.

Add public `AffineWideningPolicy` with raw values `scalar` and `packed4-to6`, Codable/Sendable, default.scalar. Thread it immutably from both Engine initializers through all Qwen4ExpModel initializer overloads to ExpertStore initialization and the widening call. Expose the effective policy read-only from Engine/pool/store. Validate an explicit fast request as JANG_6S-only before resident/pool allocation. Reject combination with existing packed-layout activation, which bypasses this source conversion. Do not use a mutable global or environment-only knob.

Add `--expert-widening scalar|packed4-to6` only to run and relevant diagnostic commands; do not put it in ModelOptions shared by serve. Serve must reject the unsupported flag before allocation. Report the effective engine policy in new run stats. Existing archived scalar stats without the field are recognized only through their verified known reference identity, not assumed for arbitrary binaries.

## Exact packed implementation

Implementation follow-up: `Sources/CAffine/affine.c` now uses NEON structure
loads/stores to expand sixteen two-byte groups per iteration. Swift retains
the original validation and overlap fallback. The C backend needs no scratch
storage, handles unaligned buffers and exact tails, and provides a portable
fallback. `Package.swift` links the small internal CAffine target. Existing
CLI/API policy names and generic scalar defaults are unchanged. Earlier full-model
TPS results were measured with the preceding Swift packed loop, not this backend.

Keep the old validation/error behavior and generic scalar routine for every other supported conversion. For a non-overlapping4→6 source/target pair, process four4-bit codes (two input bytes) into three complete output bytes. For source bytesa,b:

```
out0 = (a & 0x0f) | ((a & 0x30) << 2)
out1 = ((a & 0xc0) >> 6) | ((b & 0x0f) << 4)
out2 = (b & 0xf0) >> 2
```

The existing geometry check implies code count divisible by4 for this conversion. Write every destination byte, including unused zero bits, without a preliminary full memset. Use safe unaligned/byte access, no out-of-bounds word store at the end. Preserve zero-count behavior. Either retain generic behavior for overlapping buffers or explicitly demonstrate an already-existing non-overlap contract; do not silently change an unsafe-pointer API's observed behavior. No new scratch allocation or metadata conversion. A more elaborate vector implementation is unnecessary until this simple specialization is measured.

Tests: exhaust all65536 two-byte inputs against an independent bit-by-bit nibble-to6-bitstream oracle; compare complete packed bytes, including unused bits and prefilled0xff destinations. Exercise zero and invalid counts, integer overflow guards, unaligned slices/sentinels, first/last groups, multiple rows, realistic1638400-code projections, generic3/4/6/8 fallback conversions, and immutable per-engine policy/default propagation. Verify actual JANG source samples against scalar expanded bytes. Never substitute a requantization/dequantization approximation.

## Component measurement

Add `slotstream widening-check --synthetic --output <fresh-dir>` and mutually exclusive `--model <pinned6S>` mode. Synthetic mode runs exact tests and paired scalar/packed benchmarks with reusable bounded buffers, deterministic inputs,1/8/16 workers and consumed output checksums that prevent dead-store elimination. Both policies use identical benchmark scaffolding and balanced pair order. Record complete timings/source/compiler/binary identities and method limits; avoid extrapolating these timings to inference.

Model component mode creates only CheckpointIndex/ExpertStore, never Engine/resident trunk/pool. Compare original-reader scalar/packed outputs and read-phase times for layers0,5,22 and experts0...9: representative two-/one-/zero-widening batches. Exercise both readBatchChecked and contiguous readRunsChecked. Bound unique original source regions below256MiB and actual component physical memory<=512MB. Verify source identities before/after; record actual F_NOCACHE/F_RDAHEAD setup and observed disk-read counters where available, not advertised SSD bandwidth. Alternate policies, verify every returned tensor's exact bytes outside the timed phase, and report allocation/read/conversion/staging scopes honestly. Stop on short reads, drift, partial outputs or memory failure.

Admission: exact tests pass, packed CPU conversion is materially faster, and complete measured reader-phase evidence supports a plausible end-to-end benefit. A noisy control or unknown physical-read coverage is labeled inconclusive. No runtime performance claim follows solely from a microbenchmark. Do not broaden to metadata conversion/layout changes in this plan.

## Native parity and performance

September 15 execution correction: the first current-source qualification attempt
stopped after its nineteenth Python arm because an endpoint reported `fair`
thermal conditions. Its incomplete cohort remains at
`.build/flash/runs/widen-qualify-20260915` and supplies no qualification result.
Subsequent ordinary arms require a continuous thirty-second window of observed
nominal conditions with low-power mode off before the original two-second VM
settling interval. This reuses the cooled development benchmark's policy and
records the observations in `settling.json`. The wait is outside model timing;
all existing endpoint, paging, exact-work, memory and performance gates remain
unchanged. Restart the entire ten-pair-per-prompt cohort, with no replacement
or reuse of individual pairs from the stopped attempt.

The cooled successor at `.build/flash/runs/widen-qualify-cooled-20260915`
also stopped: its second scalar arithmetic arm began nominal and ended fair.
It preserves 44 completed launches and 22 complete pairs, with one pair
timing-ineligible. Sky-blue and Python each completed ten eligible pairs; those
subsets do not satisfy the required thirty-pair cohort. Formal qualification
remains pending a suitable thermal window. Do not automatically replace the
failed pair, pool either attempt, or describe the cooldown as a guarantee.

Extend capture with explicit policy handling while preserving existing reference schemas. Scalar capture retains its existing format and behavior. Packed capture uses a separately named `slotstream-flash-widening-output-v1` containing the effective policy and the existing computational payload; host validation admits only this specific extra configuration/version. Preserve original observer semantics and128MiB reservation, normal numerical controls, cache capacity, input IDs and shapes. Do not weaken the old reference validator or call an unmodified archive a packed implementation.

`widen_study.py parity --model <model> --reference <immutable007archive> --candidate <binary> --corpus <accepted007natural-corpus> --output <fresh>` archives the candidate, runs original scalar versus explicit packed captures serially, and requires exact full logits, ordered routes, retained state identity, input/target/continuation IDs. Also compare ordinary CLI output IDs. All model runs keep14GB,>=17GB headroom,2second settling,2048context,MTP/vision off,greedy seed7 and terminal-before-reap evidence. No reference archive replacement and no model file changes.

Then `widen_study.py screen` runs one paired untraced comparison for each already frozen cache-study prompt (sky-blue,python-deduplicate,multiply-37-by-42), preserving its actual prompt/output limits and exact output-ID equality. This is explicitly exploratory. Freeze source/binary/harness/options during each cohort; no concurrent model/GPU/I/O benchmark. If the candidate is promising, `widen_study.py qualify --pairs 10` runs a new predeclared10-pair cohort per prompt with balanced pair order, same memory and exact work. Primary metric is decode time for identical output IDs/counts; report corresponding tokens/s, prefill/first-token/request/load times separately. Required admission is>=10% median paired decode improvement, paired bootstrap95% interval excluding no improvement (10000 resamples,seed17), and no>5% first-token regression per workload. Missing/failed pairs invalidate the cohort; no selective replacements or selecting a new winning metric afterward.

The final claim is limited to this exact small prompt set/device/checkpoint/budget. This is not parent001's larger-context final qualification, and all modes stay off by default. If improvement is absent or inconclusive, retain the default scalar policy and report that result. An enabled experimental command may exist without a qualified default rollout.

## Verification and done

Implement/run pure host tests, exact named widening native checks, two-job release build and existing native/static gates. Archive source/build bytes before later rebuilds. Run synthetic/model component controls, then full-model parity and the declared performance cohorts only if preceding gates pass. Keep early/incomplete evidence separate. Update only reviewer-owned plans through the reviewer; public README performance claims require canonical db evidence in a later reporting step.

Done: reviewed exact specialization with explicit immutable policy propagation, tested generic fallback, complete parity and a measured exploratory/qualified/rejected/inconclusive result at its stated scope. No merge/push/default activation. Return STATUS/STEPS/STOPPED BECAUSE/FILES CHANGED/NOTES with exact command/evidence/commit facts.

Execution coordination: while008 completes its accepted corpus capture, the widening executor may edit only its owned Swift source. Wait for the model/harness handoff before builds, GPU/component jobs or edits under Tools/flash (including capture.py); this keeps the ongoing capture harness fixed. The quality executor owns its Python corpus/metric/env files, and root owns plans. Do not run models until component admission and root review.
