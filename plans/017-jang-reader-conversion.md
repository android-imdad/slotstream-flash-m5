# Plan017: Optimize JANG metadata conversion and lane-local scratch

## Status
- COMPLETE — both runtime candidates rejected; no runtime changes retained.
- Priority P1; performance; effort M; risk medium.
- Planned at `96a4fae`, September 16, 2026.
- Depends on completed NEON implementation and short full-model baseline.
- User authorized commit and execution of the recommended metadata/scratch follow-up. The reviewer maintains plan status and integrates approved commits; no push or release.

## Current execution

The metadata SIMD runtime candidate was rejected: the actual preserved release `readRows` assembly already contains FCVTL/FCVTL2 loops. A standalone exact-bit comparison and compiler/source receipts are retained; no metadata runtime path was added. Scratch was implemented, archived and rejected after four paired rounds in each reader case: five of six medians worsened. Candidate exact/fault checks passed; the reviewer verified the report and reran focused checks. The original runtime was restored and its full bounded check catalogue passed. No full-model follow-up was launched because the component candidate did not pass admission. [Canonical results](../db/records/measurements/jang-reader-followup-2026-09-16.md).

## Scope and current state
Repository `/Users/imdad/Documents/projects/slotstream-flash-m5`, SwiftPM macOS CLI with MLX and CAffine C target. Work in an isolated `codex/` worktree created from the planned commit. You are not alone in this repository: preserve other tasks and never revert another person's edits. The reviewer owns plans and canonical evidence/docs.

Own only `Sources/CAffine/affine.c`, `Sources/CAffine/include/affine.h`, `Sources/Slotstream/AffineRow.swift`, `Sources/Slotstream/ExpertStore.swift`, `Sources/SlotstreamDiagnostics/Diagnostics+Widening.swift`, and narrowly named new standalone benchmark/check drivers under `Tools/flash/`. Use ignored `.build/flash/` for raw artifacts. No changes to model files, routing, kernels, quantized values, precision, cache policy, planner limits, queue depths, sweep groups, serving, defaults, or generated/public docs.

`ExpertStore.readRows` lines177-190 reads a zero-filled `Data`, assembles UInt16 from bytes and calls `Float(Float16(bitPattern: bits))` per scale/bias. Lines195-203 likewise allocate `Data` for narrowed-weight source bytes. `readBatchChecked` lines330-349 dispatches multiple piece jobs per joined lane; `readRunsChecked` lines390-412 uses joined lanes with complete failure draining. `stagingArrays` transfers buffer ownership to MLX; do not recycle these GPU-owned buffers. Error handling uses `JoinedReadFailure` and publishes only complete successful destination arrays. Match the existing checked native bridge in `AffineRow.swift` and exact-widening CheckBuilder tests.

## Step 1: Price and implement metadata conversion
Inspect release assembly and measure the actual existing Swift conversion against a bounded proposed C/NEON converter on representative 25,600-value metadata pieces. Use alternating paired rounds, consume outputs, compare full bytes outside timings, avoid dead-code elimination, and preserve all raw samples/source/compiler identities in ignored output. Do not report source-level scalar code as proof of absent compiler vectorization.

If promising, add a checked metadata conversion helper with an ARM NEON path and exact portable fallback. It converts FP16 bit patterns into the same FP32 bits as existing Swift, without lowering metadata precision. Apply the optimization only to explicit `.packed4to6` policy (use the actual enum spelling in source); preserve existing generic scalar behavior. Caller must reject invalid input/output sizes and handle unaligned input, output alignment, tails and overlapping buffers safely. Prefer clear checked preconditions and bounded fallback rather than relying on undefined aliasing.

Expand existing focused exact-widening checks (no new catalogue registration needed) for all 65,536 FP16 patterns compared by FP32 bit pattern to original Swift conversion, signed zeros, subnormals/infinities/NaNs, varied vector/tail lengths, unaligned source, sentinels, invalid sizes and overlap. Gate behavior must not depend on NaN equality. Record a standalone source-independent component comparison with exact bytes.

Commit this change separately if exact checks and meaningful component benefit pass. Archive the complete release binary, metallib and build-identity/source artifacts before step 2. If conversion is already as fast or benefit cannot be demonstrated, retain evidence and do not introduce a redundant runtime path; proceed with scratch independently from unchanged production code.

## Step 2: Reuse conversion scratch within each joined lane
For the explicit packed policy only, replace per-piece zero-filled `Data` allocation with one bounded uninitialized scratch buffer owned by each synchronous joined reader lane. Determine required capacity from the lane's possible source pieces and retain at most one original source record per lane. Do not allocate multi-row buffers, create a new background worker, share a scratch pointer across lanes, or retain across requests. Preserve scale/weight conversion destination and order. On partial/failed reads no scratch bytes may be converted or published; every lane drains and frees exactly once before throwing. Preserve existing scalar path and pre-existing staging bounds.

Implement a narrowly scoped benchmark toggle only if necessary to isolate allocation from conversion, defaulting safely and not exposing another public option; prefer archived intermediate binaries. Include byte/source checks and meaningful fault injection around multi-job lane reuse, mixed source widths, and partial reads so stale scratch cannot contaminate later jobs.

Commit scratch separately only after its exact checks and bounded reader comparison pass. Compare step1-only against step1+scratch with original reads and full allocation/conversion/wrapping included; do not compare only malloc cost or claim SSD throughput from cached reads.

## Verification and resource rules
No full-model launches during executor implementation. Only one build or bounded MLX/component process at a time, with reclaimable preflight before each. The shared Mac has 48 GB RAM, live user apps, and no stress authorization. Check `pgrep -fl slotstream` and `vm_stat`; free+purgeable+file-backed pages must provide target+3 GB. Source model path for a later bounded component, only after reviewer coordination, is `/Users/imdad/Documents/Codex/2026-09-12/wha/outputs/slotstream/models/jang-6s`.

Commands from worktree:
- `python3 -m py_compile <new benchmark drivers>`; exit0.
- Build standalone C/Swift comparator using `xcrun clang -O3` and `xcrun swiftc -O`; save exact commands and output.
- `SLOTSTREAM_BUILD_JOBS=2 make build`; exit0 with verified before/after source identity and bundled metallib.
- `swift build -c release -j 2 --product slotstream-checks` then `.build/release/slotstream-checks --tier t0 --tier t1 --filter exact-widening`; focused group passes, no skip.
- `.build/release/slotstream-checks --tier t0 --tier t1`; all bounded non-model checks pass. Run once at final state, no full verify.sh campaign.
- `git diff --check`; no whitespace errors.
- Validate archived release build with `Tools/flash/common.py::validate_build_identity`, using historical validation for an intermediate source archive after later edits.

Re-use existing SwiftPM dependency caches if possible, but never symlink source files or modify shared checkout sources. Do not overwrite the archived baseline `.build/flash/runs/neon-full-model-20260916/candidate-archive`. If creating a separate build is unnecessarily expensive, ask the reviewer about a serialized shared ignored build directory; do not silently falsify build identity.

## Reviewer qualification and completion
Reviewer will inspect diffs/tests, independently run focused checks, and perform a short interleaved full-model comparison of previous NEON, metadata-only (if kept), and metadata+scratch (if kept), using fixed JANG_6S target, exact outputs/work, sampled footprint and thermal/paging eligibility. No formal ten-pair qualification or default activation. Reviewer curates raw evidence, measurement records, plans and generated docs, then commits locally under the user's authorization.

Stop and report for source drift, unexpected alias/lifetime contracts, failure to reproduce exact bits, repeated verification failure, any requirement for out-of-scope source changes, or inability to produce trustworthy measurement. Document sensible minimal adjustments rather than silently changing the objective. Report commits, exact commands, paths/hashes of archived intermediate binaries, per-stage component results and all verification outcomes; never claim end-to-end improvement from a component comparison.
