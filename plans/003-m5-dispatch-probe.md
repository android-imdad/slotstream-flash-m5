# Plan 003: Measure the existing M5 expert-kernel path

> Executor: implement only this bounded diagnostic from Plan 001's stages 1/5. Preserve inference defaults and arithmetic. Reviewer maintains the index. Commit only in `advisor/001-flash-m5`; no merge/push. Report unsupported or unobserved hardware paths explicitly.

## Status

- P1 / M / MED / diagnostics-performance.
- Planned at `91bbaf7`, 2026-09-12; Plan 002 host-tool changes were reviewed and independently verified.
- Worktree: `/Users/imdad/Documents/Codex/2026-09-12/wha/outputs/slotstream-flash-m5`.
- Depends on the reviewed Plan 002 host foundation and frozen baseline archive. Synthetic work needs no complete weights; model-backed probes wait for reviewer model-slot handoff.
- Foundation baseline archive: `.build/flash/runs/baseline-v5`; summary: `.build/flash/runs/baseline-summary-final.json`. Complete model verification and smoke succeeded. The reviewer has handed over the model slot; retain fresh admission checks and serial jobs.
- Parent plan remains `001-flash-inference-m5.md` revision 2; this is not permission to implement its custom kernels, cache policies or predictor yet.

## Why and source anchors

The pinned mlx-swift 0.31.6 includes MLX 0.31.1 NAX kernels. Slotstream may already use M5 GPU Neural Accelerators for grouped prefill. A source branch or chip name does not prove a runtime dispatch; this diagnostic must distinguish eligibility from observation before suggesting new kernels.

Read `.build/checkouts/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/quantized.cpp`: `GatherQMM::eval_gpu` chooses grouped RHS work for `M==1 && B>=16 && right_sorted_ && B/E>=4`; `gather_qmm_rhs` calls the NAX variant when device/type policy permits. `device.h:is_nax_available()` checks OS >=26.2, architecture generation and `MLX_METAL_NO_NAX`. `utils.h:enable_tf32()` defaults `MLX_ENABLE_TF32` to 1, cached per process. Record it; do not change it mid-process.

`Layers.swift:1390–1410` pads prefill to `max(16,4*group.count)` and sets `sortedIndices:true`. Decode at `Layers.swift:1253–1263` uses unsorted one-row gathered products. JANG caches FP32 metadata; real affine gather promotes x/metadata together. `CheckpointFormat.swift` sets a uniform six-bit cache for 6S, including exact widened four-bit source codes. Do not change any of these paths.

`Diagnostics+JANG.swift` demonstrates real original-row reference checks and CheckBuilder. `Sources/SlotstreamTestKit/T0Checks.swift` registers catalogue groups. `Sources/slotstream-cli/main.swift` registers JANGCheck and renders CheckReports. Match these patterns.

`GPU.startCapture` exists in `Source/MLX/GPU+Metal.swift`; its local documentation lists `MTL_CAPTURE_ENABLED=1` and MLX debug setup. Local pinned C++ `metal.cpp` calls `MTL::CaptureManager`. Treat documentation/source differences as something to verify in an owned diagnostic child. Xcode is installed at `/Applications/Xcode.app`; a read-only `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun xctrace list templates` confirmed `Metal System Trace` and `Core ML` templates. Do not change global xcode-select or require an OS upgrade.

## Scope

Create `Sources/SlotstreamDiagnostics/Diagnostics+M5.swift`, `Sources/slotstream-cli/Flash.swift`, and `Tools/flash/m5_probe.py` plus focused tests. A bounded `Tools/flash/mlx_dispatch_probe.py` helper is allowed to reproduce the private dispatch-log experiment described below. Register command/checks in `Sources/slotstream-cli/main.swift` and `Sources/SlotstreamTestKit/T0Checks.swift`. Extend Plan 002's Tools/flash gate/receipt helpers only for these diagnostics. Artifacts live under `.build/flash/`; no original checkout changes, production package/dependency mutations, predictor, ANE model, custom production kernel, public default or HTTP changes.

An isolated debug dependency/build copy is allowed under `.build/flash/` only if actually needed for capture. Do not modify `.build/checkouts`, shared artifacts, Package.swift/resolved or the baseline archive. If trace support needs broader changes, report the smallest blocked boundary and retain a useful diagnostic with observation status unknown.

Recovery findings: Documents access was restored after the user granted macOS permission. Bounded 5-second system traces now complete safely, but their exported tables show shader compilation separately from executed encoder IDs without a shader-to-encoder join. A private pinned-MLX copy may therefore log at the actual `dispatch_threadgroups` sites for `gather_qmm_rhs_nax` and `gather_qmv`, after submission. Require the selected case's GPU evaluation and numerical checks to finish, with exact PID, kernel name, bits and geometry. Preserve the source patch and all build/shader hashes. This establishes submitted/completed kernel evidence for that **instrumented diagnostic build**, not production-binary utilization. No production math, flags or dependency files change. Grouped-six-bit and decode-six-bit runs are separate controls; unrelated reference/control kernels must not supply the claimed evidence.

All capture/export processes need time, memory and output limits. Stream exports to bounded files; no unbounded `capture_output` or wildcard export of unrelated tables. Maximum profiler cap is 6 GB with at least 9 GB fresh reclaimable preflight; the synthetic MLX allocation stays below 256 MiB. Preserve earlier failed captures. No further whole-system trace is required once the mapping limitation is established.

As new diagnostics change the live source inventory, historical reference validation must use the frozen archive's internal hashes and original receipt; it must not demand equality with the changed current source tree. If a helper extension is needed, make historical validation an explicit mode and keep current-build validation strict by default. Test that a historical archive can validate after a legitimate source change while a purported fresh build with that old source set still fails.

## Steps

1. Add `slotstream m5-check --synthetic --output <fresh-dir>` and mutually exclusive `--model <path>` (original bounded samples only). Synthetic cases use deterministic generated tensors matching H=2560, intermediate=640, group=64, bits 4/6 including exact widened mixed source records, FP32 JANG metadata, and batches 1/4/16/64/256. Compare sorted grouped prefill with unsorted decode. Keep <=256 MiB live allocation, bounded cache, sequential cases; never allocate a full model/layer. Report actual shape, stride, dtype, bits, device, OS, effective TF32, build/metallib identities, expected dispatch and observed dispatch separately.
2. Register `m5-dispatch` as a meaningful small T1 check plus pure eligibility/fallback tests as appropriate. Use existing native quantizedMM/dequantization and independent scalar spot checks as references, not two calls to the same new wrapper. Include wrong indices, sort/padding, noncontiguous fallback and all three projections. Document tested numerical tolerances and measured errors; no quality claim about complete-model logits. A bookkeeping test can pass on unsupported hardware while hardware observation stays unverified.
3. Add optional trace capture in a separate owned child via m5_probe.py. First attempt one representative grouped six-bit prefill and one decode control. A capture failure/nonzero child must preserve evidence and set observed path unknown; do not kill a parent or replace the baseline. Use public MLX/Metal APIs or the installed xctrace tool. Bind capture to its exact diagnostic executable and report any difference from the production-equivalent build. Inspect actual dispatch/encoder records with kernel references; a text search finding NAX strings in a metallib or bundled source is not evidence of execution. No utilization percentage without corresponding counters.
4. Extend strict gates to require the exact registered check names, zero required skips, fresh build identity and positive Python tests. Store a versioned report with per-case results, errors, observation status/reason and complete hashes. Model mode never downloads weights, and is not a full-model benchmark. Keep NAX usage unverified if the available tooling cannot demonstrate it; the diagnostic can still report eligible paths and correctness.

## Verification

Before native verification, `make build SLOTSTREAM_BUILD_JOBS=2` must pass with current identity. Then run:

```sh
.build/release/slotstream-checks --tier t0 --tier t1 --filter m5 --json
.build/release/slotstream m5-check --synthetic --output .build/flash/runs/m5-synthetic
python3 Tools/flash/m5_probe.py --binary .build/release/slotstream --synthetic --capture --output .build/flash/runs/m5-capture
python3 -m unittest discover -s Tools/flash -p 'test_*.py'
```

Validate JSON reports with strict helper functions; exit zero with missing checks is not enough. Full catalogue T0/T1 must still pass. The Python probe uses monitored child ownership, fresh paths and the foundation artifact conventions. Test capture unavailable/failure/unsupported device, missing/stale hashes, zero actual dispatch records, and an NAX string embedded only in a source blob. Each must stay unverified rather than produce a usage claim.

## Done and STOP

Done: command/checks implemented, all numerical/component checks pass, measured allocation bound holds, exact diagnostic provenance saved, and actual NAX observation or a precise unverified reason reported. This is capability/dispatch evidence, not an LLM speedup. Only observed records may support `already_used` in the parent plan. Do not mark its stage 1 or stage 5 fully done from this subtask alone.

Stop on source drift affecting these assumptions, missing safe memory headroom, scope expansion, inability to establish owned-child cleanup, repeated verification failure or a required dependency/OS modification. No model run without the model-slot handoff. Report STATUS, STEPS/commands, STOPPED BECAUSE if needed, FILES CHANGED, NOTES including commit and evidence paths. Reviewer will inspect/re-run checks before the next stage.
