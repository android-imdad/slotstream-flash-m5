# Current execution handoff — September 15, 2026

**Public WIP research repository:** [android-imdad/slotstream-flash-m5](https://github.com/android-imdad/slotstream-flash-m5), branch `main`. The local working branch remains `advisor/001-flash-m5`; the original `origin` remote remains upstream. Publication here does not mean an upstream merge, release or default activation.

## State

Working branch: `advisor/001-flash-m5`, in this repository. No model, benchmark,
profiler or owned test process is running. Earlier active-session notes remain
in Git history; do not resume their closed process IDs or stale coordination.
Original checkpoint bytes and immutable local evidence remain unchanged.

## Completed

- Latest optimization: the explicit packed-widening path now uses an internal
  CAffine NEON backend. Release build, focused exhaustive/vector-tail checks and
  bounded original-source comparisons pass. A standalone conversion comparison
  is recorded; a [short 14 GB full-model comparison](../db/records/measurements/jang-neon-full-model-2026-09-16.md) is now complete. Formal qualification remains pending. See
  [SIMD evidence](../db/records/measurements/jang-simd-widening-2026-09-15.md).
- Plans002–008,010,012: verified JANG_6S baseline; existing M5 diagnostic dispatch;
  tokenization, bounded capture, corpus/quality metrics and terminal sampling.
- Plan011 implementation: exact packed4-to6 widening, default-off, with bounded
  parity and a matched three-pair-per-workload 24 GB development benchmark.
- Plan011 follow-up: current-source component/native/host checks, fresh exact
  parity and the fresh screen pass. Both full-study attempts are thermally
  inconclusive. The shared cooldown is implemented and tested without changing
  inference controls or acceptance thresholds.
- Stage-five CPU diagnostic: a bounded packed short-prompt sample preserves
  exact output/work. It ran under recorded fair conditions and has no eligible
  performance timings or GPU-duration claim. Nominal-only long-prompt profiles
  never launched; their owned waiters were stopped.
- Plan009: dense-ten exact; eight/six fail fidelity. Four stopped by user after
  partial work, two untested. Do not resume lower-count tests automatically.
- Plans004,013–015: cache windows, balanced read scheduling, expanded/source-native
  layout screens completed and rejected; no runtime policy or full-model repack.
- Plan016: full batch-size reader-cost grid and nine fresh matched 14 GB controls
  completed; both causal prefetch forecasts rejected by the empirical gate.
  No native prefetch worker or new ANE execution was implemented.
- Public JANG docs, canonical measurements/decisions/status, claims and generated
  projections reconciled with the findings. See the documentation verification
  logs under `.build/flash/docs-update/` for the final checks.

## Remaining work

1. The short 14 GB NEON versus previous-packed comparison is complete.
   A matched comparison at the saved 24 GB configuration remains a separate scope.
   Plan011's predeclared ten-pair qualification remains pending after two thermal
   stops. The cooled sky-blue/Python subsets are complete; arithmetic is not.
   Do not replace failed pairs or combine attempts. The separate 24 GB checkpoint
   remains development evidence.
2. Parent001's aggregate run-set, stage-seven gates/evaluator and final coverage
   remain incomplete. Proposed commands in the plan are not existing tooling.
3. Broader/long-context qualification, JANG_4M full-model evidence and
   publisher/BF16 reference reproduction remain separate gaps. The evaluation
   environment/bootstrap and large historical inputs are local, not fully
   portable clean-clone infrastructure.
4. Serving integration/default rollout, upstream integration/release and production M5 utilization
   claims remain deferred/unverified. The original-checkpoint learned Expert
   Lookahead proposal is separate from the rejected JANG heuristics.

The trained neuron predictor/sparse runtime/optional ANE predictor are **not
selected** for the failed neuron policy, rather than unfinished admitted work.
A new method or budget needs its own predeclared investigation.

## Evidence to preserve

- `.build/flash/runs/engine-benchmark-24gb-cooled-20260915`
- `.build/flash/runs/oracle-campaign-v2-20260915`
- `.build/flash/runs/read-scheduling-20260915`
- `.build/flash/runs/whole-expert-20260915`
- `.build/flash/runs/source-native-20260915`
- `.build/flash/runs/prefetch-cost-20260915`
- `.build/flash/runs/widen-parity-qualification-20260915`
- `.build/flash/runs/widen-screen-qualification-20260915`
- `.build/flash/runs/widen-qualify-20260915`
- `.build/flash/runs/widen-qualify-cooled-20260915`
- `.build/flash/runs/jang-stage5-cpu-sample-20260915`
- `.build/flash/qualification-20260915`
- `.build/flash/simd-widening-20260915`
- `.build/flash/runs/pre-simd-widening-20260915`
- `.build/flash/runs/simd-widening-source-component-20260915`

The public portable extract is
`db/sources/runs/2026/09/jang-flash-findings-20260915.json`. It preserves result
fields and original report hashes; full binaries, captures and logs remain in
ignored local archives. Completed harness snapshots must be validated against
their own frozen hashes, not against later live tool additions.

The old checkout alias resolves to this repository for historical evidence
paths. Only verified aliases may be normalized; never rewrite archive hashes
or replace an immutable reference with a new build. Run one model process at a
time, keep explicit memory targets and preflight headroom, and separate clean
benchmark timing eligibility from functional/global-paging diagnostics.

[Plan index](README.md) · [Canonical status](../db/records/plan/jang-flash-qualification-status.md) · [Public findings](../docs/JANG-FINDINGS.md)

## September 16 reader follow-up

Plan017 completed both recommended experiments. The release metadata loop already vectorizes; a separate explicit SIMD runtime path was not added. Lane-local CPU scratch passed its exact/fault checks but did not demonstrate reader benefit, so it was restored out of production. Benchmark tools, raw samples and the candidate patch are retained in [the canonical result](../db/records/measurements/jang-reader-followup-2026-09-16.md). No new full-model throughput claim or default activation follows.
