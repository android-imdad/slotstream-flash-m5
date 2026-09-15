# Current execution handoff — September 15, 2026

**Public WIP research repository:** [android-imdad/slotstream-flash-m5](https://github.com/android-imdad/slotstream-flash-m5), branch `main`. The local working branch remains `advisor/001-flash-m5`; the original `origin` remote remains upstream. Publication here does not mean an upstream merge, release or default activation.

## State

Working branch: `advisor/001-flash-m5`, in this repository. No model, benchmark,
profiler or owned test process is running. Earlier active-session notes remain
in Git history; do not resume their closed process IDs or stale coordination.
Original checkpoint bytes and immutable local evidence remain unchanged.

## Completed

- Plans002–008,010,012: verified JANG_6S baseline; existing M5 diagnostic dispatch;
  tokenization, bounded capture, corpus/quality metrics and terminal sampling.
- Plan011 implementation: exact packed4-to6 widening, default-off, with bounded
  parity and a matched three-pair-per-workload 24 GB development benchmark.
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

1. Plan011's predeclared ten-pair qualification remains pending. The separate
   three-pair 24 GB checkpoint is measured evidence, not that qualification.
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
