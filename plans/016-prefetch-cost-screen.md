# 016 — Causal SSD-prefetch cost and overlap screen

## Current disposition — September 15, 2026

**COMPLETE — both tested heuristics rejected.** Reader costs and fresh matched 14 GB controls completed; modeled gains missed admission. No native worker, 24 GB speedup claim or ANE execution.

[Evidence/current findings](../docs/JANG-FINDINGS.md) · [Canonical status](../db/records/plan/jang-flash-qualification-status.md). The original execution brief and dated review notes below are retained as history; current work is governed by [the plan index](README.md). Do not restart completed or unselected steps from a historical instruction.

Status: COMPLETE — both existing forecasts rejected by the 14 GB empirical screen. User authorized the recommended next action after the
pending-work report: complete the timing/cost screen and make an explicit
implement-or-reject decision for the existing two causal forecasts.

## Scope and controls

The existing previous-token and recent-eight-frequency forecasts have exact
native-CLOCK-reconciled traces at 14 GB, context 2048, greedy seed 7, with three
fixed development prompts. Their original scalar timings must not be combined
with packed-widening costs as though they were one execution mode. Matching
historical packed timings exist, but one fails the paging eligibility check.

This study therefore collects **three fresh untraced packed-widening controls
per workload**, rotating workload order, and reuses only the historical route
data. Every fresh control must match exact input/output IDs, completed work,
pool/plan/numerical/environment controls and native read counts/bytes. Replay
must reproduce each fresh control's counts. No historical timing enters the
admission calculation. Reference archive/receipt and original model identities
are revalidated without requiring the old collector's hashes to match today's
harness; historical evidence is preserved unchanged.

Scope stays 14 GB because those are the available matched forecast workloads.
The saved 24 GB prompts are different, and reclaimable memory at scope selection was below
the 27 GB launch preflight for a fresh 24 GB run. Headroom later recovered; the
predeclared matched 14 GB protocol was retained. There is no inferred 24 GB
speedup or universal prefetch rejection from this screen. Each 14 GB model
launch requires 17 GB reclaimable; model processes remain serial.

## New reader-cost calibration

Use the original packed-widening reader, queue depth 32, representative source
layers 0/5/22 and fixed original expert IDs. Measure every count 1 through 10,
eight repetitions each with rotated count order. Zero misses has zero reader
cost by construction. No interpolation or extrapolation is needed. Whole calls
include source reads, lossless conversion, allocation and MLX staging; complete
output-byte hashes are checked outside timing. Original regions are capped at
256 MiB, process physical footprint at 512 MB, with 4 GB reclaimable preflight.
Record successful descriptor controls, with no cold-SSD claim.

## Timing model and predeclared decision

Replay baseline CLOCK and the staging-charged CLOCK side by side. Reserve
10 × (largest original record + expanded record), round up to cache slots, and
remove them from the candidate pool. Predictions never mutate that pool. Only
past routes generate forecasts; future actual demand is used only to score them.
Reconcile all byte totals against the existing causal replay implementation.

Sum empirical per-batch median costs for baseline, charged demand, remaining
demand after useful forecasts, and all predicted reads. Normalize to each fresh
control's measured decode reader time. All remaining decode time is granted as
one **global** overlap budget, including work a real implementation might not be
able to overlap. Reader scopes must not already overlap resident/shared GPU work.

Report both:

- Free-prefetch estimate: predictions, all speculative work, insertion and
  synchronization cost nothing; only residual demand remains exposed.
- Aggregate overlap estimate: predicted reader work can overlap all non-reader
  decode time, but that budget is spent only once. Excess work is exposed.

Also test an optimistic sensitivity using observed maximum original-batch and
minimum remaining-batch costs. It applies only when a forecast is useful, so
timing spread cannot invent gains for all-wrong predictions. These sample
extrema are **not physical bounds or statistical confidence intervals**.

Use medians across the three fresh control runs. All three workloads must show
at least 10% aggregate-overlap improvement to admit a native overlap experiment.
If even the optimistic sample-range estimate misses 10% in a required workload,
reject the heuristic under this empirical screen. Otherwise keep it inconclusive.
Neither outcome is a measured native speedup. If timing is contaminated or a
required control fails, preserve partial evidence and do not issue admission.

No native prefetch worker, runtime activation, predictor training, ANE execution
or default change is part of this screening stage. Report its benchmark result
before another optimization.


## Completed result — September 15, 2026

All thirty reader-cost cases completed eight exact-output repetitions. All nine
fresh model controls passed exact prompt/output/work, plan/numerical/environment,
cache replay, terminal physical-memory and timing-eligibility checks. No historical
timing was used in the decision. Both forecasts failed the empirical gate.

The following are **modeled improvement fractions**, not observed prefetch
speedups. Negative means the charged cache loss outweighed the estimated saving.
The optimistic column uses observed cost extrema as a sensitivity check, not a
physical bound or a confidence interval.

| Workload | Forecast | Free/ideal-overlap estimate | Optimistic sample sensitivity |
|---|---|---:|---:|
| sky-blue | previous-token | 0.335% | 0.505% |
| sky-blue | recent8-frequency | 4.616% | 6.052% |
| python-deduplicate | previous-token | -0.641% | -0.393% |
| python-deduplicate | recent8-frequency | 4.774% | 6.470% |
| multiply-37-by-42 | previous-token | 0.274% | 0.485% |
| multiply-37-by-42 | recent8-frequency | 4.711% | 6.140% |

The speculative work estimate fit inside the deliberately generous global
non-reader time budget for both forecasts in each workload, so the aggregate
overlap estimate equals the free-prefetch estimate here. A real implementation
must meet individual next-layer deadlines and pay synchronization/insertion
costs; this model does neither. Even the optimistic sensitivity remained below
the required 10% in every workload. The current heuristics therefore do not
justify a native prefetch implementation at this tested setting. This does not
rule out different predictors, workloads or memory budgets.

### Fresh control measurements

These are packed-widening baseline controls with no native prefetch, at 14 GB,
2048 context, greedy seed 7, three fresh processes per workload.

| Workload | Median decode tok/s | Maximum physical peak GB |
|---|---:|---:|
| multiply-37-by-42 | 5.099 | 10.017 |
| python-deduplicate | 4.869 | 10.064 |
| sky-blue | 5.420 | 10.004 |

Reader-calibration physical peak was 373,031,920 bytes under 512 MB. Maximum
full-model control peak was 10,063,893,856 bytes under 14 GB. Native read controls
succeeded, thermal endpoints were nominal, low-power mode was off, and paging
counters were unchanged in every counted launch. All ten cooldown windows and
all nine settling records were independently validated.

### Evidence and validation

- `.build/flash/runs/prefetch-cost-20260915/report.json`, SHA-256
  `49cb662be69ad0fa866ff9c3078b7ef9aca33d332afa12ac95570b684cfb8519`.
- `verification.json` binds 155 artifacts and records independent recomputation
  of every replay, timing estimate and decision; it also binds readiness,
  settling, native completion and receipt files.
- `harness-snapshot/` preserves every host file at its frozen protocol hash;
  `archive/` preserves the exact native binary, Metal library and source archive.
- Original trace artifacts and earlier ineligible historical timing remain
  unchanged. Only the two relocation aliases for existing verification files
  were normalized; all other model/header/hash fields had to match exactly.
- Release build passed (`prefetch-cost-build-20260915.log`), 40 native T0 checks
  passed (`prefetch-cost-t0-20260915.json`), and 226 host tests passed
  (`prefetch-cost-host-tests-final-20260915.log`). All output artifacts and the
  native build identity were revalidated, and `git diff --check` passed.

No native prefetch, new inference acceleration, ANE execution or default/serving
activation was added. No model or owned test process remains. The saved 24 GB
benchmark remains a separate earlier result; this screen makes no new 24 GB claim.
