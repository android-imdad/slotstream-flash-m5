# JANG Flash findings and current status

This [public WIP research fork](https://github.com/android-imdad/slotstream-flash-m5) records local source-checkout results for
[JANG_6S on Hugging Face](https://huggingface.co/JANGQ-AI/Qwen3.8-Flash-Next-JANG_6S) on one Mac. The original PipeNetwork checkpoint remains the default; this is not an upstream release, automatic hardware support expansion, or a new default configuration.

## What works today

- The complete pinned JANG_6S checkpoint was downloaded and verified, and full-model text generation was exercised.
- Exact packed widening is available explicitly through the local run command and Engine API. It preserves the original represented weights and checkpoint files.
- Frozen tokenization, bounded activation/logit/state capture, terminal memory sampling and full-vocabulary quality metrics support reproducible comparisons.
- M5 GPU dispatch was observed in an instrumented MLX build. Production utilization remains unverified. No new Apple Neural Engine execution was implemented.

## Measured widening result

The latest implementation adds an ARM NEON CPU backend to the existing
`packed4-to6` option. Focused exact-byte and checkpoint-source checks passed.
Its component comparison is recorded separately; full-model TPS for the NEON
backend has not been measured. The development figures below belong to the
preceding Swift packed implementation. [SIMD implementation and component
evidence](../db/records/measurements/jang-simd-widening-2026-09-15.md).

A local JANG_6S benchmark at a 24 GB target measured 6.98–7.41 tok/s with explicit packed widening, about 2.2× scalar throughput.

Protocol: three paired rounds per workload on the local M5 Max with 48 GB unified memory; a 24 GB process target; 32,768 configured context tokens; greedy seed 42; up to 128 output tokens; fresh CLI processes; prefix cache, MTP and vision off.

| Workload | Scalar tok/s | Packed tok/s | First text seconds, scalar → packed |
|---|---:|---:|---:|
| Coding | 3.331 | 7.409 | 5.311 → 2.584 |
| Explanation | 3.199 | 6.983 | 5.870 → 2.930 |
| Reasoning | 3.272 | 7.266 | 6.623 → 3.235 |

All paired outputs and completed work matched exactly. The maximum observed packed-process footprint was 20.69 GB. First-text measurements exclude loading and cooldown. The configured context limit is not evidence of long-prompt quality.

This is a measured development checkpoint. Formal qualification was subsequently attempted and remains thermally inconclusive; the parent final run-set remains pending. The saved upstream target was not beaten; it uses another quantization and persistent HTTP serving with MTP/lookahead, so it is not the causal control for the speedup above.

[Canonical measurement](../db/records/measurements/jang-exact-widening-2026-09-15.md) · [Full benchmark method](../plans/009-oracle-execution.md#completed-24-gb-benchmark) · [Saved upstream target](../BASELINE.md)

## Neuron skipping: current approach rejected

The oracle masks blocks of intermediate neurons inside routed experts. It still loads dense experts and performs the full down projection; its runtime is not a sparse-inference benchmark. Comparisons are against unmodified JANG_6S, not a BF16 or publisher reference.

| Retained blocks | Positions | Mean KL | p99 KL | Top-1 agreement | PPL ratio | Result |
|---|---:|---:|---:|---:|---:|---|
| 10 | 896 | 0.0000 | 0.0000 | 100.00% | 1.0000 | Exact control passed |
| 8 | 896 | 0.2036 | 2.2378 | 83.93% | 0.9962 | Fidelity rejected |
| 6 | 896 | 0.4098 | 3.8357 | 74.33% | 0.9586 | Fidelity rejected |
| 4 | 512 | — | — | — | — | User-stopped; incomplete |
| 2 | 0 | — | — | — | — | Not run |

The small PPL-ratio improvements do not override the failed KL/top-1 gates. The lower settings were stopped at user direction; their missing results are not inferred. No partial mask is enabled. The dependent neuron predictor, sparse runtime and optional ANE predictor are not selected for this approach.

[Canonical measurement](../db/records/measurements/jang-neuron-oracle-2026-09-15.md) · [Oracle protocol and stopping decision](../plans/009-oracle-execution.md)

## Exact-loading experiments

Cache-window replay matched native CLOCK but failed its admission gate. Later reader experiments measured complete allocation, reads, conversion/copy and MLX wrapping, with output hashing outside timing.

| Candidate | Reader-time outcome | Result |
|---|---|---|
| Balanced read scheduling | 40.39% longer at default queue depth | Rejected |
| Expanded whole-expert layout | 8.48% longer paired total time | Rejected |
| Source-native whole-expert layout | 0.44% longer point estimate; essentially tied | Rejected |

The near-tie does not establish a meaningful slowdown or speedup. These are bounded reader measurements, not full-model throughput results. No full-model repack or generation activation followed. The source-native sample stored original bytes without requantization; the expanded sample stored the larger pool representation.

[Canonical measurement](../db/records/measurements/jang-loading-screens-2026-09-15.md) · [Scheduling](../plans/013-balanced-read-scheduling.md) · [Expanded layout](../plans/014-whole-expert-layout.md) · [Source-native layout](../plans/015-source-native-layout.md)

## SSD prefetch: existing heuristics rejected at the tested scope

Prefetch admission was evaluated at 14 GB using the original diagnostic workloads. Historical routes were joined to fresh matching untraced packed-widening controls. Reader costs were measured across the full supported batch-size grid; staging was charged against the cache.

| Forecast | Modeled ideal-overlap estimate | Optimistic sample sensitivity | Result |
|---|---:|---:|---|
| Previous-token | -0.64% to +0.34% | At most +0.51% | Rejected |
| Recent-frequency | 4.62–4.77% | At most 6.47% | Rejected |

These are empirical estimates, not observed native prefetch speedups. The model grants free predictions/insertion and all non-reader decode time as a shared overlap budget. Sample extrema are sensitivity checks, not physical bounds or confidence intervals. Both forecasts missed the predeclared gate; no native worker was implemented. Different predictors, workloads and memory budgets remain separate questions.

[Canonical analysis](../db/records/measurements/jang-prefetch-cost-screen-2026-09-15.md) · [Complete cost/overlap method](../plans/016-prefetch-cost-screen.md)

## M5 GPU versus Neural Engine

The instrumented grouped expert case submitted `gather_qmm_rhs_nax`; decode submitted `gather_qmv`. GPU evaluation and numerical checks completed. This verifies that diagnostic build’s existing path. It does not prove utilization in the unmodified production binary, and it does not establish a new LLM speedup from GPU changes.

The Apple Neural Engine is a separate device. The planned ANE work was limited to a useful neuron predictor. Its prerequisite failed, so no predictor was trained or converted and no ANE execution is claimed. Whole-model ANE conversion is outside the current plan.

[Foundation and dispatch evidence](../db/records/measurements/jang-flash-foundation-2026-09-15.md)

## What remains

- Complete the paired widening qualification under suitable thermal conditions. Both attempts stopped at non-nominal endpoints; completed workload subsets cannot qualify the incomplete whole cohort. Keep the earlier development result scoped to its actual workloads and budget.
- Implement the parent aggregate run-set and final qualification/reporting interfaces. Some commands in the original plan are still specifications, not existing tools.
- Qualify broader tasks and long-context behavior before making broader claims. JANG_4M full-model evidence and publisher/BF16 reproduction remain separate gaps.
- Treat CLI serving integration, default rollout and upstream integration/release as separate work. Existing scalar defaults remain.
- Keep the original-checkpoint learned Expert Lookahead plan separate: JANG heuristic rejection does not evaluate that learned approach.

The latest execution revalidated exact widening and added shared prelaunch
cooldown handling. A separate CPU stack diagnostic observed SSD reads, packed
expansion, staging and waits while preserving output/work equality. It ran under
recorded fair conditions; its timings cannot support speed claims, and CPU
stacks do not establish GPU utilization or kernel durations. Clean longer-prompt
profiling remains open. [Current attempts, completed subsets and diagnostic
limits](../db/records/measurements/jang-widening-qualification-attempts-2026-09-15.md).

[Current canonical plan](../db/records/plan/jang-flash-qualification-status.md) · [Detailed plan index](../plans/README.md) · [Admission decision](../db/records/decisions/jang-flash-admission-2026-09-15.md)

## Reproduction and evidence availability

Use [the JANG guide](JANG.md) for building, verifying weights and explicit run/library controls. [Testing](TESTING.md#jang-flash-experiments) lists the diagnostic tools and their limits.

The public [portable findings extract](../db/sources/runs/2026/09/jang-flash-findings-20260915.json) contains the approved result fields and original source-report hashes. It was exported from completed local reports; it is not a new inference run. Full binaries, logits, traces and large sample artifacts remain in ignored local archives. A fresh clone cannot reproduce a historical run without the matching archives and checkpoint. Never replace an immutable reference with a new build and keep calling it the original control.
