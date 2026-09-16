---
type: measurement
id: 01m2mepc2hc2h3vstzy9zga8cq
created: 2026-09-16T06:33:00.753520+00:00
updated: 2026-09-16T06:33:00.753520+00:00
summary: Independent JANG 512 prefill qualification
date: 2026-09-16
doc: measurements
level: '2'
machines: '[[records/machines/local-m5-max-48gb]]'
order: '1261'
runs: '[[sources/runs/2026/09/jang-prefill-512-numerics-20260916]]'
title: Independent JANG 512 prefill qualification
status: measured
---
**Disposition: the 512-token fallback is not admitted.** Both fresh prompts fail retained-state and aggregate-prefill-routing gates. Full-vocabulary logit, candidate/reference greedy-token and integer-state checks pass. Keep the ordinary 256-token prefill default; its earlier explicit packed-widening and NEON decode measurements remain unchanged.

### Prospectively frozen independent method

Plan020 selects 256 as reference, 384 as empirical rechunk control and 512 as candidate. The control is the arithmetic midpoint, selected before observing its results; it is not independently qualified or admitted as an alternative default. These are fresh captures of two previously frozen but unrun Plan019 tasks, `grounded-document` and `reader-offsets`, with 2040 and 1389 chat tokens. No old numerical/timing rows are pooled, no old candidate error extrema set the threshold, and neither prompt nor control is replaced after results.

The only source change from the Plan019 instrument adds 384 to its diagnostic allowlist and focused validation checks. Production inference, planner policy, weights and defaults are unchanged. A fresh two-job source build, the native capture self-check, all 55 T0/T1 groups and all 40 host comparator/task-driver tests pass. The actual planner/recorder confirms 885, 846 and 808 cache slots for 256, 384 and 512 at the same total target. The evaluator verifies charged workspace and pool bytes, observed pass schedules, actual chunk execution and chronological route coverage.

All arms use the actual Engine/Generator, pinned JANG_6S, explicit NEON packed4-to6, a 14 GB process target, 4096 configured context and the same deployed M5 Max controls, with prefix caching, MTP and vision off. The reference supplies eight common raw-argmax continuation IDs. All other arms teacher-force that exact list. Full logits and active state are captured after prefill, one continuation token and eight continuation tokens; routing covers every logical position and layer. The existing tensor-copy and artifact caps, real target-plus-3 GB preflight, one-model-at-a-time rule and terminal physical-footprint sampler remain in force.

All six processes complete with valid memory/terminal evidence. Maximum captured physical footprint is 13.910055560 GB, below the fixed target. These numerical runs are not speed benchmarks; global paging is a functional diagnostic and no clean-timing claim is made.

### Numerical results

The unchanged per-field gate is `candidate <= max(3 * control, 0.01)`, using reference maximum magnitude for state and reference spread for logits, with the established denominator floor. Integer state is exact; candidate/reference greedy tokens must agree. Routing uses keep-set replacements aggregated across every layer within each phase; per-layer observations are nongating diagnostics.

| Prompt | Failed state comparisons across checkpoints | Unique state fields | Additional failed gate |
|---|---:|---:|---|
| grounded-document | 36 | 12 | Aggregate prefill routing |
| reader-offsets | 26 | 10 | Aggregate prefill routing |

Counts include retained history observed again at later checkpoints. They are not counts of independent bugs or wrong answers. The complete reports retain every one of the 432 individual measurements per prompt and both aggregate routing results.

| Aggregate prefill routing | Control disagreement | Candidate disagreement | Allowed |
|---|---:|---:|---:|
| grounded-document | 0.253574% | 2.921058% | 1.000000% |
| reader-offsets | 0.409617% | 2.854272% | 1.228852% |

All six full-vocabulary logit and candidate/reference greedy-token checks pass. Integer state and both continuation-routing aggregates pass. Control deviations and control greedy choices remain visible; this relative-drift method is not an absolute model-quality oracle.

An independently checked reader example is `prefill/value.35`, shape `[1, 2, 1389, 256]`, at head 1, logical prompt row 522, component 69. Every arm identifies that row as prompt token 585. Reference and control hold `-0.287109375`; the candidate holds `36.5`. Maximum absolute delta divided by reference maximum magnitude is `0.987573406`, exceeding the bound `0.138108221`. The cache implementation appends chronological rows, growth preserves their positions, and the export excludes unused capacity. MLX's copied data is contiguous. Each arm retains the same prompt values at subsequent checkpoints, so the repeated failures do not establish continued growth in the discrepancy. The underlying computational cause remains undiagnosed; no kernel or algorithm explanation is claimed.

Independent review verifies all 2634 native artifact hashes, prompt/continuation identities, finite arrays, shapes, deployed options, binary/source/model metadata, pass schedules and pool pricing, and independently reproduces every recorded metric. No metadata, routing-alignment or cached-row-base mismatch explains these failures. Thresholds and data remain unchanged.

### Conditional work and next scope

The eight-arm objective task screen and conditional eight-arm short-request/long-completion extension were prepared and tested without model inference, then **not run after the numerical rejection**, as Plan020 prescribed. Prepared task fixtures are not task-quality or latency evidence.

Neither 512 nor 1024 is admitted by these investigations. Do not try the 384 control as a new default simply because it served as a comparator. Any further chunk-size work requires a separately declared investigation of the observed divergence. Existing Plan011 packed-widening qualification and serving/default integration remain separate pending work; the prior decode improvements measured with 256-token prefill are preserved.

Evidence: [[sources/runs/2026/09/jang-prefill-512-numerics-20260916]]. Decision: [[records/decisions/jang-prefill-512-not-admitted-2026-09-16]].
