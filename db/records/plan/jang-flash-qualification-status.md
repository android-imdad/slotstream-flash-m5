---
type: plan
id: 01m2jn131bsa68nkjdtjs8bfc0
created: 2026-09-15T13:45:14.539263+00:00
updated: 2026-09-16T04:10:47.224673+00:00
summary: JANG Flash experiment status and remaining qualification
date: 2026-09-16
doc: plan
kind: milestone
level: '2'
order: '7'
title: JANG Flash experiment status and remaining qualification
---
Public WIP repository: [https://github.com/android-imdad/slotstream-flash-m5](https://github.com/android-imdad/slotstream-flash-m5). Its `main` branch publishes the research source; this is separate from upstream integration or release. The original upstream remote and MIT attribution are retained.

Current local source and benchmark state after the September 16 comparison. This is not an upstream release or default activation.

| Track | State | Evidence and limits |
|---|---|---|
| Full JANG_6S checkpoint and baseline | Complete | Pinned download verified; real generation and physical-memory evidence exist |
| Tokenization, capture, metrics and terminal sampling | Complete at bounded scope | Immutable reference/corpus and versioned report validation |
| Existing M5 GPU dispatch | Observed in instrumented diagnostic build | Production utilization unverified; no new ANE acceleration |
| Exact packed widening | NEON CPU backend implemented under the existing explicit policy | Focused byte/source checks and standalone component comparison pass; short full-model comparison completed; formal qualification remains pending |
| Neuron-block oracle | Closed at user-revised scope | Eight/six failed fidelity; four stopped incomplete; two not run |
| Dependent predictor/sparse runtime/ANE | Not selected for that approach | No quality-passing neuron policy to advance |
| Cache windows and loading layouts | Tested and rejected | No new policy or full-model artifact admitted |
| Heuristic SSD prefetch | Empirical screen rejected at tested budget | No native worker or generation speedup |
| Parent final qualification | Pending | Aggregate run-set tooling, required final coverage and accepted candidate report remain incomplete |
| Public evidence/docs | Updated with bounded findings | Generated projections and claims retain the reported scopes |
| Serving/default rollout | Deferred | CLI serve has no widening flag; original defaults remain |

Next qualification work: Plan011 needs a fresh whole paired cohort under suitable thermal conditions, then the selected exact-arm parent run-set and final reporting. The original attempt stopped after 39 launches; the cooled successor stopped after 44. Sky-blue and Python completed their cooled subsets, but arithmetic did not complete its required cohort. Do not replace pairs or combine attempts. `Tools/flash/runset.py` and the proposed stage-seven gate/evaluator interface are not implemented; those parent-plan commands are specifications, not runnable tooling. The three-pair checkpoint is not full qualification.

JANG_4M full-model qualification, publisher/BF16 reference reproduction, broad/long-context coverage, production utilization claims and learned Expert Lookahead remain distinct uncompleted scopes. Do not transfer JANG_6S results to them. Detailed local status: `plans/README.md`; usage and findings: `docs/JANG.md` and `docs/JANG-FINDINGS.md`.

Evidence: [[records/measurements/jang-flash-foundation-2026-09-15]], [[records/measurements/jang-exact-widening-2026-09-15]], [[records/measurements/jang-neuron-oracle-2026-09-15]], [[records/measurements/jang-loading-screens-2026-09-15]], [[records/measurements/jang-prefetch-cost-screen-2026-09-15]]. Admission decision: [[records/decisions/jang-flash-admission-2026-09-15]].

Current-source follow-up: [[records/measurements/jang-widening-qualification-attempts-2026-09-15]] records passing prerequisites, both thermal stops and the separate bounded CPU sample. The CPU sample preserves exact output/work under recorded fair conditions but establishes no eligible speed or GPU-duration claim. Clean longer-prompt timing profiles never launched; owned prelaunch waiters were stopped, and no custom M5 kernel is admitted.

Latest implementation-first follow-up: [[records/measurements/jang-simd-widening-2026-09-15]] records the new CAffine NEON backend, its exact tails and portable fallback, the release build, focused checks and the short conversion comparison. Earlier full-model timings and thermal exclusions belong to the preceding packed implementation. The September 16 follow-up measured the two packed backends at a 14 GB target; the saved 24 GB setup remains a separate unmeasured NEON scope. No scalar/serving default changed and no long model qualification was repeated.

Latest bounded benchmark: [[records/measurements/jang-neon-full-model-2026-09-16]] records two alternating pairs for each short workload, exact output/work equality and monitored memory/thermal/paging conditions. It is exploratory development evidence, not formal qualification or default activation.
