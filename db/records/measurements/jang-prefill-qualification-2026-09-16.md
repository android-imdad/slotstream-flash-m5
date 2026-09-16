---
type: measurement
id: 01m2md1cg0bz6qcv9tq9fg5d7x
created: 2026-09-16T06:04:04.480251+00:00
updated: 2026-09-16T06:04:04.480251+00:00
summary: JANG larger-prefill numerical qualification
date: 2026-09-16
doc: measurements
level: '2'
machines: '[[records/machines/local-m5-max-48gb]]'
order: '1260'
runs: '[[sources/runs/2026/09/jang-prefill-numerics-20260916]]'
title: JANG larger-prefill numerical qualification
status: measured
---
**Disposition: the 1024-token candidate is not admitted.** Three retained-state fields in the prose case exceed the existing rechunking band. The code case passes the canonical numerical checks. No prefill, widening or serving default changed. These small discrepancies do not by themselves prove a bad answer; they fail the acceptance rule required before adopting the faster configuration.

### Matched instrument and scope

The new `prefill-capture` diagnostic constructs the actual Engine with the pinned JANG_6S checkpoint, explicit NEON packed4-to6, a 14 GB process target, 4096 configured context and deployed optimization controls. Prefix caching, MTP and vision stay off. The only capture hook is package-only and nil in ordinary inference; it retains the completed prefill logits and state without replaying or modifying the prefill. Ordinary one-token generation must leave exactly the prompt consumed and match the captured raw greedy choice.

Each chunk uses its actual fixed-budget planner result: 885, 808 and 653 slots at 256, 512 and 1024. Two frozen natural Plan018 prompts have 1843 and 1342 chat tokens. Each is captured at all three chunks in a separate serial process. The reference supplies eight common raw-argmax continuation IDs; the other arms teacher-force that exact list through ordinary checked single-token forwards. Full-vocabulary logits and active state are exported after prefill, one continuation token and eight continuation tokens. All logical routing observations and finite values, field/catalogue shape, integer state, hashes, controls and continuation identities are checked.

Exports limit each CPU tensor copy to 32 MiB and each arm's artifacts to 1 GiB. These are diagnostic ceilings, not extra model memory. Target-plus-3 GB reclaimable preflights, the monitored launcher, terminal lifetime sampling and per-command footprint checks remain active. All six capture processes completed inside the target; the maximum observed capture footprint was 13.466360288 GB. Numerical capture timings are deliberately ineligible for speed claims, and global paging is diagnostic for this functional work.

Before capture, two normal 1024-chunk CLI runs with the observer disabled matched the immutable Plan018 prompt/output IDs, text, completed work and effective controls exactly. These are functional predecessor checks, not new timing pairs. The original Plan018 raw evidence remains unchanged.

### Numerical outcome

The 512 arm supplies the empirical rechunk control; it is not independently qualified. The candidate must satisfy `candidate <= max(3 * control, 0.01)` for each floating state field. Relative state error is maximum absolute difference divided by reference maximum magnitude, with the existing denominator floor. Full-vocabulary logits use reference spread. Integer state is exact, and candidate greedy tokens must match the reference at the captured checkpoints.

| Prose field | Control relative error | Candidate relative error | Allowed |
|---|---:|---:|---:|
| prefill/ssm.4 | 0.334984% | 1.009814% | 1.004953% |
| step1/conv.13 | 0.558659% | 1.955307% | 1.675978% |
| step1/ssm.4 | 0.337132% | 1.025249% | 1.011396% |

All six full-vocabulary logit comparisons and their candidate/reference greedy-token checks pass. Code state comparisons pass. Canonical routing disagreement aggregates expert keep-set replacements across every layer within the phase; both prompt cases and both phases pass this aggregate criterion. Individual layer differences remain visible in the raw reports and must not be misrepresented as failures of the established aggregate gate.

The three state failures alone reject overall admission. Thresholds were not relaxed, captures were not replaced, and no new control was selected after seeing the outcome. Independent reductions over hash-verified raw tensors reproduce each discrepancy.

### Evaluation corrections and provenance

The first host evaluator incorrectly required fused RoPE, which the deployment selector does not enable on this machine. Actual capture controls exactly match the Plan018 deployment vector. V2 fixes that metadata assumption but applies a stricter per-layer routing criterion than the established aggregate rule. V3 restores the canonical aggregation while retaining every per-layer observation as a nongating diagnostic. All existing individual measurement values remain identical; only metadata validation and route classification change. Original evaluators and failed reports are preserved alongside V3. No additional model run was needed for either correction.

V3 retains 432 individual measurements per prompt and records aggregate routing separately. The source build and all 55 native T0/T1 groups pass, with zero failures or skips. The final comparator's 26 synthetic corruption/boundary tests pass; objective task-scoring tests were also verified without inference. Serializer checks cover exact integer storage, finite values, copy/quota bounds, unsafe paths, overwrite refusal and unsupported controls.

### Follow-up disposition

The corrected task suite and short-prompt/long-completion timing driver were frozen before inference. Their model runs and conditional extended timing campaign were **not run after numerical rejection**, following Plan019's stop rule. Prepared fixtures are not quality or latency evidence.

Keep the ordinary prefill default at 256. Retain the earlier larger-chunk timing results within their exact scope, but do not treat their faster replies as numerical qualification. The 512 fallback needs a separately specified independent numerical check and task/latency acceptance; its role as the control here does not admit it. Parent001 and Plan011 qualification remain separate and incomplete.

Evidence: [[sources/runs/2026/09/jang-prefill-numerics-20260916]]. Decision: [[records/decisions/jang-prefill-1024-not-admitted-2026-09-16]].
