---
type: measurement
id: 01m2jzper26f8c1kebqf6wwnzp
created: 2026-09-15T16:51:40.418693+00:00
updated: 2026-09-15T16:51:40.418693+00:00
summary: Current widening prerequisites pass; overall qualification remains thermally inconclusive
date: 2026-09-15
doc: measurements
level: '2'
machines: '[[records/machines/local-m5-max-48gb]]'
order: '1225'
runs: '[[sources/runs/2026/09/jang-widening-qualification-20260915]]'
title: JANG widening qualification attempts and CPU profile
status: measured
---
**Overall disposition: thermally inconclusive. Plan011 remains unqualified.** The current native binary passed 54 T0/T1 checks with no failure or skip. All 228 host tests passed after the shared cooldown change. Current synthetic/source widening checks and fresh full-logit, ordered-route, retained-state and ordinary-output parity passed. The fresh three-prompt exploratory screen was promising.

Both formal attempts used JANG_6S, a 14 GB process target, 2048 configured context tokens, greedy seed 7, up to 128 output tokens, MTP and vision off, and balanced scalar/packed pair order. The prescribed study requires ten pairs per prompt. The first attempt stopped after 39 launches when a scalar Python arm went from nominal to fair. The fresh cohort added a thirty-second window of nominal thermal observations before the original two-second settling interval; that cohort stopped after 44 launches when the second scalar arithmetic arm went from nominal to fair. Both arms completed functionally within the memory bound. Thermal timing exclusions remain enforced, and no pairs were replaced or pooled between attempts.

The cooled cohort contains 22 complete pairs, of which 21 are timing-eligible, against 30 required pairs. Sky-blue and Python each completed their ten eligible pairs. Arithmetic did not complete its cohort, so there is no aggregate arithmetic performance claim and no overall qualification.

### Completed workload subsets

These descriptive results are from complete subsets of an incomplete overall cohort. Throughput columns are medians of per-run tokens/second. Decode reduction is the median of paired `1 - packed_seconds / scalar_seconds`; its interval uses the fixed paired bootstrap with 10,000 resamples and seed 17. It is not calculated from a ratio of the throughput medians.

| Workload | Scalar tok/s | Packed tok/s | Paired decode-time reduction | Bootstrap 95% interval |
|---|---:|---:|---:|---:|
| Sky-blue | 2.557 | 5.443 | 52.83% | 52.11% to 53.77% |
| Python deduplication | 2.409 | 4.999 | 51.50% | 49.45% to 56.18% |

Both complete subsets met their prescribed per-workload performance thresholds, including the median first-token guardrail. This cannot satisfy the missing whole-cohort requirement. All completed paired output IDs, work counts and runtime controls were rechecked by the export. Endpoint observations do not establish continuously idle hardware or continuous nominal temperature.

### Stage-five profiling disposition

The planned nominal-only 256-/1024-token raw code-prefix timing profile never launched a model; its owned prelaunch waiters were stopped while nominal conditions remained unavailable. A separate five-second CPU stack sample at ten-millisecond intervals observed the packed Python workload under recorded fair conditions. Its output IDs and work counts matched the earlier uninstrumented packed control, and process-memory/terminal sampling passed. Instrumented timings are ineligible for speed claims.

The sample contains recurring `pread`, `AffineCodes.widen`, `ExpertStore.stagingArrays`, MLX command-submission and waiting stacks. Thread sample counts are not request wall-time fractions or GPU kernel durations. This does not establish GPU utilization or justify a custom M5 kernel. Clean long-prompt profiling and direct component cost measurements remain prerequisites to choosing that optimization. The previously rejected layouts and heuristic prefetch policies are not reopened.

Swift compiler and active simulator processes were observed separately after the cooled stop. Their contribution to the earlier thermal transition was not measured. They were left untouched. No model or profiling waiter remains from this session.

Evidence and immutable artifact hashes: [[sources/runs/2026/09/jang-widening-qualification-20260915]]. The earlier 24 GB development measurement remains a separate valid scoped result: [[records/measurements/jang-exact-widening-2026-09-15]]. Scalar and serving defaults are unchanged; parent final qualification, default activation and release remain pending.
