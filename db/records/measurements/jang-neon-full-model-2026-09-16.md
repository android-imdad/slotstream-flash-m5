---
type: measurement
id: 01m2m6gyksc1prankg9ke432p5
created: 2026-09-16T04:10:14.521560+00:00
updated: 2026-09-16T04:10:14.521560+00:00
summary: JANG NEON full-model development comparison
date: 2026-09-16
doc: measurements
level: '2'
machines: '[[records/machines/local-m5-max-48gb]]'
order: '1227'
runs: '[[sources/runs/2026/09/jang-neon-full-model-20260916]]'
title: JANG NEON full-model development comparison
status: measured
---
A short full-model comparison isolates the new ARM NEON packed widening backend against the preceding Swift packed implementation. Both arms use explicit `packed4-to6`; this is not a scalar-versus-packed comparison. The old archived binary matches the source map at `926bc10`, and the candidate matches `7670f8e`. Both build/source archives and binary hashes were validated before execution.

Protocol: JANG_6S on the local M5 Max with 48 GB unified memory; 14 GB process target; 2048 configured context tokens; greedy seed 7; at most 128 output tokens; MTP and vision off. Two predeclared interleaved rounds cover each of the three existing short prompts, reversing arm order on the second round. Each arm runs in a fresh process after thirty seconds of nominal thermal observations and the existing two-second settling interval. There is no cache purge; filesystem caching and ordinary desktop activity remain part of this local scope.

| Workload | Previous packed tok/s | NEON packed tok/s | Throughput change | Paired decode-time reduction | First token seconds, previous to NEON |
|---|---:|---:|---:|---:|---:|
| multiply-37-by-42 | 5.372 | 5.525 | +2.85% | 2.76% | 1.727 to 1.708 |
| python-deduplicate | 4.974 | 5.290 | +6.37% | 5.98% | 1.822 to 1.779 |
| sky-blue | 5.294 | 5.645 | +6.63% | 6.23% | 2.849 to 2.689 |

Throughput and first-token values are medians of the two runs per arm/workload. Throughput change is the ratio of those medians; decode-time reduction is the median of paired time reductions. Every pair had identical prompt/output IDs and completed work counts, and matching runtime controls. All twelve arms passed functional completion, terminal memory sampling and timing eligibility: nominal generator thermal endpoints, low-power mode off, and no swap-counter changes in launcher or generator intervals.

Maximum observed lifetime physical footprint across both arms was 10.056 GB, below the fixed target. Endpoint observations do not prove continuously idle hardware or continuous nominal temperature.

This is exploratory development evidence from two pairs per workload, not a statistically established speedup or formal qualification. The prior 24 GB study used a different budget/context/seed; these results cannot update its table or establish the NEON gain there. Full logits/state parity, broad context coverage, the whole qualification cohort and default/serving activation remain separate. No inference source, memory policy or default changed in this benchmark.

Evidence: [[sources/runs/2026/09/jang-neon-full-model-20260916]].
