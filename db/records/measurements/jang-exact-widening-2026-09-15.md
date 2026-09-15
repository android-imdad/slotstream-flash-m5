---
type: measurement
id: 01m2jmx3hzjwjmn2y9ebqeh39j
created: 2026-09-15T13:43:03.999719+00:00
updated: 2026-09-15T13:43:03.999719+00:00
summary: 'JANG exact widening: matched 24 GB development benchmark'
date: 2026-09-15
doc: measurements
level: '2'
machines: '[[records/machines/local-m5-max-48gb]]'
order: '1220'
runs: '[[sources/runs/2026/09/jang-flash-findings-20260915]]'
title: 'JANG exact widening: matched 24 GB development benchmark'
status: measured
---
Explicit `packed4-to6` integer-code widening preserves represented weights and original files. Scalar remains the default; the fast policy is available to the local run command and Engine API for JANG_6S. The CLI serve command has no widening activation flag.

Three paired rounds per workload ran on the local M5 Max with 48 GB unified memory. Both arms used the same JANG_6S checkpoint, a 24 GB process target, 32,768 configured context tokens, greedy seed 42, up to 128 output tokens, fresh CLI processes, prefix cache off, MTP off and vision off. Configured context is not a tested long-prompt length.

| Workload | Scalar tok/s | Packed tok/s | First text seconds, scalar → packed |
|---|---:|---:|---:|
| Coding | 3.331 | 7.409 | 5.311 → 2.584 |
| Explanation | 3.199 | 6.983 | 5.870 → 2.930 |
| Reasoning | 3.272 | 7.266 | 6.623 → 3.235 |

Packed throughput was 6.98–7.41 tok/s, about 2.2× scalar throughput. All nine pairs / eighteen arms had exact output IDs and work, matched controls, eligible timings and process peaks under target. Maximum observed packed physical footprint was 20.685329928 GB (20.69 GB rounded). First-text statistics exclude loading and cooldown.

This is a measured development checkpoint, not the original ten-pair qualification, a held-out broad task-suite result, default rollout or upstream release. The separate upstream target uses another quantization and persistent HTTP serving with MTP/lookahead; it is not the causal before arm and was not beaten.

Evidence: [[sources/runs/2026/09/jang-flash-findings-20260915]]. Original report SHA-256 `d623c0ff1af9c3f03712d2dbe52b85e0e0b6b9eec752ab3e494f75e249edf5f8`. The earlier fair-thermal attempt remains preserved as incomplete/ineligible and is not combined with the cooled cohort.
