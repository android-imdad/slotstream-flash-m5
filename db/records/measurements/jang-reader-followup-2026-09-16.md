---
type: measurement
id: 01m2m8v8s668e475009wa0v68n
created: 2026-09-16T04:50:49.766126+00:00
updated: 2026-09-16T04:50:49.766126+00:00
summary: 'JANG metadata and scratch follow-up: both runtime candidates rejected'
date: 2026-09-16
doc: measurements
level: '2'
machines: '[[records/machines/local-m5-max-48gb]]'
order: '1228'
runs: '[[sources/runs/2026/09/jang-reader-followup-20260916]]'
title: 'JANG metadata and scratch follow-up: both runtime candidates rejected'
status: measured
---
Neither proposed runtime optimization was admitted. The implementation experiments completed, but did not establish a useful additional throughput gain. The original NEON packed backend remains unchanged.

### Metadata conversion

The actual archived production `ExpertStore.readRows` disassembly already contains FCVTL/FCVTL2 vector conversion loops and scalar tails. A standalone extracted-loop comparison against explicit NEON retained all FP16-to-FP32 bit patterns exactly, including exceptional values. Its small absolute conversion differences did not justify a redundant runtime implementation. Both diagnostic sample sets and compiler/source receipts are retained separately; no full-model speedup is inferred from them. The benchmark-only candidate remains reproducible under Tools/flash.

### Lane-local CPU scratch

The candidate allocated one bounded uninitialized CPU scratch block per joined read lane, retained scalar allocation behavior, and left GPU staging ownership unchanged. Focused tests covered mixed-size reuse, invalid dimensions, partial EOF suppression and recovery. The original-reader screen included mixed layers, failure draining and exact recovery for batch and contiguous-run APIs.

| Reader | Layer | Scratch off ms | Scratch on ms | Ratio-of-median rate change |
|---|---:|---:|---:|---:|
| readBatchChecked | 0 | 2.8235 | 2.8385 | -0.53% |
| readBatchChecked | 5 | 2.9590 | 2.9700 | -0.37% |
| readBatchChecked | 22 | 3.1549 | 3.2014 | -1.45% |
| readRunsChecked | 0 | 3.5045 | 3.6462 | -3.89% |
| readRunsChecked | 5 | 3.3539 | 3.2045 | +4.66% |
| readRunsChecked | 22 | 1.7611 | 1.8027 | -2.31% |

Each case used four alternating pairs over the same original expert tensors and included allocation, reads, conversion and MLX wrapping. Five of six median reader times were worse; the one improvement did not establish broad component benefit. These small changes are exploratory and do not prove a universal slowdown. No full-model run was warranted by this screen.

All 26 native component assertions passed; the candidate focused check passed 36 assertions and the reviewer reran it successfully. Source identity remained unchanged. The bounded source regions totaled 107,520,000 bytes. Maximum sampled component footprint was 197,411,632 bytes at a 20 ms interval; this is not a terminal lifetime peak or full-model memory qualification.

After restoring the original runtime source, the preserved baseline passed all 54 T0/T1 groups, with zero failures or skips. The standalone benchmark tools and rejected candidate patch are retained; runtime source matches the pre-experiment baseline.

Next candidates are separate: bounded contiguous source reads for converted prefill runs, then native-bit cache representation research. Neither was implemented or measured in this pass.

Evidence: [[sources/runs/2026/09/jang-reader-followup-20260916]]. Decision: [[records/decisions/jang-reader-followup-rejected-2026-09-16]].
