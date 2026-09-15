---
type: measurement
id: 01m2jmx3hdasap1c6ztpb2820a
created: 2026-09-15T13:43:03.981295+00:00
updated: 2026-09-15T13:43:03.981295+00:00
summary: JANG Flash foundation and existing M5 dispatch
date: 2026-09-15
doc: measurements
level: '2'
machines: '[[records/machines/local-m5-max-48gb]]'
order: '1215'
runs: '[[sources/runs/2026/09/jang-flash-findings-20260915]]'
title: JANG Flash foundation and existing M5 dispatch
status: measured
---
The complete pinned JANG_6S checkpoint was downloaded and verified, and monitored full-model generation completed. This advances the earlier component-only checkpoint; it does not establish the publisher's quantization loss against BF16 or qualify JANG_4M full-model inference.

Frozen natural tokenization, bounded activation/full-logit/state capture, terminal memory sampling and full-vocabulary quality metrics are implemented. Exact loading comparisons use the preserved unmodified JANG_6S engine, not another quantization. The reference/cohort archives and input identities remain immutable.

An instrumented copy of pinned MLX observed `gather_qmm_rhs_nax` for the grouped expert case and `gather_qmv` for decode, followed by successful GPU evaluation and numerical checks. This establishes the existing diagnostic-build path. It does not establish production-binary utilization or a new acceleration gain. GPU Neural Accelerators and the separate Apple Neural Engine are different devices; no new ANE execution was implemented.

Evidence: [[sources/runs/2026/09/jang-flash-findings-20260915]]. The source summary and private-dispatch report hashes are included in the portable extract. Plans002–008 and010/012 retain the individual foundation acceptance methods.
