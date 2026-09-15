---
type: measurement
id: 01m2jmx3krf0xgh7rf2z0wxc9g
created: 2026-09-15T13:43:04.056104+00:00
updated: 2026-09-15T13:43:04.056104+00:00
summary: 'JANG causal SSD-prefetch: empirical cost and overlap rejection'
date: 2026-09-15
doc: measurements
level: '2'
machines: '[[records/machines/local-m5-max-48gb]]'
order: '1250'
runs: '[[sources/runs/2026/09/jang-flash-findings-20260915]]'
title: 'JANG causal SSD-prefetch: empirical cost and overlap rejection'
status: analysis
---
The existing previous-token and recent-eight-frequency forecasts were scored using exact historical routes, a complete original-reader cost grid and nine fresh matching packed-widening control runs. The historical scalar timings and a paging-ineligible old packed timing were not used. Every fresh control matched exact IDs, work, plan/numerical controls and native CLOCK read counts/bytes.

Prefetch admission was evaluated at 14 GB with the original diagnostic workloads. Reader cost calibration covered all batch sizes from one through ten in three source precision classes, with eight repetitions per case. The largest calibration peak was 373,031,920 bytes under a 512 MB cap; maximum full-model control peak was 10,063,893,856 bytes under target. Timings were eligible and original source identities were unchanged.

The model charges staging against the cache, grants free predictions/insertion, and permits all non-reader decode time as one global overlap budget. Observed cost extrema provide an optimistic sensitivity, not a physical bound or confidence interval.

| Forecast | Modeled ideal-overlap estimate | Optimistic sample sensitivity | Result |
|---|---:|---:|---|
| Previous-token | -0.64% to +0.34% | At most +0.51% | Rejected |
| Recent-frequency | 4.62–4.77% | At most 6.47% | Rejected |

Both missed the predeclared ten-percent gate in every tested workload even under the optimistic sensitivity. These are empirical estimates, not measured native prefetch speedups. No worker or serving activation was implemented. The result is scoped to these forecasts and this budget; it does not establish a universal rejection at 24 GB or for learned predictors.

Evidence: [[sources/runs/2026/09/jang-flash-findings-20260915]]. Original report SHA-256 `49cb662be69ad0fa866ff9c3078b7ef9aca33d332afa12ac95570b684cfb8519`; independent verification binds all control, calibration, readiness and settling artifacts. Plan016 retains the complete equations, per-workload table and limitations.
