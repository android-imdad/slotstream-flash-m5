---
type: run
id: 01m2m6gyjzze2znsctws2gm2nm
created: 2026-09-16T04:10:14.494842+00:00
updated: 2026-09-16T04:10:35.466937+00:00
summary: JANG NEON versus previous packed full-model benchmark
binary: 0124a9e69e0ba2a7583eba6b73b6288eda68ce33ce58f5fd44762b4bdf3050b8
captured_at: 2026-09-16
command: .build/flash/eval-env/bin/python .build/flash/neon-benchmark-20260916/run.py
discarded: 'false'
machines: '[[records/machines/local-m5-max-48gb]]'
title: JANG NEON versus previous packed full-model benchmark
tool: frozen driver using Tools/flash/widen_study.py
---
Captured native stats, launcher receipts, memory peaks, command arguments, binary and source identities, operating conditions and every completed pair in [the raw JSON](jang-neon-full-model-20260916.json). The [frozen driver](jang-neon-full-model-driver-20260916.py) reuses the existing monitored launch, readiness, exact-work, runtime-control and timing-eligibility helpers. Native stdout/stderr, sampled memory traces and archived candidate remain under `.build/flash/runs/neon-full-model-20260916`.

Two pairs per workload were predeclared before launch. No pairs were replaced or pooled with an earlier cohort. Both arms use `packed4-to6`; the baseline is the saved pre-NEON binary, not generic scalar. This is a bounded full-model benchmark, not the ten-pair formal qualification.
