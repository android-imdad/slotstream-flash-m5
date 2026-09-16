---
type: decision
id: 01m2mepc33e3hpq8wcjgw7gmzh
created: 2026-09-16T06:33:00.771701+00:00
updated: 2026-09-16T06:33:00.771701+00:00
summary: Do not adopt the 512 prefill fallback
decided_on: 2026-09-16
evidence: '[[records/measurements/jang-prefill-512-qualification-2026-09-16]]'
reversible_if: A separately declared investigation resolves numerical acceptance and passes representative task, latency and memory checks.
title: Do not adopt the 512 prefill fallback
status: standing
---
Retain the ordinary 256-token prefill default. The independent prospectively frozen 256/384/512 investigation rejects the 512 candidate on retained-state and aggregate-prefill-routing checks for both fresh prompts. The 384 midpoint is only an empirical control, not an admitted candidate. Earlier scoped prefill timings and explicit packed-widening/NEON decode results remain valid within their original limits. Further chunk work requires a separately declared investigation that explains the observed divergence and completes numerical, task, latency and memory acceptance; do not substitute controls or relax thresholds after results.
