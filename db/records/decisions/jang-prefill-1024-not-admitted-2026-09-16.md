---
type: decision
id: 01m2md1cgkpbrdnr5pxakanjav
created: 2026-09-16T06:04:04.499494+00:00
updated: 2026-09-16T06:04:04.499494+00:00
summary: Do not adopt the larger prefill candidate from timing alone
decided_on: 2026-09-16
evidence: '[[records/measurements/jang-prefill-qualification-2026-09-16]]'
reversible_if: A separately declared matched numerical investigation and representative task/latency checks pass for the candidate.
title: Do not adopt the larger prefill candidate from timing alone
status: standing
---
Retain the ordinary 256-token prefill default. The matched deployed JANG_6S investigation rejects the 1024 candidate because three retained-state comparisons exceed the predeclared rechunk band. Faster Plan018 request timings remain scoped measurements, not default admission. The 512 arm is an empirical control and needs independent qualification. Do not run the deferred long timing campaign merely to find a speed win after numerical rejection. Revisit only with a separately declared investigation that resolves numerical acceptance and completes task/latency checks without relaxing thresholds retrospectively.
