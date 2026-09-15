---
type: measurement
id: 01m2aemp6k3r2rh1ffrnjmd9s2
created: 2026-09-12T09:19:41.267528+00:00
updated: 2026-09-15T13:45:14.567775+00:00
summary: JANG component, source-policy and transport validation passed; complete-model inference, KL, memory and throughput remain unqualified.
date: 2026-09-12
doc: measurements
level: '2'
machines: '[[records/machines/local-m5-max-48gb]]'
note: Historical component-only checkpoint before the complete JANG_6S experiments; later evidence is recorded separately in the September 15 Flash measurements.
order: '1210'
runs: '[[sources/runs/2026/09/jang-support-validation]]'
title: JANG local source validation
status: measured
---
Local implementation based on upstream `f60e9a83037fea576bf210a3650d6f29e164f183`.
The accepted binary is `0014844a367f513aa3dac20906f592e1af1b77df4ae8dfa3c48e2c18370c067b`. All 156
compiled-source inputs match its build receipt.

The native catalogue completed with 46 groups, no failures or skips, and
27,484 assertions. Both JANG fixture layouts passed production expert reads,
contiguous reads, source-byte accounting, insertion, cache hits, growth and
shrink/refill. Original packed rows matched the native MLX decoder; exact
code/metadata expansion preserved represented values. Projection checks keep
the BF16 activation/cache contract explicit.

The independent source-policy executable passed 964,167 assertions and marked
hardware qualification false. The complete static suite passed, including
transport fault injection, memory guards, installer checks, fixture hashes,
projections and brain validation. The brain retains its two existing log
warnings and has no validation errors.

Reproduction and raw evidence: [[sources/runs/2026/09/jang-support-validation]].
The bounded fixture generator and original metadata/row samples are included
in Tools/jang_fixture.py and Tools/fixtures/jang/.

Qualification boundary: complete model weights were not downloaded; no
full-model generation, long-context answer test, KL reproduction, or actual
inference memory/throughput measurement was performed. JANG MTP and vision
remain disabled. This is experimental local source support, not an upstream
release or proof of end-to-end inference on this Mac.
