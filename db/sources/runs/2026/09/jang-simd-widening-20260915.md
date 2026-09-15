---
type: run
id: 01m2k34fp4aw9s60qf3q9e9vbe
created: 2026-09-15T17:51:45.860376+00:00
updated: 2026-09-15T17:51:46.420064+00:00
summary: NEON widening implementation, bounded exactness checks and standalone component comparison
binary: 0124a9e69e0ba2a7583eba6b73b6288eda68ce33ce58f5fd44762b4bdf3050b8
captured_at: 2026-09-15
command: .build/flash/simd-widening-20260915/bench
discarded: 'false'
machines: '[[records/machines/local-m5-max-48gb]]'
title: JANG NEON widening implementation and component result
tool: production widening code and bounded diagnostics
---
Implemented the NEON CPU backend and compared the actual new `AffineCodes.widen` against the previous Swift packed implementation from `926bc10`, with baseline type identifiers renamed only. Both compiled with Swift `-O`; the C object used Clang `-O3`. The small standalone driver supplies helpers only for unused row-decoding symbols; it exercises widening alone. Each arm mutates an input byte on every call and consumes output, with full-byte equality checked outside timing. The JSON preserves every paired sample and nominal endpoint observations. This is not an SSD or full-model benchmark.

[Result fields and source/build hashes](jang-simd-widening-20260915.json), [standalone driver](jang-simd-benchmark-20260915.swift), [renamed prior implementation](jang-simd-baseline-renamed-20260915.swift). The driver used the production `Sources/Slotstream/AffineRow.swift` and `Sources/CAffine/affine.c` plus a local CAffine module map. Original commands, object code and reports remain under `.build/flash/simd-widening-20260915`.

The package release build succeeded. Its C object contains the intended NEON structure loads/stores. The focused native exact-widening group passed, including every two-byte value through the vector body, vector/tail alignment boundaries, sentinels and the real projection size. The bounded checkpoint component at `.build/flash/runs/simd-widening-source-component-20260915` passed original-source byte comparisons and its memory limit. The previous binary is preserved at `.build/flash/runs/pre-simd-widening-20260915`.

No full-model TPS campaign, default activation, serving integration, router, quantization or memory-policy change was performed.
