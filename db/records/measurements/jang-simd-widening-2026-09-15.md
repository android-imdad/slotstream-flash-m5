---
type: measurement
id: 01m2k34g9j0kj7dxm1hw4e639k
created: 2026-09-15T17:51:46.482558+00:00
updated: 2026-09-15T17:51:46.482558+00:00
summary: NEON CPU widening is implemented; standalone conversion improves while full-model TPS remains unmeasured
date: 2026-09-15
doc: measurements
level: '2'
machines: '[[records/machines/local-m5-max-48gb]]'
order: '1226'
runs: '[[sources/runs/2026/09/jang-simd-widening-20260915]]'
title: JANG SIMD widening component result
status: measured
---
The explicit `packed4-to6` path now uses an ARM NEON CPU backend. It expands sixteen two-byte groups at once with structure loads/stores, writes every output byte without a clearing pass or scratch storage, and retains an exact tail and portable fallback. Existing Swift validation and overlap fallback remain intact. Generic scalar remains the default; scales, biases, cache geometry, routing and quantization are unchanged.

A standalone component comparison used a 1,638,400-code projection, nine alternating paired rounds and 200 calls per arm. The previous Swift packed implementation and the new production widening code ran in the same process. Median conversion time was 0.190276 ms before and 0.019522 ms after: a 9.75x ratio of medians. Every pair retained exact bytes; the thermal endpoints were nominal and low-power mode was off. The JSON records all samples, including the slower first round. This is a CPU conversion measurement, not a reader-throughput or full-model TPS result.

The release build passed. Its compiled C object includes NEON structure loads and stores. The focused native widening checks and bounded original-checkpoint byte comparisons passed, including exhaustive vector input pairs, unaligned boundaries, exact tails, overlap fallback and realistic projections. No long qualification campaign was repeated.

Existing development TPS numbers refer to the previous packed backend. New-backend full-model throughput and formal qualification remain unmeasured; no new overall speed claim or default activation follows from the component result. Evidence: [[sources/runs/2026/09/jang-simd-widening-20260915]].
