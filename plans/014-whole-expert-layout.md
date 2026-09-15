# 014 — Benchmark existing whole-expert storage

## Current disposition — September 15, 2026

**COMPLETE — expanded layout rejected.** Measured component result committed in 2053804; no full-model artifact or runtime activation admitted.

[Evidence/current findings](../docs/JANG-FINDINGS.md) · [Canonical status](../db/records/plan/jang-flash-qualification-status.md). The original execution brief and dated review notes below are retained as history; current work is governed by [the plan index](README.md). Do not restart completed or unselected steps from a historical instruction.

Status: COMPLETE — expanded layout rejected by the component screen. User authorized continuing after Plan 013 and committing
the completed checkpoint. Commits `65d7e82` and `ba18325` preserve that checkpoint;
large local executable archives remain ignored and their hashes are recorded.

## Scope and predeclared screen

Plan 001 step 2 requires comparing the existing expanded `PackedExpertLayout`
against original scattered tensor reads before inventing a new source-native
layout. This stage performs that comparison on a bounded sample. It reduces
nine original reads per expert to one whole-record read, with extra stored bytes
and deinterleave copying. The raw arm already uses exact packed4-to6 widening;
balanced read scheduling remains disabled in both arms.

- Three source precision classes: layers 0, 5 and 22. Ten fixed noncontiguous
  original expert IDs: 0, 7, 19, 43, 79, 131, 211, 307, 401, 511.
- Batch sizes 1, 4 and 10, eight alternating pairs per case, depth 32. Keep
  construction/verification and full output hashing outside timed reader calls.
- Build at most 256 MiB of derived expanded records. The sample identity binds
  the original checkpoint and subset mapping; normal model loading cannot
  accept it as a complete artifact. Original weights remain read-only.
- Time the complete reader: staging, scheduling, reads, conversion or copying,
  and MLX wrapping/evaluation. Verify all output bytes every pair, plus duplicate
  and reordered expert IDs outside timing.
- 512 MB physical footprint limit, 4 GB reclaimable preflight. Existing v1
  controls require successful F_NOCACHE/F_RDAHEAD calls, without proving cold SSD
  access. Nominal thermal endpoints and unchanged paging counters qualify timing.
- Advance only for at least 10% median paired **total reader time** reduction
  across the nine cases, with no case's median regression greater than 5%.
  Pair totals are an explicit equal-case screen, not an inference workload mix.

If admitted, separately price the full expanded artifact, scratch and startup
verification; qualify exact full-model behavior and benchmark the saved 24 GB
workloads against the current packed-widening engine. The component result alone
does not authorize default activation or establish generation throughput.

Implementation: bounded sample build/read helpers in `ExpertStore.swift`,
`Diagnostics+WholeExpert.swift`, `whole-expert-check` CLI,
`Tools/flash/whole_expert_study.py` and focused validation tests. Existing v1
format and public generation/serving settings are unchanged.

Benchmark this stage and report its disposition before the next optimization.


## Result — September 15, 2026

The nine-case / 72-pair screen was timing-eligible and byte-exact, but failed
admission: median paired total reader time was **8.48% longer**. The sample
contains 129,024,000 expanded bytes for 107,520,000 original source bytes.

| Layer | Expert misses | Raw median ms | Whole median ms | Paired time reduction |
|---|---:|---:|---:|---:|
| 0 | 1 | 0.922 | 0.624 | 32.19% |
| 0 | 4 | 1.714 | 1.663 | 3.28% |
| 0 | 10 | 3.203 | 3.777 | -18.43% |
| 5 | 1 | 0.769 | 0.641 | 14.84% |
| 5 | 4 | 1.540 | 1.675 | -9.45% |
| 5 | 10 | 3.163 | 3.804 | -20.47% |
| 22 | 1 | 0.628 | 0.619 | 2.58% |
| 22 | 4 | 1.512 | 1.667 | -10.26% |
| 22 | 10 | 3.332 | 3.708 | -11.71% |

Single-expert improvements do not qualify the complete layout. Ten-expert
batches regressed across all three precision classes. The result motivates a
separate source-native layout experiment; it does not prove the extra bytes
alone caused the regression. No full-model expanded repack or generation
activation was admitted.

Evidence: `.build/flash/runs/whole-expert-20260915/{report,verification}.json`,
with full native timing/output reports, sample artifact and terminal-sampled
launcher receipt under `launch/`. Peak physical footprint was 165,053,208 bytes
under 512 MB. Reclaimable memory before launch was 26.62 GB. Thermal endpoints
were nominal, low-power mode was off and paging counters did not change.

All output byte hashes matched, including duplicate/reordered expert IDs. An
injected read failure returned no partial tensors, subsequent reads recovered,
and the normal full-layout loader rejected the subset artifact. Original source
identity was unchanged. Final release build passed (`whole-expert-build-v3-20260915.log`),
40 native T0 checks passed and 219 host tests passed. The previous committed
checkpoint's full static gates remain recorded separately; this diagnostic-only
stage used its build, native, host, artifact and diff checks.
