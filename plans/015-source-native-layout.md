# 015 — Source-native whole-expert layout screen

## Current disposition — September 15, 2026

**COMPLETE — source-native layout rejected.** The original-byte layout was essentially tied overall and missed admission; committed in 01d5b6b.

[Evidence/current findings](../docs/JANG-FINDINGS.md) · [Canonical status](../db/records/plan/jang-flash-qualification-status.md). The original execution brief and dated review notes below are retained as history; current work is governed by [the plan index](README.md). Do not restart completed or unselected steps from a historical instruction.

Status: COMPLETE — rejected; no useful total reader-time improvement. Continues the user's authorized exact-loading work after
the completed Plan 014 benchmark was reported and committed as `2053804`.

## Hypothesis and scope

The expanded whole-expert layout helped some single-expert reads but made the
larger batches slower while reading 8–35% more bytes in the sampled layers.
This separate experiment bundles each expert's original nine tensor pieces
without expansion on disk. It tests whether fewer read calls help when that
extra traffic is removed. This is a hypothesis, not a demonstrated cause.

Each I/O lane reads one complete original record into bounded scratch, converts
its pieces with the same scalar F16-to-F32 and exact packed4-to6 algorithms, and
writes only its assigned output row. All lanes join before MLX sees any output;
an error discards the complete output batch. No worker constructs an MLX graph.
The ordinary raw reader and rejected balanced scheduler are unchanged.

This is a diagnostic subset, using one v1 container per sampled source layout.
Its identity binds the source checkpoint, original layer and expert mapping,
and a source-native diagnostic namespace. It is not a production on-disk format,
and the normal full-model layout loader must refuse it. A sample is tied to its
owning ExpertStore and cannot be read by another store.

## Frozen screening rules

- Same layers, expert IDs, batch sizes, queue depth, alternating eight pairs and
  exact-output verification as Plan 014. Both representations read exactly the
  original byte count per call; header/control reads remain separate.
- Original region/sample payload total 107,520,000 bytes, below 256 MiB. Charge
  source scratch and expanded output simultaneously; 512 MB actual process
  footprint, 4 GB reclaimable preflight. Existing descriptor controls must pass.
- Time complete reader calls, including lossless conversion, copy, allocation,
  scheduling and MLX staging. Exclude construction, verification and output
  hashing. Preserve every case regardless of its result.
- Same admission gate: median paired total reader-time reduction at least 10%,
  no tested case with median regression over 5%, nominal thermal endpoints,
  unchanged paging counters. This equal-case screen is not a workload-weighted
  inference estimate and makes no cold-SSD claim.
- Verify duplicate/reordered IDs, injected failure and recovery, rejection by
  the full-layout loader, source identity, output hashes and terminal memory.

Only a passing screen advances to separately budgeted runtime integration and
a matched 24 GB engine benchmark. No full-model artifact, public run/serve
activation, default change or ANE execution is included in this screen. Report
this benchmark before starting a different optimization.


## Result — September 15, 2026

The complete nine-case / 72-pair / 144 timed-call screen was timing-eligible and
byte-exact. The median paired total reader-time reduction was -0.4407%: essentially
a tie in this bounded experiment, with the candidate point estimate slightly
slower. This is not evidence of a meaningful overall slowdown or speedup.
It failed the required 10% improvement, and two cases regressed more than 5%.

| Layer | Expert misses | Raw median ms | Source-native median ms | Paired time reduction |
|---|---:|---:|---:|---:|
| 0 | 1 | 0.878 | 0.887 | -1.28% |
| 0 | 4 | 1.702 | 1.660 | 2.52% |
| 0 | 10 | 3.236 | 3.356 | -1.97% |
| 5 | 1 | 0.697 | 0.774 | -8.92% |
| 5 | 4 | 1.556 | 1.645 | -6.98% |
| 5 | 10 | 3.175 | 3.314 | -4.60% |
| 22 | 1 | 0.683 | 0.658 | 4.70% |
| 22 | 4 | 1.516 | 1.527 | -0.55% |
| 22 | 10 | 3.340 | 3.019 | 8.66% |

The source-native payload is 107,520,000 bytes, exactly the original sampled
payload size. All nine output tensors matched at every case/pair; duplicate and
reordered IDs matched. Injected read failure returned no tensors, recovery
matched the original reader, and full-model loading rejected the sample. The
original source identity remained unchanged.

Peak physical footprint was 205,964,176 bytes, under 512 MB. Reclaimable memory
before launch was 24.84 GB. Thermal endpoints were nominal, low-power mode was
off, and paging counters were unchanged. No full-model allocation was performed.

Evidence: `.build/flash/runs/source-native-20260915/` contains the source-bound
native archive, protocol and host hashes, sample payloads, raw timing reports,
terminal-sampled launcher receipt and independently recomputed `verification.json`.
The report SHA-256 is `78fca40d8290584f8e8c2c94d710b49eff3fedf57332df629e7acc69be9aa857`.
The completed Plan 014 expanded-layout evidence stays separate and unchanged.

Validation: release build passed (`source-native-build-20260915.log`), 40 native
T0 checks passed (`source-native-t0-20260915.json`), 220 host tests passed
(`source-native-host-tests-20260915.log`), all artifact bytes/build identity were
revalidated, and `git diff --check` passed. Full static-gate results for the
preceding committed foundation remain separately recorded; no new release or
full-engine qualification is claimed from this diagnostic screen.

Disposition: keep the ordinary original-reader path; source-native loading stays
diagnostic-only. Neither storage layout qualified for a full-model artifact or
24 GB generation benchmark. No new inference acceleration or ANE execution was
added. This completes the two layout screens authorized after Plan 013.
