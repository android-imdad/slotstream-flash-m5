# 013 — Balance exact expert read jobs

Status: COMPLETE — rejected by the component performance screen. User authorized continuing optimizations on September 15,
with a benchmark after each stage before another optimization.

## Decision and scope

The eight/six-block oracle failed fidelity; lower counts remain stopped by the
user. The dependent neuron predictor and optional ANE predictor are not selected.
This does not claim ANE acceleration. Previous-token and recent-frequency SSD
prefetch remain replay-only, with no timing admission. This stage investigates exact
demand loading independently, without speculation or additional checkpoint reads.

`ExpertStore.readBatchChecked` currently assigns nine pieces per expert to lanes
by stride. Weight and metadata sizes differ considerably. The experimental
schedule sorts by source plus destination byte count and greedily assigns jobs
to the least-loaded lane. The cost is a deterministic balancing heuristic, not a
prediction of SSD latency. Each original job executes once into the same buffer;
joined failure handling remains in place. The sweep's run reader is unchanged.

Initial implementation is package-only and disabled. No run/serve activation
before the component screen. Ownership: `ExpertReadSchedule.swift`, narrowly
scoped `ExpertStore.swift`, `Diagnostics+ReadScheduling.swift`, CLI diagnostic
registration, tests and this execution record. Preserve all earlier local work.

## Predeclared component screen

- Both arms use exact packed4-to6 widening, original JANG_6S files and identical
  staging/queue depths. Alternating eight pairs at layers 0, 5 and 22, with ten
  fixed noncontiguous experts and queue depths 12 and 32.
- At most 256 MiB unique original source regions, 512 MB process footprint and
  4 GB reclaimable preflight. Original files opened read-only; record successful
  F_NOCACHE/F_RDAHEAD calls without claiming a cold SSD.
- Hash every byte of all nine output tensors outside timing. Whole checked
  reader call is timed, including allocation, reads, expansion and MLX wrapping.
- Admit a full-engine benchmark only if default queue depth 32 has at least 10%
  median paired reader-time reduction pooled across cases, with no layer's
  median regression greater than 5%. Depth 12 is a sensitivity diagnostic.
- Otherwise record rejection and leave the option unavailable in generation.

If admitted, expose explicit default-off local run/capture/library selection,
verify exact logits/routes/state and run the saved three-workload 24 GB benchmark
with three interleaved pairs per workload. Both arms must use packed widening;
the immutable completed September 15 packed engine is the before arm. Apply the
existing memory, nominal thermal, paging eligibility and exact-token gates.
This remains a development benchmark, not the ten-pair final qualification.

No next optimization starts until this result is reported.


## Result — September 15, 2026

The complete six-case / 48-pair / 96-reader-call screen passed byte equality and
memory checks but rejected balanced scheduling. At the default depth of 32,
pooled median paired reader-time change was **40.39% longer**. All three layer
classes regressed. No local run/capture/library option was added; the package
switch remains false except inside the diagnostic command.

| Layer | Queue depth | Strided median | Balanced median | Median paired time reduction |
|---|---:|---:|---:|---:|
| 0 | 32 | 3.180 ms | 4.464 ms | -40.39% |
| 5 | 32 | 3.154 ms | 4.426 ms | -40.85% |
| 22 | 32 | 3.325 ms | 4.717 ms | -39.92% |
| 0 | 12 | 4.299 ms | 3.845 ms | +11.67% |
| 5 | 12 | 3.836 ms | 3.733 ms | +4.13% |
| 22 | 12 | 3.741 ms | 3.997 ms | -5.69% |

The mixed depth-12 observations do not qualify a queue-depth change. No new
optimization was started after this result. Byte-weighted lane balance did not
translate into a faster checked reader on this machine; the experiment does not
isolate the cause among scheduling overhead, I/O overlap and expansion work.

Evidence: `.build/flash/runs/read-scheduling-20260915/`. The archive binds the
native binary and original source; `protocol.json` pins the host harness and
thresholds. `launch/native/report.json` contains individual timings, output
hashes, unchanged model source identity and successful descriptor-control calls.
`launch/receipt.json` binds actual artifact bytes and samples the terminal peak
before reaping. `report.json` and its completion marker record the rejection.

- Original source regions: 107,520,000 bytes; the same regions were repeatedly
  read for paired timing (3,440,640,000 logical source bytes across 96 calls).
- Maximum measured physical footprint: 108,020,432 bytes, under 512 MB.
- Reclaimable memory before launch: 31.36 GB. Endpoint thermal state nominal,
  low-power mode off, swap-in/out counters unchanged across the launch.
- Both modes matched complete tensor-byte hashes at every case/pair. This is
  original-reader component equality, not full-model logits/state qualification.
- Release build passed; 40 native T0 checks passed, including 63 scheduler
  assertions; 215 host tests passed. Full repository static gates passed (`STATIC GATES PASS`), recorded in
  `.build/flash/read-scheduling-static-20260915.log`.

A full-engine performance run was not admitted. The preceding 24 GB packed
widening result (about 6.98–7.41 tok/s) remains the last completed engine benchmark;
this experiment establishes no additional generation speedup. All changes are
local and uncommitted, with no merge, push, serving activation or default rollout.
