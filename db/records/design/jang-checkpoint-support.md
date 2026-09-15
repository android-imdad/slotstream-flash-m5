---
type: design
id: 01m2abprd1se5kr05ch8rmt890
created: 2026-09-12T08:28:23.328829+00:00
updated: 2026-09-15T13:45:14.613677+00:00
summary: Experimental pinned JANG checkpoint loading and expert streaming, original-value preservation, fixed memory planning and explicit qualification limits.
date: 2026-09-15
doc: plan
level: '2'
order: '6'
title: JANG checkpoint support
---
This local implementation adds two explicitly pinned alternative encodings of
Qwen3.8-Flash-Next: JANG_4M and JANG_6S. It does not retarget the original
PipeNetwork default or change its existing qualification evidence.

The loader canonicalizes JANG tensor names and quantization overrides together.
JANG's already-adjusted normalization weights remain unchanged. The depthwise
PLE convolution is reshaped to the engine's layout. Dense and router projections
use their own affine settings; FP16 quantization metadata is never treated as
BF16. Expert-cache scale/bias values expand exactly to FP32 to avoid
whole-pool promotion on every QMM; the larger records are charged. Dense
projection outputs retain the original BF16 activation/cache contract.
N-gram shards retain their individual packed widths and output precision.

A global slot pool still represents all original experts and the original
router selects them. JANG_6S uses a common six-bit pool: lower-width integer
codes widen exactly on a miss without fitting a new scale or changing any
represented weight. The disk files remain untouched. This spends additional
cache capacity and CPU read preparation; it is not a performance claim.

CheckpointMemory derives resident payload from validated headers and prices
actual pool records. It adds a conservative runtime/staging allowance, context
and prefill charges, and a planning margin. Initial JANG plans are fixed and
bounded by current real memory observations. They do not use PipeNetwork's
throughput curves or elastic governor. Vision and MTP execution are explicitly
unqualified and disabled for JANG in this implementation.

Pinned raw downloads reuse the existing joined-worker range downloader with
original-file SHA256 verification, resumability, cancellation and a destination
lock. No JANG compressed package is enabled. The CLI accepts explicit model
aliases and directories; serving metadata identifies the actual quant and
rejects a request naming a different loaded checkpoint.

Verification now includes the complete pinned JANG_6S download, monitored full-model generation, bounded exact loading/logit/state comparisons, a natural development oracle cohort and paired full-model widening measurements on the local M5 Max. The original component checks also stand. These results do not reproduce the publisher's BF16/quantization reference, qualify JANG_4M full-model inference, or establish broad long-context quality.

Exact `packed4-to6` widening is an explicit local run/Engine option, with scalar as the default. No CLI serving activation was added. The neuron-block oracle failed fidelity at the completed partial settings; lower counts stopped at user direction. Its predictor, sparse runtime and optional ANE predictor are not selected. Cache windows, balanced scheduling, whole-expert layouts and the tested heuristic-prefetch policies failed their scoped admission gates. No native prefetch or new ANE execution followed.

Current qualification state: [[records/plan/jang-flash-qualification-status]]. Measured findings and limits: [[records/measurements/jang-exact-widening-2026-09-15]], [[records/measurements/jang-neuron-oracle-2026-09-15]], [[records/measurements/jang-loading-screens-2026-09-15]], [[records/measurements/jang-prefetch-cost-screen-2026-09-15]]. The larger paired study and parent final run-set remain pending; the source feature remains experimental.

Implementation: Sources/Slotstream/CheckpointFormat.swift, Checkpoint.swift,
JANGModels.swift, JANGManifests.swift, JANGPlanning.swift, ExpertStore.swift,
AffineRow.swift and the model/CLI wiring. User instructions: docs/JANG.md.

The Swift WeightStore instance retains the selected JANG manifest across
status, verification and resumable download. Original static helpers keep
their previous semantics. Expert-read counters count source bytes separately
from expanded cache bytes.
