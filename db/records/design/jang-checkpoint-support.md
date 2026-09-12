---
type: design
id: 01m2abprd1se5kr05ch8rmt890
created: 2026-09-12T08:28:23.328829+00:00
updated: 2026-09-12T09:01:11.539800+00:00
summary: Experimental pinned JANG checkpoint loading and expert streaming, original-value preservation, fixed memory planning and explicit qualification limits.
date: 2026-09-12
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

Verification is layered: existing regression checks, pure source-policy
contracts, real weight-row comparisons against native MLX, and the production
expert reader/cache with sparse metadata fixtures. These checks do not establish
full-model numerical parity, memory peaks, context quality or throughput. Those
require complete checkpoint downloads and reference inference on the target Mac.
The source feature is experimental until that qualification exists.

Implementation: Sources/Slotstream/CheckpointFormat.swift, Checkpoint.swift,
JANGModels.swift, JANGManifests.swift, JANGPlanning.swift, ExpertStore.swift,
AffineRow.swift and the model/CLI wiring. User instructions: docs/JANG.md.

The Swift WeightStore instance retains the selected JANG manifest across
status, verification and resumable download. Original static helpers keep
their previous semantics. Expert-read counters count source bytes separately
from expanded cache bytes.
