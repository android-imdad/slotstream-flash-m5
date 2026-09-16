# JANG checkpoints (experimental local build)

This checkout adds text inference for the pinned `JANG_4M` and `JANG_6S`
Qwen3.8-Flash-Next checkpoints. It streams routed experts and n-gram rows
from the original safetensors files, retaining the model's native router.
The original PipeNetwork checkpoint remains the default.

The tested model is [JANGQ-AI/Qwen3.8-Flash-Next-JANG_6S](https://huggingface.co/JANGQ-AI/Qwen3.8-Flash-Next-JANG_6S),
using [the pinned experiment revision](https://huggingface.co/JANGQ-AI/Qwen3.8-Flash-Next-JANG_6S/tree/3781190c6bbdf0a7637beda49ba179822612058a).

This is an experimental local source build, not an upstream release. The complete JANG_6S checkpoint is now verified and has full-model generation, memory and paired widening measurements. A local JANG_6S benchmark at a 24 GB target measured 6.98–7.41 tok/s with explicit packed widening. Formal qualification, broader/long-context coverage and JANG_4M full-model results remain incomplete. See [current findings and rejected experiments](JANG-FINDINGS.md).

## Build and check

From this checkout, with Apple's command line tools installed:

```sh
make build SLOTSTREAM_BUILD_JOBS=2
.build/release/slotstream jang-check
```

The check covers checkpoint configuration, quantization, memory accounting,
and small original weight samples against native MLX dequantization. It does
not download or load the full models. The repository must remain available
for the diagnostic's fixtures.

## Reproduce the bounded streaming check

This creates sparse files with real checkpoint headers and synthetic expert
payloads; it does not download the full model. Use a fresh output directory:

```sh
python3 Tools/jang_fixture.py --tier 4M --output /tmp/jang-4m-fixture
.build/release/slotstream jang-check --model /tmp/jang-4m-fixture
```

Use `--tier 6S` and another fresh directory for the other layout. These files
are **not model weights** and must never be used for generation. The test
exercises production reads, expansion, insertion, hits and resize against
independent native MLX calculations.

## Download a checkpoint

Choose a separate destination for each checkpoint. Downloads use exact
repository revisions, resumable raw ranges and SHA256 verification. JANG
has no Slotpack package; `--transport compressed` is rejected.

```sh
.build/release/slotstream pull JANG_4M --dir ./models/jang-4m
.build/release/slotstream pull JANG_6S --dir ./models/jang-6s
```

These are separate, large downloads. Either command can be interrupted and
resumed with the same command. To verify an existing download:

```sh
.build/release/slotstream pull JANG_4M --dir ./models/jang-4m --verify
```

The pinned sources and per-file digests are in
[`JANGManifests.swift`](../Sources/Slotstream/JANGManifests.swift).
A pull without `--dir` uses Slotstream's user model directory; the aliases
`JANG_4M`, `JANG_6S`, `qwen3.8-flash-next:jang-4m` and
`qwen3.8-flash-next:jang-6s` resolve there.

## Preview memory, then generate

Run one model process at a time and retain headroom for other applications.
`doctor` reads headers without loading the weights into the GPU:

```sh
.build/release/slotstream doctor --model ./models/jang-6s
.build/release/slotstream run --model ./models/jang-6s \
  --prompt "Explain why the sky is blue." --max-tokens 128 --greedy \
  --sample-footprint --stats-json ./jang-6s-run.json
```

Substitute `./models/jang-4m` for the component-supported alternative; its full-model performance is not established by the JANG_6S results. The default JANG plan
selects a conservative target from current device observations and then
keeps that cache fixed. `--memory-gb` sets a total process target; requests
that cannot fit live headroom are refused. The plan charges original resident
weights, the actual cache representation, context, prefill and runtime margin.
Its budget is not a measured physical peak. No throughput estimate is shown.

For an API server:

```sh
.build/release/slotstream serve --model ./models/jang-6s
```

Clients must request the loaded JANG name reported by the API. A request for
the original PipeNetwork quant is rejected when JANG is loaded.

## Numerical diagnostics

The source build also includes a matched numerical capture diagnostic. Its
serializer and control checks run with:

```sh
.build/release/slotstream prefill-capture --self-check
```

Full `prefill-capture` runs load the model and export logits, retained state and
common-continuation evidence through the actual Engine path. Use the fixed
budgets, monitored launcher and frozen protocol in
[the qualification record](../db/records/measurements/jang-prefill-qualification-2026-09-16.md).
That investigation did not admit the larger prefill candidate; existing defaults
remain. The [independent fallback investigation](../db/records/measurements/jang-prefill-512-qualification-2026-09-16.md)
also failed its state and routing gates. The diagnostic's presence does not
establish broader model qualification.

## Explicit exact widening

Scalar expansion remains the default. The local `run` command accepts
`--expert-widening packed4-to6` for JANG_6S only. It accelerates integer-code
expansion without changing quantization or original files. The effective policy
is recorded as `effective_expert_widening` in `--stats-json` output.

After previewing memory and checking headroom, add the flag to a JANG_6S run:

```sh
.build/release/slotstream run --model ./models/jang-6s \
  --expert-widening packed4-to6 --prompt "Explain why the sky is blue." \
  --greedy --sample-footprint --stats-json ./jang-6s-packed.json
```

This example is a functional run, not the frozen benchmark workload. The flag
is rejected for unsupported checkpoints and when combined with a packed expert
layout. The CLI `serve` command has no widening flag and keeps scalar behavior;
HTTP requests cannot select it. Neuron masks, balanced scheduling, layout
samples and prefetch-cost probes are diagnostic-only and do not activate with
this flag. No Neural Engine acceleration is added.

## Representation and boundaries

- Tensor and quantization-override names are mapped together to the existing
  text graph. Already-adjusted normalization weights are retained unchanged.
- Dense projections and the router use their per-module affine settings.
  Original FP16 scale and bias values are preserved exactly. The expert cache
  stores those values as FP32 to avoid repeated full-pool promotion. Dense
  projection outputs retain the established BF16 activation/cache contract.
- Each n-gram shard is decoded using its own packed width, including mixed
  widths within a checkpoint. Compact cache entries retain the original
  output precision.
- The shared expert pool for JANG_6S uses six-bit codes. On a miss, a four-bit
  projection's integer codes are widened exactly; its scales, biases and
  represented values do not change. This costs additional cache capacity and
  CPU work. The source checkpoint is never requantized or overwritten.
- Initial JANG support is text-only. Vision and speculative MTP execution are
  disabled and explicit requests to enable them are rejected. Their weights
  may be present in the downloaded shards but are not materialized.
- JANG uses fixed cache plans. The original checkpoint's elastic governor,
  context qualification and empirical speed curves are not evidence for JANG.

## Qualification still needed

Complete JANG_6S weights, real generation and bounded comparisons against the preserved unmodified JANG_6S engine now exist. The measured widening checkpoint does not complete the original larger paired study or parent final run-set. Broad/long-context task coverage, publisher/BF16 reference reproduction and JANG_4M full-model qualification remain separate gaps. A configured context window is not a test of a prompt reaching that limit. See [the current qualification plan](../db/records/plan/jang-flash-qualification-status.md).

## Swift library

Use the selected instance for weight status, download and verification:

```swift
let weights = WeightStore.resolving("JANG_6S")
try weights.download()
let index = try CheckpointIndex(dir: weights.modelDirectory)
let plan = try CheckpointMemory(index: index).plan(memoryGB: nil)
let engine = try await Engine(modelDir: weights.modelDirectory, plan: plan)
```

For a custom directory use `WeightStore(modelDirectory: url,
jangModel: JANGModels.jang6S)`. The original static WeightStore helpers retain
their original PipeNetwork semantics. Engine read-byte statistics count source
expert payload, not bytes added by cache expansion; physical SSD traffic still
requires operating-system measurement.

For explicit JANG_6S widening in Swift, pass `expertWidening: .packed4To6` to
the Engine initializer and inspect `engine.effectiveWideningPolicy`. Omission
keeps `.scalar`. This is a local source API; the same checkpoint/layout refusal
rules apply. It does not qualify a custom serving integration.
