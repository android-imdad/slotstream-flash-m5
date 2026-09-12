# JANG checkpoints (experimental local build)

This checkout adds text inference for the pinned `JANG_4M` and `JANG_6S`
Qwen3.8-Flash-Next checkpoints. It streams routed experts and n-gram rows
from the original safetensors files, retaining the model's native router.
The original PipeNetwork checkpoint remains the default.

This is a source implementation, not an upstream release or a qualified
full-model performance result. The component gates use real JANG weight-row
samples and checkpoint metadata. Full generation, long-context quality,
physical-memory peaks and throughput still require tests with complete weights.

## Build and check

From this checkout, with Apple's command line tools installed:

```sh
make build SLOTSTREAM_BUILD_JOBS=4
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
.build/release/slotstream doctor --model ./models/jang-4m
.build/release/slotstream run --model ./models/jang-4m \
  --prompt "Explain why the sky is blue." --max-tokens 128 --greedy \
  --sample-footprint --stats-json ./jang-4m-run.json
```

Substitute `./models/jang-6s` for the other checkpoint. The default JANG plan
selects a conservative target from current device observations and then
keeps that cache fixed. `--memory-gb` sets a total process target; requests
that cannot fit live headroom are refused. The plan charges original resident
weights, the actual cache representation, context, prefill and runtime margin.
Its budget is not a measured physical peak. No throughput estimate is shown.

For an API server:

```sh
.build/release/slotstream serve --model ./models/jang-4m
```

Clients must request the loaded JANG name reported by the API. A request for
the original PipeNetwork quant is rejected when JANG is loaded.

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

A complete weight download and an independent reference run are needed to
validate whole-model logits and useful generation on the target Mac. Then
measure cold and warm runs, prefill, sustained decode, physical footprint,
actual storage reads and long-context behavior. The component checks alone
do not establish those results or reproduce the publisher's KL scores.

## Swift library

Use the selected instance for weight status, download and verification:

```swift
let weights = WeightStore.resolving("JANG_4M")
try weights.download()
let index = try CheckpointIndex(dir: weights.modelDirectory)
let plan = try CheckpointMemory(index: index).plan(memoryGB: nil)
let engine = try await Engine(modelDir: weights.modelDirectory, plan: plan)
```

For a custom directory use `WeightStore(modelDirectory: url,
jangModel: JANGModels.jang4M)`. The original static WeightStore helpers retain
their original PipeNetwork semantics. Engine read-byte statistics count source
expert payload, not bytes added by cache expansion; physical SSD traffic still
requires operating-system measurement.
