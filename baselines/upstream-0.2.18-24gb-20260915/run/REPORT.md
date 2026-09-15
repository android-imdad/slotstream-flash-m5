# Upstream Slotstream benchmark — 15 September 2026

**Complete: the 105.3 GB checkpoint passed SHA256 verification for all 25 files, and nine measured requests completed successfully.**

## Results

Apple M5 Max with 48 GiB unified memory. Clean upstream Slotstream 0.2.18, commit `ad89ecc17bcbdd6a1ca8b80fbd09d903b6c65fd0`, standard Qwen3.8-Flash-Next MLX 4-bit checkpoint. Process target: 24 decimal GB. MTP speculative decoding and expert lookahead were enabled automatically.

| Workload | Median decode (tok/s) | Range (tok/s) | Median first visible text (s) | Median full request (s) |
|---|---:|---:|---:|---:|
| Explanation | 14.00 | 13.78–14.34 | 2.18 | 11.33 |
| Coding | 15.90 | 15.88–16.03 | 1.99 | 10.03 |
| Reasoning | 15.68 | 15.67–15.79 | 2.57 | 10.68 |

Across all nine requests, median decode speed was **15.68 tok/s**. Peak physical process memory was **19.52 GB**, below the 24 GB target. All requests were free of swap-in/out activity and memory-pressure cancellation; instrumented thermal state remained nominal with Low Power Mode off. All repeated output token sequences matched within each workload.

## Method and limits

- One fresh server, one distinct warmup excluded from the table, then three rotated rounds of explanation, coding, and reasoning prompts. Each measured response reached the 128-token output cap (1,152 measured output tokens total).
- Prompt lengths: explanation 53 tokens, coding 57, reasoning 80. Greedy decoding with seed 42. The output cap may truncate answers; this is a performance benchmark, not a quality evaluation.
- Context configured at 32,768 tokens; prefix cache disabled and zero prefix tokens reused. Prefill chunk 2,048; 4,718 expert-cache slots (about 98 per layer, 13.0 GB pool). Physical-footprint instrumentation enabled.
- The download and hash verification finished before inference. No forced operating-system cache purge; this is warm request performance after a warmup, not a cold-storage test. Normal desktop applications remained open.
- First-visible-text timing is measured at the local streaming client. Decode throughput uses the native output count and decode duration. Physical memory uses the process lifetime footprint peak, not RSS or MLX allocations alone.
- This clean upstream result is separate from the modified JANG_6S build. The JANG run used 20 GB, a different quantization and runtime, scalar widening, and no MTP; it is not a controlled optimization comparison.

## Completion

Completed at 13:24:38 Asia/Colombo. Benchmark server PID 26306 exited after the measurements. The pipeline exited, and the benchmark listener and Edge0 listener were confirmed absent. The one-time status heartbeat is being paused after this report.

## Evidence

- [Run summary](/Users/imdad/Documents/ChatGPT/slotstream/results/benchmark-20260915-132246/summary.json)
- [Manifest and exact command](/Users/imdad/Documents/ChatGPT/slotstream/results/benchmark-20260915-132246/manifest.json)
- [Raw measurements](/Users/imdad/Documents/ChatGPT/slotstream/results/benchmark-20260915-132246)
- [Build identity](/Users/imdad/Documents/ChatGPT/slotstream/results/benchmark-20260915-132246/build-identity.json)
- [Clean source status](/Users/imdad/Documents/ChatGPT/slotstream/results/benchmark-20260915-132246/source-status.txt)
- [Server log](/Users/imdad/Documents/ChatGPT/slotstream/results/benchmark-20260915-132246/server.log)
- [Cleanup](/Users/imdad/Documents/ChatGPT/slotstream/results/benchmark-20260915-132246/cleanup.json)
- [Full model verification](/Users/imdad/Documents/ChatGPT/slotstream/results/verification.log)
- [Pipeline completion](/Users/imdad/Documents/ChatGPT/slotstream/results/pipeline-completion.json)
- [Separate JANG report](/Users/imdad/Documents/ChatGPT/slotstream/results/jang-6s-20260915-114304/REPORT.md)
