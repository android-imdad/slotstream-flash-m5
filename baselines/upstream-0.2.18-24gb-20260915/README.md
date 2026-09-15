# Baseline to beat: upstream Slotstream 0.2.18

JANG_6S is the candidate. This baseline is frozen from the completed upstream run on 15 September 2026; the archived executable, Metal library, harness, requests, raw responses, and build identity are preserved together.

## Targets at 24 GB total process memory

| Workload | Median decode to exceed | Baseline first visible text |
|---|---:|---:|
| Explanation | 14.00 tok/s | 2.18 s |
| Coding | 15.90 tok/s | 1.99 s |
| Reasoning | 15.68 tok/s | 2.57 s |

Overall median: **15.68 tok/s**. Observed upstream physical peak: **19.52 GB**. The comparison budget is 24 GB; the observed 19.52 GB is not a new memory cap.

## Comparison rules

- Use the exact frozen prompts, greedy decoding, seed 42, 128 output tokens, context 32768, and a 24 GB total process target on this Mac.
- Use a separate warmup and three rotated rounds. Record effective candidate settings; an existing 20 GB scalar JANG result is not this comparison.
- Primary target: exceed every workload median decode rate. Record first-visible-text latency and total request time alongside throughput; do not conceal regressions.
- No swap activity or thermal throttling in counted intervals. Keep one model process active, pause competing heavy I/O, and record physical footprint within 24 GB.
- Keep original JANG_6S checkpoint bytes, native routing and precision. Validate optimization correctness against the unchanged JANG reference, not against a different upstream quantization. Cross-quantization output IDs may differ.
- A three-round win is exploratory. Confirm a claimed optimization gain in a larger interleaved comparison with the frozen upstream binary, including timing spread and repeatability.

Upstream used MTP and expert lookahead. JANG must declare its effective acceleration settings. The older 5.12–5.56 tok/s packed-widening screen at 14 GB and today’s 3.03 tok/s default-mode run at 20 GB are historical references, not matched measurements of this target.

Next execution step: establish an explicit packed4-to6 JANG_6S run with these frozen workloads and memory target, then profile remaining time before making source changes. Existing serving currently uses scalar widening by default; verify the selected path rather than assuming the optimization is active.

## Files

- `baseline.json`: exact requests, target numbers, and comparison rules.
- `run/`: copied measurements and build receipt.
- `bin/`: the exact baseline executable and matching Metal library.
- `harness/benchmark.py`: benchmark harness snapshot. Use the manifest and contract to point it at the archived binary when rerunning.
- `SHA256SUMS.json`: byte-level inventory. Verify before comparing; create a new baseline directory if the baseline changes.
