> Current candidate status: the matched packed-widening development benchmark is complete and did not beat this separate upstream target. See [JANG Flash findings](docs/JANG-FINDINGS.md) and [current qualification status](db/records/plan/jang-flash-qualification-status.md). The frozen baseline files retain their original bytes and historical next-step wording.

# Active JANG_6S performance target

The user selected the verified upstream Slotstream 0.2.18 benchmark as the baseline to beat with JANG_6S.

| Workload | Median decode to exceed | Baseline first visible text |
|---|---:|---:|
| Explanation | 14.00 tok/s | 2.18 s |
| Coding | 15.90 tok/s | 1.99 s |
| Reasoning | 15.68 tok/s | 2.57 s |

Use the same **24 GB total process memory target**, exact prompts, greedy decoding with seed 42, 128 output tokens, and 32,768-token context. Preserve the original JANG_6S checkpoint quality. Track latency and physical memory alongside throughput; exclude swapping or thermally throttled timing intervals.

The baseline's measured physical peak was 19.52 GB; the comparison budget remains 24 GB. Its overall median was 15.68 tok/s. The original upstream run used MTP and expert lookahead.

The full baseline archive is saved locally:

- [Baseline and comparison rules](/Users/imdad/Documents/projects/slotstream-flash-m5/baselines/upstream-0.2.18-24gb-20260915/README.md)
- [Exact requests and target numbers](/Users/imdad/Documents/projects/slotstream-flash-m5/baselines/upstream-0.2.18-24gb-20260915/baseline.json)
- [Evidence hashes](/Users/imdad/Documents/projects/slotstream-flash-m5/baselines/upstream-0.2.18-24gb-20260915/SHA256SUMS.json)

The archive includes the executable, Metal library, harness, raw results, and build receipt. Use it as the performance target for future JANG_6S work. Earlier JANG scalar and packed-widening screens are historical references, not replacements for this baseline.
