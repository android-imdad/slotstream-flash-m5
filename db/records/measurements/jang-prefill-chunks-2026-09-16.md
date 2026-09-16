---
type: measurement
id: 01m2mb1ynxh1t9bbzw74c73yka
created: 2026-09-16T05:29:25.949239+00:00
updated: 2026-09-16T05:29:25.949239+00:00
summary: JANG prefill chunk screen at a fixed memory budget
date: 2026-09-16
doc: measurements
level: '2'
machines: '[[records/machines/local-m5-max-48gb]]'
order: '1229'
runs: '[[sources/runs/2026/09/jang-prefill-chunks-20260916]]'
title: JANG prefill chunk screen at a fixed memory budget
status: measured
---
A complete bounded screen compares the existing JANG_6S prefill controls without modifying runtime source or defaults. Both larger chunks pass the predeclared timing shortlist; this does not admit either as a new default. The 1024-token arm has the largest observed request-latency benefit, while its prose decode regression is marginal against the screening limit. Manual task review is not a clean quality pass, and matched numerical qualification remains pending.

### Protocol

Local M5 Max with 48 GB unified memory; pinned JANG_6S; the validated current NEON binary; explicit packed4-to6; fixed 14 GB process target; 4096 configured context; greedy seed 42; at most 128 output tokens; MTP, vision and prefix reuse disabled. Two frozen natural repository excerpts ask for documentation summarization and code review: 1843 and 1342 chat tokens. Each workload has three rounds of 256/512/1024 chunks, rotating order so every chunk occupies each position once. All eighteen arms run serially in fresh processes, without cache purges. The planner charges the chunk before allocating the cache: 885, 808 and 653 expert slots respectively.

Each arm requires thirty seconds of nominal thermal observations and target-plus-3 GB reclaimable preflight, then records launcher and generator operating conditions, sampled physical memory and terminal lifetime footprint. No arm was replaced or excluded. All eighteen passed functional completion, effective-control/prompt-ID checks, terminal sampling and timing eligibility. Thermal endpoints were nominal, low-power mode was off, and recorded swap counters did not change. These observations do not establish continuous idle hardware or continuous nominal temperature. Global paging remains a functional diagnostic outside this timing protocol.

### Results

Times below are medians of three runs per cell, in seconds. Memory is the maximum observed physical footprint across that cell, including terminal lifetime evidence. Load/cooldown are excluded from request time and retained separately in raw evidence.

| Workload | Chunk | Prefill seconds | Request seconds | Decode tok/s | Output tokens | Maximum footprint GB |
|---|---:|---:|---:|---:|---:|---:|
| prose | 256 | 38.553 | 55.424 | 5.469 | 92 | 13.034 |
| prose | 512 | 25.246 | 42.388 | 5.319 | 91 | 12.300 |
| prose | 1024 | 17.268 | 34.923 | 5.162 | 91 | 11.776 |
| code | 256 | 25.206 | 39.469 | 5.419 | 77 | 13.228 |
| code | 512 | 17.603 | 32.931 | 5.556 | 85 | 12.232 |
| code | 1024 | 14.110 | 30.598 | 5.223 | 86 | 11.655 |

The following percentages are medians of within-round comparisons with 256; they are not ratios of the preceding table's medians. A positive decode regression means slower decoding; negative means faster.

| Workload | Chunk | Prefill-time reduction | Request-time reduction | Decode-rate regression |
|---|---:|---:|---:|---:|
| prose | 512 | 34.52% | 23.52% | 2.7417% |
| prose | 1024 | 55.66% | 37.52% | 4.9967% |
| code | 512 | 31.77% | 17.74% | -2.9553% |
| code | 1024 | 44.32% | 22.88% | 3.1281% |

The predeclared shortlist requires at least 10% median paired request-time reduction in each workload, no more than 5% median paired decode-rate regression in either, and all timing/memory checks. Both candidates pass that timing-only rule. The 1024 prose median is 4.9966566% decode regression, barely below the limit; one individual round regresses 7.97%. This is not evidence of a robust margin or a statistically established universal improvement. The baseline itself becomes faster over rounds, which is why all paired observations are retained.

Maximum footprint across all arms was 13.228349440 GB, under the fixed target. The lower observed peaks with larger chunks occur with smaller planned expert caches; this is not an increase in the available RAM budget.

### Output and qualification limits

Within each workload/chunk, all three rounds reproduce identical output IDs and text and end with `stop`. Across chunks, wording and output counts differ. Request latency therefore compares observed completed requests, not byte-identical or equal-decode-work executions. Prefill processes the same exact prompt IDs in every arm.

Manual review covers all six distinct outputs. Prose answers cover the requested facts; 256 and 1024 meet the 75-word limit, whereas 512 has 76 words. All code answers wrongly imply that widening itself destroys source contiguity. The baseline shares this error and additionally conflates packed weights with metadata. The code prompt's premise may encourage that misconception. No clear new substantive manual-quality regression was found at 1024, but none of this constitutes a clean task-quality or numerical-equivalence pass.

No matched full-vocabulary/logit/state rechunk-band test was run. The existing prefill-family diagnostic uses scalar widening, fixed slots and reference optimization controls, so its result would not qualify this deployed NEON configuration. Short prompts, long completions, other memory budgets, larger contexts and other checkpoints remain outside this screen.

The next useful step is a matched numerical and representative-task check of the 1024 candidate, retaining 512 as the lower-tradeoff alternative. Keep the default at 256 until broader acceptance supports changing it. Neither a generic default change nor server activation follows from this study.

Evidence: [[sources/runs/2026/09/jang-prefill-chunks-20260916]]. See Plan018 for the frozen protocol. The raw comparison and separate manual-review artifact preserve the distinction between timing success and incomplete quality qualification.
