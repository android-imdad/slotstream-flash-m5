# 018 — Bounded JANG prefill-chunk screen

Status: COMPLETE at bounded screening scope, September 16, 2026. Source baseline: `8f79284`.

All eighteen arms completed and passed timing/memory checks. Both larger chunks
pass the timing-only shortlist. The 1024 prose decode regression is barely under
the prescribed limit; manual review does not provide a clean quality pass.
Matched numerical and broader task qualification remain pending. Runtime source
and the default are unchanged. See the [canonical result](../db/records/measurements/jang-prefill-chunks-2026-09-16.md).

The user authorized the recommended fixed-budget prefill comparison. This is
an exploratory screen using existing controls, not a default change or the
unfinished Plan011/parent001 qualification.

## Question and scope

Does increasing JANG_6S's prefill chunk from 256 to 512 or 1024 reduce total
request latency enough to justify further numerical qualification? A larger
workspace reduces the expert cache at the same process target, so prefill speed
alone cannot decide this.

Use the current validated NEON binary, pinned JANG_6S checkpoint, 14 GB process
target, 4096-token configured context, greedy seed 42, and at most 128 output
tokens. Disable MTP, vision and prefix reuse. All three arms explicitly use
`packed4-to6` and `SLOTSTREAM_PREFILL_CHUNK`. Preserve original model bytes.

Two frozen prompts use actual repository documentation and expert-reader code,
with concise summarization/review questions. The pinned native tokenizer reports
1843 and 1342 chat tokens. Prompt text, tokenizer receipts, source identity and
exact token IDs are frozen before generation. These are development workloads,
not held-out quality benchmarks.

## Protocol

- Three complete rounds per workload, rotating chunk order to balance position:
  eighteen serial arms in fresh CLI processes. No cache purges.
- Before each arm require thirty seconds of nominal thermal observations,
  low-power mode off, model exclusivity and at least target-plus-3 GB reclaimable
  memory. Limit readiness waiting to 180 seconds and each arm to 900 seconds.
- Use the existing monitored launcher and terminal lifetime-footprint sampler.
  Stop on failed completion, target breach, source/control drift, or ineligible
  thermal/paging observations. Preserve partial evidence; do not replace arms
  or pool incomplete cohorts. Global paging exclusions apply only to this
  timing screen, not a general correctness verdict.
- Raw stats, receipts, operating observations, outputs and identities enter
  `db/sources/runs/2026/09/jang-prefill-chunks-20260916.json` before summaries.
- Check exact prompt IDs and effective runtime controls. Pool slots, expert
  work and output IDs may differ with chunk size; record rather than suppress
  them. Keep load time separate from request time.

## Screen and interpretation

Compare median paired request-time reductions, prefill and first-token times,
decode tokens/second, output-token counts, finish reason, and maximum observed
physical footprint. A candidate advances only if request time improves at least
10% in both workloads, median paired decode throughput regresses no more than
5% in either, and all arms meet the memory/timing checks. Manually review every
distinct output for task usefulness and completeness. Changed output lengths
confound direct equal-work speed claims and must be disclosed.

This screen does not establish full-vocabulary/logit/state parity. Rechunking
can legitimately change floating-point reduction order; exact output equality
is diagnostic, not the numerical acceptance rule. A passing screen is only a
candidate for the existing rechunk-band method on the actual JANG configuration.
Do not change the default from these two workloads alone.

## Artifacts and closure

Frozen driver and prompts: `.build/flash/prefill-chunks-20260916/`.
Native runs/archive: `.build/flash/runs/prefill-chunks-20260916/`.
Save a canonical run and measurement through dbmd, update this status and the
plan index, regenerate projections and run brain/projection/claims gates.
Confirm no owned model or readiness waiter remains. No commit, push, release
or default activation is implied by completing this screen.
