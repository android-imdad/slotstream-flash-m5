# Implementation plans — current status

**Public WIP research repository:** [android-imdad/slotstream-flash-m5](https://github.com/android-imdad/slotstream-flash-m5), branch `main`. The local working branch remains `advisor/001-flash-m5`; the original `origin` remote remains upstream. Publication here does not mean an upstream merge, release or default activation.

Updated September 16, 2026 after the NEON full-model benchmark and the
metadata/scratch implementation experiments. The original plans were authored on September 12;
their dated briefs/reviews are retained as history, not new pending tasks.
Execution remains on `advisor/001-flash-m5` in this repository. Source and
measurement commits do not imply merge, push, upstream release or default rollout.

| Plan | Title | Priority | Effort | Dependency | Status |
|---|---|---|---|---|---|
| [001](001-flash-inference-m5.md) | Adapt LLM in a flash to Qwen MoE streaming and qualify M5 acceleration | P1 | L | Verified JANG_6S baseline complete | IN PROGRESS |
| [002](002-flash-foundation-execution.md) | Implement the Flash experiment launcher and baseline evidence gate | P1 | M | None for host tools; model-slot handoff for generation | DONE — reviewed `91bbaf7`, verified model smoke |
| [003](003-m5-dispatch-probe.md) | Measure the existing M5 expert-kernel path | P1 | M | 002 foundation/archive | DONE — reviewed `99de033`, private dispatch confirmed; public utilization unverified |
| [004](004-expert-window-screen.md) | Screen recent-token expert cache windows against native CLOCK | P1 | M | 002 evidence and 003 reviewed commit | DONE — exact live reconciliation; window candidates rejected (best2.08% byte savings) |
| [005](005-diagnostic-reference.md) | Establish bounded logits, state and activation reference | P1 | L | 002,003 and006 | DONE — bounded reference accepted; full natural corpus remains in parent001 |
| [006](006-terminal-memory-sampling.md) | Preserve final Darwin lifetime peak before child reaping | P1 | M | Reproduced lifecycle evidence | DONE — reviewed `7f63f0b`,115 tests and real terminal-peak proof |
| [007](007-natural-tokenized-capture.md) | Freeze natural text into tokenized capture shards | P1 | M | 002,005,006 | DONE — reviewed `0b032fa`, exact reference/chat parity |
| [008](008-corpus-quality-metrics.md) | Freeze corpus and validate full-vocabulary quality metrics | P1 | L | 007,010,012 | DONE — reviewed `aa9b771`;896-position real cohort/metrics |
| [009](009-neuron-block-oracle.md) | Measure neuron-block oracle quality and possible byte savings | P1 | L | 008 corpus and metrics | COMPLETE at user-revised scope — dense-ten exact; eight/six fail fidelity; lower counts stopped by user; no sparse runtime admission |
| [010](010-prompt-tokenizer.md) | Freeze complete task prompts without inference | P1 | M | 007 tokenizer facade | DONE — reviewed `bc185df`, exact tokenizer overlap |
| [011](011-exact-widening.md) | Accelerate exact4-to6-bit expert expansion | P1 | M | 007,010; measured CPU recon | IN PROGRESS — NEON backend implemented; focused exact/source checks and component comparison pass; short full-model comparison completed; formal qualification remains pending |
| [012](012-archived-evidence-validation.md) | Verify archived harness bytes and immutable receipt joins | P1 | S | 008 uncommitted host implementation | DONE — reviewed in `aa9b771`; archive mutations rejected |
| [013](013-balanced-read-scheduling.md) | Balance exact expert read jobs | P1 | M | 011 packed reader | COMPLETE — component rejected; default-depth reader calls about 40% longer, no generation activation |
| [014](014-whole-expert-layout.md) | Measure whole-expert layout against original JANG reads | P1 | M | 011 packed widening, 013 reported | COMPLETE — expanded layout rejected; paired total reader time 8.48% longer |
| [015](015-source-native-layout.md) | Measure original-byte whole-expert records | P1 | M | 014 reported | COMPLETE — rejected; paired total reader time essentially tied, below the improvement gate |
| [016](016-prefetch-cost-screen.md) | Price causal SSD-prefetch timing and overlap | P1 | M | Existing trace replay and packed widening | COMPLETE — both forecasts rejected by matched 14 GB timing/cost screen; no native worker |
| [017](017-jang-reader-conversion.md) | Optimize metadata conversion and lane-local scratch | P1 | M | NEON benchmark complete | COMPLETE — metadata already vectorizes; scratch lacked reader benefit; original runtime retained |

## Outcome

The complete pinned JANG_6S checkpoint is verified and real generation has been
measured. Explicit exact widening has a matched development speedup. Its formal
qualification remains unfinished. Neuron masking, cache windows, balanced reads,
whole-expert layouts and the tested heuristic prefetch policies did not pass
their respective gates. Rejected diagnostics remain available for evidence;
they are not enabled inference features.

The neuron oracle closed at the user's revised scope: eight/six failed fidelity,
four stopped incomplete and two untested. Its predictor, sparse runtime and
optional Neural Engine predictor are **not selected** for this approach. Do not
resume the lower-count jobs automatically. Existing M5 GPU dispatch was observed
in an instrumented build; production utilization is still unverified and no new
ANE execution was added.

The [metadata and scratch follow-up](../db/records/measurements/jang-reader-followup-2026-09-16.md) is complete: both runtime candidates were rejected and the original implementation retained. Do not repeat these exact approaches without new evidence. Bounded contiguous prefill reads and native-bit cache storage remain separate unexecuted candidates.

## Pending, ranked

1. **Plan011 implementation follow-up:** the new NEON backend has a bounded CPU
   conversion result and a [short 14 GB full-model comparison](../db/records/measurements/jang-neon-full-model-2026-09-16.md).
   The saved 24 GB configuration has not been remeasured with NEON.
   The preceding backend's formal studies remain thermally inconclusive; no pair
   replacement, pooling, or transfer of those results to the new backend is allowed.
2. **Parent001 final tooling and qualification:** aggregate run-set creation,
   stage-seven gates/evaluator, required final coverage and selected-arm reporting
   are incomplete. Proposed interfaces in the parent plan are not runnable today.
3. **Broader scope:** long-context/task coverage, JANG_4M full-model evidence,
   publisher/BF16 reference reproduction and portable evaluation/bootstrap inputs
   remain unqualified. Do not apply JANG_6S results to them.
4. **Deferred integration:** serving/default rollout, upstream integration/release and production
   M5 utilization claims remain separate work. The original-checkpoint learned
   Expert Lookahead proposal was not evaluated by the JANG heuristic screens.

No current neuron, layout or heuristic-prefetch candidate is admitted for further
runtime implementation. A different method/configuration requires its own
predeclared quality, cost and performance investigation. Benchmark a stage before
starting the next optimization; use the actual measured result to decide whether
to proceed.

**M5 profiling follow-up:** a bounded packed-path CPU stack diagnostic completed
with exact output/work parity under recorded fair conditions. It observes reads,
packed expansion, staging and waits, not GPU kernel durations. Clean 256-/1024-token
request profiles remained unrun because nominal conditions were unavailable.
No custom kernel is admitted from this evidence. [Current qualification/profile
record](../db/records/measurements/jang-widening-qualification-attempts-2026-09-15.md).

## Evidence and documentation

- [Public findings](../docs/JANG-FINDINGS.md) — measured widening, failed fidelity,
  reader experiments, modeled prefetch estimates and explicit limits.
- [JANG usage](../docs/JANG.md) — local build, weight verification, run/library
  widening controls and serving boundaries.
- [Canonical current plan](../db/records/plan/jang-flash-qualification-status.md)
  and [admission decision](../db/records/decisions/jang-flash-admission-2026-09-15.md).
- [Portable evidence](../db/sources/runs/2026/09/jang-flash-findings-20260915.json)
  — approved fields and hashes of original completed reports.
- [Current handoff](EXECUTION-HANDOFF.md) — remaining work and immutable local
  archive locations; old active process IDs remain only in Git history.

Full binaries, logits, traces and sample payloads remain in ignored local archives.
A fresh clone does not contain all historical inputs. Preserve matching source,
model, native and harness identities; never rewrite an old hash to accept a new
binary. The old checkout path remains an alias for reading historical artifacts.
No model process is currently running.

All new runtime modes remain off by default. Full-model jobs remain serial on the
shared Mac, use explicit budgets and preflight headroom, and keep benchmark timing
eligibility separate from functional/global-paging diagnostics. This index covers
the selected Flash/M5 investigation, not unrelated server, downloader or product
roadmaps.
