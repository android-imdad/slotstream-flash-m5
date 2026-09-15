# Implementation plans

Revised with the improve skill on 2026-09-12 against local commit `93fb512` (plan revision 2).
This is a focused plan for the requested flash-inference and M5 work, not a whole-repository audit. Source code was not modified for this planning pass.

A second fresh-context review checked executor readiness against the source. The revision adds an independent M5 track, actual predictor training/evaluation commands, fresh-build and required-test gates, immutable evidence joins, and consistent opt-in activation. It also separates training/development/final data and corrects the statistical claims the small fixed task suite can support.

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
| [011](011-exact-widening.md) | Accelerate exact4-to6-bit expert expansion | P1 | M | 007,010; measured CPU recon | IN PROGRESS — opt-in path benchmarked at 24 GB (three pairs/workload); original ten-pair qualification remains pending |
| [012](012-archived-evidence-validation.md) | Verify archived harness bytes and immutable receipt joins | P1 | S | 008 uncommitted host implementation | DONE — reviewed in `aa9b771`; archive mutations rejected |
| [013](013-balanced-read-scheduling.md) | Balance exact expert read jobs | P1 | M | 011 packed reader | COMPLETE — component rejected; default-depth reader calls about 40% longer, no generation activation |
| [014](014-whole-expert-layout.md) | Measure whole-expert layout against original JANG reads | P1 | M | 011 packed widening, 013 reported | COMPLETE — expanded layout rejected; paired total reader time 8.48% longer |

Execution checkout: `../slotstream-flash-m5`, branch `advisor/001-flash-m5`, starting at `93fb512`. The executor owns implementation there; the reviewer maintains this index and reviews each bounded stage. The original checkout holds the completed, verified download. Model weights are reused read-only; mutable build artifacts are separate. No merge, push, default activation or completed qualification is implied.

September 15 continuation: the checkout now lives at `/Users/imdad/Documents/projects/slotstream-flash-m5`, on the existing `advisor/001-flash-m5` branch. The user specifically selected neuron skipping, SSD prefetching and Neural Engine execution for continued work. See [the bounded implementation and evidence checkpoint](009-foundation-review.md). Existing M5 GPU dispatch verification is separate from the pending Neural Engine predictor experiment. No approximate generation or serving mode has been activated.

Follow the dependency table, not numerical stage order. The first useful delivery is **0 → 1 → independent 2/5 → 7**: a baseline, actual M5 dispatch, exact-loading experiments, and a qualified report. Full-model jobs stay serial. Neuron work follows **1 → 3 → 4 → 7** only when the actual trained predictor passes; optional ANE follows **3 → 6 → 7**. A sparse rejection does not block M5 or exact loading.

All new modes remain **off by default after qualification**. This plan exposes explicit local run/library experiments; serving integration and default rollout are deferred. A completed investigation with no winning candidate is reported as such, never as a qualified speedup.

Each stage records `accepted`, `rejected`, `blocked`, `already_used`, or `not_selected` with evidence/reasons. The plan stays TODO/IN PROGRESS while selected required work remains; a blocked required track prevents DONE. An unselected optional script need not exist or run. Large artifacts go under the already ignored `.build/flash/`, not `bench/flash/`.

JANG_6S download, checksum verification and monitored smoke are complete (Plan 002). The M5 diagnostics are reviewed and committed after macOS file permission was granted on 2026-09-12. Basic file reads/Git were rechecked successfully and the exact leftover owned probe was stopped; no active profiler remained. Earlier system-trace budget failures and permission-blocked exports are preserved separately. An instrumented copy of pinned MLX confirmed grouped six-bit NAX submission and completed numerical evaluation; decode used qmv. Production trace utilization remains unverified and no LLM speedup is claimed. See Plan 003 review. Plans004–007 are now complete at their bounded scopes. The tokenizer/capture bridge preserves the original reference and exact chat answer boundary. Plan008 now builds the full frozen natural development corpus and quality metrics, followed by the dense-load neuron-block oracle. Storage-layout and heuristic-prefetch arms also remain unmeasured. Predictor training, sparse bundles/runtime, optional ANE, and final performance qualification remain conditional downstream work.

## Findings considered and rejected

- Treating M5 GPU Neural Accelerators as the Apple Neural Engine: they are separate execution devices with different APIs.
- Adding accelerator support from scratch before profiling: the pinned MLX source already contains NAX dispatch; verify the running path first.
- Assuming the paper's ReLU sparsity applies to Qwen SwiGLU: neuron omission requires its own accuracy and useful-block-sparsity evidence.
- Replacing native 6-bit affine weights with a convenient 4-bit tensor format: changes the selected quant and its numerical quality.
- Requantizing down-projection columns after transposition: loses the original quantization-group contract.
- Repeating the old background read-ahead design without evidence: repository measurements found it slower; any new predictor must cover all staging and coordination costs.
- Converting the whole out-of-RAM MoE to Core ML / ANE: outside this plan; the bounded predictor is the tractable initial experiment.
- Copying the paper's speedup numbers or current Pipe4 benchmarks into JANG claims: requires new same-device, same-checkpoint measurements.

Not audited: unrelated server/API behavior, downloader correctness, product UX, security, or general code quality. This plan reuses the existing test and memory boundaries and adds tests specifically for the proposed work.


September 15 next-stage result: [Plan 013](013-balanced-read-scheduling.md) tested
balanced exact read scheduling against the existing packed-widening reader. It
failed the predeclared component gate and remains diagnostic-only. The failed
neuron prerequisite makes its trained predictor and optional ANE predictor
`not_selected` for this approach. SSD prefetch remains replay-only without timing
admission. No next optimization or full-model benchmark followed the rejection.
