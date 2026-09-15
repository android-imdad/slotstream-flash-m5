# Neuron and prefetch implementation checkpoint — September 15

> Historical review/checkpoint at the stated source and evidence versions. Later completion, rejection and remaining work are reconciled in [the current index](README.md) and [JANG Flash findings](../docs/JANG-FINDINGS.md). Preserve the original evidence; do not treat historical active-work wording as a new task.

## Scope

The user clarified that continuation means neuron skipping, SSD prefetching and Neural Engine execution. This historical delivery advanced their prerequisites and the internal neuron-mask component. Plan009 and the parent were in progress at that checkpoint; Plan009 has since closed at the user-revised scope. Existing M5 GPU accelerator verification remains unchanged; no new Neural Engine execution is implemented or claimed.

Checkout: `/Users/imdad/Documents/projects/slotstream-flash-m5`, existing branch `advisor/001-flash-m5`. Changes were local and uncommitted at this checkpoint; the subsequent implementation and findings are now committed in the branch history. Prior widening changes and the separate upstream 24 GB target remain present. The new work does not change checkpoint weights, quantization, normal generation, serving defaults or dependency pins.

## Implemented

- `FlashNeuronScoring`: original FP32 values, sequential FP64 column sums and square roots, one final FP32 rounding; signed SwiGLU contribution scores; deterministic 64-neuron block selection.
- `flash-column-norms`: sample/full export through the existing expert reader. One expert at a time; no Engine or resident trunk. The model process lock, bounded writer, descriptor/path identities, metadata hashes and archived executable provenance protect the export. A small package wrapper leaves the standalone ProcessMemory compilation gate intact.
- `FlashHiddenTransform` and `FlashNeuronMaskTransform`: package-scoped, default-nil hook distinct from observation. The original hidden tensor is observed first. Ten blocks returns the same MLX tensor object; partial masks retain dtype and the full down-projection shape. Multi-token, split-rank, resident-overlap and expert-workspace combinations are rejected. Failures propagate through the checked forward before its state is committed. The charged adapter was not implemented at this checkpoint. It was subsequently completed for explicit diagnostic capture; normal run/serve still cannot activate the mask.
- `calibrate.py`: strict norm-file validation and complete development-corpus block scoring. Records zeros, contribution scores, five predeclared masks, per-category/layer summaries and individual-neuron block unions. Norms are read-only mappings; activation/scoring work is streamed. Hypothetical bundled source bytes are separated from measured savings and model quality.
- `prefetch_study.py`: previous-token and recent-eight-token frequency forecasts at the preceding layer boundary. Forecasts never inspect the future router or modify CLOCK. Charges original staging plus expanded insertion scratch by reducing the candidate pool; reports the induced cache difference, useful/wasted reads and one-batch staging bound. This is replay, not native asynchronous prefetch.
- Historical prompt validation accepts a directory alias only when it resolves to the same executable; binary, archive, metadata and receipt hashes remain required. The old checkout path is a compatibility symlink to the moved checkout; recorded archives were not rewritten.

## Evidence

All artifacts below are local under `.build/flash/`; original archives remain intact.

| Evidence | Result |
|---|---|
| `runs/neuron-norm-sample-20260915` | Real source-layout sample complete; terminal lifetime peak 331,727,640 bytes |
| `runs/neuron-norm-full-20260915` | All 48 layers / 24,576 experts; 62,914,560 norm bytes; terminal lifetime peak 421,462,904 bytes, within the 1 GB process target |
| Full norm manifest SHA-256 | `1815336f58c807b7d5597a9a512d0f2e5f5c7e25a86d310b6ce3efc194581e13` |
| `runs/neuron-calibration-launch-20260915/study` | Complete 896-position development cohort; 430,080 routed examples; terminal lifetime peak 281,870,960 bytes |
| Development activations | 6 exact zeros among 275,251,200 values. This does not support assuming ReLU-like exact sparsity; it does not establish approximate-mask quality. |
| Retain 8 / 6 / 4 / 2 blocks | Mean omitted contribution-score fractions approximately 0.162 / 0.344 / 0.541 / 0.755. These are contribution bounds, **not** output-error or logit-quality measurements. |
| Individual-neuron selection | Selecting the top 128 neurons still touches almost all ten 64-neuron blocks on average. Fine neuron counts do not imply corresponding block I/O savings. |
| `runs/causal-prefetch-budgeted-20260915` | Three original traced workloads revalidated against exact CLOCK demand. Charges 20 slots, reducing the recorded pool from 793 to 773. Previous-token potential coverage 1.775%, total read increase 3.868%; recent-frequency coverage 14.502%, total read increase 65.551%. No measured latency claim. |

The earlier `runs/causal-prefetch-20260915` report is preserved as the uncharged exploratory replay; use the budgeted report above for decisions. Global paging remains diagnostic for functional acceptance. A separate widening-study repair excludes contaminated timings from that performance study without changing functional acceptance.

## Verification and limitations

Native column/scoring orientation, signed values, tie-breaking, invalid values/counts and overflow checks passed before full export, as did the original JANG numeric checks. Host corruption, incomplete/full-vs-sample assets, forecast causality, wrong forecasts, staging bounds and byte accounting have tests. The frozen evaluation Python environment is required for NumPy-backed tests; system Python lacks NumPy.

The initial moved-checkout build failed on duplicate absolute module-cache paths; the old generated ModuleCache was moved aside and a fresh two-job release build succeeded. The initial static pass exposed the package modifier breaking standalone ProcessMemory compilation; the modifier was removed and the lock access wrapper moved into the new diagnostic file. Final build/check outcomes are recorded in the continuation handoff.

Final validation passed: release build, 201 host tests, 39 T0 checks, the seven-assertion native mask check, original JANG numerics and the complete repository static battery. The final archived binary also matched the immutable reference's full logits, routes, retained state and token IDs at all 12 positions in the accepted small natural development shard with the mask hook disabled (`runs/neuron-default-parity-20260915`). The source-bound archive is `runs/neuron-foundation-final-archive-20260915`.

This foundation checkpoint did not establish masked full-model quality or runtime speed. Subsequent oracle cohorts and the prefetch cost screen are recorded separately; no partial mask, native prefetch speedup or Neural Engine predictor was qualified. The measured upstream 24 GB target in `BASELINE.md` has not been beaten by this work.

## Follow-up disposition

1. The charged versioned oracle adapter, norm ownership, streaming masks and complete-cohort evaluator were implemented in Plan009.
2. Dense-ten parity passed. Eight/six blocks failed fidelity; four stopped incomplete at user direction and two was not tested. Do not resume the original lower-count sequence automatically.
3. The dependent free-generation/predictor/sparse-runtime/ANE path was not selected because no tested partial mask passed the gate.
4. Plan016 completed the measured reader-cost/overlap screen; both existing heuristic-prefetch policies were rejected at its matched budget. No native worker was admitted.

Reproduce the completed stages:

```sh
python3 Tools/flash/calibrate.py norms --binary .build/release/slotstream --model models/jang-6s --sample --output .build/flash/runs/<fresh-sample>
python3 Tools/flash/calibrate.py norms --binary .build/release/slotstream --model models/jang-6s --output .build/flash/runs/<fresh-full>
.build/flash/eval-env/bin/python Tools/flash/calibrate.py study --suite Tools/fixtures/flash/suite.json --capture-index .build/flash/runs/reviewer-quality-development-final/cohort --norms .build/flash/runs/neuron-norm-full-20260915/run/native --output .build/flash/runs/<fresh-study>
python3 Tools/flash/prefetch_study.py --collection .build/flash/runs/window-collection-terminal --output .build/flash/runs/<fresh-prefetch-screen>
```
