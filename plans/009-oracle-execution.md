# Oracle adapter and quality campaign — September 15, 2026

## Implemented scope

`flash-capture --mode oracle --oracle-config <file>` is diagnostic-only. Run and serve reject the new flag. Reference formats retain their existing validators; oracle output and configuration have separate versioned formats and strict validation.

The adapter charges 208 MiB total diagnostic reservation: the existing 128 MiB plus 80 MiB for the single 60 MiB CPU norm table and bounded scratch/masks. The charge precedes norm allocation and pool sizing. Original norm files are checked for complete coverage, bytes, hashes, finite nonnegative values and source metadata. One MiB is the maximum hash-read scratch. No full-table duplicate is retained.

Whole-document warmup remains dense. Each scored position installs the transform, observes the original hidden activations, masks the selected blocks and follows the current actual routes. At the last layer, the mask artifact is written before the checked forward commits its state. Mask failures and cancellation propagate; a real injected transform failure verifies that the incomplete state cannot be reused. Hooks reset between positions/documents and on errors. Dense-ten returns the unchanged hidden tensor object.

The host verifies every mask's position, layer, router ranks, actual expert IDs and Boolean cardinality. For partial masks, it independently recomputes each selection from captured original hidden values and original norms. Numerical controls and unrelated memory-ledger fields must remain unchanged; only the charged reservation and its derived cache reduction are permitted. The original full-vocabulary metric implementation and per-category thresholds are reused.

## Preliminary evidence

- Two-job release build passed: `.build/flash/oracle-build-final-20260915.log`.
- 209 host tests passed: `.build/flash/oracle-benchmark-host-tests-20260915.log`.
- T0 and mask component checks passed: `.build/flash/oracle-native-{t0,mask}-20260915.json`.
- Six CLI preallocation rejection cases passed: `.build/flash/oracle-cli-rejections-20260915.json`.
- Repository static battery passed: `.build/flash/oracle-static-20260915.log`.
- Real 64-position dense-ten smoke passed with exact logits/routes/state/token identity: `.build/flash/runs/oracle-dense-smoke-20260915`. Terminal physical peak was 9,695,433,720 bytes. Pool reduction was 854 to 834 slots.

These checks do not qualify any partial mask or speedup.

## Frozen complete campaign

`.build/flash/runs/oracle-campaign-v2-20260915` binds the archived candidate, host harness, original model descriptor/header identities, norm payloads, study, suite, reference cohort and all configurations before inference. It covers all 896 development positions across 14 shards for each retained count, in order 10, 8, 6, 4, 2. Missing or failed shards invalidate their cohort; no selective replacements. Global paging remains diagnostic for functional quality checks.

The earlier `oracle-campaign-20260915` freeze was superseded before inference to add checkpoint source-identity checks. Its withdrawal record is preserved. No outputs from separate campaigns are combined.

## Operator stopping decision and completed results

The user subsequently directed stopping below six blocks after the stronger partial settings failed. The owned serial controller and evaluator were stopped and no model process remained. `oracle-campaign-v2-20260915/operator-decision.json` binds this decision and revalidated completed evidence. The interrupted four-block receipt is explained by that operator stop, not a newly discovered correctness failure.

| Retained blocks | Positions completed | Mean KL | p99 KL | Top-1 agreement | PPL ratio | Disposition |
|---|---:|---:|---:|---:|---:|---|
| 10 | 896 | 0 | 0 | 100% | 1.0000 | Exact control passed |
| 8 | 896 | 0.203555 | 2.237785 | 83.93% | 0.99625 | Fidelity gate failed |
| 6 | 896 | 0.409843 | 3.835672 | 74.33% | 0.95860 | Fidelity gate failed |
| 4 | 512 | — | — | — | — | User-stopped; incomplete |
| 2 | 0 | — | — | — | — | Not run |

The PPL ratios alone do not override KL and top-1 fidelity requirements. The observed results justify stopping this approach as not promising; they do not prove the uncompleted settings' numerical outcomes. Partial masking remains disabled, and predictor/ANE advancement is not admitted.

Dense-ten must pass exact full-logit/route/state/token identity. Partial masks must meet mean KL ≤ 0.001, p99 KL ≤ 0.01, top-1 agreement ≥ 99%, and perplexity ratio ≤ 1.01 both pooled and per category. Full generation and predictor advancement remain separate gates even if teacher-forced masking passes.

## Benchmark before another optimization

The user explicitly requested benchmarking after this stage and before the next optimization. `Tools/flash/engine_bench.py` uses the saved upstream baseline prompts and target: 24 GB, 32K context, greedy seed 42, up to 128 output tokens, prefix cache off. It runs three paired fresh-process JANG CLI rounds per workload, alternating before/after order and requiring exact output IDs/work and matched effective controls.

The before arm is the immutable pre-widening scalar JANG binary; the after arm is the current exact `packed4-to6` path with neuron oracle disabled. This measures the current runnable exact engine, not an unqualified mask. It records decode speed, first-token/first-text/prefill/request/load timing and terminal physical memory. Contaminated timing intervals are retained but cannot support a speed comparison.

The saved upstream result is a target from another quant on a persistent HTTP server with MTP and lookahead; it is not the causal control arm, and its client latency has different boundaries. Do not claim a neuron-skipping speedup from the diagnostic oracle or this dense benchmark.

The first benchmark attempt was stopped after Foundation reported non-nominal (`fair`) thermal state. It is retained under `engine-benchmark-24gb-20260915` with an operator stop record. A fresh `engine-benchmark-24gb-cooled-20260915` cohort requires 30 seconds of nominal observations at five-second intervals before every arm and aborts on an ineligible timed arm. Cooldown time is outside inference timing. The prior oracle harness was archived byte-for-byte before this benchmark-only change.

## Completed 24 GB benchmark

The cooled cohort completed all nine pairs / eighteen arms. Every pair preserved exact output IDs and model work, matched the effective controls, stayed under the process-memory ceiling, and passed the timing-eligibility checks. The final archive still matches current native source. `verification.json` rechecks the receipts, actual artifacts, cooldown records and exact work joins.

| Workload | Scalar median tok/s | Packed median tok/s | Throughput ratio | First text, scalar → packed | Packed maximum physical peak |
|---|---:|---:|---:|---:|---:|
| Explanation | 3.199 | 6.983 | 2.18× | 5.870 → 2.930 s | 20.651 GB |
| Coding | 3.331 | 7.409 | 2.22× | 5.311 → 2.584 s | 20.650 GB |
| Reasoning | 3.272 | 7.266 | 2.22× | 6.623 → 3.235 s | 20.685 GB |

Median paired decode-time reductions were 54.19%, 55.07% and 55.02%, respectively. First-text figures are the engine's generation statistic and exclude model loading and cooldown waits. The scope is three paired fresh-process CLI rounds per saved workload, on this device and checkpoint; this is a measured checkpoint, not the original ten-pair qualification or permission for default rollout.

The current exact JANG path does **not** exceed the saved upstream decode targets of 14.003 / 15.901 / 15.684 tok/s. That upstream result uses another quant and persistent HTTP serving with MTP/lookahead, so it remains a target rather than the controlled before arm. No neuron-skipping speedup, native prefetch win or Neural Engine execution is claimed.

Evidence directory: `.build/flash/runs/engine-benchmark-24gb-cooled-20260915`.

Report SHA-256: `d623c0ff1af9c3f03712d2dbe52b85e0e0b6b9eec752ab3e494f75e249edf5f8`.

No model process remained after completion. No new optimization was started after this benchmark, and no defaults, checkpoint weights, merge or push were changed.
