---
type: measurement
id: 01m2jmx3jjjhqj590pewznhfnw
created: 2026-09-15T13:43:04.018181+00:00
updated: 2026-09-15T13:43:04.018181+00:00
summary: 'JANG neuron-block oracle: fidelity rejection and operator stop'
date: 2026-09-15
doc: measurements
level: '2'
machines: '[[records/machines/local-m5-max-48gb]]'
order: '1230'
runs: '[[sources/runs/2026/09/jang-flash-findings-20260915]]'
title: 'JANG neuron-block oracle: fidelity rejection and operator stop'
status: measured
---
The diagnostic oracle masks blocks of intermediate neurons inside each routed expert, while retaining full dense loading and down-projection computation. It does not implement sparse SSD reads. Original hidden values are scored against verified original column norms, and every selected mask is independently recomputed by the host. The extra diagnostic memory is charged before pool sizing.

These teacher-forced comparisons measure incremental error against unmodified JANG_6S, not quantization loss against BF16 or the publisher's reference.

| Retained blocks | Positions | Mean KL | p99 KL | Top-1 agreement | PPL ratio | Result |
|---|---:|---:|---:|---:|---:|---|
| 10 | 896 | 0.0000 | 0.0000 | 100.00% | 1.0000 | Exact control passed |
| 8 | 896 | 0.2036 | 2.2378 | 83.93% | 0.9962 | Fidelity rejected |
| 6 | 896 | 0.4098 | 3.8357 | 74.33% | 0.9586 | Fidelity rejected |

Four-block evaluation was stopped by the user after 512 of 896 positions; no complete verdict is claimed. Two blocks were not tested. The small PPL-ratio improvements for eight/six blocks do not override failed KL/top-1 gates. The original five-cohort protocol was not completed; the investigation closed at the user-revised scope.

No mask is admitted to generation. The dependent neuron predictor, sparse runtime and optional ANE predictor are not selected for this failed approach. Different methods would require a separately defined investigation.

Evidence: [[sources/runs/2026/09/jang-flash-findings-20260915]]. The retained-count reports and operator decision remain separate and hash-bound; Plans009 and its oracle execution record describe the full gates and stopping decision.
