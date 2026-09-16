---
type: run
id: 01m2md1bzw5b6pxpyhyg1fc8gk
created: 2026-09-16T06:04:03.964138+00:00
updated: 2026-09-16T06:04:04.418379+00:00
summary: Matched JANG prefill numerical qualification capture
binary: f46eb314c4cf682442bf6c009e9ba2649898d8a8e9dfe95d7128008c1bc0abb5
captured_at: 2026-09-16
command: .build/flash/eval-env/bin/python .build/flash/prefill-qualification-20260916/capture_run.py
discarded: 'false'
machines: '[[records/machines/local-m5-max-48gb]]'
title: Matched JANG prefill numerical qualification capture
tool: prefill-capture and frozen host comparison
---
The [original raw capture cohort](jang-prefill-numerics-20260916.json) preserves both ordinary-output smokes and all six numerical arms: commands, binary/source/harness identity, original prompt IDs, native artifact hashes, manifests, terminal memory and launcher evidence. Its original evaluator failures remain immutable.

The [final evaluation and history](jang-prefill-numerics-review-20260916.json) binds the unchanged original capture hash and the V3 comparator. It preserves corrected assessments, the metadata and routing-classification corrections, all per-field observations, and the explicit skipped task/timing follow-up. Full binary arrays and individual launcher files remain under `.build/flash/runs/prefill-qualification-20260916/numerics/`; the public record carries their hashes, not the large payloads.

The [frozen fixture bundle](jang-prefill-qualification-fixtures-20260916/SHA256SUMS.json) includes the capture orchestrator, all evaluator versions/tests, prepared task driver/tests, exact task source/tokenizer artifacts and the pre-inference protocol. Driver paths expect those contents under `.build/flash/prefill-qualification-20260916/`. Existing output paths are refused: reruns require a new declared cohort and identifiers. The candidate executable, source tarball and metallib are preserved locally under `.build/flash/runs/prefill-qualification-20260916/candidate-archive/` and validated by the captured build identity.

The task fixture preparation does not mean those model tasks ran. No new default, release, commit or push follows from this numerical investigation.
