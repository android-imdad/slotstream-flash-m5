---
type: run
id: 01m2mb1yn68wz6nbg76ft6e1sw
created: 2026-09-16T05:29:25.926069+00:00
updated: 2026-09-16T05:29:45.581587+00:00
summary: JANG fixed-budget prefill chunk comparison
binary: 0124a9e69e0ba2a7583eba6b73b6288eda68ce33ce58f5fd44762b4bdf3050b8
captured_at: 2026-09-16
command: .build/flash/eval-env/bin/python .build/flash/prefill-chunks-20260916/run.py
discarded: 'false'
machines: '[[records/machines/local-m5-max-48gb]]'
title: JANG fixed-budget prefill chunk comparison
tool: Frozen driver with monitored Flash launcher
---
Raw native statistics, outputs, sampled-memory traces, terminal receipts, operating conditions, exact commands, source/binary identities and the full frozen driver were captured in [the raw JSON](jang-prefill-chunks-20260916.json) before interpretation. The eighteen-arm run completed without replacement or exclusion. The raw report retains its original pending-review labels; [the separate completed manual review](jang-prefill-chunks-review-20260916.json) records subsequent assessment without rewriting the frozen run.

The [frozen fixture directory](jang-prefill-chunks-fixtures-20260916/SHA256SUMS.json) preserves the driver, exact prompt source/text and tokenizer receipts/artifacts. The driver's expected working location is `.build/flash/prefill-chunks-20260916/run.py` in the repository root; copy the directory contents there to inspect the original invocation. Its fixed output path refuses an existing cohort: never overwrite this run or reuse it as a fresh run identifier. The driver still requires its bound binary/model and matching harness files; this fixture bundle is not a complete clean-clone model distribution.

The full executable/metallib/source/harness archive and individual arm directories remain under `.build/flash/runs/prefill-chunks-20260916/`. Only the parent agent owned model execution. The process target was fixed at 14 GB; all model processes exited after their requests. No runtime source, default, commit, push or release changed.
