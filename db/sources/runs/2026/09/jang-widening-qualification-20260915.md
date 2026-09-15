---
type: run
id: 01m2jzpe2xfqrwttmm8kx2j5rc
created: 2026-09-15T16:51:39.741525+00:00
updated: 2026-09-15T17:03:28.775769+00:00
summary: Current-source widening evidence, two thermal exclusions and a separate non-timing CPU sample
binary: f63dc19f1ad46595074447199c96148212ef970c153361cfcdb35f6094b4c286
captured_at: 2026-09-15
command: .build/flash/eval-env/bin/python .build/flash/qualification-20260915/export_session.py
discarded: 'false'
machines: '[[records/machines/local-m5-max-48gb]]'
title: JANG widening qualification attempts and CPU diagnostic
tool: Tools/flash/widen_study.py and frozen stage-five diagnostic drivers
---
Current-source Plan011 prerequisites passed: host tests, native checks, bounded synthetic/source component tests, exact reference capture parity and the fresh three-prompt screen. The subsequent complete-study attempts did not qualify: the first stopped at a fair thermal endpoint; the cooled successor also stopped at a fair endpoint. Neither has a successful qualification receipt.

[Approved result fields and original artifact hashes](jang-widening-qualification-20260915.json) include every complete pair in the cooled attempt, endpoint conditions, memory and paging evidence, and the explicit incomplete verdict. Only complete, timing-eligible workload subsets receive descriptive aggregates. They are not a passing overall cohort.

Frozen local evidence:

- `.build/flash/runs/widen-parity-qualification-20260915`
- `.build/flash/runs/widen-screen-qualification-20260915`
- `.build/flash/runs/widen-qualify-20260915`
- `.build/flash/runs/widen-qualify-cooled-20260915`
- `.build/flash/runs/jang-stage5-cpu-sample-20260915`
- `.build/flash/qualification-20260915` for host/native checks, protocols and command logs.

The nominal-only raw code-prefix request profile never launched a model. Its owned prelaunch waiters were stopped; no other task was stopped. A separate short-prompt CPU stack sample ran under recorded fair conditions, matched the prior uninstrumented packed output/work, and is explicitly ineligible for performance claims. Its hash is in the JSON extract. Full sampling output remains local.

Frozen reproducible drivers: [request-profile preparation](jang-stage5-request-profile-driver-20260915.py), [CPU sample](jang-stage5-cpu-sample-driver-20260915.py), [approved-field export](jang-widening-export-driver-20260915.py). Run from the repository root. The request-profile preparation uses the pinned model tokenizer via the recorded tokenizers environment; model execution uses the existing evaluation environment. The saved source code and local protocols identify the exact commands; raw captures and model files are not bundled here.

Related exclusions: [[sources/runs/2026/09/jang-widening-thermal-stop-20260915]] and [[sources/runs/2026/09/jang-widening-cooled-stop-20260915]]. No new kernel, inference default, serving exposure, checkpoint or release was added.

Validation closure: all existing static gates passed, including installer, downloader, memory and planner checks. The brain/projection/claims checks passed with the two pre-existing historical-log warnings. Original validation logs remain in `.build/flash/qualification-20260915`; [validation summary and hashes](jang-widening-session-validation-20260915.json) bind them. No owned model or profiling process remains. This does not change the thermally inconclusive qualification verdict.

The complete frozen cooled-cohort harness matches Git commit `0509f71` byte-for-byte against its recorded harness hash map. A later whitespace-only cleanup of the shared readiness helper does not alter the saved evidence or its source identity.
