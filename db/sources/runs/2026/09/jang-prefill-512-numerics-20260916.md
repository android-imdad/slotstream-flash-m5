---
type: run
id: 01m2mepbg7575z1t05tkabmd52
created: 2026-09-16T06:33:00.167326+00:00
updated: 2026-09-16T06:33:00.691344+00:00
summary: Independent JANG 512 prefill qualification capture
binary: 25691b268588da85912d83e785ace0d858810a795d8daa7ec2f5919d7b8210ec
captured_at: 2026-09-16
command: .build/flash/eval-env/bin/python .build/flash/prefill-512-qualification-20260916/capture_run.py
discarded: 'false'
machines: '[[records/machines/local-m5-max-48gb]]'
title: Independent JANG 512 prefill qualification capture
tool: prefill-capture and Plan020 frozen comparison
---
The [raw numerical cohort](jang-prefill-512-numerics-20260916.json) was captured before interpretation and contains the prospective protocol, exact driver/fixture/build/harness identities, six fresh native captures, memory/terminal receipts, artifact hashes and complete comparator output. The [assessment](jang-prefill-512-assessment-20260916.json) binds that unchanged raw hash and adds independent verification plus the explicit skipped task/timing disposition.

The [frozen helper bundle](jang-prefill-512-fixtures-20260916/SHA256SUMS.json) preserves the exact capture driver, comparator/tests, conditional task driver/tests, incremental diagnostic patch and pre-inference Plan020 protocol. Drivers expect those files under `.build/flash/prefill-512-qualification-20260916/`. They reference the previously frozen, unrun Plan019 task inputs under `.build/flash/prefill-qualification-20260916/`, whose portable originals remain in [the Plan019 fixture bundle](jang-prefill-qualification-fixtures-20260916/SHA256SUMS.json). Fixed output paths refuse an existing cohort; a rerun needs newly declared identifiers and must not overwrite this record.

Large raw tensors, native launcher files and the validated executable/metallib/source archive remain under `.build/flash/runs/prefill-512-qualification-20260916/`. Public reports preserve their hashes rather than embedding the multi-gigabyte arrays. No timing rows from Plan018/019 were pooled into this independent capture.

No task-screen or extended timing model was launched. No default, ordinary inference algorithm, release, commit or push changed.
