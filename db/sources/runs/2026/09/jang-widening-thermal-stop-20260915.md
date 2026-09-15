---
type: run
id: 01m2jyf5wrm1prb2byy92j5vzv
created: 2026-09-15T16:30:13.400404+00:00
updated: 2026-09-15T16:30:39.413532+00:00
summary: Plan011 first current-source study excluded after a nominal-to-fair thermal endpoint
binary: f63dc19f1ad46595074447199c96148212ef970c153361cfcdb35f6094b4c286
captured_at: 2026-09-15
command: .build/flash/eval-env/bin/python Tools/flash/widen_study.py qualify --pairs 10 --model models/jang-6s --candidate .build/release/slotstream --corpus .build/flash/runs/reviewer-tokenized-development-final --parity .build/flash/runs/widen-parity-qualification-20260915 --output .build/flash/runs/widen-qualify-20260915
discarded: 'true'
machines: '[[records/machines/local-m5-max-48gb]]'
title: JANG widening qualification thermal stop
tool: Tools/flash/widen_study.py
---
The first current-source Plan011 qualification attempt stopped after 39 completed model launches. The scalar arm of Python pair 09 began nominal and ended fair. It completed functionally and passed process-memory accounting, but the unchanged thermal timing gate rejected it. No qualification report or completion receipt was produced.

Raw approved fields and original artifact hashes: [thermal-stop JSON](jang-widening-thermal-stop-20260915.json). Original immutable local artifacts remain under `.build/flash/runs/widen-qualify-20260915`. No pair from this attempt is reused in the fresh cooled cohort.

The continuation requires the already-used development benchmark cooldown: thirty seconds of nominal observations with low-power mode off, followed by the original two-second settling interval. Model controls, exactness, memory, paging and performance thresholds are unchanged.
