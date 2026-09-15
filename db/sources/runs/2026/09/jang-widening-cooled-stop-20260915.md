---
type: run
id: 01m2jza8d3whs0g73r0jbymbm3
created: 2026-09-15T16:45:00.706974+00:00
updated: 2026-09-15T16:45:01.317076+00:00
summary: Cooled Plan011 cohort remains thermally inconclusive after its second scalar arithmetic run
binary: f63dc19f1ad46595074447199c96148212ef970c153361cfcdb35f6094b4c286
captured_at: 2026-09-15
command: .build/flash/eval-env/bin/python Tools/flash/widen_study.py qualify --pairs 10 --model models/jang-6s --candidate .build/release/slotstream --corpus .build/flash/runs/reviewer-tokenized-development-final --parity .build/flash/runs/widen-parity-qualification-20260915 --output .build/flash/runs/widen-qualify-cooled-20260915
discarded: 'true'
machines: '[[records/machines/local-m5-max-48gb]]'
title: JANG widening cooled qualification thermal stop
tool: Tools/flash/widen_study.py
---
The fresh cooled Plan011 cohort stopped after 44 completed model launches. Its second scalar arithmetic arm began nominal and ended fair despite the thirty-second nominal prelaunch window. Functional execution and process-memory accounting passed; timing eligibility did not. The cohort is thermally inconclusive, not qualified. No failed pair is replaced and no data is pooled with the first attempt.

Raw approved fields and original artifact hashes: [cooled-stop JSON](jang-widening-cooled-stop-20260915.json). Original local evidence remains under `.build/flash/runs/widen-qualify-cooled-20260915`. Sky-blue and Python each completed all ten pairs; arithmetic did not complete its prescribed cohort. Their separate completed-workload summaries cannot pass the overall requirement for all thirty pairs.

Separate resource observations after this stop found Swift compiler processes and active simulator work. Their contribution to the earlier thermal transition was not measured; do not assert causation. Further profiling waits for nominal conditions and leaves other tasks untouched.
