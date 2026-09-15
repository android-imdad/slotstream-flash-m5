---
type: run
id: 01m2jmx3ge0y6fvjh5ntjm7zpv
created: 2026-09-15T13:43:03.950211+00:00
updated: 2026-09-15T14:07:37.952015+00:00
summary: Portable JANG Flash findings from completed local experiments
binary: Python interpreter sha256:ac60cfe0268614638d0ffa35f3b0284fc7b3a11482723793455e17eeb278509e; original native binary hashes are retained in the exported study metadata
captured_at: 2026-09-15
command: .build/flash/eval-env/bin/python Tools/flash/export_findings.py --output db/sources/runs/2026/09/jang-flash-findings-20260915.json
discarded: 'false'
machines: '[[records/machines/local-m5-max-48gb]]'
title: Portable JANG Flash findings from completed local experiments
tool: Tools/flash/export_findings.py
---
This is a tool-produced portable extract of existing completed experiments, not a new inference run. The original model files, native binaries, captures and full receipts stay unchanged in local `.build/flash/runs/` archives. Large archives are not committed to Git.

The exporter checks completed-report hashes and selected upstream provenance, copies only approved result fields, and omits local usernames, filesystem prefixes and process identifiers. Original source-report paths are repository-relative and their exact byte hashes are retained.

Captured output: [jang-flash-findings-20260915.json](jang-flash-findings-20260915.json). Its source-artifact inventory covers the baseline verification, instrumented M5 observation, exact widening benchmark, oracle stopping decision, cache-window screen, loading experiments and prefetch cost screen.

Original study drivers: `Tools/flash/engine_bench.py`, `evaluate.py`, `cache_study.py`, `read_schedule_study.py`, `whole_expert_study.py` and `prefetch_cost_study.py`. Full methods and exact archive paths are retained in Plans002–016. Each benchmark's completion/verification record remains separate. This portable extract is sufficient to inspect the reported summary, but reproducing full inference requires the original matching archives and checkpoint.
