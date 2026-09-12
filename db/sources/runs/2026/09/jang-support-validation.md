---
type: run
id: 01m2aemp5xae39a8gfq19k037t
created: 2026-09-12T09:19:41.245719+00:00
updated: 2026-09-12T09:20:16.249937+00:00
summary: Passing source build, JANG numerical and streaming fixtures, regression catalogue and static transport gates; full-model inference untested.
binary: sha256:0014844a367f513aa3dac20906f592e1af1b77df4ae8dfa3c48e2c18370c067b
captured_at: 2026-09-12T09:19:41.240884+00:00
command: make build; slotstream jang-check; slotstream-checks; context_proxy.py; static_gates.sh
discarded: 'false'
machines: '[[records/machines/local-m5-max-48gb]]'
title: JANG source implementation validation
tool: Slotstream native catalogue and acceptance tools
---
Functional validation of the local JANG source implementation. No complete
checkpoint was downloaded and no whole-model generation was run. These are
component and software-contract results, not throughput or KL measurements.

Commands, run from the checkout:

- `make build SLOTSTREAM_BUILD_JOBS=2`
- `.build/release/slotstream-checks --tier t0 --tier t1`
- `python3 Tools/context_proxy.py --out ../../work/jang-context-acceptance`
- `Tools/static_gates.sh` with the pinned dbmd executable on PATH
- `python3 Tools/jang_fixture.py --tier 4M --output ../../work/repro-jang-4M`
- `.build/release/slotstream jang-check --model ../../work/repro-jang-4M`
- The same fixture/streaming commands for tier 6S.
- `.build/release/slotstream doctor --model JANG_6S --json`

The original headers of every indexed shard are retained in the fixture pack;
expert payloads in the sparse fixture are synthetic. Separately, rows.json
contains original weight-row bytes fetched from pinned public JANG shards.
The numerical comparison uses native MLX as an independent decoder/QMM oracle.

Captured evidence in this directory:

- jang-build-verified.log and jang-build-identity.json
- jang-acceptance-catalogue.log
- jang-repro-4M.log and jang-repro-6S.log
- jang-context-acceptance.json
- jang-acceptance-static.log
- jang-fixture-hashes.txt
- jang-readiness.json and jang-validation-summary.json

Local filesystem prefixes were redacted from the public text logs. The binary
receipt contains relative source paths and exact hashes; current source and
binary bytes were checked against it. Other application activity was not
stopped, and no clean-performance claim is made.

An earlier static run failed its existing disk-full fixture: more free disk
made it exceed the compressed-manifest object limit before reaching the disk
guard. jang-static-gates.log preserves that failure. The fixture now injects a
small disk budget through the unchanged production guard. Subsequent full
static acceptance passed. Intermediate build attempts are not acceptance.
