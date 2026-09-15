# Plan 004 review — host tooling approved; live study blocked

> Historical review/checkpoint at the stated source and evidence versions. Later completion, rejection and remaining work are reconciled in [the current index](README.md) and [JANG Flash findings](../docs/JANG-FINDINGS.md). Preserve the original evidence; do not treat historical active-work wording as a new task.

Preserved commit: `d4bb01fac3a7c04dd5e1d427b090542609a6d5e0`, branch `advisor/001-flash-m5`.

The reviewer read the full implementation/tests and compared CLOCK logic to `SlotPool.ensureCore` and `victim`. Independent verification passed all 84 Python tests and 14 replay self-tests, including full synthetic collection/analyzer validation through relative and absolute paths. Pinned metadata and bounded current headers were independently checked against the completed JANG_6S checkpoint. Scope is six Python/fixture files; no runtime, math, model, dependency or default policy change.

Two review passes addressed complete cohort/evidence joins, source/revision binding, actual equal-budget comparison, bounded trace reads, Swift omitted-nil stats, and both arms' runtime controls/counts. The final correction distinguishes resolved allocation policy from volatile available-memory observations. The historical full-payload model verification is explicitly distinguished from current header/metadata checks.

Live collection was refused before launching a PID at 8.63 GB reclaimable; the independent final preflight measured 8.48 GB, against the approved 17 GB threshold. Thus actual native CLOCK reconciliation and all four window decisions are unmeasured, not rejected or passed. A synthetic rejected result only exercises the test fixture; it is not a research outcome for JANG.

Evidence:
- `.build/flash/runs/window-collection/prompts/sky-blue/untraced/receipt.json`
- `.build/flash/runs/window-collection/failure.json`
- `.build/flash/runs/window-screen/failure.json`
- `.build/flash/runs/reviewer-window-host/preflight.json`

Resume after model preflight passes, using fresh paths:

```sh
python3 Tools/flash/cache_study.py collect --model models/jang-6s --output .build/flash/runs/window-collection-resume-1
python3 Tools/flash/cache_study.py analyze --collection .build/flash/runs/window-collection-resume-1 --output .build/flash/runs/window-screen-resume-1
```

Preserve the original failed attempts. One full model job at a time. Do not weaken the resource gate or infer a byte/time benefit from host tests. At this review checkpoint Plan005 had not been dispatched. It and the later model/corpus work subsequently completed; parent001 now remains in progress for final qualification, as described in the current index. No merge or push.

## Continuation after memory recovery

On the user's request to reclaim and continue, live headroom had already recovered to approximately 24 GB; no process was stopped. Two complete-cohort attempts were started serially with unchanged archived binary/options:

- `window-collection-resume-1`: prose32 and code71 traced/untraced pairs completed and reconciled; math untraced was refused by native JANG planning before model allocation despite host preflight25.58GB.
- `window-collection-resume-2`: prose32/code71 pairs completed again; math untraced produced128 tokens with length finish and127 decode forwards. Math traced was refused by native planning before allocation despite host preflight26.00GB.

Both attempts remain incomplete. Do not combine successful arms across attempts or treat a two-prompt result as this fixed three-prompt study. No window decision was computed. Ten subsequent archived `doctor` invocations accepted the same14GB target, reporting26.0–26.2GB reclaimable and40.2GB Metal working set. The intermittent native observation is not diagnosed; Plan005 adds factual values to the refusal without changing guards.

Evidence: `.build/flash/runs/window-collection-resume-1/`, `window-collection-resume-2/`, and `memory-admission-recon/doctor-series.json`. Plan005 proceeds independently; Plan004 remains blocked on a complete, controlled cohort.

The enhanced Plan005 refusal exposed native reclaimable15.88GB with RAM51.54GB/Metal40.20GB/target14GB, while platform `vm_stat` observed about26GB. Apple's [public XNU host.c](https://github.com/apple-oss-distributions/xnu/blob/main/osfmk/kern/host.c#L539) caches third-party host_statistics responses in a1-second window after a randomized2–10 requests; platform binaries bypass that cache. A stale value from the prior model process is therefore a source-supported explanation, not an installed-kernel trace result. Plan005 adds a fixed2-second settling interval to diagnostic orchestration, retains every admission inequality, and requires fresh complete cohorts to test the mitigation.

## Final development decision after Plan006

Complete cohort `.build/flash/runs/window-collection-terminal/` contains all three traced/untraced pairs, with matching output IDs, exact native CLOCK/source-byte reconciliation, and terminal-qualified memory evidence. Root independently revalidated the full collection. Analysis is `.build/flash/runs/window-screen-terminal/`.

Windows1/2/4/8 reduced pooled decode demand bytes by0.678%,2.076%,0.150%,0.307% respectively. Best was2 tokens, below the predeclared10% admission gate. All windows are rejected for this three-prompt development cohort; no runtime eviction policy was added and no speed gain is claimed. Plan004 is DONE as an evidence-backed rejection, not a global claim about all workloads or memory budgets.
