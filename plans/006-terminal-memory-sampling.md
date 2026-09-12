# Plan 006: Capture the final Darwin lifetime peak before reaping a child

## Status and ownership

DONE at reviewed commit `7f63f0b1ebaac1e443defb09ab7e3ad6ccc6b608`. P1; effort M; risk MED. Necessary launcher prerequisite discovered during independent Plan005 review. The prior Plan005 source changes passed review/native/static gates; its reference acceptance remains blocked by this distinct pre-existing Plan002 launcher defect. Reconciled at preserved Plan005 commit `2a25359c701a78c75c3c69cbdb36d0a43d5af4b4`; native source remains frozen during this Python-only fix. Work only in `/Users/imdad/Documents/Codex/2026-09-12/wha/outputs/slotstream-flash-m5`, branch `advisor/001-flash-m5`. Reviewer owns plans; no merge or push.

## Evidence and cause

`.build/flash/runs/reviewer-capture-final/observer-on` wrote a complete valid native report/completion and exited0. The outer launcher failed because `DarwinSampler.sample()` saw current physical footprint0 at process exit. Eight earlier valid samples reached9,626,325,888 bytes. This is not an out-of-memory or native model error.

Root's bounded8MiB Python child experiment in `.build/flash/runs/sampler-exit-recon/samples.json` shows: before reaping, an exited child retains its original positive start identity, positive `ri_proc_exit_abstime`, and final positive `ri_lifetime_max_phys_footprint`, while current `ri_phys_footprint` is0. The terminal peak13,385,992 exceeded the prior live sample13,353,224. After `wait`, the rusage call returnsESRCH. Thus a terminal read can both avoid false failures and capture a peak missed by the last periodic sample.

Current `observe.py` drops exit_abstime and rejects all zero current footprints. `benchmark.py` periodically samples before calling `child.poll()`, while nonsampling loop iterations can reap via poll before any terminal read. Reordering poll first alone would lose the final lifetime peak. Never simply ignore sampler errors or bless a live zero.

## Scope

Only `Tools/flash/observe.py`, `benchmark.py`, `receipts.py`, relevant receipt schema, `test_launcher.py`/`test_receipts.py`, a focused new `test_observe.py` if useful, and a narrow gate test-loader update. Extend capture/cache-study validators only as necessary to require the new terminal policy for newly collected cohorts. No Swift/model/kernel/math/budget/default changes, no model downloads, no profiler, no process shutdown outside owned children.

## Implementation

1. Add a non-reaping child-exit observation on Darwin. This Python exposes WNOWAIT32/P_PID1/WEXITED4/WNOHANG1 but has no os.waitid; libc waitid exists. Use a small ctypes wrapper for waitid(P_PID,owned_pid,...,WEXITED|WNOHANG|WNOWAIT), with the current SDK siginfo_t ABI verified against headers/compiler. `/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/sys/signal.h:178` lists six32-bit fields, pointer, sigval union, long and seven unsigned-long pads. `sys/wait.h:174` documents WNOWAIT. Prefer an available supported wrapper over busy sampling. Validate pid, status/code, errors and ownership. Keep an injectable seam for tests; do not guess ABI or weaken identity checks.
2. Preserve the parent's child until terminal rusage is sampled. Read start/exit absolute times and lifetime peak; accept a zero current footprint ONLY for a confirmed exited, unreaped owned child with matching original start identity, positive exit time >=start and positive terminal lifetime peak. A live zero, invalid identity, missing terminal data, wait/rusage error or sampling failure remains a failure/unqualified result. Terminal samples must be labeled explicitly and cannot masquerade as live observations.
3. Then reap the child and confirm its exit status matches the non-reaping observation. Preserve nonzero exits/signals/timeouts/cancellation. Retain current owned-tree cleanup and PID/start-identity safeguards. No normal poll/wait may reap before the terminal read. Even a very short child needs actual positive measured terminal evidence; never assume zero use.
4. Record an explicit new sampling-policy identifier in fresh receipts, live/terminal sample metadata, and the final peak. Preserve readable historical receipts under their original declared/legacy policy; do not relabel historical sampled peaks as terminal-complete. New-policy memory qualification requires valid terminal evidence and the maximum of all observed live/terminal peaks <=target. Validators recompute summaries from hashed samples and reject missing/duplicate/out-of-order/mismatched terminal data, forged zero samples, or terminal peak above budget.
5. Test focused process lifecycle independently: normal and very short exit, terminal peak higher than the last live sample, terminal-only positive evidence, live zero rejection, zero terminal peak rejection, wrong PID/start/exit identity, error before/during terminal read, exit-between-check-and-read, nonzero/signal exit, cancellation/timeout cleanup, no premature reaping, and legacy receipt readability. Use tiny owned children (<=16MiB), fake samplers/clocks for adverse cases, no memory-pressure generation. A real Darwin probe must prove the child remains waitable until after terminal sampling and becomes unavailable after reaping.

## Verification and handoff

Run `python3 -m unittest discover -s Tools/flash -p 'test_*.py'`, plus a bounded real-child lifecycle gate that writes fresh evidence under `.build/flash/runs/`. Require positive counts and zero required skips. No Swift rebuild is needed for these Python-only changes; preserve current binary/source receipts. Do not run another full model until the reviewer has checked the fix. After review, rerun a fresh complete Plan005 five-process parity cohort and Plan004 six-process collection/analysis serially with unchanged2-second settling and14/17GB targets; preserve all failed cohorts.

Return STATUS/STEPS/STOPPED BECAUSE/FILES CHANGED/NOTES with exact evidence. Commit only after reviewer approval. If terminal peak cannot be obtained safely, keep memory qualification false and report the actual gap; do not revert to a sampled-only success claim. Plan005 stays blocked until a complete fresh cohort passes.
