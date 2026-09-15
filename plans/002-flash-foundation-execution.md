# Plan 002: Implement the Flash experiment launcher and baseline evidence gate

## Current disposition — September 15, 2026

**DONE at bounded scope.** Launcher/archive gates, complete pinned JANG_6S verification and monitored baseline completed; reviewed in 91bbaf7.

[Evidence/current findings](002-review.md) · [Canonical status](../db/records/plan/jang-flash-qualification-status.md). The original execution brief and dated review notes below are retained as history; current work is governed by [the plan index](README.md). Do not restart completed or unselected steps from a historical instruction.

> Executor: follow this bounded implementation plan and verify every step. It executes the foundation of Plan 001 revision 2, not its optional sparse/M5 optimizations. Your reviewer maintains the plan index. Commit only your work in the isolated worktree; do not merge or push. Report measured results and unavailable checks separately.

## Status and prerequisites

- Priority P1; effort M; risk MED; category tests/dx.
- Planned at `93fb512`, 2026-09-12; parent: `plans/001-flash-inference-m5.md` revision 2.
- Execution worktree: `/Users/imdad/Documents/Codex/2026-09-12/wha/outputs/slotstream-flash-m5`.
- Branch: `advisor/001-flash-m5`. The explicit user worktree request overrides the repository's main-branch convention.
- No dependency for host tools and small checks. Complete checkpoint verification/generation must wait for the reviewer's model-slot handoff after the existing download/smoke pipeline ends. This is resource coordination, not a request for new user approval.
- Run first: `git diff --stat 93fb512..HEAD -- Sources Tools Package.swift Package.resolved Makefile` and `git status --short`. Source starts unchanged; copied untracked plans are expected.

## Why and current state

Plan 001 needs reproducible experiments that cannot certify a stale executable, a skipped test, a failed memory sampler or partial model output. Implement this infrastructure before changing inference arithmetic.

The repository uses Swift 5 language mode with Swift tools 6. `Makefile` builds with `make build SLOTSTREAM_BUILD_JOBS=2`, colocates MLX 0.31.1's metallib and writes build identity/source receipts through `Tools/build_identity.py`. Read that tool before validating its actual schema; do not guess keys. The matching mlx-swift pin is 0.31.6.

`Tools/prefill_bench.py` supplies `preflight(needed_gb)`, `vm_snapshot`, and `terminate_child_tree`. Preflight checks the per-user Slotstream model lock and free+purgeable+file-backed RAM; it does not reserve memory. The child must reacquire the native lock/admission checks. `Sources/slotstream-checks/main.swift` supports `--tier t0 --tier t1 --json`; it exits zero with skipped groups. Its JSON contains `checks`, `passed`, `failed`, `skipped`; each CheckReport has items. Use the real types/fixtures when defining receipt validation.

Complete JANG_6S weights are downloading in the original checkout. The new worktree's `models/jang-6s` points there for read-only reuse. Do not call pull without `--verify`, delete weights, repack into that directory, or run a second downloader. Original pipeline status: `/Users/imdad/Documents/Codex/2026-09-12/wha/work/jang-6s-trial/status.json`. The reviewer will send an explicit model-slot handoff; meanwhile implement/test infrastructure.

## Scope

Create only `Tools/flash/{__init__,common,observe,benchmark,gates,receipts,test_launcher,test_receipts}.py` and `Tools/flash/schemas/{receipt-v1,selection-v1}.json`. Additional focused Python test files under `Tools/flash/` are allowed if justified. Create artifacts only under ignored `.build/flash/`. Do not modify Swift, existing Tools scripts, package pins, metallib, model files, server behavior or generated docs. The reviewer may copy plan files into this worktree; do not revert or commit their index edits.

## Step 1 — Implement a monitored launcher and strict receipts

Implement `python3 Tools/flash/benchmark.py --self-test` and `launch --output <fresh-dir> --memory-gb N --max-seconds N -- <argv...>`. Resolve repository paths from the script, never from another worktree's cwd. Use argv arrays, no shell interpolation. Reject reused output directories and invalid budgets/timeouts. The output directory must exist before the child starts, so a child can write its stats inside it.

Launch only an owned child process group; retain PID/start identity. Preflight target+3 decimal GB immediately before real model launch, while fake children/samplers are explicit test-only dependencies. Read `proc_pid_rusage` physical footprint and lifetime peak every 0.5 seconds. On missing/failed sampling, exceeded target, timeout, cancellation or nonzero exit, preserve evidence and fail; do not report zero memory. A child exiting before the first valid sample remains memory-unqualified. After joining, preserve command, allowlisted environment, VM before/after, timed memory samples, stdout/stderr and exit result. Never kill other models/apps or use a memory-pressure generator. Parent buffering is bounded.

Write a versioned receipt and atomic `completion.json` only after closing/hashing outputs. Required identities/statuses from Plan 001 may be unavailable at this foundation stage: store null with an explicit reason and `qualified=false`, never invent a run-set or model hash. Missing required fields for a requested claim are a failure. Ensure interrupted writes cannot look complete.

Tests use tiny subprocesses and injected samplers/clocks. Cover success, nonzero exit, timeout, SIGINT, invalid/fresh output checks, sampler failure, child exit before sample, budget exceedance without allocating real pressure, PID reuse, output hashing and partial receipt rejection. Read the existing cleanup helper and add a safe local wrapper if its behavior is insufficient; do not alter it.

Verify: `python3 -m unittest discover -s Tools/flash -p 'test_*.py'` and `python3 Tools/flash/benchmark.py --self-test`; require positive test counts, zero required skips/failures, and no surviving owned fake child. Tests must assert behavior, not just a printed PASS.

## Step 2 — Archive exact builds and validate required checks

Implement `archive --binary <path> --output <fresh-dir>`: copy executable, colocated metallib and actual build identity/source artifacts into `bin/`; recompute hashes; verify the source snapshot against the build receipt and current tree. Reject changed/missing files, symlink escape and incomplete snapshots. Do not rebuild implicitly or overwrite an existing archive.

Implement `gates.py --stage 0 --output <fresh-dir>`: build with two jobs, validate current build identity, run `slotstream-checks --tier t0 --tier t1 --json`, and require exact names `jang-formats` and `jang-numerics`, nonempty assertion lists, zero required failures/skips. Run named Python test classes LauncherTests and ReceiptTests with a positive discovered count. The general catalogue may include additional groups; record every result. Missing names cannot pass because another group matched a substring. Unsupported future stages fail clearly; do not create placeholder PASS implementations for stages 1–7.

Initialize `.build/flash/runs/selection.json` with schema/version, selected stage/arm IDs, dispositions and explicit pending reasons. Only stage 0 can have implementation evidence in this task. Keep future stages pending/not yet evaluated, not falsely rejected. Future run-set support will build on these schemas.

Add mutation tests: stale source, changed executable/metallib after receipt, missing artifact, all checks skipped, required name absent, zero assertions, malformed JSON and zero discovered Python tests must fail. Archive tests use tiny fixtures reflecting the real receipt schema rather than heavy model runs.

Verify: `python3 Tools/flash/gates.py --stage 0 --output .build/flash/runs/stage-0-checks` exits 0 with a fresh native build and required checks. Then `python3 Tools/flash/benchmark.py archive --binary .build/release/slotstream --output .build/flash/runs/baseline` exits 0 with matching hashes. Use new suffixed directories on a justified rerun; never overwrite failed evidence.

## Step 3 — Capture the real baseline after model-slot handoff

Do not start until the reviewer confirms the original pipeline is finished and no model process owns the slot. If handoff has not arrived, commit completed host infrastructure, report foundation-ready/full-model-pending, and remain available for continuation; the parent plan stays IN PROGRESS.

Verify the pinned checkpoint with `.build/release/slotstream pull JANG_6S --dir models/jang-6s --verify`, preserving the receipt. This reads existing weights only. Run doctor with `--model models/jang-6s --memory-gb 14 --max-context 2048 --mtp off --vision off --json`. The JANG revision must be `3781190c6bbdf0a7637beda49ba179822612058a`.

Through the monitored launcher run `jang-check --model models/jang-6s`, then `run --model models/jang-6s --memory-gb 14 --max-context 2048 --mtp off --vision off --prompt "In one sentence, explain why the sky is blue." --max-tokens 48 --greedy --sample-footprint --stats-json <fresh-run-dir>/stats.json`. Launcher target 14 GB, timeout 900 seconds, fresh preflight >=17 GB. Preserve IDs, stdout, timings, actual physical peak and stderr. Check success, finite/nonempty output, complete stats and consistent identities. A sensible sentence is a smoke result, not proof of full model correctness.

If this unmodified inference baseline fails, stop model experiments and report the precise failure. Do not fix Swift or weaken the memory target under this task.

## Done and STOP conditions

Foundation done: meaningful host tests and strict stage-0 gates pass, archive hashes match, no source outside scope changed, and a logical worktree commit exists. Stage 0 fully done additionally requires complete-checkpoint verification and a monitored successful generation. Report those separately.

Stop affected work on source drift, unknown receipt semantics, unsafe process ownership, repeated failed verification after a reasonable fix, scope expansion, unavailable headroom, or a broken complete-model baseline. Wait for handoff without blocking independent host work. Do not merge/push, change original checkout files, or update the index. Report exactly STATUS, STEPS with commands/results, STOPPED BECAUSE if needed, FILES CHANGED, NOTES (commit, paths, limitations).
