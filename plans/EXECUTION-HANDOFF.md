# Completed checkpoint — whole-expert layout screen

The user requested `commit and proceed`. Completed work through Plan 013 is now
committed locally as `65d7e82` (implementation) and `ba18325` (reports/baseline),
with no push or default activation. Large baseline executable/Metal payloads stay
in the local ignored archive; committed manifests retain their original hashes.

Plan 014 completed and was rejected (8.48% longer paired total reader time).
All native equality and host evidence checks passed; no model process remains.
Evidence is `.build/flash/runs/whole-expert-20260915`. The next bounded experiment
will test original source-native records with no expanded-byte disk overhead.

Plan 014 scope: compare the existing expanded whole-expert
layout with original JANG reads using exact packed4-to6 widening, on at most
256 MiB of original regions/derived payload. It remains diagnostic-only pending
the predeclared screen. Read `plans/014-whole-expert-layout.md` for controls and
admission criteria. Do not change completed Plan 013's rejected disposition.

Historical statements below about uncommitted work reflect their earlier dates.

---

# Latest checkpoint — exact read-scheduling experiment

September 15 continuation is complete at Plan 013's bounded scope. The user asked
for continued optimizations with a benchmark before the next change. Balanced
SSD demand-read scheduling was implemented as a package-only diagnostic switch,
then rejected by its completed eight-pair-per-case component benchmark: reader
calls at default queue depth 32 took about 40% longer, with exact tensor bytes.
No generation/capture/library option was added. No full-engine run was admitted,
and no subsequent optimization began. See `plans/013-balanced-read-scheduling.md`.

Evidence: `.build/flash/runs/read-scheduling-20260915`, owned study session 14387
completed exit 0 with a rejected performance verdict. Build v2 passed, 40 native
T0 checks, 215 host tests and full repository static gates passed. All owned work
is finished; no model or test process remains. Source remains local and uncommitted. Preserve
all earlier archives, including the successful 24 GB packed-widening benchmark.

The oracle's failed neuron prerequisite makes its predictor/ANE experiment
not_selected. Do not resume the user-stopped four/two-block runs. Prefetch remains
a replay-only screen with no native timing admission. Historical notes below are
superseded by this checkpoint where their active-work language conflicts.

---

# Current continuation checkpoint — September 15, 2026

## Active oracle execution (supersedes implementation status below)

**FINAL CHECKPOINT: no work is running.** Cooled benchmark session 95642 completed successfully; no model processes remained. All nine pairs / eighteen arms passed exact output/work equality, timing eligibility and the 24 GB ceiling. Revalidated evidence and final tables are in `plans/009-oracle-execution.md` and `.build/flash/runs/engine-benchmark-24gb-cooled-20260915/{report,verification}.json`.

Final scalar → packed medians: explanation 3.199 → 6.983 tok/s; coding 3.331 → 7.409; reasoning 3.272 → 7.266. First text roughly halved; largest packed physical peak 20.685 GB. These are about 2.2× scalar throughput, but below the separate upstream 14–15.9 tok/s targets. Neuron masks remain disabled after eight/six failed fidelity. Four is incomplete at user direction and two was not run. No native prefetch/ANE or further optimization was started. All changes remain local and uncommitted; no merge/push/default activation.

The chronological notes below are preserved, but their running-session wording is superseded by this final checkpoint.

**Latest steering and active work:** The user explicitly stopped tests below six blocks. Session 81031 and its owned evaluator/model were stopped; four-block testing has 512 validated positions and no complete verdict; two never started. No model process was left behind. `oracle-campaign-v2-20260915/operator-decision.json` records the revised stopping decision; do not resume four/two or describe the original five-cohort protocol as complete. The interrupted fourth-setting receipt reflects this intentional stop.

**Current benchmark is running in unified exec session 95642.** Log: `.build/flash/engine-benchmark-24gb-cooled-20260915.log`; output: `.build/flash/runs/engine-benchmark-24gb-cooled-20260915`. It requires 30 seconds of nominal Foundation observations before each arm (five-second polling; waits excluded from inference timing). Wait for completion, inspect all paired results/timing eligibility, update the execution record and report the benchmark before starting any new optimization.

The earlier session 85424 was deliberately stopped after thermal state became `fair`. `.build/flash/runs/engine-benchmark-24gb-20260915/operator-stop.json` preserves its ineligible, incomplete status; never mix those arms with the fresh cooled cohort. No model was left behind. Reclaimable memory before the initial launch was 30.61 GB, above the 27 GB requirement. The cooldown change passed its tests; the host suite now has 211 passing tests.

**The completed oracle harness was archived before benchmark-only edits** at `oracle-campaign-v2-20260915/harness-snapshot`, with exact file hashes matching `frozen.json`. The live harness now differs because `engine_bench.py` gained cooldown handling, so do not call the live `validate_freeze` on the completed oracle campaign and mistake that intentional later edit for corrupted inference evidence. Original frozen inputs/reports and the operator decision remain untouched; use the archived harness for a historical source audit.

The user authorized the next step and then explicitly requested a benchmark after this stage **before another optimization**. The versioned charged oracle capture adapter and complete-cohort evaluator are now implemented. `plans/009-oracle-execution.md` describes the contracts and benchmark boundaries.

Frozen campaign: `.build/flash/runs/oracle-campaign-v2-20260915`. **Do not edit Sources, Tools/flash or its schemas while the campaign runs**: all configurations, model header/descriptor identities, input bytes, native build and host harness are frozen. The earlier v1 freeze was withdrawn before inference; do not combine it with v2.

- Complete dense-ten passed all 14 shards / 896 positions with exact full logits, routes, retained state and token IDs. Mean/p99 KL 0, top-1 1, PPL ratio 1. Log: `.build/flash/oracle-dense-cohort-20260915.log`; report: campaign `retained-10/report.json`.
- Eight-block cohort completed and failed the fidelity gate: mean KL 0.2035545372, p99 2.2377849865, top-1 agreement 0.8392857143, PPL ratio 0.9962457466. Log: `.build/flash/oracle-retain8-20260915.log`.
- Six-block cohort completed and failed: mean KL 0.4098427446, p99 3.8356715824, top-1 agreement 0.7433035714, PPL ratio 0.9585999170. Log: `.build/flash/oracle-retain6-20260915.log`. The PPL ratios improving slightly does not override the original KL/top-1 fidelity gates.
- **Active unified exec session 81031** is a serial shell loop for retained counts 6, 4, 2, then `evaluate.py summarize`. Six is complete and four is starting at this checkpoint. Logs are `.build/flash/oracle-retain{6,4,2}-20260915.log`; final summary output is `.build/flash/oracle-decision-20260915.log`. **Do not launch another cohort concurrently or repeat stages that this loop owns.** Poll/wait this session until complete. All partial cohorts are required even if the strongest partial policy fails; do not fill failed cohorts from replacement shards.
- Final build passed (`oracle-build-final-20260915.log`); 209 host tests passed; native T0/mask checks, six CLI preallocation rejection cases and repository static gates passed (`oracle-static-20260915.log`). A separate 64-position dense smoke passed at 9.70 GB peak before the complete campaign.

After quality is resolved, run the matched performance checkpoint before new optimization work:

```sh
python3 Tools/flash/engine_bench.py --reference .build/flash/runs/diagnostic-reference-corpus-v2/bin/slotstream --candidate .build/release/slotstream --model models/jang-6s --baseline baselines/upstream-0.2.18-24gb-20260915/baseline.json --output .build/flash/runs/engine-benchmark-24gb-20260915
```

This compares immutable scalar JANG with current exact packed widening on the saved three prompts, three paired rounds each, 24 GB, 32K context, greedy seed 42, 128-token limit, prefix cache off. The benchmark driver records the explicit prefix-cache environment in receipts. It checks exact output IDs/work and matched controls, preserves timing-ineligible runs, and labels the persistent-server upstream quant/MTP/lookahead result as a separate target. The diagnostic oracle is disabled in this runnable-engine benchmark; do not claim it measures a sparse-reader speedup. If a partial mask passes teacher-forced quality, free-generation and trained-predictor gates still precede sparse runtime or ANE execution.

---

This section supersedes the historical coordination notes below. There are no active delegated workers from that historical session to resume.

The user explicitly means **neuron skipping, SSD prefetching and Neural Engine execution**, not another widening qualification pass. Continue the staged quality-preserving implementation in `/Users/imdad/Documents/projects/slotstream-flash-m5` on the existing `advisor/001-flash-m5` branch. Preserve the pre-existing widening and upstream-baseline work. No merge, push or default activation was performed.

Read `plans/009-foundation-review.md` for the implemented scope, results and exact next step. New files implement original down-column norms, deterministic block scoring, a package-only default-nil neuron-mask hook, complete development calibration and a causal prefetch replay with charged staging. The public oracle capture adapter does not exist yet; do not describe the internal mask hook as qualified sparse inference. A useful quality-passing trained predictor still gates Neural Engine work.

Completed evidence:

- `.build/flash/runs/neuron-norm-full-20260915`: all 24,576 original experts, 60 MiB norm table, measured 421,462,904-byte terminal peak under a 1 GB target.
- `.build/flash/runs/neuron-calibration-launch-20260915/study`: 896 positions / 430,080 routed examples from the frozen complete development cohort. Only 6 exact zeros across 275,251,200 hidden values. Best 8-block policy omits about 16.2% of the contribution score; that does **not** determine full-model quality.
- `.build/flash/runs/causal-prefetch-budgeted-20260915`: exact trace demand reconciliation, 20 slots charged for staging/insertion. Neither forecast is a qualified runtime prefetch or measured speedup.
- `.build/flash/runs/neuron-foundation-final-archive-20260915`: source-bound final native binary archive. The full norm export used its earlier archived exporter; a final-build sample reproduced the exact same norm payload hashes.
- Final release build: `.build/flash/neuron-mask-build-20260915.log` passed. Host tests: `.build/flash/neuron-host-tests-final-20260915.log`, 201 passed. Native: `.build/flash/neuron-final-t0-20260915.json`, 39 passed; `.build/flash/neuron-final-mask-20260915.json`, mask check passed with 7 assertions. The original JANG numeric check also passed.
- Final repository static battery: `.build/flash/neuron-static-final-20260915.log`, **STATIC GATES PASS**, including standalone memory, transport, planner, override and installer checks.
- `.build/flash/runs/neuron-default-parity-20260915`: the final archived binary with masking disabled matched the preserved reference's full logits, routes, retained-state identity and token IDs at all 12 natural development positions. Effective plan and numerical controls also matched. This is a default-path regression check, not partial-mask quality qualification.

The old checkout path is a compatibility directory symlink to this checkout so immutable artifacts retaining absolute paths remain readable. The prompt validator now compares the same resolved executable while still checking its exact hashes/archive. Generated compiler caches from the old absolute path were moved aside. `dbmd` was installed using the repository's pinned installer for static gates. No model weights or original evidence files were changed.

The next work is Plan 009's versioned oracle capture adapter, single norm-table ownership with an additional 80 MiB charge, mask artifacts and lifecycle, followed by dense-ten parity and the complete frozen partial-mask quality cohorts. Free generation, training, sparse reads and ANE remain conditional. Keep this checkpoint honest when updating it; the older notes below do not supply current approvals or worker ownership.

---

# Historical execution checkpoint

Current user direction: commit completed work and continue pending implementation. The later status question was answered candidly: no measured inference speedup yet; latest verified JANG_6S baseline approximately2.4–2.6tokens/s at14GB. Continue the active work; no user confirmation is needed.

Worktree `/Users/imdad/Documents/Codex/2026-09-12/wha/outputs/slotstream-flash-m5`, branch `advisor/001-flash-m5`. Root is the improve advisor: edits plans and runs verification, never source implementation. Workers own source; exact scoped commits only after review. No merge/push. Original model/checkpoint and reference archives remain read-only.

Committed milestones:0b032fa natural-text capture/tokenizer bridge;f61730d corresponding plan/index;bc185dfe4d5ab8c04f107bef90be47d7952ffc69 complete-prompt tokenizer. Earlier milestone history remains in plans002–007. Plan010 is independently approved and committed; canonical prompt archive is `.build/flash/runs/prompt-tokenizer-v1`, binarya9e21a9aa0be99a74d620bf9270f876e60dfb8a5f4cf3170546666c280fe2609. Original corpus-v2 reference `.build/flash/runs/diagnostic-reference-corpus-v2` binaryd3b0e7ffbcaf89186a8097a4dcd65493c595cf506adf7ef5098d787fc4a9c095; v1 diagnostic reference remains separately untouched.

Plan008 source is uncommitted. Its first corpus was rejected for filler/partition errors; the replacement preparedv5 now has196documents/7672positions (896training,896development,5880qualification) and actual1164 final cases. Raw public source/scorer downloads and pinned NumPy wheel live under `.build/flash/corpora/source-recon`; external dataset text is ignored. The eval environment `.build/flash/eval-env` is Python3.12.14/NumPy2.3.5 and verifies actual installed wheel files before import.

Plan008 reached its two review rounds and was blocked by a reproduced archived-harness mutation acceptance. Narrow Plan012 fixed this; root approved after33 tests and real copied-cohort mutations rejected both ordinary code drift and a rehashed forgery. A final small revision rejects nested snapshot.json extras and symlinked snapshots. Reopen008 only for the remaining actual integration, not another broad review loop. The two implementations share files, so commit their source together after integration.

Root actual tokenizer evidence: `.build/flash/corpora/reviewer-teacher-v5` (196docs,7672positions,168shards,max443fullIDs,peak382895016B); `.build/flash/corpora/reviewer-scored-prompts-v5` (1164docs,104443IDs,longest407,peak706446488B). Both use immutable approved native archives. Final suite freeze/validate passed at `.build/flash/corpora/reviewer-quality-suite-v2.json`, with preparedv5 and producer snapshot `.build/flash/corpora/quality-producers-v4`. Final repo `Tools/fixtures/flash/suite.json` was generated and validated; SHAee44f11608a11d092b81e8799ff154d87f2924ada5844906014fec917031a7d6. Root approved008+012 and asked widen_executor to commit exactly14 quality files, keeping its011diff unstaged.

COMPLETED ROOT MODEL JOB: unified exec session21424 ran `metrics.py capture-cohort --suite .build/flash/corpora/reviewer-quality-suite-v2.json --binary .build/flash/runs/diagnostic-reference-corpus-v2/bin/slotstream --mode reference-on --split development --output .build/flash/runs/reviewer-quality-development-final`.14shards/896positions, serial14GB target/17GBheadroom/2ssettle/terminal-before-reap. All14shards/896positions completed with no failure,maxpeak9808909376B. Complete validation and self-comparison passed (mean/p99KL0,top1=1,PPLratio1). Model slot was returned to widen_executor. No source/model quality claim beyond alignment.

Workers: `/root/quality_executor` exhausted context after008/012 and is idle; `/root/flash_executor` exhausted its accumulated context and is idle; `/root/prompt_executor` completed010 and can do a small scoped commit/handoff if its context permits. `/root/widen_executor` actively implements011 but is currently paused for the root model/harness window, allowing Swift-only edits/review. Return that worker the slot when21424 ends. It owns AffineRow/ExpertStore/Engine/Model/Run/Flash/widening diagnostics and Tools/flash/widen_study/capture.py extensions; no quality corpus/metric/env edits.

Plan011 is the next direct speed experiment, before neuron oracle009. It adds default-off immutable `AffineWideningPolicy` with scalar/packed4-to6, exact two-byte→three-byte specialization, JANG_6S-only validation and Run/diagnostic opt-in flag. Generic/overlap fallback preserved. Source implementation is uncommitted/unqualified. First synthetic controls passed exhaustive65536 inputs; scalar3.053ms versuspacked0.208ms single-worker (conversion only). Actual original-reader outputs matched exactly for layers0/5/22 through batch/runs paths. Initial model component physical-read coverage was unknown; worker is strengthening balanced pairs and proc_pid_rusage disk bytes plus external512MB receipts. NO full-model011parity or performance yet. Root must review source/components before allowing those runs. `/root/oracle_plan_review` is doing a bounded read-only cold review of011 and its benchmark validity.

Read-only widening recon is `.build/flash/runs/widen-profile/report.json`: historical trace120444calls/197.3billion codes; scalar2.730ms single-worker. This is not removable wall time or a TPS gain. A separate prefetch recon `.build/flash/runs/heuristic-prefetch-recon` estimates previous-token forecasts cover2.23% of missed bytes; recent8frequency covers14.21% but adds substantial unused I/O. Neither runtime prefetch nor sparse neuron loading nor ANE execution is implemented.

Next actions: finish/verify21424; generate final suite fixture; get scoped008+012source commit and root plan commits; return model/harness slot to011; review complete011diff/components; run archived-reference exact parity and exploratory paired inference, then the predeclared qualification cohort if promising. Preserve every failed/rejected attempt. Keep all new modes off by default. No latest-quant/absolute-BF16-quality claims.
