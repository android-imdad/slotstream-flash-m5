---
type: run
id: 01m2m8v8a81ea1heprv6s19qn5
created: 2026-09-16T04:50:49.288769+00:00
updated: 2026-09-16T04:54:17.430044+00:00
summary: JANG metadata and scratch optimization experiments
binary: a4e385f3401bda9a70bfce7e580316e46e61e005fc3383f13b197e6cfb99cc5f
captured_at: 2026-09-16
command: SLOTSTREAM_READER_SCRATCH_COMPONENT=1 slotstream widening-check --model <pinned JANG_6S> --output <fresh>
discarded: 'false'
machines: '[[records/machines/local-m5-max-48gb]]'
title: JANG metadata and scratch optimization experiments
tool: standalone metadata comparator and bounded original-reader diagnostic
---
The user authorized execution of metadata conversion and lane-local CPU scratch experiments following the NEON benchmark. [Evidence manifest](jang-reader-followup-20260916.json) binds all retained samples and outputs. Neither runtime candidate was admitted. No follow-up full-model cohort was launched.

Metadata: [initial diagnostic](jang-reader-metadata-20260916.json), [separate repeat](jang-reader-metadata-repeat-20260916.json), [compiler/source receipt](jang-reader-metadata-receipt-20260916.json), and [actual baseline readRows disassembly](jang-reader-metadata-production-assembly-20260916.txt). The production disassembly was obtained with xcrun llvm-objdump --disassemble-symbols on the exact readRows symbol in the archived NEON binary identified in the manifest. Reproducible standalone sources are Tools/flash/reader_metadata_bench.swift and reader_metadata_candidate.c at commit 324b26e. These component timings lack interval operating-condition observations; they are not a clean-performance or full-model claim.

Scratch: [native component report](jang-reader-scratch-component-20260916.json), [bound completion](jang-reader-scratch-completion-20260916.json), [decision](jang-reader-scratch-decision-20260916.json), [candidate patch](jang-reader-scratch-candidate-20260916.patch.gz), [build archive receipt](jang-reader-scratch-archive-20260916.json), [focused checks](jang-reader-scratch-exact-20260916.txt), [preflight](jang-reader-scratch-preflight-20260916.txt.gz) and [restored baseline checks](jang-reader-restored-baseline-checks-20260916.txt). The source/binary archives and build logs remain at the local paths in the manifest. Decompress the candidate patch with gzip and apply it to the baseline Sources, build with SLOTSTREAM_BUILD_JOBS=2 make build, then invoke the archived diagnostic command below. The package-private scratch switch affects this reader diagnostic only; it is not a serving or general run option.

Command: `SLOTSTREAM_READER_SCRATCH_COMPONENT=1 <candidate>/slotstream widening-check --model <pinned JANG_6S directory> --output <fresh output>`. Both arms use packed4-to6. Legacy scalar-named timing columns mean per-piece Data/scratch off; packed-named columns mean lane scratch on. Descriptor controls disable filesystem read caching and read-ahead on the opened checkpoint handles. Original source identities and all destination bytes match.

The first isolated build hit an absolute copied ModuleCache path; discarding only that copied worktree cache resolved it. Complete and final incremental release builds succeeded. The candidate was restored out of runtime source after the reader screen; no scalar, serving, planner, quantization or routing defaults changed.

The raw patch and preflight transcript are losslessly gzip-compressed to preserve their exact whitespace. The manifest includes their compressed and uncompressed hashes.
