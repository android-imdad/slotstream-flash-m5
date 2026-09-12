# Plan012: Verify archived harness bytes and immutable receipt joins

## Status and scope

DONE at bounded scope after one revision round. Root independently passed33 host tests and a copied real-cohort mutation test: original copied cohort passed; changed archived capture.py failed; changing its file hash plus cohort self-hashes while preserving the original launcher digest also failed. The original evidence was untouched. The final small revision rejects nested snapshot.json artifacts and symlinked roots/manifests. Source is committed with008 after its remaining real-tokenization/capture integration, not separately from its dependent uncommitted modules.

P1; effort S; risk MED. IN PROGRESS; bounded successor repair for the blocked008 host implementation after its two formal review rounds. Same isolated checkout/branch atbc185df plus uncommitted008 source. Own only metrics.py, corpus.py, their tests and a small shared snapshot-validation helper/schema if needed. No capture.py, Swift, model, optimization, dataset content or evaluator arithmetic changes. Root owns plans; widening executor owns its separate source/capture/performance files. No source commit or model run until review.

## Confirmed defect

The reviewer cloned `.build/flash/runs/quality-diagnostic-control-v5-off`, rebound the clone's harness path and its index/completion hashes, and validated the copy. Appending a comment to only the copy's archived `Tools/flash/capture.py` was still accepted by metrics.validate_cohort. See `.build/flash/runs/reviewer-quality-harness-mutation/review.json`. The original capture/archives remain untouched. metrics.py checks snapshot.json and its declared digest map, but never hashes the actual archived files.

## Required repair

1. Add one bounded snapshot validator shared where practical. Require the exact format/fields, a bounded nonempty expected inventory, safe relative paths, regular files contained in the snapshot and no symlinks/extra/missing files. Stream and verify every declared file's actual SHA256 (and byte count where declared). Check the snapshot itself is hash-bound to its owner. Execution snapshots must exactly match the launcher's recorded harness map and reside in the owning cohort. Producer snapshots must contain the exact supported producer/fixture/schema file set, not any arbitrary nonempty map. If a new helper becomes part of the producer, version/freeze that inventory explicitly.

2. Validate the complete immutable five-file native build bundle with common.validate_build_identity(historical=True), then join the launcher's recorded executable SHA256/bytes to the actual immutable binary and native source identity even when a preselected archive is passed. Preserve the invocation/resolved paths as historical evidence; do not make a valid archived cohort depend on the old live path still existing. Never search arbitrary directories for a convenient match or substitute a new reference.

3. Apply validation at both creation and later reads. A successful freeze does not allow later code-file mutation, missing files or an edited receipt to pass. Keep data-producer, execution and analysis identities separate so unrelated future working-tree edits do not invalidate archived evidence. No broad bypass or weakening of current model/corpus/row checks.

4. Preserve all current prepared suites and capture attempts. Snapshot versions may be regenerated in new directories after code is fixed; do not overwrite accepted archives or mutate original evidence to make tests pass.

## Tests and evidence

Use small synthetic snapshots for ordinary tests. Cover a valid snapshot, changed declared file with unchanged snapshot metadata, deleted/extra file, traversal/symlink, omitted producer member, malformed hash/byte count, changed snapshot owner hash, wrong launcher executable hash/bytes and a valid cohort after its original live executable disappears. Test adding an unrelated current tool leaves archived validation valid. No private model requirement for normal unit tests.

Repeat the exact reviewer mutation on a NEW copy of the existing12-row diagnostic cohort: it must pass before mutation and fail afterward, while the original still passes. Verify the immutable native bundle and all recorded harness files directly. Run the explicit pinned-environment evaluation gate and ordinary dependency-free Flash gate with structured nonzero required test coverage and no required skips.

Return the focused diff, command results and exact evidence paths for root review. On approval, root will reconcile/reopen008, run actual prompt/teacher tokenization, and only then allow its final896-position reference cohort. This repair supplies no model-quality or speedup claim. No merge/push/default activation.

Source commit: `aa9b771be2ad737afe7a4f9efadeae946df76cd5` (Freeze quality corpus and validate complete capture evidence), containing exactly the14 approved quality source/fixture/schema/test files. No model artifacts, widening source or plans were included.
