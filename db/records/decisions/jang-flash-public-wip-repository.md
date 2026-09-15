---
type: decision
id: 01m2jq7fa38afkrhmxbb14f79d
created: 2026-09-15T14:23:40.867759+00:00
updated: 2026-09-15T14:23:40.867759+00:00
summary: Publish the JANG Flash research source as a public WIP fork
decided_on: 2026-09-15
evidence: '[[records/plan/jang-flash-qualification-status]], [[records/measurements/jang-exact-widening-2026-09-15]]'
reversible_if: The maintainer explicitly changes the repository/publication scope; promotion from WIP requires separately completed qualification and release decisions.
title: Public WIP repository for JANG Flash research
status: standing
---
The user requested a new public repository, a prominent WIP label, the measured M5 Max benchmark table, and a link to the exact JANG model. Publish the current research source at https://github.com/android-imdad/slotstream-flash-m5, with `main` as the public branch.

Retain the upstream Slotstream Git history, original MIT license and clear Carlos Galarza attribution. Distinguish the fork's experiments from upstream releases/installers and marketing. Preserve the original upstream remote; publish this work through a separate remote rather than overwriting another worktree's origin.

Keep model weights, native executable/Metal archives, full captures and large local sample artifacts out of Git. Publish portable result extracts, provenance hashes, code, tests, plans and bounded findings. Public source availability does not imply final performance qualification, a release, default activation, native prefetch or ANE execution.

The README must show the existing measured JANG_6S widening result with its actual device/settings/scope and link the model card and pinned revision. Experimental benchmark claims continue to use the canonical measurement/claim gates.
