---
type: decision
id: 01m2m8v8sr0xzkemtfw0xgrbfd
created: 2026-09-16T04:50:49.784250+00:00
updated: 2026-09-16T04:50:49.784250+00:00
summary: Reject redundant metadata SIMD and unhelpful lane scratch
decided_on: 2026-09-16
evidence: '[[records/measurements/jang-reader-followup-2026-09-16]]'
reversible_if: A materially different implementation shows reproducible exact reader and matched full-model benefit.
title: Reject redundant metadata SIMD and unhelpful lane scratch
status: standing
---
Do not add a separate runtime metadata SIMD converter or enable the tested lane-local scratch implementation. The production metadata path already vectorizes; the scratch reader screen did not show broad benefit. Preserve all exactness checks, raw timings and the rejected implementation patch. Keep the existing NEON packed backend and scalar/serving defaults.

This closes only these two implementations at the tested scope. A materially different implementation with a reproducible reader benefit and subsequent matched full-model evidence may reopen the decision. Bounded multi-row coalescing and native-bit caches remain separate unevaluated ideas.
