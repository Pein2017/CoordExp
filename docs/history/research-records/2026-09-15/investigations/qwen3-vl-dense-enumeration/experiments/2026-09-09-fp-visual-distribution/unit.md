---
title: Visual distribution of annotation-relative FP after Source256 RLOO round1
description: Exact FP structural census plus seeded original-image and crop review before choosing supervision semantics.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: authorized_separately
unit_id: 2026-09-09-fp-visual-distribution
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-09-09
---

The user requests visual checking of FP distributions using full-image context
and crops before modeling TP/FP/FN supervision. This narrows the current
two-route modeling/Pro preparation to resolving the evidence semantics first.
No model training, new GPU inference, official GT changes or shared skill edits.

Source is the accepted [round1 greedy read](../2026-09-09-round1-greedy-realization/results.md):
256 images,1955 GT,1262 TP50 and1094 annotation-relative FP50. The exact global
category-consistent matcher owns sample membership; visualization colors do
not replace it. All predictions, including caps and repetitions, remain in
the population. Root owns source identity, sampling, integration and records;
three visual readers own disjoint image batches and candidate review rows.

Frozen output root:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-09-fp-visual-distribution`.
`summary.json`, `inventory.jsonl`, `sample.jsonl`, `prepare_visual_audit.py` and
`review-contract.md` own the exact preparation and visual labeling convention.
The native renderer supplies full-image GT/prediction overviews; deterministic
pixel crops add uncluttered target context without changing source pixels.

Exact structural strata (mutually exclusive in this priority order):
484 strict repeat FPs;269 non-repeat FPs near same-category GT (IoU>=.1);
14 remaining FPs overlapping another category's GT (IoU>=.5);327 weak-GT-
relation FPs. These are not hallucination or localization ground-truth labels.
Four capped images contain511 FP, overlapping rather than adding to the
repeat count.

Fixed seed20260909 initially produced uniform without-replacement samples per
stratum10/20/14/20. The user's subsequent ruling supersedes duplicate viewing:
IoU>.95 directly classifies all484 strict-repeat FP, with no visual verification
required. The active visual population is610 non-repeat FP; retain the existing
20/14/20 samples (54 total), without resampling. Extra capped-image-only checks
are cancelled. The original64-item packet remains provenance and
`active-review-scope.json` owns the narrowed54-item review. Unknown judgments
remain in the denominator. Report exact structural counts separately from
sampled visual judgments. Any weighted extrapolation must state sampling
weights, uncertainty and non-certified model-review status, not claim a full
pixel census or create training labels.

Review entity existence, category, geometry and annotation relation separately;
use one explicit primary reason for counting. Unmatched means neither phantom
nor automatically real. A crop showing an adjacent instance or a multi-instance
box must not be collapsed into generic hallucination. Inspect full context and
crop for each sampled prediction; root rechecks load-bearing ambiguity once.

Stop at54 completed non-repeat reviews, the automatic484-repeat count, a
validated summary and bounded implications for supervision/Pro questions. Do not launch
training, change a reward, relabel GT, expand to an exhaustive visual census or
modify shared runtime/skills automatically.

The bounded audit is now lead-accepted. All54 non-repeat cases have visual
reviews, exact source/image hashes and complete IDs; root inspected14 of them
and recorded five explicit adjudications while preserving original opinions.
The [results](results.md) separate the484 automatic duplicates from visual
findings. No official GT or training signal was changed.
