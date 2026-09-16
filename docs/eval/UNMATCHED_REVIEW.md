---
doc_id: docs.eval.unmatched-review
layer: docs
doc_type: reference
status: canonical
domain: eval
summary: Project-wide TIDE-aligned unmatched terminology and conservative detector-assisted review.
tags: [eval, unmatched, tide, review, proxy]
updated: 2026-09-16
---

# Unmatched analysis and review

## Scope and authority

Effective2026-09-16, per the user's explicit project-wide registration request.
Future unmatched investigations use this vocabulary, separating reference-relative
matching, physical evidence, geometry and learning admission. This document owns
interpretation and review policy; it does not claim that existing producers emit
all these fields, replace the production evaluator, or rewrite frozen results.
The [metric interpretation](INTERPRETATION.md) and each run's frozen contract
continue to own their declared matching and denominator semantics.

Current user preference: Co-DETR is the primary screening proxy; no default
Qwen-VL semantic judge. Residual ambiguity goes to bounded lead/subagent visual
review. Prefer deferring a label to training an incorrect object/category/box.

## 1. TIDE reference-relative vocabulary

Use the official [TIDE taxonomy](https://dbolya.com/tide/) and
[implementation](https://github.com/dbolya/tide/blob/master/tidecv/quantify.py)
for exact TIDE reproduction. Record reference version/digest, category mapping,
coordinate frame, IoU thresholds, prediction scores/order, max detections and
crowd/ignore handling. Native TIDE uses score-ranked matching; our research
cardinality-first matching is not interchangeable with it.

| Code | Reference-relative meaning |
|---|---|
| `Cls` | Wrong class with sufficient localization to a reference object. |
| `Loc` | Same-class object with inadequate localization. |
| `Both` | Combined class/localization error under TIDE's remaining-case rules. |
| `Dupe` | Additional prediction for a reference already consumed by another detection. |
| `Bkg` | Insufficient overlap with any eligible reference, under the background threshold. |
| `Miss` | Reference-side missed object not accounted for by the other error corrections. |

Matched, ignored and technically invalid are separate outcomes, not extra TIDE
error classes. `Miss` belongs to reference objects, not unmatched prediction rows.
Do not equate the number of `Miss` errors with every definition of total FN.
Do not report TIDE dAP/AP from a vocabulary-only analysis or invent comparable
confidence scores for autoregressive rows. If using a different assignment or
no score-ranked AP calculation, label it **TIDE-aligned diagnostics** and publish
the actual rule. Exact software output should be identified as `official_tide`.

Always distinguish `reference_kind=trusted_annotation` from
`reference_kind=detector_proxy`. A `Bkg` label means **unmatched to that reference**,
not proof that no real object exists. A detector-only `Miss` is a candidate model
omission relative to the detector, not a newly verified GT obligation.

## 2. Physical and geometry evidence are independent

These are CoordExp review extensions, not additional official TIDE classes.
Multiple tags may coexist. Reviewer confidence and evidence source are separate
from the label. A low-confidence guess must not become a definite positive or
negative merely to close an audit.

| Axis | Vocabulary / meaning |
|---|---|
| Entity | `verified_real`, `verified_absent`, `insufficient_visual_evidence`, `unresolved` |
| Category | `verified_correct`, `verified_wrong`, `unknown`, `out_of_scope` |
| Relation to annotation | `known_owner`, `unlabeled_real`, `annotation_defect`, `unresolved` |
| Geometry | `acceptable`, `single_owner_localization`, `cross_instance_box`, `extent_convention_ambiguity`, `unresolved` |
| Uniqueness | `distinct_owner`, `duplicate_owner`, `unresolved` |
| Technical debt | `malformed`, `invalid_geometry`, `coordinate_transform_error`, `truncated`; keep outside physical labels |

- `unlabeled_real` requires a verified independent owner absent from the named
  trusted ledger, not merely an IoU miss. Compare category/extent/assignment
  alternatives before declaring a new owner.
- A real unlabeled owner can still have a bad box. Physical existence and class
  support do not make its emitted coordinates trainable.
- A box containing multiple visible instances is not automatically
  `cross_instance_box`: it may be a legitimate box for one overlapping owner.
  Use that tag when the extent combines different owners or cannot be attributed
  to one owner. Repeated class names alone are not evidence.
- For single-owner localization, record which edges over/undershoot in original
  pixels and relative to the independently supported owner's width/height. Do
  not describe an x2/y2 mismatch as a proven autoregressive binding mechanism.
- `duplicate_owner` is physical identity repetition; IoU>.95 is only one
  mechanical candidate flag. Lower-IoU predictions may repeat the same owner,
  while distinct occluded owners can overlap heavily.
- Extent ambiguity includes visible/amodal, part/whole, face/person and group/
  instance annotation conventions. Name the intended convention rather than
  forcing an annotation inconsistency into model localization failure.
- Background false positive requires affirmative visual evidence, not scene
  improbability or detector non-detection. Tiny/far/occluded objects with only
  prior-based support are `insufficient_visual_evidence`, not verified absent.
- Judge visibility at the actual model-input resolution. Enlarging a crop adds
  no source information. Record privileged higher-resolution evidence separately.
  Exclude prior-only targets from future trusted teachers/required recall under
  the user's visibility ruling, with versioned reasons; no blanket area cutoff,
  deletion of raw GT, or retroactive favorable denominator change.

## 3. Review record and disposition

New review artifacts should preserve these concepts (field spelling may follow
the owning existing schema): image/proposal IDs; raw span or output digest;
checkpoint/policy; original-pixel bbox and image dimensions; reference identity;
TIDE mode/code or candidate codes; physical/category/geometry/uniqueness axes;
candidate owner IDs; evidence paths; reviewer and confidence; decision and reason.
Multiple diagnoses remain tags/axes, not a forced mutually exclusive pie chart.

Keep proposal-level frequency separate from unique-owner counts. Preserve
`candidate`, `HOLD`, `lead-accepted` and rejected dispositions; do not treat every
historical HOLD as a reviewed hard case. Fresh verified extras enter the matching
image's original-format JSONL `unlabeled`, with stable owner ID and provenance,
only after lead admission. Unknown category remains unknown. Maintain raw
`objects`, previous versions and frozen teachers/evaluation sidecars.

## 4. Co-DETR-first routing

Co-DETR crop+resize detections are a **proxy reference**, not automatically GT.
They can drive `proxy_tide` diagnostics and review priorities without changing
benchmark ground truth or granting a training label. The historical
[Co-DETR profile](../../probes/unmatched_judge/codetr_profile/README.md) combines
Co-DETR with8B; it remains an immutable predecessor, not the new detector-only
implementation.

Recommended next candidate, pending independent calibration:

1. Parse/match against the frozen trusted ledger first. Retain all parse debt.
   Cache original-image detections once per image. Use unpainted input to the
   detector; class and candidate edges are not prompts to it.
2. Add a context crop (initial inherited recipe:3x candidate width/height,
   minimum128 source pixels/side, clipped to image) and normal detector resize.
   Keep the crop window, input hash and exact resize/coordinate transform. Map
   boxes back to source pixels exactly once. Candidate-selected crops bias the
   view; agreement across views from one model is correlated evidence.
3. Compare trusted GT, rollout predictions and detector boxes separately. Store
   same/different-class overlap, four-edge residuals, containment, view agreement,
   crop truncation and competing-owner/duplicate flags. Do not average coordinates
   across potentially different owners or silently replace the rollout box.
4. Route single-owner consistent support to `supported_candidate`; observed
   category/extent/view/identity conflicts to review; absence of support to
   `unknown`. A diagnostic loose box can retain entity credit while failing
   geometry admission. Detector-supported repairs are separate proposals with
   provenance, not automatically accepted teacher targets.
5. Until qualified on the intended population, **all newly promoted teacher
   owners/category/box repairs require lead admission**, with visual evidence.
   No automated negative labels or penalties for detector non-detections. Refusing
   to add a candidate is not proof that the complete training objective is neutral
   to that object; complete-output SFT and masking semantics remain explicit.
6. Review residuals with full image plus bbox overlay and context crop. Batch by
   image and reuse accepted owner evidence; only lead/subagents needed for
   decision-bearing uncertainty and calibration/audit samples. No exhaustive
   review of all raw proposals is a freeze gate.

The old score.50 / IoU.75 rule is a screening anchor, not a safe training threshold.
Avoid tuning on already revealed64-image historical labels and calling that a
fresh validation. Independent calibration must measure false support, clean
retention, uncertainty, review workload and proposal/owner denominators. Sample
supported, conflicting and unknown groups; for distribution estimates record
sampling probabilities, weight strata by population size and report uncertainty.
A selected good-case sample cannot estimate the whole unmatched population.

## 5. What is registered versus implemented

This page is the shared terminology and user-approved conservative routing.
Existing scorers are not silently converted to TIDE; historical labels are not
automatically migrated. The detector-only retained-output diagnostic and concrete
follow-on design are owned by the
[2026-09-16 proxy record](../../research/experiments/2026-09-16-codetr-only-review-proxy/unit.md).
New multi-view inference, exact automatic classification rules and calibration
must have explicit versioned execution evidence before being called operational.
