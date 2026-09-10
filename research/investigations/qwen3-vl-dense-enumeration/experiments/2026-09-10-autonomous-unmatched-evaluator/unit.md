---
title: Automated evaluator for real unmatched detections
description: Autonomous bounded method development with a frozen image-disjoint visual reference.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: authorized_within_goal
unit_id: 2026-09-10-autonomous-unmatched-evaluator
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-09-10
---

## Current contract

The user explicitly requests an active autonomous goal: find a roughly usable
automated unmatched-detection evaluator without routine Codex image inspection.
The requested reusable profile is internal convenience, not a CLI product or
service. Available GPUs are authorized; unrelated jobs, GT and evaluation
versions remain untouched. This supersedes the earlier fixed-pilot stop rule,
not the scientific negatives in the [earlier unit](../2026-09-10-qwen4b-unmatched-judge/unit.md).

Question: can automated image evidence distinguish supported entity/category,
single-instance localization and uncertainty on new real candidate images?
The strongest alternative is proposal agreement or same-family error
correlation instead of valid visual discrimination.

Root owns methods, artifact identity, validation and final admission. A blinded
worker collected a one-time 64-image reference; those labels are provisional
visual judgments, never new GT and never the deployed inference mechanism.
No routine manual image inspection is included in runtime or latency claims.

The [frozen protocol](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-autonomous-unmatched-evaluator/contract.md)
owns detailed acceptance criteria. Existing 54 cases are development only.
Holdout v1: seed 2026091001, uniformly sample 64 of the 91 eligible images outside
the 39 development images, then one real non-strict-repeat unmatched candidate
per image. The eligible pool contains 268 candidates. This image-balanced
estimand differs from a candidate-weighted whole-FP prevalence estimate.
Protected confirmation512 is not opened.

Rough-use gate, fixed before holdout labels/predictions: >=90% selective
precision among definite reference labels, >=10 definite accepts, >=25% clean
retention, >=15% total acceptance coverage, <=20% uncertain references among
accepts. Report uncertain accepts conservatively too, and finite-sample
uncertainty; these thresholds are not a population guarantee. Hot amortized
latency <=5 seconds/candidate and cold initialization <=180 seconds on one GPU.
Rejection or box repair needs its own evidence, not inferred from accept precision.

Do not tune methods on holdout outcomes. Each local method batch has a
30-minute / 200-response bound. End unsuccessful development branches instead
of indefinitely adjusting thresholds. No training, GT mutation, paid external
API, dependency installation, shared runtime restart, or resident GPU server.

## Current evidence and branches

Artifact root:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-autonomous-unmatched-evaluator`.

- `reground-dev-v1`: completed 108 responses, independent category/box from
  point cues in two views. Fixed .75 IoU rule accepts 2 clean + 1 problematic
  out of 54; 39 unknown and 12 unvalidated repair proposals. Fails precision
  and coverage. No holdout run or automatic repair promotion.
- `blind-dev-v1`: completed 58 responses (54 development + 4 pre-existing
  technical image controls). Proposed class omitted; observe pixels, category,
  object count and edge extent. Fixed acceptance admits 3 clean + 4 problematic
  + 2 gray. Fails. Structured explanation does not establish correct decisions.
- `dino_probe.py`: frozen dedicated-detector two-view rule, superseded by the
  user's already-working Co-DETR offer before inference. Download recovery was
  told to stop; partial files are retained. This is not model-quality evidence.
  Model: IDEA-Research/grounding-dino-tiny, revision
  `a2bb814dd30d776dcf7e30523b00659f4f141c71`, public weights 689359096 bytes.
- `source_crop_probe.py`: source step2444 detector-native regeneration in
  2x/3x context crops, not a Yes/No instruction-following task. Superseded and
  stopped after 38 complete cases (exit143); partial raw evidence is retained.
  Uses existing native FP32/SDPA/DoRA/selected-embedding loader, no training;
  mapping from resized crop coordinates to original pixels is recorded.
  Same-family agreement is explicitly a possible correlated-error mechanism.
- `codetr_probe.py`: current main method after explicit user offer and lead
  selection. Reuse the installed Co-DETR ViT-L checkpoint and existing official
  `init_detector/inference_detector` entry in the `mmdet` Conda environment.
  Only source pixels enter inference: no GT, candidate class text or box outline.
  Initial development rule: same-class score >= .50 and candidate IoU >= .70
  supports row acceptance; all nonmatches remain unknown. Entity/category support at
  IoU >= .25 is separately measured and does not imply acceptable geometry.
  The detector may reproduce COCO annotation conventions and omissions;
  benchmark AP or non-detection cannot establish visual truth or falsity.

## Closeout

The retained method is the frozen3x-context Co-DETR score.50/IoU.75 gate plus
at least one matching free-category8B point observation. It provides14/64
positive candidates on the image-disjoint holdout:11 clearly usable,1 box error,
2 reference-uncertain. The original blind reference gave11/13 definite precision;
after an explicit defective-to-uncertain correction it is11/12. Conservative
verified support remains11/14 in both versions. These are provisional visual
judgments, not new GT or a population accuracy guarantee.

All predeclared rough-use gates pass under the documented uncertainty
adjudication, and the small internal callable passed a real8-case consumer
replay (80.824s cold end-to-end; all decisions identical) plus12 focused tests.
See [results](results.md) for both reference versions, receipts and limitations.
The profile is retained for screening only. No hard negative/reward, automatic
GT change, cross-prediction duplicate ruling, standing GPU service, architecture
promotion, or additional research launch is implied. Exploration is closed.
