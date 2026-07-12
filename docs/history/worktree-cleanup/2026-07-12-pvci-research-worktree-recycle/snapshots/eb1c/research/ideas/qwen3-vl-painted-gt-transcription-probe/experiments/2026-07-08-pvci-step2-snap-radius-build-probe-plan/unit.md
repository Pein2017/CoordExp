---
title: PVCI Step 2 Snap-Radius Build-Probe Plan
description: Defines the build-probe plan for testing whether anti-copy jitter training can turn painted marks into object designators.
type: idea
role: research-unit
authority: non_normative_research
promotion_status: not_promoted
unit_id: 2026-07-08-pvci-step2-snap-radius-build-probe-plan
topic: qwen3-vl-painted-gt-transcription-probe
status: complete
tags: [coordexp-swift, painted-gt, pvci, step2, build-probe, snap-radius]
updated: 2026-07-08
---

# PVCI Step-2 Snap-Radius Build-Probe Plan - 2026-07-08

## Decision

Proceed with the Step-2 snap-radius branch decider before PICD, hidden-cursor
distillation, coverage/stop mixing, or full production training.

The approved research question is:

```text
Can anti-copy jitter training transform painted marks from a geometry-copy
actuator into an object designator that recovers tight GT geometry?
```

## Rationale

Step-1 showed that Qwen3-VL strongly consumes object-coupled boundary cues, but
the strongest conflict sentinel was geometry-leakage dominant:

- `outline_only` at val100 reached high target-row debug F1 while source GT box
  and rendered mark box were identical.
- `wrong_aspect_outline` at val100 retained high target-row debug F1, but
  predictions overwhelmingly followed the rendered mark geometry instead of the
  source GT geometry.
- `background_outline` stayed poor, so the effect is not caused by arbitrary
  rectangle copying.

Therefore the next useful probe must deliberately break the correlation:

```text
rendered mark geometry == target GT geometry
```

and test whether training can restore:

```text
prediction -> tight source GT object boundary
```

instead of:

```text
prediction -> rendered painted mark boundary
```

## Approved Decisions

1. Use Step-2 snap-radius as the next primary research objective.
2. Run a no-training calibration first using the current Step-484 painted
   stepwise adapter.
3. Train object-coupled but geometry-corrupted marks; keep the target as tight
   GT phrase plus tight GT box.
4. Warm-start the first Step-2 run from the current Step-484 painted stepwise
   adapter.
5. Use a train256 tiny gate first, then evaluate train256 and val32, then
   selectively promote variants to val100.
6. Use stochastic anti-copy training and fixed deterministic evaluation bins.
7. Use green/yellow/red route verdicts, not a single aggregate metric.
8. Keep coverage/stop as a separate side probe; do not mix it into Step-2
   snap-radius training.
9. Use research notes, configs, scripts, tests, and artifact manifests as the
   carrier. Do not create OpenSpec unless a stable config/artifact/eval contract
   is promoted.

## Evidence Target

Primary evidence:

```text
snap curve before vs after training
mean IoU(pred, GT)
mean IoU(pred, mark)
closer_gt / closer_mark / closer_tie
F1 vs GT
step validity
class/phrase match
size-stratified metrics
```

Critical sentinels:

```text
tight_outline
wrong_aspect_outline
grid_snapped_box_outline_center
jitter_medium
jitter_large
background_outline
```

The leakage analyzer must continue comparing pixel-space predictions against
both:

- `painted_gt.marks[0].source_bbox_pixels`
- `painted_gt.marks[0].bbox_pixels`

Do not compare directly against coord-bin `gt[0].bbox` for leakage decisions.

## Build Requirements

Minimal probe-enablement work:

1. Add deterministic seeded anti-copy mark materialization.
2. Preserve source GT bbox and rendered mark bbox in metadata.
3. Add fixed snap-curve evaluation variants.
4. Generate train256 anti-copy jitter training JSONL.
5. Generate no-train and post-train inference configs.
6. Reuse the existing leakage analyzer after verifying its pixel-space
   comparison contract.
7. Add focused tests for deterministic materialization, metadata preservation,
   and leakage/report compatibility.

## Training Mix

First approved tiny training mix:

```text
20% tight outline
50-60% random anti-copy jittered outline
10-15% grid-snapped/coarse outline
10-15% style jitter: dashed, width/color variation, partial dropout
```

The first training mix should use object-coupled marks only. Background and
wrong-object marks remain evaluation controls, not first-pass training examples.

Suggested anti-copy jitter dimensions:

```text
scale: 1.05-2.25
center offset: up to roughly 0.25 object size
aspect warp: independent x/y factors, for example 0.7-1.5
grid snap: coarse visual-token-like cells
style: dash, thickness, color, partial edge dropout
```

The important property is that any stable mark-only affine-copy rule should be
insufficient to predict tight GT geometry.

## Probe Run

### E0 - No-Train Snap-Curve Calibration

Use the current Step-484 painted stepwise adapter:

```text
/data/CoordExp/outputs/painted_gt/train_overfit_gate/painted_gt_stepwise_teacher_prefix_geo_gate256_overfit16_warm_start_dora_all_towers_accelerate8_ebs8/checkpoints/step-484/adapter
```

Run val32 first with fixed snap-curve variants. Promote only useful variants to
val100.

Purpose:

```text
measure the pre-training copy curve
```

### E1 - Tiny Anti-Copy Training

Materialize train256 anti-copy jitter training data and warm-start from the
Step-484 adapter.

Purpose:

```text
test whether the model can learn:
mark selects object; image content defines tight boundary
```

### E2 - Post-Training Snap-Curve Evaluation

Rerun the same snap-curve variants on train256 and val32. Promote the most
diagnostic variants to val100.

Purpose:

```text
compare before/after leakage direction and F1 under mark-GT mismatch
```

### E3 - Separate Coverage/Stop Side Probe

Keep this outside the Step-2 training mix. Use dimmed outlines or hatching for
committed objects, not filled blobs.

Purpose:

```text
test whether visual coverage memory reduces re-selection, duplication, or early
stop without contaminating the snap-radius claim
```

## Route Verdict Gates

### Green - Designator Branch Alive

Signals:

```text
wrong_aspect closer_gt >= 0.60-0.70
mean IoU(pred, GT) > mean IoU(pred, mark) under wrong/aspect or medium jitter
grid/coarse F1 improves strongly, roughly near or above 0.50
tight outline F1 remains high, ideally >= 0.70
background outline remains poor
```

Consequence:

```text
VCI/PVCI designator branch becomes justified.
Feature-space cursor or PICD should distill the anti-copy snap teacher, not the
original tight-outline copy teacher.
```

### Yellow - Sketch-And-Snap

Signals:

```text
medium jitter improves
large/grid variants remain weak
small objects still fail
wrong_aspect shifts toward GT but does not flip
```

Consequence:

```text
Prefer coarse proposal -> rendered sketch -> snap refinement.
```

### Red - Boundary-Proposal Route

Signals:

```text
wrong_aspect still closer_mark
grid/coarse remains weak
pred-mark IoU dominates after training
```

Consequence:

```text
Treat the VLM as a controllable boundary transcriber/namer/verifier.
Precise localization must come from a boundary proposal or cursor module.
```

## Non-Goals

- Do not claim final COCO mAP improvement from this probe alone.
- Do not start PICD or hidden-cursor distillation before the snap-radius verdict.
- Do not mix coverage/stop supervision into the first Step-2 training run.
- Do not create a stable OpenSpec contract unless this probe promotes a durable
  public config, artifact, or metric surface.
- Do not interpret high tight-outline F1 as object-identity evidence unless
  mark geometry and GT geometry are decoupled.

## Next State

`build-probe`

Implementation should start with deterministic materialization and fixed
snap-curve eval bins, then run E0/E1/E2 before committing to the next research
route.

## Research Unit Closeout

Observed:

- The plan selected Step-2 snap-radius / anti-copy training as the next primary
  research objective before PICD, hidden-cursor distillation, coverage/stop
  mixing, or production training.
- The plan required E0 no-train calibration, E1 tiny anti-copy training, E2
  post-training snap-curve evaluation, and a separate coverage/stop side probe.

Supported:

- A bounded anti-copy probe was the correct next carrier after Step 1 geometry
  leakage evidence.
- The plan intentionally kept coverage/stop separate from snap-radius training.

Not supported yet:

- The anti-copy result itself; that is recorded in the later Step-2 results
  unit.
- Stable training, inference, config, or evaluation contracts.

Next decider:

- Execute the E0/E1/E2 snap-radius sequence and compare GT-vs-mark geometry
  following.

Promotion decision:

- Not promoted. This is a non-normative build-probe plan superseded by the
  executed Step-2 result unit.
