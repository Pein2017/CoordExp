---
title: Fixed-Prompt Clean-versus-Degraded Coordinate Branch Replication
description: Replicates the person-25 coordinate-mass observation on six crop-reviewed full-image rollout pairs and separates independent boundary error, autoregressive geometry drift, and same-class owner switching.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: authorized_within_goal
unit_id: 2026-07-18-fixed-prompt-clean-versus-degraded-coordinate-branch-replication
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-07-18
---

# Fixed-Prompt Clean-versus-Degraded Coordinate Branch Replication

Execution is complete. See [results](results.md).

## Status at closure

- Scientific status: complete within the frozen six-case scope.
- Implementation status: authorized for this bounded research unit.
- Training status: not authorized by this unit.
- Architecture status: not promoted.

## Question

When repeated full-image sampling begins from the exact same image and base
prompt, why can one rollout produce a physically coherent box while another
produces a truncated, expanded, or different-instance box of the same class?

This unit tests whether the local result previously observed for person 25 is
repeatable: a single coordinate token may win under greedy decoding even when
the aggregate probability assigned to a physically valid coordinate
neighborhood is larger.

## Competing explanations

1. **Distributed valid-boundary probability.** The model assigns meaningful
   aggregate probability to a neighborhood around the physical boundary, but
   no single valid coordinate token wins the vocabulary argmax.
2. **Earlier owner selection.** The clean and degraded rows already select
   different physical instances at the first coordinate. Later geometry is a
   consequence, not an independent extent failure.
3. **Autoregressive geometry drift.** The first coordinate difference is
   harmless, but teacher-forcing that branch changes later coordinate
   distributions and eventually produces a bad extent.
4. **Annotation or ontology ambiguity.** The apparent clean/degraded contrast
   reflects an ambiguous physical extent, such as whether an umbrella includes
   only the canopy or also the pole.
5. **Person-25 special case.** The earlier fixed-`y2` result does not replicate
   beyond its one image, checkpoint, and partial row.

## Frozen evidence scope

- Model family: Qwen3-VL 2B.
- Adapter: geometry-sorted Gaussian coordinate-target DoRA checkpoint step
  4887.
- Existing rollouts: full-image independent bagging, 16 fresh-prompt calls per
  image, temperature 0.4, top-p 0.95, repetition penalty 1.0.
- New scoring: exact base prompt and image. The primary replay uses the
  historical bfloat16 forward precision and computes all softmax probabilities
  in float32. A full-float32 forward is retained as a precision-sensitivity
  control. Both runs expose two explicitly labeled probability views: the raw
  temperature-one model distribution and temperature 0.4 plus top-p 0.95
  applied to the replay logits. The latter is an exact policy transform, but
  only the bfloat16 arm is intended to reproduce the source runtime closely.
- Primary cases: three crop-reviewed same-owner or same-extent candidates.
- Negative controls: three crop-reviewed cases in which the two rows likely
  select different same-class instances.

The case manifest is `cases.json`. Every case points to two immutable terminal
output bundles and an accepted reference box.

## Primary observations

For each case:

1. verify that both source bundles are `FULL_BAG_K`, use a fresh full-image
   prompt, refer to the same image, have identical prompt token identifiers,
   and contain a row-zero prediction with the same category;
2. find the first differing coordinate among `x1`, `y1`, `x2`, and `y2`;
3. at the first difference, teacher-force only the exact shared row prefix;
4. at every later coordinate, separately teacher-force the clean trajectory
   prefix and the degraded trajectory prefix. This distinguishes the first
   branch from geometry drift caused by an earlier branch;
5. measure the probabilities and ranks of the clean and degraded coordinate
   tokens under both the raw model distribution and the source sampling
   policy;
6. convert the accepted physical boundary into coordinate-bin space;
7. sum probability mass in accepted-boundary windows of 4, 8, 16, and 32
   bins on either side;
8. compare those sums with the isolated argmax and the two sampled tokens;
9. run one true greedy rollout with repetition penalty 1.0 and record which
   branch, if either, it follows.

The primary conclusion is distributional. Mean average precision, full
validation metrics, and rollout length are outside this unit.

## Crop-review decisions

- Image 15335, person: same small person. The degraded row incorrectly expands
  the top boundary to the canvas edge.
- Image 16451, umbrella: same umbrella. The degraded row extends below the
  canopy; this is retained as an extent/ontology-sensitive case.
- Image 7574, bowl: same white bowl. The degraded row captures only the upper
  part of the bowl.
- Image 12670, person: the two rows point to different nearby people. Negative
  owner-switch control.
- Image 6471, person: the two rows point to different spectators. Negative
  owner-switch control.
- Image 15517, bus: the degraded row shifts toward a neighboring bus segment.
  Negative owner-switch or owner-ambiguity control.

## Decision rules

- Support the distributed-valid-boundary explanation only if at least two of
  the three primary cases show, at the causally relevant branch, that raw
  valid-boundary probability mass is larger than the isolated raw coordinate
  argmax probability and that the source sampling policy retains non-trivial
  valid-boundary support rather than truncating it entirely.
- Support early owner selection if the negative controls separate at `x1` and
  later coordinates mostly follow that first choice.
- Support later geometry drift if the first difference lies inside the valid
  window but a later released coordinate leaves it.
- Do not aggregate primary cases with owner-switch controls into one rate.
- Stop after these six cases. Expansion to eight or more cases requires a
  mixed or inconclusive result, not merely a desire for tighter confidence
  intervals.

## Confounders and limitations

- The accepted ledger is stronger than official COCO alone but is not perfect
  physical ground truth.
- Coordinate neighborhoods are post-hoc diagnostics, so all four fixed window
  widths must be reported.
- A sampled token can move across the top-p boundary when the entire model
  forward changes from bfloat16 to float32. Probability arithmetic precision
  and model-forward precision must therefore remain separately labeled.
- A first differing token is not automatically the first harmful decision.
- The model can select a different same-class owner while producing a box that
  partially overlaps the reference owner.
- This unit does not establish a training loss or final architecture. It
  decides whether probability redistribution around coordinate decisions is a
  credible treatment target.

## Stop condition

Close the unit when the six crop-reviewed cases have strict bundle checks, one
true greedy row, matched-precision and full-float32 coordinate-mass results,
followed by a written comparison against the person-25 anchor. Do not begin
training from this unit.
