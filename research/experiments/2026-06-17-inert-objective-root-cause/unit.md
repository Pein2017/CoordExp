---
type: idea
role: research-unit
authority: non_normative_research
promotion_status: not_promoted
unit_id: 2026-06-17-inert-objective-root-cause
topic: prefix-denoising-sft
status: complete
title: Prefix Denoising Inert Objective Root-Cause Analysis
description: Root-cause analysis arguing that the tested prefix-denoising objective was inert and baseline comparison was confounded.
tags: [stage1, prefix-denoising, root-cause, eval-validity]
updated: 2026-06-20
---

# Prefix Denoising Inert Objective Root-Cause Analysis

## Headline

The bbox degradation is not well explained as denoising harm. The stronger read
is that the tested prefix-denoising objective was inert for this compact
coordinate model and the comparison baseline was not matched.

## Evidence

- Clean and noisy branch CE stayed nearly identical across training.
- Local-window KL was near zero from step 0.
- Teacher and student coordinate distributions were almost identical under
  clean vs noised coordinate prefixes.
- Teacher-forced coordinate distributions were broad under hard CE, with weak
  GT-local mass.
- The repaired loss path used isolated forwards, so objective inertness was not
  explained by clean-branch attention leakage.
- The reference baseline differed on objective, epoch/step budget, and adapter
  or full-merge state.

Probe scope:

```text
train: COCO rescale_32_1024_bbox_max60, sorted compact-full, LoRA r16 DoRA, 2 epochs, checkpoint-450
eval: val200, free greedy decode, temp 0, repetition_penalty 1.1, max_new_tokens 3084
```

## Interpretation

The model appeared to ground each coordinate from image and structural position
rather than from previously emitted coordinate tokens. If previous coordinate
tokens do not materially move the next-coordinate distribution, synthetic
prefix-token denoising has little mechanism to act on.

This is a negative result about the tested implementation premise, not a final
rejection of geometry-aware denoising. A geometry-aware idea may still need to
move the perturbation into a channel the model actually uses, such as
self-rollout or relative-encoding variants.

## Confounds

- The reference baseline was a 4-epoch SoftCE-coordinate full-merged model,
  while the prefix-denoising run was a 2-epoch hard-CE LoRA run.
- No matched hard-CE LoRA denoising-OFF control existed in the analyzed record.
- Decode used `repetition_penalty=1.1`, which may amplify coordinate-token
  edge saturation or arity tails.
- Axis-sort repair showed materialization and localization were separate
  problems.

## Recommended Next Probes

1. Run the matched denoising-OFF hard-CE LoRA control with the same recipe.
2. Compare against a matched coordinate-objective variant such as SoftCE.
3. Re-evaluate with `repetition_penalty=1.0` or coord-token-aware decode
   controls.
4. If pursuing denoising, test a mechanism that perturbs a channel the model
   actually conditions on.

Confidence: high for objective inertness and baseline mismatch in this record;
medium for the prediction that a matched denoising-OFF control will land near
the same AP until that control is run.

## Research Unit Closeout

Observed: clean/noisy branch CE and local-window KL were nearly identical, and
the analyzed baseline comparison differed on objective, training length,
adapter/full-merge state, and recipe.

Supported: the tested V1 objective likely did not engage the intended
coordinate-prefix mechanism, and the recorded baseline comparison is
confounded.

Not supported yet: a final rejection of geometry-aware denoising, or a
directional conclusion about denoising benefit/harm without a matched control.

Next decider: run or locate the matched denoising-OFF hard-CE LoRA control and
coordinate-objective comparison.

Promotion decision: keep as research interpretation; do not promote to
OpenSpec until a reusable metric or artifact contract emerges.

Manifest handle:

```text
source_path=progress/diagnostics/2026-06-17_prefix_denoising_inert_objective_root_cause_analysis.md
source_kind=worktree-dirty
status=??
classification=new_content_new_path
sha256=ed610b284219c2e96fe90d2a4c446978b96ededc767c0ce3118417f847846871
snapshot_path=docs/history/worktree-union/2026-06-20/snapshots/ed610b284219/progress/diagnostics/2026-06-17_prefix_denoising_inert_objective_root_cause_analysis.md
```

## Sources

- `docs/history/worktree-union/2026-06-20/snapshots/ed610b284219/progress/diagnostics/2026-06-17_prefix_denoising_inert_objective_root_cause_analysis.md`
- `docs/history/worktree-union/2026-06-20/manifest.tsv`
- `ed610b284219c2e96fe90d2a4c446978b96ededc767c0ce3118417f847846871`
- `worktree-dirty`
- `??`
