## Why

Current Stage-2 Channel-B training is stable, but the active pseudo-positive path is still much better at retaining or geometrically refining anchor-proposed objects than at teaching missing object birth under rollout. Before spending large compute on long Stage-2 runs, we need a smaller and more falsifiable birth-first contract that improves recall and stop/continue calibration without broad unmatched-negative suppression, heavier duplicate machinery, or a second teacher-forced pass.

This study is intentionally anchored to one fixed base-model plus adapter pair so the resulting training-dynamics conclusions stay attributable to the Channel-B contract rather than checkpoint-family drift. The base model stays `/data/CoordExp/model_cache_remote/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp`, and the only adapter checkpoint in scope for this change is `/data/CoordExp/output_remote/stage1_2b/coco_bbox_max60-hard_ce_soft_ce_w1_gate/epoch_4-from-base-2B/v0-20260227-050057/checkpoint-1332`.

## What Changes

- Add an opt-in birth-first Channel-B mode that keeps the anchor-first, clean-prefix, one-forward contract and reuses the existing `K=2` anchor-plus-one-explorer decision profile as the study control surface.
- Change Channel-B rollout supervision so support-positive retained anchors contribute structure-first prefix supervision instead of behaving like mostly geometry-correction state.
- Add a local continue-over-EOS calibration term for recovered-GT boundaries while keeping recovered GT objects on the existing FN-injection path and projecting the new atom through the existing rollout-text surface rather than a new authored objective module.
- Keep duplicate-burst unlikelihood as a narrow B-only guardrail instead of the main recall-improvement mechanism.
- Add explicit metrics and decision-run audit surfaces in the existing docs/artifact contract for support-positive prefix coverage, recovered-GT birth pressure, and continue-over-EOS activation.
- Update Stage-2 operator docs and decision-run tasking so every repo-authored config, smoke, and decision/long-run comparison in this change treats the fixed base model `/data/CoordExp/model_cache_remote/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp` plus adapter checkpoint `/data/CoordExp/output_remote/stage1_2b/coco_bbox_max60-hard_ce_soft_ce_w1_gate/epoch_4-from-base-2B/v0-20260227-050057/checkpoint-1332` as the only model surface in scope for this study.

## Capabilities

### New Capabilities
- None. This change reshapes existing Stage-2, loss-registry, and metrics contracts instead of introducing a new standalone subsystem.

### Modified Capabilities
- `stage2-ab-training`: define an opt-in birth-first Channel-B contract, typed `birth_first` config, reuse the existing `K=2` decision profile for the study, and tighten support-positive / recovered-GT supervision rules.
- `teacher-forcing-unified-loss-registry`: extend rollout-context subset semantics and add a canonical continue-over-EOS calibration component for evidence-backed recovered boundaries.
- `trainer-metrics-components`: add explicit birth-first metrics for support-positive prefix objects, recovered-GT boundaries, and continue-over-EOS activation.

## Impact

- Affected code: `src/trainers/stage2_two_channel/`, `src/trainers/teacher_forcing/`, `src/config/schema.py`, and related Stage-2 tests.
- Affected configs/docs: `configs/stage2_two_channel/`, `docs/training/STAGE2_RUNBOOK.md`, `docs/training/METRICS.md`, and this change-local OpenSpec artifact set.
- Repro/eval impact: changes rollout-time supervision semantics, requires config/schema coverage, and must preserve one-forward reproducibility, current geometry contracts, and current artifact naming/surface behavior.
- Study scope impact: every repo-authored config, smoke, and decision/long-run comparison for this change MUST use only the base model `/data/CoordExp/model_cache_remote/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp` together with adapter checkpoint `/data/CoordExp/output_remote/stage1_2b/coco_bbox_max60-hard_ce_soft_ce_w1_gate/epoch_4-from-base-2B/v0-20260227-050057/checkpoint-1332`.
