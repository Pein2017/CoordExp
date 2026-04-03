## Why

Corner-based bbox supervision treats all four edges as equally reliable, but in many detection datasets the object center is more stable than the exact annotated extent. We need a low-blast experimental path that can emphasize stable localization signals without changing the canonical `bbox_2d` / `xyxy` data, inference, and evaluation contracts.

## What Changes

- Introduce an opt-in internal bbox supervision mode that keeps decoded and serialized boxes in canonical `xyxy`, but augments the loss with a center-wise term and a softer size term derived from that same canonical box.
- Extend `bbox_geo` so its loss-space decomposition can combine strong center supervision, softer `log_w` / `log_h` supervision, and CIoU on canonical `xyxy`.
- As a lightweight sidecar, standardize the bbox regression math through a small shared internal helper reused by Stage-1 and Stage-2 wrappers, without changing public module names, metric keys, or pipeline contracts.
- Keep the same config-first idea across Stage-1 `custom.bbox_geo` and Stage-2 `bbox_geo.config` surfaces so the experiment can be reproduced without one-off code paths.
- Preserve decoded-box CIoU, rollout matching, parser behavior, inference artifacts, and evaluator inputs in canonical `xyxy`.
- Keep the experiment config-first under existing Stage-1 and Stage-2 YAML surfaces, with no new CLI flags.
- Make the first implementation intentionally narrow: no new public geometry key, no CoordJSON schema change, no rollout parser/evaluator contract change, and no slot-specific coord-token CE/W1 redesign in this change.

## Capabilities

### New Capabilities
- `center-size-bbox-supervision`: opt-in internal bbox loss decomposition that keeps canonical external `bbox_2d` semantics while weighting center and size terms differently in loss-space.

### Modified Capabilities
- `teacher-forcing-unified-loss-registry`: widen the shared `geo` semantics so decoded-box regression may add center-wise and `log_w` / `log_h` loss terms while CIoU remains canonical `xyxy`.
- `stage2-ab-training`: expand strict `bbox_geo.config` validation so Stage-2 AB can author the new center-size supervision knobs in the existing objective pipeline.
- `rollout-matching-sft`: expand strict `bbox_geo.config` validation so rollout-aligned Stage-2 can author the same center-size supervision knobs in the existing objective pipeline.
- `trainer-metrics-components`: clarify that existing bbox geometry metric keys remain stable while the regression term may use center-wise and `log_w` / `log_h` loss-space components, with CIoU staying canonical `xyxy`.

## Impact

- Affected code is expected to center on `src/trainers/teacher_forcing/modules/bbox_geo.py`, `src/trainers/losses/bbox_geo.py`, shared bbox geometry helpers, shared config/schema validation, and the Stage-1 / Stage-2 YAML contracts.
- Canonical data, prompt, parser, matching, inference, and evaluator contracts remain unchanged, reducing reproducibility and eval-validity risk.
- Docs/specs will need small updates in the Stage-1 objective reference, Stage-2 runbook, and metric interpretation docs if implementation lands.
- Verification must include at least one Stage-2 smoke path because the change modifies active Stage-2 capability contracts even though the public eval/infer artifact contract remains unchanged.
