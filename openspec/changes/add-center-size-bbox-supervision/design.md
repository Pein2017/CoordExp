## Context

The current CoordExp detection stack serializes boxes canonically as `bbox_2d: [x1, y1, x2, y2]` and threads that assumption through dataset rendering, teacher-forced training, rollout parsing, matching, inference artifacts, and offline evaluation. In the training path, `bbox_geo` decodes bbox quartets from teacher-forced coord logits and applies SmoothL1 plus CIoU, while Stage-2 strict pipeline validation currently allows only the existing `bbox_geo.config` keys.

The motivating intuition for this change is that many detection annotations contain more noise on object extent than on coarse object location. A full public swap to center-size text serialization would touch prompt rendering, parser/matching logic, and evaluator-facing contracts. That blast radius is unnecessary for a first experiment whose main question is whether center-heavy supervision is helpful.

## Goals / Non-Goals

**Goals:**
- Add a lightweight, opt-in bbox supervision mode that keeps outward boxes canonical `xyxy` while adding center-wise and soft size terms in loss-space.
- Keep the canonical external `bbox_2d` / `xyxy` contract unchanged across data, parsing, inference, and evaluation.
- Preserve existing Stage-1 and Stage-2 entrypoints and keep the feature entirely YAML-driven.
- When convenient while touching both code paths, standardize the shared bbox regression math so Stage-1 and Stage-2 do not grow separate `center_size` implementations.
- Make the experiment easy to validate with targeted unit/config tests, one required Stage-2 smoke path, and optional Stage-1 smoke coverage.

**Non-Goals:**
- Do not introduce a new public geometry key such as `bbox_cxcywh`.
- Do not change CoordJSON rendering, rollout parsing, matching, or evaluator artifact semantics in this change.
- Do not redesign coord-token CE / soft-CE / W1 around slot-specific center-vs-size targets in this first version.
- Do not attempt a full loss-framework rewrite or migrate Stage-1 onto the Stage-2 registry in this change.
- Do not add new CLI flags or touch upstream Hugging Face model internals.

## Decisions

### 1. Represent the new idea as an internal `bbox_geo` loss-space decomposition, not a public bbox format change

Decision:
- Add an opt-in `bbox_geo.config.parameterization: xyxy | center_size` mode.
- Keep `xyxy` as the default.
- In `center_size` mode, `bbox_geo` will still decode predicted boxes from the existing quartet, keep them canonicalized as `xyxy`, and only then derive `(cx, cy, log_w, log_h)` for the additional regression terms.
- CIoU continues to operate on canonical `xyxy` boxes in the same forward pass.

Rationale:
- This isolates the experiment to the loss path.
- It avoids touching data -> transforms/packing -> training/inference -> artifacts beyond the decoded geometry module.
- It preserves Qwen3-VL chat-template compatibility and all existing downstream artifact contracts.

Alternatives considered:
- Public `cxcywh` serialization: rejected for v1 because it would require prompt, parser, matching, inference, and eval contract changes.
- Slot-specific coord-token loss redesign: rejected for v1 because it requires broader target-building and token-supervision changes than needed to test the core intuition.

### 2. Keep the config surface narrow and shared across Stage-1 and Stage-2

Decision:
- Reuse the existing `bbox_geo` surface instead of introducing a new objective module.
- Support the following additive knobs:
  - `parameterization`
  - `center_weight`
  - `size_weight`
- Preserve existing `smoothl1_weight` and `ciou_weight`.
- For Stage-2, extend strict pipeline validation under `stage2_ab.pipeline.objective[*].config` and `rollout_matching.pipeline.objective[*].config`.
- For Stage-1, mirror the same behavior under `custom.bbox_geo`.
- Existing two-key configs must remain valid and resolve to default `xyxy`.

Rationale:
- This is the smallest config-first change that still gives us a real experimental knob.
- It minimizes implementation spread and keeps the feature discoverable in the existing bbox geometry path.

Alternatives considered:
- New standalone module: rejected because it would add pipeline complexity for behavior that is still fundamentally bbox geometry supervision.
- Hidden implementation-only toggles: rejected because experiment reproducibility depends on resolved config visibility.
- Making the new keys required everywhere: rejected because it would break existing Stage-1 and Stage-2 configs for no research benefit.

### 3. Take a small shared regression-core cleanup opportunistically, without turning this change into a framework rewrite

Decision:
- When both Stage-1 and Stage-2 bbox geometry paths are touched for `center_size`, reuse a small shared internal helper for canonical box regression decomposition and weighted reduction.
- The shared helper should own:
  - canonical `xyxy` box inputs,
  - optional `center_size` derivation `(cx, cy, log_w, log_h)` from canonical boxes,
  - epsilon-guarded width and height handling before log conversion,
  - mean-like weighted reduction for the regression branch.
- Stage-1 and Stage-2 wrappers may continue to own extraction, batching/masking, metric emission, and published pipeline state, but they should not re-implement the center-size regression math independently.
- This sidecar cleanup must not rename public modules such as `bbox_geo`, `bbox_size_aux`, or `coord_reg`, and it must not require migrating Stage-1 onto the Stage-2 objective registry.

Rationale:
- The current repo already shares low-level bbox math, but Stage-1 and Stage-2 still wrap it differently enough that a new `center_size` branch would otherwise be duplicated.
- A small shared regression core reduces divergence risk while keeping the feature scoped to the active experiment.
- This gives us part of the maintainability benefit of a broader loss standardization effort without taking on the blast radius of a full loss-framework rewrite.

Alternatives considered:
- Leave Stage-1 and Stage-2 bbox regression implementations separate: rejected because the new `center_size` math would drift quickly across the two paths.
- Full teacher-forcing loss unification or Stage-1 registry migration: deferred because it is valuable but too broad for this change.

### 4. Keep metric keys stable and document semantic branching instead of minting a new metric namespace

Decision:
- Continue emitting existing bbox geometry keys such as `loss/geo/bbox_smoothl1` and `loss/geo/bbox_ciou` in Stage-1 and their Stage-2 provenance-specific equivalents.
- Interpret `bbox_smoothl1` as “the configured bbox regression term for bbox_geo,” which is plain `xyxy` in default mode and center-wise plus soft `log_w` / `log_h` loss-space supervision in experimental mode.
- Treat `resolved_config.json` as the authoritative machine-readable discriminator for bbox regression parameterization; `run_metadata.json` stays provenance-only.
- Document that cross-run comparisons of `bbox_smoothl1` are invalid unless parameterization is joined from resolved config.

Rationale:
- This avoids unnecessary metric proliferation for a v1 experiment.
- Operators can still compare runs by inspecting the resolved config and runbook documentation.

Alternatives considered:
- New `bbox_center` / `bbox_size` namespaces: deferred because they add logging/documentation churn before we know whether the mode is useful.
- Encoding parameterization semantics into `run_metadata.json`: rejected because `resolved_config.json` is the canonical config authority.

### 5. Require one Stage-2 smoke path and optionally add Stage-1 smoke coverage

Decision:
- The minimum reproducibility target is one Stage-2 smoke config or documented command using center-size bbox supervision.
- Stage-1 smoke coverage is recommended as an extra fast loop, but it is not sufficient by itself because this change modifies active Stage-2 capability contracts.
- No new metric namespaces will be introduced in v1; resolved config plus existing bbox loss keys are sufficient.

Rationale:
- This keeps the operator-facing Stage-2 contract honest.
- An optional Stage-1 loop still provides a cheap debug path for the decoded-box supervision hypothesis before spending larger rollout budgets.

Alternatives considered:
- Stage-1-only smoke in the first pass: rejected because it can land Stage-2 contract changes without proving the active Stage-2 path.
- Stage-1 and Stage-2 smoke profiles in the same first pass: optional, but not required for the first pass if one Stage-2 smoke path is already covered.
- New experimental metric families: rejected because they would increase operator-facing surface area without changing the underlying artifact contract.

## Risks / Trade-offs

- [Risk] The lightweight version does not give true slot-wise center CE or size-specific coord-distribution targets. → Mitigation: document that this change validates only the decoded-box loss-space hypothesis; leave token-space redesign as a follow-up.
- [Risk] `log_w` / `log_h` can be numerically unstable for tiny boxes. → Mitigation: clamp width and height with a positive epsilon before log conversion and cover it with unit tests.
- [Risk] Stable metric names could hide that `bbox_smoothl1` changed semantics under the experimental mode. → Mitigation: document the branching semantics in the runbook and rely on `resolved_config.json` for exact run interpretation.
- [Risk] Existing Stage-2 configs could break if the new bbox_geo keys become required by accident. → Mitigation: specify additive optional-key semantics and require explicit backward-compat tests for legacy two-key configs.
- [Risk] Center-strong regression may improve localization stability while hurting scale calibration. → Mitigation: keep CIoU on canonical `xyxy`, expose separate `center_weight` / `size_weight`, and require targeted regression tests plus at least one smoke config.
- [Risk] The feature could be mistaken for a parser/eval contract change. → Mitigation: make `external bbox_2d remains canonical xyxy` a first-class requirement and non-goal.
- [Risk] Hidden downstream module interfaces could expand the blast radius if `bbox_geo` stops publishing canonical `xyxy` state. → Mitigation: keep the published bbox_geo state contract explicit for `bbox_size_aux`, `coord_reg`, and `coord_diag`.

## Migration Plan

1. Land schema/config validation first with default `parameterization: xyxy` behavior unchanged.
2. Implement the internal loss-space decomposition inside both Stage-1 and Stage-2 `bbox_geo` paths, preferably through a shared regression helper, while preserving the canonical shared bbox state payload.
3. Add unit/config coverage before enabling any experimental configs.
4. Add one Stage-2 smoke-ready config example or documented command with a distinct `training.run_name` and verify `resolved_config.json` reflects the intended parameterization.
5. Roll back by reverting configs to the default `xyxy` mode; no data or artifact migration is required because external contracts do not change.

## Open Questions

- None for v1. The first-pass design assumes one required Stage-2 smoke target, optional Stage-1 smoke coverage, stable existing metric keys, and canonical external eval artifacts.
