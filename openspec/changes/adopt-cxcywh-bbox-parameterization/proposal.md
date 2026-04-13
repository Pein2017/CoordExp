## Why

The original `91a1656` change tried to introduce a configurable bbox-format
switch, but it changed only part of the training/prompt/parser surface and left
contract drift behind. The subsequent draft widened the scope too far: it tried
to redesign Stage-1, Stage-2, inference, and multiple bbox loss families at
once.

For V1, the goal is narrower and more testable:

- keep canonical `xyxy` on raw data and all downstream evaluation artifacts,
- introduce an internal model-facing bbox parameterization for Stage-1,
- validate that parameterization under the simplest possible loss surface,
- keep enough gating to preserve the separation between coord tokens and
  non-coord tokens.

This change therefore scopes V1 to a Stage-1-centered experiment with
model-facing `bbox_2d: [cx, cy, u(w), u(h)]` serialization, where
`u(s) = (log(max(s, s_min)) - log(s_min)) / -log(s_min)`, plus pure hard CE on
coord slots and no regression-style bbox losses.

## What Changes

- Preserve canonical `bbox_2d: [x1, y1, x2, y2]` on raw JSONL, builder
  metadata, standardized inference artifacts, evaluation, benchmarking, and
  visualization.
- Add an internal conversion layer in Stage-1 dataset rendering / serialization
  that converts canonical `xyxy` boxes into a model-facing center-log-size
  tuple while keeping external field names unchanged:
  - field name remains `bbox_2d`
  - non-geometry fields such as `desc` remain unchanged
- Adjust dense prompt templates so model-facing bbox instructions describe
  `bbox_2d` as `[cx, cy, u(w), u(h)]`, where `u(*)` is the shared log-size
  expression, rather than `[x1, y1, x2, y2]`.
- Keep the Stage-1 loss surface intentionally minimal for this experiment:
  - pure hard coord-token CE on bbox coord slots
  - retain gating terms that discourage coord-vocab leakage across coord vs
    non-coord token families
  - defer soft-CE, W1, Smooth L1, bbox geometry regression, and bbox size aux
- Keep Stage-1 and standalone inference canonical at the external boundary by
  decoding model-facing center-log-size predictions back to `xyxy` before
  standardized artifact emission.
- Explicitly keep trainer-driven rollout/eval prompt rebuilding, Stage-2
  training, and confidence/scored artifact post-processing out of scope for V1;
  those surfaces remain `xyxy`-only and must fail fast if
  `center_log_size` is requested.
- Persist the full resolved prompt identity, including a prompt/template hash,
  in reproducibility metadata so prompt-text drift is auditable.

## Capabilities

### New Capabilities
- `bbox-parameterization-contract`: defines the Stage-1/inference-facing
  center-log-size bbox serialization while preserving canonical external
  `xyxy`.

### Modified Capabilities
- `dataset-prompt-variants`: dense prompts now support structured wording for
  the center-log-size bbox parameterization.
- `inference-engine`: inference can parse model-facing center-log-size bbox
  predictions and canonicalize them before standardized artifact emission.
- `coord-aux-loss`: Stage-1 coord-token supervision can run the center-log-size
  experiment in hard-CE-only mode while retaining explicit coord-gate and
  text-gate terms.

## Impact

- Expected code touch points are primarily:
  - `src/config/schema.py`
  - `src/config/loader.py`
  - `src/config/prompts.py`
  - `src/config/prompt_variants.py`
  - `src/datasets/dense_caption.py`
  - `src/datasets/builders/jsonlines.py`
  - `src/common/coord_standardizer.py`
  - `src/infer/engine.py`
  - `src/infer/pipeline.py`
  - `src/infer/artifacts.py`
  - `src/sft.py`
  - Stage-1-only teacher-forcing / coord-loss helpers
- Stage-2 is intentionally out of scope for this V1 change.
- Verification must cover:
  - Stage-1 prompt rendering for `[cx, cy, u(w), u(h)]`
  - internal `xyxy <-> center_log_size` conversion
  - hard-CE-only coord supervision with retained coord-gate and text-gate terms
  - canonical `xyxy` inference/eval artifacts
  - prompt/cache determinism for the new model-facing parameterization
