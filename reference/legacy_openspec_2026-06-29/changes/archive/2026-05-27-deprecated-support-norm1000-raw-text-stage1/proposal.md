## Why

CoordExp's Stage-1 training mainline is currently hard-wired to the
coord-token expression family:

- config loading requires `custom.coord_tokens.enabled: true`
- prompt resolution only supports `coord_mode=coord_tokens`
- cache identity hardcodes `custom_coord_mode=coord_tokens`
- the canonical prepared dataset surface is treated as `*.coord.jsonl` even
  though the public-data pipeline already emits parallel `*.norm.jsonl`
  artifacts on the same norm1000 lattice

That design made sense while the project was focused on `<|coord_*|>` token
experiments, but it now blocks a high-value benchmark:

- `pure CE + raw text (xyxy)`

This benchmark is specifically intended to isolate whether the added
`<|coord_*|>` vocabulary introduces optimization or duplication costs beyond
the underlying `[0,999]` norm1000 coordinate lattice. The first supported slice
is intentionally narrow:

- Stage-1 only
- pure CE only
- canonical `xyxy` only

The comparison should reuse the same canonical prepared dataset family
(`public_data/coco/rescale_32_1024_bbox_max60_lvis_proxy`) and change only the
model-facing coordinate expression:

- coord-token mode:
  - `train.coord.jsonl`
  - assistant payload uses bare `<|coord_k|>` literals
- norm1000 raw-text mode:
  - `train.norm.jsonl`
  - assistant payload uses standard JSON numeric coordinates in `[0,999]`

The current codebase already has most of the data-plane ingredients:

- canonical `*.norm.jsonl` files already exist under `public_data/`
- the dataset builder can already emit numeric arrays when
  `coord_tokens_enabled=false`
- inference text mode already understands normalized numeric predictions when
  `infer.pred_coord_mode=norm1000`
- standardized downstream artifacts already expect canonical pixel-space boxes
  after denormalization

The missing piece is architectural: the training, prompt, and cache contracts
still treat coord-token mode as the only supported geometry expression. This
change removes that bottleneck and makes norm1000 raw-text `xyxy` a supported,
reproducible Stage-1 benchmark surface.

## What Changes

- Introduce a first-class Stage-1 geometry-expression split:
  - `coord_tokens`
  - `norm1000_text`
- Keep both modes on the same canonical `xyxy` bbox chart and the same
  norm1000 `[0,999]` lattice so benchmarks differ only in expression, not
  scale.
- Restrict the supported objective surface in this change to pure CE Stage-1
  training; soft coordinate supervision, W1, bbox auxiliary losses, and Stage-2
  remain out of scope.
- Reuse the existing `custom.coord_tokens.enabled` switch as the first
  geometry-expression selector:
  - `true` -> coord-token mode
  - `false` -> norm1000 raw-text mode
- Allow Stage-1 training to consume canonical `*.norm.jsonl` directly when
  running in `norm1000_text` mode.
- Refactor prompt resolution so dense prompts can describe either:
  - bare coord tokens (`<|coord_k|>`)
  - numeric norm1000 coordinates (`0..999`)
- Keep the assistant output structure unchanged across the benchmark:
  - top-level payload remains `{"objects": [...]}`
  - object schema remains unchanged
  - only the geometry expression changes between coord-token and raw-text mode
- Replace coord-token-only config guards with mode-aware validation:
  - `custom.coord_tokens.enabled=true` means coord-token expression mode
  - `custom.coord_tokens.enabled=false` means norm1000 raw-text expression mode
- Update cache and packing fingerprints so the geometry-expression mode is
  explicit and cache-safe.
- Add numeric-span confidence scoring for raw-text canonical `xyxy` runs so
  `gt_vs_pred_scored.jsonl` reflects real score-aware benchmark ranking.
- Keep bbox chart semantics unchanged in this change:
  - `bbox_format=xyxy` remains the benchmark target
  - non-canonical bbox branches (`cxcy_logw_logh`, `cxcywh`) remain separate
    work and are not widened here
- Document a reproducible benchmark workflow that trains from:
  - `train.norm.jsonl` / `val.norm.jsonl`
  - with explicit/fixed `infer.mode=text` and
    `infer.pred_coord_mode=norm1000`
  - and always converts norm1000 outputs to pixel space with image
    `width/height` before evaluation or visualization

## Capabilities

### New Capabilities
- `geometry-expression-modes`: defines the Stage-1 contract for
  `coord_tokens` vs `norm1000_text` on the same norm1000 lattice for pure CE
  Stage-1 runs.

### Modified Capabilities
- `coord-token-mode`: narrows from a global mandatory contract to the rules for
  the coord-token branch specifically.
- `dataset-prompt-variants`: dense prompt resolution now supports numeric
  norm1000 geometry instructions in addition to coord-token wording.
- `inference-engine`: benchmark guidance and config contracts explicitly
  support raw-text norm1000 prediction evaluation via `mode=text` and
  `pred_coord_mode=norm1000`.
- `public-data-pipeline`: `*.norm.jsonl` becomes a documented first-class
  training surface for norm1000 raw-text experiments, not just an intermediate
  artifact.
- `encoded-training-cache`: cache identity must distinguish coord-token and
  norm1000 raw-text expression modes.
- `confidence-postop`: raw-text canonical `xyxy` runs gain numeric-span
  confidence scoring instead of falling back to constant scores.

## Impact

- Expected code touch points are primarily:
  - `src/config/schema.py`
  - `src/config/loader.py`
  - `src/config/prompts.py`
  - `src/datasets/dense_caption.py`
  - `src/datasets/builders/jsonlines.py`
  - `src/sft.py`
  - `src/infer/engine.py`
  - Stage-1 configs under `configs/stage1/`
- Expected documentation/spec touch points are:
  - `docs/data/CONTRACT.md`
  - `docs/data/PREPARATION.md`
  - `docs/eval/WORKFLOW.md`
  - `docs/training/STAGE1_OBJECTIVE.md`
- Verification must cover:
  - Stage-1 startup with `train.norm.jsonl`
  - numeric JSON assistant payload rendering on `[0,999]`
  - prompt parity and explicit mode hashing
  - cache invalidation across coord-token vs raw-text expression modes
  - inference/eval with `infer.mode=text` and `infer.pred_coord_mode=norm1000`
  - numeric-span confidence scoring for raw-text bbox outputs
  - guaranteed norm1000-to-pixel conversion before visualization/evaluation
