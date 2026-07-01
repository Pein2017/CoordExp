## 1. Spec And Contract Updates

- [x] 1.1 Add a new geometry-expression capability that defines Stage-1 support
  for `coord_tokens` and `norm1000_text` on the same norm1000 lattice.
- [x] 1.2 Narrow the `coord-token-mode` spec so it describes the coord-token
  branch specifically instead of imposing a global mandatory contract.
- [x] 1.3 Update prompt, public-data, inference, and cache specs so they refer
  to an explicit geometry-expression mode rather than assuming coord tokens.

## 2. Config And Prompt Refactor

- [x] 2.1 Relax config loading so `custom.coord_tokens.enabled=false` is a
  supported pure-CE Stage-1 path instead of a hard failure.
- [x] 2.1a Keep `custom.coord_tokens.enabled` as the authored expression-mode
  switch for this slice; do not introduce a new top-level config key.
- [x] 2.2 Treat `coord_tokens.enabled=true|false` as the resolved geometry
  expression mode for prompt construction, cache identity, and runtime checks.
- [x] 2.3 Add dense prompt wording/examples for numeric norm1000 `xyxy`
  geometry while preserving shared `coco_80`, ordering, and field-order logic.
- [x] 2.4 Preserve the assistant payload shell as `{"objects": [...]}` with the
  existing object schema; only geometry expression may vary by mode.

## 3. Training Dataset / Builder Support

- [x] 3.1 Support canonical `train.norm.jsonl` / `val.norm.jsonl` as direct
  pure-CE Stage-1 training inputs for raw-text norm1000 runs.
- [x] 3.2 Add startup validation that rejects mismatched surfaces such as:
  - `coord_tokens.enabled=false` with `*.coord.jsonl`
  - `coord_tokens.enabled=true` with `*.norm.jsonl` when the mode would render
    incompatible assistant payloads
  - pixel-space `*.jsonl` accidentally used as the raw-text norm1000 benchmark
- [x] 3.3 Ensure JSON rendering emits standard numeric JSON bbox arrays when
  coord tokens are disabled.

## 4. Cache And Packing Safety

- [x] 4.1 Replace hardcoded `custom_coord_mode=coord_tokens` cache metadata
  with the resolved geometry-expression mode.
- [x] 4.2 Ensure encoded-sample cache reuse, static packing, and manifest
  fingerprints all distinguish `coord_tokens` vs `norm1000_text`.
- [x] 4.3 Add tests that prove caches do not cross-reuse between
  `train.coord.jsonl` and `train.norm.jsonl`.

## 5. Inference / Evaluation Workflow

- [x] 5.1 Document and support the canonical raw-text benchmark infer path:
  - `infer.mode=text`
  - `infer.pred_coord_mode=norm1000`
  - `infer.bbox_format=xyxy`
- [x] 5.2 Add focused tests showing norm1000 numeric predictions in text mode
  denormalize correctly to canonical pixel-space artifacts.
- [x] 5.3 Keep the benchmark eval contract explicit so it does not rely on
  auto mode heuristics.
- [x] 5.4 Do not introduce a new benchmark-only infer/eval flag surface when
  explicit fixed YAML settings are sufficient.
- [x] 5.5 Add numeric-span confidence scoring for raw-text canonical `xyxy`
  outputs so `gt_vs_pred_scored.jsonl` reflects real score-aware benchmark
  ranking.
- [x] 5.6 Add focused tests for numeric-span confidence alignment, stable
  failure reasons, and score-aware AP on raw-text benchmark runs.

## 6. Configs, Docs, And Smoke Verification

- [x] 6.1 Add a Stage-1 profile for `pure ce + raw text (xyxy)` that points to
  `public_data/coco/rescale_32_1024_bbox_max60_lvis_proxy/train.norm.jsonl`
  and `val.norm.jsonl`.
- [x] 6.2 Update `docs/data/PREPARATION.md`, `docs/data/CONTRACT.md`,
  `docs/training/STAGE1_OBJECTIVE.md`, and `docs/eval/WORKFLOW.md` to
  document canonical `*.norm.jsonl` as the raw-text norm1000 benchmark surface.
- [ ] 6.3 Add an end-to-end smoke workflow that trains from `.norm.jsonl`,
  then runs infer/eval with explicit `text + norm1000` settings.
