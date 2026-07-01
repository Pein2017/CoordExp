## ADDED Requirements

### Requirement: Raw-text norm1000 benchmark inference is explicit
The inference engine SHALL support a reproducible raw-text norm1000 benchmark
path for canonical `xyxy` geometry.

Normative behavior:
- the recommended benchmark infer settings SHALL be:
  - `infer.mode=text`
  - `infer.pred_coord_mode=norm1000`
  - `infer.bbox_format=xyxy`
- this path SHALL interpret predicted numeric coordinates as norm1000 values on
  `[0,999]` and denormalize them to pixels before artifact emission,
- benchmark documentation SHALL prefer this explicit path over `infer.mode=auto`
  for raw-text norm1000 experiments.
- no benchmark-specific infer/eval flag surface is required when these explicit
  YAML settings are authored directly.

#### Scenario: Explicit raw-text norm1000 benchmark path is used
- **GIVEN** a raw-text Stage-1 benchmark checkpoint trained on `train.norm.jsonl`
- **WHEN** inference runs with `infer.mode=text` and
  `infer.pred_coord_mode=norm1000`
- **THEN** predictions are denormalized correctly to canonical pixel-space
  artifacts
- **AND** evaluation does not depend on auto-mode heuristics.
