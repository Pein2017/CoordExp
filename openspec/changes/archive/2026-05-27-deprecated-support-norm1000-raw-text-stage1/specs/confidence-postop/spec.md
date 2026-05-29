## ADDED Requirements

### Requirement: Raw-text canonical xyxy benchmark uses numeric-span confidence scoring
The confidence post-op SHALL support canonical raw-text `xyxy` benchmark runs
whose bbox coordinates are emitted as numeric norm1000 text instead of
coord-token literals.

Normative behavior:
- when a benchmark run is authored with:
  - `infer.mode=text`
  - `infer.pred_coord_mode=norm1000`
  - `infer.bbox_format=xyxy`
- the scoring path SHALL recover per-object scores from numeric bbox spans and
  token traces,
- `gt_vs_pred_scored.jsonl` for this benchmark SHALL reflect those recovered
  scores rather than a constant-score fallback.

#### Scenario: Raw-text benchmark produces score-aware scored artifact
- **GIVEN** a raw-text Stage-1 benchmark run with canonical `xyxy` output
- **WHEN** the confidence/scoring stage runs
- **THEN** `gt_vs_pred_scored.jsonl` contains real per-object scores derived
  from numeric bbox spans
- **AND** official mAP uses those scores rather than constant ones.

### Requirement: Numeric-span confidence uses raw bbox numbers, not pixel inversion
Numeric-span confidence reconstruction SHALL use the raw bbox numeric values
from `raw_output_json`, not reverse-engineered spans from canonical pixel-space
boxes.

Normative behavior:
- the scoring path SHALL extract expected bbox numeric values from
  `raw_output_json.objects[*].bbox_2d`,
- it SHALL align those numeric bbox components against `generated_token_text`
  deterministically,
- it SHALL use the aligned numeric spans plus token logprobs to derive an
  auditable per-object score,
- it SHALL NOT infer numeric spans by converting standardized pixel boxes back
  into norm1000 strings.

#### Scenario: Numeric bbox numbers are scored from raw payload spans
- **GIVEN** a raw-text prediction whose `raw_output_json` contains
  `"bbox_2d": [123, 45, 678, 910]`
- **WHEN** confidence scoring reconstructs bbox spans
- **THEN** it aligns those four numeric components against the token trace
- **AND** derives the object score from those aligned numeric spans.

### Requirement: Norm1000 predictions are denormalized before eval and visualization
For raw-text norm1000 benchmark runs, evaluation and visualization SHALL
consume canonical pixel-space boxes obtained by denormalizing norm1000
predictions with the sample's image `width/height`.

Normative behavior:
- inference SHALL decode raw-text norm1000 predictions to canonical pixel-space
  boxes before writing `gt_vs_pred.jsonl`,
- confidence scoring MAY still use raw numeric spans from the raw payload for
  scoring,
- downstream visualization and evaluation SHALL consume only the canonical
  pixel-space boxes.

#### Scenario: Raw-text norm1000 predictions render in pixel space
- **GIVEN** a raw-text benchmark prediction on the `[0,999]` lattice
- **WHEN** visualization or evaluation artifacts are materialized
- **THEN** the bbox is first denormalized using the image `width/height`
- **AND** the rendered/scored artifact uses pixel-space coordinates.
