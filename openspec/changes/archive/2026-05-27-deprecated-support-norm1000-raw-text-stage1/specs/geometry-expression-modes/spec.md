## ADDED Requirements

### Requirement: Pure-CE Stage-1 supports two geometry-expression modes on the same norm1000 lattice
The system SHALL support two first-class pure-CE Stage-1 geometry-expression
modes for canonical `xyxy` data:

- `coord_tokens`
- `norm1000_text`

Normative behavior:
- both modes SHALL use the same norm1000 `[0,999]` lattice,
- both modes SHALL keep canonical `bbox_format=xyxy`,
- the difference between the modes SHALL be expression only:
  - `coord_tokens` renders geometry as bare `<|coord_k|>` literals,
  - `norm1000_text` renders geometry as standard JSON numeric arrays,
- benchmark comparisons between these modes SHALL NOT require a change in image
  budget, bbox chart, or dataset lineage.
- this change only requires support for the pure-CE Stage-1 objective surface.

#### Scenario: Coord-token mode uses the tokenized prepared surface
- **GIVEN** a Stage-1 config authored for coord-token geometry expression
- **WHEN** training startup resolves dataset and prompt settings
- **THEN** the run consumes the canonical tokenized prepared surface
- **AND** assistant payload geometry is rendered as coord tokens.

#### Scenario: Norm1000 raw-text mode uses the numeric prepared surface
- **GIVEN** a Stage-1 config authored for raw-text geometry expression
- **WHEN** training startup resolves dataset and prompt settings
- **THEN** the run consumes the canonical norm1000 integer prepared surface
- **AND** assistant payload geometry is rendered as numeric JSON arrays.

### Requirement: Existing coord-token config toggles select geometry-expression mode for pure-CE Stage-1
The system SHALL use the existing `custom.coord_tokens.enabled` setting as the
pure-CE Stage-1 geometry-expression-mode selector for the first implementation
slice.

Normative behavior:
- `custom.coord_tokens.enabled=true` resolves geometry-expression mode to
  `coord_tokens`,
- `custom.coord_tokens.enabled=false` resolves geometry-expression mode to
  `norm1000_text`,
- this resolved mode SHALL be used consistently by:
  - config validation,
  - prompt construction,
  - dataset rendering,
  - cache identity,
  - smoke/debug metadata.

#### Scenario: Disabling coord tokens selects norm1000 raw text
- **GIVEN** a training config with `custom.coord_tokens.enabled=false`
- **WHEN** config resolution runs
- **THEN** the resolved geometry-expression mode is `norm1000_text`
- **AND** the config does not fail solely because coord tokens are disabled.

### Requirement: Assistant payload structure stays invariant across geometry-expression modes
The system SHALL preserve the existing dense assistant payload shell when
switching between `coord_tokens` and `norm1000_text`.

Normative behavior:
- the top-level assistant payload remains `{"objects": [...]}`
- object ordering and field-order semantics remain unchanged
- object keys remain unchanged
- only the geometry array expression changes between the two modes

#### Scenario: Coord-token and raw-text modes share the same outer payload structure
- **GIVEN** two Stage-1 runs over the same canonical `xyxy` sample
- **AND** one run uses `coord_tokens`
- **AND** the other uses `norm1000_text`
- **WHEN** their assistant payloads are rendered
- **THEN** both payloads use the same top-level `{"objects": [...]}`
- **AND** the same object structure
- **AND** only the geometry arrays differ in representation.

### Requirement: Canonical norm1000 numeric JSONL is a first-class pure-CE training surface
The system SHALL treat canonical `*.norm.jsonl` artifacts as a valid
model-facing training surface for `norm1000_text` pure-CE experiments.

Normative behavior:
- the canonical raw-text benchmark surface SHALL be the prepared
  `train.norm.jsonl` / `val.norm.jsonl` pair under the canonical preset root,
- the system SHALL NOT require an additional public-data derivation step to
  create a separate raw-text benchmark surface,
- startup validation SHALL reject accidental use of pixel-space `*.jsonl` for
  this benchmark unless a future change explicitly broadens that contract.

#### Scenario: Canonical norm JSONL is accepted
- **GIVEN** a Stage-1 config with `custom.coord_tokens.enabled=false`
- **AND** `custom.train_jsonl` points to `train.norm.jsonl`
- **WHEN** training startup validates the dataset surface
- **THEN** startup succeeds
- **AND** the run is treated as a valid norm1000 raw-text benchmark.

#### Scenario: Pixel JSONL is rejected for the norm1000 raw-text benchmark
- **GIVEN** a Stage-1 config with `custom.coord_tokens.enabled=false`
- **AND** `custom.train_jsonl` points to pixel-space `train.jsonl`
- **WHEN** training startup validates the dataset surface
- **THEN** startup fails fast
- **AND** the error explains that the benchmark expects canonical
  `*.norm.jsonl`.
