## ADDED Requirements

### Requirement: Training-aligned prompt construction
Inference prompt construction SHALL share or delegate to supervised-training template semantics.
Inference MUST NOT copy a divergent prompt implementation for the compact
object-box-closed task.

#### Scenario: Prompt token parity
- **WHEN** a tiny or smoke inference run renders a prompt
- **THEN** local rendered prompt token ids match backend prompt token ids

#### Scenario: Template identity recorded
- **WHEN** an inference artifact set is written
- **THEN** prompt/template identity and object-ordering policy are recorded in
  run manifest or provenance

### Requirement: Same input example family
V1 inference SHALL consume the same offline JSONL example family used by training and forward eval.
The required fields include image path, image dimensions, GT objects, and
prompt-relevant fields.

#### Scenario: Valid example row
- **WHEN** an input JSONL row contains image path, dimensions, GT objects, and
  required prompt fields
- **THEN** inference can build the prompt and preserve GT in output rows

#### Scenario: Missing image dimensions
- **WHEN** an input row lacks required image dimensions
- **THEN** benchmark-eligible inference fails before backend generation and
  does not write row artifacts that imply the input was evaluated

### Requirement: No-resize image plan
Inference SHALL use Qwen no-resize image processing with `do_resize=False`.
Every run SHALL write `image_plan.jsonl` with one row per input/raw row,
including declared dimensions, decoded dimensions, patch size, merge size,
expected `image_grid_thw`, observed `image_grid_thw`, raw patch rows, merged
visual-token count, and batch-order index.

#### Scenario: Valid no-resize image
- **WHEN** an image satisfies processor-derived no-resize constraints
- **THEN** image processing runs with `do_resize=False` and writes matching
  expected and observed grid evidence

#### Scenario: Invalid no-resize dimensions
- **WHEN** an image violates no-resize patch/merge constraints
- **THEN** inference fails before processor reshape or model generation

#### Scenario: Image-plan row count
- **WHEN** an inference run completes over N input rows
- **THEN** `image_plan.jsonl` contains N rows keyed to the same row identity as
  `gt_vs_pred.jsonl`

### Requirement: Processor model vision parity
Inference SHALL verify that Qwen processor vision parameters match the loaded model vision configuration.
The checked fields MUST include processor `patch_size`, `merge_size`,
`temporal_patch_size`, model `vision_config.patch_size`, model
`vision_config.spatial_merge_size`, and model
`vision_config.temporal_patch_size`. The verified values MUST be recorded in
`image_plan.jsonl` or run manifest evidence.

#### Scenario: Matching processor model vision fields
- **WHEN** processor and model vision fields match
- **THEN** inference records the matched patch, merge, and temporal-patch
  values in image-plan or manifest evidence

#### Scenario: Processor model vision mismatch
- **WHEN** processor patch, merge, or temporal-patch values disagree with the
  loaded model vision config
- **THEN** inference fails before backend generation and does not mark the run
  benchmark eligible

### Requirement: Compact object-box parser
V1 parsing SHALL target the compact object-box-closed format aligned with training.
V1 MUST NOT parse old JSON assistant response formats unless a later OpenSpec
change promotes that compatibility.

#### Scenario: Valid compact object
- **WHEN** generated text contains a valid compact object with schema wrappers
  and four coordinate tokens
- **THEN** the parser emits one canonical prediction object with generated-token
  evidence

#### Scenario: JSON assistant response
- **WHEN** generated text uses an old JSON assistant response format
- **THEN** V1 parser treats it as unsupported rather than silently accepting it

### Requirement: Prediction order preservation
Inference SHALL preserve model prediction order after decoding.
It MUST NOT geo-sort predictions after generation.

#### Scenario: Reversed generated order
- **WHEN** the model generates object B before object A
- **THEN** output prediction order remains B then A even if geometry sorting
  would reorder them

### Requirement: Parser diagnostics and metric eligibility
Every parsed output row SHALL carry inline parser status and metric-eligibility fields.
Required fields include parser id, parser policy, metric-bearing status, parse
status, valid prediction count, dropped prediction count, and dropped prediction
objects. Detailed `parse_diagnostics.jsonl` rows SHALL be keyed by stable row id
or line index.

#### Scenario: Accepted row
- **WHEN** all generated objects parse successfully
- **THEN** the row records metric-bearing parser status and zero dropped
  predictions

#### Scenario: Accepted with drops
- **WHEN** at least one generated object is valid and at least one object span is
  malformed
- **THEN** valid objects are preserved, dropped objects are recorded with reason
  and raw text, and row diagnostics reflect the drop count

#### Scenario: All spans dropped
- **WHEN** no generated object span is valid
- **THEN** the row remains present with empty predictions and non-metric or
  diagnostic parser status

### Requirement: Shared geometry semantics
The system SHALL reuse or deepen shared CoordExp-swift geometry semantics for generated coordinates.
This applies to coordinate-token recognition, bbox validation, and
norm1000-to-pixel conversion. Inference MUST NOT implement a second
incompatible geometry conversion policy.

#### Scenario: Coordinate conversion
- **WHEN** generated coordinate tokens represent a valid bbox
- **THEN** conversion to pixel `xyxy` follows the shared geometry helper

#### Scenario: Degenerate bbox
- **WHEN** generated coordinate tokens imply a degenerate or out-of-range bbox
- **THEN** the object is dropped or marked invalid with a geometry diagnostic
