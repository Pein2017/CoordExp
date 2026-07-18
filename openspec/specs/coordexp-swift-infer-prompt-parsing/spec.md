# coordexp-swift-infer-prompt-parsing Specification

## Purpose
TBD - created by archiving change build-coordexp-swift-inference-infra. Update Purpose after archive.
## Requirements
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
Every successful run SHALL write `image_plan.jsonl` with one row per input/raw
row, including declared dimensions, decoded dimensions, image content SHA-256,
patch size, merge size, expected `image_grid_thw`, expected raw patch rows,
expected merged visual-token count, backend projection-evidence kind, and
batch-order index. HF rows MUST include observed `image_grid_thw` from executed
tensors. vLLM rows MAY set observed `image_grid_thw` absent only when they
instead include the hash-validated executed media identity, actual returned
prompt ids, exact multimodal placeholder range/count, processor identity, and
effective `do_resize: false` kwargs; the observed placeholder count MUST equal
the expected merged visual-token count.

#### Scenario: Valid no-resize image
- **WHEN** HF executes an image satisfying no-resize constraints
- **THEN** expected and executed tensor grid evidence match

#### Scenario: Valid vLLM no-resize image
- **WHEN** vLLM executes an image satisfying no-resize constraints
- **THEN** returned prompt ids and multimodal placeholder count match the shared
  expected expansion and the row records the executed media hash

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

### Requirement: Semantic single-image decode request
The shared inference request SHALL contain authoritative executable
`chat_text`, `input_prompt_token_ids` with exactly one image placeholder,
`expected_executed_prompt_token_ids` with the no-resize visual expansion,
image path, image content SHA-256, declared and decoded image dimensions, row
identity, validated logical transform id, and shared policies. Human-facing
task prompt text MAY remain in the
prompt record as diagnostics but MUST NOT be an alternate backend execution
input. The request MUST NOT contain HF pixel tensors, vLLM multimodal objects,
or backend-owned model inputs. Multi-image and video inputs MUST fail before
backend construction.

#### Scenario: One image request
- **WHEN** a valid single-image row is prepared
- **THEN** both backends receive the same semantic prompt and image identity

#### Scenario: Geometry-augmented row
- **WHEN** a row declares `hflip`, `vflip`, or `hvflip`
- **THEN** the backend applies that transform exactly once to decoded RGB
  pixels before native projection and records the transformed RGB8 pixel hash

#### Scenario: Diagnostic prompt text diverges from chat text
- **WHEN** human-facing task prompt text differs from executable `chat_text`
- **THEN** both backends execute only `chat_text` and its two pinned token-id
  forms

#### Scenario: Multiple images
- **WHEN** a row resolves to more than one image
- **THEN** request preparation fails before GPU model loading

### Requirement: Backend-honest no-resize image evidence
Shared image planning MUST validate decoded dimensions, Qwen vision parameters,
expected no-resize grid, patch rows, visual-token counts, and image content
SHA-256 before decode. Each backend MUST reopen and hash the image immediately
before native projection and MUST fail if the bytes or decoded dimensions
differ. Each backend MUST apply the request's logical transform before native
projection and record a canonical RGB8 hash of the transformed pixels. HF MUST
record the grid observed from its executed tensors. vLLM MUST pass the same
transformed, hash-validated in-memory image with `do_resize: false` and record
the actual returned prompt ids, multimodal placeholder ranges/count, processor
identity, and no-resize kwargs. Locally materialized HF tensors MUST NOT be
described as the tensors executed by vLLM.

#### Scenario: vLLM image projection
- **WHEN** vLLM decodes a valid image
- **THEN** image-plan evidence distinguishes shared expected geometry from the
  vLLM-native executed input and records no-resize policy

#### Scenario: Dimension mismatch
- **WHEN** decoded image dimensions disagree with declared required dimensions
- **THEN** inference fails before backend generation

#### Scenario: Image changes after request preparation
- **WHEN** image bytes at the path differ from the semantic request SHA-256
- **THEN** the backend fails before native projection or generation

### Requirement: Exact backend prompt-token parity
Both initial generation and forced raw decode replay MUST submit the unexpanded
input prompt form plus the same image and `do_resize: false`. Every backend MUST return the
exact prompt token ids it executed. For vLLM, this evidence MUST come from
`RequestOutput.prompt_token_ids`, not a backend echo or self-attestation. Those
ids MUST equal `expected_executed_prompt_token_ids`, including the expanded
image placeholders and assistant generation transition. Raw replay MUST return
that same executed prompt prefix and then reproduce the authoritative generated
ids through incremental decode. Mismatch MUST be fatal even when generation
otherwise succeeds.

#### Scenario: vLLM retokenizes differently
- **WHEN** vLLM reports prompt ids different from the shared prompt record
- **THEN** the row fails before parsing and no score is published

#### Scenario: Replay submits already-expanded image ids
- **WHEN** raw replay submits executed visual-placeholder ids plus the image
- **THEN** request validation fails before replay because the raw pass must
  start from the shared unexpanded prompt form
