## ADDED Requirements

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
Both initial generation and raw replay MUST submit the unexpanded input prompt
form plus the same image and `do_resize: false`. Every backend MUST return the
exact prompt token ids it executed. For vLLM, this evidence MUST come from
`RequestOutput.prompt_token_ids`, not a backend echo or self-attestation. Those
ids MUST equal `expected_executed_prompt_token_ids`, including the expanded
image placeholders and assistant generation transition. Raw replay MUST submit
`input_prompt_token_ids + generated_token_ids` and its returned executed ids
MUST equal `expected_executed_prompt_token_ids + generated_token_ids`.
Mismatch MUST be fatal even when generation otherwise succeeds.

#### Scenario: vLLM retokenizes differently
- **WHEN** vLLM reports prompt ids different from the shared prompt record
- **THEN** the row fails before parsing and no score is published

#### Scenario: Replay submits already-expanded image ids
- **WHEN** raw replay submits executed visual-placeholder ids plus the image and
  would expand the image a second time
- **THEN** request validation fails before replay

## MODIFIED Requirements

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

#### Scenario: Valid HF no-resize image
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
