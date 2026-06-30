## ADDED Requirements

### Requirement: No-Padding Packed Training Row

V1 supervised training SHALL use one physical packed sequence per rank/step as
the standard forward path. Multiple encoded examples MAY be concatenated under
`packing.global_max_length`, but padding multiple independent examples into a
conceptual batch MUST NOT be the standard supervised path.

#### Scenario: New example fits current pack

- **WHEN** adding the next encoded example keeps the pack length at or below
  `packing.global_max_length`
- **THEN** the packer MUST append it to the current physical sequence
- **AND** preserve a segment boundary for that example.

#### Scenario: New example would overflow current pack

- **WHEN** adding the next encoded example would exceed
  `packing.global_max_length`
- **THEN** the packer MUST commit the current pack
- **AND** start a new pack with the new example.

### Requirement: Segment Isolation

Packed examples SHALL remain semantically isolated. Attention, position ids,
supervision placement, loss accounting, and metrics MUST prevent later examples
from attending to or being evaluated as continuations of earlier packed
examples unless a future continuous-stream recipe is explicitly approved.

#### Scenario: Two examples in one packed row

- **WHEN** two encoded examples are placed into one physical packed sequence
- **THEN** the forward inputs MUST include enough boundary information for Qwen
  forward and loss accounting to isolate the two segments.

### Requirement: Supervision Position Mapping

Packing SHALL convert logical target positions into physical packed positions
with an invertible mapping. The conversion MUST preserve
`target_position`, derived `logits_position = target_position - 1`,
`example_id`, segment id, token type, and span provenance.

#### Scenario: Packed atom debug trace

- **WHEN** a packed `TokenAtom` is inspected in a debug receipt
- **THEN** the receipt MUST identify its physical target position
- **AND** MUST allow reconstruction of the original encoded example id and
  logical target position.

### Requirement: Qwen MRoPE Position Inputs

Packed Qwen forward inputs SHALL use Qwen3-VL-compatible position ids and MRoPE
metadata. CoordExp code MAY call or mimic Transformers helpers, but MUST prove
that the packed inputs match Qwen3-VL expectations for text tokens, image
tokens, and segment isolation. The active upstream row-shape contract MUST be
verified: either 3-row `[t,h,w]` MRoPE ids or 4-row `[text,t,h,w]` ids when the
installed Qwen3-VL forward expects a separate text row.

#### Scenario: Two packed segments require MRoPE ids

- **WHEN** two encoded examples are concatenated into one packed row
- **THEN** Qwen position ids MUST be constructed with per-segment reset/concat
  parity against the upstream Qwen3-VL helper behavior
- **AND** validation MUST fail if installed Qwen changes the expected row
  count or row meaning.

#### Scenario: MRoPE helper behavior changes upstream

- **WHEN** a Transformers update changes Qwen3-VL position-helper output shape
  or meaning
- **THEN** Qwen forward validation MUST fail before training accepts the new
  behavior silently.

### Requirement: Qwen Forward Boundary

CoordExp-swift SHALL call the Qwen3-VL model for logits and declared model
outputs while computing all training losses in repo-owned code. V1 Qwen forward
MUST pass `labels=None`, `use_cache=False`, request full logits, avoid
`inputs_embeds`, and validate the returned output object shape instead of
depending on model-side CE.

#### Scenario: Model returns built-in loss

- **WHEN** the Qwen forward result includes a model-side loss
- **THEN** the training loss runner MUST ignore it
- **AND** compute CoordExp losses from logits and `TokenSequence`.

#### Scenario: Inputs embeds shortcut requested

- **WHEN** a V1 training path attempts to pass `inputs_embeds`
- **THEN** Qwen forward validation MUST fail because the shortcut can bypass
  Qwen3-VL visual replacement behavior.

### Requirement: FlashAttention Varlen Proof

Packed forward with FlashAttention SHALL prove explicit varlen segment
isolation using cumulative sequence lengths and maximum sequence lengths or an
equivalent upstream-backed mechanism. A 2D zero mask over a packed row MUST NOT
be accepted as the proof of isolated packed attention.

#### Scenario: FA2 enabled with ordinary 2D mask

- **WHEN** FlashAttention is enabled and the forward inputs rely only on a 2D
  padding-style mask for packed isolation
- **THEN** Qwen forward validation MUST fail before training.

### Requirement: Visual Replacement Remains In Transformers

CoordExp-swift SHALL leave Qwen visual tower execution and replacement of
image placeholders with visual features inside the Transformers Qwen3-VL
forward path or an explicitly approved Qwen wrapper. CoordExp-swift MUST
validate placeholder/grid agreement before forward, but MUST NOT hand-roll the
model architecture.

#### Scenario: Placeholder-grid mismatch

- **WHEN** the number of image placeholders does not agree with the visual grid
  metadata
- **THEN** Qwen forward validation MUST fail before model forward.

#### Scenario: Expanded image-token count validated

- **WHEN** V1 single-image forward inputs are built from a no-resize processor
  payload
- **THEN** the selected image-token count MUST equal
  `sum(image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2] //
  merge_size**2)` for the payload
- **AND** an off-by-one placeholder or grid mismatch MUST fail before model
  forward.

### Requirement: Packed Forward Receipts

Each smoke or debug run SHALL be able to emit a compact Qwen forward contract
receipt containing pack length, segment count, image grid metadata, position-id
summary, FA2 branch evidence, output shape, and supervision row counts.

#### Scenario: Vertical smoke Qwen receipt

- **WHEN** the vertical smoke completes a forward pass
- **THEN** `debug/qwen_forward_contract.json` or an equivalent linked receipt
  MUST summarize the packed forward contract for that run.
