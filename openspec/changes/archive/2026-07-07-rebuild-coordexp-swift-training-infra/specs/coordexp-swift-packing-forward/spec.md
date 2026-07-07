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

### Requirement: Deterministic Packing Cache Reuse

Packing materialization SHALL support a deterministic reusable cache for packed
micro-step plans. For the same dataset content, template, object-ordering
policy, Qwen token/processor identity, no-resize processor controls, and
`packing.global_max_length`, a cache hit MUST avoid rendering, encoding, and
packing the dataset again. The semantic fingerprint MUST also include source
identity for the renderer/template code, Qwen encoding code, packing planner,
packed supervision builder, and supervision-token construction so code changes
that alter examples cannot reuse stale caches. A cache miss MUST materialize
the cache through a forced default of 16 CPU workers. The worker count MUST be
recorded in the cache manifest or packing receipt as operational provenance,
but it MUST NOT participate in the semantic cache fingerprint because changing
worker count must not create a different training example order or supervision
contract.

#### Scenario: Same template and data are relaunched

- **GIVEN** a complete packing cache exists for the dataset/template/Qwen
  encoding/processor/global-length/code identity
- **WHEN** a later run uses the same semantic inputs
- **THEN** the training pipeline MUST load the cached micro-step plan
- **AND** MUST NOT repack the JSONL again.

#### Scenario: Renderer code changes

- **WHEN** renderer, Qwen encoding, packing planner, supervision builder, or
  supervision-token construction source identity changes
- **THEN** the packing-cache fingerprint MUST change
- **AND** the run MUST rebuild rather than trusting an older semantic cache.

#### Scenario: Cache miss on production JSONL

- **GIVEN** no complete cache exists for the resolved packing fingerprint
- **WHEN** the training pipeline materializes the packing cache
- **THEN** it MUST use 16 CPU workers by default
- **AND** the manifest or packing receipt MUST record the resolved worker
  count.

#### Scenario: Worker count changes

- **WHEN** a debug or implementation test changes the worker count without
  changing dataset, template, ordering, Qwen identity, processor controls, or
  `packing.global_max_length`
- **THEN** the cache fingerprint MUST remain unchanged
- **AND** the produced packed micro-step sequence MUST remain deterministic.

### Requirement: Supervision Position Mapping

Packing SHALL convert logical target positions into physical packed positions
with an invertible mapping. The conversion MUST preserve
`target_position`, derived `logits_position = target_position - 1`,
`example_id`, segment id, token type, and span provenance.
For standard causal losses, the derived `logits_position` MUST belong to the
same `PackedSegment` as `target_position`. A supervised atom targeting the
first physical token of any packed segment MUST be rejected or dropped by the
approved supervision builder; it MUST NOT shift to the previous segment.

#### Scenario: Packed atom debug trace

- **WHEN** a packed `TokenAtom` is inspected in a debug receipt
- **THEN** the receipt MUST identify its physical target position
- **AND** MUST allow reconstruction of the original encoded example id and
  logical target position.

#### Scenario: Causal shift would cross segment boundary

- **WHEN** a supervised target is the first physical token of a non-first
  packed segment
- **THEN** loss-context construction MUST reject or omit that atom according to
  the approved supervision policy
- **AND** it MUST NOT use the previous segment's final token as the logit row.

### Requirement: Qwen MRoPE Position Inputs

Packed Qwen forward inputs SHALL use Qwen3-VL-compatible position ids and MRoPE
metadata. CoordExp code MAY call or mimic Transformers helpers, but MUST prove
that the packed inputs match Qwen3-VL expectations for text tokens, image
tokens, and segment isolation. V1 packed training MUST use the installed
Qwen3-VL 4-row HF boundary shape `[text,t,h,w]` when the installed forward
expects a separate text row. Row 0 MUST contain text-position ids whose reset
points match the packed segment table and FA2 cumulative sequence lengths.
Rows 1-3 MUST contain temporal, height, and width MRoPE ids computed per
segment from that segment's expanded `input_ids` and visual grids before
concatenation. CoordExp MUST NOT infer positions by running the upstream helper
once over the whole packed row if that produces continuous positions across
segments.

#### Scenario: Two packed segments require MRoPE ids

- **WHEN** two encoded examples are concatenated into one packed row
- **THEN** Qwen position ids MUST be computed per segment and concatenated only
  after per-segment MRoPE construction
- **AND** text-position ids MUST reset at each packed segment start
- **AND** validation MUST assert that position-id reset points match the
  `PackedSegment` boundaries and FA2 cumulative-sequence split
- **AND** validation MUST fail if installed Qwen changes the expected row count
  or row meaning.

#### Scenario: MRoPE helper behavior changes upstream

- **WHEN** a Transformers update changes Qwen3-VL position-helper output shape
  or meaning
- **THEN** Qwen forward validation MUST fail before training accepts the new
  behavior silently.

### Requirement: Qwen Forward Boundary

CoordExp-swift SHALL call the Qwen3-VL model for logits and declared model
outputs while computing all training losses in repo-owned code. V1 Qwen forward
MUST pass `labels=None`, `use_cache=False`, avoid `inputs_embeds`, and validate
the returned output object shape instead of depending on model-side CE. The
forward MAY request a compact logits time axis for explicitly selected
supervised causal rows, but every returned row MUST keep the full vocabulary and
MUST be accompanied by a physical-pack-position map consumed by `LossContext`.

#### Scenario: Model returns built-in loss

- **WHEN** the Qwen forward result includes a model-side loss
- **THEN** the training loss runner MUST ignore it
- **AND** compute CoordExp losses from logits and `TokenSequence`.

#### Scenario: Compact supervised-row logits are requested

- **WHEN** Qwen forward uses `logits_to_keep` for supervised rows
- **THEN** the returned logits MUST have full vocabulary width
- **AND** receipts MUST record the exact physical positions kept
- **AND** `LossContext` MUST fail if any supervised atom lacks a matching kept
  physical position.

#### Scenario: Inputs embeds shortcut requested

- **WHEN** a V1 training path attempts to pass `inputs_embeds`
- **THEN** Qwen forward validation MUST fail because the shortcut can bypass
  Qwen3-VL visual replacement behavior.

### Requirement: FlashAttention Varlen Inputs And Proof

Packed forward with FlashAttention SHALL always derive explicit varlen segment
inputs from `PackedSegment` boundaries, including cumulative sequence lengths
and maximum sequence lengths, and SHALL pass those inputs to Qwen forward. A 2D
zero mask over a packed row MUST NOT be accepted as the mechanism for isolated
packed attention. Branch-level proof capture is policy-gated: smoke/debug gates
MUST record representative upstream-backed evidence that Qwen forward reached
the varlen path with cumulative sequence lengths and max lengths derived from
`PackedSegment` boundaries, while production profiles MAY disable hot-path proof
capture after that evidence exists. When emitted, "equivalent" proof MUST name
the installed upstream branch and record the observed varlen evidence.

#### Scenario: FA2 enabled with ordinary 2D mask

- **WHEN** FlashAttention is enabled and the forward inputs rely only on a 2D
  padding-style mask for packed isolation
- **THEN** Qwen forward validation MUST fail before training.

#### Scenario: FA2 branch proof emitted

- **WHEN** a packed forward debug receipt claims FlashAttention segment
  isolation
- **THEN** it MUST include cumulative sequence lengths, max sequence lengths,
  segment count, resolved attention implementation, and evidence of the
  upstream varlen branch used for the call.

#### Scenario: FA2 branch proof disabled for production throughput

- **WHEN** a packed supervised training config sets the FA2 branch-proof policy
  to a production-disabled mode after representative FA2 evidence has already
  been recorded by smoke/probe artifacts
- **THEN** the runtime MUST still pass explicit FA2 varlen inputs to Qwen
  forward
- **AND** forward receipts MUST record the resolved branch-proof policy
- **AND** forward receipts MUST NOT claim branch-proof evidence for calls where
  hot-path proof capture was disabled.

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
