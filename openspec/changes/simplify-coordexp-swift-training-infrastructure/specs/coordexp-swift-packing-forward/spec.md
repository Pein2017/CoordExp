## ADDED Requirements

### Requirement: Packed Forward Proof Is Explicit And Non-Durable By Default

Packed-forward contract proof SHALL be collected through focused unit tests or
an explicitly enabled smoke/probe. Normal training MUST NOT emit a per-step
Qwen forward receipt. An explicit proof artifact MUST identify its scope and
contain the pack length, segment count, image-grid summary, position-reset
summary, FA2 varlen evidence, output shape, and supervision-row mapping needed
for the claim being tested.

#### Scenario: Production training runs normally

- **WHEN** a production profile executes packed Qwen forward
- **THEN** runtime MUST perform all required input and output validation
- **AND** MUST NOT emit a per-step forward proof file.

#### Scenario: Vertical smoke requests proof

- **WHEN** the explicit vertical smoke enables packed-forward proof capture
- **THEN** one bounded proof artifact MUST summarize the representative call
- **AND** the proof MUST NOT become a stream copied across every step or rank.

## MODIFIED Requirements

### Requirement: Deterministic Packing Cache Reuse

Packing materialization SHALL support a deterministic reusable cache for packed
micro-step plans. For the same dataset content, template, object-ordering
policy, augmentation semantics, Qwen token/processor identity, no-resize
processor controls, and `packing.global_max_length`, a current-version cache
hit MUST avoid rendering, encoding, and packing the dataset again. The semantic
fingerprint MUST include source identity for renderer/template code, Qwen
encoding/position/FA2/forward code, packing planner, packed supervision
builder, and supervision-token construction so code changes that alter packed
semantics cannot reuse stale caches. A cache miss MUST materialize through the
resolved worker policy, with 16 CPU workers as the production default. Worker
count MUST be recorded in the current cache manifest but MUST NOT participate
in semantic identity or change packed order. Old-version, incomplete, corrupt,
or mismatched caches MUST be rejected and rebuilt rather than migrated.

#### Scenario: Same template and data are relaunched

- **GIVEN** a complete current-version packing cache exists for the resolved
  semantic fingerprint
- **WHEN** a later run uses the same semantic inputs
- **THEN** the training pipeline MUST load the cached micro-step plan
- **AND** MUST NOT repack the JSONL again.

#### Scenario: Semantic producer code changes

- **WHEN** renderer, Qwen encoding/position/FA2/forward, packing planner,
  supervision builder, or supervision-token source identity changes
- **THEN** the packing-cache fingerprint MUST change
- **AND** the run MUST rebuild rather than trust the older cache.

#### Scenario: Cache miss on production JSONL

- **GIVEN** no complete current-version cache exists for the resolved
  fingerprint
- **WHEN** the pipeline materializes the cache
- **THEN** it MUST use 16 CPU workers by default
- **AND** the cache manifest MUST record the resolved worker count.

#### Scenario: Worker count changes

- **WHEN** a debug or implementation test changes worker count without
  changing semantic inputs
- **THEN** the cache fingerprint MUST remain unchanged
- **AND** the produced packed micro-step sequence MUST remain deterministic.

#### Scenario: Older payload version exists

- **WHEN** an otherwise complete cache uses an older payload version
- **THEN** the reader MUST treat it as a miss
- **AND** rebuild MUST occur before training consumes packed micro-steps.

### Requirement: Qwen Forward Boundary

CoordExp-Swift SHALL call the Qwen3-VL model for logits and declared model
outputs while computing all training losses in repo-owned code. Qwen forward
MUST pass `labels=None`, `use_cache=False`, avoid `inputs_embeds`, and validate
the returned output object shape instead of depending on model-side CE. Forward
MAY request a compact logits time axis for explicitly selected supervised
causal rows, but every returned row MUST keep the full vocabulary and MUST be
accompanied by an in-memory physical-pack-position map consumed and validated
by `LossContext`. Normal training MUST NOT persist that map as a per-step
receipt.

#### Scenario: Model returns built-in loss

- **WHEN** the Qwen forward result includes a model-side loss
- **THEN** the training loss runner MUST ignore it
- **AND** compute CoordExp losses from logits and `TokenSequence`.

#### Scenario: Compact supervised-row logits are requested

- **WHEN** Qwen forward uses `logits_to_keep` for supervised rows
- **THEN** the returned logits MUST have full vocabulary width
- **AND** the in-memory map MUST identify the exact physical positions kept
- **AND** `LossContext` MUST fail if any supervised atom lacks a matching kept
  physical position.

#### Scenario: Inputs embeds shortcut requested

- **WHEN** training attempts to pass `inputs_embeds`
- **THEN** Qwen forward validation MUST fail because the shortcut can bypass
  Qwen3-VL visual replacement behavior.

### Requirement: FlashAttention Varlen Inputs And Proof

Packed forward with FlashAttention SHALL always derive explicit varlen segment
inputs from `PackedSegment` boundaries, including cumulative sequence lengths
and maximum sequence lengths, and SHALL pass those inputs to Qwen forward. A 2D
zero mask over a packed row MUST NOT be accepted as the mechanism for isolated
packed attention. Explicit smoke/debug proof MUST capture representative
upstream-backed evidence that Qwen forward reached the varlen path with
cumulative and maximum sequence lengths derived from `PackedSegment`
boundaries. Production profiles MUST disable hot-path proof capture after that
evidence exists while retaining all runtime input validation.

#### Scenario: FA2 enabled with ordinary 2D mask

- **WHEN** FlashAttention is enabled and forward inputs rely only on a 2D
  padding-style mask for packed isolation
- **THEN** Qwen forward validation MUST fail before training.

#### Scenario: Explicit FA2 proof emitted

- **WHEN** an explicit packed-forward smoke/probe claims FlashAttention segment
  isolation
- **THEN** its bounded proof MUST include cumulative sequence lengths, maximum
  lengths, segment count, attention implementation, and the observed upstream
  varlen branch.

#### Scenario: Production proof capture disabled

- **WHEN** a production profile disables FA2 proof capture after representative
  evidence exists
- **THEN** runtime MUST still pass explicit FA2 varlen inputs
- **AND** normal forward calls MUST not emit proof artifacts.

## REMOVED Requirements

### Requirement: Packed Forward Receipts

**Reason**: Requiring forward contract receipts as part of normal artifact
infrastructure encourages per-step proof persistence and couples production
logging to Qwen internals.

**Migration**: Use focused module tests and one explicitly enabled, bounded
smoke/probe proof artifact. Keep normal training validation in memory and emit
no per-step Qwen receipt files.
