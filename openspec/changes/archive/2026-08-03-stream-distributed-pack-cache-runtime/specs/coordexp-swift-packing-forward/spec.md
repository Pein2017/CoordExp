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
Distributed train assembly MUST perform no more than one full
digest-and-payload validation pass before forward, and that pass MUST
preserve the exact canonical rank-local pack sequence.

#### Scenario: Same template and data are relaunched

- **GIVEN** a complete current-version packing cache exists for the resolved
  semantic fingerprint
- **WHEN** a later run uses the same semantic inputs
- **THEN** the training pipeline MUST load the cached micro-step plan
- **AND** MUST NOT repack the JSONL again.

#### Scenario: Renderer code changes

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

#### Scenario: Distributed rank consumes a prepared train cache

- **GIVEN** a complete current-version packing cache and resolved schedule
- **WHEN** a rank assembles its eager rank-local train tuple
- **THEN** structural manifest admission MUST NOT decode the payload
- **AND** the eager rank loader MUST perform exactly one full validated payload
  pass before forward
- **AND** the resulting sequence MUST match the canonical rank-local order.
