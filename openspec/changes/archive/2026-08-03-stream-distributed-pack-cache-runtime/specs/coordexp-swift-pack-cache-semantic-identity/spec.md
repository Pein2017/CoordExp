## MODIFIED Requirements

### Requirement: Cache Payload Is Current-Version-Only

The supervised packing cache SHALL be disposable acceleration state. Cache
verification MUST use an explicit level. `manifest` verification MUST validate
the current format version, matching semantic fingerprint, complete status,
valid contiguous chunk plan, declared counts, safe existing payload paths, and
declared digest syntax without reading payload bytes. `payloads` verification
MUST additionally match every declared SHA256, restricted-unpickle every chunk,
and validate tuple, count, and micro-step types. Preparation, cache publication,
cache completeness, existing-cache reuse, and eager eval MUST use `payloads`.
Distributed train resolution MAY use `manifest` only when one eager validated
rank load immediately follows and completes before forward. Missing, partial,
corrupt, old-version, or semantically mismatched caches MUST be treated as
cache misses and rebuilt under the applicable verification level. Invalid
caches MUST fail closed; no legacy migration or decoder is permitted.

#### Scenario: Old cache version is discovered

- **WHEN** the cache root contains a complete cache written with an older
  format version
- **THEN** the reader MUST reject it as a cache miss
- **AND** the training pipeline MUST rebuild it with the current format before
  use.

#### Scenario: Cache manifest is incomplete

- **WHEN** a manifest is missing chunks, contains a chunk gap, has mismatched
  counts, or does not declare complete status
- **THEN** the cache MUST NOT be consumed
- **AND** rebuild MUST be the recovery path.

#### Scenario: Preparation verifies payloads

- **WHEN** the single-process preparation command validates a cache
- **THEN** it MUST use `payloads` verification
- **AND** a digest-valid forbidden or malformed payload MUST fail preparation.

#### Scenario: Distributed train resolution avoids a duplicate payload pass

- **WHEN** a distributed rank resolves an already prepared train cache
- **THEN** it MUST use `manifest` verification without decoding payloads
- **AND** the following eager rank load MUST verify every digest and decoded
  payload exactly once before forward.

#### Scenario: Payload changes after preparation

- **WHEN** a payload becomes corrupt after preparation but before rank loading
- **THEN** manifest admission MAY succeed
- **AND** the eager rank load MUST reject the changed payload before forward
- **AND** the distributed rank MUST NOT rebuild it in place.
