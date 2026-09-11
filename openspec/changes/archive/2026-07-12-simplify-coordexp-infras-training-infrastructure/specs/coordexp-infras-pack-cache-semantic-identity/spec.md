## ADDED Requirements

### Requirement: Cache Payload Is Current-Version-Only

The supervised packing cache SHALL be disposable acceleration state. A cache
reader MUST accept only the current cache format version, matching semantic
fingerprint, complete status, valid contiguous chunk plan, declared counts,
and readable current payload files. Missing, partial, corrupt, old-version, or
semantically mismatched caches MUST be treated as cache misses and rebuilt.
The implementation MUST NOT provide cache migrations, legacy manifest
tolerance, or old payload decoders.

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

## MODIFIED Requirements

### Requirement: Forward-Side Source Identity In Cache Fingerprint

The supervised packing cache SHALL include Qwen forward-side source producers
in its semantic fingerprint determinants. The determinant set MUST include the
source identities for `src/qwen/positions.py`, `src/qwen/fa2.py`, and
`src/qwen/forward.py` in addition to renderer/template, Qwen encoding, packing
planner, packed supervision builder, and supervision-token construction
sources.

#### Scenario: Qwen position source changes

- **WHEN** the source identity for `src/qwen/positions.py` changes
- **THEN** the packing-cache fingerprint MUST change
- **AND** a later run MUST rebuild rather than reuse the previous cache.

#### Scenario: Qwen FA2 source changes

- **WHEN** the source identity for `src/qwen/fa2.py` changes
- **THEN** the packing-cache fingerprint MUST change
- **AND** a later run MUST rebuild rather than trust the previous cache.

#### Scenario: Qwen forward source changes

- **WHEN** the source identity for `src/qwen/forward.py` changes
- **THEN** the packing-cache fingerprint MUST change
- **AND** the current cache manifest MUST expose the determinant evidence that
  caused invalidation.

### Requirement: Cache Materialization Provenance Is Not Semantic Identity

Packing-cache materialization strategy and worker count SHALL be recorded in
the current cache manifest as operational provenance, but MUST NOT participate
in the semantic cache fingerprint. Changing only materialization worker count
MUST NOT change the cache fingerprint or packed micro-step order.

#### Scenario: Worker count changes

- **WHEN** the resolved packing-cache worker count changes for an otherwise
  identical materialization
- **THEN** the semantic fingerprint and packed micro-step sequence MUST remain
  identical
- **AND** a newly written current-version cache manifest MUST record the
  resolved worker count.

## REMOVED Requirements

### Requirement: Existing Cache Payload Shape Remains Stable

**Reason**: Packing-cache payloads are derived and rebuildable. Preserving old
payload shapes and optional legacy manifest fields turns an internal
acceleration format into permanent compatibility code.

**Migration**: Increment the cache version for this change. Reject older cache
directories and rebuild them from source data and the current semantic
fingerprint.
