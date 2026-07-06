## ADDED Requirements

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
- **AND** a later run MUST not reuse a cache written under the previous
  position-source identity.

#### Scenario: Qwen FA2 source changes

- **WHEN** the source identity for `src/qwen/fa2.py` changes
- **THEN** the packing-cache fingerprint MUST change
- **AND** a later run MUST rebuild rather than trusting a cache produced under
  the previous FA2-source identity.

#### Scenario: Qwen forward source changes

- **WHEN** the source identity for `src/qwen/forward.py` changes
- **THEN** the packing-cache fingerprint MUST change
- **AND** the cache manifest or receipt MUST expose enough source-identity
  evidence to explain the invalidation.

### Requirement: Cache Materialization Provenance Is Not Semantic Identity

Packing-cache materialization strategy and worker count SHALL be recorded as
operational provenance, but MUST NOT participate in the semantic cache
fingerprint. Changing only the materialization worker count MUST NOT change
the cache fingerprint or the packed micro-step order.

#### Scenario: Worker count changes

- **WHEN** the resolved packing-cache worker count changes from 16 to another
  positive internal test value
- **THEN** `build_packing_cache_fingerprint()` MUST return the same semantic
  fingerprint for otherwise identical inputs
- **AND** the cache manifest or receipt MUST still record the resolved worker
  count as provenance for newly written caches.

### Requirement: Existing Cache Payload Shape Remains Stable

This change SHALL preserve the existing supervised packing-cache payload shape.
Existing readable cache manifests MUST remain readable when they omit newer
source-identity or materialization provenance fields, but new cache writes MUST
record the expanded determinant evidence.

#### Scenario: Legacy manifest loaded

- **WHEN** a previously written cache manifest lacks the newer forward-side
  source identity fields or materialization provenance
- **THEN** cache loading MUST fail only if semantic validation requires a
  rebuild
- **AND** the loader MUST NOT crash because optional provenance fields are
  absent.

#### Scenario: New manifest written

- **WHEN** a new supervised packing cache is materialized
- **THEN** the manifest or packing receipt MUST record the expanded source
  identity determinant set
- **AND** the payload format MUST remain compatible with the existing
  micro-step cache reader.
