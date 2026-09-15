# coordexp-swift-pack-cache-semantic-identity Specification

## Purpose
TBD - created by archiving change harden-coordexp-swift-cache-and-handoff-contracts. Update Purpose after archive.
## Requirements
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

### Requirement: Distributed Training Consumes Prepared Caches

Multi-process training SHALL treat packing-cache preparation as a distinct
single-process startup phase. The preparation phase MUST load processor and
tokenizer state without loading model weights, MUST materialize and strictly
validate both configured train and eval caches, and MUST complete successfully
before distributed workers are launched. During distributed training, every
rank MUST independently derive and validate its semantic cache fingerprint.
A distributed cache miss MUST fail immediately with the preparation command;
it MUST NOT make workers wait inside a long-lived distributed collective or
allow any worker to rebuild the cache.

#### Scenario: Production cache is absent

- **WHEN** a multi-process training rank cannot validate its expected cache
- **THEN** startup MUST fail with `training.pack_cache_not_prepared`
- **AND** the error MUST identify `python -m src.prepare_train_cache` as the
  recovery command
- **AND** no distributed rank may begin cache materialization.

#### Scenario: Preparation succeeds before launch

- **WHEN** `python -m src.prepare_train_cache --config <path>` completes
  successfully
- **THEN** its receipt MUST identify the resolved config and strictly validated
  train and eval cache manifests
- **AND** it MUST attest that model weights were not loaded
- **AND** a later multi-process launch may consume those immutable caches
  without a cache-status collective.
