# coordexp-swift-pack-cache-semantic-identity Specification

## Purpose
TBD - created by archiving change harden-coordexp-swift-cache-and-handoff-contracts. Update Purpose after archive.
## Requirements
### Requirement: Dataset Timestamps Are Not Semantic Identity

The semantic packing-cache fingerprint SHALL identify dataset inputs by
content, not by filesystem timestamps. Dataset determinants MUST include the
dataset content SHA256 and byte size and MUST NOT include modification-time
fields. Path, sample-limit, template, ordering, augmentation, processor,
token-identity, packing, and code-identity determinants remain unchanged;
this requirement MUST NOT broaden cache reuse across changed dataset content.

#### Scenario: Timestamp changes without content change

- **WHEN** a dataset file is touched or copied such that only its
  modification time differs while bytes are identical
- **THEN** the semantic cache fingerprint MUST remain identical
- **AND** a prepared cache for that fingerprint MUST remain reusable.

#### Scenario: Content changes

- **WHEN** dataset bytes change in any way
- **THEN** the dataset content SHA256 determinant MUST change the semantic
  fingerprint
- **AND** the previous cache MUST NOT be reused.

### Requirement: Rank-Selective Chunk Consumption Preserves Verification

Rank-local cache loading SHALL be able to skip chunks whose declared index
range contains none of the rank's required micro-step indices. Skipping MUST
NOT weaken verification: manifest-level validation of every chunk declaration
(contiguity, counts, digest syntax, path safety, file existence) MUST cover
skipped chunks, and every chunk whose payload is decoded MUST first match its
declared SHA256 and pass restricted unpickling and payload type validation.
Baseline behavior MUST NOT substitute prepare-time verification for
load-time digest verification of decoded chunks; any such trust mode requires
its own separately justified change.

#### Scenario: Corrupt chunk outside the required set

- **WHEN** a chunk not required by the rank's schedule has a corrupted payload
  while its manifest declaration remains structurally valid
- **THEN** that rank's load MAY succeed without decoding the corrupt chunk
- **AND** any rank that requires the chunk MUST fail closed on its digest.

#### Scenario: Corrupt chunk inside the required set

- **WHEN** a required chunk's payload does not match its declared SHA256
- **THEN** the rank load MUST fail closed before forward
- **AND** the rank MUST NOT rebuild the cache in place.

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
declared digest syntax without reading payload bytes — for every declared
chunk, including any chunk a distributed rank does not require. `payloads`
verification MUST additionally match every declared SHA256, restricted-unpickle
every chunk, and validate tuple, count, and micro-step types. Preparation,
cache publication, cache completeness, existing-cache reuse, and eager eval
MUST use `payloads` over every declared chunk with no rank-selective skipping.
Distributed train resolution MAY use `manifest` only when one eager validated
rank load immediately follows and completes before forward; that eager rank
load MUST digest-verify, restricted-unpickle, and type/count-validate every
chunk whose declared index range intersects the rank's resolved-schedule
required indices, and MAY skip the payload read entirely for any chunk whose
declared range does not intersect that set — skipping never weakens the
unconditional manifest-level admission above. Missing, partial, corrupt,
old-version, or semantically mismatched caches MUST be treated as cache misses
and rebuilt under the applicable verification level. Invalid required
payloads MUST fail closed; no legacy migration or decoder is permitted.

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
- **THEN** it MUST use `payloads` verification over every declared chunk
- **AND** a digest-valid forbidden or malformed payload MUST fail preparation.

#### Scenario: Distributed train resolution avoids a duplicate payload pass

- **WHEN** a distributed rank resolves an already prepared train cache
- **THEN** it MUST use `manifest` verification without decoding payloads
- **AND** the following eager rank load MUST verify every digest and decoded
  payload required by that rank's resolved schedule exactly once before
  forward, skipping the payload read only for chunks outside that required
  set.

#### Scenario: Payload changes after preparation (required chunk)

- **WHEN** a payload required by this rank's resolved schedule becomes
  corrupt after preparation but before rank loading
- **THEN** manifest admission MAY succeed
- **AND** the eager rank load MUST reject the changed payload before forward
- **AND** the distributed rank MUST NOT rebuild it in place.

#### Scenario: Payload changes after preparation (non-required chunk)

- **WHEN** a payload NOT required by this rank's resolved schedule becomes
  corrupt after preparation but before rank loading, while its manifest
  declaration remains structurally valid
- **THEN** manifest admission MAY still succeed for that declaration
- **AND** this rank's eager load MAY succeed without reading or rejecting that
  payload
- **AND** any other rank whose resolved schedule requires that chunk MUST
  still reject it before forward.

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
