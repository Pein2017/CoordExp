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
declared digest syntax without reading payload bytes for every declared chunk.
`payloads` verification MUST additionally match every declared SHA256,
restricted-decode every chunk, and validate tuple, count, and micro-step types.
Preparation, publication, completeness checks, existing-cache reuse, and eager
eval MUST use `payloads` over every declared chunk. Distributed train
resolution MAY use `manifest` only when an eager validated rank load follows
before forward; that load MUST authenticate and decode every chunk intersecting
the rank's resolved schedule and MAY skip payload reads outside that set.

Missing, partial, corrupt, old-version, or semantically mismatched caches MUST
be rejected as non-current. A missing current target or retired-version cache
MAY be recovered only by publishing the expected current cache to a previously
absent version namespace and fingerprint. An invalid directory already
occupying the expected current target is an immutable collision and MUST fail
closed. Preparation and training MUST NOT rewrite, replace, repair, rename,
delete, migrate, or garbage-collect an existing cache target.

#### Scenario: Old cache version is discovered

- **WHEN** a complete retired cache exists but the current version resolves a
  different absent namespace and fingerprint
- **THEN** the reader MUST reject the retired cache as non-current
- **AND** preparation MAY publish only to the absent current target while
  leaving the retired cache unchanged.

#### Scenario: Expected current target is invalid

- **WHEN** the expected current target exists but is incomplete, mismatched,
  corrupt, or otherwise invalid
- **THEN** admission and preparation MUST fail as an immutable collision
- **AND** the target MUST NOT be repaired, replaced, renamed, or deleted
- **AND** the error MUST NOT claim that rerunning preparation can overwrite it.

#### Scenario: Existing current target is valid

- **WHEN** the expected current target passes full current-version validation
- **THEN** preparation MUST reuse it as a read-only cache hit
- **AND** no manifest or payload byte may change.

#### Scenario: Preparation verifies payloads

- **WHEN** the single-process preparation command validates a cache
- **THEN** it MUST use `payloads` verification over every declared chunk
- **AND** a digest-valid forbidden or malformed payload MUST fail preparation.

#### Scenario: Distributed train resolution avoids a duplicate payload pass

- **WHEN** a distributed rank resolves an already prepared train cache
- **THEN** it MAY use manifest-only admission
- **AND** the following eager rank load MUST authenticate and validate every
  required chunk exactly once before forward.

#### Scenario: Required payload changes after admission

- **WHEN** a payload required by the rank's resolved schedule changes after
  manifest admission but before rank loading
- **THEN** eager rank loading MUST reject it before forward
- **AND** the rank MUST NOT rebuild or repair the target in place.

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

### Requirement: Realized Cache Payload Identity Is Complete

The packing-cache fingerprint MUST bind the content of every input and declared
producer that can change a realized cached micro-step. This includes raw data
and referenced images, rendering and parsing semantics, tokenizer and processor
assets used during preparation, realized vocabulary groups, packing and
supervision configuration, runtime fields serialized into cached micro-steps,
and the source owners that construct or serialize those payloads. The manifest
MUST name the identity schema version and MUST reject payloads whose recorded
determinants differ from the current resolved determinants.

#### Scenario: A transitive cached-payload producer changes

- **WHEN** a declared data, template, encoding, packing, supervision,
  position, micro-step-construction, or serialization owner changes in a way
  covered by source identity
- **THEN** the semantic cache fingerprint MUST change
- **AND** the old payload MUST NOT be admitted as current.

#### Scenario: A front-end asset changes at the same path

- **WHEN** tokenizer, processor, template, trusted-code, image, or raw dataset
  content used by preparation changes without changing its path
- **THEN** the semantic cache fingerprint MUST change.

#### Scenario: Determinants drift during preparation

- **WHEN** a bound input, asset, vocabulary grouping, or producer identity
  changes after staged payload construction begins
- **THEN** publication MUST re-resolve identity and reject the drift before
  installing the target
- **AND** no stale-fingerprint final target may become visible.

### Requirement: Cache Admission Precedes Expensive Training Setup

The training entrypoint MUST resolve and validate the expected train and eval
cache identities, manifests, publication state, and rank-required payloads
before loading model weights, creating adapters or optimizers, constructing the
training runtime, or making material GPU allocations. Direct and distributed
launches SHALL use the same fail-closed admission semantics.

#### Scenario: A required cache is absent

- **WHEN** a direct or distributed launch cannot admit the required cache
  fingerprint
- **THEN** it MUST fail before expensive training setup
- **AND** the error MUST report the expected cache target and deterministic
  single-process preparation command.

#### Scenario: A required payload is corrupt

- **WHEN** a required chunk fails schema, checksum, path-safety, restricted
  decoding, or structural validation
- **THEN** launch MUST fail before forward or model mutation
- **AND** distributed workers MUST NOT rebuild the cache in place.

#### Scenario: Required caches are admitted

- **WHEN** train and eval cache admission succeeds
- **THEN** expensive training setup MAY begin
- **AND** the run record MUST bind the admitted fingerprints and manifest
  identities.
