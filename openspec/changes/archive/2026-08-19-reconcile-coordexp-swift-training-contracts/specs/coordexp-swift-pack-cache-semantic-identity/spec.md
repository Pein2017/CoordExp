## ADDED Requirements

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

## MODIFIED Requirements

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
