## ADDED Requirements

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

## MODIFIED Requirements

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
