## ADDED Requirements

### Requirement: Realized Cache Payload Identity Is Complete

The packing-cache fingerprint MUST bind every input and transitive producer that
can change a realized cached micro-step. At minimum, the identity SHALL include
the raw dataset content identity, rendering and parsing owners, packing and
supervision configuration, model/config assets used by preparation, the content
of all tokenizer and processor assets used to encode the sample, the complete
realized token-vocabulary grouping, and the source owners that construct or
serialize cached payloads. The manifest MUST name the identity schema version
and MUST reject a payload whose recorded determinants do not match the current
resolved determinants.

#### Scenario: Tokenizer content changes at the same path

- **WHEN** any tokenizer asset content changes without changing its filesystem
  path
- **THEN** the resolved cache fingerprint changes and the old payload is not
  admitted as current

#### Scenario: The model loader discovers a new front-end asset

- **WHEN** a new regular file under the resolved local model root could be
  selected by processor, tokenizer, configuration, template, or trusted-code
  dispatch
- **THEN** the complete non-weight front-end envelope changes the fingerprint
  or fails closed if the file cannot be classified safely
- **AND** a fixed filename allowlist cannot silently omit the new asset

#### Scenario: The model-root inventory exceeds a declared bound

- **WHEN** recursive discovery exceeds the declared total regular-file count,
  model-weight-index declaration count, hashed front-end file count, or hashed
  byte count
- **THEN** determinant construction fails closed with a bounded diagnostic even
  when the excess entries would otherwise be classified as excluded weights

#### Scenario: A model-root subtree cannot be inventoried

- **WHEN** recursive discovery encounters an unreadable directory, symlink, or
  other unsupported entry under the resolved model root
- **THEN** determinant construction fails closed instead of returning a
  shortened asset envelope

#### Scenario: Only a classified model-weight payload changes

- **WHEN** only bytes of an exact model-weight payload named by the
  content-bound weight index or by the declared conventional standalone-weight
  policy change, while all preparation-time front-end assets and semantic
  determinants remain equal
- **THEN** the packing-cache fingerprint remains unchanged because model-free
  sample preparation does not read model weights

#### Scenario: A nested file reuses a conventional weight basename

- **WHEN** a nested front-end or trusted-code directory contains a file named
  `model.safetensors`, `pytorch_model.bin`, or another conventional standalone
  weight basename that is not declared by a content-bound weight index
- **THEN** that nested file remains content-bound and changing it changes the
  cache fingerprint
- **AND** only an exact model-root-level conventional standalone payload or an
  index-declared shard is excluded from content hashing

#### Scenario: Realized vocabulary grouping changes

- **WHEN** a special-token ID, control-token classification, vocabulary size,
  or any member of the realized token-vocabulary groups changes
- **THEN** the resolved cache fingerprint changes even if the grouping source
  code and configuration text are otherwise unchanged

#### Scenario: A runtime field serialized into each micro-step changes

- **WHEN** model-logits dtype, FA2 proof policy, or another current config value
  serialized by the production `SupervisedMicroStep` constructor changes
- **THEN** the resolved cache fingerprint changes
- **AND** both the constructor source owner and micro-step schema owner are
  present in the declared determinant inventory

#### Scenario: A transitive payload owner changes

- **WHEN** a renderer, parser, raw-data geometry helper, image loader, pack
  planner, supervision mapper, position-ID owner, or cache serializer that can
  alter cached state changes
- **THEN** the cache identity changes or preparation fails because that owner is
  absent from the declared determinant inventory
- **AND** completeness is checked against an independent hard-coded owner/source
  matrix rather than a matrix derived from the production registry

#### Scenario: A cache identity version is retired

- **WHEN** a payload uses an older or unknown cache identity schema version
- **THEN** current training rejects it with the expected current version and an
  exact preparation command, without deleting or mutating the old payload

#### Scenario: A determinant changes while a cache is being prepared

- **WHEN** a dataset, referenced image, model-front-end asset, realized
  vocabulary group, or declared owner changes after payload construction starts
- **THEN** publication re-resolves the complete determinant registry after
  staged-payload validation and rejects the drift before installing the target
- **AND** no partial or stale-fingerprint final target becomes visible

### Requirement: Cache V3 Publication Is Immutable

Every cache v3 publication MUST resolve to a version namespace and semantic
fingerprint that identify one immutable committed directory. Publication MUST
make a complete staged payload visible only at a previously absent target and
MUST NOT overwrite, replace, repair, or delete an existing cache directory.
`Rebuild` SHALL mean resolving the current determinants and publishing to a
previously absent v3 namespace/fingerprint; it SHALL NOT mean rewriting the
path of a retired, incomplete, mismatched, or corrupt cache. Normal cache
preparation and training MUST NOT perform retention or garbage collection.

#### Scenario: A new v3 fingerprint is published

- **WHEN** current determinants resolve a canonical v3 target with a lowercase
  64-hex fingerprint and that target does not exist
- **THEN** preparation atomically publishes the complete staged cache at that
  version namespace/fingerprint
- **AND** no existing cache directory is changed

#### Scenario: A caller supplies a noncanonical publication path

- **WHEN** a writer receives an arbitrary directory, wrong version namespace,
  traversal-like fingerprint, or target whose basename differs from the
  resolved fingerprint
- **THEN** publication rejects the request before creating a lock, stage, or
  final cache directory

#### Scenario: Valid cache bytes are copied outside the canonical v3 target

- **WHEN** a public manifest or payload reader is given a byte-identical cache
  outside the caller-selected
  `<cache-root>/coordexp-swift-pack-cache-v3/<fingerprint>` even if the alternate
  path has the same version/fingerprint shape
- **THEN** admission rejects the noncanonical location before returning or
  hydrating cached micro-steps

#### Scenario: A canonical-looking cache path contains a symlink

- **WHEN** the version namespace, fingerprint directory, or another component
  of the caller-selected public cache target is a symlink
- **THEN** every public manifest reader, payload reader, completeness probe, and
  writer rejects the target before consuming a manifest or creating a stage

#### Scenario: A chunk path changes between authentication and decode

- **WHEN** a required chunk path is replaced, becomes a symlink or non-regular
  file, grows, shrinks, or changes metadata while admission reads it
- **THEN** admission either decodes the original manifest-authenticated bounded
  byte snapshot or fails closed
- **AND** the restricted decoder never consumes bytes different from those whose
  SHA-256 was compared with the manifest

#### Scenario: The exact v3 target is already valid

- **WHEN** the resolved v3 namespace/fingerprint already contains a complete
  payload that passes current validation
- **THEN** preparation reuses it as a read-only cache hit without changing any
  manifest or payload byte

#### Scenario: The exact v3 target already exists but is invalid

- **WHEN** the resolved v3 target is incomplete, mismatched, corrupt, or
  otherwise fails current validation
- **THEN** publication fails closed as an immutable collision
- **AND** the existing target is not repaired, replaced, renamed, or deleted
- **AND** the error records the exact target, current version, expected
  fingerprint, and bounded validation category
- **AND** the error declares automatic recovery unavailable rather than
  presenting the same-target preparation command as recovery

#### Scenario: A retired cache requires rebuild

- **WHEN** a v1 or v2 cache is present but current preparation requires v3
- **THEN** rebuild may publish only to the newly resolved v3
  namespace/fingerprint
- **AND** the retired cache remains unchanged

### Requirement: Cache Admission Precedes Expensive Training Setup

The training entrypoint MUST resolve and validate expected train and evaluation
cache identities, manifests, publication state, and rank-required payloads
before loading model weights, creating adapters or optimizers, preparing the
Accelerate runtime, or making material GPU allocations. Direct and distributed
launches SHALL use the same admission semantics.

#### Scenario: The train cache is missing

- **WHEN** a direct or distributed training launch cannot admit the required
  train-cache fingerprint
- **THEN** it fails before model loading and reports the deterministic cache
  preparation command and expected cache root

#### Scenario: A required payload is corrupt

- **WHEN** a manifest exists but a payload required by the current rank fails
  schema, checksum, or structural validation
- **THEN** the launch fails before expensive training setup and identifies the
  exact corrupt payload

#### Scenario: All required caches are valid

- **WHEN** train and evaluation cache admission succeeds
- **THEN** model loading and distributed training setup may begin and the run
  receipt records the admitted fingerprints and manifest identities

### Requirement: Rank-Selective Evaluation Cache Consumption

When evaluation work is partitioned across ranks, each rank MUST load only the
payload chunks or indexed steps assigned to that rank. The rank assignments
MUST form an exact, disjoint cover of canonical evaluation ordinals and MUST
preserve the accepted metric semantics.

#### Scenario: Disjoint evaluation shards are admitted

- **WHEN** an evaluation cache is valid and evaluation is partitioned across
  multiple ranks
- **THEN** each rank hydrates only its required payloads and the union of rank
  ordinals covers every canonical example exactly once

#### Scenario: An unassigned shard is corrupt

- **WHEN** a shard not required by the current rank is corrupt but is required
  by another rank
- **THEN** the responsible rank rejects the run, and no rank silently replaces
  the missing work or produces a partial global metric

#### Scenario: Selective and full hydration are compared

- **WHEN** the same fixed evaluation cache is consumed by selective hydration
  and by the compatibility full-hydration reference
- **THEN** predictions, canonical ordinals, and aggregate metrics are equal
  within the declared numerical tolerance

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
old-version, or semantically mismatched caches MUST be rejected as non-current.
A missing current target or a retired-version cache MAY be recovered only by
publishing the expected current cache to a previously absent version
namespace/fingerprint. An invalid directory that already occupies the expected
current target is an immutable collision and MUST fail closed. Rebuild MUST NOT
rewrite, replace, rename, or delete an existing cache directory; no legacy
migration or decoder is permitted.

#### Scenario: Old cache version is discovered

- **WHEN** the cache root contains a complete cache written with an older
  format version
- **THEN** the reader MUST reject it as non-current
- **AND** preparation MAY publish the current format only under a previously
  absent version namespace/fingerprint while leaving the old cache unchanged.

#### Scenario: Cache manifest is incomplete

- **WHEN** a manifest is missing chunks, contains a chunk gap, has mismatched
  counts, or does not declare complete status at the expected current target
- **THEN** the cache MUST NOT be consumed
- **AND** preparation MUST fail as an immutable collision without mutating,
  replacing, renaming, or deleting the incomplete directory.

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
