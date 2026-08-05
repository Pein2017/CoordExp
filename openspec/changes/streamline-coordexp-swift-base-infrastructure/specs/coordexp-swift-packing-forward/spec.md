## ADDED Requirements

### Requirement: Bounded Forward-Input Lookahead

Supervised training MAY overlap CPU-side Qwen forward-input construction with
GPU execution through a forward-input provider owned at the trainer/pipeline
boundary. The provider lifecycle MUST be scoped to one planned step: it
receives exactly the moved micro-step sequence of the current planned step,
prepares at most one micro-step beyond the executing one, and MUST discard
unconsumed prepared items when the step ends — normally, on producer error,
on consumer error, or on a finite-gate early break — so no stale item can
carry into a later planned step. Consumption MUST follow canonical micro-step
order with an explicit ordinal/pack identity check that fails closed on skew.
The producer MUST perform CPU-only work (file read, image content-hash
verification, decode, geometry transform, processor execution, CPU
tensorization, FA2 varlen planning) and MUST NOT execute GPU work or device
transfers; device transfer and model forward remain on the consuming step.
Queue handoff MUST be cancellation-aware in both directions so a full
depth-one queue can never deadlock shutdown; shutdown MUST join the producer
within a bounded time on every exit path, including scheduled eval and
checkpoint transitions between steps. Prepared inputs MUST be byte-identical
to synchronous construction and every existing validation (including image
content SHA256) MUST run unchanged in the producer. A producer failure MUST
surface the original error at exactly the micro-step it affects. The resolved
provider mode (overlapped or synchronous) MUST be recorded once in the
compact run record; a debug profile MAY select which provider mode is
active for measurement, and both modes MUST be semantically identical.

#### Scenario: Lookahead overlaps within a planned step

- **WHEN** the provider is active during a planned step
- **THEN** at most one prepared micro-step beyond the executing one may be
  resident
- **AND** the tensors reaching Qwen forward MUST be byte-identical to
  synchronous construction
- **AND** consumption order MUST match the canonical micro-step order with an
  identity check that fails closed on skew.

#### Scenario: Producer materialization fails

- **WHEN** image identity or materialization validation fails while preparing
  micro-step k ahead of time
- **THEN** the original contract error MUST be raised when micro-step k is
  consumed
- **AND** earlier micro-steps MUST complete normally
- **AND** the provider MUST shut down cleanly without deadlock.

#### Scenario: Finite-gate early break discards prepared work

- **WHEN** a pre-backward finite-gate consensus stops a planned step before
  its final micro-step
- **THEN** the provider MUST cancel and discard any prepared-but-unconsumed
  item for that step
- **AND** the next planned step MUST begin from its own micro-step sequence
  with no stale or skewed item
- **AND** intervening scheduled eval or checkpoint handlers MUST NOT receive
  provider state.

#### Scenario: Provider mode is selected for measurement

- **WHEN** a debug profile selects a non-default provider mode
- **THEN** training MUST produce the same loss stream and gate decisions as
  the other mode on the same inputs
- **AND** the run record MUST state which provider mode was active.

## MODIFIED Requirements

### Requirement: Deterministic Packing Cache Reuse

Packing materialization SHALL support a deterministic reusable cache for
packed micro-step plans. For the same dataset content, template,
object-ordering policy, augmentation semantics, Qwen token/processor identity,
no-resize processor controls, and `packing.global_max_length`, a
current-version cache hit MUST avoid rendering, encoding, and packing the
dataset again. The semantic fingerprint MUST include source identity for
renderer/template code, Qwen encoding/position/FA2/forward code, packing
planner, packed supervision builder, and supervision-token construction so
code changes that alter packed semantics cannot reuse stale caches. A cache
miss MUST materialize through the resolved worker policy, with 16 CPU workers
as the production default; worker count MUST be recorded in the current cache
manifest but MUST NOT participate in semantic identity or change packed
order. Distributed train assembly MUST perform at most one
digest-and-payload validation pass before forward, and that pass MAY be
restricted to the chunks that contain indices required by the rank's resolved
schedule. Every chunk whose payload is decoded MUST first match its declared
digest and pass restricted-unpickle and type validation; chunks with no
required indices MAY be skipped without payload reads while their manifest
declarations remain validated. The resulting sequence MUST preserve the exact
canonical rank-local pack sequence. Old-version, incomplete, corrupt, or
mismatched caches MUST be rejected and rebuilt rather than migrated.

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
- **AND** the eager rank loader MUST perform at most one validated payload
  pass over the chunks its schedule requires before forward
- **AND** the resulting sequence MUST match the canonical rank-local order.

#### Scenario: Chunk without required indices is skipped

- **WHEN** a rank's required index set does not intersect a chunk's declared
  range
- **THEN** the rank MUST NOT read or decode that chunk's payload
- **AND** every decoded chunk MUST still be digest-verified and
  restricted-unpickled
- **AND** the produced rank-local sequence MUST be identical to a
  full-payload-pass load.
