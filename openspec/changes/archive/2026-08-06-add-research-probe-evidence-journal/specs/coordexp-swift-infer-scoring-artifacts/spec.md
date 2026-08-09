## ADDED Requirements

### Requirement: Opaque execution context is strict and durable
Inference artifact execution MAY receive one caller-owned opaque execution
context and an optional execution-journal plan reference. The plan reference
MUST bind only immutable pre-execution evidence: journal schema, execution
identifier, plan fingerprint, and plan-file SHA-256. It MUST NOT require or
predict a terminal fingerprint. When supplied, the
pipeline MUST validate their complete canonical JSON values and fingerprints
before backend session creation or model work. Successful and terminal-failure
artifact publication SHALL preserve the exact context, context fingerprint,
and journal plan reference in the run manifest and summary or in one atomically
bound sidecar referenced by both.

Artifact serialization MUST reject unsupported values and non-finite numbers;
it MUST NOT silently coerce them to strings. The artifact layer SHALL NOT
interpret the context or require research-specific fields.

#### Scenario: Context survives successful inference
- **WHEN** a caller supplies valid nested context and inference completes
- **THEN** the completed artifact family preserves the same canonical context
  and fingerprint

#### Scenario: Context survives terminal failure
- **WHEN** a caller supplies valid context and inference ends through a terminal
  failure path
- **THEN** terminal status evidence preserves the same canonical context and
  fingerprint without publishing benchmark-looking scored artifacts

#### Scenario: Invalid context fails before model work
- **WHEN** caller context contains a live object, non-string mapping key, or
  non-finite number
- **THEN** inference fails before backend session creation, model work, or
  artifact publication instead of stringifying or omitting the value

#### Scenario: No-context compatibility
- **WHEN** an existing caller supplies no execution context or journal reference
- **THEN** its valid artifact schema and inference behavior remain compatible
  with the pre-change path

#### Scenario: Preflight reference does not depend on completion
- **WHEN** the current inference call will produce a work-item record for the
  referenced journal
- **THEN** preflight succeeds from the immutable plan reference without a
  terminal fingerprint or mutation of the context after model work

### Requirement: Sharded execution context agrees exactly
Before worker launch, the data-parallel controller SHALL materialize one
immutable canonical execution-context byte source and bind its locator,
file SHA-256, canonical-value fingerprint, and journal plan reference in every
worker launch contract. Each worker MUST load and verify those exact bytes
before backend session creation and preserve them in rank-local success or
terminal-failure evidence. Strict merge MUST reject a missing or mismatched
locator, file digest, value fingerprint, context value, or plan reference, and
merged success or terminal-failure evidence SHALL preserve the
controller-owned bytes.

#### Scenario: Matching shard context merges
- **WHEN** every successful shard reports the exact controller-declared context
  bytes, file digest, value fingerprint, and journal plan reference
- **THEN** merge preserves them in the top-level artifact family

#### Scenario: Shard context mismatch fails closed
- **WHEN** one shard omits or changes the declared execution context or journal
  reference
- **THEN** merge emits terminal failure evidence and does not publish
  benchmark-looking top-level scored artifacts
