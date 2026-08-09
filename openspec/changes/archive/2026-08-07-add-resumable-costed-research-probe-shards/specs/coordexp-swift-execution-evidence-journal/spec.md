## ADDED Requirements

### Requirement: Validated read-only journal diagnostics
The journal SHALL expose a read-only validated projection of accepted records
and process attempts. Each record view MUST include its sequence, work-item
identifier, attempt identifier, opaque payload, payload fingerprint, and record
digest. Each attempt view MUST distinguish a validated start, a validated
mechanical failure outcome, and a start with no accepted outcome. The
projection SHALL identify the last durable record when one exists.

The projection MUST perform the same complete plan, attempt, record, sequence,
payload, digest, and terminal validation as journal reload. It MUST NOT expose
uncommitted temporary files as records, infer an operating-system exit code,
classify payload meaning, authorize continuation, or change any persisted
journal bytes. The journal disk schema SHALL remain version 1.

#### Scenario: Caller reads durable records after interruption
- **WHEN** a fresh process inspects a non-terminal journal with accepted records
  and an attempt start without an outcome
- **THEN** it receives the validated record payloads in sequence order, the
  unfinished attempt state, and the exact last durable record identity without
  mutating the journal

#### Scenario: Persisted diagnostic input is corrupt
- **WHEN** a record or attempt digest, sequence, work-item identity, payload, or
  plan binding is invalid
- **THEN** diagnostic inspection fails with the existing typed journal artifact
  error instead of returning a partial trusted view

#### Scenario: Caller asks diagnostics to interpret an outcome
- **WHEN** an accepted opaque payload contains research-specific status or
  unmatched fields
- **THEN** the journal returns the validated payload unchanged and assigns no
  scientific label or retry decision
