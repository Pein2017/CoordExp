## MODIFIED Requirements

### Requirement: Deterministic Packing Cache Reuse

Packing materialization SHALL support a deterministic reusable cache for
packed micro-step plans. For the same semantic inputs, a current-version cache
hit MUST avoid rendering, encoding, and packing the dataset again. Distributed
train assembly MUST perform no more than one full digest-and-payload validation
pass before forward, and that pass MUST preserve the exact canonical rank-local
pack sequence. Worker count MUST remain recorded but outside semantic identity,
with 16 CPU workers as the production preparation default. Old-version,
incomplete, corrupt, or mismatched caches MUST be rejected rather than migrated.

#### Scenario: Distributed rank consumes a prepared train cache

- **GIVEN** a complete current-version packing cache and resolved schedule
- **WHEN** a rank assembles its eager rank-local train tuple
- **THEN** structural manifest admission MUST NOT decode the payload
- **AND** the eager rank loader MUST perform exactly one full validated payload
  pass before forward
- **AND** the resulting sequence MUST match the canonical rank-local order.

#### Scenario: Worker count changes

- **WHEN** a debug or implementation test changes worker count without changing
  semantic inputs
- **THEN** the cache fingerprint MUST remain unchanged
- **AND** the produced packed micro-step sequence MUST remain deterministic.
