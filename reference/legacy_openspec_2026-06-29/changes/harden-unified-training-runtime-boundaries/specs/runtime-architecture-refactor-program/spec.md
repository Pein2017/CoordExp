## ADDED Requirements

### Requirement: Boundary-hardening slices pass the deletion test

Runtime architecture refactors SHALL move responsibilities into modules whose
deletion would remove a real owned behavior rather than a label around behavior
still owned by a broad launcher, trainer, or owner object.

Normative behavior:

- each implementation slice MUST name the owner of runtime projection,
  target-construction policy, backend/decode adaptation, artifact/provenance
  semantics, and architecture gates;
- `src/sft.py` MUST NOT be the final owner of Stage-2 runtime policy
  resolution, pipeline manifest assembly, or rollout/eval prompt/decode
  projection;
- broad `owner: Any` or trainer-private method access MAY remain only in
  explicit migration adapters and MUST NOT become the final interface between
  shared runtime modules;
- final architecture gates MUST verify the moved ownership boundary rather than
  only banning a small set of legacy import strings;
- shadow, audit-only, or historical surfaces MUST be labeled as such in docs
  and MUST NOT be routed as current runtime authority.

#### Scenario: Boundary extraction removes real ownership from the caller

- **GIVEN** a refactor slice claims to extract a runtime or trainer
  responsibility into a dedicated module
- **WHEN** architecture review applies the deletion test
- **THEN** deleting the new module would remove the owned behavior
- **AND** the behavior is not still assembled through `src/sft.py`,
  `Stage2RolloutRuntime`, or a broad owner-shaped object.

### Requirement: Architecture gates follow moved responsibility boundaries

Runtime architecture cleanup SHALL add gates that match the final boundary after
each responsibility is moved.

Normative behavior:

- import and search gates MUST include active `src`, `scripts`, `tests`,
  `configs`, `docs`, and non-archived OpenSpec references when a legacy surface
  is declared removed;
- owner-coupling gates MUST distinguish allowed edge adapters from shared
  runtime modules that should consume resolved facts only;
- deletion gates MUST cover retired A/B or Channel-B public surfaces, removed
  manifest families, and retired specs that should not remain active
  authority;
- gates SHOULD be tightened only after the corresponding behavior is moved, so
  tests do not codify a transitional state as final architecture.

#### Scenario: Gate rejects reintroduced owner-shaped runtime coupling

- **GIVEN** prompt, backend, decode, or artifact behavior has moved behind a
  resolved shared-runtime adapter
- **WHEN** a future change adds direct trainer-private owner probing back into
  the shared runtime module
- **THEN** an architecture gate fails before the coupling becomes a supported
  interface.
