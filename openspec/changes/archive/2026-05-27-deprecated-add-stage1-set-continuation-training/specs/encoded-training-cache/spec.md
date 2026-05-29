## MODIFIED Requirements

### Requirement: Encoded cache is limited to explicitly eligible deterministic runs
The encoded-sample cache SHALL remain limited to payloads that are pure
functions of stable base-record identity and fixed configuration.

For `stage1_set_continuation`, sampled branch continuations are runtime-derived
and MUST NOT be loaded from ordinary one-sequence encoded-sample cache entries.

Additional normative behavior:
- v1 SHALL treat encoded-sample cache as ineligible unless the implementation
  proves and records metadata-only eligibility,
- metadata-only eligibility means cached payloads preserve `assistant_payload`,
  `messages` or equivalent image/prompt identity, metadata, sample identity,
  and provenance, while branch continuations are still built at runtime,
- when cache is enabled and ineligible with `ineligible_policy=error`, startup
  MUST fail fast,
- when cache is enabled and ineligible with `ineligible_policy=bypass`, startup
  MUST continue uncached and record an explicit bypass reason in logs and run
  artifacts.

#### Scenario: Branch continuations are not reused from ordinary cache
- **GIVEN** `custom.trainer_variant: stage1_set_continuation`
- **AND** encoded-sample caching is enabled
- **WHEN** branch continuations are built
- **THEN** sampled prefix/candidate continuations are generated from current
  resolved config and seeded sample identity
- **AND** they are not read from an ordinary one-sequence SFT encoded cache.

#### Scenario: Ineligible cache request follows policy
- **GIVEN** `custom.trainer_variant: stage1_set_continuation`
- **AND** encoded-sample caching is ineligible
- **WHEN** cache policy is evaluated
- **THEN** `ineligible_policy=error` fails fast
- **AND** `ineligible_policy=bypass` continues uncached with recorded
  provenance.
