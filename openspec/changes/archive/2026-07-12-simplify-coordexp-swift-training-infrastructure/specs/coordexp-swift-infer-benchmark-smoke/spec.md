## MODIFIED Requirements

### Requirement: Adapter-enabled smoke

After tiny base-path trace validation passes, the system SHALL run a real
adapter-enabled smoke. The smoke MUST configure the base model, explicit
adapter payload path, and any selected-token embedding-delta path, then verify
the exact identity checks promised by those loaders and scored artifact
production. It MUST NOT depend on checkpoint-final or handoff metadata path
resolution.

#### Scenario: Explicit adapter checkpoint payload

- **WHEN** smoke config points directly to an adapter directory and optional
  selected-token embedding-delta directory
- **THEN** runtime MUST load those payloads, validate their declared identity
  boundaries, and record the identities actually loaded.

#### Scenario: Adapter checkpoint final

- **WHEN** a new canonical smoke config depends on `checkpoint-final`,
  `checkpoint.json`, or `checkpoint_handoff.json` to resolve payload paths
- **THEN** config or smoke-contract validation MUST fail before generation.

#### Scenario: Wrong adapter identity

- **WHEN** smoke injects an adapter that violates the standard PEFT/DoRA
  identifier, model, target, tensor, or load-result contract
- **THEN** runtime MUST fail before generation.
