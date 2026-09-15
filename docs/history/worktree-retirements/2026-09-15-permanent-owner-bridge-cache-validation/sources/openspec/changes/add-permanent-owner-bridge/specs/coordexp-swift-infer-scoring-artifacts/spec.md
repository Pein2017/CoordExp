## ADDED Requirements

### Requirement: Permanent bridge inference provenance

Single-rank and merged inference manifests for a permanent-bridge run SHALL bind the bridge payload fingerprint, manifest fingerprint, architecture profile, companion adapter/embedding identities, HF backend mode, and lifecycle-sidecar identity. Strict shard merge MUST reject disagreement in any of those fields. The lifecycle sidecar is diagnostic and MUST NOT change the existing per-prediction score formula or evaluator row schema.

#### Scenario: Bridge shards disagree
- **WHEN** two completed HF shards name different bridge payloads, profiles, or lifecycle policies
- **THEN** strict merge MUST fail before publishing a canonical top-level scored artifact

#### Scenario: Bridge lifecycle sidecar is moved with artifacts
- **WHEN** a complete bridge inference artifact family is relocated
- **THEN** manifest and provenance validation MUST continue to resolve it through content identity rather than an original absolute path

