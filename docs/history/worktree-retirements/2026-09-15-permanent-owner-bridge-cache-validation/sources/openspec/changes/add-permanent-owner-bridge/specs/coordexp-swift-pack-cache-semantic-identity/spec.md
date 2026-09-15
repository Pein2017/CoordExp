## ADDED Requirements

### Requirement: Owner-bridge cache semantic identity

A cache containing permanent-owner-bridge training records SHALL bind the bridge record schema version, architecture-profile identity, annotation trust class and its provenance, source-owner identities and boxes, realized presentation order, presentation seed identity, boundary covered/uncovered records, typed row regions, renderer identity, tokenizer identity, processor identity, and forward-side source identity. Materialization worker count and filesystem location MUST NOT change semantic identity.

Older payloads that lack any required owner or presentation field MUST be rejected and rebuilt rather than upgraded in place. Cache validation MUST verify the payload fingerprint before distributed training consumes it.

#### Scenario: Random presentation changes
- **WHEN** `random-1` and `random-2` realize different object-row orders for the same example
- **THEN** their bridge cache semantic identities MUST differ

#### Scenario: Scalar-density cache is present
- **WHEN** a cache built for the removed scalar owner-density proposal is discovered
- **THEN** Stage 1 validation MUST reject it as schema-incompatible and rebuild from source data

#### Scenario: Annotation trust changes
- **WHEN** identical rendered content is reclassified from `ordinary_partial` to `trusted_exhaustive`
- **THEN** its cache semantic identity MUST change before any negative or final-null supervision changes
