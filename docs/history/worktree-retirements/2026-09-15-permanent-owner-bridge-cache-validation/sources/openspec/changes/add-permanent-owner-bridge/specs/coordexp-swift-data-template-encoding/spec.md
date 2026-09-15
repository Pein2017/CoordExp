## ADDED Requirements

### Requirement: Presentation-aware owner-row rendering

Stage 1 rendering SHALL preserve one stable source-owner identity for every labeled row independently of realized row order and SHALL carry an explicit per-example annotation trust class. The first production leaf accepts only `ordinary_partial`. Test/fixture data MAY exercise `trusted_exhaustive`, but the trust class MUST come from resolved data provenance and MUST NOT be inferred from filename, object count, or annotation exhaustion. The two `geo_sorted` presentations MUST preserve and validate authored source order. Each random presentation MUST use a deterministic permutation derived from the run seed, example identity, and presentation identity, and the realized order MUST be written into reproducible run or cache evidence.

The renderer SHALL emit typed spans for each complete object row and its description and geometry regions, plus boundary records that identify the gold covered and known-uncovered source-owner sets. Permutation MUST move complete rows atomically and MUST NOT change row text, box coordinate order, wrapper tokens, terminal suffix semantics, or source-owner identity.

#### Scenario: Complete row is permuted
- **WHEN** a multi-owner example is rendered in a random presentation
- **THEN** description, wrapper, four coordinates, and row close for each owner MUST move together
- **AND** each rendered row MUST retain its source-owner identity

#### Scenario: Packed cache is rebuilt for another presentation
- **WHEN** the same source example is rendered under a different presentation identity
- **THEN** its realized order and boundary covered/uncovered records MUST reflect that presentation
- **AND** reproducibility MUST NOT depend on data-loader worker timing

#### Scenario: Trust class is absent
- **WHEN** a Stage 1 source example has no annotation trust class after resolved data defaults are applied
- **THEN** validation MUST fail before rendering instead of guessing whether unmatched atoms or exhaustion are trustworthy
