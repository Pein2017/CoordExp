## MODIFIED Requirements

### Requirement: Teacher-forcing target IR supports first-error correction atoms

Teacher-forcing validation SHALL support first-error correction atoms where the
selected supervised token intentionally differs from the live token at
`input_ids[target_position]`.

Normative additions:

- By default, `selected_token_id` MUST still match
  `input_ids[target_position]`.
- A mismatch is allowed only when atom provenance explicitly marks
  `allow_target_token_mismatch=true`.
- A mismatch atom MUST still have `selected_token_id` inside
  `valid_token_ids`.
- Loss modules MUST compute probability against `valid_token_ids`, not against
  the live token.
- Metrics SHOULD expose mismatch counts so operators can detect whether OPD is
  actively correcting self-prefix errors.

#### Scenario: Mismatch requires explicit provenance

- **WHEN** an atom's selected token differs from the live target-position token
- **AND** the atom does not set `allow_target_token_mismatch=true`
- **THEN** target IR validation fails fast.
- **WHEN** the provenance flag is present
- **THEN** validation succeeds and the loss supervises the selected oracle token.
