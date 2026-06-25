# teacher-forcing-unified-loss-registry Delta

## ADDED Requirements

### Requirement: Token-type mass uses an exclusive four-family hard-SFT target

The stable `token_type_mass` teacher-forcing objective SHALL train an exclusive
token-family partition for the promoted hard-SFT detection stack.

Normative behavior:

- the objective families MUST be exactly `schema`, `coord`, `desc`, and
  `stop`;
- the four families MUST be mutually exclusive for objective purposes;
- semantic stop supervision MUST target the assistant stop token `<|im_end|>`
  as `stop`, not as `schema`;
- coordinate supervision MUST use the coordinate-token family `coord`;
- description supervision MUST use the free-description family `desc`;
- compact structural/template supervision MUST use `schema`;
- control, pad, and tokenizer special tokens outside these four families MUST
  be excluded from the family objective denominator;
- at each supervised atom, the objective MUST increase probability mass for the
  target family and suppress mass assigned to the other three families;
- under `objective.profile: hard_sft`, the target family MUST be the one-hot
  `selected_token_role` family from the teacher-forced target IR.

#### Scenario: Schema supervision suppresses non-schema family mass

- **GIVEN** a hard-SFT supervised atom whose `selected_token_role` is `schema`
- **WHEN** `token_type_mass` is computed
- **THEN** the target family is `schema`
- **AND** `coord`, `desc`, and `stop` are treated as negative families for that
  atom.

#### Scenario: Stop supervision is distinct from schema supervision

- **GIVEN** a hard-SFT supervised atom whose selected token is `<|im_end|>`
- **WHEN** token-family targets are built
- **THEN** the target family is `stop`
- **AND** `<|im_end|>` is not counted as `schema`.

#### Scenario: Hard-SFT mixed valid sets use selected-role targets

- **GIVEN** a hard-SFT target IR atom with multiple valid token candidates
- **AND** a concrete teacher-forced selected token
- **WHEN** `token_type_mass` builds its family target
- **THEN** the objective uses the one-hot `selected_token_role` family
- **AND** it does not derive a soft family target from valid-set candidate
  weights for this hard-SFT contract.

#### Scenario: Out-of-family tokens are excluded from the family objective

- **GIVEN** tokenizer tokens that are pad, image, or unrelated special/control
  tokens outside `schema`, `coord`, `desc`, and `stop`
- **WHEN** family probability masses are computed
- **THEN** those tokens do not contribute to any positive or negative family
  denominator for `token_type_mass`.
