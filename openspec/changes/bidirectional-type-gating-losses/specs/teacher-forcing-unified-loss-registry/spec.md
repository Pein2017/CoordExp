# teacher-forcing-unified-loss-registry Delta

## ADDED Requirements

### Requirement: Token-type mass uses an exclusive four-family hard-SFT target

The stable `token_type_mass` teacher-forcing objective SHALL train an exclusive
token-family partition for the promoted hard-SFT detection stack.

Normative behavior:

- the objective families MUST be exactly `schema`, `coord`, `desc`, and
  `stop`;
- the contract family labels MUST map to live target IR roles as:
  - `schema -> TokenRole.SCHEMA`
  - `desc -> TokenRole.TEXT`
  - `coord -> TokenRole.COORD`
  - `stop -> TokenRole.STOP`
- this contract MUST NOT rename or require renaming the live
  `TokenRole.SCHEMA`, `TokenRole.TEXT`, `TokenRole.COORD`, or
  `TokenRole.STOP` IR roles;
- older registry names `struct` and `eos` are legacy aliases superseded by this
  contract's `schema` and `stop` labels, respectively;
- older registry names `desc` and `coord` remain compatible semantic labels,
  but the authoritative live IR roles for this contract are `TokenRole.TEXT`
  and `TokenRole.COORD`;
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
  selected token role from the teacher-forced target IR after applying the
  family-label mapping above.

#### Scenario: Schema supervision maps to TokenRole.SCHEMA

- **GIVEN** a hard-SFT supervised atom whose selected token role is
  `TokenRole.SCHEMA`
- **WHEN** `token_type_mass` is computed
- **THEN** the target family is `schema`
- **AND** `coord`, `desc`, and `stop` are treated as negative families for that
  atom.

#### Scenario: Stop supervision maps to TokenRole.STOP

- **GIVEN** a hard-SFT supervised atom whose selected token is `<|im_end|>`
- **WHEN** token-family targets are built
- **THEN** the target family is `stop`
- **AND** the live IR role is `TokenRole.STOP`
- **AND** `<|im_end|>` is not counted as `schema`.

#### Scenario: Hard-SFT mixed valid sets use selected-role targets

- **GIVEN** a hard-SFT target IR atom with multiple valid token candidates
- **AND** a concrete teacher-forced selected token
- **WHEN** `token_type_mass` builds its family target
- **THEN** the objective uses the one-hot selected token role from the target IR
- **AND** it does not derive a soft family target from valid-set candidate
  weights for this hard-SFT contract.

#### Scenario: Legacy registry aliases do not rename the IR

- **GIVEN** older registry text that refers to `struct` or `eos`
- **WHEN** this contract is applied to new hard-SFT token-type mass behavior
- **THEN** `struct` is treated only as a legacy alias for the `schema` contract
  label
- **AND** `eos` is treated only as a legacy alias for the `stop` contract label
- **AND** the live target IR continues to use `TokenRole.SCHEMA`,
  `TokenRole.TEXT`, `TokenRole.COORD`, and `TokenRole.STOP`.

#### Scenario: Out-of-family tokens are excluded from the family objective

- **GIVEN** tokenizer tokens that are pad, image, or unrelated special/control
  tokens outside `schema`, `coord`, `desc`, and `stop`
- **WHEN** family probability masses are computed
- **THEN** those tokens do not contribute to any positive or negative family
  denominator for `token_type_mass`.
