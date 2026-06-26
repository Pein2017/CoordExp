## ADDED Requirements

### Requirement: Stage-1 token-type-mass families are compact-specific
The Stage-1 `token_type_mass` objective SHALL define compact-specific
type-family names without renaming the shared teacher-forcing role registry.

Normative behavior:

- the shared teacher-forcing role vocabulary remains `struct`, `desc`,
  `coord`, and `eos` for existing Stage-2 residual-correction contracts,
- the Stage-1 `token_type_mass` family named `schema` maps to the shared
  structural role and contains compact schema/structural tokens, including
  `<|object_ref_start|>`, `<|object_ref_end|>`, `<|box_start|>`,
  `<|box_end|>`, and compact row separators when those tokens are present in
  the tokenizer,
- the Stage-1 family named `desc` maps to the shared description role and
  contains free-text object description tokens after excluding schema, coord,
  stop, padding, unknown, and control-special tokens,
- the Stage-1 family named `coord` maps to the shared coord role and contains
  exactly `<|coord_0|>` through `<|coord_999|>`,
- the Stage-1 family named `stop` maps to the shared eos role and contains the
  Qwen chat stop marker `<|im_end|>` only,
- text terminators such as `<|endoftext|>` or `<|end_of_text|>` MUST NOT be
  used as semantic stop targets for compact Stage-1 teacher-forcing; they
  remain padding/control tokens outside the four active Stage-1 type families.

#### Scenario: Stage-1 families do not rewrite Stage-2 roles
- **WHEN** compact Stage-1 target IR is evaluated by `token_type_mass`
- **THEN** it may report family names `schema`, `desc`, `coord`, and `stop`
- **AND** the shared Stage-2 role names `struct`, `desc`, `coord`, and `eos`
  remain valid for residual-correction contracts.

### Requirement: Token-type-mass is bidirectional and exclusive
The `token_type_mass` term SHALL compute an exclusive four-way family-mass loss
over Stage-1 schema, coord, desc, and stop families.

Normative behavior:

- `objective.terms.token_type_mass` MUST expose only `enabled` and `weight`.
- The selected atom role MUST map to exactly one target family.
- Family logits MUST be computed by logsumexp over the full-vocabulary
  log-probabilities for each family.
- The type loss denominator MUST include only the four active family logits.
  Padding, unknown, control-special, and other out-of-family tokens are excluded
  from the type-loss denominator.
- The per-atom type loss MUST be the negative log-probability of the target
  family after softmax over the four family logits.
- This objective is bidirectional: increasing the target-family mass lowers the
  loss, while increasing any non-target active-family mass raises the loss.
- The term MUST NOT replace ordinary CE or valid-set likelihood; it is an
  additive family-mass pressure term.

#### Scenario: Object transition is an exclusive schema-versus-stop decision
- **GIVEN** a compact teacher-forcing atom whose next token is
  `<|object_ref_start|>`
- **WHEN** `token_type_mass` is evaluated
- **THEN** the target family is `schema`
- **AND** probability mass on `<|im_end|>` contributes through the competing
  `stop` family.

#### Scenario: Assistant stop is not schema
- **GIVEN** a compact teacher-forcing atom whose next token is `<|im_end|>`
- **WHEN** `token_type_mass` is evaluated
- **THEN** the target family is `stop`
- **AND** schema structural tokens are competing non-target family mass.

#### Scenario: Coordinate tokens use the coordinate family
- **GIVEN** a compact teacher-forcing atom whose selected token role is `coord`
- **WHEN** `token_type_mass` is evaluated
- **THEN** the target family is `coord`
- **AND** the coordinate family contains exactly the 1000 coord tokens.
