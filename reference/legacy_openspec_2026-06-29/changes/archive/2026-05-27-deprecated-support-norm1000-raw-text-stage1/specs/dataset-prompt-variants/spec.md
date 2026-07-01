## MODIFIED Requirements

### Requirement: Variant Registry and Deterministic Resolution
Prompt resolution inputs for dense prompts MUST include:
- prompt variant key,
- ordering policy,
- object field order, and
- geometry expression mode.

#### Scenario: Deterministic repeated resolution
- **WHEN** the same variant key is resolved repeatedly with the same prompt
  inputs (ordering, object field order, and geometry expression mode)
- **THEN** the resolver MUST return byte-identical system and user prompt text
  across calls.

### Requirement: Fixed Base Prompt Invariants
Dense prompt resolution SHALL keep universal invariants in a fixed base prompt
that is independent of dataset variant selection.

#### Scenario: Coord-token invariant text
- **WHEN** dense prompt resolution uses geometry expression mode
  `coord_tokens`
- **THEN** the fixed base prompt MUST describe geometry using bare
  `<|coord_k|>` literals with `k in [0,999]`.

#### Scenario: Norm1000 raw-text invariant text
- **WHEN** dense prompt resolution uses geometry expression mode
  `norm1000_text`
- **THEN** the fixed base prompt MUST describe geometry as standard JSON
  numeric coordinates on the same `[0,999]` lattice
- **AND** it MUST NOT instruct the model to emit `<|coord_k|>` tokens.
