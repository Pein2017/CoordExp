## ADDED Requirements

### Requirement: Channel-B first-error OPD uses oracle valid actions, not live-token positives

Online compact-full Channel-B residual-set training SHALL treat the model's
self-prefix tokens as context observations. At a correction position, the
positive token set SHALL be derived from residual-state `ValidAction` records.

Normative behavior:

- A live self-prefix token that is not one of the oracle valid actions MUST NOT
  be added to the positive token set.
- The atom MAY record the live token as provenance for diagnostics.
- If the oracle selected token differs from `input_ids[target_position]`, the
  target atom MUST be marked as a first-error correction and validated as an
  intentional target-token mismatch.
- Deterministic schema, stop, and singleton object-internal actions are
  singleton valid-set CE cases.

#### Scenario: Wrong live token is corrected instead of reinforced

- **GIVEN** a self-prefix token at the target position is text token `A`
- **AND** the residual-state oracle valid action set contains only text token
  `B`
- **WHEN** the residual-set target IR is built
- **THEN** `valid_token_ids` contains `B`
- **AND** `valid_token_ids` does not contain `A`
- **AND** the atom records that the live token was corrected.

### Requirement: Multiple positives are scoped to active residual-state ambiguity

The residual-set objective SHALL allow multiple positive next tokens only when
the current residual-state trie position is genuinely ambiguous.

Normative behavior:

- Multiple positives are allowed when the current prefix still maps to multiple
  valid residual candidates and `enumerate_valid_actions(...)` returns multiple
  next-token actions for the same slot.
- Once the teacher-forced prefix selects a branch that narrows the residual
  candidate set to a singleton, subsequent positions for that branch MUST use
  singleton CE unless a later residual-state ambiguity is explicitly reached.
- Coordinate positions MAY have multiple positives only when multiple residual
  candidates remain active at that coordinate slot.
- The loss MUST expose diagnostics for ambiguous targets, singleton strict
  targets, and coordinate ambiguous targets.

#### Scenario: Shared description prefix collapses before strict suffix CE

- **GIVEN** two residual objects share the first description token
- **WHEN** the first shared token is supervised
- **THEN** the valid token set may be singleton while candidate identity remains
  ambiguous.
- **WHEN** a later description or coordinate token selects one object
- **THEN** subsequent object-internal positions are strict singleton CE for that
  selected teacher branch.
