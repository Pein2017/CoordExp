## MODIFIED Requirements

### Requirement: Stage-2 AB residual-state trie objectives correct self-prefix first errors

Stage-2 AB `stage2_trie_ce` and `residual_set_correction` SHALL use online
self-prefix correction semantics for Channel-B compact-full rollouts.

Normative additions:

- Self-prefix tokens are roll-in context; they are not automatically positive
  labels.
- Residual-state valid actions define the positive next-token set.
- First-error target atoms may supervise an oracle token that differs from the
  live token at `target_position`.
- Multiple-positive trie supervision is allowed only while residual candidates
  remain ambiguous under the current teacher-forced prefix.
- After branch collapse, object-internal supervision is strict singleton CE
  until another residual-state ambiguity is reached.

#### Scenario: Stage-2 residual trie does not reinforce same-role malformed drift

- **WHEN** a compact-full Channel-B rollout produces a same-role but non-oracle
  token at a correction position
- **THEN** the residual trie objective trains the oracle valid action set
- **AND** the live token is counted as a corrected mismatch, not added as a
  positive.
