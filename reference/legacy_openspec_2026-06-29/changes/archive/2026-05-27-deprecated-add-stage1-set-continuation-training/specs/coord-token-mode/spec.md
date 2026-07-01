## MODIFIED Requirements

### Requirement: Single-forward Stage-1 loss composition
For ordinary one-sequence Stage-1 coord-gated SFT, the system MUST continue to
compute non-coord CE and coord `softCE+W1` from a single model forward pass, and
coord-token targets MUST remain masked out of ordinary full-vocab CE.

The `stage1_set_continuation` trainer variant is a repeated-forward branch
scoring variant, not an ordinary one-sequence SFT variant.

Additional normative behavior for `stage1_set_continuation`:
- each candidate branch uses one independent forward,
- non-coord candidate-entry labels contribute full-vocab logprob to the MP
  candidate score,
- coord-token candidate-entry labels contribute coord-vocab-normalized logprob
  to the MP candidate score,
- ordinary full-vocab CE MUST NOT be used as the coord-token contribution to
  the candidate score,
- branch-local coord auxiliary losses MAY reuse the same branch logits but MUST
  remain separate from the MP candidate score.

#### Scenario: Set-continuation coord slots use coord-vocab scoring
- **GIVEN** `custom.trainer_variant: stage1_set_continuation`
- **AND** a candidate entry contains `<|coord_*|>` labels
- **WHEN** the candidate score is computed
- **THEN** coord-token slots are scored over the coord-token vocabulary
- **AND** non-coord slots are scored over the full vocabulary.

#### Scenario: Ordinary Stage-1 single-forward contract is unchanged
- **GIVEN** ordinary Stage-1 SFT with coord-gated `softCE+W1`
- **WHEN** training computes loss
- **THEN** the existing single-forward ordinary Stage-1 contract remains in
  force.
