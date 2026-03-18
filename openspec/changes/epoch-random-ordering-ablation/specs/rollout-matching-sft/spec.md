# rollout-matching-sft Spec Delta

This is a delta spec for change `epoch-random-ordering-ablation`.

## MODIFIED Requirements

### Requirement: Canonical encoding and supervision index sanity checks
The ONE teacher-forced forward pass SHALL use the exact same prompt/messages encoding (chat template + image tokens placement) as rollout generation.

Dense prompt/message construction used by rollout generation and any trainer-driven evaluation prompt rebuilding SHALL resolve from the same shared dense prompt contract as training, using equivalent values for:
- prompt variant,
- object field order, and
- object ordering.

Labels SHALL align to assistant response tokens only; prompt tokens MUST be `ignore_index` (or equivalent).

The trainer MUST implement three engineering sanity checks:
- (a) prompt+image prefix tokenization matches generation (e.g., `len` and/or hash of the prompt token IDs),
- (b) all supervised `coord_token_indices` fall within the assistant-label span (never into the prompt span),
- (c) trainer-driven evaluation prompt rebuilding resolves equivalent dense policy instructions to rollout generation for equivalent prompt inputs.

#### Scenario: Random-order rollout and trainer-eval prompts stay aligned
- **GIVEN** a rollout-aware trainer with dense prompting enabled
- **AND** the active config resolves `object_ordering: random`
- **WHEN** the trainer builds rollout-generation prompts and trainer-driven evaluation prompts
- **THEN** both surfaces use equivalent random-order policy instructions
- **AND** prompt-prefix sanity checks continue to compare against the resolved prompt tokenization.

#### Scenario: Supervision indices are validated against assistant span
- **GIVEN** a sample with computed coord token indices for self-context supervision in the rollout prefix
- **WHEN** the trainer builds loss masks for the forward pass
- **THEN** it asserts every supervised coord index lies within the assistant portion of the encoded sequence
- **AND** it errors clearly if any index points into the prompt/image prefix (preventing silent misalignment).
