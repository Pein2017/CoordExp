## ADDED Requirements

### Requirement: Residual-set builders compile into shared teacher-forcing IR

The teacher-forcing objective pipeline SHALL receive residual-set Stage-2
supervision through shared IR records, not through Stage-2-specific loss-module
logic.

Normative behavior:

- Stage-2 residual builders MAY create upstream `CorrectionEvent`,
  row-observation, scan-state, and UL-provenance records.
- Objective runners MUST consume only the shared IR atom fields defined by the
  residual-set correction spec, including `logit_position`, `target_position`,
  `allowed_token_roles`, `selected_token_role`, `valid_token_ids`,
  `selected_token_id`, `loss_weight`, `coord_role`, `loss_tags`, and
  `provenance`.
- Loss modules MUST NOT parse raw rollout text, redo bbox legality, redo
  duplicate detection, redo GT/UL matching, rerun UL promotion, or recompute the
  semantic remaining set.
- Every atom MUST pass next-token alignment validation before training loss is
  trusted.

#### Scenario: Loss module does not inspect Stage-2 provenance logic

- **WHEN** a dirty-prefix correction event compiles to a `SupervisionAtom`
- **THEN** the objective pipeline computes loss from atom-local fields
- **AND** it does not inspect whether the event originated from FN, FP,
  duplicate, malformed, or UL-mining provenance.

### Requirement: Residual-set selected roll-in follows ValidAction transitions

The teacher-forcing objective pipeline SHALL construct selected roll-in paths
from residual-state `ValidAction` records, not from bare token ids.

Normative behavior:

- `SupervisionAtom.valid_token_ids` MUST be derived from the action set returned
  by the residual-state machine.
- When a selected token is needed for the corrected teacher-forced input, the
  selected token MUST come from a `ValidAction`.
- The selected action's next state MUST drive subsequent suffix and atom
  construction.
- If a selected action has invalid or empty next state, strict validation MUST
  fail before loss computation.

#### Scenario: Selected x1 drives later bbox context

- **WHEN** a corrected roll-in selects one valid `x1` action from an ambiguous
  same-description object set
- **THEN** subsequent teacher-forced input and atoms use that action's next
  state
- **AND** object-coordinate mixing is not possible through an independent
  transition path.

### Requirement: Residual-set target sequence assembly preserves causal alignment

The teacher-forcing objective pipeline SHALL assemble residual-set sequences so
that each atom supervises the correct next-token logits.

Normative behavior:

- Rollout prefix labels MUST be masked by default.
- Constructed suffix labels/atoms MAY be active according to the target IR.
- Correction atoms may attach to validated logit positions inside a kept raw
  rollout prefix, but they supervise expert next-token choices, not the actual
  generated next token.
- `logit_position + 1 == target_position` MUST hold whenever
  `target_position` is represented.
- The selected target token MUST match the constructed/corrected input sequence
  at `target_position`.
- Bounds, padding, prompt/assistant boundary crossing, and template span
  alignment MUST be validated.

#### Scenario: Early EOS correction uses logits before EOS

- **WHEN** a rollout emits `<|im_end|>` while remaining objects are nonempty
- **THEN** the correction atom reads logits at the token immediately before EOS
- **AND** the valid target is the next valid continuation under the residual
  set, not the emitted EOS token.

### Requirement: Template adapter owns schema spans for residual-set suffixes

The teacher-forcing objective pipeline SHALL use the template boundary adapter
as the source of truth for residual-set suffix schema and spans.

Normative behavior:

- Constructed suffix ids MUST come from adapter-rendered/tokenized Stage-1
  compatible template output.
- The adapter MUST use the existing detection template/tokenization surfaces
  rather than Stage-2 string helpers for compact row rendering.
- Template-rendered schema tokens, including separators/newline if present, are
  deterministic schema singleton atoms.
- Stage-2 residual builders MUST pass supervision objects and suffix-start
  state to the adapter rather than authoring schema fragments directly.
- Adapter validation MUST prevent duplicate deterministic schema tokens at
  prefix/suffix joins.

#### Scenario: Suffix join does not duplicate schema

- **WHEN** a kept raw prefix already includes a deterministic separator
- **THEN** the adapter starts the suffix after that separator
- **AND** emitted atoms do not train a duplicated separator.

### Requirement: Residual-set pipeline rejects coordinate repair spans

The teacher-forcing objective pipeline SHALL NOT require or implement local
raw-rollout coordinate repair spans for residual-set Stage-2 v1.

Normative behavior:

- A single coordinate failure MUST NOT compile to `bbox_tail_from_anchor`.
- Raw-rollout coordinate divergence is handled as an uncommitted row or dirty
  context according to residual-set scan policy.
- Coordinate atoms may be emitted only for constructed teacher-forced suffix
  positions or optional clean GT stabilizer positions.
- Loss modules MUST remain agnostic to whether a coordinate token came from GT
  or promoted UL; that distinction is represented by atom weight/provenance.

#### Scenario: Invalid coordinate row emits no coordinate repair span

- **WHEN** row classification marks a bbox row invalid because geometry is not
  positive-area
- **THEN** event-to-IR compilation emits no coordinate repair atoms for that row
- **AND** later valid boundary atoms may still be emitted if recovery is valid.
