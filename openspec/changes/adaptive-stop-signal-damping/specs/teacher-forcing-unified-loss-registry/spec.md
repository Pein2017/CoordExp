# teacher-forcing-unified-loss-registry Specification (Delta)

## MODIFIED Requirements

### Requirement: Loss component names and contexts are canonical and shared
The system SHALL define canonical loss component names and context types that
are used consistently across Stage-1 and Stage-2 code paths.

Normative contexts:
- `gt`: pure GT teacher forcing logits/targets (Stage-1; also the CE anchor for Stage-2 Channel-A).
- `self_context`: Channel-A final-iteration logits under soft/ST coord-slot self-context.
- `rollout`: Channel-B clean-prefix teacher-forced logits under clean accepted prefix + FN injection.

Normative loss component names (minimum set; can be extended):
- `struct_ce`: token cross entropy on structure tokens, including closure-tail
  structure and EOS enforcement. When `stop_signal_ce` is enabled, semantic
  stop-branch positions are excluded from `struct_ce` and are supervised
  through `stop_signal_ce` instead.
- `desc_ce`: token cross entropy on description tokens.
- `stop_signal_ce`: branch-local weighted token cross entropy on semantic
  stop-branch positions for eligible dense object-list teacher forcing
  (optional; typically `context=gt`).
- `loss_dead_anchor_suppression`: duplicate-certified unlikelihood over
  clean-boundary divergence tokens in Channel-B rollout context.
- `coord_token_ce`: token cross entropy on coord vocabulary tokens (optional;
  typically GT context only).
- `coord_reg`: coord-subspace regularizers computed from logits/probabilities
  (optional; includes distribution/ordinal terms on coord positions and
  vocab-partition gate terms).
- `geo`: bbox-level geometry loss computed on decoded boxes.

Normative behavior:
- A single implementation of the above components MUST be reused across
  stages/channels (no duplicated definitions).
- Module pipelines and trainers MUST use these stable canonical component names
  for registry identity and objective semantics.
- `stop_signal_ce` MUST supervise only semantic stop-branch positions,
- the positive target token for `stop_signal_ce` MUST remain the terminal
  `']}'`,
- the competing `']},'` token MUST influence `stop_signal_ce` only through the
  branch-local damping calculation,
- `stop_signal_ce` MUST aggregate as a weighted mean over eligible semantic
  stop-branch positions,
- when `stop_signal_ce` is enabled, semantic stop-branch positions MUST
  contribute to `stop_signal_ce` and MUST NOT be double-counted inside
  `struct_ce`,
- when `stop_signal_ce` is disabled, those same semantic stop-branch positions
  MUST fall back to ordinary `struct_ce`,
- public training logs for registry-defined objective modules MUST follow the
  canonical metric emission contract in `trainer-metrics-components` rather
  than inventing trainer-specific aliases.

#### Scenario: Shared naming prevents silent drift
- **GIVEN** Stage-1 and Stage-2 both report semantic stop supervision
- **WHEN** metrics are logged
- **THEN** both code paths refer to the same canonical component name
  `stop_signal_ce`
- **AND** the component is not hidden inside stage-specific aliases.

#### Scenario: Semantic stop branch is counted exactly once
- **WHEN** stop-signal damping is enabled on an eligible dense object-list
  target
- **THEN** the first terminal `']}'` contributes to `stop_signal_ce`
- **AND** that same position does not contribute a second time inside
  `struct_ce`.

## ADDED Requirements

### Requirement: Semantic stop supervision remains tokenizer-aware and EOS-distinct
The unified loss registry SHALL treat semantic stop supervision as a
tokenizer-aware subset of structure supervision rather than as a generic EOS
replacement.

Normative behavior:

- semantic stop selection MUST use the tokenizer-aware `']},'` vs `']}'` branch
  pair from the encoded target,
- the later closure-tail `']}'` and `<|im_end|>` tokens MUST remain outside
  `stop_signal_ce`,
- EOS MUST remain token type `eos` and MUST continue to be counted exactly
  once,
- the stop-signal experiment MUST NOT introduce a new token type solely for
  semantic stop branches,
- if tokenizer segmentation changes such that the semantic stop branch cannot be
  identified unambiguously, the system MUST fail fast rather than silently
  reverting to raw-brace heuristics.

#### Scenario: EOS remains separate from semantic stop supervision
- **WHEN** an eligible assistant target ends with semantic stop `']}'`,
  closure-tail `']}'`, and `<|im_end|>`
- **THEN** only the first terminal `']}'` is eligible for `stop_signal_ce`
- **AND** `<|im_end|>` remains supervised only through the existing EOS path.
