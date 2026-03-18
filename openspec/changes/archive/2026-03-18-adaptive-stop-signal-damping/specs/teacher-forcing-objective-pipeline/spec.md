# teacher-forcing-objective-pipeline Specification (Delta)

## ADDED Requirements

### Requirement: Teacher-forcing context exposes packing-safe stop-branch metadata
The teacher-forcing objective pipeline SHALL expose canonical semantic
stop-branch metadata for eligible dense object-list targets.

Normative behavior:

- the derived teacher-forcing context MUST provide segment-local semantic
  stop-branch metadata that is sufficient to identify:
  - the semantic stop token position (first terminal `']}'`),
  - the closure-tail positions that follow it (later `']}'` and `<|im_end|>`
    when present),
  - the tokenizer-specific stop/continue branch pair used by the loss
    computation,
- the metadata MUST be derived from the encoded target sequence and tokenizer,
  not from raw-string brace counting alone,
- when sequences are packed, the metadata MUST remain segment-local and MUST
  NOT leak branch positions across segment boundaries,
- if stop-signal damping is enabled and the semantic stop metadata is missing or
  ambiguous for an otherwise eligible sequence, the objective pipeline MUST fail
  fast.

#### Scenario: Packed sequences preserve local stop-branch indices
- **WHEN** packed training sequences include multiple dense object-list targets
- **THEN** each segment exposes its own semantic stop-branch metadata
- **AND** no objective module can read stop-branch indices from another
  segment.

### Requirement: token_ce accepts strict stop-signal damping configuration
The teacher-forcing pipeline SHALL support an optional stop-signal damping
configuration inside `token_ce`.

Normative behavior:

- `token_ce.config.stop_signal_damping` MUST be an optional mapping,
- the mapping MUST accept only these keys:
  - `enabled`
  - `min_weight`
  - `max_weight`
  - `branch_temperature`
  - `curve_gamma`
  - `detach_gate`
- unknown keys MUST fail fast with actionable diagnostics,
- omitted optional keys MUST resolve to these defaults:
  - `enabled: false`
  - `min_weight: 0.2`
  - `max_weight: 1.0`
  - `branch_temperature: 1.0`
  - `curve_gamma: 2.0`
  - `detach_gate: true`
- `min_weight` and `max_weight` MUST satisfy `0 <= min_weight <= max_weight`,
- `branch_temperature` MUST be strictly positive,
- `curve_gamma` MUST be strictly positive,
- `branch_temperature` MUST scale the two-token stop-vs-continue competition
  before pair normalization,
- `curve_gamma` MUST control the exponent applied to the pair-local stop belief
  before the bounded stop weight is formed,
- `detach_gate: true` MUST detach / stop-gradient the branch-local weight path,
  while `detach_gate: false` MAY allow gradients to flow through the weight
  computation,
- when enabled, stop-signal damping MUST apply only on eligible dense
  object-list targets in teacher-forced `context=gt` text CE surfaces,
- enabling stop-signal damping MUST NOT change Channel-B rollout duplicate-ul,
  geometry, or coord-regularization routing.

#### Scenario: Unknown stop-signal damping config key fails fast
- **WHEN** `token_ce.config.stop_signal_damping` contains an unknown key
- **THEN** config validation fails before training starts
- **AND** the error message names the invalid key and the allowed key set.

#### Scenario: Rollout-only objectives remain unaffected
- **WHEN** stop-signal damping is enabled for teacher-forced `context=gt`
  token CE
- **THEN** Channel-B rollout-only objectives continue to use their existing
  routing and semantics
- **AND** the stop-signal experiment does not implicitly alter duplicate-ul or
  geometry modules.
