# adaptive-stop-signal-damping Specification (Delta)

## ADDED Requirements

### Requirement: Dense object-list stop branches are identified from tokenizer-aware serialized targets
The system SHALL identify semantic stop branches from the actual encoded
assistant target used for teacher forcing, not from raw JSONL samples or
generic EOS-only heuristics.

Normative behavior:

- for dense CoordJSON assistant targets serialized as a top-level
  `{"objects":[...]}` container, the semantic stop branch MUST be the first
  terminal `']}'` token emitted immediately after the final object's last coord
  token,
- the competing continue token MUST be the non-final object-boundary token
  `']},'` from the same tokenizer and chat-template pair,
- the later top-level closing `']}'` and `<|im_end|>` tokens MUST remain
  closure-tail signals and MUST NOT be treated as the semantic stop branch,
- eligibility MUST be determined from the exact encoded sequence used for the
  loss computation,
- if the feature is enabled and the semantic stop branch cannot be located
  unambiguously, training MUST fail fast rather than silently falling back to
  raw brace counting.

#### Scenario: Multi-object sample exposes one semantic stop branch
- **WHEN** a dense assistant target contains multiple serialized objects
- **THEN** each interior object boundary is recognized as `']},'`
- **AND** only the first terminal `']}'` after the final object is treated as
  the semantic stop branch.

### Requirement: Stop-signal damping uses a continuous bounded branch-local weight
The system SHALL compute stop-signal damping from the branch-local competition
between stopping and continuing.

Normative behavior:

- at the semantic stop branch row, the system MUST derive stop-vs-continue
  belief from the logits assigned to `']}'` and `']},'`,
- the stop-signal weight MUST be a continuous bounded function of that
  branch-local belief,
- the weight MUST monotonically reduce stop supervision as continuation belief
  becomes stronger,
- the weight MUST recover ordinary or near-ordinary stop supervision when stop
  belief dominates,
- the positive supervised token MUST remain the terminal `']}'`; `']},'` MUST
  influence the loss only through the branch-local damping calculation.

#### Scenario: Continue-preferring branch reduces stop pressure
- **WHEN** the branch-local logits favor `']},'` over `']}'`
- **THEN** the effective stop-signal weight is lower than the ordinary
  undamped stop CE weight
- **AND** the positive target token remains `']}'`.

### Requirement: Adaptive stop-signal damping is opt-in and default-preserving
The system SHALL keep adaptive stop-signal damping as an authored-YAML
experiment that preserves baseline behavior unless explicitly enabled.

Normative behavior:

- the experiment MUST be disabled by default,
- enablement MUST come from authored YAML rather than a new CLI flag,
- when disabled, teacher-forcing loss behavior MUST match the baseline token CE
  behavior for the same batch,
- when enabled, only semantic stop-branch positions may receive damped stop CE;
  later closure-tail `']}'` and `<|im_end|>` supervision MUST remain on the
  ordinary closure path.

#### Scenario: Disabled experiment preserves baseline supervision
- **WHEN** `token_ce.config.stop_signal_damping.enabled` is false or omitted
- **THEN** no semantic stop-branch positions receive special damping treatment
- **AND** ordinary structure and EOS supervision remain unchanged.

### Requirement: Adaptive stop-signal damping preserves serialization and downstream-readout invariants
The system SHALL preserve the existing dense serialization contract while
making the experiment auditable through the existing downstream readout
surfaces.

Normative behavior:

- enabling or disabling stop-signal damping MUST NOT change assistant-target
  serialization, object ordering, coord ordering, or the geometry contract,
- the experiment MUST remain compatible with the existing Qwen3-VL
  chat-template boundary tokens used for semantic stop-branch discovery,
- the experiment MUST NOT require tokenizer-vocab edits or upstream HF model
  file changes,
- downstream effect measurement MUST reuse the existing `rollout/*`,
  `eval/parsing/*`, and `eval/detection/*` metric families rather than
  introducing a second stop-signal-specific rollout or eval namespace.

#### Scenario: Enabled experiment leaves serialized targets unchanged
- **WHEN** stop-signal damping is enabled for an eligible dense object-list
  sample
- **THEN** the serialized assistant target and coord ordering remain unchanged
- **AND** only the weighting of the semantic stop-branch supervision changes.
