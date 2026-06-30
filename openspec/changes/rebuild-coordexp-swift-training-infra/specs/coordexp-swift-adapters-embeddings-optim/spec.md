## ADDED Requirements

### Requirement: Adapter Loading And Initialization

Qwen loading SHALL support base-only model loading, base plus existing adapter
loading, and base plus newly initialized adapter tuning. When adapter tuning is
enabled without an adapter path, runtime MUST initialize the approved adapter
type automatically after source-study gates are satisfied.

#### Scenario: Adapter path supplied

- **WHEN** a config provides `adapter.path`
- **THEN** model loading MUST compose the base model with the adapter payload
- **AND** MUST record the base model identity and adapter identity in setup
  receipts.

#### Scenario: Adapter tuning enabled without path

- **WHEN** adapter tuning is enabled and no adapter path is supplied
- **THEN** runtime MUST initialize a new adapter from config
- **AND** MUST record the initialized adapter type and target policy.

### Requirement: DLoRA Source Gate

`adapter.type: dlora` SHALL NOT validate until dLoRA has been defined against
DoRA/`use_dora` or an explicit CoordExp-owned mechanism, and a minimal
round-trip probe has verified initialization, save, load, and forward
compatibility. The first adapter-enabled vertical smoke MUST use dLoRA only
after this gate passes.

#### Scenario: DLoRA requested before source study

- **WHEN** a config sets `adapter.type: dlora` before the dLoRA source gate is
  marked passed
- **THEN** config or setup validation MUST fail
- **AND** the diagnostic MUST name the missing dLoRA source-study/probe gate.

### Requirement: Adapter Target Discovery

Adapter target selection SHALL be explicit and auditable. `target_modules:
all_linear` MUST discover supported linear modules in the configured Qwen
towers while excluding `lm_head` unless a later approved contract changes that
rule.

#### Scenario: Language-only dLoRA configured

- **WHEN** adapter tuning targets the language tower with `all_linear`
- **THEN** setup MUST enumerate matched modules in a receipt
- **AND** `lm_head` MUST be absent from the adapter target set.

#### Scenario: Vision or aligner adapter configured

- **WHEN** adapter tuning targets vision or aligner modules
- **THEN** setup MUST verify that target discovery works for those supported
  choices even if production defaults target language only.

### Requirement: Special-Token Embedding Source Gate

Selected special-token embedding training SHALL be implemented only after a
source study compares custom Qwen wrappers, PEFT `TrainableTokens`, and LoRA
`trainable_token_indices`. The chosen mechanism MUST support fully trainable
selected embedding deltas, tied input/output behavior for Qwen3-VL when
applicable, compact checkpoint payloads, and base-plus-adapter-plus-delta
composition.

#### Scenario: Source study not complete

- **WHEN** training config requires special-token embedding deltas before the
  embedding source gate is marked passed
- **THEN** setup MUST fail before optimizer construction.

### Requirement: Trainable Special Tokens

The default selected-token embedding group SHALL include
`<|object_ref_start|>`, `<|object_ref_end|>`, `<|box_start|>`, `<|box_end|>`,
and `<|coord_0|>` through `<|coord_999|>`. Other embedding rows MUST remain
frozen unless a later approved config explicitly changes the selected group.

#### Scenario: Gradient outside selected tokens

- **WHEN** gradients are computed for the base embedding matrix
- **THEN** the implementation MUST mask or prevent updates outside the selected
  token set
- **AND** the optimizer receipt MUST identify the selected token ids.

### Requirement: Compact Embedding Delta Checkpoints

Checkpoints SHALL save selected special-token embedding deltas as compact
payloads, not as full embedding or full head exports. Inference and later
training MUST load the base model plus optional adapter plus optional selected
embedding delta explicitly.

#### Scenario: Checkpoint with embedding deltas

- **WHEN** a checkpoint includes selected embedding deltas
- **THEN** it MUST contain `special_token_embeddings.safetensors` and
  `special_token_embeddings.json` or approved equivalents
- **AND** metadata MUST record token strings, token ids, dtype, shape, and
  tied/untied behavior.

### Requirement: Explicit Optimizer Groups

Every trainable parameter SHALL match exactly one explicit optimizer group with
approved learning rate and weight decay. Supported initial groups MUST include
vision, aligner, language, adapter parameters, and selected special-token
embedding deltas. Missing or duplicate group matches MUST fail fast.

#### Scenario: Trainable parameter not matched

- **WHEN** optimizer construction finds a trainable parameter that matches no
  configured group
- **THEN** optimizer construction MUST fail before training begins.

#### Scenario: Parameter matched twice

- **WHEN** a trainable parameter matches multiple optimizer groups
- **THEN** optimizer construction MUST fail with a group-conflict diagnostic.

### Requirement: Trainable Surface Receipt

Adapter, embedding, freeze, and optimizer setup SHALL emit a trainable-surface
receipt that lists frozen towers, trainable towers, adapter targets, selected
embedding tokens, parameter counts, optimizer groups, and unmatched/frozen
reason summaries.

#### Scenario: Vertical smoke setup completes

- **WHEN** the vertical smoke finishes model and optimizer setup
- **THEN** run artifacts MUST include a receipt that proves the intended
  trainable surface and optimizer groups before the first backward pass.
