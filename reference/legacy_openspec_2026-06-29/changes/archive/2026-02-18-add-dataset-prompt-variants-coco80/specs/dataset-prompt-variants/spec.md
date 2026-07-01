## ADDED Requirements

### Requirement: Variant Registry and Deterministic Resolution
The system SHALL provide a centralized prompt-variant registry for dense detection prompts and SHALL resolve prompt variants by explicit key with deterministic behavior.

#### Scenario: Default variant fallback
- **WHEN** no prompt variant is specified in training or inference configuration
- **THEN** the system MUST resolve to the backward-compatible `default` variant

#### Scenario: Unknown variant rejection
- **WHEN** a configuration specifies a prompt variant key that is not registered
- **THEN** configuration resolution MUST fail with an error that includes the unknown key and available variant keys

#### Scenario: Deterministic repeated resolution
- **WHEN** the same variant key is resolved repeatedly with the same prompt inputs (ordering and coord mode)
- **THEN** the resolver MUST return byte-identical system and user prompt text across calls

#### Scenario: Deterministic cross-surface resolution
- **WHEN** training and inference resolve the same variant key with equivalent prompt inputs
- **THEN** both surfaces MUST produce equivalent policy instructions in system and user prompts

### Requirement: Built-in COCO-80 Prompt Variant
The system SHALL include a built-in `coco_80` prompt variant for dense mode that constrains `desc` generation to the canonical 80 COCO class names using a compact class-list instruction.

#### Scenario: Canonical class source
- **WHEN** the built-in `coco_80` list is defined
- **THEN** it MUST be aligned to the canonical COCO 2017 class names/order from the repository source-of-truth snapshot (`public_data/coco/raw/categories.json`) and frozen in code for deterministic resolution

#### Scenario: COCO variant instruction content
- **WHEN** `coco_80` is selected for dense prompting
- **THEN** resolved prompts MUST include instruction text that limits object descriptions to the canonical COCO-80 class set in compact list form

#### Scenario: COCO list uniqueness
- **WHEN** `coco_80` prompt text is assembled
- **THEN** each canonical class name MUST appear at most once in the compact class-list instruction

#### Scenario: Summary prompt behavior remains unchanged
- **WHEN** summary mode is active
- **THEN** summary prompts MUST remain unchanged and MUST NOT be replaced with dense COCO-80 instructions

### Requirement: Cross-Surface Prompt Parity
Training and inference prompt construction SHALL use the same variant-aware resolver so that a selected variant produces equivalent policy instructions across surfaces.

#### Scenario: Training variant application
- **WHEN** `custom.extra.prompt_variant` is set in a training config
- **THEN** training prompt resolution MUST apply that variant for dense prompts

#### Scenario: Inference variant application
- **WHEN** `infer.prompt_variant` is set in an inference pipeline config
- **THEN** inference message construction MUST apply the same variant semantics to system and user prompts

#### Scenario: Explicit parity guidance
- **WHEN** a checkpoint was trained with a non-default prompt variant
- **THEN** documentation and config examples MUST specify that inference SHOULD use the same variant for reproducible evaluation

### Requirement: Config-First Variant Selection
Prompt variant selection SHALL be configuration-driven and SHALL NOT require new CLI flags.

#### Scenario: YAML-based selection only
- **WHEN** users configure prompt variants for training or inference
- **THEN** variant selection MUST be available via YAML configuration keys and MUST NOT require new command-line arguments

#### Scenario: Training YAML path
- **WHEN** users configure training prompt variant selection
- **THEN** the supported key path MUST be `custom.extra.prompt_variant`

#### Scenario: Inference YAML path
- **WHEN** users configure inference prompt variant selection
- **THEN** the supported key path MUST be `infer.prompt_variant`

### Requirement: Fusion Prompt Override Precedence
Existing fusion dataset-level prompt overrides SHALL preserve precedence over variant defaults.

#### Scenario: Fusion override precedence retained
- **WHEN** a fusion dataset entry provides explicit `user_prompt` and/or `system_prompt`
- **THEN** those explicit overrides MUST take precedence over prompts derived from any selected variant

### Requirement: Variant Metadata in Inference Artifacts
Inference artifacts SHALL include the resolved prompt variant for reproducibility auditing.

#### Scenario: Resolved config metadata
- **WHEN** an inference pipeline run emits `resolved_config.json`
- **THEN** the file MUST record the resolved prompt variant under inference configuration metadata

#### Scenario: Summary metadata
- **WHEN** inference emits summary output
- **THEN** the summary MUST include the resolved prompt variant value used to build generation messages
