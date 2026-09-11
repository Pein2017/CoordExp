## MODIFIED Requirements

### Requirement: Adapter Loading And Initialization

Qwen loading SHALL support base-only model loading, base plus existing adapter
loading, and base plus newly initialized adapter tuning. When adapter tuning is
enabled without an adapter path, runtime MUST initialize the approved adapter
type automatically after source-study gates are satisfied. Setup MUST validate
the configured base/adapter identity and selected target policy, but normal
training MUST NOT require a separate durable adapter setup receipt.
The standard adapter boundary SHALL validate the configured base identifier,
model compatibility, `peft_type: LORA`, `use_dora: true`, target modules,
tensor shapes, and nonempty LoRA A/B plus DoRA magnitude-vector state. It MUST
NOT claim immutable base-config or tokenizer-content validation that standard
PEFT adapter metadata does not carry. Optional selected-token embedding deltas
retain their existing stronger base-config and tokenizer hash validation.

#### Scenario: Adapter path supplied

- **WHEN** a config provides `adapter.path`
- **THEN** model loading MUST compose the base model with the adapter payload
- **AND** MUST validate that the loaded adapter is compatible with the
  configured base identifier, model, target policy, and required LoRA/DoRA
  tensors.

#### Scenario: Base contents change at the same configured path

- **WHEN** an adapter-only inference config points to a base path whose
  contents changed without changing the standard PEFT base identifier
- **THEN** the adapter loader MUST apply its declared identifier/model/target/
  tensor checks
- **AND** MUST NOT claim immutable hash validation absent from the adapter
  payload.

#### Scenario: Adapter tuning enabled without path

- **WHEN** adapter tuning is enabled and no adapter path is supplied
- **THEN** runtime MUST initialize a new adapter from config
- **AND** MUST validate the initialized adapter type and target policy before
  optimizer construction.

### Requirement: Adapter Target Discovery

Adapter target selection SHALL be explicit and validated. `target_modules:
all_linear` MUST discover supported linear modules in the configured Qwen
towers while excluding `lm_head` unless a later approved contract changes that
rule. Discovery MUST return the complete matched target set to adapter setup
and tests; it MUST NOT require a separate durable receipt in every run.

#### Scenario: Language-only DoRA configured

- **WHEN** adapter tuning targets the language tower with `all_linear`
- **THEN** setup MUST enumerate and validate the matched module set
- **AND** `lm_head` MUST be absent from that set.

#### Scenario: Vision or aligner adapter configured

- **WHEN** adapter tuning targets vision or aligner modules
- **THEN** setup MUST verify target discovery for those supported choices even
  if production defaults target language only.

### Requirement: Adapter Seed Mode Execution Hierarchy

Adapter setup SHALL execute the config seed-mode hierarchy after the DoRA
source gate passes. `initialize_new` MUST create every configured target from
fresh DoRA initialization. `load_existing` MUST load the configured
`adapter.path` and validate that loaded adapter targets match requested target
discovery. `warm_start_expand_dora` MUST create every DoRA target required by
the new config, then decide reuse versus initialization independently for each
required target module. For each required target module, setup MUST reuse the
source checkpoint tensors when the source adapter contains the complete DoRA
tensor set for that target; setup MUST keep the new initialization when the
source adapter contains no tensors for that target; and setup MUST fail when
the source adapter contains only a partial tensor set for a required target.
Selected special-token embedding deltas MUST be loaded from the configured
compact embedding payload before optimizer construction. Setup MUST return
validated copied/initialized target information to its caller and tests, but
normal training MUST NOT persist a per-run adapter receipt graph.

#### Scenario: Fresh adapter setup

- **WHEN** `adapter.seed_mode: initialize_new` is configured
- **THEN** setup MUST create all configured target modules from fresh DoRA
  initialization
- **AND** MUST validate the configured target towers.

#### Scenario: Existing adapter setup

- **WHEN** `adapter.seed_mode: load_existing` is configured
- **THEN** setup MUST load `adapter.path`
- **AND** MUST fail if loaded targets do not match configured discovery.

#### Scenario: Source adapter lacks a newly requested tower

- **WHEN** a config requests language, vision, and aligner DoRA targets while
  the source adapter contains only language tensors
- **THEN** setup MUST copy the complete language tensor set
- **AND** MUST initialize the vision and aligner DoRA tensors as trainable
  parameters.

#### Scenario: Source adapter already contains a requested tower

- **WHEN** a config requests a target module that already has a complete DoRA
  tensor set in the source adapter
- **THEN** setup MUST copy that tensor set by exact key into the new adapter
- **AND** the copied target set MUST remain available for validation and tests.

#### Scenario: Source adapter has a partial required target

- **WHEN** the source adapter contains some but not all DoRA tensors for a
  target required by the new config
- **THEN** setup MUST fail before optimizer construction
- **AND** the diagnostic MUST name the missing source tensor keys.

### Requirement: Trainable Special Tokens

The default selected-token embedding group SHALL include
`<|object_ref_start|>`, `<|object_ref_end|>`, `<|box_start|>`, `<|box_end|>`,
and `<|coord_0|>` through `<|coord_999|>`. Other embedding rows MUST remain
frozen unless a later approved config explicitly changes the selected group.
Optimizer construction MUST validate the selected token ids without requiring
a separate durable optimizer receipt.

#### Scenario: Gradient outside selected tokens

- **WHEN** gradients are computed for the base embedding matrix
- **THEN** the implementation MUST mask or prevent updates outside the selected
  token set
- **AND** optimizer construction MUST expose the selected token ids to its
  validation tests.

### Requirement: Explicit Optimizer Groups

Every trainable parameter SHALL match exactly one explicit optimizer group with
approved learning rate and weight decay. Supported initial groups MUST include
vision, aligner, language, adapter parameters, and selected special-token
embedding deltas. In this version tower names are semantic namespaces for
adapter targets and optimizer validation; full base-model parameter fine-tuning
is unsupported. Any trainable base weight outside approved adapter modules and
selected special-token embedding deltas MUST fail unless a later approved
contract adds that mode. Optimizer matching MUST be validated against
post-adapter parameter names, including PEFT wrapper prefixes and DoRA
magnitude-vector parameters when present. Missing or duplicate group matches
MUST fail fast. A durable optimizer-group receipt MUST NOT be required.

#### Scenario: Trainable parameter not matched

- **WHEN** optimizer construction finds a trainable parameter that matches no
  configured group
- **THEN** optimizer construction MUST fail before training begins.

#### Scenario: Parameter matched twice

- **WHEN** a trainable parameter matches multiple optimizer groups
- **THEN** optimizer construction MUST fail with a group-conflict diagnostic.

## REMOVED Requirements

### Requirement: Trainable Surface Receipt

**Reason**: Adapter, embedding, freeze, and optimizer correctness remains a
setup invariant, but persisting a separate receipt graph in every run
duplicates resolved config and validation output.

**Migration**: Validate trainable targets, selected tokens, parameter counts,
and optimizer-group coverage through setup return values and focused tests.
Keep only compact terminal/setup facts needed in `run.json`; do not create a
receipt directory.
