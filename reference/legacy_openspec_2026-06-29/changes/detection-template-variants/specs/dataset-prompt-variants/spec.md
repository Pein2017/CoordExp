## MODIFIED Requirements

### Requirement: Variant Registry and Deterministic Resolution
The system SHALL provide a centralized prompt-variant registry for dense
detection prompts and SHALL resolve prompt variants by explicit key with
deterministic behavior.

Prompt resolution inputs for dense prompts MUST include:

- prompt variant key,
- ordering policy,
- object field order,
- coord mode, and
- detection template id.

#### Scenario: Default variant fallback
- **WHEN** no prompt variant is specified in training or inference configuration
- **THEN** the system MUST resolve to the backward-compatible `default` variant

#### Scenario: Unknown variant rejection
- **WHEN** a configuration specifies a prompt variant key that is not registered
- **THEN** configuration resolution MUST fail with an error that includes the
  unknown key and available variant keys

#### Scenario: Deterministic repeated resolution
- **WHEN** the same variant key is resolved repeatedly with the same prompt
  inputs (ordering, object field order, coord mode, and detection template id)
- **THEN** the resolver MUST return byte-identical system and user prompt text
  across calls
- **AND** `coord mode` refers to the resolver `coord_mode` input (current
  contract: `coord_tokens`).

#### Scenario: Deterministic cross-surface resolution
- **WHEN** training, trainer-driven rollout/eval prompt rebuilding, and
  standalone inference resolve the same variant key with equivalent prompt
  inputs
- **THEN** all of those surfaces MUST produce equivalent policy instructions in
  system and user prompts

#### Scenario: Template-specific compact row pattern
- **GIVEN** a compact detection template id
- **WHEN** dense prompt text is resolved
- **THEN** the prompt's compact output pattern matches the canonical renderer
  for that template id
- **AND** line-vs-no-line behavior is derived from the template id.

#### Scenario: Backend prompt parity
- **GIVEN** a fixed inference config with a compact `detection_template.id`
- **WHEN** HF, local vLLM, and server-backed vLLM request builders resolve
  prompts for that config
- **THEN** the final payload handed to each backend contains byte-equivalent
  system and user prompt text
- **AND** they record the same prompt hash and detection template id
- **AND** backend adapters do not rewrite compact row pattern, separator, or
  closure-token instructions
- **AND** final-newline instructions for `compact_object_box_closed_lines` are
  preserved through backend request construction.
