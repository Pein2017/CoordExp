## ADDED Requirements

### Requirement: Detection template id participates in parser and provenance policy
The shared inference runtime SHALL treat resolved `detection_template.id` as
part of the strict parser policy and provenance for compact detection artifacts.

Normative behavior:

- shared runtime prompt construction, backend request preparation, parsing, and
  artifact materialization MUST consume the resolved semantic template id rather
  than independently authored compact parse modes, row separators, or
  `compact_full` format names,
- parser policy metadata MAY include internal parser ids, but those ids MUST be
  derived from `detection_template.id`,
- prompt policy fingerprints for compact detection MUST include
  `detection_template.id`,
- parser/provenance metadata recorded in `resolved_config.json` and
  `summary.json` MUST include `detection_template.id`,
- comparable compact artifacts MUST fail strict comparison or official eval
  preflight when required template provenance is missing,
- backend adapters MUST NOT rewrite template-specific compact prompt patterns,
  row separators, or closure-token instructions.

#### Scenario: Shared runtime derives parser policy from template id
- **GIVEN** a compact inference config with
  `detection_template.id: compact_object_box_closed`
- **WHEN** the shared runtime prepares prompt, decode, parse, and provenance
  policy
- **THEN** the parser policy is derived from
  `compact_object_box_closed`
- **AND** no independent compact parse-mode or row-separator value is accepted
  as the source of truth.

#### Scenario: Prompt fingerprint changes with template id
- **GIVEN** two compact inference configs that differ only by
  `detection_template.id`
- **WHEN** the shared runtime computes prompt policy fingerprints
- **THEN** the fingerprints differ.

#### Scenario: Missing template provenance blocks comparable artifacts
- **GIVEN** a compact artifact family without resolved detection template
  provenance
- **WHEN** a comparable evaluation or official reporting path loads the artifact
- **THEN** loading fails with a missing-provenance diagnostic before metrics are
  compared.
