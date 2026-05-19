# inference-pipeline Delta

## ADDED Requirements

### Requirement: Compact-full parsing supports strict marker mode and legacy compatibility mode

The inference pipeline SHALL distinguish new marker-delimited compact-full
parsing from historical newline-compatible parsing.

Normative behavior:

- new teacher-forcing inference configs MUST default to strict
  marker-delimited compact-full parsing with
  `infer.parsing.compact_full.mode: marker_delimited_strict`;
- strict marker parsing MUST reject legacy newline-delimited rows;
- strict marker parsing MUST be all-or-nothing for primary AP/AR;
- historical old-checkpoint inference/eval configs MAY opt into
  `infer.parsing.compact_full.mode: legacy_compatible` parsing that accepts
  newline-delimited compact-full outputs;
- parse artifacts MUST record parse mode, observed separator style, and stable
  parse error codes;
- resolved configs and parse artifacts MUST record the compact-full
  serialization policy separately from the template id;
- strict marker parsing MUST use the stable error-code taxonomy:
  `empty_output`, `legacy_separator_in_new_format`,
  `missing_object_ref_start`, `missing_box_start`, `empty_description`,
  `forbidden_description_token`, `wrong_coord_arity`,
  `invalid_coord_token`, `trailing_garbage`, and `invalid_geometry`.

#### Scenario: Legacy newline is rejected in strict new mode

- **GIVEN** `infer.parsing.compact_full.mode: marker_delimited_strict`
- **WHEN** generated compact text uses newline-separated rows
- **THEN** parsing fails with `legacy_separator_in_new_format`
- **AND** no salvaged objects are used for primary scoring.

#### Scenario: Missing object marker uses stable strict error code

- **GIVEN** `infer.parsing.compact_full.mode: marker_delimited_strict`
- **WHEN** generated compact text omits the first `<|object_ref_start|>`
- **THEN** parsing fails with `missing_object_ref_start`
- **AND** no generic parse-failed code replaces the stable taxonomy.

### Requirement: Compact grammar decoding is optional and serialization-policy aware

Compact grammar decoding SHALL remain optional for new teacher-forcing models
and SHALL declare its serialization policy when enabled.

Normative behavior:

- new teacher-forcing inference configs MUST default
  `infer.generation.compact_grammar.enabled` to `false`;
- grammar-enabled runs MUST be labeled as secondary constrained decode modes;
- when enabled for new models, compact grammar MUST use
  `serialization_policy: marker_delimited`;
- marker-delimited grammar MUST allow `<|object_ref_start|>` or `<|im_end|>`
  after four coordinate tokens, not newline;
- historical configs MAY explicitly request `legacy_newline_delimited` grammar.

#### Scenario: Marker grammar does not force newline after bbox

- **GIVEN** compact grammar is enabled with `serialization_policy:
  marker_delimited`
- **WHEN** the decoder has emitted the fourth coordinate token of an object
- **THEN** allowed structural continuations are `<|object_ref_start|>` and
  `<|im_end|>`
- **AND** newline is not an allowed structural delimiter.
