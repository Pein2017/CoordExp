## ADDED Requirements

### Requirement: Canonical Example Chain

The data pipeline SHALL use `RawExample -> RenderedExample -> EncodedExample ->
PackedSequence` as the canonical transformation chain. `example_id` and
`examples` MUST be used for stable semantic data instances; `sample` MAY be
used only for stochastic selection or informal inspection wording.

#### Scenario: JSONL row loaded for training

- **WHEN** a JSONL row is loaded into the training pipeline
- **THEN** it MUST become a validated `RawExample`
- **AND** downstream stages MUST preserve its `example_id` through rendering,
  encoding, packing, metrics, and debug receipts.

### Requirement: Raw Example Validation

`src/data` SHALL own JSONL loading, image path resolution, image dimension
validation, and bbox coordinate validation. V1 training examples MUST be
single-image examples with coordinate-bin bbox values in `x1,y1,x2,y2` order.
Video and multi-image payloads MUST fail before template rendering.

#### Scenario: Video payload in V1 training data

- **WHEN** a raw row contains a video payload
- **THEN** data validation MUST reject the row before tokenization or Qwen
  processor calls.

#### Scenario: Coordinate-bin bbox order

- **WHEN** an object box is accepted by the raw data validator
- **THEN** the stored canonical bbox order MUST be `x1,y1,x2,y2`
- **AND** object order MUST NOT be silently changed by geometry validation.

### Requirement: Template Rendering And Object Ordering

`src/templates` SHALL render validated examples into English Qwen chat
messages, prompt text, `supervised_response_text`, typed character spans, and
realized object order. V1 MUST support `source_order` and deterministic
`random` object-ordering policies, and the defining smoke fixture MUST use
`source_order`. Legacy `sorted` MUST be rejected by V1 config validation.
Future geometric sorting MUST use a deliberately approved name such as
`geometry_sorted` with an explicit key definition.

#### Scenario: Legacy sorted object ordering configured

- **WHEN** `object_ordering: sorted` is configured
- **THEN** config validation MUST reject the legacy value
- **AND** MUST NOT reinterpret it as source order or a geometric sort.

#### Scenario: Random object ordering configured

- **WHEN** `object_ordering: random` is configured
- **THEN** the renderer MUST use a run-controlled deterministic seed
- **AND** the realized order MUST be reproducible from run artifacts.

### Requirement: Assistant Supervision Boundary

The supervised assistant target SHALL include assistant answer content and the
terminal `<|im_end|>` transition token. The trailing newline in the
`<|im_end|>\n` convention MUST be represented as ignored text. Prompt, system,
user, image-placeholder, chat-boundary, role/header, and other control tokens
MUST NOT create supervised `TokenAtom`s.

#### Scenario: Assistant suffix tokenization

- **WHEN** Qwen tokenization maps the assistant terminal suffix to
  `<|im_end|>` followed by newline
- **THEN** only `<|im_end|>` MUST be supervised as `eos`
- **AND** the newline token MUST be ignored by CE and gate losses.

#### Scenario: First assistant answer token

- **WHEN** the token immediately after the assistant start boundary belongs to
  the assistant answer content
- **THEN** that token MUST be the first supervised target token
- **AND** `<|im_start|>` or role/header tokens MUST NOT be supervised.

### Requirement: Span Alignment

Rendered spans SHALL use Python string half-open intervals over
`supervised_response_text`. Loss-bearing characters MUST be covered by exactly
one typed leaf span. Properly nested parent spans MAY exist for provenance;
crossing spans MUST fail validation. Special-token leaf spans MUST cover whole
special-token literals.

#### Scenario: Crossing rendered spans

- **WHEN** two rendered spans overlap without proper nesting
- **THEN** template validation MUST fail before Qwen encoding.

#### Scenario: Uncovered loss-bearing character

- **WHEN** a supervised character is not covered by a typed leaf span
- **THEN** rendering or alignment MUST fail instead of inventing a token type
  later.

### Requirement: Qwen Token Identity Preflight

Qwen setup SHALL verify that the loaded tokenizer contains the expected
CoordExp tokens before training. The expected trainable target specials are
`<|object_ref_start|>`, `<|object_ref_end|>`, `<|box_start|>`,
`<|box_end|>`, and `<|coord_0|>` through `<|coord_999|>`. Invalid aliases such
as `<|object_start|>` and `<|object_end|>` MUST fail.

#### Scenario: Missing coordinate token

- **WHEN** `<|coord_999|>` is missing from the loaded tokenizer
- **THEN** Qwen setup MUST fail before model forward or optimizer setup.

#### Scenario: Invalid wrapper alias

- **WHEN** a template or config references `<|object_start|>`
- **THEN** validation MUST reject it as a non-canonical alias.

### Requirement: No-Resize Image Encoding

Qwen encoding SHALL use explicit no-resize image processing. The processor call
path MUST use `do_resize=False`, and any local image-grid computation MUST be
proven equivalent to the actual no-resize processor output before it can be
used for pack-cost planning. V1 no-resize validation MUST derive admissible
spatial dimensions from the loaded processor's `patch_size` and `merge_size`,
or from a source-study-proven equivalent upstream rule. For the common Qwen
processor path, height and width MUST be divisible by
`patch_size * merge_size` before the image is accepted. The resolved config
MUST also provide explicit no-resize limits for maximum raw pixels and maximum
merged visual tokens, and the Qwen setup or encoding receipt MUST record the
loaded `patch_size`, `merge_size`, raw image dimensions, raw-pixel count,
`image_grid_thw`, and merged visual-token count used for validation.

#### Scenario: Smoke fixture image encoded

- **WHEN** the smoke fixture image is encoded
- **THEN** the decoded image dimensions MUST match the fixture metadata
- **AND** the resulting `image_grid_thw` MUST be recorded or checkable in the
  Qwen setup/encoding receipt.

#### Scenario: No-resize image dimensions are not admissible

- **WHEN** a V1 training image has height or width that violates the loaded
  no-resize processor's admissible-dimension rule
- **THEN** data or Qwen encoding validation MUST fail before the Qwen processor
  attempts its patch reshape
- **AND** the diagnostic MUST include the image id or path, measured
  dimensions, loaded `patch_size`, loaded `merge_size`, and required factor or
  source-study-proven rule.

#### Scenario: No-resize image exceeds budget

- **WHEN** a no-resize image has admissible dimensions but exceeds the resolved
  raw-pixel limit or merged-visual-token limit
- **THEN** data or Qwen encoding validation MUST fail before processor or model
  forward
- **AND** the diagnostic MUST report the measured raw pixels, measured merged
  visual tokens, and resolved limits.

### Requirement: EncodedExample Bounds

An `EncodedExample` SHALL NOT exceed `packing.global_max_length` by itself.
Encoding MUST validate token ids, image placeholders, visual payload metadata,
and token span alignment before the example can enter packing.

#### Scenario: Single encoded example too long

- **WHEN** one encoded example exceeds `packing.global_max_length`
- **THEN** the encoder or packer MUST fail with the example id and measured
  length
- **AND** it MUST NOT truncate silently.
