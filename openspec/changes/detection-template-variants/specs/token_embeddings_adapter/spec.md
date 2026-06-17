## MODIFIED Requirements

### Requirement: Token-embeddings adapter tuning for special token rows
The system SHALL provide an opt-in `custom.token_embeddings_adapter` adapter that adds trainable offsets for
configured token row IDs to both the token embedding and lm_head, while leaving
base weights frozen and storing the offsets with the adapter checkpoint (PEFT).

For generic coord-token use, the configured row IDs remain the coord vocab IDs
by default. For compact detection token-row adaptation, the configured row IDs
MUST be derived from the selected `detection_template.id` and MUST include the
required compact structural token rows in addition to the 1000 coord-token rows.
The structural row IDs are the native Qwen/CoordExp special-token IDs:
`<|object_ref_start|>` = 151646, `<|object_ref_end|>` = 151647,
`<|box_start|>` = 151648, and `<|box_end|>` = 151649. Implementations MUST
verify those tokens resolve as single-token entries before training or adapter
checkpoint validation relies on them.

The persisted module name MUST be `token_embeddings_adapter`; config validation
and docs MUST describe compact detection usage as token-row adaptation when
structural rows are included.

#### Scenario: Token embeddings adapter enabled
- GIVEN custom.token_embeddings_adapter.enabled is true in the training config
- AND custom.token_embeddings_adapter.groups resolves the coord vocab IDs (default 151670-152669)
- WHEN the model runs forward
- THEN embeddings for those IDs include the offset addition, logits include the
  head offset, and non-coord IDs are unchanged.

#### Scenario: Compact detection derives structural rows
- GIVEN compact detection token-row adaptation is enabled
- AND `detection_template.id: compact_object_box_closed`
- WHEN training config validation resolves offset rows
- THEN the configured row IDs include the 1000 coord-token ids
- AND they include `<|object_ref_start|>`, `<|object_ref_end|>`,
  `<|box_start|>`, and `<|box_end|>`.

#### Scenario: Base weights remain frozen
- GIVEN custom.token_embeddings_adapter.enabled is true
- WHEN training with token-embeddings adapter active
- THEN gradients for base embed_tokens and lm_head weights remain zero, and only
  token-embeddings adapter parameters receive updates.

#### Scenario: Saving and loading
- GIVEN a trained model with token_embeddings_adapter enabled
- WHEN saving the adapter checkpoint
- THEN token-embeddings adapter parameters are saved with the adapter (PEFT
  `adapter_model.safetensors`, via `modules_to_save=["token_embeddings_adapter"]`)
  and restored on load without extra steps.

## ADDED Requirements

### Requirement: Template-derived offset adapter checkpoint validation
The system SHALL validate compact detection offset-adapter checkpoints against
the exact row-id set derived from the resolved `detection_template.id`.

Normative behavior:

- adapter-shorthand or PEFT checkpoints with `token_embeddings_adapter` MUST validate
  against the selected compact template id,
- validation MUST reject missing row ids, extra row ids, duplicate row ids, and
  structural row omissions,
- saved `token_ids`, embedding offset rows, and lm-head offset rows when present
  MUST have matching row counts,
- `modules_to_save` MUST include `token_embeddings_adapter` when offset rows are
  expected from the adapter checkpoint,
- validation errors MUST name the resolved template id and any missing or extra
  structural rows,
- full or merged checkpoints without an offset-adapter module MUST still carry
  resolved template metadata, but adapter tensor row-shape validation is not
  applicable to them.

#### Scenario: Adapter checkpoint accepts exact object-box row set
- GIVEN an adapter checkpoint with `token_embeddings_adapter`
- AND `detection_template.id: compact_object_box_closed`
- WHEN checkpoint validation runs
- THEN the checkpoint is accepted only if its row ids are exactly the 1000 coord
  ids plus `<|object_ref_start|>`, `<|object_ref_end|>`, `<|box_start|>`, and
  `<|box_end|>`.

#### Scenario: Missing box-end row is rejected
- GIVEN an adapter checkpoint with `token_embeddings_adapter`
- AND `detection_template.id: compact_box_closed`
- WHEN the checkpoint row ids omit `<|box_end|>`
- THEN checkpoint validation fails
- AND the diagnostic names `compact_box_closed` and `<|box_end|>`.

#### Scenario: Extra or duplicate adapter rows are rejected
- GIVEN an adapter checkpoint with compact detection offset rows
- WHEN checkpoint validation finds duplicate row ids or rows outside the
  template-derived row set
- THEN checkpoint validation fails before inference or training uses the
  checkpoint.
