# token_embeddings_adapter Specification

## Purpose
Define the token-embeddings adapter contract used to tune selected special token rows without changing the base tokenizer/vocab or upstream model internals.
## Requirements
### Requirement: Token-embeddings adapter tuning for special token rows
The system SHALL provide an opt-in `custom.token_embeddings_adapter` surface that adds trainable offsets for role-resolved token IDs to both the token embedding and lm_head, while leaving base weights frozen and storing the offsets with the adapter checkpoint (PEFT).

#### Scenario: Token embeddings adapter enabled
- GIVEN custom.token_embeddings_adapter.enabled is true in the training config
- AND custom.token_embeddings_adapter.groups resolves coord vocab IDs and required schema token IDs
- WHEN the model runs forward
- THEN embeddings for those IDs include the offset addition, logits include the head offset, and non-coord IDs are unchanged.

#### Scenario: Base weights remain frozen
- GIVEN custom.token_embeddings_adapter.enabled is true
- WHEN training with token-embeddings adapter active
- THEN gradients for base embed_tokens and lm_head weights remain zero, and only token-embeddings adapter parameters receive updates.

#### Scenario: Saving and loading
- GIVEN a trained model with token_embeddings_adapter enabled
- WHEN saving the adapter checkpoint
- THEN token-embeddings adapter parameters are saved with the adapter (PEFT `adapter_model.safetensors`, via `modules_to_save=["token_embeddings_adapter"]`) and restored on load without extra steps.

### Requirement: Dedicated optimizer buckets for token-embeddings adapter offsets
The system SHALL support a multimodal optimizer variant that assigns separate learning-rate groups to token-embeddings adapter embedding and head parameters, distinct from vision/aligner/LLM dlora groups.

#### Scenario: Distinct LR applied
- GIVEN optimizer is set to multimodal_token_embeddings_adapter
- AND custom.token_embeddings_adapter.embed_lr/head_lr are configured
- WHEN optimizer param groups are built
- THEN token-embeddings adapter params appear in their own groups with the specified LRs, and dlora parameters retain their existing grouping.

### Requirement: Default safety when disabled
The system SHALL preserve existing dlora behavior when custom.token_embeddings_adapter.enabled is false.

#### Scenario: Feature off
- GIVEN custom.token_embeddings_adapter.enabled is false (or omitted)
- WHEN training initializes
- THEN no token-embeddings adapter parameters are created, optimizer grouping matches current dlora multimodal behavior, and model outputs match the previous pipeline.

### Requirement: Compact token rows are independent of field order
Compact token-row adaptation SHALL derive required structural rows from
`detection_template.id` and SHALL NOT alter the row set based on
`custom.object_field_order`.

The implementation MUST validate the exact structural row set for the selected
template id before training or adapter-checkpoint use when compact token-row
adaptation is active.

#### Scenario: Bbox-first and desc-first require identical rows
- **GIVEN** two compact configs that differ only by `custom.object_field_order`
- **AND** both use `detection_template.id: compact_object_box_closed`
- **WHEN** token-row adaptation resolves required rows
- **THEN** both require the same coord rows plus `<|object_ref_start|>`,
  `<|object_ref_end|>`, `<|box_start|>`, and `<|box_end|>`.

#### Scenario: Object-closed row set excludes box end
- **GIVEN** `detection_template.id: compact_object_closed`
- **WHEN** compact structural rows are resolved
- **THEN** `<|object_ref_end|>` is required
- **AND** `<|box_end|>` is not required.
