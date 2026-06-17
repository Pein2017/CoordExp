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
