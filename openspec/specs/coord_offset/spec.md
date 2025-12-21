# coord_offset Specification

## Purpose
TBD - created by archiving change add-coord-offset-tuning. Update Purpose after archive.
## Requirements
### Requirement: Coord-offset tuning for coord tokens
The system SHALL provide an opt-in coord-offset adapter that adds trainable offsets for the coord token IDs to both the token embedding and lm_head, while leaving base weights frozen and storing the offsets with the adapter checkpoint (PEFT).

#### Scenario: Coord offsets enabled
- GIVEN coord_offset.enabled is true in the training config
- AND coord_offset.ids is set to the coord vocab IDs (default 151670–152669)
- WHEN the model runs forward
- THEN embeddings for those IDs include the offset addition, logits include the head offset, and non-coord IDs are unchanged.

#### Scenario: Base weights remain frozen
- GIVEN coord_offset.enabled is true
- WHEN training with coord-offset active
- THEN gradients for base embed_tokens and lm_head weights remain zero, and only coord-offset parameters receive updates.

#### Scenario: Saving and loading
- GIVEN a trained model with coord-offset enabled
- WHEN saving the adapter checkpoint
- THEN coord-offset parameters are saved with the adapter (PEFT `adapter_model.safetensors`, via `modules_to_save`) and restored on load without extra steps.

### Requirement: Dedicated optimizer buckets for coord offsets
The system SHALL support a multimodal optimizer variant that assigns separate learning-rate groups to coord-offset embedding and head parameters, distinct from vision/aligner/LLM dlora groups.

#### Scenario: Distinct LR applied
- GIVEN optimizer is set to multimodal_coord_offset
- AND coord_offset.embed_lr/head_lr are configured
- WHEN optimizer param groups are built
- THEN coord-offset params appear in their own groups with the specified LRs, and dlora parameters retain their existing grouping.

### Requirement: Default safety when disabled
The system SHALL preserve existing dlora behavior when coord_offset.enabled is false.

#### Scenario: Feature off
- GIVEN coord_offset.enabled is false (or omitted)
- WHEN training initializes
- THEN no coord-offset parameters are created, optimizer grouping matches current dlora multimodal behavior, and model outputs match the previous pipeline.

