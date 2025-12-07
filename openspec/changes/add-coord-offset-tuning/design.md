# Design: coord-offset tuning for coord tokens

## Goals
- Enable learning for coord token rows in `embed_tokens` and `lm_head` without altering base weights or non-coord vocab.
- Keep existing dlora coverage on other linear layers (LLM, vision, aligner).
- Allow separate learning rates for the coord offsets.

## Approach
1) **Offset adapter (embedding + head)**
   - New module injects trainable `embed_offset` and `head_offset` tensors for a configured ID list (default 151670–152669; exclude `<|coord_*|>` 151669).
   - Base `embed_tokens` and `lm_head` remain frozen; forward adds offsets to embeddings and adds logit bias to the corresponding vocab positions.
   - Offsets must live under a submodule on the PEFT-wrapped model **and be registered via `modules_to_save`** so PEFT/ms-swift include them in `adapter_model.safetensors` and in export flows. Separate sidecar files are not compatible with the existing save/load/export path.

2) **Optimizer grouping**
   - Introduce `multimodal_coord_offset` optimizer variant:
     - Group A: coord embed offsets (dedicated LR/WD).
     - Group B: coord head offsets (dedicated LR/WD).
     - Group C/D/E: existing vision / aligner / LLM (dlora) groups, reusing the current multimodal splitter.
   - Defaults: inherit dtype from model (bf16) and weight_decay=0 unless overridden.

3) **Config surface**
   - `coord_offset.enabled` (bool), `ids` (list/range), `embed_lr`, `head_lr`, optional `weight_decay`, `dtype` (default model dtype).
   - Overlay config for `configs/dlora/sft_base.yaml` to enable the feature for 8B.

4) **Behavioral guarantees**
   - Non-coord vocab rows unchanged (base frozen; offsets only for listed IDs).
   - LoRA/DoRA remains active for all other linear layers.
   - Works with Deepspeed ZeRO-2 and flash-attn (offset params are tiny; no kernel changes).

## Validation plan
- Unit: forward diff shows only coord IDs shift; backward shows grads only on offset params.
- Optimizer: inspect param groups to confirm distinct LRs.
- Config: toggle on/off leaves default behavior untouched when disabled.
