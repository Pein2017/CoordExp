---
doc_id: docs.standards.upstream.qwen-vl
layer: docs
doc_type: standard
status: canonical
domain: standards
summary: Transformers Qwen-VL model, processor, and CoordExp integration notes.
updated: 2026-06-07
---

# Transformers Qwen-VL

Source scope: local `transformers==4.57.1`, with primary focus on `qwen3_vl`,
`qwen2_5_vl`, and `qwen2_vl`. Resolve `${TRANSFORMERS_ROOT}` with:

```bash
python -c "import pathlib, transformers; print(pathlib.Path(transformers.__file__).parent)"
```

## Architecture Map

- Qwen3-VL generated files say upstream edits belong in
  `modular_qwen3_vl.py`, not installed generated files. CoordExp must not patch
  local HF model files.
- `Qwen3VLConfig` composes vision and text configs and carries multimodal token
  ids: `vision_start_token_id`, `vision_end_token_id`, `image_token_id`, and
  `video_token_id`.
- `Qwen3VLForConditionalGeneration` wraps `Qwen3VLModel` and `lm_head`; the
  backbone exposes `model.visual` and `model.language_model`.
- Qwen3-VL vision uses patch embedding, learned/interpolated position
  embeddings, vision rotary embeddings, ViT blocks, a merger, and DeepStack
  merger outputs from configured vision layers.
- Qwen3-VL text is not pure text-only: DeepStack visual features can be injected
  into early hidden states through `visual_pos_masks` and
  `deepstack_visual_embeds`.
- Auto mappings keep Qwen2-VL, Qwen2.5-VL, and Qwen3-VL separate. Do not
  collapse their processor, RoPE, video, or attention behavior.

## Processor And Grid Rules

- `Qwen3VLProcessor` wraps an image processor, Qwen2 tokenizer, and Qwen3-VL
  video processor.
- Image placeholders expand from `image_grid_thw` using visual merge size.
  CoordExp should build messages through the processor/template path and then
  verify placeholder counts, not manually synthesize repeated image tokens.
- Qwen3 video placeholders differ from Qwen2.5: Qwen3 inserts text timestamps
  and frame blocks; Qwen2.5 uses `second_per_grid_ts` flows.
- Qwen3 `get_rope_index()` splits video grids into per-frame rows and uses text
  timestamps rather than absolute video time ids.
- Beam/generation expansion is visual-tensor aware because `pixel_values` is
  flattened across visual tokens and `image_grid_thw` is flattened across
  images, not ordinary batch-major metadata.

## Position IDs

- Qwen3-VL accepts 3-row MRoPE `position_ids` for temporal/height/width.
- If a 4-row tensor is supplied, row 0 is used as text positions for causal mask
  and text attention, while rows 1-3 are MRoPE geometry.
- During generation, Qwen3-VL recomputes position IDs with cached
  `rope_deltas` and drops visual tensors after the first cache step.
- Manual `position_ids` should be treated as a high-risk diagnostic override.
  If used, preserve either the 3-row MRoPE contract or the 4-row text-plus-MRoPE
  contract exactly.

## Logits

- Qwen3-VL forward accepts `logits_to_keep`.
- It slices hidden states before `lm_head`; `0` keeps all logits.
- CoordExp token diagnostics, coordinate scoring, and segment-aware sidecar
  remapping require full logits unless a separate logit-projection map exists.

## CoordExp Rules

- Keep `processor_do_resize=false`. HF Qwen image processors can default to
  resizing; CoordExp geometry assumes offline-prepared images and aligned
  coordinates.
- Preserve `pixel_values`, `image_grid_thw`, `pixel_values_videos`,
  `video_grid_thw`, `attention_mask`, and cache state together. Do not reorder
  grids independently of token order.
- For LoRA/freezing, inspect loaded checkpoint names. Common Qwen3-VL prefixes:
  `model.visual.blocks.*`, `model.visual.merger.*`,
  `model.visual.deepstack_merger_list.*`, and
  `model.language_model.layers.*`.
- Make `attn_implementation` explicit in run metadata, especially when comparing
  `eager`, `sdpa`, and `flash_attention_2`.

## Troubleshooting

- `qwen3_vl` import or AutoModel errors usually indicate an older Transformers
  install.
- Image token/features mismatch: inspect `image_grid_thw`, merge size, chat
  template image-token count, and placeholder-mask construction.
- Duplicate visual work in generation: inspect `cache_position`; Qwen3-VL should
  only forward pixels on the first cache step.
- Video temporal drift: record fps metadata, Qwen3 timestamp insertion, frame
  sampling, and `do_resize`.
- Unexpected `token_type_ids`: Qwen3-VL examples may drop them; local processor
  defaults do not require returning them.

## Handles

- Local config:
  `${TRANSFORMERS_ROOT}/models/qwen3_vl/configuration_qwen3_vl.py`
- Local model:
  `${TRANSFORMERS_ROOT}/models/qwen3_vl/modeling_qwen3_vl.py`
- Local processor:
  `${TRANSFORMERS_ROOT}/models/qwen3_vl/processing_qwen3_vl.py`
- Local video processor:
  `${TRANSFORMERS_ROOT}/models/qwen3_vl/video_processing_qwen3_vl.py`
- Upstream docs:
  [Qwen3-VL v4.57.1](https://huggingface.co/docs/transformers/v4.57.1/model_doc/qwen3_vl)
- Upstream source:
  [qwen3_vl tag v4.57.1](https://github.com/huggingface/transformers/tree/v4.57.1/src/transformers/models/qwen3_vl)
