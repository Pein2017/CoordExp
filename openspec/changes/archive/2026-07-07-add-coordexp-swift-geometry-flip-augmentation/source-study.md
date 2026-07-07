# Source Study Notes

## Renderer Object Order

Current renderer behavior in `src/templates/renderer.py`:

- `source_order`: returns `tuple(enumerate(raw_example.objects))` unchanged.
- `geo_sorted`: does not sort; it asserts the incoming object tuple is already
  top-to-bottom then left-to-right.
- `random`: requires an explicit `object_order_seed` and derives the
  reproducibility key as:

```text
{object_order_seed}:{raw_example.example_id}
```

Augmentation must therefore prepare `geo_sorted` presentations before calling
the renderer, and must receipt the exact object-order seed/key for `random`
presentations.

## Qwen Image Materialization

Current Qwen image behavior in `src/qwen/images.py`:

- `build_no_resize_image_plan` derives width, height, `image_grid_thw`,
  `raw_patch_rows`, and `merged_visual_tokens` from `RawExample.image`.
- `plan_qwen_image` returns lazy `QwenImageEncoding` with pixels unset.
- `materialize_qwen_image_encoding` and
  `materialize_qwen_image_encoding_batch` call `_load_rgb_image_from_plan`
  immediately before `image_processor(..., do_resize=False)`.

The minimal V1 hook is the image plan plus `_load_rgb_image_from_plan`: carry a
logical transform id in `QwenNoResizeImagePlan`, preserve no-resize dimensions,
and apply the flip in memory before the processor call.

## Pack Cache Identity And Receipts

Current packing-cache behavior in `src/training/pack_cache.py`:

- semantic determinants include dataset path/stat/sha, template, packing
  length, processor config, train order, runtime seed, Qwen identity, and code
  identity;
- worker count is materialization provenance only and does not affect the
  semantic fingerprint;
- cache receipts already expose `materialization` and return the manifest
  materialization block through `_resolve_or_build_pack_cache`.

Augmentation must add resolved augmentation config, augmentation policy version,
augmentation source seed, augmentation code identity, and Qwen image
materialization code identity to the semantic fingerprint. Cache miss receipts
must also carry the compact transform-count and randomized object-order
seed/key provenance required by the spec.

## Fixture

Existing fixture:

```text
tests/fixtures/smoke/qwen3_vl_single_image_pack/config.yaml
tests/fixtures/smoke/qwen3_vl_single_image_pack/examples.jsonl
```

It already uses `packing.global_max_length: 12000` and contains two training
rows with two objects each, so it can be reused for augmentation smoke checks.

## Runtime Boundary

This change has not touched adapter, DoRA/PEFT, special-token embedding,
Accelerate, or DeepSpeed modules. If a later GPU smoke reuses adapter-enabled
training, the existing DoRA and special-token source-study gates still apply.
