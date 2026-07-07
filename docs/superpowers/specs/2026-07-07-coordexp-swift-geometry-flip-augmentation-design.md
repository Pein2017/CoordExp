# CoordExp-Swift Geometry Flip Augmentation Design

## Objective

Add a minimal, auditable data-augmentation pipeline for CoordExp-Swift V1 that
supports train-only deterministic geometry flips while preserving the existing
Raw Dataset -> Dataset Processing -> Chat Template -> Packing -> Tokenization
-> Visual Processing -> Model Construction -> Forward Pass -> Loss Computation
flow.

The accepted V1 approach is **RawExample deterministic expansion with in-memory
image transforms**. Resizing, crop, color, random per-epoch augmentation, and
eval-time augmentation are out of scope.

## Dataset Scope

The initial dataset family is the current COCO `len12000` source:

```text
/data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000/train.coord.jsonl
/data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000/val.coord.jsonl
```

Do not route this design through the older `rescale_32_1024_bbox_max60` family.
Existing analysis configs may still reference `max60`, but the new
CoordExp-Swift augmentation design targets `coord_jsonl_len12000` rows and
`packing.global_max_length: 12000`.

## Contract

The augmentation stage sits after raw JSONL loading and before template
rendering:

```text
load_raw_examples
-> augmentation processor
-> render_example
-> encode_rendered_example
-> plan_packed_sequences
-> supervised training
```

The augmentation processor owns derived examples. It must produce validated
`RawExample` instances and must not mutate rendered text, encoded tokens,
packed supervision, or loss atoms directly.

V1 supports these transform variants:

- `identity`: original example.
- `hflip`: horizontal flip.
- `vflip`: vertical flip.
- `hvflip`: horizontal plus vertical flip.

When both horizontal and vertical flips are enabled, V1 includes the composed
`hvflip` variant. The deterministic expansion order is:

```text
identity, hflip, vflip, hvflip
```

If only one axis is enabled, the expansion order is:

```text
identity, hflip
```

or:

```text
identity, vflip
```

## Geometry

CoordExp-Swift raw examples use coordinate-bin boxes in canonical
`x1,y1,x2,y2` order over the inclusive `0..999` token range. Flips are defined
in that coordinate-bin space.

Horizontal flip:

```text
x1' = 999 - x2
y1' = y1
x2' = 999 - x1
y2' = y2
```

Vertical flip:

```text
x1' = x1
y1' = 999 - y2
x2' = x2
y2' = 999 - y1
```

Horizontal plus vertical flip:

```text
x1' = 999 - x2
y1' = 999 - y2
x2' = 999 - x1
y2' = 999 - y1
```

Every transformed bbox must pass the same `RawObject` bbox validation as
loaded examples. Degenerate or out-of-range transformed boxes are contract
errors, not silently dropped samples.

## Image Handling

V1 does not materialize flipped image files.

Instead, augmented examples keep the original image path and attach a logical
image-transform tuple to image planning metadata. Qwen image materialization
applies the transform in memory immediately before calling the Qwen image
processor with `do_resize=False`.

Because flips preserve image width and height, the Qwen no-resize planning
contract remains stable:

- `image_grid_thw` does not change.
- merged visual-token count does not change.
- `packing.global_max_length` pressure does not change.
- no extra image cache directory is introduced.

The image artifact/receipt path must still make the transform visible. A
debugger should be able to distinguish:

```text
original image pixels from image.jpg
hflip view of image.jpg
vflip view of image.jpg
hvflip view of image.jpg
```

without requiring four physical files.

## Example Identity And Metadata

Derived example ids must be stable and readable:

```text
{original_example_id}::aug:hflip
{original_example_id}::aug:vflip
{original_example_id}::aug:hvflip
```

The original example keeps its original id.

Derived examples must record augmentation provenance in metadata:

- source example id;
- transform id;
- transform sequence;
- original object index for each derived object;
- original bbox for each derived object;
- transformed bbox for each derived object;
- source source-provenance handle inherited from the raw row.

Object ids may remain unchanged inside the derived example because
`RawExample` uniqueness is scoped to one example. The derived `example_id`
separates training/eval/debug identity across variants.

## Object Order

The renderer remains responsible for asserting the configured template object
ordering. The augmentation processor is responsible for preparing derived
examples that satisfy that contract.

For `source_order`:

- preserve original object tuple order after bbox transform.

For `geo_sorted`:

- transform bboxes first;
- reorder the derived example's object tuple into top-to-bottom then
  left-to-right order;
- record the pre-sort source index in metadata;
- let `render_example(... object_ordering="geo_sorted")` assert the result.

For `random`:

- preserve the derived example's object tuple order;
- let the existing renderer randomization use its run-controlled seed.

This keeps the existing meaning of `geo_sorted`: the renderer receives data
that is already geometrically sorted; it does not become a sorting stage.

## Config Surface

Keep the V1 config narrow:

```yaml
data:
  augmentation:
    train:
      geometry_flips:
        horizontal: true
        vertical: true
        include_composed: true
```

Rules:

- if `horizontal: false` and `vertical: false`, the processor is a no-op;
- `include_composed` is valid only when both axes are enabled;
- augmentation applies to train data only;
- eval data is never augmented in V1;
- no probability field is exposed;
- no runtime worker knob is exposed.

## Module Shape

Add a small `src/augmentation/` package:

- `src/augmentation/geometry.py`
  - pure bbox transform functions;
  - validates transformed `0..999` bins;
  - no config loading, image IO, or pipeline ownership.

- `src/augmentation/processor.py`
  - `NoopAugmentationProcessor`;
  - `GeometryFlipAugmentationProcessor`;
  - converts `RawExample -> tuple[RawExample, ...]`;
  - owns deterministic expansion order and augmentation metadata.

- `src/augmentation/factory.py`
  - builds the processor from resolved config;
  - pipeline receives only the processor interface.

The training pipeline should not branch on individual transforms. It should
only load raw examples, call the configured processor, and pass the expanded
examples into the existing render/encode/pack path.

## Cache And Receipts

Packing-cache identity must include:

- resolved augmentation config;
- augmentation code identity:
  - `src/augmentation/geometry.py`
  - `src/augmentation/processor.py`
  - `src/augmentation/factory.py`
- image-transform policy version.

The pack-cache or training receipt should include one compact augmentation
summary:

```json
{
  "split": "train",
  "policy": "geometry_flips",
  "enabled_transforms": ["hflip", "vflip", "hvflip"],
  "input_example_count": 1000,
  "output_example_count": 4000,
  "augmented_example_count": 3000,
  "dataset_family": "rescale_32_1024_bbox_len12000"
}
```

Do not create a broad augmentation report framework in V1.

## Non-Goals

- No resizing augmentation.
- No crop, rotation, color, blur, mosaic, mixup, or random policy.
- No eval/test-time augmentation.
- No materialized flipped image files.
- No separate augmented JSONL source.
- No change to prompt wording, coordinate-token grammar, loss semantics,
  optimizer grouping, or evaluator metric reduction.
- No `max60`-specific design path.

## Required Evidence After Implementation

- Config-load tests for disabled, horizontal-only, vertical-only, and
  horizontal-plus-vertical policies.
- Unit tests for bbox transforms, including boundary boxes at `0` and `999`.
- Processor tests proving deterministic expansion order and derived
  `example_id`s.
- Processor tests proving `geo_sorted` derived examples are re-sorted after
  transform and still pass renderer assertion.
- Qwen image tests proving in-memory flips preserve no-resize dimensions and
  produce distinct pixel views without materializing files.
- Packing-cache fingerprint tests proving augmentation config and code identity
  participate in cache identity.
- One tiny dry-run or smoke pipeline check on the `len12000` fixture/path with
  augmentation enabled and cache receipt evidence.

## Acceptance Boundary

This document approves the design direction only. The next step is to draft a
focused implementation plan and, if compatibility-sensitive config/cache
contracts are introduced, an OpenSpec change before source implementation.
