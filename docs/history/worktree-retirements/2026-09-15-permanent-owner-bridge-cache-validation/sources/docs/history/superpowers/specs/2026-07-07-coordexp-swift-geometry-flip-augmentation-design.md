# CoordExp-Swift Geometry Flip Augmentation Design

## Objective

Add a minimal, auditable data-augmentation pipeline for CoordExp-Swift V1 that
supports train-only horizontal and vertical geometry flips while preserving the
existing Raw Dataset -> Dataset Processing -> Chat Template -> Packing ->
Tokenization -> Visual Processing -> Model Construction -> Forward Pass ->
Loss Computation flow.

The accepted V1 approach is a **static stochastic view per run**:

- every source training example produces exactly one training presentation;
- horizontal and vertical flip decisions are sampled once when the pack cache is
  materialized;
- the sampled view is cached and repeated across epochs;
- changing the augmentation seed, probabilities, policy version, or code
  identity creates a different cache identity;
- eval and inference data are never augmented in V1.

Resizing, crop, color, epoch-varying online augmentation, test-time
augmentation, and sample-count expansion are out of scope.

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

The augmentation processor owns training presentations. It must produce
validated `RawExample` instances and must not mutate rendered text, encoded
tokens, packed supervision, or loss atoms directly.

V1 supports four realized transform ids:

- `identity`: no flip.
- `hflip`: horizontal flip only.
- `vflip`: vertical flip only.
- `hvflip`: horizontal and vertical flip.

These are not expansion variants. They are the four possible outcomes of two
independent Bernoulli decisions per source example.

The training input count is invariant:

```text
output_example_count == input_example_count
```

## Sampling

The V1 config surface is:

```yaml
data:
  augmentation:
    train:
      geometry_flips:
        enabled: true
        horizontal_prob: 0.5
        vertical_prob: 0.5
```

Rules:

- augmentation applies to train data only;
- eval and inference rows are never augmented;
- each probability is validated in `[0.0, 1.0]`;
- `enabled: false` is a no-op regardless of probability values;
- when enabled, horizontal and vertical decisions are sampled independently;
- if both decisions are true, the realized transform id is `hvflip`;
- no public worker knob is exposed;
- no epoch-varying or per-loader-resample mode is exposed in V1.

Sampling must be deterministic under cache identity. A valid seed source is:

```text
{runtime.seed}:{split}:{source_example_id}:geometry_flips:v1
```

The exact seed construction may include a stable source-row index if duplicate
source example ids are ever allowed, but it must not depend on worker order,
process id, JSONL chunking, GPU rank, Python object identity, or cache-hit vs
cache-miss behavior.

Same resolved config plus same seed plus same code identity means the same
sampled view and the same cache. To resample the stochastic view intentionally,
change the seed or the augmentation probabilities.

## Matrix Geometry

CoordExp-Swift raw examples use coordinate-bin boxes in canonical
`x1,y1,x2,y2` order over the inclusive `0..999` token range. Flips are defined
as affine transforms in that coordinate-bin plane.

Use homogeneous coordinate matrices:

```text
identity:
[[ 1,  0,   0],
 [ 0,  1,   0],
 [ 0,  0,   1]]

hflip:
[[-1,  0, 999],
 [ 0,  1,   0],
 [ 0,  0,   1]]

vflip:
[[ 1,  0,   0],
 [ 0, -1, 999],
 [ 0,  0,   1]]

hvflip:
[[-1,  0, 999],
 [ 0, -1, 999],
 [ 0,  0,   1]]
```

Transform bbox corners, then canonicalize:

```text
corners = (x1,y1), (x2,y1), (x2,y2), (x1,y2)
p_i' = M p_i
bbox' = min_x, min_y, max_x, max_y
```

This avoids procedural "do vertical then do horizontal" ambiguity. The same
transform id must drive both the coordinate transform and the in-memory image
view.

Every transformed bbox must pass the same `RawObject` bbox validation as loaded
examples. Degenerate or out-of-range transformed boxes are contract errors, not
silently dropped samples.

## Image Handling

V1 does not materialize flipped image files.

Instead, augmented presentations keep the original image path and attach a
logical image-transform descriptor to image planning metadata. Qwen image
materialization applies the transform in memory immediately before calling the
Qwen image processor with `do_resize=False`.

Because flips preserve image width and height, the Qwen no-resize planning
contract remains stable:

- `image_grid_thw` does not change;
- merged visual-token count does not change;
- `packing.global_max_length` pressure does not change;
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

Receipt metadata should include the transform id and coordinate matrix:

```json
{
  "transform_id": "hvflip",
  "coord_matrix": [[-1, 0, 999], [0, -1, 999], [0, 0, 1]],
  "image_transform": "hvflip"
}
```

## Example Identity And Metadata

V1 separates source identity from presentation identity.

- `source_example_id` is the id from the JSONL row.
- `presentation_id` is the id for the sampled training view.

For `identity`, `presentation_id` may equal `source_example_id`. For flipped
views, use stable readable ids:

```text
{source_example_id}::aug:hflip
{source_example_id}::aug:vflip
{source_example_id}::aug:hvflip
```

The `RawExample.example_id` passed into rendering should be the presentation
id, while augmentation metadata must retain the source example id. This makes
pack artifacts and debugger output reflect the actual content being trained.

Augmented presentations must record provenance in metadata:

- source example id;
- presentation id;
- transform id;
- coordinate matrix;
- sampled horizontal and vertical booleans;
- original object index for each derived object;
- original bbox for each derived object;
- transformed bbox for each derived object;
- source-provenance handle inherited from the raw row.

Object ids may remain unchanged inside a presentation because `RawObject`
uniqueness is scoped to one example. The presentation id separates
training/debug identity across sampled views.

## Object Order

The renderer remains responsible for asserting the configured template object
ordering. The augmentation processor is responsible for preparing presentations
that satisfy that contract.

For `source_order`:

- preserve original object tuple order after bbox transform.

For `geo_sorted`:

- transform bboxes first;
- reorder the presentation's object tuple into top-to-bottom then left-to-right
  order;
- record the pre-sort source index in metadata;
- let `render_example(... object_ordering="geo_sorted")` assert the result.

For `random`:

- support augmentation and random order together;
- sample the transform first;
- keep the post-transform presentation object tuple as the canonical input to
  the renderer;
- let the renderer apply its explicit run-controlled random order;
- require an explicit object-order seed/key receipt for every randomized
  presentation;
- record both transform provenance and object-order seed/key in receipts or
  metadata sufficient to reproduce the presentation.

This keeps the existing meaning of `geo_sorted`: the renderer receives data
that is already geometrically sorted; it does not become a sorting stage.

## Module Shape

Add a small `src/augmentation/` package:

- `src/augmentation/geometry.py`
  - pure matrix and bbox transform functions;
  - validates transformed `0..999` bins;
  - no config loading, image IO, or pipeline ownership.

- `src/augmentation/processor.py`
  - `NoopAugmentationProcessor`;
  - `GeometryFlipAugmentationProcessor`;
  - converts `RawExample -> RawExample`;
  - owns deterministic sampling, presentation identity, ordering preparation,
    and augmentation metadata.

- `src/augmentation/factory.py`
  - builds the processor from resolved config;
  - pipeline receives only the processor interface.

The training pipeline should not branch on individual transforms. It should
load raw examples, call the configured processor once per source example, and
pass the resulting presentations into the existing render/encode/pack path.

## Cache And Receipts

Packing-cache identity must include:

- resolved augmentation config;
- augmentation seed source;
- augmentation policy version;
- augmentation code identity:
  - `src/augmentation/geometry.py`
  - `src/augmentation/processor.py`
  - `src/augmentation/factory.py`
- image-transform materialization code identity, including the relevant Qwen
  image planning/materialization code.

Worker count is provenance only and must not participate in the semantic cache
key. Cache hits load the existing sampled view and must not resample.

The pack-cache or training receipt should include one compact augmentation
summary:

```json
{
  "split": "train",
  "mode": "static_stochastic_view",
  "policy": "geometry_flips",
  "seed": 42,
  "horizontal_prob": 0.5,
  "vertical_prob": 0.5,
  "input_example_count": 1000,
  "output_example_count": 1000,
  "transform_counts": {
    "identity": 252,
    "hflip": 249,
    "vflip": 248,
    "hvflip": 251
  },
  "dataset_family": "rescale_32_1024_bbox_len12000"
}
```

The receipt does not need a broad augmentation report framework in V1.

## Non-Goals

- No deterministic sample-count expansion.
- No resizing augmentation.
- No crop, rotation, color, blur, mosaic, mixup, or learned policy.
- No eval/test-time augmentation.
- No materialized flipped image files.
- No separate augmented JSONL source.
- No epoch-varying resampling.
- No change to prompt wording, coordinate-token grammar, loss semantics,
  optimizer grouping, or evaluator metric reduction.
- No `max60`-specific design path.

## Required Evidence After Implementation

- Config-load tests for disabled and probability-enabled policies.
- Unit tests for matrix and bbox transforms, including boundary boxes at `0`
  and `999`.
- Processor tests proving output count equals input count.
- Processor tests proving same seed/config yields same transform assignments,
  and changing seed changes assignments.
- Processor tests proving `geo_sorted` presentations are re-sorted after
  transform and still pass renderer assertion.
- Processor tests proving augmentation plus `random` ordering is deterministic,
  reproducible, and independent of worker completion order.
- Qwen image tests proving in-memory flips preserve no-resize dimensions and
  produce distinct pixel views without materializing files.
- Packing-cache fingerprint tests proving augmentation config, seed, policy
  version, and code identity participate in cache identity.
- Cache-hit tests proving cached transformed presentations are reused without
  resampling.
- One tiny dry-run or smoke pipeline check on the `len12000` fixture/path with
  augmentation enabled and cache receipt evidence.

## Acceptance Boundary

This document approves the design direction only. The next step is to draft a
focused OpenSpec change or implementation plan. Source implementation should
wait until the compatibility-sensitive config/cache/artifact contracts are
recorded in the appropriate spec surface.
