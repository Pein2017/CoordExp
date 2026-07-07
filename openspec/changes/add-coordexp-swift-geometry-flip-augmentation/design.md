# CoordExp-Swift Geometry Flip Augmentation Design

## Context

CoordExp-Swift is a clean, agent-readable rebuild of the CoordExp supervised
training stack. The accepted V1 baseline owns raw JSONL loading, prompt/template
rendering, Qwen encoding, packing, token-wise supervision, loss computation,
training runtime, receipts, metrics, checkpoints, and inference/eval handoff.

The new augmentation path must preserve the pipeline order:

```text
Raw Dataset
-> Dataset Processing
-> Chat Template
-> Packing
-> Tokenization
-> Visual Processing
-> Model Construction
-> Forward Pass
-> Loss Computation
```

The motivating design note is
`docs/superpowers/specs/2026-07-07-coordexp-swift-geometry-flip-augmentation-design.md`.
That note replaced an earlier deterministic-expansion idea with the accepted
static stochastic one-view-per-source-example policy.

The target dataset family is the current COCO `len12000` coordinate JSONL
family, not older `max60` fixtures. Training uses Qwen no-resize image
processing, so augmentation must not introduce hidden resize, padding, or
materialized alternate image files.

DoRA, PEFT, special-token embedding deltas, Accelerate, and DeepSpeed behavior
are unchanged by this change. Their existing source-study and runtime gates
remain owned by the broader training-infrastructure changes; augmentation must
not add a new adapter/runtime dependency or weaken those gates.

## Goals / Non-Goals

**Goals:**

- Add train-only horizontal/vertical geometry flip augmentation.
- Preserve one source example -> one training presentation.
- Keep augmentation deterministic under resolved config, seed, source identity,
  policy version, and code identity.
- Apply the same realized transform to coordinate bins and image pixels.
- Make object-order interactions auditable for `source_order`, `geo_sorted`,
  and `random`.
- Include augmentation semantics in packing-cache identity and receipts.
- Keep cache hits stable: a cache hit reuses the materialized sampled view and
  does not resample transforms.

**Non-Goals:**

- No deterministic sample-count expansion.
- No epoch-varying online augmentation.
- No resizing, crop, rotation, color, blur, mosaic, mixup, learned policies, or
  test-time augmentation.
- No generated augmented JSONL files and no materialized flipped image files.
- No changes to prompt wording, coordinate-token grammar, loss semantics,
  optimizer groups, checkpoint format, evaluator reduction, inference decode, or
  Qwen resize policy.
- No broad augmentation registry or future placeholder framework in V1.

## Decisions

### Static Stochastic Presentation

Each source training example is transformed into exactly one presentation during
pack-cache materialization. Horizontal and vertical decisions are sampled as
independent Bernoulli variables from:

```yaml
data:
  augmentation:
    train:
      geometry_flips:
        enabled: true
        horizontal_prob: 0.5
        vertical_prob: 0.5
```

The realized transform id is one of `identity`, `hflip`, `vflip`, or `hvflip`.
The same sampled presentation is reused across epochs through the pack cache.

Alternative considered: deterministic expansion into identity/hflip/vflip/hvflip
variants. Rejected because it changes the number of examples and optimizer
steps, increases packing pressure, and makes comparisons to previous production
runs harder to interpret.

### Module Ownership

The implementation should be small and direct:

- `src/config/`: strict config schema and validation.
- `src/augmentation/geometry.py`: pure affine matrices, bbox corner transform,
  canonicalization, and validation over `0..999`.
- `src/augmentation/processor.py`: deterministic transform sampling,
  presentation construction, ordering preparation, metadata, and receipt
  summary inputs.
- `src/augmentation/factory.py`: construct the processor from resolved config.
- `src/data/` and `src/training/pipeline.py`: insert the processor after raw
  JSONL loading and before template rendering.
- `src/templates/renderer.py`: keep current ordering assertions; do not turn
  the renderer into an augmentation stage.
- `src/qwen/images.py`: apply the logical image transform in memory immediately
  before Qwen processing with `do_resize=False`.
- `src/training/pack_cache.py`: include augmentation determinants in semantic
  cache identity and write augmentation receipt provenance.

No package-level broad registry is needed in V1. A small factory is enough
because there is only one approved policy.

### Matrix Geometry

Flips are defined as affine matrices in inclusive coordinate-bin space:

```text
hflip: x' = 999 - x
vflip: y' = 999 - y
hvflip: both axes
```

The implementation transforms all four bbox corners and canonicalizes
`min_x,min_y,max_x,max_y`. This is preferred over procedural "do horizontal
then vertical" logic because the matrix is the single source of truth for the
coordinate transform, the image transform descriptor, and receipt evidence.

### Image Transform

Qwen image plans keep the original image path and add logical transform
metadata. Pixel flipping happens in memory before the Qwen image processor is
called. Because flips preserve width and height, no-resize grid planning,
visual-token counts, and packing cost stay unchanged.

Alternative considered: write flipped files into an image cache. Rejected for V1
because it adds storage lifecycle and stale-file risks without improving
training semantics.

### Object Ordering

Ordering behavior stays explicit:

- `source_order` preserves source object tuple order after bbox transform.
- `geo_sorted` transforms bboxes first, then reorders the presentation into
  top-to-bottom/left-to-right order before renderer assertion.
- `random` samples transform first, then lets the renderer apply its explicit
  run-controlled random order.

For `random`, every randomized presentation must emit an explicit object-order
seed/key receipt. This protects reproducibility when workers complete in a
different order and when a flipped presentation has a different
`presentation_id` from its `source_example_id`.

### Cache Identity And Receipts

Packing-cache identity includes resolved augmentation config, seed source,
policy version, augmentation code identity, and Qwen image materialization code
identity. Worker count remains provenance only and must not affect the semantic
cache key.

The receipt records mode `static_stochastic_view`, probabilities, seed,
input/output counts, transform counts, and enough per-presentation provenance
to reproduce transform and randomized object order. It is intentionally compact;
V1 does not create a broad report framework.

## Risks / Trade-offs

- Static view gives less stochastic diversity than epoch-varying augmentation
  -> accepted for V1 because reproducibility, cache reuse, and schedule
  comparability matter more.
- Flipped presentation ids can change renderer random ordering if the seed key
  is careless -> mitigate with explicit object-order seed/key receipts and
  deterministic key construction independent of worker completion order.
- Matrix math over inclusive bins can hide off-by-one mistakes -> mitigate with
  boundary tests at `0` and `999`, round-trip checks, and smoke artifact
  inspection.
- In-memory image transforms can drift from coordinate transforms -> mitigate by
  using the same transform id/matrix descriptor in image planning receipts and
  bbox metadata.
- Cache identity can become too coarse if code paths are omitted -> mitigate by
  including augmentation modules and relevant Qwen image materialization code in
  cache determinants.

## Migration Plan

1. Add the strict config schema with augmentation disabled by default.
2. Add pure matrix/bbox geometry helpers and tests.
3. Add the augmentation processor and factory, including deterministic sampling,
   presentation ids, metadata, `geo_sorted` preparation, and `random`
   object-order receipt inputs.
4. Insert the processor in the training data path before rendering/encoding.
5. Extend Qwen image planning/materialization to carry and apply logical flips
   in memory with `do_resize=False`.
6. Extend packing-cache fingerprint and receipts.
7. Add smoke evidence on the `len12000` path or an approved fixture with cache
   miss and cache hit behavior.

Rollback is simple: set `data.augmentation.train.geometry_flips.enabled: false`
or use a config without the augmentation block. Runs with augmentation disabled
must preserve previous training behavior.

## Open Questions

None blocking. Implementation may choose the exact private seed-string helper
shape, but it must satisfy the spec: stable under source row identity and
independent of worker order, process id, rank, and cache-hit behavior.
