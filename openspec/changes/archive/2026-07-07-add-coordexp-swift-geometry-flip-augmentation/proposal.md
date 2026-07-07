## Why

CoordExp-Swift needs a small geometry augmentation path that can improve
robustness without breaking the strict image/coordinate/template alignment that
the new training infrastructure was built to expose. The previous provisional
augmentation note used deterministic sample expansion, but the accepted design
is a static stochastic one-view-per-source-example contract that keeps pack
size, schedule meaning, and cache reuse simple.

The change protects accuracy and precision first by making the exact sampled
flip, coordinate matrix, image transform, object order, and cache identity
auditable. It improves training efficiency by preserving the one-example-in,
one-presentation-out invariant and reusing packed caches. It keeps the
implementation simple by supporting only horizontal and vertical flips in V1,
while leaving a clean augmentation processor boundary for future policies.

## What Changes

- Add train-only static stochastic geometry flip augmentation for
  CoordExp-Swift supervised training.
- Support independent horizontal and vertical flip probabilities:
  `data.augmentation.train.geometry_flips.enabled`,
  `horizontal_prob`, and `vertical_prob`.
- Sample exactly one realized transform per source training example during
  pack-cache materialization: `identity`, `hflip`, `vflip`, or `hvflip`.
- Preserve `output_example_count == input_example_count`; no deterministic
  expansion or duplicated augmented JSONL is introduced.
- Define flip behavior as affine matrices over the inclusive `0..999`
  coordinate-token plane and require bbox canonicalization through transformed
  corners.
- Apply the same realized transform to in-memory Qwen image materialization
  with `do_resize=False`; flipped image files are not materialized.
- Make transform assignments deterministic under resolved config, runtime seed,
  policy version, and source example identity; cache hits must reuse the sampled
  view without resampling.
- Include augmentation config, seed, policy version, augmentation code identity,
  and Qwen image materialization code identity in the packing-cache semantic
  fingerprint.
- Record compact augmentation receipt metadata, including transform counts and
  object-order seed/key receipts for randomized object ordering.
- Keep eval and inference unaugmented in V1.

No breaking change is intended for runs with augmentation disabled.

## Capabilities

### New Capabilities

- `coordexp-swift-geometry-augmentation`: train-only static stochastic
  horizontal/vertical flip augmentation, including config validation, matrix
  bbox transforms, in-memory Qwen image transforms, object-order interaction,
  packing-cache identity, receipts, and smoke evidence.

### Modified Capabilities

None. The relevant training, packing, Qwen image, and cache contracts currently
live in active changes rather than stable specs in this worktree, so this change
introduces a focused new capability instead of modifying a stable one.

## Impact

Affected surfaces:

- `src/config/` for the augmentation config schema and validation.
- `src/augmentation/` for matrix geometry, deterministic sampling, presentation
  construction, and metadata.
- `src/data/` and `src/training/pipeline.py` for inserting augmentation after
  raw JSONL loading and before rendering/encoding/packing.
- `src/templates/renderer.py` interaction for `source_order`, `geo_sorted`, and
  `random` object ordering.
- `src/qwen/images.py` for in-memory logical image transforms during Qwen image
  materialization.
- `src/training/pack_cache.py` for semantic fingerprint and receipt updates.
- `tests/` for config, bbox transform, processor/order, cache identity,
  cache-hit, Qwen image, and dry-run/smoke coverage.
- `docs/superpowers/specs/2026-07-07-coordexp-swift-geometry-flip-augmentation-design.md`
  as the proposal-level design note that motivated this OpenSpec contract.

The implementation must not change prompt wording, coordinate-token grammar,
loss semantics, optimizer grouping, checkpoint format, evaluator metric
reduction, inference behavior, or Qwen resizing behavior.
