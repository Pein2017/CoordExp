## 1. Source Study And Fixture Pinning

- [x] 1.1 Inspect current `src/templates/renderer.py` object-order behavior for `source_order`, `geo_sorted`, and `random`, and record the exact seed/key inputs augmentation must preserve.
- [x] 1.2 Inspect current `src/qwen/images.py` no-resize image planning/materialization path and identify the minimal hook point for logical in-memory flips.
- [x] 1.3 Inspect current `src/training/pack_cache.py` semantic fingerprint and receipt construction to identify every determinant and receipt field that must include augmentation.
- [x] 1.4 Pin or reuse a small `len12000` smoke fixture/config that exercises at least two objects and can run with `packing.global_max_length: 12000`.
- [x] 1.5 Confirm this change does not alter DoRA, PEFT, special-token embedding, Accelerate, or DeepSpeed code paths; if any adapter-enabled smoke is reused, keep the existing DoRA source-study gate as a prerequisite.

## 2. Config And Pure Geometry Tests

- [x] 2.1 Add failing config tests for omitted augmentation, disabled augmentation, valid probabilities, unknown fields, and invalid probability values.
- [x] 2.2 Add failing pure-geometry tests for `identity`, `hflip`, `vflip`, and `hvflip` affine matrices over boundary boxes at `0` and `999`.
- [x] 2.3 Add failing tests that transformed bboxes are canonicalized from transformed corners and contract-error on degenerate or out-of-range results.
- [x] 2.4 Implement the strict augmentation config schema under `src/config/` with augmentation disabled by default.
- [x] 2.5 Implement `src/augmentation/geometry.py` with matrix descriptors, bbox corner transform, canonicalization, and validation.

## 3. Processor And Object-Order Contract

- [x] 3.1 Add failing processor tests proving one source example emits exactly one presentation and `output_example_count == input_example_count`.
- [x] 3.2 Add failing processor tests proving same seed/config/source identity yields the same transform assignment and changing seed can change assignments.
- [x] 3.3 Add failing processor tests for `source_order`, transformed-then-asserted `geo_sorted`, and augmented `random` ordering with explicit object-order seed/key receipt.
- [x] 3.4 Add failing worker-order determinism tests proving presentation transform ids and random object-order seed/keys do not depend on completion order.
- [x] 3.5 Implement `src/augmentation/processor.py` and `src/augmentation/factory.py` with deterministic static stochastic sampling, presentation ids, metadata, and ordering preparation.

## 4. Pipeline And Qwen Image Materialization

- [x] 4.1 Insert the augmentation processor after raw JSONL loading and before template rendering/encoding in the training pipeline.
- [x] 4.2 Add failing Qwen image tests proving logical flips preserve no-resize image dimensions, `image_grid_thw`, visual token count, and packing image-token cost.
- [x] 4.3 Extend Qwen image plans/materialization to carry logical transform metadata and apply the in-memory pixel transform before Qwen processing.
- [x] 4.4 Add artifact/debug evidence showing original image path plus logical transform id for flipped presentations without materialized flipped image files.

## 5. Packing Cache Identity And Receipts

- [x] 5.1 Add failing fingerprint tests proving augmentation config, seed source, policy version, augmentation code identity, and Qwen image materialization code identity affect the semantic packing-cache fingerprint.
- [x] 5.2 Add failing fingerprint tests proving internal pack-cache materialization worker count remains provenance only and does not affect semantic cache identity.
- [x] 5.3 Add failing cache-hit tests proving cached sampled presentations are reused without resampling transforms.
- [x] 5.4 Implement packing-cache determinant updates for augmentation modules and relevant Qwen image materialization code.
- [x] 5.5 Implement compact augmentation receipts with mode, policy, probabilities, seed/source, input/output counts, transform counts, dataset family, and object-order seed/key receipts when random ordering is used.

## 6. Smoke And Contract Verification

- [x] 6.1 Run targeted tests for config, geometry, processor/order, Qwen image materialization, packing cache, and pipeline assembly.
- [x] 6.2 Run an augmentation-enabled dry run with `packing.global_max_length: 12000`, sample-limited `len12000` data, and a fresh cache root; verify the receipt records static stochastic view and `output_example_count == input_example_count`.
- [x] 6.3 Re-run the same dry run with the same cache root and verify it reports a cache hit and does not resample transforms.
- [x] 6.4 Run a five planned-step smoke with two `eval.forward` steps, real `global_max_length: 12000`, metrics, checkpoint metadata, and `checkpoint-final` when GPU availability allows.
- [x] 6.5 Run residue checks for stale deterministic-expansion wording, stale `include_composed`, stale `max60` authority in the augmentation path, and old OpenSpec/old-src authority wording.
- [x] 6.6 Run `openspec validate add-coordexp-swift-geometry-flip-augmentation --strict` and `git diff --check`.
