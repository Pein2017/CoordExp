# Qwen3-VL Single-Image Pack Smoke Fixture

This fixture is the permanent source anchor for the first CoordExp-swift vertical smoke. It is intentionally small, local, and boring: two real COCO-derived single-image examples, each with exactly two objects, canonical integer coordinate-bin boxes, and copied local image files.

## Files

- `examples.jsonl`: canonical `RawExample` JSONL. It uses `example_id`, `image`, `objects`, and `metadata`; source-dialect fields such as `images`, `desc`, and `bbox_2d` appear only inside provenance.
- `images/train2017/*.jpg`: copied fixture-local images. Training and tests must not depend on the external source image paths at runtime.
- `checksums.json`: fixture-specific SHA-256 image checksums, source row hashes, source row numbers, source file stats, selected object ids, and selection rationale.
- `config.yaml`: self-contained smoke training config. It uses `adapter.type: dora`, `packing.global_max_length: 12000`, `training.effective_batch_size: 2`, `training.max_steps: 5`, and `eval.forward.steps: [2, 4]`.
- `expected_rendered.json`: frozen renderer-generated snapshot produced through the real `RawExample -> RenderedExample` path. It is review-gated and must not be updated by the same test that compares it.

## Invariants

- All image paths in `examples.jsonl` are fixture-relative and point under `images/`.
- Each example has one image and two source-order objects with distinct descriptions.
- Bboxes are `[x1, y1, x2, y2]` integer coordinate bins in `[0, 999]`.
- Image dimensions are copied from the current `len12000` source rows and are divisible by 32 for the approved Qwen no-resize processor path.
- Smoke run outputs must go under `run.artifact_root` (`outputs/smoke/qwen3_vl_single_image_pack/`), not under this fixture directory.

## Source Selection

The fixture was materialized from `/data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000/train.coord.jsonl`.

- Row 3 -> `coco2017_train_000000000030__smoke2obj`
- Row 5 -> `coco2017_train_000000000036__smoke2obj`

They are the first two source-order rows found that are single-image, exactly two-object, distinct-description, description-safe, existing-image, and no-resize-compatible with 32-aligned dimensions. No object reduction was applied.

## Intended Run

After the new training infrastructure exists:

```bash
python -m src.train --config tests/fixtures/smoke/qwen3_vl_single_image_pack/config.yaml
```

Before full training exists, config and fixture validators should still be able to load this directory, check path locality, parse `examples.jsonl`, verify image checksums, and confirm that generated artifacts would land under the ignored smoke output root.
