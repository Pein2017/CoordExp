# Data Contract + Dataset Pipeline

Contract:
- Global JSONL contract is documented in `docs/data/JSONL_CONTRACT.md`.
- Each record has:
  - `images: [str]` (paths resolved relative to JSONL dir unless absolute),
  - `objects: [...]` where each object has non-empty `desc` and exactly ONE geometry (`bbox_2d` OR `poly` (+ optional `poly_points`)),
  - `width`, `height` always present,
  - optional `summary`, `metadata`.

Coord conventions (important for training):
- The training runner currently enforces `custom.emit_norm: none` (see `CustomConfig.__post_init__` in `src/config/schema.py`).
- That means runtime normalization is disabled; numeric geometry used in text MUST already be within `[0, 999]` (norm1000).
- Coord-token mode (`custom.coord_tokens.enabled: true`) expects `<|coord_k|>` tokens (k in [0,999]) in the JSONL.

Runner behavior:
- `src/sft.py` auto-sets `ROOT_IMAGE_DIR` to the parent directory of `custom.train_jsonl` (or `custom.fusion_config`) if unset.
- In fusion mode, `custom.fusion_config` loads multiple datasets via `FusionConfig` and builds `FusionCaptionDataset`.

Per-sample flow (single JSONL path):
- Dataset: `BaseCaptionDataset.from_jsonl()` in `src/datasets/dense_caption.py`.
- `__getitem__` roughly does:
  - epoch-seeded index permutation (deterministic shuffle),
  - deep copy record -> optional preprocessors (validation, augmentation),
  - optional object reordering (`custom.object_ordering`: sorted|random),
  - if coord_tokens enabled: `annotate_coord_tokens(record)` (`src/coord_tokens/validator.py`),
  - build messages via `JSONLinesBuilder` (`src/datasets/builders/jsonlines.py`),
  - encode via the ms-swift template -> tensors (`input_ids`, `labels`, `pixel_values`, `image_grid_thw`, etc),
  - attaches raw `messages` + `assistant_payload` / `objects` metadata for downstream trainers.

Augmentation:
- YAML-driven augmentation under `custom.augmentation` and optional curriculum (`custom.augmentation_curriculum`).
- Implemented in `src/datasets/augmentation/*` and geometry helpers in `src/datasets/geometry.py` (affine transforms update geometry atomically).
- Training assumes images are already prepared offline (no runtime "smart resize" in the training path).
