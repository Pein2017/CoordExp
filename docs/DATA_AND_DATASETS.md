# Data & Datasets

Comprehensive guide to data format, schema, dataset builders, and preprocessing pipeline.

**Source of truth**: `src/datasets/`, `src/datasets/data_details.md`, `src/datasets/geometry.py`

**Raw annotation intake** is covered in `DATA_PREPROCESSING_PIPELINE.md` (how dataset converters + `public_data/scripts/rescale_jsonl.py`/`convert_to_coord_tokens.py` produce the train/val JSONL that feed this pipeline).

---

## Table of Contents
- [Data Format](#data-format)
- [Dataset Pipeline](#dataset-pipeline)
- [Builders](#builders)
- [Preprocessors](#preprocessors)
- [Best Practices](#best-practices)

---

## Data Format

### JSONL Schema

Each record in your training data follows this structure:

```json
{
  "images": ["path/to/img1.jpg", "path/to/img2.jpg"],
  "objects": [
    {"bbox_2d": [x1, y1, x2, y2], "desc": "object description"},
    {"poly": [x1, y1, x2, y2, x3, y3, ...], "poly_points": M, "desc": "polygon description"},
    {"line": [x1, y1, ..., xn, yn], "line_points": N, "desc": "line description"}
  ],
  "width": 1920,
  "height": 1080,
  "summary": "optional: single-line English summary"
}
```

**Key Rules**:
- Image paths resolve relative to JSONL file directory (absolute paths also allowed)
- Exactly ONE geometry field per object (`bbox_2d`, `poly`, or `line`)
- For polygons: `poly` is a flat, even-length list (≥6 values / ≥3 points). `poly_points` is optional metadata but should match `len(poly) / 2` when present.
- For lines: `line_points` should equal number of coords ÷ 2 (optional but recommended; validation falls back to the coord count when absent)
- Coordinates are in pixel space with original `width`/`height`

### Geometry Types

| Type | Format | Use Case |
|------|--------|----------|
| **bbox_2d** | `[x1, y1, x2, y2]` | Axis-aligned boxes |
| **poly** | `[x1,y1, x2,y2, x3,y3, ...]` | Arbitrary polygons (even-length list, ≥3 points). Use `poly_points` to record vertex count. |
| **line** | `[x1,y1, ..., xn,yn]` + `line_points: N` | Polylines (cables, fibers) |

### Coordinate Normalization

**Pipeline default (post-conversion):**
- All public data JSONLs we ship/train on are pre-normalized to norm1000 `[0, 999]` for both numeric text and coord-token files. Pixel-space intermediates are temporary only.
- Training should therefore set `custom.emit_norm: none` for numeric runs (no runtime scaling). Coord-token runs bypass numeric scaling entirely.

**If you bring your own pixel JSONL:**
- You must either pre-normalize offline or set `custom.emit_norm: norm1000` to let the builder normalize; mixed coord systems are not supported.

**Enforcement:**
- Numeric runs with pre-normalized data are validated to be within `[0, 999]` and will raise if out-of-range.

### Modes: Dense vs Summary

**Dense Mode** (default):
```json
{
  "object_1": {"bbox_2d": [100, 200, 300, 400], "desc": "..."},
  "object_2": {"line_points": 4, "line": [50, 60, 80, 120, 130, 180, 180, 220], "desc": "..."}
}
```
## Dataset Pipeline

### Critical Configuration Requirement

**REQUIRED in all config files**:
```yaml
data:
  dataset: ["dummy"]  # NEVER REMOVE - required for ms-swift TrainArguments validation
```

**Why this is needed**:
- ms-swift's `TrainArguments.__post_init__()` validates that `dataset` or `cached_dataset` is non-empty
- This check happens during config initialization, before custom dataset loading
- Even though we load actual datasets via `custom.train_jsonl` and pass them directly to the trainer, the validation must pass first
- The `["dummy"]` placeholder satisfies the validation but is never actually used
- Removing this will cause: `ValueError: self.dataset: [], self.cached_dataset: []. Please input the training dataset.`

**Source**: `/data/ms-swift/swift/llm/argument/train_args.py:162-164`

### Architecture Overview

```
JSONL → DenseCaptionDataset → Collator → Trainer
```

**Key Components**:
1. **DenseCaptionDataset**: Mode selection (dense/summary), augmentation config, per-item orchestration
2. **Preprocessors**: Validation, augmentation (plugged into the dataset)
3. **Builder**: Message formatting (JSONLinesBuilder)
4. **Collator**: Tensor preparation with standard padding; optional packing wrapper flattens pre-packed lists when enabled.

### Visual Feature Distillation (optional)

- Enable via `custom.visual_kd` when you want to lock the vision/aligner stack to a teacher while giving the language tower more room.
- The dataset already supplies `pixel_values` and `image_grid_thw`; as long as a record contains images, the trainer captures and distills the corresponding activations automatically.
- Batches without images (e.g., summary-only validation groups) skip the extra loss—no action required.

### DenseCaptionDataset

**Role**:
- Selects dense vs summary mode per sample
- Applies augmentation/preprocessing
- Attaches metadata for downstream processing and template encoding

**Configuration**:
```yaml
custom:
  train_jsonl: /path/to/train.jsonl
  val_jsonl: /path/to/val.jsonl
  use_summary: false                # true → summary-only mode
  emit_norm: norm1000              # Coordinate format in text
```

## Conversion & QA Tooling

If your source is a human-annotation export, start with the intake guide (`docs/DATA_PREPROCESSING_PIPELINE.md`) and run the dataset converter plus `public_data/scripts/rescale_jsonl.py` (and optionally coord-token conversion) to produce train/val/tiny JSONL that already satisfy this contract.

- **Public datasets (`public_data/`)**:
  - See `PUBLIC_DATA.md` + `public_data/README.md` for LVIS/COCO/Objects365 download, conversion, sampling, visualization, and pytest coverage.
  - Each converter produces JSONL that matches this document’s schema; polygons include `poly_points`. **Legacy toggle**: `--poly-max-points N` can downgrade very high-vertex polygons to `bbox_2d`, but it is disabled by default and **not used by training** (we rely on sequence-length limits + dataset-level filtering instead).
  - For LVIS training, we typically use the **filtered** JSONLs to remove dense/low-diversity images (see `docs/DATA_PREPROCESSING_PIPELINE.md`):
    - `public_data/lvis/rescale_32_768_poly_20/train.filtered_max100_dense50_u3_t095.coord.jsonl`
    - `public_data/lvis/rescale_32_768_poly_20/val.filtered_max100_dense50_u3_t095.coord.jsonl`
- **Visualization**:
  - `vis_tools/vis_augment_compare.py` and friends overlay objects/summaries to validate augmentation and JSONL integrity. See `vis_tools/README_CROP_VIS.md`.
- **Chat template inspection**:
  - `scripts/inspect_chat_template.py --jsonl <path> --index 0` shows the exact rendered chat text and token IDs for a sample with the current prompts and Qwen3-VL chat template.

**Fusion status**: Multi-dataset fusion is temporarily disabled. Training currently assumes a single LVIS JSONL provided via `custom.train_jsonl` / `custom.val_jsonl`. Fusion helpers remain in the codebase for potential future use but are not wired into the runner.

For the universal JSONL record contract shared by all domains, see `docs/DATA_JSONL_CONTRACT.md`.

---

## Builders

### JSONLinesBuilder

**Purpose**: Formats single-image records into single-turn conversation messages

**Dense Mode**:
```python
# User message: embed the image followed by the prompt
[
  {"type": "image", "image": "path"},
  {"type": "text", "text": prompt}
]

# Assistant message: minimal object hierarchy (no per-image wrapper)
{
  "object_1": {"bbox_2d": [...], "desc": "类型/属性/..."},
  "object_2": {"line_points": 4, "line": [...], "desc": "..."}
}
```


**Key Behavior**:
- Attaches top-level `objects` with pixel coords (for template normalization)
- Geometries normalized based on `emit_norm` setting
- Deterministic ordering of object indices (`object_1`, `object_2`, ...)
- Consumes validated `ConversationRecord` objects and exposes augmentation telemetry (`pipeline.last_summary`) for downstream health checks.

---

## Preprocessors

### DenseCaptionPreprocessor

**Purpose**: Validation and light filtering

**Checks**:
- Schema validity (required fields present)
- Geometry field uniqueness (exactly one per object)
- Line point count matches `line_points`
- Image paths resolve correctly

**Action**: Raises `ValueError` on invalid records (fail-fast)

### AugmentationPreprocessor

**Purpose**: Apply geometry-aware augmentations

**Features**:
- Atomic updates (image + geometries transformed together)
- Preserves coordinate alignment
- See [DATA_AUGMENTATION.md](DATA_AUGMENTATION.md) for details
- Reads standardized telemetry (`AugmentationTelemetry`) with crop coverage, kept indices, and skip reasons to audit augmentation pipelines.

**Example**:
```yaml
custom:
  augmentation:
    enabled: true
    bypass_prob: 0.1              # 10% clean samples
    ops:
      - name: hflip
        params: { prob: 0.5 }
      - name: rotate
        params: { max_deg: 25.0, prob: 0.4 }
      - name: random_crop
        params: { scale: [0.7, 1.0], prob: 0.3 }
      - name: resize_by_scale
        params: { lo: 0.9, hi: 1.1, prob: 0.5 }
      - name: color_jitter
        params: { brightness: [0.75, 1.25], prob: 0.5 }
      # ✅ MUST be last: ensures final padding to multiple of 32
      - name: expand_to_fit_affine
        params: { multiple: 32 }
```

### Domain Context

CoordExp now targets general open-domain detection/grounding (public datasets such as LVIS/COCO/Objects365). Legacy telecom corpora are still supported via the same JSONL contract but no longer drive prompt wording or hierarchy requirements.

---

## Best Practices

### Data Preparation

✅ **Do**:
- Keep pixel coords on disk (template normalizes during training)
- Use relative image paths when possible
- Validate schema before training
- Include `width`/`height` metadata
- Test with small dataset first

❌ **Don't**:
- Mix coordinate systems in same file
- Omit required fields (`width`, `height`)
- Use absolute paths unnecessarily
- Skip validation (fail early is better)

### Schema Validation

```bash
# Recommended: validate before training
python -m src.datasets.validate_jsonl --input train.jsonl --verbose
```

**Common Issues**:
- Missing `line_points` for line geometries
- Multiple geometry fields per object
- Path resolution failures
- Width/height mismatch

### Performance Tips

1. **Image Loading**: Use relative paths from JSONL directory for portability
2. **Augmentation**: Enable only needed ops (each adds overhead)
3. **Packing**: Optional via the packing wrapper (training.packing). Enables padding_free collation with packed sequences; leave disabled for evaluation unless explicitly testing packed eval.

### Debugging

**Enable debug mode**:
```bash
python -m src.sft --config config.yaml --debug
```

**Check first batch**:
```python
from src.datasets import DenseCaptionDataset
ds = DenseCaptionDataset(config)
item = ds[0]
print(item.keys())  # input_ids, labels, pixel_values, ...
```

---

## Collation (padding vs. packing)

- **Default**: Standard padded batches (`padding_free=false`).
- **Packing (opt-in)**: Set `training.packing: true` plus `packing_length/buffer/min_fill_ratio` to enable the packing wrapper; this sets `template.packing=true` and uses padding_free collation. Training metrics are aggregate-only when packing is on; per-dataset telemetry is dropped. Evaluation stays un-packed unless explicitly enabled.

---

## Verification Checklist

Before training:

- [ ] JSONL schema valid (all required fields present)
- [ ] Geometry fields correct (one per object)
- [ ] Line objects have `line_points` matching coord count
- [ ] Image paths resolve correctly
- [ ] Width/height metadata present
- [ ] Summary field present (if using summary mode)
- [ ] Coordinates pre-normalized to `[0, 999]` (or plan to normalize offline before training)
- [ ] No duplicate objects or malformed geometries
- [ ] Test with `--debug` flag first

### Token-type metrics (coord vs text)

- Enable with `custom.token_type_metrics.enabled: true`; defaults to `include: ["lvis"]`, `exclude: []`.
- Works on padded and packed batches: token types are computed per sample pre-pack and concatenated; if alignment fails the metrics are skipped (training continues).
- Metrics are aggregate-only: logs `token_acc_top5`, `text_token_acc`, and per-type breakdowns (`desc_token_frac`, `format_token_frac`, `coord_token_frac`, plus `desc_token_acc` / `format_token_acc` and their top-5 variants); no per-dataset buckets.
- NaN-safe: batches with zero supervised tokens are skipped.

### Coord distribution loss (coord tokens)

CoordExp trains coordinate tokens with **distribution-based supervision** only:

- Standard full-vocab CE is applied **only to non-coordinate tokens** (text + JSON structure).
- At `<|coord_*|>` positions, the model is supervised via:
  - `softCE`: soft cross-entropy between predicted coord-bin distribution `p` and a unimodal Gaussian soft label `q`
  - `W1`: 1D Wasserstein-1 distance on discrete bins via CDF differences between `p` and `q`
  - `gate`: coord-vocab gate loss that penalizes probability mass leaking to non-coord tokens

```yaml
custom:
  coord_soft_ce_w1:
    enabled: true
    # total_loss += soft_ce_weight * softCE + w1_weight * W1 + gate_weight * gate
    soft_ce_weight: 1.0
    w1_weight: 1.0
    gate_weight: 1.0
    temperature: 1.0
    target_sigma: 2.0
    target_truncate: 16
```

Notes:
- Coord-token positions are identified from **labels** (teacher forcing), never from model predictions.
- No decoded coordinates (argmax/expectation/median) are computed for training or metrics.
- Logged losses (train/eval parity, eval uses `eval_` prefix): `coord_softce_w1/loss`, `coord_softce_w1/soft_ce`, `coord_softce_w1/w1`, `coord_softce_w1/gate`, plus `coord_softce_w1/coord_vocab_mass` and `coord_softce_w1/coord_tokens`.

---

## See Also

- **Augmentation**: [DATA_AUGMENTATION.md](DATA_AUGMENTATION.md) - Geometry-aware transforms
- **Training**: [REFERENCE.md](REFERENCE.md#training) - Full training guide
- **Architecture**: [README.md](README.md#architecture) - End-to-end pipeline
- **Upstream Models**: [UPSTREAM_DEPENDENCIES.md](UPSTREAM_DEPENDENCIES.md) - HF Qwen3-VL + ms-swift background

---

**Last Updated**: 2025-11-24 (geometry schema + links)
