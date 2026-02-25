# Fusion Dataset (Multi-JSONL Training)

CoordExp can train/evaluate on **multiple JSONL datasets** without pre-merging by using a *fusion config* file.

- Enable it in your training YAML via `custom.fusion_config: <path/to/fusion.yaml>`.
- When `custom.fusion_config` is set, the runner **ignores** `custom.train_jsonl` / `custom.val_jsonl`.

Note:
- Fusion-config training is supported, but it is not the default workflow in this repo. Prefer single-dataset training unless you explicitly need mixing.

## 1) Fusion Config Schema

Fusion configs are YAML/JSON mappings with Qwen3-VL-style containers:

- `targets`: list of dataset entries
- `sources`: optional list of dataset entries (accepted for compatibility; treated the same as `targets`)
- `extends`: optional string or list of strings for inheritance (paths relative to the current file)

At least one dataset entry across `targets` and `sources` is required.

### Dataset Entry Fields

Each dataset entry supports:

- Required:
  - `dataset`: wrapper key (e.g. `lvis`, `coco`, `objects365`, `vg`, `jsonl`)
  - `train_jsonl`: path to train JSONL
  - `template`: template ID (validated; unknown values error)
- Optional:
  - `name`: dataset ID override (must be unique across all entries)
  - `val_jsonl`: path to eval JSONL, or `null` to skip eval for this dataset
  - `ratio`: float, defaults to `1.0`
  - `sample_without_replacement`: bool, defaults to `false` (prevents upsampling beyond the pool)
  - `user_prompt`: per-dataset user prompt override (string; useful for bbox-only vs poly-prefer supervision)
  - `system_prompt`: per-dataset system prompt override (string; use sparingly)
  - `augmentation_enabled`: bool (upstream-compat; only applies if you configured augmentation globally)
  - `curriculum_enabled`: bool (upstream-compat; only applies if you configured augmentation curriculum globally)
  - `mode`: `dense` / `summary` (validated for compatibility, but currently ignored; use `custom.use_summary` to control mode)
  - `max_objects_per_image`: optional object-cap preprocessor (drops extra objects when an image has too many instances)

Notes:

- Dataset ID = `name` if provided, otherwise `dataset`.
- Unknown `dataset` wrapper keys error (to avoid silently ignoring typos).
- JSONL path resolution:
  - absolute paths are kept as-is
  - paths starting with `./` or `../` are resolved relative to the fusion config file directory
  - other relative paths (e.g. `public_data/...`) are treated as repo-root/CWD relative (launchers run from repo root)

### Extends Merge Semantics

When using `extends`, dataset entries are merged by **dataset ID**:

- Dataset ID = `name` if provided, otherwise `dataset`.
- If both base and override define the same dataset ID, the effective entry is a deep-merge (override keys win).
- Base ordering is preserved; new override-only entries are appended.

### Known Template IDs (CoordExp v1)

Fusion config `template:` is currently validated against the known IDs:

- `bbox_only`
- `poly_preferred`

CoordExp v1 uses a single runtime template instance; `template` is primarily a typo-guard and a compatibility hook.

## 2) Sampling Semantics (CoordExp v1)

CoordExp treats **all dataset entries uniformly** (no target/source semantic split).

### Training quota per dataset

For each dataset entry `i`:

- Load its pool `pool_i` from `train_jsonl`
- Compute per-epoch quota:

`quota_i = round(len(pool_i) * ratio_i)`

Sampling behavior:

- If `quota_i <= len(pool_i)`: sample without replacement (shuffle + take first `quota_i`)
- If `quota_i > len(pool_i)`: include the whole pool and sample the extra with replacement
  - Set `sample_without_replacement: true` on the dataset entry to prevent upsampling (cap at `len(pool_i)`).

All per-dataset samples are concatenated and then shuffled for the epoch.

### Evaluation

The eval dataset is built from **any entry with a non-null `val_jsonl`**.

- `val_jsonl: null` (or missing) => that dataset contributes **no** eval samples
- Eval ordering does not matter

## 3) Interaction With Packing + Coord Tokens

- **Packing (Stage-1)**: fusion datasets are **not** compatible with Stage-1 dataset-level static packing.
  - Stage-1 packed training requires `training.packing_mode: static` and fails fast on fusion datasets (order-sensitive sample schedule).
  - Recommended: disable `training.packing` for fusion runs, or materialize an offline merged JSONL if you need packing.
- **Coord-token mode (required)**: fusion datasets require `custom.coord_tokens.enabled: true` and `custom.coord_tokens.skip_bbox_norm: true`.

## 4) Example

Fusion config (containers + ratios + `val_jsonl: null`):

```yaml
# configs/fusion/examples/lvis_vg.yaml

targets:
  - dataset: lvis
    name: lvis
    train_jsonl: public_data/lvis/rescale_32_768_bbox_max60/train.bbox_only.max60.coord.jsonl
    val_jsonl: public_data/lvis/rescale_32_768_bbox_max60/val.bbox_only.max60.coord.jsonl
    template: poly_preferred
    ratio: 1.0

sources:  # accepted for compatibility; treated the same as targets
  - dataset: vg
    name: vg
    train_jsonl: public_data/vg/rescale_32_768_bbox/train.coord.jsonl
    val_jsonl: null
    template: poly_preferred
    ratio: 0.2
```

Training YAML:

```yaml
# configs/fusion/sft_lvis_vg.yaml

extends: ../base.yaml
training:
  packing: false  # Stage-1 packing is not supported for fusion datasets (see section 3)
custom:
  fusion_config: configs/fusion/examples/lvis_vg.yaml
```

## 5) When To Prefer Offline Merge

Runtime fusion is supported and is a good fit for quick multi-dataset mixing ablations.

Offline merge can still be useful when:

- you need a single JSONL for external tools
- you want a fixed materialized dataset snapshot
- you need Stage-1 packing (offline merge produces a single stable dataset schedule)

You can materialize a fused JSONL using the helper in `src/datasets/fusion.py` (`build_fused_jsonl`) or the thin wrapper script `public_data/scripts/merge_jsonl.py`.
