# Visual Genome (VG) -> CoordExp JSONL

This repo consumes detection/grounding data via a single JSONL contract (`docs/DATA_JSONL_CONTRACT.md`).
Visual Genome is supported via a downloader + converter script that follows the HuggingFace dataset loader:
`ranjaykrishna/visual_genome` (the HF repo hosts the loader; the data are downloaded from the official VG mirrors).

## Quick Start

All commands should be run from the repo root:
`/data/home/xiaoyan/AIteam/data/CoordExp`

## Datasets supported

This repo supports two VG-derived datasets that share the same image pool:

- `vg` (objects): `desc = object name` (short label), `bbox_2d = object box`
- `vg_ref` (region phrases): `desc = region phrase` (free-form text), `bbox_2d = region box`

Note: region boxes are not 1:1 aligned with object boxes; treat `vg_ref` as a separate dataset/task.

### Preferred: unified runner (one command)

The unified public-data runner is the recommended entrypoint:

```bash
./public_data/run.sh vg all --preset rescale_32_768_bbox -- --objects-version 1.2.0
```

VG region phrases (recommended to run after `vg download` so images can be reused):

```bash
./public_data/run.sh vg_ref all --preset rescale_32_768_bbox
```

For a smoke run that stops downloads after 300 seconds:

```bash
./public_data/run.sh vg all --preset rescale_32_768_bbox -- --objects-version 1.2.0 --max-seconds 300
```

Tip: `all` forwards passthrough args after `--` only to dataset-specific steps (download/convert). If you need to tune
shared preprocessing options (e.g., `--image-factor`, `--max-pixels`), run `rescale`/`coord` separately.

### 0) (Optional) Network proxy

If you need a proxy for HuggingFace access, export it in your shell before running:

```bash
export http_proxy=http://127.0.0.1:9090
export https_proxy=http://127.0.0.1:9090
```

### 1) Download + convert (pixel JSONL)

This writes everything under `public_data/vg/`:

```bash
PYTHONPATH=. conda run -n ms python public_data/scripts/prepare_visual_genome.py \
  --download \
  --objects-version 1.2.0 \
  --val-mod 5
```

For a smoke run that stops downloads after 300 seconds:

```bash
PYTHONPATH=. conda run -n ms python public_data/scripts/prepare_visual_genome.py \
  --download \
  --objects-version 1.2.0 \
  --max-seconds 300
```

Outputs:
- `public_data/vg/raw/train.jsonl`
- `public_data/vg/raw/val.jsonl` (deterministic split; see below)

### 2) Smart-resize (recommended)

Follow the shared intake pipeline (`docs/DATA_PREPROCESSING_PIPELINE.md`).

Example:

```bash
PYTHONPATH=. conda run -n ms python public_data/scripts/rescale_jsonl.py \
  --input-jsonl public_data/vg/raw/train.jsonl \
  --output-jsonl public_data/vg/rescale_32_768_bbox/train.jsonl \
  --output-images public_data/vg/rescale_32_768_bbox \
  --image-factor 32 \
  --max-pixels $((32*32*768)) \
  --min-pixels $((32*32*4)) \
  --num-workers 8 \
  --relative-images
```

Larger budget example (`max_pixels = 32*32*1024`):
```bash
PYTHONPATH=. conda run -n ms python public_data/scripts/rescale_jsonl.py \
  --input-jsonl public_data/vg/raw/train.jsonl \
  --output-jsonl public_data/vg/rescale_32_1024_bbox/train.jsonl \
  --output-images public_data/vg/rescale_32_1024_bbox \
  --image-factor 32 \
  --max-pixels $((32*32*1024)) \
  --min-pixels $((32*32*4)) \
  --num-workers 8 \
  --relative-images
```

### 3) Convert to coord tokens (train format)

CoordExp training defaults to coord-token supervision. Convert pixel coords to `<|coord_k|>`:

```bash
PYTHONPATH=. conda run -n ms python public_data/scripts/convert_to_coord_tokens.py \
  --input public_data/vg/rescale_32_768_bbox/train.jsonl \
  --output-tokens public_data/vg/rescale_32_768_bbox/train.coord.jsonl
```

Training config knobs (already set in `configs/dlora/sft_base.yaml`):
- `custom.coord_tokens.enabled: true`
- `custom.coord_tokens.skip_bbox_norm: true`

### 4) Merge VG + LVIS for single-dataset training

Multi-dataset fusion is deprecated in CoordExp. To train on LVIS + VG together, merge JSONLs offline.

Important: merging requires rewriting relative image paths, because loaders resolve images relative to the merged JSONL directory.
Use:

```bash
PYTHONPATH=. conda run -n ms python public_data/scripts/merge_jsonl.py \
  --inputs \
    public_data/lvis/rescale_32_768_poly_20/train.filtered_max100_dense50_u3_t095.coord.jsonl \
    public_data/vg/rescale_32_768_bbox/train.coord.jsonl \
  --output public_data/mix/lvis_vg/train.coord.jsonl \
  --strategy round_robin
```

Then point training to the merged file (see `configs/dlora/sft_lvis_vg_mix.yaml`).

## Dataset-specific Notes

### No official splits

VG does not provide official train/val splits. The converter creates a deterministic split:
`val` if `image_id % val_mod == 0` (default `val_mod=5` => ~20% val / 80% train).

### Bounding box conversion and clipping

VG object boxes are provided as `(x, y, w, h)`; we convert to `bbox_2d=[x1,y1,x2,y2]`.

We clip coordinates into `[0,width-1] / [0,height-1]` by default. This is important because
`public_data/scripts/convert_to_coord_tokens.py` scales pixels by `coord = round(v/width*1000)`,
and a coordinate equal to `width` would map to 1000 (out of the allowed 0-999 range).

### Object names

VG provides `names: [str, ...]` per object. The converter uses the first non-empty name and sanitizes it
to keep it compatible with CoordExp's JSON emission constraints (single line; no control whitespace).

### Region phrases (`vg_ref`)

If you have `region_descriptions.json` available, you can convert region phrases into the same JSONL contract:

```bash
./public_data/run.sh vg_ref convert
./public_data/run.sh vg_ref rescale --preset rescale_32_768_bbox
./public_data/run.sh vg_ref coord --preset rescale_32_768_bbox
```

Tip: if `public_data/vg/raw/images/` already exists, `vg_ref` will skip re-downloading images and reuse them.

## Validation Checklist

- Pixel JSONL sanity (bbox-only validator):

```bash
PYTHONPATH=. conda run -n ms python public_data/scripts/validate_jsonl.py \
  public_data/vg/rescale_32_768_bbox/train.jsonl
```

- Prompt/template compliance on coord-token JSONL:

```bash
PYTHONPATH=. conda run -n ms python scripts/inspect_chat_template.py \
  --jsonl public_data/vg/rescale_32_768_bbox/train.coord.jsonl \
  --index 0
```

## Included sample

`public_data/vg/sample/` contains a tiny synthetic VG-like annotation pair plus the converted JSONL.
It is meant only to demonstrate the format and validate the pipeline tooling without downloading the full dataset.
