# COCO 2017 (Instances / 80 Classes) → CoordExp JSONL

This folder contains a **small, reproducible** pipeline to download the official
COCO 2017 images + annotations and convert them into CoordExp's JSONL contract:
`docs/data/JSONL_CONTRACT.md`.

Outputs live under:
- `public_data/coco/raw/` (downloaded artifacts + converted JSONL)
- `public_data/coco/<preset>/` (shared unified pipeline artifacts via `public_data/run.sh`)

## What you get
- Raw COCO 2017 download (images + `instances_{train,val}2017.json`)
- Conversion to JSONL with:
  - `images`: `["images/<split>/<file_name>"]` (relative to JSONL directory)
  - `objects`: per-instance `{bbox_2d: [x1,y1,x2,y2], desc: <category_name>}`
  - Additional provenance fields for convenience: `image_id`, `file_name`, `category_id`, `category_name`

## Canonical COCO 2017 URLs
These are the standard hosted artifacts on `images.cocodataset.org`:
- `http://images.cocodataset.org/zips/train2017.zip`
- `http://images.cocodataset.org/zips/val2017.zip`
- `http://images.cocodataset.org/annotations/annotations_trainval2017.zip`

The downloader writes SHA256 checksums after download to keep runs reproducible.

## Recommended workflow (via unified runner)
From repo root (`/data/CoordExp`):

Download (full):
```bash
./public_data/run.sh coco download
```
Notes:
- This uses the runner's fast path (`aria2c` multi-connection) when available; install `aria2` and `unzip`.
- If you pass passthrough args after `--`, the runner falls back to the dataset plugin downloader.

Convert (full):
```bash
./public_data/run.sh coco convert
```

Validate raw JSONL contract (structure + bbox sanity):
```bash
./public_data/run.sh coco validate --raw-only --skip-image-check
python public_data/scripts/validate_coco2017_instances.py public_data/coco/raw/train.jsonl --categories_json public_data/coco/raw/categories.json --require-80 --sample_out public_data/coco/raw/sample_first5_train.jsonl
```

Optional: run the shared pipeline to generate training-ready preset artifacts
(offline resize + coord tokenization):
```bash
./public_data/run.sh coco all --preset rescale_32_768_bbox
```

Canonical preset artifacts:
- `train.jsonl` / `val.jsonl` (pixel-space)
- `train.norm.jsonl` / `val.norm.jsonl` (norm1000 integers)
- `train.coord.jsonl` / `val.coord.jsonl` (coord tokens)

Optional max-object filter (off by default):
```bash
PUBLIC_DATA_MAX_OBJECTS=60 ./public_data/run.sh coco all --preset rescale_32_768_bbox
```
When enabled, output preset naming uses canonical suffix `..._max60`.

## Prompt Variant for COCO-80

For COCO closed-class experiments, use the built-in `coco_80` prompt variant on
both training and inference configs.

Prompt composition follows:
- `{fixed_base_prompt} + {dynamic_variant_suffix}`
- fixed base keeps sorted object-order and coord-token instructions (universal)
- `coco_80` suffix adds the closed-class label policy

Training YAML:
```yaml
custom:
  extra:
    prompt_variant: coco_80
```

Inference YAML:
```yaml
infer:
  prompt_variant: coco_80
```

Parity guidance:
- Keep the same prompt variant between training and inference for reproducible eval.
- Verify artifacts include the resolved variant:
  - `<run_dir>/resolved_config.json` (`infer.prompt_variant`)
  - `<run_dir>/summary.json` (`infer.prompt_variant`)

The canonical COCO-80 class source-of-truth snapshot is
`public_data/coco/raw/categories.json`; the built-in `coco_80` variant is frozen
in code for deterministic behavior across machines.

## One-command smoke test (tiny, no images required)
This downloads only annotations, converts a small sample, validates, and emits a 5-image sample file:
```bash
./public_data/run.sh coco download -- --annotations-only && \
./public_data/run.sh coco convert -- --max_samples 100 && \
python public_data/scripts/validate_coco2017_instances.py public_data/coco/raw/train.jsonl \
  --categories_json public_data/coco/raw/categories.json \
  --require-80 \
  --sample_out public_data/coco/raw/sample_first5_train.jsonl
```

## Directory layout (after full download)
```text
public_data/coco/raw/
  downloads/
    train2017.zip
    val2017.zip
    annotations_trainval2017.zip
    SHA256SUMS.txt
  images/
    train2017/*.jpg
    val2017/*.jpg
  annotations/
    instances_train2017.json
    instances_val2017.json
  train.jsonl
  val.jsonl
  categories.json
  conversion_stats.json
```
