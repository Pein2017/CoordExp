# COCO Test-Dev Submission Infrastructure

Date: 2026-03-10

## Status

Infrastructure landed for building a ready-to-upload official COCO detection
submission artifact from CoordExp inference outputs.

This note documents the current supported path. It does **not** record an
official submission result yet.

## What Was Added

- COCO public-data download support for official-test assets:
  - `test2017.zip`
  - `image_info_test2017.zip`
- COCO converter support for:
  - `test-dev` -> `image_info_test-dev2017.json`
  - `test` -> `image_info_test2017.json`
- Repo-native raw JSONL outputs:
  - `public_data/coco/raw/test-dev.jsonl`
  - `public_data/coco/raw/test.jsonl`
- Dedicated export entrypoint:
  - `scripts/export_coco_submission.py`
- YAML template for submission export:
  - `configs/eval/coco_submission.yaml`
- Inference template targeting official test-dev:
  - `configs/infer/ablation/coco80_testdev_desc_first.yaml`

## Supported Workflow

1. Download official-test assets:

```bash
./public_data/run.sh coco download -- --include-test
```

2. Convert the official test-dev JSONL:

```bash
./public_data/run.sh coco convert -- --include-test --test-split test-dev
```

3. Build the 1024-budget resized inference input:

```bash
PYTHONPATH=. conda run -n ms python public_data/scripts/rescale_jsonl.py \
  --input-jsonl public_data/coco/raw/test-dev.jsonl \
  --output-jsonl public_data/coco/rescale_32_1024_bbox/test-dev.jsonl \
  --output-images public_data/coco/rescale_32_1024_bbox \
  --image-factor 32 \
  --max-pixels $((32*32*1024)) \
  --min-pixels $((32*32*4)) \
  --relative-images
```

4. Run inference:

```bash
PYTHONPATH=. conda run -n ms python scripts/run_infer.py \
  --config configs/infer/ablation/coco80_testdev_desc_first.yaml
```

5. Score predictions:

```bash
PYTHONPATH=. conda run -n ms python scripts/postop_confidence.py \
  --config configs/postop/confidence.yaml
```

6. Export the official submission JSON:

```bash
PYTHONPATH=. conda run -n ms python scripts/export_coco_submission.py \
  --config configs/eval/coco_submission.yaml
```

## Output Artifacts

- scored prediction artifact:
  - `<run_dir>/gt_vs_pred_scored.jsonl`
- official submission JSON:
  - `<run_dir>/coco_submission.json`
- export summary:
  - `<run_dir>/submission_summary.json`
- optional semantic mapping report:
  - `<run_dir>/semantic_desc_report.json`

## Important Notes

- The export path joins the scored predictions back to the source COCO JSONL by
  record order, recovers the real COCO `image_id`, and projects boxes from the
  resized 1024-budget inference resolution back to the original COCO test-dev resolution.
- This avoids widening the main `gt_vs_pred.jsonl` contract just for official
  submission export.
- The shared `public_data/run.sh` preset pipeline still manages `train` / `val`
  splits only. Official test-dev currently uses the raw JSONL path under
  `public_data/coco/raw/`.

## Remaining Manual Steps

- Choose the final checkpoint path to benchmark.
- Run the full inference + confidence pipeline on `test-dev`.
- Upload `coco_submission.json` to the official COCO evaluation server.
- Record the returned AP / AP50 / AP75 / APs / APm / APl metrics in a benchmark note.
