# Detection Evaluator (CoordExp)

Offline evaluator to compute COCO-style metrics and/or an F1-ish set-matching metric from CoordExp JSONL.

## Inputs
- `pred_jsonl`: Pixel-space predictions produced by `scripts/run_infer.py` (unified engine); **must include inline `gt`** per line.
- Geometry objects live under `gt` / `pred` arrays with fields `{type, points, desc, score}`; legacy `predictions`/raw-text parsing is no longer used.
- Width/height are taken from the inline GT; size mismatches are counted but do not abort.

## Behavior
- Pixel-ready consumption: no coord-mode inference/denorm is performed; polygons are exported directly as COCO `segmentation` (mask IoU), bboxes as `bbox`. Lines are carried through in reports but excluded from metrics (and currently unused in CoordExp datasets).
- One geometry per object; degenerate/invalid entries are dropped with counters and recorded per-image for diagnostics.
- Categories use exact desc strings; unknowns bucket to `unknown` by default (configurable to drop).
- Scores are fixed at 1.0 (greedy decoding outputs have no reliable confidence); any provided `score` fields are ignored.
- COCOeval runs for bbox and segm (when polygons exist). TODO: polygon GIoU hook.
- Optional F1-ish mode runs greedy 1:1 matching by IoU, then reports set-level counts (matched / missing / hallucination) and semantic-on-matched correctness (exact or embedding similarity).
- GPU is required (CUDA must be available) for the CLI and the training callback.

## CLI
```
python scripts/evaluate_detection.py \
  --pred_jsonl <path> \
  --out_dir eval_out \
  [--metrics coco|f1ish|both] \
  [--unknown-policy bucket|drop|semantic] \
  [--semantic-model <hf-id>] [--semantic-threshold 0.6] [--semantic-device auto] \
  [--f1ish-iou-thrs 0.3 0.5] \
  [--strict-parse] [--no-segm] [--iou-thrs 0.5 0.75] \
  [--overlay --overlay-k 12]
```
Artifacts (always): `metrics.json` (metrics + counters), `per_image.json`, optional `overlays/` when enabled.

Artifacts (when COCO is enabled via `--metrics coco|both`): `per_class.csv`, `coco_gt.json`, `coco_preds.json`.

Artifacts (when F1-ish is enabled via `--metrics f1ish|both`):
- `matches.jsonl` for the primary IoU threshold (0.5 if present in `--f1ish-iou-thrs`, else the max threshold).
- Optional `matches@<thr>.jsonl` for additional IoU thresholds (formatted with two decimals, e.g. `matches@0.30.jsonl`).
- `per_image.json` gains a stable `f1ish` field keyed by IoU threshold strings (e.g. `"0.50"`), containing per-image TP/FP/FN and semantic-on-matched counts.

## Staged workflow
1) Run unified inference to get `pred.jsonl` (pixel-space, with inline gt):
```
bash scripts/infer.sh --gt <data.jsonl> --ckpt <ckpt> --mode coord|text --out output/pred.jsonl
```
2) Evaluate (uses inline gt):
```
bash scripts/eval.sh --pred output/pred.jsonl --out_dir eval_out
```
3) Visualize (uses inline gt):
```
bash scripts/vis.sh --pred output/pred.jsonl --save_dir vis_out
```

## Training hook
Use `src.callbacks.DetectionEvalCallback` with pre-generated prediction JSONL:
```python
from src.callbacks import DetectionEvalCallback

callback = DetectionEvalCallback(
    gt_jsonl="path/to/gt.jsonl",
    pred_jsonl="path/to/preds.jsonl",  # generated separately
    out_dir="eval_out",
    mode="text",
)
trainer.add_callback(callback)
```
Metrics are logged under `eval_det_*` keys; callback skips if CUDA is unavailable.
