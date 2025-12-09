# Detection Evaluator (CoordExp)

Minimal offline evaluator to compute COCO-style metrics from CoordExp JSONL.

## Inputs
- `pred_jsonl`: Pixel-space predictions produced by `scripts/run_infer.py` (unified engine); **must include inline `gt`** per line.
- Geometry objects live under `gt` / `pred` arrays with fields `{type, points, desc, score}`; legacy `predictions`/raw-text parsing is no longer used.
- Width/height are taken from the inline GT; size mismatches are counted but do not abort.

## Behavior
- Pixel-ready consumption: no coord-mode inference/denorm is performed; polygons are exported directly as COCO `segmentation` (mask IoU), bboxes as `bbox`. Lines are carried through in reports but excluded from metrics.
- One geometry per object; degenerate/invalid entries are dropped with counters and recorded per-image for diagnostics.
- Categories use exact desc strings; unknowns bucket to `unknown` by default (configurable to drop).
- Scores are fixed at 1.0 (greedy decoding outputs have no reliable confidence); any provided `score` fields are ignored.
- COCOeval runs for bbox and segm (when polygons exist). TODO: polygon GIoU hook.
- GPU is required (CUDA must be available) for the CLI and the training callback.

## CLI
```
python scripts/evaluate_detection.py \
  --pred_jsonl <path> \
  --out_dir eval_out \
  [--unknown-policy bucket|drop] [--strict-parse] [--no-segm] \
  [--iou-thrs 0.5 0.75] [--overlay --overlay-k 12]
```
Artifacts: `metrics.json` (metrics + counters), `per_class.csv`, `per_image.json`, `coco_gt.json`, `coco_preds.json`, optional `overlays/` when enabled.

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
