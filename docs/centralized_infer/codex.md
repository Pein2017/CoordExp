# Centralized Inference Refactor (CoordExp)

Date: 2025-12-08

## What changed (implementation snapshot)
- Added unified engine at `src/infer/engine.py` with explicit `mode` (`coord`/`text`), shared scaling/validation, and deterministic generation (seeded `torch.Generator`).
- Rebuilt CLI `scripts/run_infer.py` to require `--gt_jsonl`, `--model_checkpoint`, `--mode`, and generation flags only; emits `pred.jsonl` + `pred.summary.json`. Bash wrapper: `scripts/infer.sh`.
- Visualization now reads only `pred.jsonl` (with inline gt); no external GT inputs. Bash wrapper: `scripts/vis.sh`.
- Detection evaluator consumes `pred.jsonl` with inline gt only; polygons via COCO `segmentation`, lines ignored for metrics. Bash wrapper: `scripts/eval.sh`.
- Eval bash wrapper added: `scripts/eval.sh` (wraps `scripts/evaluate_detection.py`).

## Unified output schema (per-line `pred.jsonl`)
```
{
  "index": int,
  "image": "path/to/img.jpg",
  "width": W,
  "height": H,
  "mode": "coord" | "text",
  "coord_mode": "norm1000" | "pixel",   # trace only
  "gt":   [{"type","points","points_text","desc","score":1.0,"_coord_mode": "..."}],
  "pred": [{"type","points","points_text","desc","score":1.0,"_coord_mode": "..."}],
  "raw_output": "<model text>",
  "errors": ["..."]
}
```
- All `points` are absolute pixels; polygons preserved as single rings; lines carried through but not scored.
- `points_text` is a pixel-space text rendering of `points` to make downstream
  text-only consumers consistent; `_coord_mode` records the detected source
  space before scaling (norm1000 or pixel).

## Mode handling
- **coord mode**: GT and preds expected in norm1000; enforce 0–999; denorm both using width/height; mode/GT mismatch -> error and skip generation.
- **text mode**: GT pixels; preds denorm if tokens or norm-like; pixel preds accepted via `coords_are_pixel` heuristic; mode/GT mismatch flagged.

## Error handling & summary
- Per-sample `errors` list; critical issues (size missing, multi-image, image load failure, mode mismatch) skip generation but still emit a line.
- Canonical counters (invalid_json/geometry/coord/size_mismatch/empty_pred/mode_gt_mismatch/…) aggregated into `pred.summary.json` alongside totals and distinct error codes.

## Evaluator alignment
- Evaluator consumes `gt`/`pred` objects directly (pixel space); polygons exported as COCO `segmentation` for mask IoU; `line` objects excluded from metrics but retained in reports.

## Visualizer alignment
- Reads pixel-space `pred.jsonl` (or runs inference via engine) and renders bbox/poly/line overlays to `save_dir`; no legacy norm/pixel dual paths.

## Notes / gaps
- Requested `docs/centralized_infer/aug-gpt.md` was not found in the repo; continue without it.
- Legacy fields (`predictions`, raw text parsing in evaluator) are removed; older outputs must be regenerated or adapted externally.
