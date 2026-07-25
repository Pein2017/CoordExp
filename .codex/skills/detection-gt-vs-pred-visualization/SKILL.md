---
name: detection-gt-vs-pred-visualization
description: Render current CoordExp-Swift per-row GT/prediction images or two-run comparisons from scored inference artifacts.
---

# CoordExp-Swift Detection Visualization

Use the shared renderer:

- CLI: `scripts/visualize_detection.py`
- API: `src.vis.render_gt_vs_prediction` and
  `src.vis.render_prediction_comparison`
- Owners: `src/vis/api.py`, `normalization.py`, `matching.py`, and
  `rendering.py`

## Render

1. Bind an exact run directory containing matching `gt_vs_pred.jsonl` and
   `gt_vs_pred_scored.jsonl`, or the exact supported scored JSONL.
2. Select ordered row IDs or a bounded limit.
3. Run `gt-vs-pred` for one run or `compare` for two runs with identical ordered
   row IDs and GT.
4. Verify every selected row produced one PNG and that `manifest.json` records
   its source.

The renderer owns image resolution, GT norm1000-to-pixel conversion, prediction
pixel geometry, class-aware matching, duplicate hints, colors, and panel
semantics. Reject missing images, invalid dimensions or geometry, row mismatch,
or unsupported artifact names instead of adding renderer-local fallback logic.

Each sample gets its own image. A multi-sample collage is a presentation-only
exception that requires an explicit user request.

## Interface

```bash
python scripts/visualize_detection.py gt-vs-pred   --run-dir <run-dir> --out-dir <output-dir> --row-id <row-id>

python scripts/visualize_detection.py compare   --left-run-dir <left-run-dir> --right-run-dir <right-run-dir>   --left-label <left-label> --right-label <right-label>   --out-dir <output-dir> --limit <n>
```

Choose a fresh output directory when preservation matters; the renderer has no
collision guard.

## Report

Return the manifest, ordered PNG paths, selected row IDs, source run or runs,
and any contract failure.
