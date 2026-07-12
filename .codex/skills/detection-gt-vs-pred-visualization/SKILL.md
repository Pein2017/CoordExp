---
name: detection-gt-vs-pred-visualization
description: Use for current CoordExp-Swift per-row GT/pred PNGs or two-run comparisons via scripts/visualize_detection.py or src.vis, including row selection, matching, duplicate hints, manifests, and renderer failures.
---

# CoordExp-Swift Detection Visualization

Use the shared renderer. Do not build a bespoke PIL, matplotlib, normalization,
or matching path when `src/vis/` supports the request.

## Current route

- CLI: `scripts/visualize_detection.py`
- Public API: `src.vis.render_gt_vs_prediction` and
  `src.vis.render_prediction_comparison`
- Implementation: `src/vis/api.py`, `normalization.py`, `matching.py`, and
  `rendering.py`
- Input: a run directory containing both `gt_vs_pred.jsonl` and
  `gt_vs_pred_scored.jsonl`, or the exact scored JSONL path

Legacy `src/infer/vis.py`, `src/vis/gt_vs_pred.py`,
`src/vis/comparison.py`, evaluator overlays, guarded/proxy artifacts, and
`vis_resources/` sidecars are not the current Swift renderer.

## Render

Render GT versus predictions:

```bash
conda run -n ms python scripts/visualize_detection.py gt-vs-pred \
  --run-dir <run-dir> \
  --out-dir <output-dir> \
  --limit <n>
```

Select exact rows by repeating `--row-id <row-id>`. Compare two runs with
identical ordered row IDs and GT:

```bash
conda run -n ms python scripts/visualize_detection.py compare \
  --left-run-dir <left-run-dir> \
  --right-run-dir <right-run-dir> \
  --left-label <left-label> \
  --right-label <right-label> \
  --out-dir <output-dir> \
  --limit <n>
```

Both commands accept `--duplicate-iou-threshold`; the default is `0.30`.
Matching uses class agreement and IoU `0.50`.

Programmatic equivalents:

```python
from src.vis import render_gt_vs_prediction, render_prediction_comparison

render_gt_vs_prediction(run_dir, out_dir, row_ids=[row_id], limit=1)
render_prediction_comparison(left_run, right_run, out_dir, limit=20)
```

## Contract and output

The loader requires raw/scored row-count and row-ID parity. It uses raw GT and
scored predictions, resolves the image path, converts GT norm1000 coordinate
bins to pixel `xyxy`, and draws prediction `bbox` as already-pixel `xyxy`.
Reject missing images, invalid dimensions or geometry, mismatched GT, and
unsupported JSONL names instead of adding renderer-local fallback logic.

Each selected row produces a separate PNG. The output directory also receives
`manifest.json` and `README.md`. Do not create contact sheets or multi-sample
collages unless the user explicitly asks. Choose a fresh output directory when
preservation matters because the renderer has no collision guard.

GT-versus-prediction layout and colors:

- left panel: GT; right panel: predictions;
- green: matched GT or prediction;
- yellow: missing GT / false negative;
- red: unmatched prediction / false positive;
- purple dashed: possible duplicate prediction.

Prediction comparison hides matched GT, shows each run's predictions, and uses
the same green/yellow/red/purple meanings. Preserve these semantics unless the
user explicitly requests a presentation-only variant.

## Reporting

Return the manifest path, ordered PNG paths, selected row IDs, source run(s),
and any contract failure. Keep one image per sample by default so review remains
legible and artifact identity stays obvious.
