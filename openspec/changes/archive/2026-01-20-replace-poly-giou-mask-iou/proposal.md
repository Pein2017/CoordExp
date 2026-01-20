# Change: Replace poly bbox-GIoU with mask-IoU auxiliary loss

## Archive Note (2026-01-20)
This change targets the legacy `custom.coord_loss` (aux L1/GIoU/poly losses) path. As of 2026-01-14, Stage-1
switched to distributional coord supervision via `custom.coord_soft_ce_w1`, and `custom.coord_loss` is removed
from the runtime config surface (configs fail fast in `src/config/schema.py`).

Accordingly, this change is archived as legacy and does not update current specs.

## Why
Poly supervision currently collapses to bbox GIoU, which only provides gradients at extrema and does not shape the full polygon. We want a differentiable polygon loss that preserves the existing coord-token pipeline, teacher-forcing behavior, and packing compatibility while improving shape fidelity.

## What Changes
- Replace poly->bbox GIoU in coord auxiliary loss with soft mask-IoU computed from differentiable polygon rasterization.
- Add a closed-curve smoothness regularizer for polygon predictions only.
- Keep bbox_2d GIoU unchanged; line spans remain L1-only.
- Introduce poly mask hyperparameters in `custom.coord_loss` (mask size and soft-raster settings), reuse `giou_weight` as the poly loss weight, and log poly-specific metrics.
- Clamp decoded polygon coords to [0,1] before rasterization for stability.
- Update `configs/dlora/sft_coord_loss.yaml` to include poly mask defaults.

## Impact
- Affected specs: coord-aux-loss
- Affected code: `src/metrics/dataset_metrics.py`, new helper in `src/coord_tokens/`, `src/config/schema.py`, `configs/dlora/sft_coord_loss.yaml`
- Backward compatibility: bbox aux loss unchanged; poly aux loss switches from bbox GIoU to mask-IoU when enabled.
