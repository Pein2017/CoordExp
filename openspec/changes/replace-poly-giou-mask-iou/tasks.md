## 1. Implementation
- [x] 1.1 Add a soft polygon rasterizer helper (grid cache, winding soft-inside, softmin distance, mask sigmoid).
- [x] 1.2 Replace the poly->bbox GIoU branch in `CoordAuxLossMixin` with mask-IoU + smoothness; clamp coords to [0,1].
- [x] 1.3 Extend `CoordLossConfig` with poly mask parameters and defaults; reuse `giou_weight` as the poly mask-IoU weight.
- [x] 1.4 Add poly metrics logging (`coord_loss/poly_mask_iou`, `coord_loss/poly_smooth`) for train/eval.
- [x] 1.5 Update `configs/dlora/sft_coord_loss.yaml` with poly mask defaults.

## 2. Validation
- [ ] 2.1 Run a short debug config (e.g., `configs/dlora/debug-sft_coord_loss.yaml`) and confirm poly metrics appear in logs.
- [ ] 2.2 Verify no regressions in bbox GIoU metrics on a small eval run.
