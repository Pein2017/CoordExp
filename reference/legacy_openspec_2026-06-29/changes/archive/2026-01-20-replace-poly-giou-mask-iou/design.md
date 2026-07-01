# Design: Poly soft mask IoU auxiliary loss

## Goals
- Provide polygon-shape supervision with gradients across all vertices/edges.
- Preserve the existing coord-token decode, coord_spans alignment, and packing pipeline.
- Keep bbox_2d auxiliary GIoU unchanged; apply new loss only for `geom_type == "poly"`.

## Non-goals
- No new detection heads or model architecture changes.
- No enclosure GIoU-style penalty in v1.
- No changes to line spans beyond existing L1 loss.

## Data Flow Integration
1. Reuse `topk_expectation_decode` to obtain normalized coords in [0,1).
2. Use coord_spans to slice per-object polygon coords.
3. For poly spans, reshape to `V_pred`/`V_gt` of shape [N, 2], clamp to [0,1].
4. Rasterize each polygon into a soft mask `M_pred`, `M_gt` at resolution 64x64.
5. Compute soft IoU loss and add smoothness regularizer on predicted vertices.
6. Scale poly loss by `giou_weight` and smoothness by `poly_smooth_weight`.

## Soft Rasterization (Differentiable Polygon Mask)
- Grid: H=W=64, cell centers at `(i+0.5)/W`, `(j+0.5)/H`.
- Cache the grid per `(device, dtype, mask_size)` inside the mixin or helper to avoid per-step rebuilds.

### Soft inside probability (winding)
For each grid point `g`, compute angle sum over polygon edges:
- `angle_i = atan2(cross(V_i-g, V_{i+1}-g), dot(V_i-g, V_{i+1}-g) + eps)`
- `w = sum(angle_i) / (2*pi)`
- `q = sigmoid((abs(w) - 0.5) / tau_inside)`

### Soft distance to boundary
- Compute point-to-segment distances `dist_i` for each edge.
- Use softmin: `d = -logsumexp(-beta_dist * dist_i) / beta_dist`.

### Soft signed distance and mask
- `s = (2*q - 1) * d`
- `M = sigmoid(s / sigma_mask)`

This yields values in [0,1] with smooth gradients near boundaries.

## Losses
- Soft IoU: `IoU = sum(M_pred * M_gt) / (sum(M_pred + M_gt - M_pred*M_gt) + eps)`
- Poly loss: `L_poly = 1 - IoU`
- Smoothness (closed curve): `L_smooth = sum(||v_{i+1} - 2*v_i + v_{i-1}||^2)` with wraparound.
- Total poly aux: `giou_weight * L_poly + poly_smooth_weight * L_smooth`

## Defaults
- `poly_mask_size = 64`
- `poly_sigma_mask = 1.5 / poly_mask_size`
- `poly_tau_inside = 0.08`
- `poly_beta_dist = 100`
- `poly_smooth_weight = 0.05`

## Logging
- `coord_loss/poly_mask_iou` logs the IoU value (higher is better).
- `coord_loss/poly_smooth` logs the smoothness term.
- Eval uses `eval_` prefix via existing metric wrapper.

## Compatibility Notes
- Coordinates are normalized from coord tokens by `/999`; clamping ensures valid [0,1] input.
- Packing is unaffected: spans remain aligned with coord positions in labels.
- Bbox GIoU and line L1 paths are unchanged.
