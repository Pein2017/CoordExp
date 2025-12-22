## MODIFIED Requirements
### Requirement: GIoU for bbox_2d and mask-IoU for poly
The auxiliary loss SHALL compute GIoU for `bbox_2d` spans only. For `poly` spans, it SHALL compute a mask-IoU loss from differentiable soft polygon rasterization and SHALL NOT convert polygons to bboxes for loss.

#### Scenario: Poly uses mask IoU
- **GIVEN** a `poly` object with at least 3 points
- **WHEN** coord auxiliary loss runs
- **THEN** the loss uses mask-IoU and does not use bbox GIoU for that polygon.

## ADDED Requirements
### Requirement: Poly rasterization inputs
Polygon coordinates used for mask-IoU SHALL be interpreted in normalized [0,1] space, clamped to [0,1], and rasterized on a fixed grid using cell centers.

#### Scenario: Clamp normalized coords
- **GIVEN** decoded polygon coords outside [0,1]
- **WHEN** rasterization runs
- **THEN** coords are clamped before mask generation.

### Requirement: Poly mask configuration defaults
The coord loss config SHALL include polygon mask parameters with defaults:
- `poly_mask_size`: 64
- `poly_sigma_mask`: 1.5 / `poly_mask_size`
- `poly_tau_inside`: 0.08
- `poly_beta_dist`: 100
- `poly_smooth_weight`: 0.05

#### Scenario: Defaults apply
- **GIVEN** `custom.coord_loss.enabled` is true and no poly mask parameters are set
- **WHEN** aux loss runs for polygons
- **THEN** the defaults above are used.

### Requirement: Poly smoothness
The auxiliary loss SHALL add a closed-curve second-order smoothness term computed on predicted polygon vertices only and scaled by `poly_smooth_weight`.

#### Scenario: Smoothness uses predicted vertices only
- **GIVEN** a polygon prediction and GT polygon
- **WHEN** aux loss runs
- **THEN** smoothness is computed on the prediction only and added to the aux loss.

### Requirement: Poly mask loss weighting
Poly mask-IoU loss SHALL be scaled by `giou_weight`.

#### Scenario: Shared poly weight
- **GIVEN** `giou_weight = 1.0`
- **WHEN** poly mask-IoU loss is computed
- **THEN** the contribution is scaled by 1.0.

### Requirement: Poly mask metrics
The system SHALL log polygon metrics with keys:
- `coord_loss/poly_mask_iou` (IoU value; higher is better)
- `coord_loss/poly_smooth`

#### Scenario: Eval logging
- **GIVEN** evaluation with coord aux loss enabled
- **WHEN** poly spans are present
- **THEN** eval metrics include `eval_coord_loss/poly_mask_iou` and `eval_coord_loss/poly_smooth`.
