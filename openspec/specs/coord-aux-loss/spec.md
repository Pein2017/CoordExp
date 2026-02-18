# coord-aux-loss Specification

## Purpose
Define the canonical auxiliary geometry-loss contract for coord-token training, including configuration knobs, per-geometry loss behavior, and logging expectations.

## Requirements


### Requirement: Configurable coord auxiliary loss
The system SHALL expose a long-term config block at `custom.coord_loss` for coord auxiliary loss that includes:
- enable flag (default: false)
- L1 weight (default: 0.0)
- GIoU weight (default: 0.0)
- coord CE weight (default: 1.0)
- non-coord CE weight (default: 1.0)
- top-k expectation parameters (top_k default: 0.1; temperature default: 1.0)

#### Scenario: Aux loss disabled by default
- **GIVEN** no coord auxiliary loss config is provided
- **WHEN** training runs
- **THEN** no auxiliary loss is computed and behavior matches current SFT.

#### Scenario: CE weighting disabled when aux loss is off
- **GIVEN** `custom.coord_loss.enabled` is false
- **WHEN** training runs
- **THEN** coord vs non-coord CE weights are not applied and base CE remains unchanged.


### Requirement: Top-k expectation decoding on coord tokens
Expectation decoding SHALL operate over the ordered coord-token id list (`<|coord_0|>`..`<|coord_999|>`) and support top-k selection where `top_k` may be a fraction (0 < top_k < 1) or an integer count. Fractional values SHALL be converted via `ceil(top_k * 1000)` and clamped to [1, 1000]. Top-k selection SHALL use the highest logits; tie order is implementation-defined.

#### Scenario: Fractional top_k
- **GIVEN** `top_k = 0.1`
- **WHEN** expectation decoding runs
- **THEN** the top 10% of coord-token logits are used to compute E(top_k) over bins 0..999.


### Requirement: L1 per coord token in normalized space
The auxiliary L1 loss SHALL be computed per coord token using normalized targets in [0,1].

#### Scenario: L1 uses coord tokens
- **GIVEN** coord tokens appear in the assistant sequence
- **WHEN** aux loss is enabled
- **THEN** L1 is computed per coord token using the expected coord value and the target coord token.


### Requirement: GIoU for bbox_2d and poly via bbox
This requirement MUST be interpreted as a legacy coord-loss formulation used by earlier experiments and retained for audit traceability.

Legacy behavior summary:
- `bbox_2d` used GIoU directly.
- `poly` was converted to axis-aligned bbox via min/max before GIoU.
- `line` was excluded from GIoU.

Current canonical training behavior is defined by `Requirement: GIoU for bbox_2d and mask-IoU for poly` below and MUST take precedence whenever both contracts are visible in this file.

#### Scenario: Poly GIoU-style loss
- **GIVEN** historical runs/logs produced before mask-IoU migration
- **WHEN** those artifacts are interpreted
- **THEN** poly loss entries can reflect bbox-converted GIoU semantics
- **AND** current training configs MUST follow the mask-IoU requirement below.


### Requirement: Packed and non-packed batch support
Aux loss computation SHALL work in both packed and non-packed batches by carrying per-object coord spans through the collator and aligning them with coord-token positions in labels.

#### Scenario: Packed coord spans
- **GIVEN** packing is enabled and multiple samples are packed
- **WHEN** the collator builds a batch
- **THEN** coord spans are concatenated in pack order and the aux loss is computed correctly.


### Requirement: Train/eval logging parity
When enabled, aux loss components SHALL be logged for both train and eval with matching key names. Keys SHALL include:
- `coord_loss/total`
- `coord_loss/l1`
- `coord_loss/giou`
- `coord_loss/coord_ce`
Eval logs SHALL use the framework prefix `eval_` (e.g., `eval_coord_loss/total`).

#### Scenario: Eval logging
- **GIVEN** evaluation runs with coord aux loss enabled
- **WHEN** eval_step executes
- **THEN** L1/GIoU/coord CE components are logged under eval metrics.


### Requirement: Coord vs non-coord CE weighting
The system SHALL allow separate CE weights for coord tokens vs non-coord tokens via loss scaling when `custom.coord_loss.enabled` is true.

#### Scenario: Coord-weighted CE
- **GIVEN** `custom.coord_loss.enabled` is true
- **AND** coord_weight != non_coord_weight
- **WHEN** training computes token CE
- **THEN** coord tokens use coord_weight and non-coord tokens use non_coord_weight.


### Requirement: GIoU for bbox_2d and mask-IoU for poly
The auxiliary loss SHALL compute GIoU for `bbox_2d` spans only. For `poly` spans, it SHALL compute a mask-IoU loss from differentiable soft polygon rasterization and SHALL NOT convert polygons to bboxes for loss.

#### Scenario: Poly uses mask IoU
- **GIVEN** a `poly` object with at least 3 points
- **WHEN** coord auxiliary loss runs
- **THEN** the loss uses mask-IoU and does not use bbox GIoU for that polygon.


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
