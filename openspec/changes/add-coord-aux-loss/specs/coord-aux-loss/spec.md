## ADDED Requirements
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
The auxiliary GIoU loss SHALL be computed for `bbox_2d` objects and for `poly` objects by converting their points to an axis-aligned bbox using min/max. `line` objects SHALL be excluded from GIoU.

#### Scenario: Poly GIoU-style loss
- **GIVEN** a poly object with >=3 points
- **WHEN** aux loss is enabled
- **THEN** the system converts the polygon to a bbox and computes GIoU on that bbox.

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
