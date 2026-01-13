## ADDED Requirements
### Requirement: Loss-scale weighting for coord/text CE
The system SHALL apply coord/text CE weights via ms-swift `loss_scale` when `coord_loss` is enabled and `coord_ce_weight` or `non_coord_ce_weight` differs from 1.0. The `loss_scale` tensor SHALL be built from labels as follows: 0.0 for `labels == -100`, `coord_ce_weight` for coord-token labels, and `non_coord_ce_weight` for all other supervised labels. The tensor length SHALL match `labels` for both padded and packed batches so that ms-swift’s shift (`roll(-1)`) aligns weights to target tokens.

#### Scenario: Weighted coord CE
- **GIVEN** `coord_loss.enabled` is true and `coord_ce_weight=2.0`, `non_coord_ce_weight=1.0`
- **WHEN** a batch contains both coord and text labels
- **THEN** `loss_scale` is attached with 2.0 at coord-label positions, 1.0 at non-coord label positions, and 0.0 at masked positions
- **AND** ms-swift applies the weights via the loss_scale path in CE computation.

#### Scenario: Default weights avoid loss_scale
- **GIVEN** `coord_ce_weight=1.0` and `non_coord_ce_weight=1.0`
- **WHEN** a batch is collated
- **THEN** `loss_scale` is omitted and the default CE loss path is used.

#### Scenario: Packed batch alignment
- **GIVEN** packing is enabled for training or eval
- **WHEN** `loss_scale` is attached
- **THEN** its length matches packed labels length and weights correspond to packed label positions.

### Requirement: Aux losses remain coord-only
Auxiliary coord losses (L1, GIoU, poly mask/smooth) SHALL continue to be computed only on coord-token positions, independent of `loss_scale` values.

#### Scenario: Aux loss excludes text
- **GIVEN** a batch with both coord and text labels
- **WHEN** auxiliary coord losses are computed
- **THEN** only coord-token positions contribute to aux losses and text tokens do not.
