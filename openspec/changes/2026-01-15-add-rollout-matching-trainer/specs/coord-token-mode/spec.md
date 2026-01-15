## MODIFIED Requirements

### Requirement: Coord-token augmentation compatibility
The system SHALL support data augmentation on records whose geometries are expressed as coord tokens (`<|coord_k|>`, 0–999), converting to numeric values before augmentation and restoring coord tokens after augmentation when coord-token mode is enabled. Public geometry fields (`bbox_2d`, `poly`) MUST remain token strings after augmentation.

#### Scenario: Coord tokens round-trip through identity augmentation
- **GIVEN** `custom.coord_tokens.enabled` is true
- **AND** an object geometry is provided as coord tokens
- **WHEN** augmentation runs with no-op/identity transforms
- **THEN** the geometry remains tokenized with identical token values after preprocessing/augmentation.

#### Scenario: Coord tokens with active affine augmentation
- **GIVEN** `custom.coord_tokens.enabled` is true
- **AND** an object geometry is provided as coord tokens
- **WHEN** an affine/geometry-changing augmentation runs
- **THEN** the system converts tokens to numeric values for the transform
- **AND** rounds transformed coordinates to the nearest integer (consistent with existing numeric path)
- **AND** converts the transformed integer values back to coord tokens in the output record, keeping geometry fields as tokens.

