## MODIFIED Requirements

### Requirement: Coord-token mode is explicit rather than globally mandatory
- The system SHALL support `custom.coord_tokens.enabled=true` as the explicit
  coord-token expression path.
- The system SHALL support `custom.coord_tokens.enabled=false` as a separate
  norm1000 raw-text expression path for Stage-1.
- `custom.coord_tokens.skip_bbox_norm: true` remains mandatory when
  coord-token mode is enabled.
- `custom.coord_tokens.skip_bbox_norm` SHALL NOT be used to reject the
  raw-text norm1000 path when coord tokens are disabled.

#### Scenario: Coord-token mode still requires skip-bbox-norm
- **GIVEN** a training config with `custom.coord_tokens.enabled=true`
- **AND** `custom.coord_tokens.skip_bbox_norm=false`
- **WHEN** config loading runs
- **THEN** loading fails with a clear error indicating that coord-token mode
  requires `skip_bbox_norm=true`.

#### Scenario: Raw-text norm1000 mode does not fail on disabled coord tokens
- **GIVEN** a training config with `custom.coord_tokens.enabled=false`
- **WHEN** config loading runs
- **THEN** loading succeeds
- **AND** the system resolves a non-coord-token geometry-expression mode.
