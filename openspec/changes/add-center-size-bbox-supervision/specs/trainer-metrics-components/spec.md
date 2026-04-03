## ADDED Requirements

### Requirement: BBox geometry metric keys remain stable across internal bbox loss-space modes
The training metrics contract SHALL keep existing bbox geometry metric key names stable when bbox regression changes internal loss-space decomposition.

Normative behavior:
- Existing bbox geometry metric keys such as `loss/geo/bbox_smoothl1`, `loss/geo/bbox_ciou`, `loss/coord/bbox_smoothl1`, `loss/coord/bbox_ciou`, and `loss/B_coord/bbox_smoothl1`, `loss/B_coord/bbox_ciou` MUST remain stable.
- `bbox_ciou` metrics MUST continue to reflect CIoU computed on canonical decoded `xyxy` boxes.
- `bbox_smoothl1` metrics MAY reflect either canonical `xyxy` regression or internal `center_size` loss-space regression, depending on the authored bbox geometry parameterization in the resolved config.
- Runs using non-default bbox geometry loss-space MUST remain distinguishable through `resolved_config.json` rather than new metric key families.
- Comparisons of `bbox_smoothl1` across runs with different bbox geometry loss-space modes MUST be treated as invalid unless the comparison explicitly joins against the resolved config parameterization.

#### Scenario: Center-wise loss-space mode preserves canonical metric keys
- **WHEN** bbox geometry supervision runs with `parameterization: center_size`
- **THEN** training logs still use the canonical `bbox_smoothl1` and `bbox_ciou` metric key families
- **AND** the resolved config remains the authoritative source for interpreting which regression parameterization produced the run.
