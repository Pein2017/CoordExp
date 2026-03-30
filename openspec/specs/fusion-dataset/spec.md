# fusion-dataset Specification

## Purpose
Document that legacy fusion-config dataset authoring is temporarily disabled in
the canonical training surface while dormant examples/modules remain in-tree
for future reactivation.

## Requirements
### Requirement: custom.fusion_config Is Rejected
Fusion-config training is currently not an active authored path.
Training configs MUST NOT use `custom.fusion_config`.

#### Scenario: legacy fusion config is rejected
- **WHEN** a training config authors `custom.fusion_config`
- **THEN** config loading SHALL fail fast
- **AND** the error SHALL tell the operator that fusion-config training has
  been temporarily disabled in favor of the single-dataset hierarchy for now.
