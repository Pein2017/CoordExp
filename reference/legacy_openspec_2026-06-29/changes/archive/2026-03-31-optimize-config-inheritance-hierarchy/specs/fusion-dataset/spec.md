# fusion-dataset Specification (Delta)

## MODIFIED Requirements

### Requirement: Fusion Config Overrides Standard JSONL Paths
The supported training surface MUST reject authored `custom.fusion_config`
with guidance to migrate onto the single-dataset hierarchy.

#### Scenario: custom.fusion_config is rejected
- **WHEN** a training config authors `custom.fusion_config`
- **THEN** config loading fails fast
- **AND** the error tells the operator that fusion-config training has been
  temporarily disabled.

### Requirement: Fusion Config File Schema (Qwen3-VL Compatible Containers)
Canonical training runs MUST reject fusion-config file authoring even though
dormant examples remain in-tree.

#### Scenario: Fusion config file schema is no longer accepted
- **WHEN** a config points at a legacy fusion-config file
- **THEN** the run fails before dataset construction
- **AND** the error indicates that fusion-config authoring is currently
  disabled.

### Requirement: Dataset Entry Schema + Template Validation
Canonical training runs MUST NOT parse or validate fusion dataset entries while
fusion-config authoring is disabled.

#### Scenario: Fusion dataset entries are not parsed anymore
- **WHEN** a legacy fusion config contains dataset entries
- **THEN** the system does not attempt to validate or materialize them
- **AND** the run fails on the removed fusion surface itself.

### Requirement: Extends Merge Semantics
Canonical training config loading MUST NOT apply fusion-specific `extends`
merge semantics while the fusion surface is disabled.

#### Scenario: Fusion extends semantics are unavailable
- **WHEN** a legacy fusion config relies on fusion-specific `extends`
- **THEN** the run fails because fusion-config loading is removed
- **AND** no fusion merge semantics are applied.

### Requirement: Per-Dataset Ratio Quotas (No Target/Source Semantics)
Canonical training runs MUST NOT compute fusion-specific per-dataset ratio
quotas while authored fusion runs are disabled.

#### Scenario: Fusion ratio quotas are not computed
- **WHEN** a legacy fusion config sets per-dataset `ratio`
- **THEN** the run fails before any fusion quota computation occurs.

### Requirement: Eval Dataset Uses Any Non-Null val_jsonl
Canonical training runs MUST NOT assemble fusion eval datasets while authored
fusion runs are disabled.

#### Scenario: Fusion eval assembly is removed
- **WHEN** a legacy fusion config provides `val_jsonl` entries
- **THEN** the run fails before fusion eval dataset construction.

### Requirement: Dense-Caption Only (v1)
The system MUST fail fast rather than enter fusion dense-caption mode because
fusion is not a currently supported authored mode.

#### Scenario: Fusion dense-caption mode is unavailable
- **WHEN** a training run attempts to enable fusion training
- **THEN** the run fails fast instead of entering fusion dense-caption mode.

### Requirement: Encoded fusion samples include stable join metadata
Canonical training runs MUST NOT rely on or emit fusion-specific encoded-sample
join metadata while fusion runs are disabled.

#### Scenario: Fusion metadata contract is gone
- **WHEN** fusion training is attempted
- **THEN** the run fails before any fusion sample encoding path is entered.

### Requirement: Prompt injection is restored deterministically
Canonical training runs MUST NOT execute fusion-specific prompt injection or
restoration paths while the fusion dataset path is disabled.

#### Scenario: Fusion prompt injection path is retired
- **WHEN** a config attempts to use fusion dataset training
- **THEN** the run fails before any fusion prompt override path can execute.

### Requirement: Compatibility With Coord-Token Mode And Packing
The system MUST reject fusion authoring before coord-token or packing
compatibility logic would matter because fusion authoring is temporarily
disabled.

#### Scenario: Fusion compatibility path is removed
- **WHEN** a legacy config combines `custom.fusion_config` with CoordExp
  defaults such as coord tokens or packing
- **THEN** the run still fails on the disabled fusion surface
- **AND** the error does not imply that fusion remains supported.
