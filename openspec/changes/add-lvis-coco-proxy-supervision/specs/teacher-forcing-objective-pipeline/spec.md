# teacher-forcing-objective-pipeline Specification (Delta)

## MODIFIED Requirements

### Requirement: Pipeline modules consume a teacher-forcing context contract
The teacher-forcing context SHALL expose derived object-local supervision views
when metadata-driven proxy supervision is enabled.

Normative behavior:

- the shared context / target-builder layer MUST be able to recover:
  - object-local mapping class labels
  - object-local supervision tier labels
  - object-local desc spans
  - object-local bbox groups
  - object-local desc weights
  - object-local coord weights
- objective modules MUST consume these derived views rather than re-parsing
  ad-hoc trainer-specific metadata independently,
- count mismatches between object metadata, desc spans, and bbox groups MUST
  fail fast.

#### Scenario: Metadata-driven proxy weighting uses shared derived views
- **WHEN** proxy-supervision metadata is present and object weighting is enabled
- **THEN** `token_ce`, `bbox_geo`, and `coord_reg` observe aligned object-local
  spans / groups from the shared context
- **AND** the runtime does not silently guess object alignment independently in
  each module
- **AND** the derived views remain sufficient to distinguish
  `same_extent_proxy` from `cue_only_proxy` at runtime.

### Requirement: Teacher-forcing objective is declared as an ordered YAML pipeline
The teacher-forcing pipeline SHALL support metadata-driven object weighting
through canonical module config keys.

Normative behavior:

- `token_ce.config` MAY include:
  - `object_weight_mode`
- `bbox_geo.config` MAY include:
  - `object_weight_mode`
- `coord_reg.config` MAY include:
  - `object_weight_mode`
- `object_weight_mode` MUST support:
  - `none`
  - `metadata`
- unknown object-weight modes MUST fail fast.

#### Scenario: Unknown object weight mode fails fast
- **WHEN** a pipeline module config sets `object_weight_mode` to an unsupported
  value
- **THEN** config validation fails fast with actionable guidance.
