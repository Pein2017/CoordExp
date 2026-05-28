## ADDED Requirements

### Requirement: Trainer variant resolution fails fast for unknown active variants

Training runtime resolution SHALL reject unknown non-empty
`custom.trainer_variant` values unless an explicit extension contract declares
that arbitrary variants are supported.

Normative behavior:

- removed variants MUST continue to fail with replacement guidance;
- typo-like unknown variants MUST NOT silently fall back to the generic Stage-1
  runtime plan;
- if extension variants are needed, they MUST have an explicit registration or
  documented extension mechanism and tests proving they cannot accidentally
  activate Stage-2 semantics.

#### Scenario: Typo in Stage-2 variant fails before trainer construction

- **GIVEN** a config with `custom.trainer_variant: stage2_rollout_correcton`
- **WHEN** runtime planning resolves the variant
- **THEN** loading fails before trainer construction
- **AND** the error identifies the unknown trainer variant.

### Requirement: Stage-2 runtime projection records authored and resolved policy sources

Stage-2 config loading and bootstrap SHALL expose a resolved runtime projection
that records which authored namespace supplied each effective runtime policy.

Normative behavior:

- `stage2_rollout_correction.*` remains the objective/correction namespace;
- `rollout_matching.*` remains the rollout runtime/backend/decode/eval
  namespace;
- compatibility fallback from legacy or custom fields MAY remain during
  migration only if the resolved projection records the fallback source;
- run manifests, effective runtime artifacts, and policy provenance MUST be
  able to distinguish authored keys from compatibility-derived effective
  values.

The projection MAY be named `Stage2RuntimeProjection` or equivalent, but it
MUST be a named contract resolved before trainer construction.

The named projection MUST include, at minimum:

- authored source and fallback source for each effective policy;
- effective train/eval prompt policy;
- effective decode and backend policy for train rollout and eval rollout;
- eval materialization policy;
- post-rollout packing owner and packing settings;
- geometry, bbox format, detection sequence format, object ordering, and object
  field-order policy;
- Stage-2 pipeline manifest payloads and policy provenance payloads.

#### Scenario: Prompt variant fallback is visible in provenance

- **GIVEN** a compatibility path derives an effective rollout or eval prompt
  variant from a legacy custom field
- **WHEN** the runtime projection and run provenance are written
- **THEN** the authored source and effective resolved value are both visible
- **AND** the fallback is not indistinguishable from a direct
  `rollout_matching.*` authoring choice.
