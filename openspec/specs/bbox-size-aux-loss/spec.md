# bbox-size-aux-loss Specification

## Purpose
Define the canonical decoded-box size auxiliary contract shared across Stage-1,
Stage-2 AB, and rollout-aligned teacher-forcing paths.
## Requirements
### Requirement: Decoded-box size auxiliary is plugin-first and aligned with the current coord-token + `bbox_2d` contract
The system SHALL support an optional bbox size auxiliary computed from decoded
continuous boxes without changing tokenizer format, coord vocabulary, sequence
format, or decode protocol.

Normative behavior:

- the canonical bbox expression remains `bbox_2d: [x1, y1, x2, y2]` where
  `(x1, y1)` is top-left and `(x2, y2)` is bottom-right after canonicalization,
- coord-token alignment remains unchanged:
  - coords are still expressed through the current `<|coord_k|>` tokens where
    `k in [0, 999]`,
  - the size auxiliary MUST consume decoded boxes derived from the current
    coord-token path rather than introducing a second coordinate encoding,
- the canonical implementation MUST be reusable as a plugin-style loss module
  (`bbox_size_aux`) for pipeline-driven trainers,
- Stage-1 MAY invoke that same implementation through a thin adapter, but MUST
  NOT create a second divergent bbox-size loss definition,
- the helper MUST accept predicted boxes in `xyxy`,
- the helper MUST canonicalize boxes before measuring size:
  - `x_lo = min(x1, x2)`
  - `x_hi = max(x1, x2)`
  - `y_lo = min(y1, y2)`
  - `y_hi = max(y1, y2)`
- width and height MUST be computed as:
  - `w = max(x_hi - x_lo, eps)`
  - `h = max(y_hi - y_lo, eps)`
- the helper MUST support:
  - matched log-width/log-height loss,
  - matched log-area loss,
  - thresholded oversize penalty,
- the helper MUST return numerically stable scalars and MUST treat empty masked
  inputs as zero contribution rather than crashing.

#### Scenario: Canonicalized exact match yields near-zero matched size loss
- **GIVEN** predicted and target boxes that become identical after
  canonicalization
- **WHEN** matched log-width/log-height loss is computed
- **THEN** the returned scalar is near zero
- **AND** the computation does not depend on the original corner ordering.

### Requirement: Oversize penalty is opt-in and must not create a default small-box prior
The system SHALL treat oversize regularization as a weak optional safeguard,
never as the default geometry objective.

Normative behavior:

- oversize regularization MUST be disabled by default,
- when enabled it MUST penalize only above an explicit threshold:
  - log-width threshold, or
  - log-height threshold, or
  - area-fraction threshold,
- the default behavior MUST NOT encourage all boxes to become smaller in the
  absence of matched supervision.

#### Scenario: Box below threshold has zero oversize penalty
- **GIVEN** oversize regularization is enabled with an explicit threshold
- **WHEN** a predicted box remains below the threshold
- **THEN** the oversize penalty is exactly zero for that box.

### Requirement: Stage-1 config surface is nested and explicit
Stage-1 SHALL expose bbox size auxiliary supervision through a dedicated nested
config block under `custom.bbox_size_aux`.

Normative keys:

- `enabled`
- `log_wh_weight`
- `log_area_weight`
- `oversize_penalty_weight`
- `oversize_area_frac_threshold`
- `oversize_log_w_threshold`
- `oversize_log_h_threshold`
- `eps`

Validation behavior:

- all weights MUST be `>= 0`,
- `eps` MUST be `> 0`,
- if `enabled=true`, at least one of:
  - `log_wh_weight`
  - `log_area_weight`
  - `oversize_penalty_weight`
  MUST be non-zero.

#### Scenario: Enabled Stage-1 size aux with all-zero weights fails fast
- **GIVEN** `custom.bbox_size_aux.enabled: true`
- **AND** every Stage-1 size-aux weight is `0`
- **WHEN** config validation runs
- **THEN** validation fails fast with actionable guidance.

### Requirement: Stage-1 SHALL execute bbox size aux through a plugin host
When Stage-1 bbox size auxiliary supervision is enabled, the runtime SHALL
execute it through a reusable plugin/module host rather than bespoke
feature-local loss logic.

Normative behavior:

- the Stage-1 path MAY keep a nested `custom.bbox_size_aux` config surface in
  v1,
- but runtime execution MUST route through a reusable Stage-1 aux plugin host,
- the Stage-1 host MUST reuse the canonical `bbox_size_aux` plugin
  implementation and MUST NOT define a second divergent bbox-size loss
  implementation.

#### Scenario: Stage-1 bbox size aux is hosted as a plugin module
- **GIVEN** Stage-1 bbox size auxiliary is enabled
- **WHEN** the Stage-1 trainer computes the auxiliary loss
- **THEN** execution passes through a reusable Stage-1 aux plugin host
- **AND** that host reuses the canonical `bbox_size_aux` plugin
  implementation.

### Requirement: Stage-1 size aux requires explicit bbox grouping semantics
When Stage-1 bbox size auxiliary is enabled, the runtime SHALL supervise only
true bbox groups and MUST NOT infer bbox groups from raw coord-token counts when
that would be ambiguous.

Normative behavior:

- v1 Stage-1 support is bbox-only for this plugin path,
- the Stage-1 adapter MAY group bbox coord tokens in chunks of four only for
  bbox-only geometry,
- if the active Stage-1 dataset/batch includes non-bbox geometry while the
  plugin is enabled, the runtime MUST fail fast,
- missing required bbox grouping metadata when the feature is enabled MUST fail
  fast with actionable diagnostics.
- the Stage-1 adapter MUST reuse the canonical `bbox_size_aux` plugin
  implementation once bbox groups are resolved.

#### Scenario: Ambiguous Stage-1 geometry grouping fails fast
- **GIVEN** Stage-1 bbox size aux is enabled
- **AND** the runtime encounters non-bbox geometry in the batch
- **WHEN** the auxiliary loss is about to run
- **THEN** training raises with actionable diagnostics
- **AND** does not silently apply the loss to ambiguous coord spans.
