# teacher-forcing-objective-pipeline Specification (Delta)

## MODIFIED Requirements

### Requirement: Teacher-forcing objective is declared as an ordered YAML pipeline
The teacher-forcing objective SHALL be declared as an ordered YAML pipeline so
the enabled modules, their execution order, and their stable semantics are all
auditable from config alone.

Normative behavior:

- unknown or missing presets MUST fail fast with actionable diagnostics.
- `loss_dead_anchor_suppression` is a valid objective module name.
- `bbox_size_aux` is a valid objective module name.
- `loss_dead_anchor_suppression` MUST be declared with `channels: [B]`.
- `loss_dead_anchor_suppression.config` MUST be validated strictly and MUST be
  `{}` in v1.
- For canonical Stage-2 AB clean-prefix configs, the ordered objective list
  MUST place `loss_dead_anchor_suppression` after `token_ce` and before
  `bbox_geo`.
- When `bbox_size_aux` is enabled, canonical Stage-2 objective order MUST place
  it after `bbox_geo` because `bbox_size_aux` depends on the decoded
  canonicalized box state produced on the same matched supervision path.
- Canonical Stage-2 routing presets MUST be:
  - `token_ce.application.preset: anchor_text_only`
  - `loss_dead_anchor_suppression.application.preset: rollout_only`
  - `bbox_geo.application.preset: anchor_only`
  - `bbox_size_aux.application.preset: anchor_only`
  - `coord_reg.application.preset: anchor_only`
- Stage-2 AB configs MUST reject self-context-era presets that imply a final
  Channel-A pass, including `anchor_text_plus_final_struct`,
  `anchor_if_single_iter_else_final`, `final_only`, and `anchor_and_final`.
- `bbox_size_aux.config` MUST accept the canonical decoded-box size-aux keys:
  - `log_wh_weight`
  - `oversize_penalty_weight`

#### Scenario: Canonical Stage-2 objective presets no longer use final-pass routing
- **WHEN** a canonical Stage-2 AB objective pipeline is authored after this
  deprecation
- **THEN** Channel-A uses `anchor_text_only` and `anchor_only`
- **AND** the pipeline does not rely on self-context-era final-pass presets.

### Requirement: Pipeline identity is recorded for reproducibility
The trainer SHALL record a stable resolved objective-pipeline identity so two
runs with materially different objective semantics cannot share the same
manifest/checksum.

Normative behavior:

- The trainer MUST log the resolved ordered list of objective modules and
  diagnostics modules at initialization.
- The trainer MUST emit a stable pipeline checksum derived from the
  fully-resolved pipeline identity payload (`objective`, `diagnostics`, and
  semantics-only `extra`).

Normative checksum definition (this repo; required for implementers):

- The pipeline checksum MUST be the hex digest of `sha256` over UTF-8 bytes of
  a canonical JSON serialization of a fully-resolved pipeline identity object.
- The identity object MUST include, at minimum:
  - `objective`: ordered list of resolved module identity entries
  - `diagnostics`: ordered list of resolved module identity entries
  - `extra`: mapping (default `{}`) for trainer-specific identity fields that
    affect objective/metrics semantics. If `extra` is used, it MUST be included
    in the checksum input.
- The `extra` mapping MUST use stable, fully-qualified key names (avoid
  ambiguous short keys).
  For this repo, implementers MUST use the following reserved keys when
  applicable:
  - `variant` (string; trainer variant name, for example `stage2_two_channel`)
- `stage2_ab.coord_ctx_embed_mode` MUST NOT appear in active pipeline identity
  payloads because it is part of the deprecated self-context loop surface.
- `stage2_ab.coord_decode_mode` and `rollout_matching.coord_decode_mode` MUST
  NOT appear in active pipeline identity payloads because those authored decode
  toggles are deprecated and unsupported.
- Each resolved module identity entry MUST be normalized before checksum:
  - `name: str`
  - `enabled: bool`
  - `weight: float`
  - `channels: list[str]` (if omitted in config, normalize to `["A","B"]`; if
    provided, normalize ordering as `["A","B"]` filtered to those present)
  - `config: mapping` (module-resolved config with defaults applied; unknown
    keys are already rejected)
- Canonical JSON requirements:
  - object keys MUST be serialized with lexicographic key ordering
    (`sort_keys=true`)
  - list ordering MUST be preserved (execution order is semantically
    meaningful)
  - serialization MUST be whitespace-free (`separators=(",", ":")`)
  - floats MUST be finite (fail fast on NaN/Inf) and MUST normalize `-0.0` to
    `0.0` before serialization

#### Scenario: Deprecated self-context identity fields are absent from the checksum payload
- **WHEN** a Stage-2 two-channel run initializes its pipeline identity payload
- **THEN** `stage2_ab.coord_ctx_embed_mode`,
  `stage2_ab.coord_decode_mode`, and
  `rollout_matching.coord_decode_mode` do not appear in the active checksum
  payload.
