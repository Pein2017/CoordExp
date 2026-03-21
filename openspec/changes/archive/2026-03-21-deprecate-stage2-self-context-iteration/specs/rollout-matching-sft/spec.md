# rollout-matching-sft Specification (Delta)

## MODIFIED Requirements

### Requirement: Rollout pipeline specs are explicit and complete (no implicit defaults)
Rollout pipeline module specs MUST be authored with explicit fields and
complete module configs to prevent silent drift from default injection.

Normative behavior:
- Each entry in `rollout_matching.pipeline.objective[]` and
  `rollout_matching.pipeline.diagnostics[]` MUST include:
  - `name`, `enabled`, `weight`, `channels`, `application`, `config`.
- `channels` MUST be explicitly authored as a subset of `{A,B}`.
- `application` MUST be an explicitly authored mapping with exactly one key:
  - `preset`
- `application.preset` MUST be valid for the referenced module:
  - `token_ce`: `anchor_text_only`, `rollout_text_only`
  - `bbox_geo`, `bbox_size_aux`, `coord_reg`:
    - `anchor_only`
- Repo-owned rollout-aligned configs MUST reject self-context-era/final-pass
  preset names so the shared preset surface no longer advertises invalidated
  behavior:
  - `anchor_text_plus_final_struct`
  - `anchor_if_single_iter_else_final`
  - `final_only`
  - `anchor_and_final`
- `config` MUST include exactly the allowlisted keys for the referenced module:
  - missing required keys MUST fail fast (no implicit defaults)
  - unknown keys MUST fail fast (no escape-hatch aliases)

#### Scenario: Rollout-aligned final-pass preset fails fast
- **WHEN** a rollout-aligned config authors
  `rollout_matching.pipeline.objective[*].application.preset: final_only`
- **THEN** configuration parsing fails fast before trainer initialization
- **AND** the error explains that repo-wide self-context-era final-pass presets
  are deprecated and unsupported.
