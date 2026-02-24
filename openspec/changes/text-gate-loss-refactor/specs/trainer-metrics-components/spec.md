# trainer-metrics-components Specification (Delta)

## ADDED Requirements

### Requirement: Loss metrics emit canonical registry-derived keys only
For registry-defined loss components, trainers MUST emit only canonical `loss/<component>` metric keys and MUST NOT emit trainer-specific loss aliases.

Normative behavior:
- Canonical keys include (minimum set):
  - `loss/token_ce`
  - `loss/struct_ce`
  - `loss/desc_ce`
  - `loss/geo`
  - `loss/coord_reg`
  - `loss/coord_gate`
  - `loss/text_gate`
- Trainer-specific aliases MUST be removed (non-exhaustive):
  - `loss/ce`
  - `loss/coord`
  - `loss/coord_prefix`
  - `loss/coord_tail`
  - `loss/token_ce_anchor`
  - `loss/token_ce_self_context`
  - `loss/struct_ce_self_context`
  - `loss/desc_ce_self_context`
  - `loss/<module>_obj`

#### Scenario: Canonical-only loss keys
- **WHEN** a Stage-2 or rollout-aligned training step logs loss metrics
- **THEN** emitted keys include only canonical `loss/<component>` keys for registry-defined components
- **AND** legacy alias keys are absent.

### Requirement: Rollout-only metrics are sparse-emitted
Trainers MUST NOT emit rollout-specific monitor metrics on steps where no rollout was executed.

Normative behavior:
- “Rollout executed” MUST be determined by runtime evidence (e.g., non-zero rollout generation time, non-zero parsed rollout length, or equivalent authoritative signal), not merely by decode configuration.
- When rollout was not executed, the trainer MUST omit (not emit with `0.0`) rollout-specific keys, including (non-exhaustive):
  - `rollout/precision`, `rollout/recall`, `rollout/f1`
  - `rollout/*` parse/gating/length/coverage diagnostics
  - `time/rollout_generate_s`, `time/rollout_parse_match_s`, `time/rollout_teacher_encode_s`

#### Scenario: A-only Stage-2 does not spam rollout zeros
- **WHEN** Stage-2 AB runs with `stage2_ab.schedule.b_ratio = 0.0`
- **AND** no rollout is executed for the current optimizer step
- **THEN** the emitted training log line contains no `rollout/*` scalar keys (they are absent rather than constant zeros).

### Requirement: Zero-valued timing keys are sparse-emitted
To reduce constant-noise monitors, trainers SHOULD omit timing keys that are identically `0.0` for the current run.

Normative behavior:
- `time/mask_build_s` MUST be omitted when it is not measured by the current trainer (`0.0` placeholder values are disallowed).

## REMOVED Requirements

### Requirement: Metric key schema remains backward compatible during refactor
**Reason**: This change intentionally breaks legacy loss metric aliases to eliminate drift and enforce a single canonical loss registry contract.

**Migration**: Update dashboards/scripts to consume canonical keys (`loss/struct_ce`, `loss/desc_ce`, `loss/geo`, `loss/coord_reg`, `loss/coord_gate`, `loss/text_gate`) and stop relying on trainer-specific alias keys.
