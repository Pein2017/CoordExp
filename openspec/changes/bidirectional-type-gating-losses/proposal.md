## Why

The active hard-SFT detection stack needs a stable, config-first contract for
bidirectional token-family pressure before implementation touches loss math.
The current research plan promotes only the type-gating term; ledger,
continuation, geometry, smoke, and salvage behavior remain experiment-only.

## What Changes

- Define `objective.terms.token_type_mass` as the stable config term for the
  active hard-SFT token-family objective.
- Specify that `enabled: true` on the promoted hard-SFT stack means an additive,
  bidirectional, exclusive four-family objective over `schema`, `coord`,
  `desc`, and `stop`.
- Forbid a stable `mode` key on `objective.terms.token_type_mass`; the
  four-family exclusive behavior is the contract.
- Specify hard-SFT targeting through the one-hot `selected_token_role` family.
- Define metric semantics for `teacher_forcing/loss/token_type_mass` and any
  emitted contribution accounting.
- Leave valid-set likelihood, within-valid coverage, coverage ledger,
  continuation, `bbox_positive_area`, train/smoke gates, and diagnostic salvage
  outside this stable change.

## Capabilities

### New Capabilities

- None.

### Modified Capabilities

- `stage1-detection-objectives`: Adds the config contract for
  `objective.terms.token_type_mass` on the promoted hard-SFT detection stack.
- `teacher-forcing-unified-loss-registry`: Defines the stable four-family
  token-type mass objective and hard-SFT selected-role target semantics.
- `trainer-metrics-components`: Defines token-type mass loss and contribution
  metric semantics under the `teacher_forcing/...` namespace.

## Impact

- Affected config surface: `objective.terms.token_type_mass.enabled` and
  `objective.terms.token_type_mass.weight`.
- Affected objective semantics: hard-SFT teacher-forcing loss aggregation gains
  an additive family-partition term when the promoted stack enables it.
- Affected metrics: `teacher_forcing/loss/token_type_mass` and optional
  contribution accounting for that term.
- No new CLI flags, dependencies, services, production training launch, or
  stable contract for ledger/continuation/geometry/salvage behavior.
