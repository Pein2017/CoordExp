## ADDED Requirements

### Requirement: Stage-2 Window-Aware Post-Rollout Packing (Training Only)
The stage_2 rollout-matching trainer (`custom.trainer_variant: rollout_matching_sft`) MUST support an optional
training-only mode that improves post-rollout packing utilization by accumulating segments across a full
gradient-accumulation window and making packing decisions with window-level visibility.

Config contract:
- The knob `custom.extra.rollout_matching.post_rollout_pack_scope` MUST accept `micro` or `window`.
- If unset, it MUST default to `micro` (current behavior).
- If set to any other value, the trainer MUST fail fast with actionable error text.

This change MUST preserve the existing gradient-accumulation semantics and training math:
- It MUST NOT collapse an accumulation window into fewer than `gradient_accumulation_steps` forward/backward calls.
- It MUST NOT change loss scaling/normalization relative to the existing micro-step contract.

#### Scenario: Default behavior preserved
- **GIVEN** `custom.extra.rollout_matching.post_rollout_pack_scope` is unset (or set to `micro`)
- **WHEN** stage_2 training runs
- **THEN** post-rollout packing and teacher-forced training MUST behave as they do today.

#### Scenario: Window-aware packing enabled (no GA collapse)
- **GIVEN** `custom.extra.rollout_matching.post_rollout_pack_scope: window`
- **WHEN** stage_2 training processes a full gradient-accumulation window
- **THEN** the trainer MUST accumulate per-micro post-rollout segments within the window
- **AND** compute packing selections with visibility over all segments in that window
- **AND** execute exactly `gradient_accumulation_steps` teacher-forced forward/backward micro-steps for that optimizer step.

#### Scenario: No cross-step carry and no silent dropping
- **GIVEN** `custom.extra.rollout_matching.post_rollout_pack_scope: window`
- **WHEN** an optimizer step boundary is crossed
- **THEN** no post-rollout segment from the prior optimizer step may be carried into the next optimizer step
- **AND** the trainer MUST NOT silently drop any segments produced for supervision in a full accumulation window
- **AND** if the window's segments cannot be scheduled into the window's micro-steps without dropping (given `packing_length`),
  the trainer MUST fail fast with actionable guidance.

#### Scenario: Deterministic window scheduling
- **GIVEN** `custom.extra.rollout_matching.post_rollout_pack_scope: window`
- **AND** identical post-rollout segments in identical insertion order with identical `encoded_len`
- **WHEN** window scheduling runs twice
- **THEN** it MUST produce the same per-micro packed selections in the same order both times.

#### Scenario: Infeasible window fails fast
- **GIVEN** `custom.extra.rollout_matching.post_rollout_pack_scope: window`
- **AND** the total `encoded_len` of all segments produced in a full accumulation window is greater than
  `gradient_accumulation_steps * packing_length`
- **WHEN** the trainer attempts to build the window schedule
- **THEN** it MUST fail fast with actionable guidance (e.g., increase `global_max_length`, reduce `max_new_tokens`, or disable packing).

#### Scenario: Evaluation and prediction are unaffected
- **GIVEN** stage_2 evaluation or prediction is running
- **WHEN** the stage_2 pipeline executes
- **THEN** teacher-forced training (including post-rollout packing) MUST NOT run
- **AND** the evaluation pipeline MUST remain unchanged
- **AND** rollout buffer reuse MUST be disabled (behave as `m_steps=1`) to keep metrics interpretable.
