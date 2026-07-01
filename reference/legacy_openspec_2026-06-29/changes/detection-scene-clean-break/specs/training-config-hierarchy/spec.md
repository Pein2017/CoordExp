## ADDED Requirements

### Requirement: Clean-break detection configs use current concept names

Canonical clean-break detection configs SHALL use public names that match the
DetectionScene architecture rather than historical mechanism names.

Normative behavior:

- new canonical Stage-1 detection teacher-forcing configs SHOULD live under
  `configs/stage1/detection_teacher_forcing/`;
- new canonical Stage-2 rollout-correction configs SHOULD live under
  `configs/stage2/rollout_correction/`;
- new public Stage-1 config/docs vocabulary SHOULD use
  `stage1_detection_teacher_forcing` rather than `recursive_detection_ce`;
- new public Stage-2 rollout/decode/eval policy SHOULD be owned by the
  Stage-2 rollout-correction surface rather than `rollout_matching.*`;
- historical config roots MAY remain only when classified as archive history,
  migration handles, rejection/absence tests, or temporary implementation
  details scheduled for rename/delete;
- strict typed config loading and deterministic resolution MUST remain retained
  invariants for canonical configs.

#### Scenario: Canonical config root exposes current concept names

- **WHEN** a new canonical Stage-1 or Stage-2 detection config is added after
  the clean-break migration
- **THEN** its path and public vocabulary use detection teacher-forcing or
  rollout-correction names
- **AND** it does not introduce `dense_caption`, `recursive_detection_ce`, or
  `rollout_matching` as target public concepts.
