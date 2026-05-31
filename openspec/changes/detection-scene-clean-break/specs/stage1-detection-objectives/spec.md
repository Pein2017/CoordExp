## ADDED Requirements

### Requirement: Stage-1 detection teacher forcing uses scene projection vocabulary

The clean-break Stage-1 detection surface SHALL use scene projection vocabulary
instead of public recursive-detection historical naming.

Normative behavior:

- the target public Stage-1 concept is `stage1_detection_teacher_forcing`;
- Stage-1 target construction MUST project `DetectionScene` through
  `RenderedDetectionSequence` into `DetectionSupervisionView`;
- template semantics MUST be owned by `DetectionSequenceTemplate`;
- token labels, masks, coordinate spans, and sidecars MUST be owned by
  `DetectionSupervisionView` construction;
- new public config/docs names MUST NOT use `recursive_detection_ce` as the
  target architecture name;
- legacy recursive-detection names MAY remain only as archive history,
  migration handles, or rejection/absence tests until deleted.

#### Scenario: Stage-1 supervision is derived from DetectionScene

- **GIVEN** a Stage-1 detection teacher-forcing training example
- **WHEN** supervision is constructed
- **THEN** `DetectionScene` is rendered through a `DetectionSequenceTemplate`
- **AND** token-level labels and masks are represented as a
  `DetectionSupervisionView`.
