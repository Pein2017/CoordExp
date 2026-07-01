## ADDED Requirements

### Requirement: GT-vs-pred visualization resources are derived review views

The clean-break detection architecture SHALL treat GT-vs-pred visualization
resources as derived review views rather than as a competing canonical scene
schema.

Normative behavior:

- visualization resources MAY continue to use existing `gt_vs_pred`-family
  filenames and review sidecars;
- visualization inputs MUST be derived from `DetectionScene`,
  `DecodedDetectionResult`, `DetectionEvalRecord`, or
  `ScoredDetectionEvalRecord` concepts rather than defining a second semantic
  object model;
- visualization-specific sorted GT order, prediction order preservation,
  matching overlays, labels, and rendering metadata MUST remain visualization
  view policy;
- visualization view policy MUST NOT redefine training object ordering,
  Stage-2 assignment semantics, eval metric meaning, or raw/scored artifact
  interpretation.

#### Scenario: Visualization consumes scene/eval concepts without becoming canonical semantics

- **GIVEN** a detection eval artifact and optional matching annotations
- **WHEN** visualization resources are prepared
- **THEN** the renderer receives a visualization view derived from the semantic
  scene/eval-record concepts
- **AND** the visualization resource does not become the canonical source for
  training, rollout correction, inference parsing, or official metrics.
