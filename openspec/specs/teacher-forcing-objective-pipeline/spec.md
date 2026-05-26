## Purpose
Define the teacher-forcing objective pipeline module contract, including stable
module names and rollout-correction duplicate-suppression ownership.

## Requirements

### Requirement: duplicate-burst unlikelihood is removed from live objectives
The teacher-forcing objective pipeline SHALL reject
`loss_duplicate_burst_unlikelihood` as a live Stage-2 objective module while
retaining duplicate filtering and diagnostic metadata.

Normative behavior:

- live Stage-2 objective lists MUST omit `loss_duplicate_burst_unlikelihood`,
- pipeline validation MUST reject `loss_duplicate_burst_unlikelihood` instead
  of creating a compatibility alias,
- the retained runtime diagnostic metadata MUST encode duplicate-control
  non-survivor first-divergence records aligned to the final clean prefix
  after the pre-match duplicate-control step and existing rollout-correction matching /
  triage flow,
- the runtime metadata producer MUST preserve deterministic record ordering for
  identical rollout inputs,
- the runtime metadata producer MUST NOT require a second teacher-forced
  forward or any post-hoc confidence signal.
- duplicate-control diagnostics and counters SHALL remain valid after the
  objective module is removed from live canonical training.

#### Scenario: Removed loss_duplicate_burst_unlikelihood is rejected
- **WHEN** rollout-correction v3 provides canonical cluster-aware duplicate
  first-divergence records
- **AND** a config declares `loss_duplicate_burst_unlikelihood`
- **THEN** pipeline validation fails fast
- **AND** duplicate-control diagnostics remain available without the removed
  objective module.

#### Scenario: Identical duplicate clusters produce stable diagnostic ordering
- **WHEN** the same rollout sample is prepared twice with identical parsed bbox
  objects and triage inputs
- **THEN** the emitted duplicate-control first-divergence records are identical across
  preparations
- **AND** training reproducibility does not depend on container iteration order.
