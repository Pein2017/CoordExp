## MODIFIED Requirements

### Requirement: loss_duplicate_burst_unlikelihood remains the canonical B-only suppression module
The teacher-forcing objective pipeline SHALL continue to use
`loss_duplicate_burst_unlikelihood` as the canonical Channel-B local
suppression module for the v3 contract.

Normative behavior:

- `loss_duplicate_burst_unlikelihood` remains a valid objective module name,
- `loss_duplicate_burst_unlikelihood` MUST continue to declare `channels: [B]`,
- `loss_duplicate_burst_unlikelihood.config` MUST remain `{}` in v1,
- the runtime metadata consumed by
  `loss_duplicate_burst_unlikelihood` MUST encode canonical duplicate-control
  non-survivor first-divergence targets projected onto the final clean prefix
  after the pre-match duplicate-control step and existing Channel-B matching /
  triage flow,
- the runtime metadata producer MUST preserve deterministic target ordering for
  identical rollout inputs,
- the runtime metadata producer MUST NOT require a second teacher-forced
  forward or any post-hoc confidence signal.

#### Scenario: loss_duplicate_burst_unlikelihood accepts cluster-aware duplicate continuation metadata
- **WHEN** Channel-B v3 provides canonical cluster-aware duplicate
  first-divergence targets
- **THEN** the `loss_duplicate_burst_unlikelihood` module may consume them
  without requiring a new module name
- **AND** pipeline validation still treats
  `loss_duplicate_burst_unlikelihood` as the canonical B-only suppression
  module.

#### Scenario: Identical duplicate clusters produce stable target ordering
- **WHEN** the same rollout sample is prepared twice with identical parsed bbox
  objects and triage inputs
- **THEN** the emitted duplicate-unlikelihood target list is identical across
  preparations
- **AND** training reproducibility does not depend on container iteration order.
