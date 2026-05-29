# trainer-metrics-components Specification (Delta)

## MODIFIED Requirements

### Requirement: Stable metric and batch key names
The training metrics contract SHALL expose canonical observability for
metadata-driven proxy supervision when it is enabled.

Normative behavior:

- the trainer SHOULD emit canonical proxy-supervision gauges or counters for:
  - real object count
  - strict object count
  - plausible object count
  - effective desc-weight sum
  - effective coord-weight sum
- the corresponding key names MUST be documented in
  `docs/training/METRICS.md`,
- these metrics MUST reflect the effective supervision after packing-aware
  aggregation rather than raw pre-pack sample counts only.

#### Scenario: Proxy-supervision metrics expose effective weight mass
- **WHEN** an augmented batch contains both strict and plausible objects
- **THEN** training logs expose both object-tier counts and effective desc /
  coord weight totals
- **AND** operators can verify that plausible supervision is present but
  downweighted.
