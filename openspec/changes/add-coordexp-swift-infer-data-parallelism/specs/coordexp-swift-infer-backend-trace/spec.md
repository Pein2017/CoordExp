## ADDED Requirements

### Requirement: Rank-aware trace diagnostics
Data-parallel inference SHALL add rank and device identity to trace and diagnostic sidecars.
Rank/device fields MUST be diagnostic metadata only and MUST NOT change
selected-token scoring semantics, token alignment semantics, or evaluator row
schema. Rank-local and merged token trace and diagnostic sidecars MUST preserve
rank, world size, assigned parent-visible token, worker logical device, and
worker CUDA visibility evidence where available.

#### Scenario: Token trace rank evidence
- **WHEN** a token trace row is written by a data-parallel worker
- **THEN** it includes rank, world size, assigned device token, and logical
  device fields
- **AND** selected-token score recomputation remains based on token ids,
  generated-step indices, and logprobs

#### Scenario: Backend-neutral trace record
- **WHEN** a future backend writes trace sidecars through the same shard
  contract
- **THEN** rank/device diagnostic fields do not require exposing raw backend
  output objects outside the backend module
