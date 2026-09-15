## ADDED Requirements

### Requirement: Permanent bridge backend-neutral lifecycle trace

HF decode results for a permanent-bridge composition SHALL carry bounded backend-neutral lifecycle evidence for each sequence without exposing model tensors outside the backend. The evidence MUST bind request id, bridge payload/profile identity, boundary generated-step index, aggregate atom/null mass, selected atom id when an opener occurs, opener/null disagreement, latch step, row-close step, clear outcome, admission RMS ratio, row-write RMS summary, and terminal lifecycle status. Bridge state MUST align to the same prompt/generated token ids and stop semantics as the policy likelihood trace.

#### Scenario: Two batched sequences have different bridge phases
- **WHEN** one request is inside an object row and another has reached a boundary or stop
- **THEN** backend-neutral results MUST preserve distinct lifecycle events for both request ids
- **AND** trace validation MUST fail if an atom id or clear event crosses request identity

#### Scenario: Lifecycle trace is misaligned
- **WHEN** a latch, box-end clear, or boundary route names a generated-step index whose token is incompatible with that transition
- **THEN** backend trace integrity MUST fail before canonical scored artifacts publish

