## ADDED Requirements

### Requirement: Dynamic HF permanent-bridge execution

Dynamic HF inference SHALL compose the validated permanent bridge with the base model, DoRA adapter, and selected-token delta before generation. Generation MUST compute static atoms once per image-conditioned sequence, maintain independent router and row-latch state for every batch element, apply only aggregate availability before native opener/stop selection, and apply the hard owner value from opener through row close. The runtime MUST NOT force an opener, owner row, duplicate suppression, or terminal token outside native model logits.

The supported Stage 1 execution path SHALL be deterministic greedy HF decoding. Any generation mode whose sequence branching or cache reordering cannot preserve exact per-sequence bridge lifecycle MUST fail before generation rather than share or lose latch state.

#### Scenario: Batched rows are at different lifecycle positions
- **WHEN** one batch element is inside an object row while another is at a boundary or has stopped
- **THEN** each element MUST retain its own atom inventory, boundary distribution, active latch, and clear event
- **AND** no bridge state may leak between batch elements

#### Scenario: Native opener disagrees with null
- **WHEN** native decoding emits `<|object_ref_start|>` while the router ranks null first
- **THEN** the runtime MUST latch the best non-null atom for that row
- **AND** inference diagnostics MUST record the null/opener disagreement

### Requirement: Bridge-aware inference evidence

Bridge-enabled inference SHALL preserve the existing raw, scored, token-trace, parse, and failure-accounting artifacts and add bounded bridge identity and lifecycle evidence. The run manifest MUST bind the bridge payload fingerprint and architecture profile. Bounded per-boundary evidence MUST include non-null/null mass, selected atom identity, latch/clear outcome, and admission RMS ratio; bounded per-row evidence MUST include row-write RMS ratio and any lifecycle disagreement.

#### Scenario: Bridge lifecycle fails mid-generation
- **WHEN** a row closes without one valid active latch or a latch survives past its clear boundary
- **THEN** the affected backend result MUST fail trace integrity
- **AND** the scored run MUST not silently publish that row as canonical evidence

