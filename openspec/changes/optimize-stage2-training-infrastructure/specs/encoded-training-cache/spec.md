# encoded-training-cache Specification (delta: bounded worker residency)

## Purpose
Extend the encoded training cache contract so cache reuse remains memory-bounded under persistent worker execution.

## Requirements

## ADDED Requirements

### Requirement: Encoded cache shard residency MUST be explicitly bounded
When encoded-sample caching is used with persistent workers, worker-local cache residency MUST be bounded by explicit policy rather than allowed to grow without limit across the epoch.

Normative behavior:
- The cache runtime MUST enforce an explicit shard-residency policy, such as:
  - a maximum number of resident shards,
  - an eviction strategy,
  - or an equivalent bounded-access mechanism.
- Cache hits MUST remain functionally equivalent whether the accessed shard was already resident or had to be reloaded after eviction.
- The chosen residency policy MUST NOT change the encoded payload contract delivered to downstream training code.

#### Scenario: Worker shard reuse remains bounded over a long epoch
- **GIVEN** encoded-sample caching is enabled
- **AND** dataloader persistent workers remain alive across many shard accesses
- **WHEN** a worker traverses enough samples to touch more shards than the configured residency bound
- **THEN** older shard state is evicted or otherwise bounded
- **AND** sample payload correctness remains unchanged.
