## Purpose

Observe final provider request identity and cache-reporting evidence safely enough to distinguish local prefix changes from unexplained backend cache losses.

## ADDED Requirements

### Requirement: Final-request observation is opt-in and behavior-neutral
Diagnostics SHALL be disabled by default and observe the supported ordinary Web and context-management compaction request paths after all payload transformations and before dispatch. Observation MUST NOT alter request bytes, dispatch count, response consumption, cancellation, retry policy or model context. Unsupported or oversized bodies and observation failures MUST remain non-blocking and explicitly unobserved rather than creating false final-payload evidence. Final coverage means supported HTTP dispatch; WebSockets and bodies that cannot be inspected without consumption SHALL remain explicitly unobserved. Diagnostics MUST NOT change transport to obtain coverage or replace final-dispatch evidence with an earlier hook snapshot. Adapters that reject injected fetch MUST be left untouched. Supplied transforming transports SHALL remain unqualified unless their final boundary is proven.

#### Scenario: Later payload hook changes input
- **WHEN** a payload transformation changes input after an earlier observer could have seen it
- **THEN** the diagnostic fingerprint represents the actually dispatched body, not that earlier snapshot

#### Scenario: Unsupported adapter
- **WHEN** an adapter does not support injected fetch
- **THEN** enabling diagnostics leaves its options and operation unchanged and reports unobserved coverage

#### Scenario: Observation failure
- **WHEN** a diagnostic sink fails or the body is unsupported or exceeds admission bounds
- **THEN** the original request behavior is retained and no complete final-payload coverage is claimed

### Requirement: Records are bounded and content-free
Records SHALL correlate session/request identity, request kind, model/API, cache-field presence and controlled prefix fingerprints/lengths with response completion and raw usage-field presence. They MUST NOT contain prompt text, opaque signatures, credentials or arbitrary headers. Fingerprint comparison SHALL identify its comparison epoch; bounded retention MUST NOT grow unbounded with requests, sessions or input length, including a stalled asynchronous writer's pending backlog. Public observations MUST contain only immutable body strings or detached allowlisted terminal data, never mutable provider event objects. Observer listeners and tracking MUST be released on disposal/replacement; dropped or ambiguous coverage MUST be disclosed.

#### Scenario: Cache report absent versus zero
- **WHEN** one completed response omits the cached-token field and another explicitly reports zero
- **THEN** diagnostics distinguish those states even when normalized usage represents both as zero

#### Scenario: Concurrent sessions and retries
- **WHEN** requests from two sessions interleave or a request retries
- **THEN** each actual HTTP attempt has a distinct identity and completion remains attributable or explicitly unknown without exchanging fingerprints or usage between sessions

#### Scenario: Warming and late callbacks
- **WHEN** captured request callbacks are reused for cache warming or complete after wrapper disposal
- **THEN** they are not falsely associated with an earlier ordinary attempt and disposed tracking is not resurrected

#### Scenario: Mutating participant and stalled writer
- **WHEN** a public observation listener mutates received data or the diagnostic writer stalls
- **THEN** provider events and checkpoint output remain unchanged, pending memory stays within declared limits and dropped coverage is disclosed

#### Scenario: Restart changes fingerprint epoch
- **WHEN** the observer restarts with a new private fingerprint key
- **THEN** records identify the new epoch and do not claim cross-epoch prefix equality
