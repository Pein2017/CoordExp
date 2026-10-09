# pi-web-observed-decode-tps Specification

## Purpose
Provide a transparent, reload-stable session-level estimate of model output decode throughput from Pi Web's server-observed stream, without attributing shell/tool or user wait time to generation.
## Requirements
### Requirement: Measure observed model-output generation intervals
Pi Web SHALL measure each completed streamed assistant response using a monotonic server clock at the first and last non-empty assistant output delta observed before SSE forwarding. A response is measurable only when it has at least two such deltas, a positive elapsed interval, and a positive final provider-reported output-token count. The response rate SHALL be its final `usage.output` divided by the elapsed interval. Responses that are incomplete, non-streamed, lack final output usage, or have insufficient timing data SHALL NOT be represented as zero-rate samples.

The interval SHALL exclude time to first output and time outside that response, including shell/tool execution, user idle time, and waits between responses. Delays between the first and last server-observed deltas remain in the interval. The metric SHALL be labeled as host-observed output decode TPS and SHALL NOT claim to be the provider's internal decoder/GPU rate.

#### Scenario: Measure a completed streamed response
- **WHEN** a completed assistant response has multiple non-empty output deltas, positive final provider output usage, and a positive server-measured first-to-last-delta interval
- **THEN** Pi Web records that response's output-token count and interval as a measurable sample

#### Scenario: Exclude time outside model output streaming
- **WHEN** a session has time before the first output delta, shell/tool execution, user idle time, or a wait between separate assistant responses
- **THEN** none of that time is added to any response interval or the session TPS denominator

#### Scenario: Exclude responses without a valid measurement
- **WHEN** a response is incomplete, non-streamed, has fewer than two output-bearing deltas, has missing/non-positive output usage, or has invalid elapsed time
- **THEN** it is not included as a zero or fabricated rate in the session TPS calculation

### Requirement: Aggregate session TPS and report measurement coverage
Pi Web SHALL calculate session observed decode TPS as the sum of final provider-reported output tokens divided by the sum of elapsed intervals for all measurable responses recorded in that session. It SHALL NOT use an unweighted arithmetic mean of per-response rates. It SHALL also calculate that ratio independently for each provider/model represented in the samples. Session details SHALL show the all-session rate in tokens per second and its measurable-versus-tracked-completed-response count; when more than one provider/model is represented, it SHALL also expose the separate per-provider/model rates. It SHALL show an unavailable state rather than zero when no measurable response exists.

#### Scenario: Aggregate multiple responses by their measured work and time
- **WHEN** a session contains a 100-token response measured over 2 seconds and a 300-token response measured over 3 seconds
- **THEN** its aggregate rate is 80 tokens per second, calculated as 400 divided by 5, rather than the arithmetic mean of 50 and 100

#### Scenario: Show a usable rate and coverage
- **WHEN** at least one response in a session is measurable
- **THEN** session details show the aggregate tokens-per-second value and the measured-response count relative to tracked completed assistant responses

#### Scenario: Keep model-specific rates distinct
- **WHEN** measurable responses in one session use more than one provider/model
- **THEN** session details retain the all-session aggregate and expose a separate sum-of-tokens divided by sum-of-duration rate for each provider/model group

#### Scenario: Show unavailable when no response is measurable
- **WHEN** no response in a session is measurable
- **THEN** session details show that the metric is unavailable and do not display zero, infinity, or a rate inferred from legacy messages

### Requirement: Persist measurement metadata without changing conversation usage
Pi Web SHALL persist versioned measurement metadata associated with the exact completed assistant session entry so the aggregate can be reconstructed after reload and across session branches. This metadata SHALL NOT be sent as model context, displayed as conversation content, or included in Pi's standard token, cost, or message totals. Existing session data without measurement metadata SHALL remain valid and SHALL NOT be assigned inferred timing or TPS.

#### Scenario: Reconstruct metrics after reloading a session
- **WHEN** a session containing persisted measurement metadata is reopened
- **THEN** Pi Web recomputes the same session TPS and coverage from that metadata

#### Scenario: Preserve legacy session behavior
- **WHEN** a session predates measurement metadata or contains only legacy assistant messages
- **THEN** Pi Web keeps the transcript and standard usage statistics intact and reports the decode-TPS metric as unavailable unless measured records exist

#### Scenario: Keep telemetry out of model context and standard totals
- **WHEN** a session with measurement metadata is used for a later model request or its standard token/cost/message statistics are computed
- **THEN** the metadata contributes no conversation content and does not alter those standard totals

