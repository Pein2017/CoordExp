## Why

Pi Web exposes token counts and per-response streaming speed, but session details do not show an aggregate generation-rate metric. Users need a session-level average based on provider-reported output tokens while excluding time before generation and shell/tool execution; the value must be clearly described as host-observed decode throughput, not the provider's internal GPU measurement.

## What Changes

- Measure eligible streamed assistant responses in the Pi Web server before SSE transport, and persist versioned timing metadata alongside the corresponding session-log entry without changing Pi's standard usage totals.
- Show a session-level, output-token-weighted observed decode TPS and its measurement coverage in session details, with per-provider/model rates when a session used multiple models and an explicit unavailable state when no response qualifies.
- Keep the metric distinct from total session duration, time-to-first-token, tool/shell runtime, and the existing per-turn streaming estimate.

## Capabilities

### New Capabilities
- `pi-web-observed-decode-tps`: Session-level observed provider-output-token throughput, its eligibility and aggregation rules, persistence, and presentation.

### Modified Capabilities

None.

## Impact

- Pi Web server-side agent event handling and Pi session-log custom metadata.
- Session statistics/API and the session-details UI.
- Tests for timing/eligibility, aggregation, persistence/reload, and details presentation. No provider integration, Pi Core API change, release, or deployment is included.
