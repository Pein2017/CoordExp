## Context

See `proposal.md` for motivation and `specs/pi-web-observed-decode-tps/spec.md` for the behavior contract.

Pi Web's `AgentSessionWrapper.start()` receives Pi AgentSession events before forwarding them to browser/SSE listeners, so it can capture server-side stream timing without relying on browser delivery cadence. Assistant final usage and provider/model identity are available when the response completes. Session details currently flow through `GET /api/sessions/[id]`, `computeSessionStats()`, the `SessionStatsInfo` client type, and the session panel in `components/AppShell.tsx`; the live `get_session_stats` RPC path is separate and must expose the same additional metric.

Pi session logs already support app-owned custom entries, used by Pi Web for other session metadata. Such entries are not conversation messages, but they advance the session leaf and the branch navigator currently counts skipped non-message nodes. The telemetry implementation must preserve that UI behavior.

## Goals / Non-Goals

**Goals:**

- Measure intervals from raw server-observed assistant output deltas, independently of SSE buffering and the browser's per-turn estimate.
- Persist exact response-level timing/eligibility data and derive session and provider/model summaries from the log.
- Keep telemetry out of the model context, transcript, and existing token/cost/message totals.
- Present the aggregate, eligible/tracked response coverage, and provider/model breakdown in session details, including a truthful unavailable state.

**Non-Goals:**

- Measuring provider-internal GPU/decode timestamps, network-free inference speed, or time-to-first-token.
- Changing model-provider APIs, Pi Core, the existing per-turn badge, or ordinary session usage totals.
- Retroactively estimating TPS for existing logs, changing stable OpenSpec specs, releasing/deploying, or restarting Pi Web.

## Decisions

1. **Observe at the server event boundary, before SSE.** Add narrowly scoped tracking to the `AgentSessionWrapper` subscription in `lib/rpc-manager.ts`, before its existing event forwarding. Use a monotonic clock on non-empty assistant output deltas; finalize a candidate from the completed assistant message's final usage/provider/model. Do not time the whole agent run or derive timing from client events. The requested sol-6.1 read-only design review supported this boundary and ratio-based aggregation, while emphasizing that host-observed intervals must not be presented as an exact provider-internal rate.

2. **Correlate and persist only after the assistant entry exists.** Buffer per-response observations while the run is active. At `agent_end`, when completed messages have been appended to the session manager, correlate each finalized response to its exact assistant session-entry ID. Append a versioned Pi Web custom entry containing that ID, provider/model, final output-token count when available, elapsed milliseconds when measurable, and an eligibility/exclusion status. If an event cannot be matched unambiguously to a persisted assistant entry, do not synthesize a sample. Recording exclusion statuses for completed responses lets the UI report measured-versus-tracked coverage; interrupted responses that never produce a completed entry remain outside that denominator.

   A custom entry is preferred over a sidecar because it travels with the session file and its branches and needs no separate cleanup/backup lifecycle. It is preferred over modifying provider usage or assistant message fields because it does not alter Pi's standard usage contract or model input. Do not add this data to the assistant content or a standard `usage` entry.

3. **Compute summaries from persisted samples.** Add a pure, validated aggregation helper over session entries. For every measurable sample calculate `outputTokens / (elapsedMs / 1000)`; session TPS is `sum(outputTokens) / sum(elapsedMs / 1000)`, never a mean of per-response rates. Compute the same ratio per provider/model. Session details show one all-session rate and show grouped rates when more than one provider/model is present. Ignore malformed, duplicate, unsupported-version, or uncorrelated records. Read all logged branches, consistent with existing whole-session token/cost totals.

4. **Carry the metric through both live and persisted statistics paths.** Add an optional typed decode-TPS summary to `SessionFileStats`/`SessionStatsInfo`. The session API calculates it from session entries; the live `get_session_stats` RPC augments Pi's standard stats with the same helper. The existing client merge path preserves the server-computed summary, and the session details' token/stat rows render a localized label, rate and response coverage. Keep equivalent demo/mock types and fixtures aligned where those mirror production UI/statistics.

5. **Keep telemetry structurally invisible.** Custom telemetry entries remain ignored by normal conversation and usage aggregation. Teach the branch navigator to treat this dedicated custom-entry type as transparent for its user-visible skipped-entry count, without changing how other custom entries are counted. Tests must cover following a telemetry leaf with another prompt and branching/reloading a session so the data neither becomes prompt content nor distorts navigation.

6. **Communicate the measurement limitation.** Label the value `Observed decode TPS` (or localized equivalent), not an unqualified vendor TPS. The specification and acceptance record retain the measurement boundary: the clock covers first-to-last server-observed deltas, includes delivery/proxy pauses inside that interval, excludes TTFT and all time outside the response, and uses final provider output-token usage. Since provider usage can include hidden reasoning not represented by emitted deltas, the estimate may differ from a provider's internal decoder rate. Do not apply a speculative correction.

## Risks / Trade-offs

- [Users interpret the metric as exact provider/GPU throughput] → Use an explicit observed-rate label and retain the boundary in the owner records; never call it a provider-internal measurement.
- [Proxy/network buffering or pauses reduce the observed rate] → This is inherent in the chosen host-observed signal; retain those pauses in the denominator and document the boundary rather than silently claiming pure decode time.
- [Final `usage.output` includes tokens not represented by streamed deltas, such as hidden reasoning] → Disclose this mismatch and avoid correction factors or false precision.
- [A Pi session custom entry advances the branch leaf or appears as a skipped node] → Append only after completion, keep it out of context/usage, and make this dedicated type transparent to branch-navigation counts with regression tests.
- [Legacy or incomplete responses lack sufficient evidence] → Never infer historic timestamps; use explicit exclusion status for completed tracked responses and show unavailable when no valid sample exists.
- [Message/event correlation is ambiguous under retries or multiple assistant messages in one run] → Correlate against persisted entries and drop any ambiguous sample rather than associating it with the wrong response; test multiple message/tool cycles.

## Migration Plan

No migration is needed. Existing session files remain readable and their historical messages are not backfilled. Newly completed responses add versioned Pi Web custom telemetry entries. Rollback is code-only: older Pi Web builds ignore custom entries, while standard messages and usage remain intact. Do not archive this change, synchronize stable specs, publish, deploy, or restart Pi Web as part of this work.

## Independent acceptance repair (2026-10-07)

The user removed explanatory UI prose after initial implementation; the details panel displays metric rows only. The measurement limitation remains in this design. Independent installed-SDK acceptance found that start/update events copy message objects while end carries the finalized object. Stable-reference mocks therefore do not qualify response timing correlation. Repair must follow the actual per-response lifecycle, associate the finalized response with its exact persisted entry, and retain an installed-SDK HTTP-loopback regression. Historical excluded/unmeasured records remain unchanged. The user subsequently authorized fixes and Web redeployment after acceptance; that adoption does not include plugin installation or enabling experimental cache options.
