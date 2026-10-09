## 1. Measurement contract and event capture

- [x] 1.1 Add the versioned telemetry entry schema/parser and pure session/provider-model aggregation helper; verify sum-over-sum TPS, coverage, model grouping, unavailable cases, and malformed/duplicate-record rejection with focused tests.
- [x] 1.2 Capture assistant output-delta intervals at the server event boundary using a monotonic clock, correlate completed responses to exact persisted assistant entries, and append measured/excluded metadata after completion; verify multiple assistant/tool cycles and ineligible responses with wrapper-level tests.

## 2. Statistics and persistence integration

- [x] 2.1 Expose the computed metric through both the session-details API and live `get_session_stats` RPC, and carry it through `SessionFileStats`, `SessionStatsInfo`, and the client merge path; verify session reload and live-session stats tests.
- [x] 2.2 Verify custom telemetry records survive session reopen/branching without changing token, cost, or message totals or entering model context; keep telemetry nodes transparent to branch-navigation counts while preserving existing behavior for other custom entries.

## 3. Session details presentation

- [x] 3.1 Add a localized observed-decode-TPS row, coverage, multi-model breakdown and unavailable state to the session details view; retain the measurement boundary in the design after the user's request to remove explanatory UI prose, and verify the production and demo UI render measurable, mixed-model, and unavailable cases.
- [x] 3.2 Run focused session statistics, event-stream, session-tree/navigation, and UI tests plus the project's applicable type/lint checks; resolve regressions without modifying unrelated active work.

## 4. Independent acceptance repair (2026-10-07)

- [x] 4.1 Repair response timing correlation for the installed SDK's copied message events; retain an actual-SDK loopback RED/GREEN covering durable metadata, reopen and distinct response/tool/retry boundaries.
- [x] 4.2 Repair the three stale Web test expectations without weakening branch-lock, selected-resource or exact-prompt behavior; verify the final Web consumer and record repaired-source acceptance and authorized redeployment separately.
