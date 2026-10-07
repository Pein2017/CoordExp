# Independent repair acceptance — 2026-10-07

The original installed-SDK acceptance failed: copied message objects caused real streamed responses to be persisted as `not-streamed`. The failure and original reproduction remain sealed at Web `.local/diagnostics/acceptance-20261007/acceptance.md` and `tps-sdk-repro.log`. This record describes the repaired candidate, not a replacement of the historical failure.

## Repaired response boundary

`lib/session-decode-tps.ts` follows the SDK's sequential start/update/end lifecycle instead of indexing copied update objects. It measures non-empty text/thinking/tool deltas with the existing monotonic clock, then associates the finalized message object with its exact durable assistant entry. Overlapping lifecycles, unfinished/orphan events, reused final objects and ambiguous persisted entries cannot fabricate timing. Separate responses, tool gaps, interrupted attempts and automatic retries retain distinct samples and exclusions. Historical records and provider usage remain unchanged.

Actual installed Pi SDK1.0.3 → local HTTP/SSE → real Web wrapper → SessionManager persistence → reopen/continuation tests are maintained in `lib/session-decode-tps-sdk.test.mjs`. Before the repair, all three initial SDK regressions failed on lost timing. After the repair, four SDK cases plus deterministic tracker tests pass17/17. SDK cases cover reopen and a real tool cycle, abort then a fresh prompt, automatic retry, and a positive-usage final-only response excluded as nonstreamed. Callback brackets verify the measured intervals; tool execution lies outside them. No external model calls occurred.

The lead independently reran the original failing reproduction against repaired source:2 deltas/output100 become measured1/tracked1, with finite reopened TPS. Its separate GREEN log preserves the original RED log. Production and demo continue displaying concise metric rows per the user's later instruction; no explanatory UI prose or fabricated historical TPS was restored.

## Verification and limits

Lead combined TPS, UI/navigation, wrapper/exact-prompt, usage/statistics/projection and installed-SDK request-observation/crossrepo suite:145/145, exit0. Root Web and demo TypeScript, scoped changed-file ESLint and whitespace checks pass. Evidence is under `/data/CoordExp/codex-tools/pi-web/.local/diagnostics/acceptance-20261007/`.

Three previously failed tests are repaired without changing production policy: localized branch-lock notice, selected-resource discovery/tool scope, and exact-prompt factory/hook composition. In-memory nearest-wrong mutations widening extension discovery or dropping exact prompt factories fail. These checks do not establish unrelated feature correctness.

The wider Web npmtest run is **not green**:2275/2391,116 failures.108 component failures assert English output against the pre-existing Simplified Chinese default. Other failures concern inherited process warnings/environment, unrelated timing or stale configuration/statistics expectations. Relevant stale cache-category pagination expectations are separately repaired; the original full log remains retained. Do not claim all Web tests pass or package publication qualification.

Hosted cache benefit remains HOLD with zero external requests. This TPS repair does not measure provider-internal GPU throughput, establish cache improvement, or authorize adopting an experimental compact release. Source qualification and local Web deployment are distinct; deployment observations will be recorded below after the user's authorized restart.

## Authorized local Web deployment

The user explicitly confirmed no active Web Pi sessions and requested direct deployment. Before stopping, the lead rechecked the running API (empty), exact launcher/child/cwd and descendants (only known PostCSS build workers). SIGTERM stopped the exact launcher2086926; launcher and server2087430 exited and port12345 closed before a replacement was started. The canonical `bin/local-dev.sh` was started once in tmux `pi-web`.

New launcher2924369/server2924777 retain `/data/CoordExp/.pi`, official managed Pi1.0.3, proxy127.0.0.1:9090 and listener127.0.0.1:12345. Homepage, all38 referenced compiled assets, session-list/running/detail APIs return200. Detail usage categories reconcile with unchanged lifetime totals; historical TPS remains null/0-of44 rather than being backfilled. No runtime errors appear after the new startup. These are local HTTP/API and SDK lifecycle observations; browser visual operation and a real provider request were not tested.

Pagination accounting regression repaired18/18; original full-suite failure remains in its historical log. The other7 non-UI failures were independently classified as inherited Undici warnings/host environment, stale tracing-root expectation or receive-callback timing assumption; no material runtime defect was demonstrated. This classification does not clear those ordinary full-suite failures.

Deployment receipts: Web `.local/diagnostics/acceptance-20261007/deployment-stop.json`, `deployment-final.json`, `runtime-smoke.json`, and `tps-sdk-repair-green.log`. No npm publication, Git commit, upstream push, plugin installation/adoption, credential/profile edit, historical journal rewrite or diagnostic/affinity option activation occurred. Newly completed Web responses use the repaired tracker; past lost timing remains unavailable.
