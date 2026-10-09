# Verification report — pi-cache-efficiency-and-diagnostics

## Summary

| Dimension | Result |
| --- | --- |
| Completeness | 12/12 implementation/source-qualification tasks; 9 requirements mapped |
| Correctness | 23/23 specified scenarios have deterministic coverage or the explicitly allowed hosted-HOLD result |
| Coherence | Three ordered waves; advisor corrections applied; existing source owners/public APIs/default-off compatibility retained |
| Global release qualification | NOT GREEN: qualified root test6463/6464; no release/adoption acceptance |
| Hosted cache benefit | HOLD before dispatch;0 external model calls; no cache-hit guarantee |

Planning home is `/data/CoordExp/openspec/changes/pi-cache-efficiency-and-diagnostics`; no separate store, archive or scientific record is created. Acceptance and detailed receipt pointers are in acceptance.md. This report verifies the affected change, not every unrelated dirty change in the shared repositories.

## Requirement and scenario mapping

Paths below are relative to source owners **W**=`codex-tools/pi-web`, **C**=`codex-tools/pi-extensions/packages/pi-codex-compact`.

| Requirement | Implementation | Scenario/consumer evidence |
| --- | --- | --- |
| Bounded journal projection preserves evidence | W `bin/pi-diagnostics.mjs`, `lib/session-diagnostic-projection.mjs` | `session-diagnostic-projection.test.mjs`: large signature before usage, malformed/partial lines, finite ranges, source unchanged; frozen A318/B176 receipts show all matched rows and byte reduction |
| Explicit artifact selectors | Same helper/CLI | Pointer errors/prototype/opaque keys, exact primitive types,9007199254740993 and long-decimal visible rejection, input admission/display omissions; generic tools untouched |
| Categorized usage reconciles | W `lib/session-stats.ts`, `lib/session-usage-breakdown.ts`, demo mirrors/types, RPC | `session-usage-breakdown.test.mjs`, `session-stats.test.mjs`, `session-stats-rpc.test.mjs`: all categories, abandoned branches/context edits, unknown/tool usage, exact token and USD1e-9 cost reconciliation; lead replacement counterexample fixed |
| Cache fractions/display unambiguous | W AppShell/ChatWindow production/demo and locale rows | Row/localization/empty-denominator and memo refresh consumers; append/reload/live RPC reconciles; A ordinary90.951866/pooled86.986773%, B93.787264/89.595802% |
| Final observation opt-in/neutral | W `lib/request-diagnostic-observer.ts`, RPC integration; C `request-observation.ts`, `remote-context-management.ts`, settings/menu | SDK loopback late payload hook equals server body; unsupported adapters untouched, supplied transforming fetch unknown, no Response consumption/extra dispatch; source-owned event enum RED/GREEN join |
| Records bounded/content-free | W `request-diagnostics.ts`, `request-diagnostic-sink.ts`; independent compact producer | `request-diagnostics.test.mjs`, observer/SDK/RPC/crossrepo tests: absent/zero/invalid, epoch/key, interleaving/retry/warming/native summaries, unknown attribution, replacement/disposal, mutation/throws/stalled writer/drop/rotation/private permissions/sentinel exclusion |
| Compatible compact affinity | C settings/menu, codex-compact/remote-context-management | `cache-affinity.test.ts`, settings/menu tests: defaultfalse/true, queued durability/reversible opt-out, other protocols byte equivalence, provider/API-key/OAuth/environment/compatibility/current model branches;24 installed-provider cases |
| Qualified prefix without private coupling | C `cache-affinity.ts`, opt-in context preparation | Persisted systems/sections/legacy/forced prompts/ordered deltas/opaque marker tests; explicit constraint/future-field override RED/GREEN; actual installedSDK final source/generated prefix facts before/after and checkpoint epoch |
| Checkpoint/task survives | Existing checkpoint/replay ownership, enabled path | `affinity-sdk.test.mjs`5/5 and crossrepo2/2: generated Jiti, cancellation, publication/reopen/repeated compaction, exact one opaque item/new tail, task continuation and tool once; guarded hosted preflight qualifies allowed HOLD branch |

## Final evidence and scope

Lead final Web combined cache suite **81/81** on the current tree, exit0: journal/stat/reader/category/live RPC plus all request diagnostics and compact crossrepo tests. Log/exit: W `.local/diagnostics/pi-cache-efficiency-wave2/lead-final-cache-suite.*`. Worker root/demo TypeScript and targeted lint passed; no Next build, dev server or browser reload was performed.

Compact final package suite **420/420**, installed-provider affinity24/24, generated installedSDK/Jiti5/5, hosted guard6/6 and strict focused TypeScript passed. `npm run check`, local build/pack/tar inventory/zero-model offline CLI smoke and36-package README heading audit passed. No production extension imports another extension's private implementation; qualification-only crossrepo source imports are not runtime coupling. Settings retain the existing file/queue/validation/rollback/menu owner. Input/body/queue/tracking/ring bounds are declared and tested, not an unbounded tracing service.

Before correction, two load-bearing main-agent counterexamples failed: category-only live replacement inflated buckets despite unchanged lifetime totals; visible-only tool equality discarded explicit constrainedSampling overrides. Both are fixed with tests. Prefix duplicate preparation was observed in installedSDK final bytes before opt-in repair. Default-off request behavior and retained checkpoint contracts remain qualified, not inferred from settings alone.

## Issues / limitations

**CRITICAL for affected implementation:** none unresolved. This is not a statement that release gates are all green.

**WARNING — repository-wide test gate:** original root result6462/6464 had two fixture failures caused by inherited TMPDIR inside the Web Git checkout. Source/config of those owners is unchanged; scoped `/tmp` falsification passed30/30 without modifying tests/global environment. One qualified root rerun6463/6464 retained `test/benchmark-extension-startup.test.ts` timeout-fixture ENOENT on capture.jsonl. That test/benchmark script is unchanged. No repair, weakening or retry was made. Root npmtest exit1 remains recorded; a fully green repository/release gate is not claimed. Resolve/qualify this separate timing harness before claiming repository-wide release readiness.

**WARNING — visual/live-API coverage:** deterministic row/memo and live RPC consumers passed; browser layout and real HTTP session API were not exercised. Supported final HTTP coverage is not universal: WebSocket/non-materialized/unsupported transports and terminal ordinary inference remain unobserved; independently constructed live subagents bypass this RPC installation.

**Planned HOLD, not fabricated failure/success:** selected OAuth branch omits emitted output cap even with explicit maxTokens. Deadline/eight actual fetches do not justify USD2 exposure. No hosted run started; API-key substitution, extra protocol and retry are prohibited. Hosted cache improvement, historical isolated misses and arbitrary ephemeral context-hook equality are not established by local tests.

No commits, profile/credentials/runtime installation, active-session restart/reload, publication or archive were performed. Local source and qualification tarball are not production adoption. The change may be considered source-qualified with these recorded limits; it is not declared ready for unrestricted release.
