# Acceptance — pi-cache-efficiency-and-diagnostics

## Scope and authority

Source qualification only; no commit, deployment, release installation, shared-profile change, restart or active-session reload. Original journals are read-only and frozen at A lines1–318 / B lines1–176. Advisor review is in review.md. Wave acceptance is ordered, not inferred from a worker report.

## Wave 1 — lead accepted

Owned source: `codex-tools/pi-web`. Explicit CLI/helper: `bin/pi-diagnostics.mjs`, `lib/session-diagnostic-projection.mjs`; statistics and UI through existing production/demo session-stat consumers, live RPC, AppShell rows, ChatWindow memo keys and locales. Owner instructions: `docs/bounded-artifact-diagnostics.md`, `docs/session-usage.md`, `docs/agents/sessions.md`.

Worker RED/GREEN: projection 11/11, stats/reader/category/live-RPC 48/48 after one lead-discovered correction. Lead exercised final CLI/helper/live-RPC/row consumers: 17/17; then the two focused replacement/ambiguous-category regressions passed. Root/demo typechecks and targeted lints passed in worker checks. The baseline TPS failure later passed on the current concurrently modified tree without this change weakening TPS checks; improvement is not attributed to cache work.

Load-bearing counterexample: assistant10 tokens/$0.1 replaced by tool-result10/$0.1 initially yielded lifetime10/$0.1 but category20/$0.2. Corrected merging now uses the unchanged authoritative lifetime delta, retains billed baseline buckets, adds zero for zero net growth, and uses other for ambiguous positive attribution. Append/reload, compaction reduction and monetary regrouping remain qualified (exact integer tokens; USD1e-9 cost tolerance).

Frozen arithmetic: A ordinary90.951866%, pooled86.986773%, cacheRead10,330,624; B ordinary93.787264%, pooled89.595802%, cacheRead5,251,072. This confirms denominator distinctions, not backend-cache causality.

Projection receipt: `codex-tools/pi-web/.local/diagnostics/pi-cache-efficiency-wave1/frozen-session-projection-measurements.json`. A selected raw5,061,058B ->26,395B,99.4785% reduction; B1,716,074B ->16,446B,99.0416%. All100/60 matched usage rows displayed, complete parsing, source unchanged during reads. Large signatures, malformed/partial lines, explicit pointers, unsafe integers/long decimals, source read-only and bounded omissions tested. Output size reduction is bytes, not a measured tokenizer/cache benefit.

Unverified: browser visual layout and real HTTP API route; deterministic presentation/memo and live RPC consumers were tested, no dev server/build/reload was performed.

## Wave 2 — lead accepted

Web default-off `PI_WEB_REQUEST_DIAGNOSTICS=1` observer uses public SDK streamFunction/fetch/eventBus; compact default-false `requestDiagnostics` owns its independent versioned event producer. No core/private-extension import in production. Loopback final-boundary/unsupported adapters/custom fetch/retry/interleaving/warming/native summaries/disposal tests qualify only supported materialized HTTP paths, not WebSocket or terminal ordinary inference. Independently constructed live subagents bypass this Web installation and are explicitly outside coverage.

Web diagnostic suite22/22, root typecheck and targeted lint passed. Compact21files/390tests, package typecheck and installedSDK1.0.3 observation6/6 passed. Broader Web RPC suite39/41 retains two unrelated source-regex baseline failures, not weakened. Lead reran the cross-repository final-consumer test2/2: actual compact source -> installedSDK1.0.3 loopback -> publicEventBus -> Web sink -> journal publication/reopen/opaque replay/continuation, observer off/on/adversarial equivalent. Six HTTP requests, three compactions; absent/zero distinguished; four mutations rejected/four throwing participant calls isolated, zero host rejections and no sentinels persisted. Final compact body2,666B HMAC equals server-received bytes. Checkpoint ID/time are generated, remaining contracts and usage are identical.

One integration RED exposed incompatible unobserved-reason enums; compact now uses the same canonical versioned enum, cross-repository consumer GREEN. Public ambiguous terminal `attemptId:"unknown"` cannot inherit an earlier request; host persists attribution only after validating association.

Hard bounds are implemented/tested: body8MiB, record16KiB, pending128records/256KiB, sessions64/attempts128, files2x1MiB with0700/0600; random process HMAC epoch, detached frozen terminal usage and body strings, no raw source queued. Failures/drop/eviction disclosed; a permanently stalled sink cannot claim complete durable coverage. Observation does not await IO or alter dispatch.

Receipts: Web `.local/diagnostics/pi-cache-efficiency-wave2/` (web-receipt.md, crossrepo-consumer.json, red/green logs); extension `.local/qualification/pi-cache-efficiency-wave2/receipt.md`. Two accidental initial extension checks from the wrong cwd are explicitly excluded; correctly scoped checks are authoritative. Generated packaged-runtime gates remain Wave3-owned. No external model request or runtime adoption performed.

## Wave 3 — lead accepted for affected source; global release gate not green

Owner `codex-tools/pi-extensions/packages/pi-codex-compact`. `preserveCacheAffinity:false` is global-only and validated. True affects only Context Management: originating public session ID plus provider-default retention; absent/false and other protocols retain their old request contract. System/tool preparation removes only proven redundant outer declarations; all persisted systems/sections/ordered deltas remain. Legacy and differing forced/effective prompts are retained conservatively. Private shared-memory/context-hook coupling is absent.

InstalledSDK pre-fix evidence demonstrated duplicate preparation changing input position0. Current ordinary-to-maintenance comparison preserves input through position4 before intentional assistant/maintenance append; reopened checkpoint epoch preserves through position3 and replays one exact opaque item plus tail. Generated active-task continuation executes the completed tool once. Forced-effective conservative case still diverges at position0; equal-prefix behavior for arbitrary ephemeral context is not claimed.

Lead falsification also found visible tool equality could discard explicit constrainedSampling:false/config overrides. Corrected comparison now requires every defined explicit outer field to match persisted data; absent discovery fields do not erase persisted constraints. RED/GREEN covers false/config/matching/mismatch/persisted-only/future fields. The README table row was repaired and interface reviewed; experimental warnings and one Changeset retained.

Final checks: compact23files/420tests, installed provider affinity24/24, installedSDK/generated Jiti5/5, guard6/6+focused strictTS, unchanged Wave2 cross-repository2/2; build/package dry-pack/local tarball inspection/zero-model offline CLI load passed. Root npmruncheck passed (build/Biome/boundaries/workspace types), README heading audit36packages passed. All were serial relative to root/Kit gates.

**Root npmtest is not green.** Original environment6462/6464 retained two pi-file-context failures; inherited TMPDIR was inside the Web Git checkout. Two unchanged files passed30/30 under command-scoped TMPDIR/TMP/TEMP=/tmp. The single qualified full rerun passed6463/6464, with an untouched startup benchmark timeout fixture failing ENOENT on capture.jsonl. Source/config diffs for these owners are empty. No unrelated test was changed, weakened or retried to manufacture green. This remains a repository-wide qualification warning; no release/adoption acceptance is issued.

Hosted preflight: **HOLD before network,0 hosted model calls**. InstalledSDK1.0.3 OAuth-shaped payloads omit max_output_tokens even with explicit maxTokens128000; catalog maxTokens does not establish billed-output exposure. Eight actual calls/15minutes cannot justify the declared USD2 bound. API-key contrast is not permission to substitute the selected route. Dispatch guard locally seals on first401 and blocks installed-core summary fallback before a second fetch; deadline/8-call/cancel/overlap/endpoint bounds passed. Cache benefit remains unverified, and historical A188/A255 losses remain unexplained without observed final-wire evidence.

Receipts: extension `.local/qualification/pi-cache-efficiency-wave3/receipt.md`, `final-prefix-after.json`, final gate .log/.exit files, and `hosted-preflight.json/.md`. Local qualification tarball is unadopted, not a release. No active qualification jobs remain.

## Final verification

Affected current diff and nine requirements/23 scenarios reviewed against design/advisor corrections; complete mapping in verification.md. Strict OpenSpec validation passed. Tasks are complete for the implementation/source-qualification scope including the explicitly allowed hosted-HOLD branch, not a claim that every global test or hosted cache benefit passed.

No commit, archive, release publication/installation, shared-profile mutation, credentials change, active-user restart/reload or scientific source/artifact edit was performed. Web has substantial unrelated concurrent dirty work, preserved. Browser visual layout/real HTTP API, unsupported/WebSocket/terminal ordinary and independently constructed live-subagent diagnostic coverage, arbitrary context-hook prefix equality, hosted benefit and adoption remain unverified or out of scope as documented.

## Independent repair and Web deployment — 2026-10-07

This follow-up preserves the preceding historical receipts. The user authorized fixes and direct local Web deployment, confirming no active Web sessions. The companion TPS source defect and stale Web assertions are repaired; actual SDK lifecycle/HTTP-loopback consumer checks and145 combined Web feature tests pass. Relevant pagination exact-statistics expectations now explicitly include all usage categories while retaining old totals;18/18 pass and a nearest-wrong category mutation fails despite unchanged lifetime totals.

Extension tooling repair is confined to `scripts/benchmark-extension-startup.mjs` and `test/benchmark-extension-startup.test.ts`: a matching-id get_state readiness probe precedes the command deadline, with a separate bounded startup deadline. Hung-after-ready retains100ms; false/missing/mismatched readiness and early exit remain bounded and cleaned. A socket-gated deterministic nearest-wrong timer-placement mutation fails. The real managed offline Pi CLI caller passes with two isolated fresh profiles and zero model calls.

Current extension root gates are now GREEN: `TMPDIR=/tmp TMP=/tmp TEMP=/tmp npm run check` exit0; subsequent `npm test`493files/6469tests exit0. The original6463/6464 result remains historical. New gate evidence: `codex-tools/pi-extensions/.local/qualification/benchmark-readiness-20261007/receipt.md` and root-check/root-test logs/exit files. No compact production source was changed by this benchmark repair.

The ordinary full Web suite remains non-green (the retained run2275/2391,116failures), mostly English SSR expectations under the existing Chinese default plus inherited warnings/environment and unrelated stale/timing expectations. This is bounded source/consumer acceptance and a local dev-service deployment, not full Web package-release acceptance.

Local Web restart completed via the canonical launcher after exact-process and idle checks. Homepage,38 compiled assets and real running/list/detail APIs return200; live API usage categories reconcile. The restart retains the shared profile/managed SDK/proxy and existing extension release. `PI_WEB_REQUEST_DIAGNOSTICS` remains off; compact diagnostics and preserveCacheAffinity are not enabled, and the new experimental compact release is not installed. Hosted cache benefit still HOLD with zero external calls. Runtime receipts and the companion TPS acceptance are linked under Web `.local/diagnostics/acceptance-20261007/` and `../pi-web-observed-decode-tps/acceptance.md`.
