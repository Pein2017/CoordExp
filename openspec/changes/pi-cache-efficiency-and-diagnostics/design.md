## Context

See proposal.md for motivation and the four delta specs for behavior. OpenSpec resolves one central root at `/data/CoordExp`; nested Pi repos do not have separate planning stores. Pi Web is heavily dirty with independent TPS/subagent/runtime work, including statistics, RPC and locales. Compact source is currently clean relative to its own HEAD; unrelated monorepo tests/subagent files are dirty. No resets, broad formatting, commits or live-user reloads are part of this change.

Original journal evidence is frozen at Luna lines1-318 and Sol lines1-176, approximately2026-10-06T17:36:40Z: ordinary90.9519/93.7873%, pooled86.9868/89.5958%. The current statistics deliberately include maintenance. The cache parser maps both absent cached_tokens and zero to normalized zero. Existing network records lack final-body/session attribution.

Installed SDK1.0.3 public APIs expose `session.agent.streamFunction`, request-local `fetch`, raw provider stream callbacks and originating session IDs. HTTP fetch does not cover WebSocket dispatch. Compact already has an injected fetch wrapper downstream of its payload expansion. `pi.getSettings()` has no core cache-retention preference; cache warming off is not cache disabling. Shared-memory recall is ephemeral and must not be accessed through private state or frozen for reuse.

## Goals / Non-Goals

**Goals:** reduce diagnostic input size; distinguish usage populations without changing totals; capture bounded HTTP request/usage evidence without semantic changes; experimentally preserve compact affinity and qualify prefix transitions.

**Non-Goals:** change generic read output, scientific manifests/claims, opaque checkpoint semantics, normal CLI instrumentation, proxy behavior, credentials, official Pi core or managed node_modules. Do not promise90% cache hits. Do not invent an inherited retention setting or invoke another extension's private hooks. Scientific ledger selection is caller-owned structural projection, not a new scientific validator. Unsupported numeric representations must be rejected visibly before a rounded value is emitted; compare normalized JSON numeric lexemes with the emitted representation, including unsafe integers and long decimal adversarial fixtures.

## Decisions

### 1. Keep one explicit diagnostic consumer, not a hidden output filter

Implement a local CLI and reusable small projection helper in Pi Web, beside its current journal/stat consumers. Journal mode incrementally parses raw JSONL without opening it through a migration-capable SessionManager; it projects only requested metadata/usage, tool calls or bounded text. Reuse existing accounting/extraction where practical, but do not silently reuse a loader that discards malformed lines or rewrites legacy files. Explicit finite line range supports growing journals; malformed records and partial final lines produce disclosed incomplete coverage. Input byte limits are admission checks, separate from output truncation. JSON mode requires explicit JSON Pointer selectors, reports collection/text omissions and retains exact JSON primitive values. Missing/invalid selectors fail visibly. Preserve raw evidence and built-in tools.

Alternative rejected: repeated raw read pagination, which includes signatures and nested tool output; or a global read redaction hook, which breaks exact/scientific consumers. Research Probes files are test inputs only; no changes to active research source or ledgers.

### 2. Add categorized statistics to the existing lifetime accumulator

Keep the existing `tokens` and `cost` fields unchanged and add a disjoint breakdown: inference, compaction, branchSummary, cacheWarm, other. Unknown `usage.kind` and nested/tool-result usage go to the explicit other bucket, not dropped and not assumed to be model inference. Other remains counted in all totals; this is an accounting classification, not rejection of unknown kinds. Token amounts reconcile with exact integer equality; monetary regrouping is compared at a documented USD1e-9 tolerance to accommodate floating accumulation order, while the existing chronological total remains authoritative. Mirror production/demo, structural SDK-facing types and incremental merge; ensure component memo keys see the breakdown. Ordinary ratio uses only inference; maintenance means compaction+branchSummary+cacheWarm, while other remains separately visible.

Use the current API/RPC stats paths, preserving TPS fields and existing all-entry accounting. Alternative rejected: change the pooled denominator or recompute only active compacted context.

### 3. Observe the final HTTP dispatch, not an early hook

Web installs a request-local wrapper around the public captured `agent.streamFunction`, preserving all options/callbacks and composing a fetch observer immediately before the underlying fetch only for qualified built-in OpenAI Responses HTTP adapters. Unsupported adapters (including Google/Vertex which reject injected fetch) remain untouched and explicitly unobserved. Never globally monkeypatch fetch or provider objects. A supplied transforming fetch is not qualified as final-wire coverage: delegate unchanged and mark coverage unknown unless a local test establishes the true boundary. Allocate HTTP attempt IDs at each actual dispatch, separate from logical operation, originating session and wrapper generation; captured warming callbacks, native summaries and retries must not reuse ordinary-request identity. Ambiguous operation/completion attribution is unknown, not inferred ordinary. Preserve materialized fetch arguments and return the original Response without consuming its body. Bodies requiring stream reads are marked unobserved; do not force SSE or alter transport for coverage.

The host diagnostic implementation owns fingerprinting and the bounded sink. Compact forwards opt-in transport/terminal-event observations over a documented, versioned extension-neutral public event protocol; request evidence is an immutable body string, and terminal/usage observations are detached allowlisted primitive data, never a mutable raw provider event or mutable request arguments. Include absent/zero/invalid cache-field states. Catch observation errors without logging exception text. The observer's privacy guarantee covers only its own records, not other fully privileged extensions; the Web host listens by matching originating session and attempt, without importing compact private source. Standalone compact remains functional with no observer participant. Only materialized request bodies may be supplied transiently to this privileged local callback; records contain no raw body, credentials, arbitrary headers, signatures or error text. This public protocol is observational, not a mechanism for context preparation or affinity policy.

Normal Web observation is enabled by an explicit host diagnostic switch; compact observation uses its own global-only `requestDiagnostics` boolean defaultfalse, handled by the existing settings runtime/menu. The protocol and enablement must be documented; terminal ordinary requests remain out of scope. The observer uses a random per-process HMAC key and explicit comparison epoch, bounded section/input fingerprints and finite in-memory session/attempt tracking. Parsed request admission is at most8MiB; oversized/non-string bodies get coverage markers without parsing. Diagnostic sink is private (0600 files/0700 directory) under Pi Web `.local/diagnostics/`, uses bounded rotation (at most two1MiB files) and never writes session journals. Records are at most16KiB; pending writes are capped at128 records and256KiB, with explicit drop/coverage counters; retain at most64 session tracking slots and128 attempts, with eviction disclosed. Never queue a raw body or event. Release observers on wrapper replacement/disposal and test late callbacks/warming overlap. Writes and observation failure cannot prevent dispatch. Raw usage capture records cached-field presence separately from value and correlates completion/error/abort with a request attempt without creating normalized zeros for absent fields.

Alternative rejected: onPayload-only evidence (can precede other hooks); global transport interception (mutates unrelated consumers); raw prompt dumps; unkeyed low-entropy hashes; extension-private cross-imports; an additional independent diagnostics package.

### 4. Change compact affinity only under an experimental opt-in

Add validated global-only `preserveCacheAffinity:false` through the existing settings implementation, menu and README. Default-off must produce the existing compact request. When true and protocol=context-management, forward public originating sessionId and omit the forced cacheRetention:none so the provider chooses its current short/long/default compatibility branch from resolved environment/model/auth. Other protocols remain unchanged. Explicitfalse is the supported opt-out; do not invent `ctx.settingsManager`, a core retention property, or treat `PI_CACHE_RETENTION=none` as an SDK disable switch.

Preserve authoritative system and ordered tool definitions, checkpoint expansion, maintenance-only suffix, storefalse and tool_choicenone. Use normalized final-request contrasts to identify duplicate declarations or canonical-prefix changes; fix proven redundant preparation only inside the opt-in context-management path, leaving default-off request equivalence intact. Qualify persisted-system, forced effective prompt, legacy history, ordered tool deltas and existing checkpoints against installed SDK normalization; do not blindly drop every system message or the effective prompt. Unsupported fidelity is reported rather than guessed. Do not make a private dependency on shared-memory to align ephemeral recall. Affinity can improve routing without reproducing the ordinary full prefix; report the first divergence and unchanged prefix regions explicitly.

Alternative rejected: unconditional cache-setting change, hardcoded long TTL, freezing recalled memory, switching backend/protocol, or forcing a plaintext summary.

## Risks / Trade-offs

- Final HTTP observation is not universal wire coverage → mark WebSocket/non-materialized paths unobserved and never use earlier snapshots as final evidence.
- Public observational events could be mishandled by privileged participants → version the contract, make source payloads read-only/ephemeral, isolate listener/sink failures and test request equality and absent participants.
- HMAC record arrays/logging can grow or stall → cap input/body/records/session maps and rotate private output; no await of durable disk writes in streaming callbacks.
- Category merges and TPS edits overlap → one writer per affected file during each batch, exact targeted edits and reload live source before editing; preserve current unrelated changes.
- Affinity does not guarantee cache eligibility or backend reports → deterministic request fidelity first, then a bounded hosted contrast; never claim cause from normalized usage alone.
- Runtime package source differs from active packaged release → qualify generated runtime through installed SDK; source acceptance is not production adoption.

## Migration Plan

1. Complete artifacts and bounded ask-advisor review, resolve specific blockers, then apply under the user's explicit implementation instruction using the fast-forward/apply workflow.
2. Wave1: offline tests and fixed historical projections; verify totals and input reduction. Lead accepts before wave2.
3. Wave2: diagnostic code/protocol and default-off/HTTP-loopback SDK tests, privacy and lifecycle falsification. Lead accepts before wave3.
4. Wave3: settings/affinity candidate, final-payload contrasts and installedSDK checkpoint/continuation regressions, generated runtime and package checks.
5. Hosted qualification is isolated synthetic Luna history, never a user's session: at most8 model endpoint requests (baseline/candidate each two ordinary requests, onecompact and oneresume), no automatic retry/warming, no tool/GPU jobs, at most15minutes and estimated cost capUSD2 using declared catalog pricing. A dispatch-level guard counts all actual model endpoint calls, including fallback/native summary, and prevents dispatch after the first external failure or exhausted bound; first-compaction failure must be falsified locally to show fallback cannot escape the guard. Stop after the first external/entitlement failure; no rerun or alternate protocol. ChatGPT sign-in may omit max_output_tokens, so do not treat model.maxTokens as a billed-output bound. Preflight the maximum declared exposure using actually emitted controls/context window/pricing; HOLD hosted execution if USD2 cannot be justified. Bind effective options and synthetic inputs in receipt. A failure produces HOLD on hosted benefit, not a fabricated success.
6. Keep raw receipts under the owning repo `.local/diagnostics/` or `.local/qualification/`; retain concise human acceptance at this change and package docs. No root outputs.
7. No commit, publication, versioned release installation, shared-profile mutation or active-session reload without separate adoption authority. Opt-in defaults preserve current production behavior; disabling the new options restores old behavior without changing historical checkpoints.

## Open Questions

Only backend-observed cached-token reporting and attainable cache benefit remain empirical. These do not change the deterministic request/checkpoint contracts; the hosted receipt can leave them explicitly unverified.
