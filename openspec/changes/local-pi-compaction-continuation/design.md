## Context

See proposal.md for motivation. Pi 1.0.3 owns the execution loop; `context-management` is the qualified provider protocol. The plugin already persists v3 opaque checkpoints and the completed maintenance suffix. The current session can resume from that checkpoint after a fresh user instruction.

## Goals / Non-Goals

Goals: repair visible continuation ordering using existing hooks and replay, then activate a locally built release. Non-goals: new auth, provider/SDK changes, an unconditional continuation loop, publication, or interrupting user's active sessions.

## Decisions

- Source owner: independent Git checkout `codex-tools/pi-extensions`; preserve upstream monorepo/history and a fetch remote. Package owner remains `packages/pi-codex-compact`. Local documentation and release helper own deployment.
- Append one exact deterministic user continuation item after the validated completed maintenance suffix for non-manual compaction with a live `ctx.signal`, or overflow with `willRetry`. Persist it inside v3 replacement history. Newer real input follows it naturally. This avoids extra turns and transient request-prefix changes.
- Keep provider output validation strict. Only the persisted-history reader additionally recognizes the exact final continuation item; reject arbitrary user suffixes and preserve old v3 history.
- Runtime consumes a versioned unpacked local tarball, not editable `src/`. The release records source identity and tarball digest; local package installation replaces the npm registration only after load verification. Existing active wrappers retain their loaded extension until safe reload.

## Risks / Trade-offs

- Opaque semantic recall is not proven by HTTP success → qualify a bounded hosted task separately from deterministic SDK ordering.
- Repeated side effects or extra turns → real SDK tool-counter, manual/settled, cancellation and newer-input regressions, with no `sendUserMessage`/`triggerTurn` calls.
- Release dependency/loading drift → generated-entry loader smoke and installed SDK check; retain upstream authentication and checkpoint recovery tests.

## Applicable guides and verification

Read `docs/extension-conventions.md`, `docs/extension-settings.md`, `docs/readme-conventions.md` in the fork. Touched MUST areas: deterministic/idempotent context and stable prefix (exact replay tests), cancellation/ownership (SDK regressions), strict persistence boundary (malformed/history tests), package loading/build/contents (Jiti and pack smokes), and repository gates (`npm run check`, `npm test`). No command/menu/settings interface changes. Local release helper is tested for versioned packaging and refusal to overwrite an existing release.
