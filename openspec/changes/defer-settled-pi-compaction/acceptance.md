# Deferred completed-answer compaction

The requested policy keeps a normally completed answer final, then compresses
before the next prompt's normal model request. Manual compaction executes
immediately and waits; active-task threshold compaction and overflow recovery
retain the existing continuation. This aligns the relevant Codex default timing,
not every optional Codex policy or threshold.

## Source and consumer qualification

The local fork adds `deferPostAnswerCompaction` to its existing settings owner,
defaulting false and intended to be enabled in the shared CLI/Web profile. Public
`agent_end`, `input` and `before_agent_start` distinguish completed-run maintenance
from next-prompt preflight; no Pi core or Web scheduling patch is used.

Managed SDK: Pi 1.0.3 under `codex-tools/pi-core/.local/install/releases/1.0.3`.
The real SDK consumes fixture Responses SSE bytes through its built-in adapter;
all requests are intercepted and constrained to the localhost fixture endpoint.

- RED: `.local/defer-settled-sdk103-red.log` in `codex-tools/pi-extensions`
  shows completed-answer and reopened-session assertions reject the old behavior.
  It also exposes the old overflow replay mismatch.
- Source GREEN: `.local/defer-settled-sdk103-green-5.log` records 41 passing
  SDK, checkpoint, settings and menu tests before the additional queued-follow-up
  and eligibility coverage.
- Generated consumer: `.local/defer-settled-sdk103-built.log` records all ten
  actual-SDK cases passing through packaged `dist/index.ts` with Pi's Jiti loader.
  Cases cover active continuation, manual idle behavior, opt-out maintenance,
  deferral, JSONL reopen, queued follow-up, overflow retry, steering, abort and
  persisted checkpoint replay. Tool effects execute once.
- Repository checks: `.local/defer-settled-check-green.log` records successful
  build, Biome, extension boundaries and workspace typechecks.
- Full repository suite: `.local/defer-settled-test.log` reports 6,414 passing
  and four failing tests, confined to the previously reproduced unchanged
  Starship, Subagents, Typesafe Compact and runtime-builder failures. All compact
  plugin tests pass; those unrelated modules remain unqualified here.
- Pack dry run: `.local/defer-settled-pack-dry-run.json` inventories the candidate
  package, generated entrypoint and lazy settings-menu chunk.

Preflight and overflow qualification exposed raw-entry replay fingerprints that
included Pi-managed system declarations and omitted failed attempts. The producer
now uses Pi's public session projection; the reader preserves system declarations
and matches retained conversation strictly. Changed conversation and missing
legacy fingerprints still fail closed. Native continuation and JSONL replay tests
verify the opaque checkpoint reaches the actual following request.

## Adoption boundary

Candidate version: `0.55.0-coordexp.2`. The canonical local release command requires
clean committed source and never overwrites a release. The shared profile is
`/data/CoordExp/.pi`; active Web sessions must finish before their extension reload.

The user authorized a local commit and install. Plugin source is committed at
`5b026a006b6b465d5089daf77dd93e12c072e3bf` on local `main`; its worktree is clean,
with no push, tag, registry publication or Pi core change. The immutable local
release records that commit and upstream
`87f29eaff190925d682927c5ff9b6e042e661bbb` in
`codex-tools/pi-extensions/.local/releases/0.55.0-coordexp.2/release.json`.
The installed package's ten managed-SDK/Jiti regressions pass again in
`.local/defer-settled-sdk103-installed.log`.

The existing plugin settings runtime and public Pi SettingsManager publish the
flag and package selection, preserving every unrelated JSON field. Private prior
configuration and adoption receipts are under
`codex-tools/pi-extensions/.local/qualification/defer-settled-install/`.
The shared profile selects only the new immutable package, remains enabled on
`context-management` with its previous retry settings, and sets
`deferPostAnswerCompaction: true`.

An ephemeral CLI RPC startup discovers `codex-compact` from that package's
`dist/index.ts`, submits no prompt and exits zero. The requested Web session
`01a10ce3-4348-7204-ab11-5e1af9835c40` was verified idle with no queued messages,
reloaded using public `reload`, and discovers the same release path. Its entire
existing JSONL is byte-identical across that idle reload. A newly created Web
session `01a11249-e876-728e-864a-e4f7671c1c72` also uses the new release.

Two already running Web sessions, `01a1120f-726d-728e-864a-e4f222389428` and
`01a111fc-4f11-728e-864a-e4f0ca7fa36c`, retain their loaded previous release.
They were observed without reload or interruption; once idle, `/reload` adopts
the shared selection. Default installation and the requested session's adoption
are complete; no claim is made that an active run changes its loaded plugin.

No paid request was added. The prior hosted continuation attempt remains HOLD;
local SDK regressions establish scheduling and transport plumbing, not hosted
checkpoint recall or provider reliability.
