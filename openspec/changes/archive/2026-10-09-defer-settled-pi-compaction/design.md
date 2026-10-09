## Context

Pi 1.0.3 checks threshold compaction after a completed run, during prompt preflight,
and before the next assistant request. The public extension context exposes an
active run signal only for the request-time path. The shared CLI/Web profile uses
the immutable local `pi-codex-compact` package; Web forwards SDK lifecycle events.

## Goals / Non-Goals

**Goals:** Defer completed-answer threshold maintenance to the next request using
Pi's existing checks, with the current remote checkpoint backend and continuation.

**Non-Goals:** Patch Pi internals, add a Web scheduler, disable threshold checks,
change overflow recovery, or infer task completion from an assistant's prose.

## Decisions

- Add the explicit `deferPostAnswerCompaction` boolean to the plugin's existing
  settings owner, default false for existing installations, enabled in our shared
  profile. Preserve the existing loader, validation, atomic updates and menu.
- Track a completed run at public `agent_end`, clearing that session-owned marker
  at `input` and `before_agent_start`. Cancel threshold interception only in that
  completed-run phase with no active signal and a normally stopped response.
  The next prompt's preflight remains enabled and compacts before its user text is
  recorded or sampled. Reopened sessions need no persisted scheduling state.
- Pi's request-time check runs between assistant responses, not before the initial
  response to a new prompt. A signal-only interception cancelled preflight too;
  the real SDK regression rejected that implementation.
- Native preflight/overflow tests exposed retained-message mismatches: raw entries
  included system declarations moved by the SDK and responses omitted by context
  edits. Generate replay fingerprints from Pi's public session projection, match
  conversation messages strictly and preserve Pi-owned system declarations.
  Missing legacy fingerprints still fail closed; do not invent missing history.
- Keep manual, active-task, error/cancellation and overflow decisions unchanged.
  Qualify actual source and generated release consumers with the managed Pi 1.0.3
  SDK and local Responses SSE bytes; no paid endpoint is used.
- Adopt a new immutable local package through the existing release command and
  shared profile. Reload Web extensions only while the affected sessions are idle.
  The release command requires reviewed committed source; do not bypass that guard.

## Risks / Trade-offs

- Public signal/request timing is SDK-specific → reproduce final-answer deferral,
  next-prompt ordering, reopen, manual, active and overflow cases on installed Pi.
- Pi emits start/cancel events for intercepted maintenance → no remote request or
  checkpoint is created; cancellation must not appear as an error or restart work.
- A newly submitted prompt waits for compression → intentional scheduling trade-off;
  its text and tools must reach the first normal request after compression.
- Active Web wrappers retain the previous plugin → preserve their work and verify
  idle reload adoption rather than restarting active runs.
