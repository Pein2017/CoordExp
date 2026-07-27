# Dynamic Model Routing

> Current empirical guidance for lead agents. This is not stable repository
> policy and is not consumed directly by workers.

## Audience And Scope

- Codex leads and leads started through `CC:spawn` may consult and update this
  file when substantial delegation, conclusion-critical work, or an uncertain
  route makes the choice material.
- A lead passes the selected surface, model, effort, goal, and stop boundary in
  the worker prompt. Spawned workers do not read this file.
- Simple local work does not need this document or delegation.

## Context Handoff

- Set `fork_turns: "none"` explicitly on every Codex `spawn_agent` call and on
  every `CC:spawn` interface that exposes it. Never omit the argument or choose a
  conversation-history window.
- The lead writes a self-contained brief with the goal, exact cwd and owned
  paths, relevant current evidence, non-negotiable constraints, expected output,
  completion condition, and stop boundary.
- If the required context cannot be distilled without changing its meaning,
  keep the work with the lead instead of forwarding raw history.

## Current Priors

- Candidate Codex worker models include Terra and Sol. Candidate Claude Code
  worker models include Sonnet and Opus. Both surfaces may expose multiple
  reasoning-effort levels; verify live availability before dispatch.
- There is no fixed role-to-model table or reliable benchmark yet. Choose the
  delegation surface first from required tools, runtime access, permissions,
  subscription, and isolation, then choose model and effort from task
  verifiability, complexity, and consequence.
- Prefer the least costly route that can reliably finish the bounded lane, and
  increase capability or effort when evidence, failure, or consequence warrants
  it.

## Maintenance

- The lead may revise these priors in concise natural language when real work
  changes its judgment.
- Keep only current recommendations and observations that still distinguish
  routes. Delete stale or absorbed observations instead of building an
  append-only history.
- Do not require a benchmark, profiling pass, per-task ledger, or formal routing
  matrix before exercising judgment.
