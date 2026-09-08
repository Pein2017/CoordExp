# CoordExp Project Contract

> User-wide orchestration is loaded first from
> `/data/CoordExp/.codex/AGENTS.md`. This file adds CoordExp-specific rules for
> this tree; a nested `AGENTS.md` specializes them, while a sibling
> `AGENTS.override.md` replaces that directory's `AGENTS.md`.

## Authority

- Shared agent configuration lives under `/data/CoordExp/.codex`; do not depend
  on `/root` state.
- When ownership is unclear, use `/data/CoordExp/docs/AGENT_INDEX.md`; give each
  decision and write surface one current owner. Reviews, memories, handoffs,
  and historical worktrees are evidence, not authority.
- The user also owns research meaning and compatibility.

## Research

- Act as an independent research collaborator: question decision-relevant
  assumptions, disagree when evidence warrants it, and propose falsifiable
  alternative explanations ranked by the cheapest discriminating evidence.
- Separate observation, hypothesis, inference, and speculation. Investigate
  autonomously within the agreed scope without changing research meaning,
  architecture, material cost, claims, or stop rules; converge when the
  question is answered rather than opening an unbounded research program.
- Keep data, geometry, order, prompts, tokens, objectives, metrics, artifacts,
  and claims aligned.
- Before costly research or launch, state the question, contrast,
  decision-owning outcome, strongest alternative, primary evidence, and stop
  rule. Bind claims to config, checkpoint, artifact root, counters, metrics,
  and evidence scope.
- Use the shortest evidence path: plumbing smoke is not model-quality evidence,
  decision-bearing or cited results must be reproducible, and exploratory
  probes should not be overbuilt.

## Development

- OpenSpec owns scope, research meaning, compatibility, and completion for its
  changes. Superpowers is an on-demand execution toolbox; its artifacts link to
  rather than restate the owning change.
- Here, test-first is required only for faults, fail-closed paths, bug fixes,
  and frozen-contract refactors: reproduce bugs first and characterize frozen
  contracts first. Prefer fail-fast validation for probes, glue, config, and
  fast-failure code; the user-wide load-bearing-test requirement still applies.
- Guard silent-corruption surfaces such as masking, supervision positions,
  loss accounting, and parity with invariants and golden fixtures; probe-tier
  work gets no RED/GREEN ceremony.
- Before broad implementation or costly launch, retire conclusion-changing
  execution risks with the smallest production-shaped vertical slice; leaf
  tests and mocks do not close real-entry, distributed, scale, persistence,
  finalization, or downstream-consumer risk.
- Before scaling data, ranks, or GPUs, declare and measure relevant bounds for
  model forwards, collective order, cache or materialization passes, wall time,
  RSS, workers, artifact payload, and finalizer or consumer behavior.

## Runtime

- Follow the user-wide default `ms` Conda execution policy unless the named
  artifact explicitly requires another runtime.
- Shared GPU activity and dirty changes are expected; adapt only after a
  concrete OOM or operational conflict, and preserve unrelated work.

## Records

- Put active investigations, interpretation, negative results, and continuation
  context in `research/`; raw provenance belongs in `docs/history/`; create no
  new `progress/` records.
