# Agent Guide - CoordExp

> Canonical, agent-neutral guidance for `/data/CoordExp`. A linked worktree may
> add a small `AGENTS.override.md` for stable local differences, never a copy of
> this base.

## Authority

- Shared agent configuration lives under `/data/CoordExp/.codex`; do not depend
  on `/root` state.
- When current ownership is unclear, use `docs/AGENT_INDEX.md` to find the
  smallest current owner.
- Give each decision and write surface one current owner. Reviews, memories,
  handoffs, and historical worktrees are evidence, not authority.
- The user owns research meaning, compatibility, publication, material cost,
  irreversible behavior, and changes to architecture, claim scope, or stop rule.
  Agents own discoverable facts and reversible implementation details.

## Research

- Keep data, geometry, order, prompts, tokens, objectives, metrics, artifacts,
  and claims semantically aligned.
- Before a costly launch or research implementation, state the question, contrast,
  decision-owning outcome, strongest alternative, primary evidence, and stop rule.
- Bind experiment claims to their config, checkpoint, artifact root, counters,
  metrics, and evidence scope.
- Use the shortest evidence path for exploration: a plumbing smoke is not
  model-quality evidence. Decision-bearing or cited results must be reproducible;
  do not overbuild exploratory probes.

## Runtime

- Run Python through the default `ms` Conda environment:
  `conda run -n ms <command>`, unless the named artifact requires another runtime.
- Shared GPU activity is expected and normally reusable; adapt only after a
  concrete OOM or operational conflict.
- Dirty changes are expected; inspect their ownership before treating them as
  task work.

## Records

- Put active investigations, interpretation, negative results, and continuation
  context in `research/`; raw provenance belongs in `docs/history/`; create no
  new `progress/` records.
