# Agent Guide - CoordExp

> Canonical, agent-neutral guidance for work under `/data/CoordExp`. This is the
> only shared policy source. An active linked worktree may add one small
> `AGENTS.override.md` containing stable local differences, never a copy of this
> base.

## Grounding And Authority

- Start from the exact path, worktree, artifact, config, spec, diff, run, or
  question the user names. Inspect it before explaining or changing it, and
  revalidate before crossing roots.
- Shared agent configuration lives under `/data/CoordExp/.codex`; workflows must
  not depend on `/root` state.
- Give each decision and write surface one current owner. Reviews, memories,
  handoffs, and historical worktrees are evidence, not live authority.
- The user owns choices that change research meaning, compatibility,
  publication, material cost, or irreversible behavior. Agents own discoverable
  facts and reversible implementation details; ask only when that boundary is
  unresolved.
- When an approved route changes architecture, claim scope, material cost, or
  stop rule, restate old versus new and obtain the owning decision before
  proceeding.

## Research Judgment

- Accuracy and precision come first. Keep data, geometry, order, prompts,
  tokens, objectives, metrics, artifacts, and claims semantically aligned.
- Before a costly launch or research implementation, make the question,
  contrast, decision-owning outcome, strongest alternative, primary evidence,
  and stop rule explicit.
- An exploratory slice should follow the shortest evidence-bearing path to the
  primary observation; keep optional optimization, hardening, and future
  comparators detached. A plumbing smoke is not model-quality evidence.
- Decision-bearing or cited results must be reproducible; exploratory probes
  should not be overbuilt for reproducibility.
- Treat a reusable interface as provisional until stable semantics or a second
  real consumer establishes the seam.

## Lead And Workers

- The lead agent owns decomposition, global context, cross-lane decisions,
  research interpretation, synthesis, and final acceptance. Delegate bounded
  implementation, investigation, or review lanes only when parallelism or
  attention isolation materially helps; keep simple work local.
- Distill every delegation into a self-contained brief with exact ownership,
  permissions, relevant evidence, completion conditions, and stop boundaries;
  do not forward unfiltered conversation history. Keep write surfaces
  non-overlapping and reconcile every result in the lead.
- The lead alone owns model and effort routing. Use the available
  `agent-routing` skill for substantial delegation, conclusion-critical work,
  or an uncertain route; workers receive the chosen route in their prompt.

## Runtime And Parallel Work

- Python normally runs in the `ms` conda environment.
- GPU use is shared by default. Check live processes, utilization, and memory;
  co-locate only with safe headroom and acceptable interference, use explicit
  device placement, and never kill or evict unrelated processes. Coordinate
  timing-sensitive or near-full-cluster jobs.
- Multiple agents may modify one worktree concurrently. Dirty or unfamiliar
  changes are expected: inspect and preserve them, and never revert, overwrite,
  stage, or commit work outside the current task.

## Engineering Taste

- Fail fast on invalid, unknown, retired, or semantically incompatible inputs
  before expensive work. Do not hide them behind fallback defaults.
- Do not over-design, over-engineer, over-optimize, or over-audit. Stop when the
  requested outcome and proportionate evidence are satisfied.
- Minimize semantic surface and total cognitive load; use raw lines of code only
  as a tie-breaker between otherwise equal designs.
- Prefer code, names, types, and tests that explain the mechanism. Comments
  explain why, invariants, units, ordering, or non-obvious constraints rather
  than narrating the code.
- Reuse stable semantics, not imagined variation. Small local duplication is
  preferable to a premature abstraction.
- Performance work targets time-to-primary-observation and end-to-end
  wall-clock, including GPU idle time and CPU preparation or admission. Profile
  the measured critical path before adding parallelism or infrastructure.
- Remove temporary scaffolding and smoke artifacts created by the current task.
  Delete dead branches or superseded shims caused by the change without turning
  cleanup into a redesign or removing evidence owned elsewhere.

## Durable Owners

- Follow the user's language in conversation. Keep code, paths, commands,
  configs, schemas, formulas, experiment identifiers, plans, reviews, and
  handoffs in English unless requested otherwise.
- Put new investigations, interpretations, negative results, and continuation
  context in `research/`; raw provenance belongs in `docs/history/`; create no
  new `progress/` records.
- Stable cross-worktree skills live in `$CODEX_HOME/skills`; worktree-local
  skills are explicit experimental deltas. Preserve official and vendor-managed
  ownership.
- When named evidence does not reveal the current owner, consult one narrow
  route from `docs/catalog.yaml` or `docs/AGENT_INDEX.md`, not both by default.
