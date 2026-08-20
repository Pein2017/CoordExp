# CoordExp Project Contract

> User-wide orchestration is loaded first from
> `/data/CoordExp/.codex/AGENTS.md`. This file owns only CoordExp-specific rules
> for the current Git root. Each maintained linked worktree carries this full
> project contract and folds stable local differences into its own root
> `AGENTS.md`; a sibling `AGENTS.override.md` would replace, not append to, it.

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

## Development

- When Superpowers executes an OpenSpec-owned change, OpenSpec remains the sole
  authority for scope, research meaning, compatibility, and completion.
  Superpowers artifacts are execution-only and must link rather than restate
  the owning change.
- Superpowers is an on-demand toolbox, never a default posture: invoke a
  skill when the current task needs it. Its planning, worktree, and
  subagent-driving skills stay in use; no skill imposes a workflow the task
  did not ask for.
- Test-first is required only for fault or fail-closed paths, bug fixes
  (reproduction test first), and frozen-contract refactors (characterization
  first). Elsewhere prefer fail-fast runtime validation over test-first
  ritual; exploratory probes, glue, config, and fast-failure code never get
  ceremony. A load-bearing test, whenever written, must have been observed
  to fail for the right reason at least once (RED or a demonstrated
  sensitivity check); a green-only test is unverified evidence.
- Silent-corruption surfaces (masking, supervision positions, loss
  accounting, parity) are guarded by invariant assertions and golden
  fixtures rather than unit TDD; probe-tier work gets no RED/GREEN
  ceremony, matching its single-review-round budget.
- Before broad implementation or a costly launch, retire the smallest set of
  conclusion-changing execution risks with a production-shaped vertical slice.
  Leaf tests and mocks do not close real-entry, distributed, scale, persistence,
  finalization, or downstream-consumer risk.
- Before scaling data, ranks, or GPUs, declare and measure the relevant bounds
  for model forwards, collective order, cache or materialization passes, wall
  time, RSS, workers, artifact payload, and finalizer or consumer behavior.
- At design-to-implementation and fixed-implementation-to-launch or recovery
  boundaries, prefer a fresh task with a compact handoff when inherited
  conversation state is no longer required.

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
