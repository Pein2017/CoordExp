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

## Multi-agent orchestration

Spawn contract:

- Every spawn sets `fork_turns`, `model`, and `reasoning_effort` explicitly.
  Builders and reviewers use `fork_turns: "none"`; `"all"` is forbidden once
  session context is non-trivial (>~50k tokens); design advisors fork at most
  the last 5 turns.
- One task = one fresh, narrow spawn. No cross-task follow-up chains on a
  worker thread. Hierarchy depth 2 (lead → workers) by default; add a third
  cheap layer only for genuinely parallel bounded subtasks.
- Every brief states: frozen goal + non-goals, authoritative constants, exact
  acceptance commands, output contract, known failure modes, budget (max
  spawns, max review rounds, token/time), and tier (`probe` | `production`).

Progress gating (checkpoint, not poll):

- Workers end their turn at defined checkpoints (plan, first diff, RED test,
  GREEN) and hand back a deliverable; the lead reviews and corrects course
  there. Never poll a running worker for status.
- The lead never accepts a worker's self-report. Acceptance requires
  independently verifiable receipts: command replay, regression-count diff,
  file mtimes, artifact paths.
- Waiting: one `wait_agent` sized to the expected duration (config allows
  3600s); batch multiple targets into one wait; no narration between waits.

Review cadence:

- Task-level review sol/high; milestone/mechanism review xhigh; the strongest
  adviser only at claim/launch/irreversible gates.
- One bundled correction round per task. A third same-class finding = stop
  and fix the shared invariant, or escalate — never a fourth review round.
- Probe-tier work: single review round, no anti-forgery hardening.

Decision authority (standing):

- Ask the user: research meaning or direction, claim scope, stop rules,
  material cost (GPU launches, paid models), irreversible or outward-facing
  actions, architecture changes, publication.
- Decide autonomously and log: implementation details, reversible refactors,
  test/tooling choices, worker routing, retries.
- If blocked >30 min on a decision not on the ask-user list, take the
  reversible option, record it, continue. Never idle-wait on the user.

Durability:

- Frozen decisions (grill-me outcomes, contracts, recommendations) live in
  files, not conversation context; compaction has flipped an in-context
  recommendation before.

Model routing priors (verify live availability; effort is search depth, not a
fix for role mismatch — on semantic or architectural uncertainty change model
family, do not escalate effort):

- Read-only scout: luna/medium (haiku is a provider-diverse peer; neither owns
  writes, review, or conclusions).
- Bounded builder: terra/high or sonnet/high; sonnet/medium for small explicit
  work with a deterministic verifier.
- Semantic builder (math, autograd, research semantics, silent correctness):
  sol/high; opus/medium-high as peer.
- Lifecycle builder (compatibility, serialization, source archaeology):
  opus/high.
- Semantic review: sol/xhigh; lifecycle review: opus/xhigh. Review requires a
  frozen target; target drift invalidates findings.
- Major decisions: sol/max, opus or fable xhigh/max advise only; lead/user
  retain authority. Never run two writers on one semantic surface.
- Optimize time to final acceptance = builder latency + correction + review +
  runtime wait + lead intervention; spend is a tie-breaker.

## Runtime

- Run Python through the default `ms` Conda environment:
  `conda run -n ms <command>`, unless the named artifact requires another runtime.
- Shared GPU activity is expected and normally reusable; adapt only after a
  concrete OOM or operational conflict.
- Dirty changes are expected; inspect their ownership before treating them as
  task work.
- For non-interactive long-running asynchronous work, use `yield_time_ms >=
  180000` for empty `write_stdin` polls and `functions.wait`; prefer `300000`
  when intermediate output is unnecessary.
- Do not use a short poll merely to report that work is still running. In
  `functions.exec`, set outer `@exec yield_time_ms` at least 30000 ms longer
  than its longest nested wait. Completion returns early; exempt only
  non-empty interactive `write_stdin` calls from this rule. `wait_agent` is
  NOT exempt: it follows the orchestration waiting rule — one wait sized to
  the expected duration, batched targets, no narration between waits.

## Records

- Put active investigations, interpretation, negative results, and continuation
  context in `research/`; raw provenance belongs in `docs/history/`; create no
  new `progress/` records.
