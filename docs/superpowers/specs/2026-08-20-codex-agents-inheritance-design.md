# Codex AGENTS Inheritance and Adaptive Topology Design

**Status:** Implemented and committed in the main checkout and three maintained
worktrees.

## Goal

Make Codex load one compact user-level orchestration contract plus one honest
project/worktree contract, then remove instruction files whose real discovery
semantics shadow or duplicate those contracts.

## Confirmed Codex semantics

- `CODEX_HOME` is `/data/CoordExp/.codex` in the active environment.
- Global discovery reads `AGENTS.override.md` first and otherwise `AGENTS.md`,
  using only one non-empty global file.
- Project discovery walks from the nearest Git root to the current working
  directory. At each directory it selects at most one file in this order:
  `AGENTS.override.md`, `AGENTS.md`, configured fallbacks.
- Therefore `AGENTS.override.md` replaces, rather than appends to, a sibling
  `AGENTS.md`.
- Instructions are assembled once per Codex run/session. A fresh session is
  required to observe a changed chain.

## Instruction ownership

### User-level owner: `/data/CoordExp/.codex/AGENTS.md`

Own only reusable behavior that should apply across repositories opened with
this Codex profile:

- authorization and reversible-action boundaries;
- adaptive topology selection;
- lead, worker, and reviewer authority;
- spawn briefs and context inheritance;
- checkpoint states, 60-minute default `wait_agent` joins, acceptance, and
  review cadence;
- durable handoffs and context-lifetime boundaries;
- worktree safety and time-to-final-acceptance optimization.

It must not contain CoordExp research semantics, Conda commands, GPU policy,
artifact locations, OpenSpec ownership, or record-placement rules.

### Project owner: each Git root `AGENTS.md`

Own only CoordExp-specific behavior:

- current-owner routing through `docs/AGENT_INDEX.md`;
- research meaning and evidence boundaries;
- CoordExp development, OpenSpec, test, vertical-slice, and scale rules;
- `conda run -n ms`, shared-GPU, dirty-tree, and artifact rules;
- `research/`, `docs/history/`, and `progress/` placement.

Every linked worktree is a separate Git root, so its root `AGENTS.md` must
contain the complete CoordExp project contract. This migration maintains only
`CoordExp-swift`, `research-probe-infras`, and `research-probes`; all other
worktrees are temporary and explicitly out of scope. Worktree-local additions
are folded into the same root file when present.

### Nested owner: directory-local `AGENTS.md`

Use a nested file only when its rules apply to that directory and descendants.
Do not use a root-level `AGENTS.override.md` as an additive mechanism.

## Adaptive topology contract

The lead chooses topology from dependency and ownership before choosing a
model:

1. Handle a small task directly when delegation cannot reduce time to
   acceptance.
2. Reuse one worker for sequential checkpoints only while goal, non-goals,
   semantic owner, write surface, authoritative constants, permissions, tier,
   and acceptance contract remain unchanged.
3. Start fresh workers for independent semantic or write surfaces. Parallelize
   only disjoint surfaces.
4. Start a fresh worker when any reuse invariant changes, when a phase crosses
   design/implementation/launch/recovery, or when inherited context is no
   longer reliable.
5. Keep the lead as scheduler and final acceptor. A worker may create a third
   layer only when its brief explicitly authorizes cheap, bounded, independent
   subtasks and includes their cost and receipts.
6. Use `candidate`, `lead-accepted`, and `user-accepted` as distinct success
   states. `NEEDS_CONTEXT`, `HOLD`, `BLOCKED`, and `SUPERSEDED` are valid
   non-success outcomes.
7. Checkpoint only at a decision-bearing boundary, not mechanically at every
   plan, first diff, RED, or GREEN event.
8. Review according to risk. Target drift invalidates affected claims; use a
   narrow delta review when owner, assumptions, and contract remain stable.

## Migration

Order is load-bearing:

1. Create the global contract before removing any project instruction source.
2. Slim `/data/CoordExp/AGENTS.md` to the CoordExp project contract.
3. Apply that project contract only to `CoordExp-swift`,
   `research-probe-infras`, and `research-probes`, folding any live local rules
   into the root file.
4. Delete root `AGENTS.override.md` only after any folded local rules are
   present in `AGENTS.md`.

Do not touch any other worktree, including its root or nested instruction
files. Also exclude app-owned `/data/CoordExp/.codex/worktrees/**`, `/tmp`
worktrees, external repositories, plugin caches, generated `.pi-worker`
fixtures, and unrelated dirty files.

## Verification

- The main checkout and three maintained worktrees have a non-empty root
  `AGENTS.md`.
- None of the three maintained worktrees has a root `AGENTS.override.md`.
- Worktree-local contract text remains present exactly once.
- Global and project files contain no contradictory topology or inheritance
  claims and pass `git diff --check` in each checkout.
- A deterministic discovery audit confirms global → project ordering and no
  same-directory shadowing.
- A live fresh-session instruction-source smoke is optional because it invokes
  a model; run it only with explicit material-cost authorization.

## Residual caveat

Because `CODEX_HOME` is inside `/data/CoordExp`, a task whose cwd is under
`/data/CoordExp/.codex` can discover the global file again as a project-nested
file. Keep the global contract compact and do not launch normal repository work
from `.codex`; verify this edge separately before changing `CODEX_HOME`.
