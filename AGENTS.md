# Agent Guide - CoordExp

> Canonical base guidance for all Codex work under `/data/CoordExp`. Codex treats
> `/data/CoordExp` as the project root; an active linked worktree may append a
> small `AGENTS.override.md` delta. Put task procedures in skills, current routes
> in `docs/`, and experiment state in its owning research unit.

## Grounding And Ownership

- User request > a deliberately scoped nested override > this guide > surface
  defaults.
- Start from the exact path, worktree, artifact, config, spec, diff, run, or
  question the user names. Inspect it before explaining or changing it.
- Work in the exact checkout in scope. Revalidate before crossing roots;
  branches, worktrees, memories, and old notes are not interchangeable.
- Shared Codex configuration lives under `/data/CoordExp/.codex`. Do not make
  workflows depend on `/root` state.
- This file is the only shared policy source. A worktree override contains only
  additions or explicit local replacements; it never copies this base.
- Give each decision and write surface one current owner. Treat reviews,
  memories, handoffs, and historical worktrees as evidence, not live authority.
- The user owns choices that change research meaning, compatibility,
  publication, material cost, or irreversible behavior. The agent owns
  discoverable facts and reversible implementation details.
- Ask only when an unresolved choice crosses that boundary. Otherwise inspect,
  choose the conservative repository-local interpretation, and continue.

## Judgment And Research

- Make the smallest reversible change that satisfies the request and its
  verification. Avoid speculative knobs, compatibility layers, abstractions,
  and cleanup outside the evidence-backed scope.
- Before a costly launch or research implementation, make the question,
  contrast, decision-owning outcome, strongest alternative, primary evidence,
  and stop rule explicit.
- For an exploratory slice, implement only what obtains the primary observation
  or protects its interpretation. Prefer one representative real smoke over
  production-shaped preflight.
- Keep model, data, geometry, order, prompts, tokens, objectives, metrics, and
  artifacts semantically aligned. State proxy-to-outcome assumptions.
- Treat a reusable interface as provisional until runtime evidence or a second
  real consumer establishes the seam.

## Orchestration And Delegation

- Use the main process as the orchestrator: it owns decomposition, global
  context, cross-lane decisions, research interpretation, synthesis, and final
  acceptance. Delegate bounded implementation, investigation, or review lanes
  when parallelism or attention isolation materially helps; keep simple
  one-lane work local.
- Give each worker a self-contained brief with its goal, exact ownership,
  permissions, evidence, completion condition, and stop boundary. Assign
  non-overlapping write surfaces and reconcile all worker output in the main
  process; independent reviews inform the decision rather than voting on it.
- Choose the worker surface first from required tools, runtime access,
  permissions, subscription, and isolation; then choose model and reasoning
  effort from task verifiability, complexity, and consequence. Codex worker
  surfaces include Terra and Sol; the CC Plugin can launch Claude Code workers,
  normally Sonnet or Opus. Verify live availability before dispatch.
- As a starting heuristic, use Terra or Sonnet for routine bounded
  implementation and mechanical checks, and Sol or Opus for complex,
  high-consequence, or conclusion-critical work. Adapt effort and escalate from
  evidence rather than preserving a fixed role table.
- Pass only the context a lane needs. Prefer a fresh self-contained brief for an
  independent worker; include conversation history only when the task truly
  depends on it.

## Shared Compute

- Python checks in this repository normally use the `ms` conda environment.
- The expected host capacity is eight GPUs, but live state is authoritative.
  Verify topology, processes, utilization, and free memory before launch.
- GPU use is shared by default: a device with an existing process is not
  automatically reserved. Co-locate only when projected peak memory plus
  headroom fits without OOM risk and compute interference is acceptable; use
  explicit device placement and never kill or evict unrelated processes.
- Coordinate before sharing occupied GPUs for timing- or benchmark-bearing work
  and before jobs that need most or all GPUs. Label diagnostic, reduced-scale,
  and benchmark-bearing runs honestly.

## Parallel Work And Hygiene

- Multiple Codex and Claude instances may modify the same worktree
  concurrently. Dirty or unfamiliar changes are expected: inspect them,
  preserve them, and never revert, overwrite, stage, or commit them unless the
  current task owns them.
- Use explicit paths for Git and destructive operations. Preserve credentials,
  unrelated artifacts, and parallel work; do not use broad cleanup to make a
  tree look tidy.
- Remove temporary scaffolding, caches, and smoke artifacts created by the
  current task once they no longer serve verification. Do not delete
  evidence-bearing artifacts, fixtures, or another worker's outputs without
  authority.
- Keep the codebase compact: remove dead branches and superseded shims caused by
  the current change, but do not turn local cleanup into a redesign.

## Evidence And Communication

- Use explicit config and schema contracts; unknown or retired surfaces should
  fail visibly instead of becoming hidden defaults.
- Verify installed upstream or runtime behavior when it owns the claim. Plans,
  mocks, banners, and receipts do not substitute for executed semantics.
- Every change needs proportionate evidence: a targeted test, real smoke, parse,
  artifact or manifest check, metric check, replay, residue search, or an
  explicit reason the check was skipped. Narrow checks first and label their
  scope and residual risk.
- Follow the user's language in conversation. Keep code, paths, commands,
  configs, schemas, formulas, experiment identifiers, plans, reviews, and
  handoffs in English unless requested otherwise.
- Put new investigations, interpretations, negative results, and continuation
  context in `research/`; use `docs/history/` for raw provenance and create
  no new `progress/` records.
- Lead reports with the outcome and crucial evidence. Reviews lead with
  evidence-backed findings and a decision; implementation reports include
  verification, skipped checks, and residual risk.

## Progressive Disclosure

- Use the narrowest matching skill. Stable cross-worktree skills live only in
  `$CODEX_HOME/skills`; worktree-local skills are explicit experimental deltas.
  Preserve official and vendor-managed ownership.
- When named evidence does not reveal the current owner, search
  `docs/catalog.yaml` or `docs/AGENT_INDEX.md` for one narrow route; do not
  load both or follow a fixed read order by default.
- Keep the instruction hierarchy to two semantic levels: this base and, only
  where needed, one worktree-root delta. Do not add nested instruction files for
  task state that belongs in a skill, document, spec, or research unit.
