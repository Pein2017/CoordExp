---
name: agent-routing
description: Compose and iteratively calibrate bounded Codex and Claude Code teams by choosing whether to delegate, worker roles, surface, model, reasoning effort, verifier, escalation, joining, and stopping. Use when a lead is deciding how to delegate, selecting subagent models or effort, splitting work among scouts, builders, reviewers, judges, or planning advisers, recovering failed agent work, comparing team compositions, or updating routing priors from real task evidence. Skip when a task is clearly small enough to keep local with no routing decision.
---

# Agent Routing

> Experimental. Iterate this worktree-local skill from real use; do not treat
> its current team patterns as permanent model rankings.

## Ground The Route

1. Keep the active primary lead, normally Codex, responsible for global
   context, decomposition, synthesis, research meaning, and final acceptance.
2. Read [references/model-priors.md](references/model-priors.md) when choosing a
   surface, model, or effort, or when new evidence may change a prior. Verify
   the live spawn interface before selecting a non-default route.
3. Keep simple work in the lead. Delegate only when parallelism,
   specialization, provider independence, or attention isolation materially
   helps.
4. Respect the live concurrency cap. Use a bounded worker pool and batch or
   queue excess partitions; do not create one agent per item by reflex.

## Build The Team

Choose in this order:

1. **Role:** scout, builder, mechanical verifier, reviewer, research judge, or
   architecture and planning adviser. Combine roles only when independence is
   unimportant.
2. **Acceptance:** decide whether success is mechanically checkable by a test,
   schema, exact comparison, runtime receipt, or artifact check, or instead
   requires judgment about semantics, trade-offs, or research meaning.
3. **Verifier:** select the acceptance mechanism before a cheap worker. It must
   cover the consequential failure, not merely syntax.
4. **Surface:** choose Codex or Claude Code from required tools, runtime access,
   permissions, isolation, quota, and the value of provider independence.
5. **Model and effort:** map the role through the current priors. Use the least
   costly combination likely to finish reliably; increase capability or effort
   only for demonstrated reasoning need, silent-error risk, or consequential
   judgment.

If a judgment-shaped task can be narrowed into a mechanically checkable lane
without changing its meaning, narrow the role and brief before up-tiering.

Useful starting compositions:

- **Scout -> lead:** collect bounded evidence; let the lead interpret it.
- **Builder -> checker:** use for implementation with executable acceptance.
- **Builder -> independent reviewer -> lead:** use when semantic or integration
  risks survive tests.
- **Blind judges -> counterexample -> lead:** use for a conclusion-critical
  question that benefits from model or provider independence.
- **Architecture adviser -> builder -> reviewer:** use only for a material fork;
  the lead retains the top-level decision.

Do not summon every model, run duplicate writers on one semantic surface, or
create a review chain when one owner plus a checker is enough.

## Hand Off Context

- Set `fork_turns: "none"` explicitly for every Codex or Claude Code spawn.
  Never omit it or forward a conversation-history window.
- Provide a self-contained brief with goal, exact cwd and owned paths, relevant
  evidence, permissions and write intent, non-negotiable constraints, expected
  output, acceptance method, completion condition, and stop boundary.
- Include exact commands, tool recipes, or known traps only when they are
  non-obvious and material. Otherwise preserve the worker's judgment.
- Keep work with the lead when the required context cannot be distilled without
  changing its meaning.

## Cascade And Escalate

Use cheap-first composition only when a reliable, inexpensive verifier exists:

1. Run the least costly plausible worker at low or medium effort.
2. Evaluate outside the worker's own prose.
3. Give one focused correction when the failure is local and diagnosed.
4. If it still fails, escalate by failure type and re-run the verifier.

Do not cascade when no verifier can catch the important error, a silent mistake
can contaminate downstream work, or the output decides a research or
architecture direction. Route directly to a stronger judgment lane instead.

- Keep path, command, and deterministic-test failures within the builder lane.
- Escalate cross-file contract ambiguity to a stronger builder or reviewer.
- Escalate competing research meanings or route-changing contradictions to an
  independent research or architecture judge.
- Higher effort is not evidence of correctness, and a stronger model does not
  replace an executable check.

## Join And Stop

- Classify each lane as `required`, `parallel-then-join`, or
  `explicitly-detached`. Join and synthesize required work before answering;
  detachment requires explicit user intent.
- Observe long-running lanes at meaningful boundaries. Steer with one narrow
  follow-up when evidence changes the task; do not restart the same brief
  blindly.
- Prefer a mechanical checker to an LLM judge. When judgment is unavoidable
  and independence matters, use a different model or provider from the worker;
  never decide by majority vote or model prestige.
- Never blind-loop. Every iteration needs an out-of-band evaluator. Keep the
  best verified result, stop on first regression, and vary the angle only for
  useful diversity.

## Learn From Real Work

Optimize the team first for accuracy, precision, semantic coverage, and
acceptance success; then for end-to-end wall time and lead correction burden;
then for model spend when comparable receipts exist.

- Prefer representative real work. Run a paired routing trial only when route
  uncertainty is material enough to justify it.
- In a comparison, hold brief, evidence, tools, permissions, write surface,
  output contract, and verifier fixed. Change only the declared routing
  variable.
- Separate worker quality from handoff quality, verifier quality, and surface
  or runtime reliability.
- Treat one result as provisional. Change a default only after repeated
  comparable evidence; weaken it when counterevidence appears.
- Update [references/model-priors.md](references/model-priors.md) only when an
  observation changes a routing decision. Edit priors in place; do not build a
  benchmark, profiling requirement, per-task ledger, or formal routing matrix.
