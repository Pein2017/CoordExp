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
   the live spawn interface before selecting a route. When that interface
   advertises Luna, treat it as the default Codex worker model rather than as a
   legacy-only option.
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
5. **Model and effort:** choose these jointly as a routing pair, not as a model
   ranking followed by a monotonic "more thinking is better" knob. The same
   effort can affect models differently, and excess effort can increase scope
   expansion, redesign, or review churn. Map the pair through the current
   priors and use the least costly combination likely to finish reliably;
   increase capability or effort only for demonstrated reasoning need,
   silent-error risk, or consequential judgment.

For a worker-shaped Codex lane with a reliable verifier, prefer Luna when the
live interface exposes it. Start at the least effort likely to pass; `high` is
a reasonable ceiling for ordinary bounded scouting, feasibility, code
navigation, or mechanical checking. Use Luna `xhigh` or `max` only when a
localized depth gap has been observed and the same verifier still decides the
result. Route directly to Terra, Sol, or Claude when the role itself requires
their current strengths—such as unresolved integration ownership,
silent-correctness review, research judgment, or provider independence—rather
than treating effort escalation as a substitute for role fit.

A reliable verifier is necessary but not sufficient to make a lane
Luna-shaped. Do not give Luna ownership of crash consistency, durability,
concurrency, backward-compatible schema evolution, or a multi-owner lifecycle
seam merely because focused tests can be written. Those tasks have integrative
or silent-correctness blast radius: use Terra or a broad builder for
implementation ownership, and retain an independent high-consequence reviewer
when the verifier cannot enumerate every bad state.

Judge the semantic dataflow, not the number of edited files or whether an API
is frozen. Preserving one new value across controller, worker, shard, merge,
success, failure, and old no-value paths is an integrative lifecycle lane even
when every writer owns disjoint paths. A lane that spans three or more such
stages, or must preserve old-reader or old-writer compatibility, is not an
ordinary Luna task; split out only the mechanical leaf work and route the
end-to-end owner to Terra, Sonnet, Opus, or another integration-fit builder.

For the current Codex 5.6 family, use this provisional responsibility split:

- **Luna:** bounded evidence collection, feasibility, code and convention
  scouting, and mechanical checks with an external verifier.
- **Terra:** stable cross-file implementation ownership, interface synthesis,
  and contract closure when the lane is too integrative for a scout but does
  not own the research conclusion.
- **Sol:** conclusion-critical mechanism interpretation, research judgment,
  architecture trade-offs, and silent-correctness audit. Prefer `high`; admit
  `xhigh` only for a named high-consequence uncertainty that survived a
  narrower pass.

The lead sets a cost and correction budget before spawning. Do not assign
several overlapping Sol judges to rediscover the same ambiguity. Narrow the
question in the lead, give one lane decision ownership, and use cheaper scouts
to prepare evidence when that preparation is mechanically verifiable.

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

Admit parallel writers only when their owned paths are disjoint and any shared
interface is already frozen. If lanes share a state, lifecycle, or contract
seam, assign one explicit integrator or serialize the waves. A worker's focused
checks are a lane receipt, not integration acceptance; after joining, the lead
runs the cross-lane or full verifier.

Disjoint write surfaces permit parallelism; they do not determine model fit.
Trace each lane's invariant through runtime stages before calling it bounded.

## Hand Off Context

- Set `fork_turns: "none"` explicitly for every Codex or Claude Code spawn.
  Never omit it or forward a conversation-history window.
- Provide a self-contained brief with goal, exact cwd and owned paths, relevant
  evidence, permissions and write intent, non-negotiable constraints, expected
  output, acceptance method, completion condition, and stop boundary.
- State whether the lane needs a fresh process or continuity. Use fresh agents
  for session, tool-registration, build, auth, or stale-runtime acceptance; use
  follow-ups to close prior findings, but never relabel them as fresh or
  independent evidence.
- Include exact commands, tool recipes, or known traps only when they are
  non-obvious and material. Otherwise preserve the worker's judgment.
- Keep work with the lead when the required context cannot be distilled without
  changing its meaning.

## Cascade And Escalate

Use cheap-first composition only when a reliable, inexpensive verifier exists:

1. When available and suitable, run Luna as the first Codex worker at the least
   effort likely to pass. Give one focused correction or move through `high`
   when the observed gap is bounded implementation depth; do not automatically
   climb to `xhigh` or `max` before reconsidering the role and brief.
2. Evaluate outside the worker's own prose.
3. Give one focused correction when the failure is local and diagnosed.
4. If it still fails, escalate by failure type and re-run the verifier.

Do not cascade when no verifier can catch the important error, a silent mistake
can contaminate downstream work, or the output decides a research or
architecture direction. Route directly to a stronger judgment lane instead.

Likewise, do not cheap-first a persistence, concurrency, compatibility, or
cross-owner lifecycle implementation as one undifferentiated lane. First split
off any truly mechanical subtask; route the remaining invariant-owning seam by
its consequence and integration breadth.

- Keep path, command, and deterministic-test failures within the builder lane.
- Escalate cross-file contract ambiguity to a stronger builder or reviewer.
- Escalate competing research meanings or route-changing contradictions to an
  independent research or architecture judge.
- When a worker is overthinking, broadening scope, or redesigning beyond the
  brief, first tighten the lane or lower effort; raising effort usually
  amplifies that failure mode. Raise effort when the observed failure is
  insufficient depth on a bounded uncertainty, not merely because a task
  failed.
- Higher effort is not evidence of correctness, and a stronger model does not
  replace an executable check.

## Join And Stop

- Classify each lane as `required`, `parallel-then-join`, or
  `explicitly-detached`. Join and synthesize required work before answering;
  detachment requires explicit user intent.
- While lanes run, advance useful independent lead work first. Wait only when
  the critical path is blocked, and prefer one completion-first join over
  repeated short polling. A progress or unchanged wake must change scheduling
  to justify another observation; do not immediately wait again by reflex.
  Use agent listing or message reads for targeted diagnosis or steering, not as
  a status-polling loop.
- Observe long-running lanes at meaningful boundaries. Steer with one narrow
  follow-up when evidence changes the task; do not restart the same brief
  blindly.
- Prefer a mechanical checker to an LLM judge. When judgment is unavoidable
  and independence matters, use a different model or provider from the worker;
  never decide by majority vote or model prestige.
- Bind consequential review to an exact commit/tree, runtime identity, and
  evidence set. Any unresolved blocker holds release; after a change, recheck
  the affected gate at the new fixed point, using narrow confirmation when only
  non-behavioral evidence changed. Freeze the declared surface and establish a
  no-edit window before a fixed-tree audit starts. If a necessary correction
  lands during review, invalidate that certification attempt, publish a new
  identity, and re-audit the bounded delta; do not accept a reviewer who merely
  continued reading a moving target.
- Never blind-loop. Every iteration needs an out-of-band evaluator. Keep the
  best verified result, stop on first regression, and vary the angle only for
  useful diversity.

## Learn From Real Work

Optimize the team first for accuracy, precision, semantic coverage, and
acceptance success; then for end-to-end wall time and lead correction burden;
then for model spend when comparable receipts exist.

Track both accepted-work cost and fully loaded correction cost. A completed
agent with no observed follow-up is only an acceptance proxy; a follow-up is
correction burden, not proof of failure. For consequential routing updates,
record explicit lead or verifier dispositions and count rework spend in the
effective cost denominator. Use a usage ledger to find concentration and
outliers, not to turn heterogeneous roles into a global model leaderboard.

### Evaluate Each Lane

Before spawning, keep a transient lead-owned record of the route pair, narrow
task shape, acceptance verifier, consequential failure it covers, correction
budget, and escalation boundary. After joining and running integration
acceptance, assign exactly one disposition:

- `accepted_first_pass`: the lane receipt and integration verifier pass without
  a substantive correction;
- `accepted_after_correction`: one bounded same-agent correction closes a
  diagnosed local gap and the verifier then passes;
- `rejected_or_escalated`: the verifier exposes a semantic, integration, or
  scope failure that the lane does not close within budget; or
- `surface_failure`: auth, quota, plugin, client, runtime, or missing-receipt
  failure prevents a model-quality comparison.

Keep the lane disposition separate from the target verdict. A reviewer or
judge can be `accepted_first_pass` by returning a reproducible `HOLD` that finds
the consequential defect; the audited tree is rejected, but the review lane
succeeded. Conversely, target drift during a fixed-tree certification makes
that certification attempt a `surface_failure` even when the reviewer usefully
detects the drift. Repairing the audited target and asking the same reviewer to
verify a new fixed point is a new review pass, not a correction to the
reviewer's output. Use `accepted_after_correction` only when the agent's own
delivery needed the bounded correction.

When an observation changes a routing decision, summarize it as `route pair ->
narrow task shape -> required verifier -> observed advantage -> failure
boundary`. Call a route a best current scenario only at that granularity and
only after comparable accepted evidence; never infer an absolute model ranking
from different roles. Keep the disposition transient unless it changes a
prior, and never treat worker prose or completion status as acceptance.

- Let the lead judge model and effort fit dynamically from representative real
  work, verifier results, correction burden, and downstream integration. Do not
  spend substantial tokens assigning the same full task to multiple model or
  effort arms merely to produce a ranking. Run a narrow paired routing trial
  only when route uncertainty is decision-relevant and ordinary work evidence
  is insufficient.
- In a comparison, hold brief, evidence, tools, permissions, write surface,
  output contract, and verifier fixed. Change only the declared routing
  variable.
- Draw a comparative conclusion only when every required arm returns a usable,
  comparable receipt. Missing, interrupted, auth, plugin, client, or runtime
  failures are surface evidence rather than model-quality evidence; do not
  silently substitute a different model or surface and score it as the same
  arm.
- Treat `model type × thinking effort` as an interacting pair. To learn the
  interaction, compare efforts within one model or models at one effort before
  changing both; do not attribute a paired-route result to model type alone.
- Separate worker quality from handoff quality, verifier quality, client-surface
  acceptance, and runtime reliability; client success is not implementation
  evidence for that model.
- Treat one result as provisional. Change a default only after repeated
  comparable evidence; weaken it when counterevidence appears.
- Update [references/model-priors.md](references/model-priors.md) only when an
  observation changes a routing decision. Edit priors in place; do not build a
  benchmark, profiling requirement, per-task ledger, or formal routing matrix.
