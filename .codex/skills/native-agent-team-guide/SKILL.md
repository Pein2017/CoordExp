---
name: native-agent-team-guidance
description: Choose native agent-team topology, package ownership, model and effort, and concise message flow when delegation can reduce time or lead context; choose direct, flat, or bounded nested delegation by package shape and authorization.
---

# Native Agent Team Guide

Use native tools, not another scheduler or mandatory team workflow. The
[agent contract](../../../.codex/AGENTS.md) owns authority, delegation permission, and
acceptance. Loading this skill does not authorize a research launch or exceed
the active delegation boundary.

## Choose the smallest useful team

Handle small, tightly coupled, already-understood work directly. Delegate a
complete, independently verifiable outcome when briefing, integration, and
acceptance cost less than the work saved. A frozen command plus an event monitor
often needs no agent. Scouting is optional, not a required first stage.

Choose direct work, a flat `L0 lead -> L1 package owner(s)` shape, or bounded
`L0 -> L1 -> L2` delegation inside an owned package. Do not default to flat or
nested topology, and do not add a scout or nesting ceremony unless it reduces
lead work while preserving genuine integration. These are responsibilities, not
permanent posts or model ranks:

- **Lead:** user intent, scientific/spec meaning, cross-package interfaces,
  resource allocation, integration, and final acceptance.
- **Package owner:** own one independently assessable research or engineering
  question, gather the required evidence, and deliver a coherent result within
  the agreed scope. Implementation and real entry/consumer checks apply when
  the task requires them.
- **Optional specialist:** answer one evidence question, challenge one named
  failure mode, or analyze one difficult idea. No automatic reviewer per task.

Split by independently acceptable outcomes, not by files or process steps.
If one lane's findings would repeatedly redefine another lane's task, settle
that dependency first or keep the coupled work with one owner.
Keep one bug's diagnosis/fix/check loop together unless a real independent lane
exists. Parallelize independent reads or disjoint writes, not competing fixes,
shared-state debugging, or the same scarce runtime. Do not assign a scout if
the lead will repeat its entire investigation anyway.

## Model and effort routing

Prefer Luna `max` and Sol for subagents; keep Astra primarily in the main
thread for framing, integration, and final acceptance. This is the user's
cost-conscious routing preference, not a measured claim of equal quality.
Choose directly by task difficulty and verifier strength, not a mandatory
escalation ladder. Check actual callable models, efforts, and fork inheritance
constraints before dispatch. Effort buys search depth, not authority.

| Model | Working range and roles |
|---|---|
| `gpt-5.6-luna` | Prefer `max` for bounded implementation or analysis with clear invariants and a strong verifier; `medium/high` remains sufficient for simple extraction, inventory, or mechanical work. Own a complete package, not only scouting. |
| `gpt-5.6-sol` | Default for work needing more semantic judgment or complex implementation: `high` as a starting prior, `medium` for clear bounded tasks, `xhigh/max` for difficult reasoning or debugging. Select the needed effort directly. |
| `gpt-6-astra` | Primarily the main-thread lead. Use a subagent only for a concrete capability gap or consequential uncertainty that Luna/Sol cannot resolve economically; do not add an automatic Astra reviewer. |

For clear, readily verifiable work, let Luna `max` own the package. For more
complex work, prefer Sol as owner, with independent Luna work only when it
reduces total effort. If briefing Luna requires the lead to solve the task,
choose Sol rather than expanding the prompt. Task length alone does not require
Astra. Terra remains outside the default rotation, not unavailable.

Allow the same owner to correct a concrete failed verifier or counterexample
when the scope remains valid. Include retries and additional lead review in the
cost comparison; cheaper attempts can still produce a cheaper accepted outcome.
Do not cycle through every effort or persist when corrections expose a semantic
or capability gap. Bring decision-bearing ambiguity to the lead, which can
resolve it or assign a focused stronger adviser. Advice does not transfer
user-owned decisions.

Avoid duplicating the worker's investigation in the lead. Use concise invariants
and real consumer checks, not over-detailed briefs or routine second reviews.
Route summaries by semantic risk, not their label. Model preferences do not
create review gates or change the running lead setting. Preserve acceptance
standards; do not infer success rates from completed workers or mixed task costs.

## Scout and collector outputs

When a Luna scout or collector is used, return a compact handoff containing:
the question and scope; a small evidence table with counts and exact artifact
paths; and gaps or unknowns. Do not return raw logs. The lead should be able to
integrate the evidence without repeating the inventory.

## Brief once; share changes, not whole histories

Give a self-contained brief with only execution-changing information:

```text
goal / non-goals; overall decision this result informs;
cwd; authoritative artifact and version;
owned read/write paths; permissions and frozen invariants;
real acceptance command/evidence; deliverable and stop rule;
delegation allowed or forbidden (plus bounds if allowed).
```

### Implementation briefs for Luna and Sol

Improve first-pass acceptance by removing execution-changing ambiguity, not by
asking the worker to "get it right in one shot." One pass means one assignment
through implementation and its own correction/check loop, not one edit without
tests. Treat model-specific adjustments below as working priors to validate on
real tasks, not proven capability limits.

For implementation, specialize the brief above with only the missing facts:

- Observable before/after behavior and a concrete acceptance example; distinguish
  settled semantics from implementation choices the worker owns.
- Known entrypoint/caller and the closest existing pattern, with exact paths or
  symbols when already discovered. Mark suspected locations as hypotheses;
  let the worker locate unknown ones rather than inventing APIs or defaults.
- Relevant preserved behavior, failure behavior, and write boundaries. Include
  a discriminating edge case when it changes correctness, not an exhaustive
  speculative checklist.
- The nearest real acceptance command and expected outcome, or the required
  consumer behavior if the command is not yet known. Reuse the project's test
  policy; bugs need a reproducer, not tests that merely mirror the patch.

For **Luna max**, prefer a settled interface and a small coherent implementation
surface, a concrete example, and the relevant existing pattern. Resolve
user-owned ambiguity before dispatch; leave local implementation to the worker.
If this requires the lead to design every step, give the package to Sol instead.
For **Sol**, state the overall intent and constraints while leaving room to
trace dependencies and choose the implementation. Name already-settled design
choices so it can finish the implementation without reopening them; ask it to
surface evidence that invalidates those choices rather than silently redesign.
Neither model needs a long persona, repeated rules, or a compulsory plan report.

Ask the worker to inspect relevant code, implement, run the required verifier,
and repair failures within its scope before returning. Missing semantic input,
an invalid frozen contract, or an out-of-scope change goes to the lead with the
exact conflict; ordinary discoverable details do not. Return a candidate with
changed paths, observed check results, and remaining limitations. Stop when the
acceptance behavior is verified; do not broaden review to improve confidence
without a concrete unresolved risk.

Evaluate these briefs on naturally occurring implementation tasks: record whether
the first submission passed lead acceptance, the reason for substantive rework,
and worker plus attributable lead usage in existing task evidence. Separate brief
gaps, implementation errors, and environment failures. Do not claim improved
one-shot rates from a dry run, mix unlike tasks, or add mandatory duplicate runs
or a new accounting system. Shorten or adjust the brief from observed failures.

Distinguish the assignment, not another topology: a **research-question owner**
gets the frozen question, claim/resource boundaries, decision-bearing evidence,
and stop rule, with freedom to choose methods inside those bounds. Do not solve
the investigation in its brief. An **execution owner** implements the accepted
design and verifies its real entry/consumer; it does not reopen frozen research
meaning. Escalate conflicts rather than silently switching assignments.

For evolving research, use a short current-context section in the existing
authoritative record: overall question and outcome criterion, fixed boundaries
and open decisions, what L0 is deciding or awaiting, active owners and their
dependencies, and the latest changes that affect work. Give that section one
owner; link package evidence rather than duplicating it. All nodes need the
overall purpose; each needs detail about its own dependencies, not every log.
Identify relevant peers, which changes require notification, and this entry
point in the brief. Read it on assignment or resumption and refresh affected
inputs after a relevant change, before dependent costly execution or submission.
Do not add periodic polling or require a separate board for a simple task.

Reference the existing spec, config, or research record instead of copying it
into every message. Identify mutable inputs by a checkpoint/hash when drift
would change the result. Shared files are not automatically shared knowledge.
Choose `fork_turns` dynamically: consider `all` when research history and prior
tradeoffs materially affect the assignment, limited turns when recent context
suffices, and `none` for self-contained work or an independent challenge that
benefits from isolation. Supply current scope and missing evidence either way.
Inherited history is a spawn-time snapshot, not a live feed of later parent or
sibling work; mark superseded assumptions explicitly. Respect the selected
fork mode's model and effort inheritance constraints; no role mandates a mode.

Use the native v2 tools according to their actual semantics:

- `spawn_agent`: one named package or specialist, not a permanent department.
- `send_message`: deliver a fact/question without starting an idle target's
  turn. Use canonical paths such as `/root/package/worker` across branches.
- `followup_task`: continue a non-root target and start a turn if idle. Reuse
  follows the agent contract's continuity and authority checks.
- `wait_agent`: event-driven mailbox wait; follow contract wait bounds. Use
  `list_agents` to discover peers and reconcile the roster when needed, not as
  a polling loop. Names and status do not establish responsibilities; use the
  brief or ask the owner.
- `interrupt_agent`: reconcile interrupted work before handing off its surface;
  do not assume interrupting a reasoning turn terminates its external jobs.

Send decision-bearing changes, blockers, interface facts, and completion, not
routine progress. A useful message is: finding -> evidence -> impact -> action
needed. Peers may exchange evidence directly; they may not expand each other's
authority, redirect another owner's work, or change a shared contract. Route
cross-package decisions to L0 and update the single authoritative artifact.
Do not broadcast logs/transcripts; native completion normally returns to the
immediate parent, so a package owner must synthesize its children's results.

Use existing research records for shared evidence, with each agent updating
its owned file or section. Avoid a duplicate board or competing writers.
Record updates do not notify peers automatically: send affected peers a short
change summary and evidence path when their decisions or execution may change.
For a changed invariant, decision-bearing input, or producer/consumer interface,
identify the affected work and whether existing results remain valid, need
recomputation, or still need checking. The affected owner acknowledges and
reconciles that change before further dependent execution or submission; do
not require acknowledgments for routine evidence updates. Use `followup_task`
if an idle owner must act. Peer interface facts can travel directly while the
package owner retains integration; shared understanding does not replace a
real consumer check on a changed artifact.

## Bounded L0 -> L1 -> L2 package delegation

Under the user's standing authorization, an L1 package owner may dispatch
bounded Luna scouts, collectors, or mechanical workers at L2 without a fresh
permission request for each helper. Stay inside the owned package and its
shared resource allowance; this does not authorize L3, extra material GPU use,
or expansion of semantic scope. Name levels
unambiguously: L0 is root, L1 its child, L2 its grandchild. Native capability
does not grant permission beyond this standing bound; do not infer a hard v2
depth guard from `agents.max_depth` (the inspected v2 implementation ignores
it). Check current tool availability and limits before relying on them; do not
change runtime configuration to satisfy this skill.

Nesting is useful when a package contains independent, disjoint subwork and L1
can absorb its implementation detail, correction, and integration instead of
forwarding it to L0. If L1 merely relays a task or L0 must redo all its work,
keep the package flat. Do not invent extra workers to justify a layer.

For an authorized nested package:

- L0 freezes package/spec meaning, interfaces, acceptance, owned paths,
  delegation ceiling, total worker/resource allowance, and allowed model/effort
  choices. All descendants share that allowance; it is not multiplied per L1.
- L1 decomposes only that package, dispatches with explicit model/effort/fork,
  names upstream inputs and affected peers, assigns disjoint write ownership,
  integrates L2 candidates, and runs the
  package's consumer-facing checks. It is not a second portfolio scheduler.
- L2 implements or investigates its bounded part; it cannot spawn L3, alter the
  package contract, or bypass L1's integration. L1 does not concurrently edit
  an L2-owned surface; ownership returns only after reconciliation.
- L1 returns the candidate, exact artifact/diff and verifier results, unresolved
  risks/decisions, and the bounded cost/result evidence required by the contract;
  no transcript packet. L0 checks the package boundary and decision-bearing
  evidence, not every internal step. Only L0 marks final lead acceptance.

If this topology helps one package, that does not authorize deeper recursion
or carry delegation permission into a different goal or phase.

## Close the loop without duplicate work

Inspect the exact candidate and replay the smallest decision-bearing verifier.
Test the real caller/consumer where leaf tests cannot establish acceptance.
Use a specialist review only for a named risk; after correction, recheck that
counterexample and the acceptance commands, not another broad review round.
Fresh contexts are appropriate when the contract or foundational assumptions
change; otherwise let the same owner finish its correction loop.

Before an expensive rerun, distinguish failed execution from a derived-only
evaluation defect. For research work, use the retained-evidence recovery rule in
[research-flow](../research-flow/SKILL.md) before deciding whether execution must
repeat; a failed reducer alone is not a reason to repeat valid model work.

Optimize the whole accepted outcome: worker usage, lead intervention, rework,
integration/verification, and runtime wait. Prefer lower total cost at the
required acceptance standard; waiting, integration, and rework are not free. Cached tokens are not unique
text or a complete cost measure; completed workers are not accepted-task
evidence. When cost bears on a routing decision, separate cached input,
uncached input, and output using dated applicable rates; missing usage or
rates remain unknown, not zero. Compare like tasks at like acceptance, including
failed attempts, substantive rework and attributable lead integration.
Use naturally occurring acceptance evidence to revise routing, not a
mandatory model alternation, duplicate runs, or new accounting infrastructure.
Stop when acceptance is satisfied.
