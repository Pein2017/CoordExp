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

Prefer Astra `low/medium` for complex or consequential work. Prefer Luna
`medium/high` for routine scouts, collectors, evidence extraction, summaries,
and bounded dirty or mechanical work; use Luna `xhigh/max` for bounded,
verifiable implementation with clear invariants. These are routing priors, not
authority or automatic quality gates. Check actual callable models/efforts and
select directly rather than forcing an escalation ladder. Effort buys search
depth, not authority or automatic quality. In the current native tool, the
user's "light" maps to callable `low`; do not pass an unsupported `light`
effort.

`gpt-5.6-sol` is temporarily retired from default routing. Use it only after
explicit user opt-in for a comparison or fallback; never reactivate it as an
automatic fallback or escalation tier.

| Model | Working range and roles |
|---|---|
| `gpt-5.6-luna` | `medium/high` for cheap scouting, mechanical work, evidence extraction, and summaries; `xhigh/max` for simple or moderate implementation with clear invariants and a strong verifier. Not scout-only. |
| `gpt-6-astra` | `low` (user's light) for complex implementation; consider `medium` when uncertainty warrants more depth. `high` or above for consequential review/audit; `xhigh/max` for decisive reasoning, brainstorming, or a focused adviser. Advice does not transfer user-owned decisions. |
| `gpt-5.6-sol` | Explicit-opt-in comparison or fallback only; never part of default routing. |

Two provisional operating patterns:

- For clear, readily verifiable work that can tolerate latency, consider Luna
  `xhigh/max` owning the whole package rather than duplicating its work in Astra.
- For complex or time-sensitive work, consider Astra `low/medium` as owner,
  with Luna handling independent evidence or mechanical work in parallel.
  Useful owner work need not wait for unrelated scouts to finish.

Treat Luna's source token price as a low-weight concern, not its waiting,
integration, or rework cost. Cheap calls favor useful delegation, not
redundant reports.
Route summaries by semantic risk, not the label "summary" alone.

Astra is a normal worker/adviser choice in this routing, not exceptional.
Task length alone does not require it, and a model preference does not create
a review gate. Terra remains outside the default rotation, not unavailable.

If making Luna reliable requires the lead to solve the task in an over-detailed
prompt, prefer Astra `low/medium` rather than solving the task for Luna. A
concise invariant plus a real counterexample/consumer
check is more useful than a longer list of instructions. Do not claim that
lower lead effort preserves quality until real acceptance evidence supports it;
escalate proactively at ambiguous semantic or high-consequence boundaries.
Writing a preferred effort in a brief does not change the running lead setting.

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
integration/verification, and runtime wait. Luna's token price is low-weight;
waiting, integration, and rework are not free. Cached tokens are not unique
text or a complete cost measure; completed workers are not accepted-task
evidence. When cost bears on a routing decision, separate cached input,
uncached input, and output using dated applicable rates; missing usage or
rates remain unknown, not zero. Compare like tasks at like acceptance, including
failed attempts, substantive rework and attributable lead integration.
Use naturally occurring acceptance evidence to revise routing, not a
mandatory model alternation, duplicate runs, or new accounting infrastructure.
Stop when acceptance is satisfied.
