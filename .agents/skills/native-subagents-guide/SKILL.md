---
name: native-subagents-guidance
description: Choose native agent-team topology, package ownership, model and effort, and concise message flow when delegation can reduce time or lead context. Flat by default; nested package delegation only when authorized and worthwhile.
---

# Native Agent Team Guide

Use native tools, not another scheduler or mandatory team workflow. The
[agent contract](../../../.codex/AGENTS.md) owns authority, delegation permission, and
acceptance. Loading this skill does not authorize a launch or deeper topology.

## Choose the smallest useful team

Handle small, tightly coupled, already-understood work directly. Delegate a
complete, independently verifiable outcome when briefing, integration, and
acceptance cost less than the work saved. A frozen command plus an event monitor
often needs no agent. Scouting is optional, not a required first stage.

Default to `L0 lead -> L1 package owner(s)`. These are responsibilities, not
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
Keep one bug's diagnosis/fix/check loop together unless a real independent lane
exists. Parallelize independent reads or disjoint writes, not competing fixes,
shared-state debugging, or the same scarce runtime. Do not assign a scout if
the lead will repeat its entire investigation anyway.

## Model and effort: broad roles, task-shaped choices

These are user-preferred starting hypotheses, not measured model rankings.
Check actual callable models/efforts; select directly rather than forcing an
escalation ladder. Effort buys search depth, not authority or automatic quality.
Compare model x effort pairs, not just families: lower-effort Astra can be a
candidate against higher-effort Sol. Treat supplied benchmarks as task-specific
priors, not proof of lead quality, local prices, or monotonic effort gains.

| Model | Working range and roles |
|---|---|
| `gpt-5.6-luna` | Use `medium/high/xhigh/max`, not low. Medium for exact evidence and mechanics; high for bounded fixes and checks; xhigh/max for deeper or multi-file implementation with clear invariants and a strong verifier. Not scout-only. |
| `gpt-5.6-sol` | Use `low/medium/high/xhigh/max`. Low for clear small implementation/review; medium for ordinary complete packages; high for hard/long implementation, integration, recovery, and audit; xhigh/max for difficult debugging, hidden correctness, architecture, or mechanism analysis. Not a fallback-only model. |
| `gpt-6-astra` | Consider every supported effort and role, including low for bounded work. Medium/high are the normal lead candidates; use xhigh/max/ultra when concrete uncertainty justifies them, often as a focused adviser rather than an always-deep lead. |

Sol high/xhigh is a normal review/audit choice. Astra subagents are exceptional,
not forbidden: use them for complex consequential review, uncertain ideas, or
a demonstrated task-specific advantage. Task length alone does not require
Astra. Terra remains outside the default rotation, not unavailable.

If making Luna reliable requires the lead to solve the task in an over-detailed
prompt, prefer Sol. A concise invariant plus a real counterexample/consumer
check is more useful than a longer list of instructions. Do not claim that
lower lead effort preserves quality until real acceptance evidence supports it;
escalate proactively at ambiguous semantic or high-consequence boundaries.
Writing a preferred effort in a brief does not change the running lead setting.

## Brief once; share changes, not whole histories

Give a self-contained brief with only execution-changing information:

```text
goal / non-goals; cwd; authoritative artifact and version;
owned read/write paths; permissions and frozen invariants;
real acceptance command/evidence; deliverable and stop rule;
delegation allowed or forbidden (plus bounds if allowed).
```

When coordination matters, identify relevant peers and their responsibilities,
which changes require notification, and the shared record location.

Reference the existing spec, config, or research record instead of copying it
into every message. Identify mutable inputs by a checkpoint/hash when drift
would change the result. Shared files are not automatically shared knowledge.
Let the lead choose `fork_turns` dynamically under the agent contract; worker
roles do not impose a fixed mode or turn cap. Inherit context when it saves
rebriefing, and supply task scope and missing evidence explicitly. Respect the
selected fork mode's model and effort inheritance constraints.

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

## Optional L0 -> L1 -> L2 package delegation

Use a nested package only when explicitly authorized under the agent contract.
Name levels unambiguously: L0 is root, L1 its child, L2 its grandchild. Native
capability does not grant permission; do not infer a hard v2 depth guard from
`agents.max_depth` (the inspected v2 implementation ignores it). Check current
tool availability and limits before relying on them; do not change runtime
configuration to satisfy this skill.

Nesting is useful when a package contains independent, disjoint subwork and L1
can absorb its implementation detail, correction, and integration instead of
forwarding it to L0. If L1 merely relays a task or L0 must redo all its work,
keep the package flat. Do not invent extra workers to justify a layer.

For an authorized nested package:

- L0 freezes package/spec meaning, interfaces, acceptance, owned paths,
  delegation ceiling, total worker/resource allowance, and allowed model/effort
  choices. All descendants share that allowance; it is not multiplied per L1.
- L1 decomposes only that package, dispatches with explicit model/effort/fork,
  assigns disjoint write ownership, integrates L2 candidates, and runs the
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

Optimize the whole accepted outcome: worker usage, lead intervention, rework,
integration/verification, and runtime wait. Cached tokens are not unique text
or a complete cost measure; completed workers are not accepted-task evidence.
Use existing checks, receipts, and usage when available, not duplicate model
benchmarks or new accounting machinery. Stop when acceptance is satisfied.
