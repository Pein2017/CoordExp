# Contract and receipt shapes

Omit genuinely irrelevant fields. Do not replace them with copied conversation
history. General delegation and acceptance policy remains owned by the user-wide
`AGENTS.md`.

## L0 to L1 package contract

```text
package_id and frozen target identity:
decision-owning outcome:
goal and non-goals:
user-owned decisions that remain frozen:
cwd and authoritative paths/constants:
owned read/write surfaces and permissions:
material-cost boundary:
acceptance verifier/evidence:
effective_depth_plan: 1 | 2
depth-2 predicate:
  independent output A:
  independent output B:
  expected time or L0-context advantage:
route plan: model x effort x layer
mandatory escalation triggers:
known failure modes:
tier and budget:
stop rule:
required acceptance-packet fields:
```

The contract grants local implementation latitude only inside this frozen
boundary. If fewer than two L2 outputs materialize, the package becomes
`effective_depth: 1`; L1 completes it directly.

## Terra to Sol escalation witness

Before spawning a Sol L1 as a capability escalation, record:

```text
prior Terra attempt and route:
unchanged frozen target and verifier:
acceptance-changing counterexample or semantic conflict:
one bundled Terra correction and replay result:
why the verifier or Terra inspection still cannot close it:
fresh Sol package id:
```

No witness means no capability escalation. A Sol reviewer may still be selected
under the user-wide review policy, but its model label does not expand authority.

## L1 to L2 execution ticket

Use this ticket only for an output named by the depth-2 predicate.

```text
ticket_id and parent package_id:
one executable output:
exact cwd and owned paths:
required inputs and authoritative constants:
interface invariants:
model and effort:
permissions and explicit non-authority:
deterministic verifier or evidence request:
expected receipt:
known failure modes:
tier and budget:
stop rule and typed failure status:
```

An L2 reports to its L1 and never spawns. If the ticket needs Sol-level semantic
or lifecycle judgment, promote it to a separate L1 adviser.

## L1 to L0 acceptance packet

```text
package_id and frozen target identity:
status: candidate | NEEDS_CONTEXT | HOLD | BLOCKED | SUPERSEDED
effective_depth: 1 | 2
materialized L2 outputs and evidence handles:
model x effort x layer routes:
outcome and decision impact:
changed paths/artifacts or evidence handles:
L1-replayed verifier and exact decisive results:
L1 correction rounds: 0 | 1
review findings and disposition:
contract deviations and escalations:
residual risks and unexecuted gates:
recommended L0 action:
```

Raw logs, long diffs, and full L2 transcripts stay behind evidence handles. Any
acceptance-changing, permission-changing, or user-owned finding remains visible.

## Per-package pilot receipt

L0 completes this after its acceptance decision:

```text
package_id and task class:
lead disposition: accepted | rework | escalated | failed
effective_depth: 1 | 2
depth-2 predicate result and materialized output count:
model x effort x layer routes:
accepted outcome and verifier:
L0 raw L2 transcript reads: integer
L0 direct package code edits: integer and handles
L1 correction rounds: 0 | 1
semantic escalations and role/write-surface drift:
critical-path wall time:
measured tokens and priced/unpriced usage coverage:
observed benefit:
observed overhead or failure:
retain | revise | reject recommendation:
```

`followup_aware` completion is not a strict lead disposition. Do not claim
context, latency, cost, or model superiority without a matched depth-1 package.
