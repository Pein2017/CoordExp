# Contract and receipt shapes

Use these shapes as concise recipes. Omit fields that are genuinely irrelevant;
do not replace them with copied conversation history.

## L0 to L1 package contract

```text
package_id:
decision-owning outcome:
frozen goal and non-goals:
user-owned decisions that remain frozen:
cwd and authoritative paths/constants:
owned read/write surfaces and single-writer rule:
permissions and material-cost boundary:
required L1 responsibilities:
acceptance commands/evidence:
mandatory escalation triggers:
output status and acceptance-packet fields:
known failure modes:
tier and budget:
stop rule:
```

The contract grants local implementation latitude only inside the frozen
boundary. It does not grant architecture, publication, destructive, costly, or
external authority that L0 does not already possess.

State the effective depth and why it is cheaper than direct L1 execution. For a
shared physical worktree/index, state that writer packages and commits are
serialized or name the isolated worktrees that make parallel writes safe.

## L1 to L2 execution ticket

```text
ticket_id and parent package_id:
one executable output:
exact cwd and owned paths:
required inputs and authoritative constants:
interface invariants:
permissions and explicit non-authority:
deterministic verifier or evidence request:
expected receipt:
known failure modes:
tier and budget:
stop rule and typed failure status:
```

Select only `gpt-5.6-luna` or `gpt-5.6-terra`. Use Luna for cheap read-only
extraction and tightly mechanical tasks; use Terra for bounded implementation,
integration checks, or stronger local reasoning. Model availability is checked
live. Lack of an allowed model is a route failure, not permission to substitute.

## L1 to L0 acceptance packet

```text
package_id and frozen target identity:
status: candidate | NEEDS_CONTEXT | HOLD | BLOCKED | SUPERSEDED
outcome and decision impact:
changed paths/artifacts or evidence handles:
L1-replayed verification and exact decisive results:
review findings and disposition:
contract deviations and escalations:
residual risks and unexecuted gates:
recommended L0 action:
```

Raw logs, long diffs, and full child transcripts stay behind referenced handles.
Any P0/P1, claim-changing, permission-changing, or user-owned finding remains
visible even when the rest of the packet is compressed.

Expected generated drift, pin changes, fixture propagation, or formatting
shape differences that were already authorized belong in `review findings and
disposition`; they are not `HOLD` or `BLOCKED` unless they require new authority.

## Generic example

A long service migration has three independent packages: schema compatibility,
client implementation, and deployment verification. L0 keeps the migration
semantics and rollout decision. Three L1 leads may own those packages if their
write surfaces are disjoint. The client L1 can spawn a Terra builder and a Luna
test-receipt worker; it integrates and verifies both. A security review needing
Sol is a separate read-only L1, never an L2. Only package packets and material
decision deltas return to L0.

## Per-package pilot receipt

```text
task class and package size:
effective topology and model/effort routes:
accepted outcome and verifier:
L0 context or token evidence available:
L0 implementation interventions:
L1 correction rounds:
semantic escalations and role/write-surface drift:
wall time and priced/unpriced usage evidence:
observed benefit:
observed overhead or failure:
retain | revise | reject recommendation (user decides):
```
