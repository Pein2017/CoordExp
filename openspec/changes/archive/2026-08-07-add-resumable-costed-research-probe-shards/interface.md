# Resumable support-completion adapter interface

The sealed support-completion plan remains the logical authority.  The adapter
reads its exact bytes, derives a separate eight-slot LPT schedule from declared
scalar-equivalent-forward cost, and never changes `shard_index`: that field
continues to identify only the legacy terminal receipt partition.

For each physical slot, `open_slot_journal` binds the plan byte/content
identities, schedule digest, slot number, adapter version, and caller-owned
execution identity before an observer/model is called.  `execute_slot` starts
one attempt and appends one complete opaque observation per context.  An
explicit continuation uses the same root and identities, starts a new attempt,
and skips all accepted context records.  It never retries an accepted success,
failure-shaped, unmatched, or other opaque payload.

Only mechanically terminal journals may reach
`materialize_legacy_shard_receipts`.  The materializer validates every slot,
including its mandatory execution identity, exact physical-slot identity,
schedule/context fingerprint, and scheduled work-item denominator. It regroups
observations by the sealed legacy shard and plan position and calls its owned
`build_legacy_receipt` to produce the unchanged receipt schema. The complete
in-memory receipt set must pass the caller-supplied, digest-bound unchanged
merger validator before any write-once receipt is published. It refuses missing,
foreign, duplicate, non-measured, merger-ineligible, or otherwise ineligible
payloads. Slot/attempt/process fields never enter the terminal scientific receipt.

`launch_slot_worker` is a parent-only mechanics surface.  It observes a child
PID/return code/signal and last durable record, then write-once publishes a
mechanics receipt.  It never retries or authorizes continuation.  A GPU smoke
must supply an absent fresh root, an explicit `CUDA_VISIBLE_DEVICES=N` parent
environment, child logical `cuda:0`, and a separately invoked continuation.

`run_resumable_natural_boundary_support_shard.py` is the executable consumer
example and adoption boundary. It hash-binds the exact active runner, its
transitive active `src` runtime tree, sealed plan/census, authored and resolved
configuration authority, exact non-symlink base-model file denominator and
bytes, adapter tensor, embedding delta, and embedding source-gate receipts
before model work. It uses the active runner's
candidate bank, scalar scorer, and support-feature function; one complete
observation is returned to `execute_slot` per context. An accepted failed or
partial observation is durable and is not retried automatically.

Initial invocation omits `--resume` and requires an absent journal root. A
continuation repeats every identity argument and adds only `--resume` plus a
new attempt-specific runtime receipt path. The parent may pass the same worker
command to `launch_slot_worker(..., terminate_after_first_durable_record=True)`
for the bounded signal gate; production execution leaves that flag false.

The bounded two-context mechanics projection uses
`project_logical_contexts` only for the real signal smoke. It retains the
sealed parent plan digests and exact context mappings, derives a separate
one-slot schedule, and can publish only
`materialize_bounded_mechanics_terminal`; it cannot publish or masquerade as a
scientific legacy shard receipt.

The adoption glue must first compare the active consumer and merger bytes to
the fixed digests in the proposal, then call the active consumer's
`validate_execution_plan` with the exact plan byte hash and census before
opening a journal or model.  This worktree intentionally does not apply that
glue to the active untracked research unit.

Downstream topology is explicit: materialized legacy receipts are consumed by
the unchanged merger; the analyzer consumes census-v3, not shard receipts. The
compatibility receipt proves both owned contracts independently and preserves
the pre-existing prior-support-ledger to census-v3 bridge failure as a separate
diagnostic rather than altering old evidence or weakening a validator.
