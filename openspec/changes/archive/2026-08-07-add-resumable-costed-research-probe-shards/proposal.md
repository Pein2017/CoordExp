## Why

`scripts/research/run_natural_boundary_support_completion.py` currently keeps a
whole shard's observations in memory and publishes its receipt only after the
shard returns. A process interruption can therefore discard hours of completed
support contexts even though the accepted execution-evidence journal already
provides independently durable work items and exact-identity continuation.

The sealed 200-context plan also embeds a legacy hash partition whose largest
shard costs 13,867 scalar forwards. A deterministic longest-processing-time
schedule over the same declared context costs has a 9,689-forward maximum on
the current plan, a 30.129% reduction, without changing the logical work or the
legacy receipt partitions consumed by the merger.

## What Changes

- Add a consumer-specific resumable execution adapter for the support-context
  lifecycle of `run_natural_boundary_support_completion.py`; this is not a
  general probe framework.
- Treat each of the sealed 200 contexts as one independently durable journal
  work item. An explicitly requested exact-identity continuation executes only
  contexts without accepted records.
- Separate the immutable logical work-item plan from a derived physical worker
  schedule. Derive the schedule deterministically from declared scalar-forward
  cost with stable tie-breaking, while preserving the plan's legacy
  `shard_index` solely as receipt-partition identity.
- Deterministically materialize the existing support-completion shard receipt
  schema from validated journal records, regrouped by legacy receipt shard, so
  the current merger retains its receipt contract and the current analyzer
  retains its separate census-v3 scientific contract. Do not repair or
  reinterpret a pre-existing ledger-to-census bridge defect.
- Add mechanics-only attempt, process-exit, signal, and last-durable-record
  diagnostics. Do not automatically retry, classify scientific outcomes, or
  treat a mechanics receipt as research evidence.
- Prove canonical terminal shard receipts are byte-identical between an
  uninterrupted run and an interruption followed by explicit exact-identity
  continuation. Preserve attempt-history differences only in separate
  mechanics diagnostics.
- Run a bounded single-GPU real `SIGTERM` then continuation smoke after CPU
  gates pass. The smoke proves durability and continuation mechanics only.
- Keep every existing sealed research root immutable. Do not recover or
  reinterpret lost v3/v4 in-memory results, and do not change the research
  question, support rule, denominator, checkpoint, prefix, candidate set, or
  outcome semantics.
- Deliver implementation only in this worktree as a fixed commit, interface
  note, adapter patch or consumer example, tests, and mechanics receipts. Do
  not merge or cherry-pick it into the active `research-probes` unit.

## Capabilities

### New Capabilities

- `coordexp-infras-natural-boundary-support-shards`: Consumer-specific logical
  planning, deterministic cost-aware physical scheduling, per-context durable
  execution, legacy shard-receipt materialization, and interruption mechanics
  for the natural-boundary support-completion runner.

### Modified Capabilities

- `coordexp-infras-execution-evidence-journal`: Add a validated read-only view of
  accepted record payloads, attempt state, and last durable record for caller
  diagnostics without changing the journal's on-disk schema or scientific
  authority.

## Impact

- Affected stable owner: `src/artifacts/evidence_journal.py`, limited to
  validated read-only projections and diagnostics.
- New worktree-local consumer adapter and tests under `scripts/research/` and
  `tests/research/`; the active consumer is referenced by exact file digest and
  is not edited in place.
- The adapter targets the current consumer bytes
  `sha256:9ade7ad861e7c822458f65fbdb39ddda702648d62339533a3faf5e0dd37a52b3`
  and current merger bytes
  `sha256:9eb7534641be4c87768698719ed84c2f84119959d9fa324b0fc5a72f7cd5edf1`.
- No new model, checkpoint, tokenizer, research-input, support-policy, merger,
  analyzer, training, or automatic-retry dependency is introduced.
- Existing journal directories remain readable; the journal disk schema stays
  at version 1.
