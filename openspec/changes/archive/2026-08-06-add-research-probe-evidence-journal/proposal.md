## Why

Long-running research probes can finish expensive model work yet lose every
completed work-item result when a late metadata or terminal-finalization value
is not serializable. Existing per-request decode receipts and atomic final
artifact families attest individual inference calls, but they do not provide a
pre-accelerator evidence-plan check or durable progress across probe work
items.

## What Changes

- Add a small execution-evidence journal that validates and persists an
  immutable run envelope and expected work-item plan before accelerator work.
- Require recursively strict canonical JSON for persisted evidence: string
  mapping keys, finite numbers, and no implicit `default=str` coercion of live
  Python objects.
- Serialize and durably append each completed work-item record independently,
  with stable identity and content digests, before the next item can become the
  only in-memory owner of that evidence.
- Publish terminal completion or failure separately from completed records, so
  a late finalizer failure cannot erase readable prior work. Resume exposes
  completed work-item identities only when run and plan identities match
  exactly; it never retries work or decides whether evidence is scientific.
- Let inference artifact publication carry an optional caller-owned opaque
  execution context and journal reference through successful, terminal-failure,
  sharded, and merged paths. Invalid context fails before model work, and shard
  context disagreement fails closed.
- Add a production-shaped CPU write/read/crash fixture and a minimal real-case
  integration gate. These establish mechanics only and do not count as model
  quality or mechanism evidence.
- Keep research-owned cohort construction, conditioning, intervention,
  estimand, thresholds, unmatched meaning, arm comparisons, claim boundaries,
  and stop rules outside the capability.
- Do not add a generic probe runner, experiment schema, hook framework,
  scheduler, reducer, automatic retry policy, or GPU resource broker.

## Capabilities

### New Capabilities

- `coordexp-swift-execution-evidence-journal`: Strict evidence-plan preflight,
  independently durable work-item records, terminal publication, validation,
  and exact-identity resume discovery without scientific interpretation.

### Modified Capabilities

- `coordexp-swift-infer-scoring-artifacts`: Preserve optional opaque execution
  context and journal references across success, failure, shards, and merge;
  reject non-JSON evidence instead of silently stringifying it.

## Impact

- `src/artifacts/` gains the shared strict canonical-JSON and execution-journal
  owners, with CPU contract and interruption tests under `tests/artifacts/`.
- `src/inference/artifacts.py`, `src/inference/pipeline.py`, and
  `src/inference/merge.py` adopt the strict evidence envelope and preserve its
  digest through existing artifact families; focused inference tests cover
  compatibility and shard disagreement.
- Existing valid training and inference artifacts remain readable. Callers
  that currently rely on unsupported values being converted to strings must
  instead provide an explicit receipt mapping before execution.
- Research runners may adopt the journal as an outer execution dependency, but
  their scientific record schemas and result interpretation remain local.
