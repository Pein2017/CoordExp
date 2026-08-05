# Streamline CoordExp-Swift Base Infrastructure

## Why

A 2026-08-03 full-infrastructure audit of this worktree found no confirmed
correctness defect on the production training or inference paths, but it
identified four structural efficiency losses and one weight/exactness backlog
(actual performance measurement is owned by M0-M6 in `measurement-plan.md`):

- The training loop has no CPU/GPU overlap: every micro-step re-reads,
  re-hashes, re-decodes, and re-preprocesses its images synchronously between
  GPU steps (`src/qwen/forward.py:272-296`, `src/qwen/images.py:604-658`),
  every epoch presentation.
- Rank cache loading decodes every chunk on every rank even though each rank
  needs only its schedule's indices (`src/training/pack_cache.py:352-389`);
  production caches are 11-12 GB with world size 8.
- `eval.forward` replicates the full eval set on every rank
  (`src/training/pipeline.py:566-568`, `src/eval/forward.py:132-198`), so eval
  wall time never benefits from data parallelism.
- Dataset `mtime_ns` sits inside semantic cache determinants next to the
  strictly stronger content SHA256 (`src/training/pack_cache.py:145`), so a
  `touch` or timestamp-losing copy forces a multi-hour rebuild; the cache root
  currently holds three near-identical 11-12 GB caches.
- Exactness/weight backlog: train `acc_top1`/`acc_top5` cross-rank reduction is
  mean-of-rank-means rather than atom-weighted; the schema default
  `fa2_branch_proof: every_forward` puts per-layer host syncs on any config
  that omits the key while all production configs override it; run artifacts
  carry no step timing; and a production-dead non-streaming loss path plus
  several dead helpers remain as divergence liabilities.

## What Changes

This is a bounded efficiency, observability, exactness, and code-weight pass.
It preserves all proven training/inference semantics: renderer, tokenization,
supervision boundaries, packed FA2/MRoPE construction, loss formulas, backend
gradient scaling, checkpoint payloads, prompt parity, coordinate spaces, and
every existing fail-closed validation remain untouched.

Primary (correctness-preserving efficiency):

1. Bounded one-step CPU lookahead for Qwen forward-input/image
   materialization, overlapped with the current GPU step. No GPU work in the
   producer, queue depth exactly one, no ordering or semantic change,
   deterministic error propagation at the affected step, strict image
   identity/hash validation preserved, bounded host memory.
2. Rank-selective pack-cache loading: skip chunks whose declared index range
   does not intersect the rank's required indices. Every chunk that is decoded
   remains SHA256-verified and restricted-unpickled exactly as today. Trusting
   prepare-time verification instead is explicitly rejected as baseline and
   recorded only as a separately justified deferred option.
3. Rank-sharded `eval.forward` with exact global aggregation: every durable
   scalar in the canonical eval row has a defined exact sufficient statistic
   and reducer (global denominators plus summed partial numerators for
   segment-balanced terms; summed integer correct/atom counts for top-k;
   count-weighted sums for token-weighted diagnostics; global sums for
   example/pack counts; single emission for global-denominator-derived
   fields), so the entire row and best-checkpoint selection match the current
   fully replicated evaluator within declared numerical tolerance.
4. Remove dataset `mtime_ns` from semantic cache determinants; the dataset
   content SHA256, byte size, resolved path, and all data/template/code/
   processor/image-plan/packing determinants remain. Cache reuse is not
   broadened across changed content.

Secondary (independently gated):

- Default `model.fa2_branch_proof` to `first_micro_step`; explicit
  `every_forward` remains available for debugging.
- Make train `acc_top1`/`acc_top5` exact across ranks by summing rank-local
  integer sufficient statistics (correct counts and local atom counts) before
  forming the ratio; never by weighting with an already-global count or
  reconstructing counts from rounded ratios.
- Add low-overhead step-duration and input-build timing fields to the normal
  wide train logging row, reduced as the all-rank maximum (critical-path
  semantics) through the existing metric collective.
- Reduce gradient finite-gate host synchronization only behind proven decision
  and diagnostic equivalence plus a measured benefit.
- Delete the production-dead non-streaming loss path after a current
  call-graph/consumer proof; keep one canonical streaming path for train and
  eval; port tests instead of retaining a dead branch.
- Remove confirmed dead helpers and the duplicated forward-result
  synchronization calls; consolidate tiny duplicated helpers only where
  ownership is already clear; no generic utilities layer.
- Hoist the per-pack example-id map rebuild and adopt bisect-based token-span
  lookup only with exact boundary-equivalence tests.

## Non-Goals

- No inference architecture rewrite; inference scoring interval redesign is
  out of scope absent current evidence of need.
- No change to renderer, tokenization, supervision, packed FA2/MRoPE, loss
  formulas, or backend gradient scaling semantics.
- No optimizer-state resume.
- No epoch-dependent augmentation or ordering redesign.
- No compatibility layers for hypothetical consumers.
- No weakening of cache, artifact, checkpoint, prompt, coordinate, or failure
  validation.

## Evidence Vs. Proposal Separation

- Current evidence: audit findings above, live cache inventory
  (`.cache/coordexp_swift`, 50 GB, three 11-12 GB near-duplicates), production
  configs (`configs/coordexp_swift/prod/*.yaml`), and the completed
  `stream-distributed-pack-cache-runtime` measurements (about 10.5 minutes of
  GPU-idle admission at eight ranks before its fix).
- Proposed behavior: the delta specs in this change.
- Implementation direction: `design.md`.
- Deferred work: prepare-time-trust chunk verification, inference-side
  refactors, and any slice whose measured gate fails (see stop rules).

## Measured Acceptance, Stop, And Rollback Rules

Correctness-preserving requirements (exact aggregation, determinant hygiene,
verification preservation, deletion proofs) gate on tests and equivalence
probes. Benchmark-dependent optimizations (lookahead, rank-selective loading,
eval sharding, finite-gate batching, encode micro-optimizations) additionally
gate on measured improvement on representative production-shaped smokes
defined in `measurement-plan.md`. No speculative speedup numbers are encoded
as requirements. If a benchmark-dependent slice does not improve its declared
end-to-end wall-clock measurement, that slice is reverted and recorded as
rejected-with-evidence; the change remains valid without it. Implementation
may stop after any completed wave.

## Capabilities

### New Capabilities

None. All changes modify or extend existing stable capabilities.

### Modified Capabilities

- `coordexp-swift-packing-forward`: add the bounded forward-input lookahead
  contract; allow rank-selective chunk consumption inside the single validated
  payload pass.
- `coordexp-swift-pack-cache-semantic-identity`: exclude dataset filesystem
  timestamps from semantic identity; add the rank-selective loaded-chunk
  verification contract.
- `coordexp-swift-supervision-losses`: make cross-rank top-k accuracy
  reduction atom-weighted.
- `coordexp-swift-training-artifacts`: allow rank-sharded eval.forward with
  exact global aggregation; add step-timing fields to the wide train row.
- `coordexp-swift-config-runtime`: default packed-forward proof capture to one
  representative first-micro-step capture.

## Sequencing Dependency

The completed change `stream-distributed-pack-cache-runtime` remains untouched
and MUST be synced/archived before this change's
`coordexp-swift-packing-forward` and
`coordexp-swift-pack-cache-semantic-identity` deltas are applied or archived.
That sync is a blocking reconciliation, not a mechanical archive: its
`Deterministic Packing Cache Reuse` and `Cache Payload Is Current-Version-Only`
deltas carry fewer scenarios than the current stable requirements, so the
union of stable scenarios (same-template relaunch, renderer code change,
production cache miss, worker count, older payload version, old-version and
incomplete-manifest rejection) MUST be preserved at sync time. To make later
archive order safe regardless, this change's `Deterministic Packing Cache
Reuse` MODIFIED delta (in `coordexp-swift-packing-forward`) and its
`Cache Payload Is Current-Version-Only` MODIFIED delta (in
`coordexp-swift-pack-cache-semantic-identity`) each carry the full union of
stable scenarios, the prior change's scenarios, and this change's new
rank-selective-loading scenarios, qualified so that only chunks required by a
rank's resolved schedule need be decoded and digest-verified — a corrupt
payload outside that required set MAY be skipped by this rank, while any rank
that requires it MUST still fail closed. The delta text was drafted against
the live code inspected at commit `a19ffceeb`; implementation revalidates
against live HEAD (tasks 0.1) rather than treating that commit as authority.

## Impact

- Code: `src/training/pipeline.py`, `src/training/pack_cache.py`,
  `src/training/supervised_trainer.py`, `src/eval/forward.py`,
  `src/losses/runner.py`, `src/runtime/train_runtime.py`,
  `src/runtime/finite_gates.py`, `src/qwen/forward.py`, `src/qwen/encoding.py`,
  `src/config/models.py`, `src/artifacts/run_writer.py`, plus focused tests.
- Artifacts: additive-only fields in `logging.jsonl` train rows; no field is
  renamed or removed; eval row schema unchanged.
- Caches: any edit to determinant fields or to `code_identity` source files
  changes semantic fingerprints, so it is the final settled source state of
  this change — not the `mtime_ns` removal alone — that induces one new
  production fingerprint. Intermediate measurement work runs against
  dedicated temporary cache roots and sample-limited derived configs so the
  shared production cache is never churned; exactly one production
  materialization for the new fingerprint happens after all cache-determinant
  sources settle, in a user-authorized window. Existing cache directories
  (including the current 50 GB root) are never deleted or mutated
  automatically; they remain historical/disposable state whose cleanup is
  user-owned. No cache format version change; invalid caches still fail
  closed.
- Configs: no production config edits required; omitting `fa2_branch_proof`
  changes meaning from `every_forward` to `first_micro_step`. All production
  configs already set it explicitly; the eight smoke configs that omitted it
  were pinned to `every_forward` during implementation so their resolved
  behavior did not change.
- User-owned decisions: none beyond those resolved in the lead review, except
  authorizing the single production cache-materialization window above
  (launch/material cost).
