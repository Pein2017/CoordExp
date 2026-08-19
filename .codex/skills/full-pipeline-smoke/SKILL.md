---
name: full-pipeline-smoke
description: Use when a CoordExp change has material risk at a real entry, distributed, scale, checkpoint, artifact, finalizer, or downstream integration seam, before broad implementation or completion.
---

# Full Pipeline Smoke

Use a production config with fewer samples. Preserve model, template, packing,
length, geometry, decoding, checkpoint, and artifact contracts unless one is the
factor under test.

## Choose The Mode

- **Early risk-retirement slice:** before broad implementation, name the smallest
  set of execution risks that could invalidate the architecture and exercise
  each through the real entry, installed runtime, production-owned wrapper, and
  minimum representative data. A leaf helper, mock, or in-process shortcut does
  not close an integration risk.
- **Final frozen integration smoke:** after implementation is fixed, traverse the
  complete conclusion-bearing production path and bind evidence to that exact
  tree, runtime, and artifact set.

For the early slice, rank risks by decision impact and cheapest discriminator.
Stop broad implementation when a top risk fails. If a risk cannot yet be
exercised, record it as `unproven` with the evidence and authority required to
proceed; do not silently convert it to passed.

## Design

1. Name the production configuration and changed behavior.
2. Name the top execution risks and the exact real seam that exposes each one.
3. Choose the smallest sample, rank count, and step count that reaches those
   seams without changing their execution shape.
4. Force at least one checkpoint save when production saves.
5. Force at least one evaluation step when production evaluates.
6. State the artifacts, counters, and resource bounds that prove each stage
   before launch.

Allowed smoke overrides are output location, sample limits, enough steps to
cross required save/eval boundaries, shortened save/eval cadence with unchanged
mode, and stability settings such as zero loader workers. Preserve optimizer,
learning rate, template, packing, maximum lengths, decode policy, checkpoint
mode, and evaluation semantics unless explicitly under test.

## Scale And Resource Contract

Before scaling data, ranks, or GPUs, declare the relevant bounds and measure a
representative case:

- model forwards per sample, pack, and rank, including counterfactual branches;
- wrapped versus unwrapped calls, collective sequence, and optimizer mutation;
- cache or materialization full passes, representative bytes, and I/O
  amplification;
- wall time, peak RSS, worker count, and any concurrency cap;
- per-rank and merged artifact payload size;
- process-tree ownership, heartbeat and exit evidence, finalizer ownership, and
  downstream reload or consumption;
- activation count, partial-activation behavior, and recovery ceiling when the
  operation is at most once.

An implementation is not scale-ready when these quantities are unknown or
unbounded. For an irreversible or at-most-once action, follow the
"Multi-agent orchestration" section of AGENTS.md (decision authority and
model routing priors) for executor and recovery authority.

## Required Path Through The System

Exercise every conclusion-bearing stage that applies:

- data read and sample contract;
- multimodal template and encoding;
- packing and position identifiers when enabled;
- forward/backward/optimizer or rollout step;
- checkpoint save and load-adjacent behavior;
- inference, parsing, matching, scoring, and evaluation;
- canonical serialization, hashing, atomic publication, fresh reload, and the
  production downstream consumer;
- logs, metrics, runtime identity, process exit, and reproducibility artifacts.

Discover the live launcher from current authority and CLI help. Run in the
checkout's configured environment and preserve the effective runtime identity.
Worktree-local data or model links may restore expected logical locations, but
do not stage runtime links.

When behavior can vary by rank, exercise at least one real multi-process case
whose ranks have intentionally different local work while preserving the
required collective choreography. A single-rank result cannot close that risk.

## Acceptance

- The run reaches its planned terminal step, not only the first optimizer step.
- The checkpoint contains the same class of model, adapter, and auxiliary
  modules that production saving promises.
- When the change affects checkpoint payload, loading, resume, or downstream
  behavior, start a fresh process, load the saved checkpoint through the
  production-owned path, and execute at least one conclusion-bearing forward or
  evaluation step. Verify required payload keys, shapes, dtypes, and behavioral
  integrity at the tolerance owned by the feature.
- Evaluation runs when production enables it, emits finite metrics, and receives
  normal model outputs.
- Feature metrics appear under the current namespace; retired or debug-only
  namespaces are absent.
- Effective runtime evidence records launch shape, batch/accumulation, save and
  eval cadence, and packing state.
- Inference/evaluation outputs are complete, metric-bearing, and traceable to
  the run identity rather than merely present after process exit.
- Canonicalization, publication, reload, and the downstream consumer succeed on
  a non-empty production-shaped artifact before expensive broad execution.
- Measured resource quantities remain within the declared bounds; a correct but
  unbounded or repeatedly materialized path is a failed scale gate.

For server-mode rollouts, verify local endpoint reachability and proxy bypass.
For coordinate experiments, bind serialization and evaluation to the checkpoint
that was trained on that representation; do not transfer evidence across
incompatible coordinate surfaces.

Report production identity, smoke reductions, sample scope, checkpoint,
backend and launch shape, stages crossed, artifacts and metrics checked,
failures or skips, and the exact claim this smoke supports.
