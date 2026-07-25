---
name: full-pipeline-smoke
description: Validate a CoordExp change through a production-like data, train-or-rollout, checkpoint, inference, evaluation, and artifact smoke rather than a narrow unit or config check.
---

# Full Pipeline Smoke

Use a production config with fewer samples. Preserve model, template, packing,
length, geometry, decoding, checkpoint, and artifact contracts unless one is the
factor under test.

## Design

1. Name the production configuration and changed behavior.
2. Choose the smallest sample and step count that reaches that behavior.
3. Force at least one checkpoint save when production saves.
4. Force at least one evaluation step when production evaluates.
5. State the artifacts and counters that will prove each stage before launch.

Allowed smoke overrides are output location, sample limits, enough steps to
cross required save/eval boundaries, shortened save/eval cadence with unchanged
mode, and stability settings such as zero loader workers. Preserve optimizer,
learning rate, template, packing, maximum lengths, decode policy, checkpoint
mode, and evaluation semantics unless explicitly under test.

## Required Path Through The System

Exercise every conclusion-bearing stage that applies:

- data read and sample contract;
- multimodal template and encoding;
- packing and position identifiers when enabled;
- forward/backward/optimizer or rollout step;
- checkpoint save and load-adjacent behavior;
- inference, parsing, matching, scoring, and evaluation;
- logs, metrics, runtime identity, and reproducibility artifacts.

Discover the live launcher from current authority and CLI help. Run in the
checkout's configured environment and preserve the effective runtime identity.
Worktree-local data or model links may restore expected logical locations, but
do not stage runtime links.

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

For server-mode rollouts, verify local endpoint reachability and proxy bypass.
For coordinate experiments, bind serialization and evaluation to the checkpoint
that was trained on that representation; do not transfer evidence across
incompatible coordinate surfaces.

Report production identity, smoke reductions, sample scope, checkpoint,
backend and launch shape, stages crossed, artifacts and metrics checked,
failures or skips, and the exact claim this smoke supports.
