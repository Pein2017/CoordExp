## Why

The predecessor K-trajectory experiment stopped before any update because
vLLM sampling and HF replay could not satisfy its frozen exact-policy parity
contract.  The core algorithm therefore remains untested.  For the selected
Human-13 overfit probe, it is worth spending more sampling compute to put
sampling, replay, and gradient construction on one trainable HF numerical
surface and reach a real one-update behavioral result.

## What Changes

- Add an experiment-local all-HF stepwise sampler and vectorized replay that
  use the same live Source model object, DoRA representation, BF16 dtype,
  FlashAttention-2 backend, model mode, repetition-penalty transform, and
  no-cache causal-forward implementation.  A parity gate owns the remaining
  stepwise-versus-teacher-forced shape difference.
- Keep K=16 and four logical groups of four trajectories, but deliberately
  disable sampler KV/prefix reuse and accept repeated image/prompt computation.
  Replay batches the four completed trajectories so backward remains feasible.
- Add a one-image image-1584 admission slice at training RP 1.0.  It first
  checks semantic lineage and the predecessor's unchanged `0.02` maximum and
  `0.002` mean chosen-token processed-logprob tolerances, then immediately
  performs one full trajectory-credit + sparse greedy-compiler + preservation
  update if the shared surface passes.
- Audit that private proposal with clean greedy at RP 1.0 and RP 1.10, report
  H gained, G lost, net unique owners, duplication, malformed rows, STOP/cap
  burden, and restore Source exactly.  No checkpoint is promoted.
- Add a conditional full-panel successor: only a positive, protected
  one-image proposal may authorize one 13-image K16 shared-surface update.
- Reuse the predecessor's pure trajectory ledger, compiler, AdamW proposal,
  preservation, matcher, transaction, and analyzer semantics; do not reopen or
  reinterpret its verified cross-engine negative result.
- Keep the implementation bounded and research-local: no vLLM path, generic RL
  trainer, multi-update loop, K-miss supervision, production default, or
  validation/generalization claim.

## Capabilities

### New Capabilities

- `coordexp-swift-human13-all-hf-shared-surface-trajectory-credit-vertical`:
  Experiment-local contracts for same-surface HF K16 sampling and replay,
  one complete private update with trajectory credit/compiler/preservation,
  dual-RP greedy audit, exact rollback, and conditional 13-image continuation.

### Modified Capabilities

None.  Existing production training, HF inference, vLLM inference, optimizer,
checkpoint, and evaluation contracts remain unchanged.

## Impact

- New research unit, OpenSpec change, Superpowers design/plan, leaf configs,
  all-HF sampler/replay receipts, focused tests, and one production-shaped
  one-image vertical entry.
- Reuse of `human13_trajectory_credit.py`, `human13_greedy_compiler.py`, the
  RP-crossover proposal/preservation/runtime helpers, Human-13 model assembly,
  clean-greedy evaluator, matcher, and training-state transaction.
- Higher exploratory compute and wall time: no-cache full-history BF16/FA2
  forwards repeat image and prompt work for sampling and gradient replay.
- No external dependency, stable API change, accepted checkpoint, deployment
  behavior change, or population-level conclusion.
