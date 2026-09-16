## Why

The predecessor K-trajectory experiment stopped before any update because
vLLM sampling and HF replay could not satisfy its frozen exact-policy parity
contract.  The core algorithm therefore remains untested.  For the selected
Human-13 overfit probe, it is worth spending more sampling compute to put
sampling, replay, and gradient construction on one trainable HF numerical
surface and reach a real one-update behavioral result.

## What Changes

- Add an experiment-local all-HF stepwise sampler and sampler-step-aligned replay that
  use the same live Source model object, DoRA representation, BF16 dtype,
  FlashAttention-2 backend, model mode, repetition-penalty transform, and
  no-cache causal-forward implementation.  A parity gate owns the remaining
  stepwise-versus-teacher-forced shape difference.
- Keep K=16 and four logical groups of four trajectories, but deliberately
  disable sampler KV/prefix reuse and accept repeated image/prompt computation.
  Replay reconstructs each recorded active-batch/history-length step with
  position-selective logits and bounded activation checkpointing so one later
  backward remains feasible on the live 80-GB-class surface.
- Add a one-image image-1584 admission slice at training RP 1.0.  It first
  checks semantic lineage and the predecessor's unchanged `0.02` maximum and
  `0.002` mean chosen-token processed-logprob tolerances, then immediately
  performs one full trajectory-credit + sparse greedy-compiler + preservation
  update if the shared surface passes.
- Treat the BF16/FlashAttention-2 training surface as the sole authority for
  sampling, replay, trajectory credit, greedy compilation, preservation, and
  the update token/path.  Construct its Source projection, compiler boundary,
  witness bank, and post-apply margin probe from BF16-native free-running
  outputs; never feed fp32/SDPA token paths into the BF16 policy.  Keep the
  fp32/SDPA surface as a separate paired owner-level clean-greedy audit at both
  repetition penalties.  The shared choke point enforces strict identity only
  for model/checkpoint/adapter/embedding/tokenizer/prompt/image/manifest and
  declared processor policy.  Cross-surface token, row, coordinate, and owner
  differences are retained as `diagnostic_only` evidence and do not decide
  admission.  This restores the design boundary that the audit surface need
  not be numerically or token-identical to BF16/FA2, while BF16 sampler-to-
  replay parity remains strict and unchanged.  The superseded coordinate-alias
  artifact remains immutable as the evidence that invalidated that gate.
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

- `coordexp-infras-human13-all-hf-shared-surface-trajectory-credit-vertical`:
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

## Closeout disposition (2026-08-24)

This production-shaped route is retired without a completed private update or
model-quality result. Its live attempts established preflight, Source-audit,
and K16/replay infrastructure evidence, but the admitted path never reached
backward or an optimizer step. K16 receipts bind a real compiler ledger and
complete the compiler materialization task, while four canonical Source
baselines were durably admitted. No compiler objective/backward or proposal
audit followed. The final prelaunch, one-update/audit, independent result
audit, and conditional full-panel tasks remain intentionally unchecked. The
route's bounded results, research graph, project memory, strict validation,
residue check, and independent closeout review are complete only as
documentary retirement, not as an algorithm result or archive.

The later standalone probes deliberately removed this route's production
admission machinery and complete compiler/projection contract in order to
obtain direct scientific evidence. They do not complete this proposal. The
current N=13 K4/K8 conclusion is bounded to the exact Human-13 cohort, four
updates, LR `3e-6`, paired banks, and dual-RP physical-owner evaluation; see the
[continuity note](../../../memories/notes/2026-08-24-human13-n13-k4-k8-factorial-result.md).
It supports keeping K4 over K8 for a future redesigned objective, not resuming
this retired vertical, generalization, deployment, or checkpoint promotion.
