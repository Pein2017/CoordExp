# Task 4 report — guarded dual-GPU entry and continuation gate

## Scope

Task 4 now has a value-only leaf config and a guarded one-image entry at
`scripts/research/run_human13_all_hf_shared_surface_vertical.py`.  The entry
is dry-run by default.  A live caller must provide `--execute`, explicit
`--user-model-gpu-authority`, two distinct suitable cards (GPU 0 training /
GPU 1 HF fp32/SDPA batch-one audit), and a newly confirmed-absent output root.
The entry exposes only injected Task-1/2/3 service seams; it does not load a
model, reserve a card, write an output root, or execute the one-image update.

The audit analyzer delegates prediction ordering, duplicate exclusion, and
one-to-one owner matching to the canonical Human-13 K-union analyzer.  It
reports per-RP source/proposal owner IDs, H gains, G losses, M gains, net
unique delta, duplicate/unmatched/malformed burdens, cap stops, rows, and
tokens.  The continuation receipt requires the exact H-positive/net-positive
RP-1.0 result, zero own-G loss under both audits, and non-increasing duplicate,
malformed, and cap burdens.  Full-panel admission remains hash-bound to a
passing one-image terminal and is model/GPU inert.

All injected failure paths attempt private proposal cleanup and Source
reproduction/rollback before closing borrowed sessions.  Retry, fallback, and
checkpoint promotion are forbidden in the resource and terminal receipts.

The fail-closed correction bundle also joins prebuilt resources to exact GPU 0
training/GPU 1 audit roles; validates the manifest image and its parent binding
envelope, source checkpoint/adapter/embedding paths and hashes,
tokenizer/prompt/panel/image provenance, canonical matcher identity, output
RP/image/provenance/stop fields, canonical parser/status and arm/trajectory/run
lineage, required checkpoint paths, distinct Source and proposal checkpoint
payload/path identities, and disjoint G/H/M owner strata before update
admission.  Partial session opens and writer failures close/clean
borrowed state, with one rollback attempt and no retry/fallback.  Full-panel
admission requires an issued in-process sealed passing terminal, not an
arbitrary matching hash or forged receipt.

## Verification evidence

Focused Task-4 tests pass with CPU/value doubles:

```text
27 passed — tests/research/test_run_human13_all_hf_shared_surface_vertical.py
```

The terminal seal is issued only by the guarded one-image path; a direct
attempt to seal a value without its private issuer is rejected, and full-panel
admission requires that issued in-process seal.

The default CLI was exercised against the leaf config and produced a terminal
receipt with zero model loads, forwards, backwards, optimizer steps, GPU
allocations, network actions, and output creations.  No live HF/model/GPU,
network, checkpoint, or output action was performed.

Task 4.6 remains intentionally unchecked pending the requested independent
bounded prelaunch review.  Task 5 owns any future live-state reservation and
execution decision.

The prelaunch smoke reached the public CLI help/config/dry-run and the
injected Task-1/2/3 seams, but it is **HOLD** for the live handoff: Task 3's
current owner is deliberately CPU-only (its full model, trainable parameters,
replay tensors, and compiler tensors reject non-CPU devices), while Task 2's
production-shaped HF shared-surface session and `human13_live_model` assembly
bind the same BF16/FA2 model and replay tensors to CUDA.  No CPU-to-CUDA bridge
preserving autograd graph ownership, optimizer identity, and exact rollback is
implemented or proven.  This is an integration contract decision, not evidence
that a real model or GPU run merely needs more time.  Task 4.6 and Task 5 stay
open until the user chooses either a CUDA-capable Task-3 owner, a new explicit
CUDA adapter/owner, or a documented stop at the injected boundary.

The reviewer’s bounded Task-1/2/3/live-model/evaluator set (the focused suite
plus seven adjacent files) passes 256 tests after the strict parser/lineage and
distinct-proposal regressions.  A broader nine-file Task-1/2/3 and
live-model/evaluator run passes 280 tests including the focused suite.  The
exact 11-file smoke command below passes 328 tests:

```text
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 pytest -q \
  tests/research/test_run_human13_all_hf_shared_surface_vertical.py \
  tests/research/test_human13_hf_shared_surface.py \
  tests/research/test_human13_hf_shared_surface_live.py \
  tests/research/test_human13_all_hf_vertical.py \
  tests/research/test_human13_trajectory_credit.py \
  tests/research/test_human13_greedy_compiler.py \
  tests/research/test_human13_adamw_proposal_preservation.py \
  tests/research/test_human13_training_transaction.py \
  tests/research/test_human13_live_model.py \
  tests/research/test_human13_live_eval.py \
  tests/research/test_analyze_human13_k_union.py
Pytest: 328 passed, 2 warnings
```

All sets use CPU/value doubles only.  Compileall, Ruff, strict OpenSpec
validation, Serena diagnostics, and Pyright (`0 errors, 0 warnings, 0
informations`) are clean.  No model/GPU/network action was used for any
diagnostic.

The Task-4 boundary remains explicit: real model assembly, GPU reservation,
private checkpoint bytes, output-root reservation, and Task-5 live adapters are
not proven by these injected tests or the zero-action CLI dry-run.
