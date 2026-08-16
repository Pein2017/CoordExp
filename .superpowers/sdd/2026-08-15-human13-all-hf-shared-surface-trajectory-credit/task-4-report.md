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

The bounded CUDA seam is now explicit in
`scripts/research/human13_cuda_cpu_adapter.py`.  It accepts either an injected
objective or the admitted Task-2 sampled/replay groups plus Task-3 trajectory
and compiler ledgers, preserves model/parameter/optimizer/transaction object
identity, moves only detached witness-solver evidence to the live device, and
records Source/applied/restored parameter, version-counter, CUDA-RNG, and
transaction digests.  The one-update probe always rejects/restores the
transaction; a failed reject emits a typed terminal receipt without retry.
Full-model frozen parameters, buffers, module modes, registry identity, version
counters, and Task-2 model-config identity/backend/cache fields are included
in the source fingerprint and are restored or fail closed on drift; storage
dtype/device/layout drift emits a typed terminal restore receipt rather than
claiming an exact success.  Task-2 replay mapping keys, tensor identity,
device/layout, graph leaves, and content hashes are snapshotted and
revalidated around the realized-margin probe; drift is restored when possible
and never yields a sealed receipt.  Task-2/Task-3 objective binding includes
the admitted surface, full-model, witness, replay-group/tensor,
trajectory/compiler, denominator, and coefficient hashes.  Receipts are
sealed in-process and a consumer must present the exact issued object.  The
lower-level device-aware projection and apply wrappers live beside the
existing CPU owner; the CPU entry remains the default and its behavior is
regression-tested unchanged.  The injected realized-margin probe remains a
read-only measurement callback, with the replay-evidence guard enforcing that
precondition for the admitted Task-2 path.

The adapter's `human13_cuda_objective_binding.v1` is intentionally a converted
consumer-side binding, distinct from Task 3's
`human13_all_hf_objective_binding.v1`; Task 5 must issue that converted
`ProposalBinding`/sealed receipt from the production service lineage before
calling this seam.  No direct cross-schema reuse is claimed.

## Production-shaped Task-2 no-update witness

On 2026-08-16, the public assembly boundary was exercised with the frozen
image-1584 Source checkpoint/adapter/selected-token delta and the real manifest
prompt skeleton on GPU 0.  The assembly was BF16, FlashAttention-2, language-
only DoRA, eval-mode shared surface, with explicit `use_cache=False`.  Four
K16 groups sampled and then replayed through the same model object.  Replay
reconstructed every recorded sampler step (same active request IDs and causal
history length), selected only the needed causal logits, and used non-reentrant
activation checkpointing; no padding-based batch replay was used.

The no-update parity receipt was:

```text
sample_forward_count=463
replay_forward_count=463
total_forward_count=926
no_cache_forward_count=926
group parity: max_abs_error=0.0, mean_abs_error=0.0 for all four groups
cleanup_state=closed, cleanup_call_count=1, retained_graph_count=0
session_held_reference_count=0
```

The durable raw record is
`research/investigations/qwen3-vl-dense-enumeration/experiments/2026-08-16-human13-all-hf-shared-surface-trajectory-credit-vertical/no-update-k16-parity-witness.md`.
It records the capture-time worktree/config/manifest hashes and lists all four
sampled/replayed group hashes; its v2 recapture binds the receipt to the
post-cleanup current source/test tree as well.

The run performed no backward, optimizer step, private checkpoint write,
audit-model load, network action, or output-root write.  The configured output
root was not written by this run; it still contains a pre-existing stale
`run-reservation.json` for PID 377949 with zero recorded model actions and no
terminal/resource/parity fields.  This reservation is not claimed as the
witness above and remains an unresolved Task-5 recovery/ownership issue.  This
is a Task-2 no-update parity witness only; it does not close Task 4.6 or Task 5, because the production
Task-5 service, dual-GPU audit, private proposal/checkpoint ownership, durable
rollback receipt, and downstream consumer wiring remain unexercised.

## Verification evidence

Focused Task-4 tests pass with CPU/value doubles:

```text
27 passed — tests/research/test_run_human13_all_hf_shared_surface_vertical.py
```

The CUDA adapter focused suite passes 18 tests (including one CUDA test on the
available injected device and a skipped-device-safe path when CUDA is absent).
The adapter plus the CPU preservation and vertical owner suites pass 95 tests
with two expected CUDA-unavailable skips under
`CUDA_VISIBLE_DEVICES=''`.

The terminal seal is issued only by the guarded one-image path; a direct
attempt to seal a value without its private issuer is rejected, and full-panel
admission requires that issued in-process seal.

The default CLI was exercised against the leaf config and produced a terminal
receipt with zero model loads, forwards, backwards, optimizer steps, GPU
allocations, network actions, and output creations.  No live HF/model/GPU,
network, checkpoint, or output action was performed.

Task 4.6 remains intentionally unchecked.  The bounded adapter now proves the
lower-level CUDA objective/proposal/projection/apply/transaction seam with
injected values and the actual admitted Task-2/Task-3 fixture receipts, while
leaving Task 3's CPU-only `PreparedAllHFVertical` owner unchanged.  The
independent prelaunch review still holds the live handoff: production BF16 /
FlashAttention-2 model assembly, replay-tensor production, checkpoint/output
ownership, and Task-5 service wiring have not been exercised here and require a
separate live-state acceptance witness.

The prelaunch smoke reached the public CLI help/config/dry-run and the
injected Task-1/2/3 seams, but it is **HOLD** for the live handoff: Task 3's
current owner is deliberately CPU-only (its full model, trainable parameters,
replay tensors, and compiler tensors reject non-CPU devices), while Task 2's
production-shaped HF shared-surface session and `human13_live_model` assembly
bind the same BF16/FA2 model and replay tensors to CUDA.  No CPU-to-CUDA bridge
preserving autograd graph ownership, optimizer identity, and exact rollback is
claimed beyond the bounded adapter seam described above.  This is an
integration acceptance boundary, not evidence that a real model or GPU run
merely needs more time.  Task 4.6 and Task 5 stay open until the adapter is
wired to the production service receipts and a no-output live-state witness is
explicitly accepted.

The reviewer’s bounded Task-1/2/3/live-model/evaluator set (the focused suite
plus seven adjacent files) passes 256 tests after the strict parser/lineage and
distinct-proposal regressions.  A broader nine-file Task-1/2/3 and
live-model/evaluator run passes 280 tests including the focused suite.  The
exact 11-file smoke command below passes 330 tests:

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
Pytest: 330 passed, 2 warnings
```

The broad adjacent sets use CPU/value doubles; the adapter's CUDA case uses
only a tiny injected CUDA surface, not a live HF model.  Compileall, Ruff,
strict OpenSpec validation, Serena diagnostics, and Pyright (`0 errors, 0
warnings, 0 informations`) are clean.  No live model, network, checkpoint, or
output action was used for any diagnostic in the injected/static gates; the
separate production-shaped Task-2 no-update witness above is the sole real
model/GPU action and wrote no checkpoint, network artifact, or output.

The Task-4 boundary remains explicit: real model assembly, GPU reservation,
private checkpoint bytes, output-root reservation, and Task-5 live adapters are
not proven by these injected tests or the zero-action CLI dry-run.
