# Handoff: Task 4.6 / Task 5 boundary

## Semantic supersession (2026-08-20)

The former cross-surface exact-token/coordinate-alias admission rule is
superseded by the active OpenSpec/design decision.  The immutable
`one-image-successor-20260820T-reconcile-v2` artifact is retained as the
evidence that fp32/SDPA and BF16/FA2 are distinct policies: BF16 preserved all
protected G owners and additionally covered `gt:1584:12`, while other token
and owner differences were not coordinate quantization aliases.  Future
admission must therefore freeze independent baselines: BF16-native Source,
compiler, witness, and preservation inputs on GPU0; fp32/SDPA Source baselines
at RP 1.0 and 1.10 on GPU1.  Cross-surface divergence is diagnostic-only.
Strict BF16 sampler/replay parity, model/checkpoint/adapter/tokenizer/prompt/
image/manifest identity, protected BF16 G presence, K16, LR, objective,
dual-RP outcome gate, private proposal, rollback, and no-promotion remain
unchanged.

The current live diagnostic root is decision-bearing but pre-update only:

`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-15-human13-all-hf-shared-surface-trajectory-credit-vertical/one-image-successor-20260820T-reconcile-v2`

It recorded RP1.0 pass, RP1.1 diagnostic divergence, 204 source-audit
forwards, zero sample/replay/backward/update/checkpoint actions, and a clean
source-only failed close.  Its compact evidence is under
`receipts/006-source_audit_rp_1.1.json`.

Date: 2026-08-16 UTC

## Objective and decision

Continue the frozen OpenSpec change
`add-human13-all-hf-shared-surface-trajectory-credit-vertical` from the
approved Task-2.5 boundary toward the real one-update vertical.  Task 2.5 is
closed.  Task 4.6 and Tasks 5.1--5.5 remain intentionally open: the real
no-update parity path is proven, but no update/audit/rollback/publication claim
is admitted.

## Current authority and evidence

- Exact tree: commit `0e647270db16d92ac5d590d0b7fffd529b010c38`.
- OpenSpec authority: `openspec/changes/add-human13-all-hf-shared-surface-trajectory-credit-vertical/`.
- Task-2 report: `.superpowers/sdd/2026-08-15-human13-all-hf-shared-surface-trajectory-credit/task-2-report.md`.
- Task-4 report: `.superpowers/sdd/2026-08-15-human13-all-hf-shared-surface-trajectory-credit/task-4-report.md`.
- Durable current-tree K16 receipt:
  `no-update-k16-parity-witness.md` in this directory.  The v2 recapture is
  bound to the current live/test hashes and records 463 sampling forwards,
  463 replay forwards, 926 total/no-cache forwards, four zero-error parity
  groups, one cleanup call, zero retained graphs, and zero post-close session
  group lists.
- Verification: focused live 50 passed; exact current 11-file smoke 330
  passed; Pyright 0; Ruff/compileall/Serena clean; strict OpenSpec 28/28.

## Closed boundary

The same admitted BF16/FA2 Qwen model object performs sampling and exact
sampler-step replay with position-selective logits and non-reentrant
checkpointing.  The witness is no-update only.  It does not consume or imply
backward, AdamW, private checkpoint, HF-fp32/SDPA audit, rollback, continuation
gate, output publication, or downstream consumer evidence.

## Open blocker and stop rule

The production service and split CUDA lifecycle now exist, but the next live
owner revision must replace the former cross-surface reconciliation gate with
surface-separated admission.  It still needs:

1. a BF16-native free-running Source projection on the same CUDA session;
2. BF16-native compiler Source boundary, remaining-owner state, frozen witness
   bank/Jacobians, and post-apply margin probe;
3. durable fp32/SDPA Source baselines at RP 1.0 and 1.10, with proposal audits
   compared only within that fp32 surface;
4. strict protected-BF16-G and internal sampler/replay parity admission, while
   retaining cross-surface divergence as diagnostic-only evidence;
5. the existing private checkpoint, rollback reproduction, durable receipt,
   and downstream/full-panel consumer wiring; and
6. append-only recovery of the stale configured `one-image/run-reservation.json`
   (PID 377949 is dead) under an explicit owner.  It must not be deleted or
   overwritten by a continuation run without a verified recovery receipt.

Do not mark 4.6 or 5.x complete until a production-shaped no-update receipt
binds the current tree, config, BF16-native Source/witness/compiler inputs,
fp32 baselines, resource cards, and diagnostic divergence.  Only then may the
same reserved root perform one K16 private update.  A missing protected BF16 G,
internal parity failure, or fp32 identity drift remains a typed HOLD; a
cross-surface token/owner difference alone does not.

## Minimal next reading path

1. OpenSpec `tasks.md`, `design.md`, and the Task-4/Task-2 reports above.
2. `scripts/research/run_human13_all_hf_shared_surface_vertical.py` for the
   guarded protocol and fail-closed ordering.
3. `scripts/research/human13_hf_shared_surface_live.py` for the admitted live
   sampler/replay resource receipt.
4. `scripts/research/human13_cuda_cpu_adapter.py` and
   `scripts/research/human13_all_hf_vertical.py` for the adapter/owner seam.
5. `no-update-k16-parity-witness.md` for the current real receipt and its
   stale-reservation boundary.

The unrelated untracked memory note
`memories/notes/2026-08-14-scalable-k-trajectory-successor-direction.md` is
preserved and is not part of this handoff.
