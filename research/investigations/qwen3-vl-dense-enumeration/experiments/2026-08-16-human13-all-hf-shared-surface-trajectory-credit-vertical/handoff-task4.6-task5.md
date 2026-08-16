# Handoff: Task 4.6 / Task 5 boundary

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

The guarded entry still has only an injected `OneImageServices` protocol.  No
production implementation is present for the complete Task-5 path.  The CUDA
adapter is a bounded seam, but production wiring still needs:

1. conversion from Task-3 `human13_all_hf_objective_binding.v1` to the adapter's
   `human13_cuda_objective_binding.v1`;
2. live Task-2 replay/trajectory/compiler/witness ownership on the same CUDA
   model and transaction;
3. GPU-1 HF fp32/SDPA source/proposal audit ownership;
4. private proposal checkpoint bytes, rollback reproduction, durable failure or
   success receipt, and downstream/full-panel consumer wiring; and
5. recovery of the stale configured `one-image/run-reservation.json` (PID
   377949 is dead) under an explicit owner.  It must not be deleted or
   overwritten by a continuation run without a verified recovery receipt.

Do not call `--execute`, do not perform an update, and do not mark 4.6 or 5.x
complete until these owners exist and a production-shaped dry-run/update
receipt binds the current tree, config, source assembly, resource cards,
checkpoint/output paths, and rollback consumer.  If the next task cannot
resolve the service and reservation ownership without changing research
meaning or execution policy, leave the tasks unchecked and report HOLD.

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
