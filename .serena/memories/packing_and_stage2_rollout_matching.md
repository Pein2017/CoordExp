# Packing + Stage2 Runtime (Memory)

Role separation:
- Memory role: quick runtime semantics for packing and stage2 trainer behavior.
- Canonical docs: `docs/data/PACKING.md`, `docs/training/STAGE2_RUNBOOK.md`, `docs/training/METRICS_LOSSES.md`.
- Canonical runtime code: `src/sft.py`, `src/trainers/rollout_matching_sft.py`, `src/trainers/stage2_ab_training.py`.
- Update trigger: when packing parser keys, stage2 namespaces, or trainer packing mechanics change.

Packing reminders:
- Packing knobs are parsed in `src/sft.py` runtime (`training.packing*`, `training.eval_packing`).
- Packing length is derived from template/model max length.
- `training.packing_length` is unsupported by schema validation.
- Non-stage2 packing forces train batch size 1 and uses dataset packing wrappers.

Stage2 reminders:
- Applies to `rollout_matching_sft` and `stage2_ab_training`.
- Dataset-level packing wrappers are disabled; stage2 uses post-rollout trainer-internal packing.
- Rollout config contract is top-level `rollout_matching.*`.
- `custom.extra.rollout_matching.*` is rejected and should never be used.
