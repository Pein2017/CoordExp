# Runtime Footguns (Memory)

Role separation:
- Memory role: quick recall for runtime and config mistakes that are easy to miss during implementation.
- Canonical docs: `docs/SYSTEM_OVERVIEW.md`, `docs/IMPLEMENTATION_MAP.md`, `docs/data/CONTRACT.md`, `docs/training/STAGE1_OBJECTIVE.md`, `docs/training/STAGE2_RUNBOOK.md`, `docs/training/METRICS.md`, `docs/eval/WORKFLOW.md`.
- Update trigger: when config contracts, trainer variants, packing behavior, or infer/eval entrypoints change.

High-signal reminders:
- Training is YAML-first; `template` and `custom` are required, and unknown keys fail fast in `src/config/schema.py`.
- Keep `data.dataset: ["dummy"]` as the ms-swift placeholder; dataset source comes from `custom.train_jsonl` or legacy `custom.fusion_config`.
- Default posture is single-dataset training; `custom.fusion_config` is legacy or experimental.
- Offline-prepared JSONL is the contract: keep `custom.emit_norm: none`, pre-normalize coords to norm1000 or coord tokens, and keep image paths relative.
- Coord-token enablement and coord distribution loss are independent; changing one does not automatically enable the other.
- `training.packing_length` is unsupported; use `global_max_length` or `template.max_length`.
- For `custom.trainer_variant: stage2_two_channel`, `training.effective_batch_size` is required and must divide learner world size.
- Stage-2 Channel-B uses the clean-prefix contract; the raw rollout prefix is diagnostic-only, not the positive teacher-forced source.
- Put rollout settings under top-level `rollout_matching.*`; never under `custom.extra.rollout_matching.*`.
- Offline evaluator metrics log under `eval_det_*`; trainer-native rollout eval logs under `eval_rollout/*`.
