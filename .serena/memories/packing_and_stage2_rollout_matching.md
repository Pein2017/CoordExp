# Packing and Stage-2 (Rollout-Matching) Notes

Packing (stage-1 / standard trainers):
- YAML knobs live under `training.*` and are parsed in `src/sft.py::_parse_packing_config()`:
  - `training.packing` (bool), `training.packing_buffer`, `training.packing_min_fill_ratio`,
    `training.packing_drop_last`, `training.packing_allow_single_long`, `training.eval_packing`.
  - `training.packing_length` is deprecated; prefer `global_max_length` -> `template.max_length`.
- `ConfigLoader.build_train_arguments()` strips packing-only knobs before TrainArguments init and stores them in `train_args._packing_overrides`.
- When packing is enabled and trainer_variant != `rollout_matching_sft`, `src/sft.py`:
  - forces `per_device_train_batch_size=1`,
  - wraps the dataset with `PackedCaptionDataset` (`src/datasets/wrappers/packed_caption.py`),
  - requires a finite `max_steps` (auto-estimated only if `len(dataset)` is known).
- Eval packing is opt-in via `training.eval_packing`; eval batch size is also forced to 1.

Stage-2 rollout matching (`custom.trainer_variant: rollout_matching_sft`):
- Trainer lives in `src/trainers/rollout_matching_sft.py`.
- Pipeline: rollout (no grad) -> strict parse -> match -> build one teacher-forced target -> masked losses.
- Packing is supported POST-ROLLOUT ONLY (trainer-internal). Dataset-level packing wrappers are explicitly disabled.
- Canonical runbook: `docs/training/STAGE2_RUNBOOK.md`.
- Metrics reference: `docs/training/METRICS_LOSSES.md` (keys under `rollout/*`, `packing/*`, `time/*`).
- Most stage-2 knobs live under `custom.extra.rollout_matching.*` (rollout backend, decoding, matching thresholds, buffering, offload, desc monitor).
