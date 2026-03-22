# Runtime Footguns (Memory)

Role separation:
- Memory role: quick recall for runtime and config mistakes that are easy to miss during implementation.
- Canonical docs: `docs/SYSTEM_OVERVIEW.md`, `docs/IMPLEMENTATION_MAP.md`, `docs/data/CONTRACT.md`, `docs/training/STAGE1_OBJECTIVE.md`, `docs/training/STAGE2_RUNBOOK.md`, `docs/training/METRICS.md`, `docs/eval/WORKFLOW.md`, `openspec/specs/runtime-architecture-refactor-program/spec.md`.
- Update trigger: when config contracts, trainer variants, packing behavior, or infer/eval entrypoints change.

High-signal reminders:
- Training is YAML-first; `template` and `custom` are required, and unknown keys fail fast in `src/config/schema.py`.
- Keep `data.dataset: ["dummy"]` as the ms-swift placeholder; dataset source comes from `custom.train_jsonl` or legacy `custom.fusion_config`.
- Default posture is single-dataset training; `custom.fusion_config` is legacy or experimental.
- Offline-prepared JSONL is the contract: keep `custom.emit_norm: none`, pre-normalize coords to norm1000 or coord tokens, and keep image paths relative.
- When `custom.offline_max_pixels` is authored, launcher/dataset prechecks use it as the offline image-budget contract rather than `template.max_pixels`.
- Coord-token enablement and coord distribution loss are independent; changing one does not automatically enable the other.
- `training.packing_length` is unsupported; use `global_max_length` or `template.max_length`.
- For `custom.trainer_variant: stage2_two_channel`, `training.effective_batch_size` is required and must divide learner world size.
- `custom.trainer_variant: stage2_ab_training` is removed; use `stage2_two_channel`.
- `custom.trainer_variant: rollout_matching_sft` is removed; use `stage2_rollout_aligned`.
- `custom.trainer_variant: stage2_rollout_aligned` must author `rollout_matching.pipeline.*`; `stage2_ab.pipeline.*` is invalid there.
- Removed self-context-era knobs such as `stage2_ab.n_softctx_iter`, `stage2_ab.coord_ctx_embed_mode`, `stage2_ab.coord_decode_mode`, and `rollout_matching.coord_decode_mode` should fail fast rather than be silently ignored.
- Stage-2 Channel-B uses the clean-prefix contract; the raw rollout prefix is diagnostic-only, not the positive teacher-forced source.
- Put rollout settings under top-level `rollout_matching.*`; never under `custom.extra.rollout_matching.*`.
- Server-mode Stage-2 launches go through `scripts/train_stage2.sh` and `src.launchers.stage2_vllm_server`; GPU topology is split by `server_gpus` vs `train_gpus`, not by new CLI flags.
- Refactored runtime ownership is now split across `src/bootstrap/`, `src/infer/{engine,artifacts,backends}.py`, `src/eval/{detection,artifacts,orchestration}.py`, and `src/trainers/rollout_runtime/`.
- `custom.trainer_variant: stage2_rollout_aligned` is still supported, but rollout-target/eval ownership now routes through `src/trainers/rollout_aligned_targets.py` and `src/trainers/rollout_aligned_evaluator.py` rather than staying fully inline.
- The YAML infer pipeline writes `resolved_config.path` next to `gt_vs_pred.jsonl`; use it to recover authoritative run metadata when artifacts are consumed outside `run_dir`.
- Offline evaluator callback metrics log under `eval_det_*`; trainer-native Stage-2 rollout eval logs under `eval/detection/*`, `eval/parsing/*`, `eval/description/*`, `eval/config/*`, and `eval/runtime/*`.
