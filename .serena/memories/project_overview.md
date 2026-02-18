# CoordExp Overview (Memory)

Role separation:
- Memory role: quick routing for day-to-day implementation turns (what to open next).
- Canonical docs: `progress/full_idea.md`, `docs/data/README.md`, `docs/training/STAGE2_RUNBOOK.md`, `public_data/README.md`.
- Update trigger: when entrypoints, trainer variants, or default workflow posture changes.

Current workspace posture:
- Default: single-dataset training.
- Primary efficiency lever: packing.
- Fusion-config training: legacy/experimental.

Primary entrypoints:
- Training: `src/sft.py`
- Inference pipeline: `scripts/run_infer.py`
- Detection evaluation: `scripts/evaluate_detection.py`

Trainer variants (quick map):
- Default ms-swift trainer (via `TrainerFactory`).
- `rollout_matching_sft`.
- `stage2_ab_training`.
- `gkd_monitor` (with `rlhf_type: gkd`).

Memory topic map (owner memory per domain):
- Config contract: `.serena/memories/config_yaml_guide.md`
- Data contract + dataset flow: `.serena/memories/data_contract_and_datasets.md`
- Coord-token/offset specifics: `.serena/memories/coord_tokens_and_coord_offset.md`
- Packing + stage2 runtime behavior: `.serena/memories/packing_and_stage2_rollout_matching.md`
- Inference/eval tooling: `.serena/memories/evaluation_inference_and_tools.md`
- Public-data runner/output contract: `.serena/memories/public_data_module.md`
- Coding/process guardrails: `.serena/memories/style_and_conventions.md`
- Command cribsheet: `.serena/memories/suggested_commands.md`
- Completion checklist: `.serena/memories/task_completion.md`
