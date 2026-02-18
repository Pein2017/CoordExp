# Task Completion Checklist (CoordExp)

- Config sanity:
  - `custom.train_jsonl` (or `custom.fusion_config`) set; `custom.json_format: standard`; `custom.emit_norm: none` (enforced).
  - `data.dataset: ["dummy"]` present (ms-swift TrainArguments validation).
  - If `custom.coord_tokens.enabled: true`, ensure `custom.coord_soft_ce_w1.enabled: true` and usually `custom.coord_tokens.skip_bbox_norm: true`.
  - If KD enabled (`rlhf_type: gkd` with `rlhf.llm_kd_weight>0` or `custom.visual_kd.enabled`), set `rlhf.teacher_model` and verify vocab sizes match.
- Run quick smoke checks:
  - `scripts/tools/inspect_chat_template.py` on a sample JSONL row.
  - `--debug` run (health check prints image token counts; optional `custom.dump_conversation_text`).
- Packing / performance:
  - If `training.packing: true` (non-stage2), confirm `per_device_train_batch_size=1` and `max_steps` is finite (sft auto-sets only when dataset length known).
  - For stage2 rollout matching, remember packing is post-rollout only (trainer-internal).
- Tests:
  - `conda run -n ms python -m pytest tests/` (or at least the relevant subset).
- Docs/OpenSpec:
  - If you changed data contract, trainer behavior, or evaluation outputs, update docs and follow `openspec/` governance.
