# YAML Config Guide (Memory)

Role separation:
- Memory role: fast config footguns/defaults for implementation turns.
- Canonical docs: `docs/data/README.md`, `docs/training/STAGE2_RUNBOOK.md`.
- Canonical code contracts: `src/config/loader.py`, `src/config/schema.py`.
- Update trigger: when schema keys/validation behavior or loader merge logic changes.

High-signal defaults/invariants:
- Training is YAML-first; runtime CLI flags are limited to config path/base config/debug/verbose.
- Inheritance uses `extends` or `inherit`; cycles fail fast.
- `template` and `custom` sections are required.
- `data.dataset: ["dummy"]` remains the ms-swift placeholder.
- YAML `prompts:` overrides are disabled by loader validation.

`custom` contract reminders:
- Must provide dataset source via `custom.train_jsonl` or `custom.fusion_config`.
- `custom.user_prompt` is required.
- `custom.emit_norm` must be `none`.
- `custom.json_format` must be `standard`.
- `custom.coord_tokens.*` and `custom.coord_soft_ce_w1.*` are independent toggles.

Training helper reminders:
- `training.effective_batch_size` drives gradient accumulation derivation.
- `stage2_ab_training` requires `training.effective_batch_size` with divisibility checks.
- `global_max_length` propagates to model/template max-length fields.
- `training.packing_length` is unsupported; use `global_max_length`/`template.max_length`.
