# YAML Config Guide (loader + schema)

Entrypoint:
- `src/sft.py` takes only `--config` and optional `--base_config`; all hyperparams are in YAML.

Inheritance / merge:
- Supported keys: top-level `extends:` or `inherit:` (str or list[str]).
- Paths are resolved relative to the current YAML; bases merge in order (earlier = lower precedence); cycles error.
- Implementation: `ConfigLoader.load_yaml_with_extends()` in `src/config/loader.py`.

Required / validated sections:
- `template:` must exist (used by ms-swift to build the chat template).
- `custom:` must exist and validates dataset + CoordExp-specific knobs (see `CustomConfig` in `src/config/schema.py`).
- For ms-swift TrainArguments validation, configs should include:
  - `data.dataset: ["dummy"]` (placeholder; actual dataset comes from `custom.train_jsonl` / fusion).

Prompt selection:
- YAML `prompts:` overrides are DISABLED (must be empty); edit `src/config/prompts.py`.
- `ConfigLoader.resolve_prompts()` selects default prompts based on:
  - `custom.use_summary` (summary vs dense),
  - `custom.object_ordering` (sorted vs random),
  - `custom.coord_tokens.enabled` (coord_tokens vs numeric).
- `CustomConfig.output_variant` is derived (not user-set) from these prompts.

CustomConfig gotchas (enforced by schema):
- `custom.emit_norm` MUST be `"none"` (runtime coord normalization disabled; data must already be norm1000 or tokens).
- `custom.json_format` MUST be `"standard"`.
- Must provide either `custom.train_jsonl` (or legacy `custom.jsonl`) OR `custom.fusion_config`.
- `custom.coord_tokens.enabled: true` REQUIRES `custom.coord_soft_ce_w1.enabled: true`.

Training section helpers:
- `training.effective_batch_size` auto-computes `training.gradient_accumulation_steps` using world size:
  `effective = per_device_train_batch_size * world_size * grad_accum`.
- Packing knobs under `training.*` (e.g. `packing`, `packing_buffer`, `eval_packing`) are popped before TrainArguments init
  and consumed later by `src/sft.py` (see `train_args._packing_overrides`).

Other common knobs:
- `global_max_length` sets both `model.max_model_len` and `template.max_length`.
- Save delay: set `training.save_delay_steps` or `training.save_delay_epochs`; loader attaches a `SaveDelayConfig` and `sft.py` adds `SaveDelayCallback`.
- Debug-only overrides: `debug.enabled`, `debug.output_dir`, `debug.{train,val}_sample_limit` (see `DebugConfig` in `src/config/schema.py`).
