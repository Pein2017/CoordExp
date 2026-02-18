# Coord Tokens, Losses, and Coord-Offset Adapter

Coord tokens:
- Text form: `<|coord_k|>` where `k in [0, 999]`.
- Codec helpers: `src/coord_tokens/codec.py` (`is_coord_token`, `token_to_int`, `ints_to_tokens`, masks for coord-id ranges).

Coord-token mode (data + template):
- Record annotation: `src/coord_tokens/validator.py::annotate_coord_tokens(record)` adds cached token/int forms under `*_coord_*` keys.
- Builder behavior: `src/datasets/builders/jsonlines.py` emits either numeric `[0,999]` or token strings depending on `coord_tokens_enabled`.
- Template adapter: `src/coord_tokens/template_adapter.py::apply_coord_template_adapter()` patches `template.normalize_bbox` to a no-op when
  `custom.coord_tokens.enabled: true` and `custom.coord_tokens.skip_bbox_norm: true` (prevents double scaling).

Distributional coord supervision (stage-1 + parts of stage-2):
- Core math: `src/coord_tokens/soft_ce_w1.py` (Gaussian soft targets, soft CE, W1 on CDF).
- Trainer integration: `CoordSoftCEW1LossMixin` in `src/metrics/dataset_metrics.py`.
- Config: `custom.coord_soft_ce_w1.*` in `src/config/schema.py`.
- Schema guard: `custom.coord_tokens.enabled` REQUIRES `custom.coord_soft_ce_w1.enabled`.

Coord-offset adapter (optional):
- Purpose: train only coord-token rows without updating the full vocab.
- Implementation: `src/coord_tokens/offset_adapter.py`:
  - `install_coord_offset_adapter(model, coord_ids=...)` freezes base `embed_tokens.weight` + `lm_head.weight` and adds trainable offsets.
  - `reattach_coord_offset_hooks(model)` is required after PEFT/Swift wrapping so the active adapter instance hooks the wrapped modules.
- Runner wiring: `src/sft.py`:
  - installs adapter before `sft.prepare_model()`;
  - adds `coord_offset_adapter` to `modules_to_save`;
  - sanity-checks `coord_offset.ids` vs `model.config.vocab_size`.
- Optimizer: set `training.optimizer: multimodal_coord_offset`; registered in `src/optim/coord_offset_optimizer.py`.

Export/merge note:
- Some workflows require merging LoRA + coord offsets; see `scripts/merge_coord.sh` and `README.md`.
