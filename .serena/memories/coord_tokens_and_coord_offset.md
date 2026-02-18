# Coord Tokens + Coord Offset (Memory)

Role separation:
- Memory role: quick recall for coord-token mechanics and adapter wiring.
- Canonical docs: `docs/training/COORD_OBJECTIVE_AND_ADAPTER.md` (plus stage docs when applicable).
- Canonical code paths: `src/coord_tokens/`, `src/trainers/metrics/mixins.py`.
- Update trigger: when coord tokenization, coord losses, or offset adapter wiring changes.

Core reminders:
- Coord token form: `<|coord_k|>`, `k in [0, 999]`.
- Annotation helper: `annotate_coord_tokens(...)` caches `_coord_tokens`, `_coord_token_ints`, `_coord_token_norm`, and sets `_coord_tokens_enabled`.
- Dataset builder emits numeric or tokenized coordinates based on coord-token mode.

Loss/optimization reminders:
- Coord distributional objective utilities live in `src/coord_tokens/soft_ce_w1.py`.
- Trainer mixin lives in `src/trainers/metrics/mixins.py` (re-exported elsewhere).
- Coord-token enablement does not force coord-soft-ce enablement at schema level.

Coord-offset adapter reminders:
- Adapter install/reattach lives in `src/coord_tokens/offset_adapter.py`.
- Runner wiring + module-save integration is in `src/sft.py`.
- Optimizer path: `training.optimizer: multimodal_coord_offset`.
