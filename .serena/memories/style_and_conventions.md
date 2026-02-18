# Style and Conventions (CoordExp)

- YAML-first workflow: prefer changing/adding knobs in `configs/` + schema in `src/config/schema.py` over new CLI flags.
- Prompt overrides: YAML `prompts:` section must be empty; edit `src/config/prompts.py`. `custom.user_prompt` can override dense user prompt (summary prompt is fixed in code).
- Data invariants: never drop/reorder geometry points; one geometry per object (`bbox_2d` OR `poly`); keep `width/height`.
- Coord normalization: training currently enforces pre-normalized norm1000 (`[0, 999]`) numeric coords or `<|coord_k|>` tokens; runtime normalization is disabled (`custom.emit_norm: none`).
- Compatibility: do not edit upstream HF model internals (e.g., Qwen3-VL modeling files); implement via adapters/wrappers (coord template adapter, coord offset adapter).
- Reproducibility: seed everything; prefer deterministic shuffles (dataset uses epoch-seeded permutations).
- OpenSpec governance: any capability/contract shift or breaking change should go through `openspec/` (see `openspec/AGENTS.md`).
- Code hygiene: minimal comments, ASCII, clear errors; avoid reverting unrelated dirty-tree changes.
