# Coding Guardrails

Role: implementation-time invariants only.

Canonical pointers:
- `docs/PROJECT_CONTEXT.md`
- `docs/standards/README.md`
- `docs/standards/CODE_STYLE.md`
- `docs/standards/REPO_HYGIENE.md`
- `openspec/specs/`

Guardrails:
- Stay config-first; prefer YAML or schema changes over adding new CLI flags.
- Preserve geometry semantics via `src/datasets/geometry.py`; never drop, reorder, or silently reinterpret coordinates.
- Keep training on the offline-resized path (`do_resize=false`) unless docs or specs explicitly change.
- Preserve Qwen3-VL chat-template compatibility.
- Do not edit upstream HF model files such as `modeling_qwen3_vl.py`; extend through wrappers or adapters.
- Route contract or architecture changes through OpenSpec.
- Keep edits scoped and do not revert unrelated dirty changes.
- Prefer the current small runtime seams over re-growing monolithic entrypoints.