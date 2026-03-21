# Codex Agent

## Mission
- Evolve CoordExp into a general grounding/detection research stack; keep runs reproducible and paper-ready.
- Follow precedence: `docs/PROJECT_CONTEXT.md` -> `docs/SYSTEM_OVERVIEW.md` -> `docs/IMPLEMENTATION_MAP.md`.
- Use `progress/README.md` only for historical context.

## Defaults
- Single-dataset training.
- Packing is the primary efficiency lever.
- Fusion-config training is legacy/experimental.

## Guardrails
- Config-first; avoid new CLI flags; keep Qwen3-VL chat-template compatibility.
- Preserve geometry (never drop/reorder coords); use `src/datasets/geometry.py`; training uses `do_resize=false`.
- Do not edit upstream HF model files like `modeling_qwen3_vl.py`.
- For architectural/contract changes, follow OpenSpec governance.

## Workflow
- Explain decisions only when they affect correctness/reproducibility/eval validity/maintainability.
- State assumptions when underspecified; choose the simplest viable approach; do not invent metrics/results.
- Fail fast on unexpected behavior and resolve root causes.

## Repo Safety
- Never run destructive cleanup commands unless explicitly asked.
- Dirty changes from parallel work are expected; focus only on your edits.
- Prefer small, incremental commits during large refactors.

## Navigation
- Use `coordexp-codebase` for entrypoints and workflow pointers.
- Use `coordexp-research-context` for broad background, read-order, and historical Stage-2 context.
- For any `*.py` file, Serena MCP is mandatory for exploration and editing.

## Model
- Subagents must use `gpt-5.4` (not mini) by default.

## Environment
- Repo root: `.`
- Use `conda run -n ms python ...` for tests.
- Use `temp/` for one-off debug artifacts; clean up when done.
