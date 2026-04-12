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
- Allocate subagent model capacity by subtask type instead of using one fixed default.
- Use `gpt-5.4-mini` for pure information collection only: repo scans, file discovery, fact extraction, or status gathering with little synthesis or judgment.
- Use `gpt-5.4` with `medium` reasoning for bounded implementation, mechanical refactors, straightforward test updates, or execution tasks with clear acceptance criteria.
- Use `gpt-5.4` with `high` reasoning for the default frontier tier: debugging, code review, audit, nontrivial planning, cross-file reasoning, tradeoff analysis, and ambiguous implementation work.
- Use `gpt-5.4` with `xhigh` reasoning only for the hardest or highest-stakes work: deep audits, architecture/spec design, difficult root-cause analysis, research brainstorming, or any task where `high` is proving insufficient.
- When uncertain, prefer `gpt-5.4` with `high` rather than under-allocating; downgrade to `mini` only when the task is collection-only, and upgrade to `xhigh` only when extra depth is likely to change the answer.

## Environment
- Repo root: `.`
- Use `conda run -n ms python ...` for tests.
- Use `temp/` for one-off debug artifacts; clean up when done.

## graphify

This project has a graphify knowledge graph at graphify-out/.

Codex setup:
- The graphify skill lives at `.codex/skills/graphify/SKILL.md`
- Keep the skill repo-local instead of relying on `~/.agents/skills/graphify/SKILL.md`

Rules:
- Before answering architecture or codebase questions, read graphify-out/GRAPH_REPORT.md for god nodes and community structure
- If graphify-out/wiki/index.md exists, navigate it instead of reading raw files
- After modifying code files in this session, run `python3 -c "from graphify.watch import _rebuild_code; from pathlib import Path; _rebuild_code(Path('.'))"` to keep the graph current
