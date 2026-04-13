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
- Prefer `rtk` for noisy shell workflows first: docs/prose reads, `git`, search, logs, test output, and other multi-line command output where compact summaries help.
- For any `*.py` file, Serena MCP is mandatory for exploration and editing.
- For Python work, use `rg`/`rtk` to narrow files or directories first, then switch to Serena MCP for symbol-level analysis, reference tracing, and precise edits.
- Do not force `rtk` into exact-output workflows; fall back to raw commands when machine-readable stdout, delicate quoting, or verbatim output matters.

## Self-Improving
- Activate the `self-improving` skill when the user explicitly names it, asks to remember a reusable preference/correction/workflow, asks what has been learned, or wants repeated mistakes captured for future sessions.
- Also activate it proactively when a session shows repeated mistakes, repeated rework, multi-turn debugging/relaunch loops, environment footguns, or a clearly reusable successful workflow that is likely to recur in this workspace.
- It is acceptable to trigger the skill just to inspect existing repo-local memory, decide whether a reusable lesson exists, or briefly propose capturing that lesson for future sessions; do not wait for the user to say the exact skill name once the pattern is clear.
- Keep memory writes conservative: prefer proposing or recording the smallest reusable correction/workflow lesson rather than writing broad behavioral rules from a single one-off request.
- Keep mutable self-improving memory under `.self-improving/`; do not write workspace memory into `.codex/skills/self-improving/` or a machine-global home directory.
- Treat `.self-improving/` as exported repo-local project state rather than private scratch; the files under it are intended to be shared/visible in this workspace when relevant.

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
