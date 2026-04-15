# Codex Agent

## Mission
- Evolve CoordExp into a general grounding/detection research stack; favor reproducible, paper-ready workflows and compatibility-preserving changes.
- Follow precedence: `openspec/specs/` -> `docs/PROJECT_CONTEXT.md` -> `docs/SYSTEM_OVERVIEW.md` -> `docs/IMPLEMENTATION_MAP.md` -> relevant domain docs under `docs/` -> `openspec/changes/<active-change>/` -> `progress/`.
- Use `docs/AGENT_INDEX.md` and `docs/catalog.yaml` for routing; use `progress/` only for historical context, diagnostics, or empirical evidence.

## Defaults
- Offline-prepared single-dataset JSONL is the default training surface; keep runtime transforms minimal and reproducible.
- Treat both Stage-1 baseline SFT and Stage-2 rollout-aware training as active first-class surfaces; fusion-config training remains legacy/experimental.
- Packing, cache reuse, and manifest/artifact completeness are primary operational levers.

## Guardrails
- Config-first; avoid new CLI flags; keep Qwen3-VL chat-template compatibility and current artifact contracts.
- Preserve geometry and image alignment end-to-end (never drop/reorder coords); use `src/datasets/geometry.py`; training uses `do_resize=false`.
- Do not edit upstream HF model files like `modeling_qwen3_vl.py`.
- For stable training/eval behavior, config contracts, loss semantics, or artifact-name changes, follow OpenSpec governance and update docs/specs in the same change.

## Workflow
- Explain decisions only when they affect correctness/reproducibility/eval validity/maintainability.
- State assumptions when underspecified; choose the smallest viable change; do not invent metrics/results.
- Fail fast on unexpected behavior, resolve root causes, and verify on the narrowest realistic surface first (targeted tests, caches, or artifacts before broad suites).

## Repo Safety
- Never run destructive cleanup commands unless explicitly asked.
- Dirty changes from parallel work are expected; isolate your edits and do not revert unrelated work.
- Prefer small, logically scoped commits during large refactors or incident response.

## Navigation
- Use `coordexp-codebase` for entrypoints and workflow pointers.
- Use `coordexp-research-context` for broad background, read-order, and historical Stage-2 context.
- Prefer `rtk` for noisy shell workflows first: broad repo scans, multi-hit `rg`, `git`, logs, tests, long docs/prose reads, and other multi-line command output where compact summaries help.
- Do not force `rtk` into exact-output workflows. Prefer raw commands for narrow line reads (for example `sed -n`, `nl -ba ... | sed -n`), machine-readable stdout, delicate quoting, or commands where `rtk` could obscure exact interpreter/environment binding.
- When a command already depends on a project-specific environment wrapper, keep that wrapper under `rtk` instead of dropping it. In this repo, tests should prefer `rtk conda run -n ms python -m pytest ...` over bare `rtk pytest ...`.
- Serena MCP is available beyond Python whenever the target file type is supported and symbol-aware navigation or editing would reduce ambiguity or risk.
- For any `*.py` file, Serena MCP is mandatory for exploration and editing.
- For non-Python code, prefer Serena MCP when working in large files, doing cross-reference tracing, symbol-level edits, or nontrivial refactors. For plain-text or non-symbolic files such as Markdown, YAML, JSON, or exact line-based inspections, direct shell reads and `apply_patch` are usually the better fit.
- For code work, use `rg`/`rtk` first to narrow candidate files or directories, then switch to Serena MCP when symbol-level understanding or editing is useful.

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
