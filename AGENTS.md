# Codex Agent

## Mission
- Evolve CoordExp into a general grounding/detection research stack; keep runs reproducible and paper-ready.
- Direction: `progress/full_idea.md`. Prefer YAML-first experiments in `configs/` over ad-hoc scripts.
- Defaults: single-dataset training; packing is the primary efficiency lever; fusion-config training is legacy/experimental.

## Working Style (Research-Grade)
- Explain decisions only when they affect correctness/reproducibility/eval validity/maintainability; keep rationales short.
- State assumptions when underspecified; choose the simplest viable approach; do not invent metrics/results.
- Code & architecture style guidance: `docs/standards/CODE_STYLE.md` (Transformers-inspired “Option A”)

## Guardrails
- Config-first; avoid new CLI flags; keep Qwen3-VL chat-template compatibility.
- Preserve geometry (never drop/reorder coords); use `src/datasets/geometry.py`; training uses `do_resize=false`.
- Do not edit upstream HF model files like `modeling_qwen3_vl.py` (off-limits).
- For architectural/contract changes, follow OpenSpec governance.

## Repo Safety (Prevent Data Loss)
- Never run destructive cleanup commands unless the user explicitly asked (examples: `git restore`, `git clean -fd`, `git reset --hard`, `git checkout -- <path>`, `rm -rf`). If suggesting such a command, warn that it can delete uncommitted/untracked work.
- Before making edits, check `git status --porcelain` and call out any dirty files; if there are unexpected changes (especially in files we didn’t touch), stop and ask how to proceed.
- Prefer small, incremental commits (or a `git stash`) during large refactors so work can’t be accidentally discarded.

## Environment
- Repo root: `.`.
- Use `conda run -n ms python ...` for commands (e.g., tests).
- Target models: `/data/home/xiaoyan/AIteam/data/Qwen3-VL`.
- Use `temp/` for one-off debug artifacts; clean up when done.

## Navigation (Progressive)
- Source of truth: `docs/` (do not duplicate docs into global instructions).
- Use the `coordexp-codebase` skill for doc index, entrypoints, and config workflow pointers.
- For any file matching `*.py`, **Serena MCP is mandatory** for exploration and editing (symbol-aware navigation and edits). Serena MCP is the authoritative and precise method for all Python code operations.
- Do **not** use Serena MCP for non-Python files (e.g., `*.md`, `*.sh`, `*.json`, `*.txt`). Use standard tools such as `rg`, `cat`, or appropriate editors for those.
- Use Serena MCP’s `activate_project` when exploring Python code in external libraries or repositories outside the current working directory.

## Codex Sub-Agents (Async Reviews)
- Prioritize quality first, then speed.
- Spawn sub-agents whenever parallel work can improve correctness or throughput.
- No fixed cap: use as many sub-agents as needed.
- Give each sub-agent a narrow, independent scope.
- Main agent must verify sub-agent findings against repo sources before applying.
- Main agent owns final decisions, edits, and output quality.

## Scope
- In: coord vocab/expectation decoding, set matching losses, rollout-based consistency, grounding evaluation.
- Out (unless approved): large architecture forks, bespoke RL loops, custom vision backbones.
