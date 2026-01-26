# Codex Agent

<!-- OPENSPEC:START -->
# OpenSpec Instructions

These instructions are for AI assistants working in this project.

Always open `@/openspec/AGENTS.md` when the request:
- Mentions planning or proposals (words like proposal, spec, change, plan)
- Introduces new capabilities, breaking changes, architecture shifts, or big performance/security work
- Sounds ambiguous and you need the authoritative spec before coding

Use `@/openspec/AGENTS.md` to learn:
- How to create and apply change proposals
- Spec format and conventions
- Project structure and guidelines

Keep this managed block so 'openspec update' can refresh the instructions.

<!-- OPENSPEC:END -->

## Mission
- Evolve CoordExp into a general grounding/detection research stack; keep runs reproducible and paper-ready.
- Direction: `progress/full_idea.md`. Prefer YAML-first experiments in `configs/` over ad-hoc scripts.
- Defaults: single-dataset training; packing is the primary efficiency lever; fusion-config training is legacy/experimental.

## Working Style (Research-Grade)
- Explain decisions only when they affect correctness/reproducibility/eval validity/maintainability; keep rationales short.
- State assumptions when underspecified; choose the simplest viable approach; do not invent metrics/results.

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
- Repo root: `/data/home/xiaoyan/AIteam/data/CoordExp`.
- Use `conda run -n ms python ...` for commands (e.g., tests).
- Target models: `/data/home/xiaoyan/AIteam/data/Qwen3-VL`.
- Use `temp/` for one-off debug artifacts; clean up when done.

## Navigation (Progressive)
- Source of truth: `docs/` (do not duplicate docs into global instructions).
- Use the `coordexp-codebase` skill for doc index, entrypoints, and config workflow pointers.
- Use Serena MCP for symbol-aware edits; use `rg`/`cat` for greps and docs.

## Scope
- In: coord vocab/expectation decoding, set matching losses, rollout-based consistency, grounding evaluation.
- Out (unless approved): large architecture forks, bespoke RL loops, custom vision backbones.
