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

## Purpose & Mission
- Act as an embedded collaborator evolving CoordExp from its telecom QC heritage into a general detection/grounding research platform.
- Aim for SOTA results and a paper-ready story for ICLR/CVPR/ECCV; keep everything reproducible and auditable.

## Background & Legacy
- Repo is partially copied from the prior telecom quality-control project: Qwen3-VL SFT + LoRA with data/vision preprocessing, chat template orchestration, and inference already proven.
- Infrastructure (training, preprocessing, inference) is battle-tested—reuse first, then extend.

## Project Overview (CoordExp)
- Research-first fork that extends Qwen3-VL with coordinate-specialized tokens, expectation-based continuous box decoding, and order-invariant (Hungarian/OT) matching.
- Goal: improve grounding accuracy/precision and training efficiency via fully differentiable geometry loss while keeping the native SFT pipeline.
- Benchmark across public/common detection datasets to reach SOTA open-vocabulary detection and grounding.

## Current Priorities
- Generalize the legacy pipeline beyond telecom to broad detection/grounding benchmarks; validate geometry-aware decoding and matching.
- Iterate via YAML configs first; minimize code churn unless the design demands it.
- Keep compatibility with layouts/configs used in `/data/home/xiaoyan/AIteam/data/Qwen3-VL`.
- Anchor direction and narrative in `progress/idea.md`; keep artifacts paper-ready.

## Standard Workflow Outline (adapted from Qwen3-VL)
1) Data intake/normalization: prepare detection/grounding datasets with consistent geometry and chat templates; validate boxes before training.
2) Experiment config/fusion: edit YAML in `configs/` (seeds, LoRA/full FT, coord vocab choices, dataset mixes) and keep configs under version control.
3) Train/finetune: run `src/sft.py` via the `ms` env; favor config-driven hooks for CoordExp losses/matching.
4) Evaluate & ground: run evaluation/inference passes on held-out sets; capture qualitative grounding visuals when touching geometry or vocab.
5) Documentation/governance: when behavior or contracts change, update `docs/` and follow OpenSpec proposal flow for material changes.

## Codebase Layout
- `src/` — training/inference code (datasets, config, trainers, callbacks, utils, `sft.py`).
- `configs/` — YAML experiment configs; default surface for new knobs.
- `scripts/` — small utilities (coord vocab expansion/verification); keep thin wrappers only.
- `docs/` — authoritative guidance; sync when workflows/configs change.
- `openspec/` — change management specs; start here for proposals.
- `progress/` — evolving ideas and direction (`progress/idea.md`).
- `patent/` — patent draft/background.

## Relationship to Qwen3-VL
- Reuse the Qwen3-VL training stack (data preprocess, chat templates, ms-swift trainer wiring).
- Add CoordExp modules (coord vocab, expectation decoding, set matching losses) without forking core HF modeling; prefer lightweight wrappers/adapters.
- Maintain compatibility with configs/layout used in `/data/home/xiaoyan/AIteam/data/Qwen3-VL`.

## Environment
- Always invoke `/root/miniconda3/envs/ms/bin/python` (and `/root/miniconda3/envs/ms/bin/torchrun` or `/root/miniconda3/envs/ms/bin/swift`) directly; do not use `conda activate` or `conda run`.
- `transformers` path: `/root/miniconda3/envs/ms/lib/python3.12/site-packages/transformers`.
- `ms-swift` available at `/data/home/xiaoyan/AIteam/data/ms-swift`.
- Run commands from repo root `/data/home/xiaoyan/AIteam/data/CoordExp` unless stated.
- **Serena MCP**: Available via MCP server; project configured at `.serena/project.yml`. Activate with "activate the project Qwen3-VL" or by path. Project-specific memories stored in `.serena/memories/`. **Do not use Serena MCP for pure document retrieval or reading** — it doesn't benefit document/text reading tasks; use standard file reading tools instead.
- **When to prefer Serena MCP**: Use MCP when you need semantic, symbol-level operations—finding symbols, references, or performing structured symbol edits—especially across large files. Skip MCP for straightforward file reads or document retrieval where `read_file`/`rg` is faster.
- **MCP workflow for breadth then depth**: Start with symbol overviews (`find_symbol` / `get_symbols_overview`) to map large files, then pull specific bodies or references only where needed; pair with `rg` for quick presence/usage checks; fall back to `read_file` for prose/docs or when you truly need the whole file. Favor symbol edits for code changes; use plain reads for simple text.


## Development Approach
- **Code exploration**: Prefer Serena MCP tools (semantic search, symbol navigation, find_referencing_symbols) for understanding code structure, relationships, and making targeted edits. Use standard file reading only when you need full file contents.
- Keep `sft.py` entry and trainer flow unchanged; plug CoordExp loss + matching via config and modular hooks.
- Avoid editing HF `modeling_qwen3_vl.py`; extend via adapters/monkeypatch only if necessary.
- Preserve chat template and serialization compatibility with base detection tasks.
- Follow OpenSpec proposal workflow (`@/openspec/AGENTS.md`) for features or other non-trivial changes.

## Design Principles
- Configuration-first: prefer YAML in `configs/` to new CLI flags.
- Prefer defining runtime arguments at the top of entry scripts (e.g., `scripts/infer.sh`, `scripts/train.sh`) instead of adding new CLI flags; avoid new CLI arguments when possible.
- Explicit over implicit: validate early, no silent defaults; clear errors with remediation.
- Type-safe, frozen configs where possible; small public interfaces and clean imports.
- Geometry-aware data handling; visualize/validate when touching boxes or quantization.
- Reuse over custom: prefer ms-swift/transformers primitives before new code.
- Fail fast & test small: add minimal probes/visual checks for new coord logic.

## Common Rules & Preferences (borrowed from Qwen3-VL)
- Keep docs in lockstep with code: touch mapped docs when you change behavior/configs.
- Deterministic runs: seed everything and log seeds in new entrypoints.
- Geometry & grounding: never drop or reorder geometry silently; reuse helpers in `src/datasets/` for canonicalization/quantization.
- Logging: use project logger utilities; include remediation hints in errors.
- Third-party deps: prefer existing ms-swift/transformers; justify new deps and record behavioral impact.
- Validate before merge: run existing tests or targeted probes when altering datasets/geometry or decoding logic.

## Scope & Non-Goals
- In-scope: coord vocab design, expectation decoding, set matching losses, rollout-based consistency, evaluation for grounding.
- Out-of-scope (unless explicitly approved): large architecture forks, bespoke RL loops, custom vision backbones.

## Important
- Interrupt for clarification whenever requirements are ambiguous or assumptions feel shaky.
- Keep paper-readiness in mind: preserve experiment metadata, configs, and qualitative examples.
