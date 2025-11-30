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

## Project Overview
CoordExp is a research-first fork that extends Qwen3-VL for open-vocabulary detection and grounding using coordinate-specialized tokens, expectation-based continuous box decoding, and order-invariant (Hungarian/OT) matching. Goal: improve grounding accuracy/precision and training efficiency via fully differentiable geometry loss, avoiding heavy RL while keeping the native SFT pipeline. We will benchmark across public and common detection datasets, aiming for SOTA by fine-tuning Qwen3-VL with CoordExp.

## Relationship to Qwen3-VL
- Reuse the existing Qwen3-VL training stack (data preprocess, chat templates, ms-swift trainer wiring).
- Add CoordExp modules (coord vocab, expectation decoding, set matching losses) without forking core HF modeling; prefer lightweight wrappers/adapters.
- Maintain compatibility with configs/layout used in `/data/home/xiaoyan/AIteam/data/Qwen3-VL`.

## Environment
- Use `ms` conda env (`/root/miniconda3/envs/ms`) for all Python.
- `transformers` path: `/root/miniconda3/envs/ms/lib/python3.12/site-packages/transformers`.
- `ms-swift` available at `/data/ms-swift`.
- Run commands from repo root `/data/home/xiaoyan/AIteam/data/CoordExp` unless stated.

## Design Principles
- Configuration-first: prefer YAML in `configs/` to new CLI flags.
- Explicit over implicit: validate early, no silent defaults; clear errors with remediation.
- Type-safe, frozen configs where possible; small public interfaces and clean imports.
- Geometry-aware data handling; visualize/validate when touching boxes or quantization.
- Reuse over custom: prefer ms-swift/transformers primitives before new code.
- Fail fast & test small: add minimal probes/visual checks for new coord logic.

## Development Approach
- Keep `sft.py` entry and trainer flow unchanged; plug CoordExp loss + matching via config and modular hooks.
- Avoid editing HF `modeling_qwen3_vl.py`; extend via adapters/monkeypatch only if necessary.
- Preserve chat template and serialization compat with base detection tasks.
- When adding features or non-trivial changes, consult `@/openspec/AGENTS.md` for proposal workflow.

## Scope & Non-Goals
- In-scope: coord vocab design, expectation decoding, set matching losses, rollout-based consistency, evaluation for grounding.
- Out-of-scope (unless explicitly approved): large architecture forks, bespoke RL loops, custom vision backbones.

## Quick Pointers
- Source lives in `src/`; configs in `configs/`.
- Patent draft/background: `patent/draft.md`.
- Prefer incremental, well-commented changes; keep scripts thin and delegate logic to modules.
