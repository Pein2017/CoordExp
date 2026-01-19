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

## Mission & Context
- Evolve CoordExp from telecom QC roots into a general grounding/detection research stack; keep runs reproducible and paper-ready (ICLR/CVPR/ECCV).
- Anchor direction in `progress/full_idea.md`; prefer reproducible configs over ad-hoc scripts.

## Current Priorities
- Broaden datasets beyond telecom; validate coord-token + expectation decoding and order-invariant matching.
- Iterate via YAML in `configs/` first; keep compatibility with `/data/home/xiaoyan/AIteam/data/Qwen3-VL`.
- Maintain single-dataset training as the default; fusion-config training is legacy/experimental; packing remains the default efficiency lever.

## Codebase Map
- `src/` core: `sft.py` entry, `config/loader.py` (YAML merge + prompt resolve), datasets/augment/collators, coord token adapters, callbacks, trainers, eval/infer.
- `configs/` experiments (dlora, eval, tests, packing defaults).
- `scripts/` thin utilities (inspect chat template, convert/verify coord tokens, run/eval/vis).
- `docs/` authoritative guides (data contract, preprocessing, packing guide, evaluator).
- `openspec/` change governance; `progress/` ideas/roadmap; `patent/` background.

## Run Environment
- Use `conda run -n ms python *` for commands (e.g., `conda run -n ms python -m pytest tests/`).
- Run commands from repo root `/data/home/xiaoyan/AIteam/data/CoordExp`.
- `ms-swift` at `/data/home/xiaoyan/AIteam/data/ms-swift`; HF transformers in the ms env.

## Tools & Navigation
- Serena MCP: best for code symbol discovery/edits; avoid for plain docs (use `rg`/`cat`).
- For detailed Serena MCP workflows and common patterns, use the `serena-mcp-navigation` skill.

## Config Workflow (YAML-first)
- CLI is minimal: `python -m src.sft --config <yaml> [--base_config <yaml>] [--debug|--verbose]`.
- Inheritance via `extends`/`inherit`; cycles fail fast. Prompt overrides in YAML are **disabled**—edit `src/config/prompts.py`.
- Required: `custom.user_prompt` and either (`custom.train_jsonl` / `custom.jsonl`) OR `custom.fusion_config`; `ROOT_IMAGE_DIR` auto-set from the JSONL/fusion-config dir if unset.
- `custom.use_summary` toggles summary mode; ordering controlled by `custom.object_ordering` (`sorted`/`random`).
- `custom.coord_tokens.enabled` applies the template adapter; if data are pre-tokenized, set `custom.coord_tokens.skip_bbox_norm: true`.
- KD guards: enabling GKD or visual KD requires `rlhf.teacher_model`; vocab sizes must match teacher/student.
- `effective_batch_size` auto-computes `gradient_accumulation_steps`; packing keys are stripped before TrainArguments and re-consumed in `sft.py`.

## Data Contract & Prep
- Follow `docs/DATA_JSONL_CONTRACT.md`: exactly one geometry per object (`bbox_2d|poly|line`), width/height required, paths relative to JSONL dir.
- Pixel coords are canonical; template normalizes to norm1000 at encode time. If using coord tokens in JSONL, keep `width/height` and enable `skip_bbox_norm` to avoid double scaling.
- Validate/preview with `scripts/inspect_chat_template.py` and dataset validators in `src/datasets/`.

## Training Loop Guardrails (sft.py)
- Always call `sft.prepare_model(...)` (already done in entrypoint) before trainer to ensure LoRA/PEFT wrapping; otherwise full weights train/save.
- Coord-offset adapter: enable via `custom.coord_offset`; ids must fit tokenizer vocab; module is added to `modules_to_save` and hooks are reattached after PEFT wrap.
- Fusion is supported (legacy/experimental): set `custom.fusion_config` to a fusion YAML/JSON; it overrides `custom.train_jsonl` / `custom.val_jsonl`. Prefer offline JSONL merge when possible.
- Packing: enabling forces `per_device_train_batch_size=1` and requires finite `max_steps`; `_parse_packing_config` uses template/global_max_length. Eval packing is opt-in (`training.eval_packing`).
- Save-delay: `training.save_delay_steps|epochs` or structured `save_delay_config`; checkpoints blocked until the delay passes.
- Augmentation: YAML-built pipeline (ops registered via `datasets/augmentation/ops`); curriculum requires the augmenter and a computable `total_steps`.
- Health checks (`--debug`) dump conversation text and image token counts; `custom.dump_conversation_text` writes to output_dir.

## Packing Defaults (see `docs/PACKING_MODE_GUIDE.md`)
- Default target: `global_max_length: 16000`, `per_device_train_batch_size: 1`, `effective_batch_size: 12`, `packing_buffer: 256`, `packing_min_fill_ratio: 0.7`, `eval_packing: true`.
- Packing replaces padding; adjust `eval_steps/save_steps` (~80) and `save_delay_steps` (~200) for 4-epoch runs on 4×A100.

## Design Principles & Do/Don't
- Config-first; avoid new CLI flags. Prefer YAML knobs and small adapters/wrappers over editing HF model files (`modeling_qwen3_vl.py` is off-limits).
- Preserve geometry: never drop/reorder coords; use helpers in `src/datasets/geometry.py`; training uses `do_resize=false`.
- Keep chat-template compatibility with Qwen3-VL; avoid custom token hacks outside coord tokens/offset adapters.
- Deterministic runs (seed everything), clear validation/errors, minimal comments; ASCII only unless file already uses Unicode.
- Update docs alongside behavior changes; follow OpenSpec for any capability/contract shifts.

## Evaluation & Inference
- Use `scripts/eval.sh` / `scripts/evaluate_detection.py` with YAML-driven configs; `configs/eval/detection.yaml` is the template.
- Inference helpers: `scripts/run_infer.py`, `scripts/vis.sh`, `scripts/run_vis_coord.sh`; keep adapters loaded via `--adapters` or merged weights as appropriate.

## Scope
- In: coord vocab/expectation decoding, set matching losses, rollout-based consistency, grounding evaluation.
- Out (unless approved): large architecture forks, bespoke RL loops, custom vision backbones.

## Working Style
- Ask for clarification when assumptions feel shaky; keep experiment metadata/artifacts paper-ready (configs, logs, qualitative vis).
- Use `temp/` folder for temporary test scripts, debug code, or experimental utilities; create sub-folders and organize files flexibly as needed for the task; always cleanup temporary files/scripts once the task or test is completed to avoid clutter and one-shot console debugging sessions.
