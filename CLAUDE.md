# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

CoordExp extends Qwen3-VL with coordinate-specialized tokens to achieve SOTA open-vocabulary detection/grounding. The core innovation:

- **Coordinate Tokens**: 1000-bin vocab (`<|coord_0|>` through `<|coord_999|>`) for normalized geometry
- **CoordExp Decoding**: Softmax on coord-subvocab + expectation yields continuous boxes with smooth gradients
- **Order-Invariant Matching**: Hungarian/OT matching for set-structured object supervision
- **Fully Differentiable**: L1 + GIoU losses applied directly to LM head outputs

Research goal: paper-ready results for ICLR/CVPR/ECCV on public detection benchmarks (LVIS, COCO, Objects365).

## Common Commands

```bash
# Environment: Always use ms conda env directly (no conda activate)
PYTHON=/root/miniconda3/envs/ms/bin/python
SWIFT=/root/miniconda3/envs/ms/bin/swift

# Run tests
$PYTHON -m pytest tests/ -v
$PYTHON -m pytest tests/coord_tokens/ -v  # coord token tests only

# Training
$PYTHON -m src.sft --config configs/dlora/sft_base.yaml --base_config configs/base.yaml
$PYTHON -m src.sft --config configs/debug.yaml --debug --verbose  # debug mode

# Data preprocessing pipeline (LVIS example)
bash public_data/scripts/lvis_full_pipeline.sh
MAX_BLOCKS=1024 FACTOR=32 bash public_data/scripts/lvis_full_pipeline.sh

# Expand vocabulary (one-time setup)
$PYTHON scripts/expand_coord_vocab.py \
  --src model_cache/Qwen3-VL-8B-Instruct \
  --dst model_cache/Qwen3-VL-8B-Instruct-coordexp

# Merge LoRA + coord offsets
ADAPTERS=output/coord/checkpoint-* OUTPUT_DIR=output/merged GPU_DEVICES=0 bash scripts/merge_coord.sh

# Inference
bash scripts/infer.sh --model output/merged --gt-jsonl data.jsonl --output pred.jsonl --mode coord

# Evaluation
bash scripts/eval.sh --pred pred.jsonl --output metrics.json

# Verify coord tokens
$PYTHON scripts/verify_coord_tokens.py --model model_cache/Qwen3-VL-8B-Instruct-coordexp

# Inspect chat template rendering
$PYTHON scripts/inspect_chat_template.py --jsonl path/to/data.jsonl --index 0
```

## Architecture

### Entry Point & Training Flow
- `src/sft.py`: Pure YAML-driven SFT entry; loads config → builds dataset → creates trainer → trains
- Config inheritance: `--base_config` + `--config` (override YAML extends base)
- Uses ms-swift (`SwiftSft`/`SwiftRLHF`) with modular hooks for CoordExp losses

### Core Modules (`src/`)
```
src/
├── sft.py                    # Main entry point
├── config/                   # YAML loading, TrainingConfig schema, prompts
├── datasets/
│   ├── dense_caption.py      # BaseCaptionDataset (loads JSONL, applies template)
│   ├── augmentation/         # Registry-based ops (hflip, rotate, crop, etc.)
│   ├── builders/             # JSONLinesBuilder (record → Qwen3-VL messages)
│   ├── preprocessors/        # Row-level transforms (normalization, augmentation)
│   └── wrappers/             # PackedCaptionDataset (bin-packing for efficiency)
├── coord_tokens/
│   ├── codec.py              # Token↔int conversion (token_to_int, int_to_token)
│   ├── template_adapter.py   # Patches Qwen3-VL template for coord tokens
│   ├── offset_adapter.py     # Trainable offsets for coord token embeddings
│   └── validator.py          # Coordinate range validation
├── optim/                    # Custom optimizers (multimodal_coord_offset)
├── infer/                    # Inference engine (coord/text modes)
├── eval/                     # COCO-style detection metrics (AP@50, AP@75, mAP)
├── callbacks/                # Training callbacks (augmentation curriculum)
├── trainers/                 # Custom trainer variants
└── metrics/                  # Token type accuracy metrics (coord vs text)
```

### Configuration System (`configs/`)
```yaml
# Inheritance: extends: ../base.yaml
# Key config blocks:
model:
  model: model_cache/Qwen3-VL-8B-Instruct-coordexp
  torch_dtype: bfloat16
  attn_impl: flash_attention_2

tuner:
  train_type: lora
  use_dora: true
  target_modules: [all-linear]
  lora_rank: 16

training:
  optimizer: multimodal              # or multimodal_coord_offset
  learning_rate: 4.0e-4
  vit_lr: 2.0e-4
  aligner_lr: 8.0e-4
  packing: true

custom:
  emit_norm: none                    # Pre-normalized data required
  train_jsonl: path/to/train.jsonl
  val_jsonl: path/to/val.jsonl
  coord_tokens:
    enabled: true
    skip_bbox_norm: true
  coord_offset:                      # Optional: trainable coord embeddings
    enabled: true
    embed_lr: 4.0e-4
    head_lr: 4.0e-4
```

### Data Contract (JSONL)
```json
{
  "images": ["relative/path.jpg"],
  "objects": [
    {"desc": "red cup", "bbox_2d": ["<|coord_100|>", "<|coord_200|>", "<|coord_300|>", "<|coord_400|>"]},
    {"desc": "table", "poly": ["<|coord_10|>", "<|coord_20|>", ...], "poly_points": 4}
  ],
  "width": 768,
  "height": 512,
  "summary": "red cup x1, table x1"
}
```
- Geometry: exactly one of `bbox_2d`, `poly`, or `line` per object
- Coords: either pixel floats OR `<|coord_k|>` tokens (k ∈ [0,999])
- See `docs/DATA_JSONL_CONTRACT.md` for full schema

### Coord Token IDs
- `<|coord_*|>` (wildcard): ID 151669
- `<|coord_0|>` through `<|coord_999|>`: IDs 151670–152669
- Offset adapter only affects 151670–152669 (skips wildcard)

## Environment

- Python: `/root/miniconda3/envs/ms/bin/python` (always use directly, no conda activate)
- Transformers: `/root/miniconda3/envs/ms/lib/python3.12/site-packages/transformers`
- ms-swift: `/data/home/xiaoyan/AIteam/data/ms-swift`
- Working directory: `/data/home/xiaoyan/AIteam/data/CoordExp`
- Serena MCP: Available for semantic code navigation; configured at `.serena/project.yml`. Use for symbol search/navigation; skip for document reads.

## Development Guidelines

### Configuration-First
- Prefer YAML edits in `configs/` over new CLI flags
- Define runtime args at top of entry scripts (`scripts/train.sh`, `scripts/infer.sh`)
- Use `extends:` for config inheritance

### Code Conventions
- Keep `sft.py` entry flow unchanged; extend via config hooks
- Avoid editing HF `modeling_qwen3_vl.py`; use adapters/wrappers instead
- Geometry handling: never drop/reorder coords silently; validate early
- Seed everything for reproducibility; log seeds in new entrypoints

### Testing
- Add tests for coord token logic, geometry transformations
- Run `pytest tests/ -v` before significant changes
- Use `--debug --verbose` flags during development

### Documentation
- Keep docs in sync when changing behavior/configs
- Follow OpenSpec proposal workflow for material changes (`openspec/AGENTS.md`)
- Research direction lives in `progress/idea.md`

## Key Files Reference

| File | Purpose |
|------|---------|
| `src/sft.py` | Training entry point |
| `src/config/loader.py` | YAML config loading with inheritance |
| `src/config/schema.py` | TrainingConfig dataclass schema |
| `src/datasets/dense_caption.py` | Main dataset class |
| `src/coord_tokens/codec.py` | Token↔int conversion |
| `src/coord_tokens/offset_adapter.py` | Trainable coord offsets |
| `configs/base.yaml` | Shared defaults |
| `configs/dlora/sft_base.yaml` | Standard coord training config |
| `docs/DATA_JSONL_CONTRACT.md` | Data schema specification |
| `progress/idea.md` | Research roadmap |

## Scope

**In-scope**: coord vocab design, expectation decoding, set matching losses, detection evaluation, dataset preprocessing

**Out-of-scope** (unless approved): large architecture forks, bespoke RL loops, custom vision backbones
