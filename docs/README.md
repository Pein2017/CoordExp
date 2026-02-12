# Documentation Index

Welcome to the CoordExp documentation. This index will help you navigate the documentation structure.

## Start Here (5-10 minutes)

1) **Understand the data contract**: [`docs/data/JSONL_CONTRACT.md`](data/JSONL_CONTRACT.md)
2) **Prepare data (public or private)**: [`docs/data/INTAKE_PIPELINE.md`](data/INTAKE_PIPELINE.md)
3) **Train**:
   - Stage-1 / baseline SFT: start from `configs/base.yaml` and follow [`docs/data/README.md`](data/README.md)
   - Stage-2 / rollout-matching + AB: [`docs/training/STAGE2_RUNBOOK.md`](training/STAGE2_RUNBOOK.md)
4) **Infer + evaluate**: [`docs/eval/README.md`](eval/README.md)
5) **Interpret logs**: [`docs/training/METRICS_LOSSES.md`](training/METRICS_LOSSES.md)

All runnable commands in this repo assume repo root (`/data/CoordExp`) and prefer:

```bash
PYTHONPATH=. conda run -n ms python ...
```

## Quick Commands (Copy/Paste)

```bash
# Train (YAML-first)
PYTHONPATH=. conda run -n ms python -m src.sft --config <yaml> [--base_config <yaml>] [--debug]

# Inspect how one JSONL record renders under the current Qwen3-VL chat template
PYTHONPATH=. conda run -n ms python scripts/tools/inspect_chat_template.py --jsonl <path.jsonl> --index 0

# Validate JSONL structure (public_data validator)
PYTHONPATH=. conda run -n ms python public_data/scripts/validate_jsonl.py <path.jsonl>
```

## [Data & Datasets](data/README.md)
*   **[README](data/README.md)**: Main guide to data format, schema, and dataset pipeline.
*   **[JSONL Contract](data/JSONL_CONTRACT.md)**: The authoritative schema for JSONL records.
*   **[Intake Pipeline](data/INTAKE_PIPELINE.md)**: Guide to raw annotation intake, conversion, and preprocessing.
*   **[Packing](data/PACKING.md)**: Guide to packing modes for efficient training.
*   **[Visual Genome](data/VISUAL_GENOME.md)**: Specifics for Visual Genome data.
*   **[Fusion Dataset](data/FUSION_DATASET.md)**: Guide to multi-dataset fusion.

## [Training](training/)
*   **[Stage-2 Runbook](training/STAGE2_RUNBOOK.md)**: Runbook for Rollout-Matching and Stage-2 AB training.
*   **[Metrics & Losses](training/METRICS_LOSSES.md)**: Detailed explanation of training metrics and loss functions.
*   **[Coord Objective & Adapter](training/COORD_OBJECTIVE_AND_ADAPTER.md)**: SoftCE/W1 losses and offset adapters.

## [Evaluation](eval/README.md)
*   **[Detection Evaluator](eval/README.md)**: Guide to the offline detection evaluator.

## [Standards & Meta](standards/)
*   **[Repo Hygiene](standards/REPO_HYGIENE.md)**: Constitution for repository hygiene and best practices.
*   **[Code & Architecture Style](standards/CODE_STYLE.md)**: Transformers-inspired style guidelines (Option A).
*   **[Upstream Dependencies](standards/UPSTREAM.md)**: Information about upstream dependencies (HF Qwen3-VL, ms-swift).
*   **[Porting](standards/PORTING.md)**: Guide for porting features or models.

## [Notes](notes/)
*   **[Patent](notes/patent/draft.md)**: Patent draft notes.
