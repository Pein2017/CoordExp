# Implementation Notes — `add-dataset-prompt-variants-coco80`

## Reproducibility Record (Prompt Variants)

Use this checklist for any run that uses a non-default prompt variant.

| Field | Value / Path |
| --- | --- |
| Training config path | `<repo>/configs/.../*.yaml` with `custom.extra.prompt_variant` |
| Inference config path | `<repo>/configs/infer/*.yaml` with `infer.prompt_variant` |
| Run name | `run.name` from inference/training YAML |
| Seed | `infer.generation.seed` (and training seed in YAML) |
| Selected prompt variant | `default` or `coco_80` |
| Unified predictions artifact | `<run_dir>/gt_vs_pred.jsonl` |
| Inference summary artifact | `<run_dir>/summary.json` |
| Resolved pipeline config artifact | `<run_dir>/resolved_config.json` |

## Artifact Checks

- `resolved_config.json` MUST include `infer.prompt_variant`.
- `summary.json` MUST include `infer.prompt_variant`.
- For COCO reproducibility, train and infer variants SHOULD match (`coco_80` ↔ `coco_80`).

## Validation Evidence (this implementation)

- Prompt/inference/fusion regression tests:
  - `PYTHONPATH=. conda run -n ms python -m pytest -q tests/test_prompt_variants.py tests/test_infer_batch_decoding.py tests/test_unified_infer_pipeline.py tests/test_fusion_config.py`
- OpenSpec strict validation:
  - `openspec validate add-dataset-prompt-variants-coco80 --type change --strict`
