---
doc_id: docs.eval.unmatched-proposal-verifier-study
layer: docs
doc_type: runbook
status: draft
domain: eval
summary: Config-first runbook for the offline unmatched-proposal verifier ablation study.
updated: 2026-03-13
---

# Unmatched Proposal Verifier Study

Purpose: run the offline ablation that compares teacher-forced proposal-verification proxies for unmatched rollout objects.

Primary entrypoint:
- `scripts/analysis/run_unmatched_proposal_verifier.py`

Primary configs:
- `configs/analysis/unmatched_proposal_verifier/default.yaml`
- `configs/analysis/unmatched_proposal_verifier/smoke.yaml`

## What The Study Reuses

- rollout proposal collection:
  - `src/infer/pipeline.py`
  - `src/infer/engine.py`
- matched / unmatched bucketing:
  - `src/eval/detection.py`
- desc-span parsing:
  - `src/trainers/rollout_matching/parsing.py`
- teacher-forced scoring:
  - `src/analysis/unmatched_proposal_verifier.py`

The study does not add a new detector head or training loop.
Proposal collection stays on the existing infer/eval path, and scoring stays on the existing HF teacher-forced model path.

## Recommended Commands

Smoke:

```bash
conda run -n ms python scripts/analysis/run_unmatched_proposal_verifier.py \
  --config configs/analysis/unmatched_proposal_verifier/smoke.yaml
```

Default two-checkpoint matrix:

```bash
conda run -n ms python scripts/analysis/run_unmatched_proposal_verifier.py \
  --config configs/analysis/unmatched_proposal_verifier/default.yaml
```

## Important Config Keys

Smoke policy:
- `smoke.yaml` is the low-memory harness-validation profile.
- It is currently allowed to use `collection.backend_mode: hf` when the main vLLM path is too memory-heavy for the shared machine.
- `--smoke` on the CLI is only a cardinality override on top of whatever config you pass; it does not replace the config with `smoke.yaml`.

Dataset / subset:
- `dataset.jsonl_path`
- `dataset.sample_count`
- `dataset.seed`

Prompt controls:
- `prompts.prompt_variant`
- `prompts.object_field_order`

Rollout collection:
- `collection.backend_mode`
- `collection.temperature`
- `collection.repetition_penalty`
- `collection.batch_size`
- `collection.gpu_memory_utilization`
- `collection.max_model_len`

Eval bucketing:
- `eval.f1ish_pred_scope`
- `eval.f1ish_iou_thrs`
- `eval.semantic_model`

Verifier scoring:
- `scoring.device`
- `scoring.gt_batch_size`
- `scoring.masked_batch_size`
- `scoring.mask_fill`

Checkpoint list:
- `checkpoints[].path`
- `checkpoints[].name`
- optional per-checkpoint prompt overrides:
  - `checkpoints[].prompt_variant`
  - `checkpoints[].object_field_order`

## Artifact Layout

The study writes under:
- `<run.output_dir>/<run.name>/`

Key artifacts:
- `subset/sampled.coord.jsonl`
- `subset/sampled.coord.jsonl.meta.json`
- `gt/gt_positives.jsonl`
- `gt/gt_hard_negatives.jsonl`
- `checkpoints/<checkpoint-slug>/pipeline_config.yaml`
- `checkpoints/<checkpoint-slug>/checkpoint_manifest.json`
- `checkpoints/<checkpoint-slug>/gt_vs_pred.jsonl`
- `checkpoints/<checkpoint-slug>/eval/matches.jsonl`
- `checkpoints/<checkpoint-slug>/proposal_table.jsonl`
- `checkpoints/<checkpoint-slug>/gt_proxy_scores.jsonl`
- `checkpoints/<checkpoint-slug>/proposal_proxy_scores.jsonl`
- `checkpoints/<checkpoint-slug>/summary.json`
- `report.md`
- `study_manifest.json`

## Notes

- If the configured checkpoint path is missing, the study also tries `result/<checkpoint-basename>` as a lightweight fallback and records which path was used in `checkpoint_manifest.json`.
- When the sampled subset lives outside the source dataset directory, the subset meta records the original dataset root and the generated infer pipeline config sets `run.root_image_dir` accordingly.
- Calibration is explicitly skipped in v1 because the study reports raw teacher-forced log-probability proxies rather than calibrated probabilities.
- The main study path remains vLLM-oriented. The smoke profile is allowed to diverge toward a lighter backend/runtime so the scorer, report, and artifact contracts can still be validated on a busy shared machine.
