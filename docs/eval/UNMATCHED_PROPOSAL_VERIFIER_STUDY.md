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

The authority-first workflow is now staged:
- `prepare`
- `collect`
- `gate`
- `score`
- `audit`
- `report`

Collection-health summarization and gating live in the explicit `gate` stage. Rollout proposal scoring is the only part gated off by collection validity; the clean GT benchmark still runs per checkpoint.

Later stages may reuse frozen earlier artifacts instead of forcing a monolithic rerun.

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

Authoritative temperatures:
- `0.0`
- `0.3`
- `0.5`
- `0.7`

The default single-run config now uses `temperature=0.3`. Multi-temperature diagnosis should be expressed as multiple study configs or repeated launches over those four values; non-authoritative temperatures belong in appendix-only analysis.

## Important Config Keys

Smoke policy:
- `smoke.yaml` is the low-memory harness-validation profile.
- It is currently allowed to use `collection.backend_mode: hf` when the main vLLM path is too memory-heavy for the shared machine.
- `--smoke` on the CLI is only a cardinality override on top of whatever config you pass; it does not replace the config with `smoke.yaml`.

Dataset / subset:
- `dataset.jsonl_path`
- `dataset.sample_count`
- `dataset.seed`
- `run.stages`

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
- `collection.max_num_seqs`
- `collection.enforce_eager`
- `collection.disable_custom_all_reduce`

Backend notes:
- `collection.backend_mode: stage2_parity_vllm` is the preferred study collection path when proposal realism matters.
- It reuses the real Stage-2-style vLLM message/template contract rather than the older generic infer-engine local-vLLM path.
- Keep `tensor_parallel_size=1` and `batch_size=16` when reproducing the Stage-2-aligned collection diagnosis.

Collection-validity gate:
- `collection_gate.nonempty_pred_image_rate_min`
- `collection_gate.pred_count_total_min`
- `collection_gate.unmatched_count_min`

Eval bucketing:
- `eval.f1ish_pred_scope`
- `eval.f1ish_iou_thrs`
- `eval.semantic_model`

Verifier scoring:
- `scoring.device`
- `scoring.gt_batch_size`
- `scoring.masked_batch_size`
- `scoring.mask_fill`

Manual audit:
- `manual_audit.sample_count`
- `manual_audit.score_key`
- `manual_audit.label_path`

Local reviewer:
- `scripts/analysis/run_manual_audit_reviewer.py`
- accepts an existing `manual_audit_by_temp.csv` or a packaged combined CSV
- serves a small browser UI with:
  - next / prev navigation
  - keyboard shortcuts
  - label buttons
  - note editing
  - save-back to CSV / JSONL

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
- `subset/subset_manifest.json`
- `gt/gt_positives.jsonl`
- `gt/gt_hard_negatives.jsonl`
- `gt/gt_manifest.json`
- `checkpoints/<checkpoint-slug>/pipeline_config.yaml`
- `checkpoints/<checkpoint-slug>/collection_manifest.json`
- `checkpoints/<checkpoint-slug>/checkpoint_manifest.json`
- `checkpoints/<checkpoint-slug>/gt_vs_pred.jsonl`
- `checkpoints/<checkpoint-slug>/eval/matches.jsonl`
- `checkpoints/<checkpoint-slug>/proposal_table.jsonl`
- `checkpoints/<checkpoint-slug>/collection_health.json`
- `checkpoints/<checkpoint-slug>/gt_proxy_scores.jsonl`
- `checkpoints/<checkpoint-slug>/proposal_proxy_scores.jsonl`
- `checkpoints/<checkpoint-slug>/scoring_manifest.json`
- `checkpoints/<checkpoint-slug>/summary.json`
- `manual_audit/candidates.jsonl`
- `manual_audit/labeled.jsonl`
- `manual_audit/manifest.json`
- `manual_audit/summary.json`
- `report_manifest.json`
- `collection_health_by_temp.csv`
- `gt_clean_proxy_metrics_by_temp.csv`
- `rollout_proxy_metrics_by_temp.csv`
- `manual_audit_by_temp.csv`
- `report.md`
- `study_manifest.json`

For packaged manual-audit queues such as
`output/analysis/unmatched-proposal-verifier-manual-audit-v1/`, launch:

```bash
conda run -n ms python scripts/analysis/run_manual_audit_reviewer.py \
  --audit-csv output/analysis/unmatched-proposal-verifier-manual-audit-v1/manual_audit_recommended96.csv \
  --port 8765
```

Then open `http://127.0.0.1:8765` locally.

The reviewer writes labels back to:
- the selected audit CSV itself (`audit_label`, `audit_notes`)
- `manual_audit_labels.jsonl` next to that CSV

That JSONL uses the canonical fields:
- `audit_id`
- `audit_label`
- `audit_notes`

So it can be reused later as `manual_audit.label_path` when rerendering
summaries or reports.

## Notes

- If the configured checkpoint path is missing, the study also tries `result/<checkpoint-basename>` as a lightweight fallback and records which path was used in `checkpoint_manifest.json`.
- When the sampled subset lives outside the source dataset directory, the subset meta records the original dataset root and the generated infer pipeline config sets `run.root_image_dir` accordingly.
- Calibration is explicitly skipped in v1 because the study reports raw teacher-forced log-probability proxies rather than calibrated probabilities.
- The main study path remains vLLM-oriented. The smoke profile is allowed to diverge toward a lighter backend/runtime so the scorer, report, and artifact contracts can still be validated on a busy shared machine.
- Collection-invalid runs remain visible in `collection_health.*` and per-checkpoint summaries, but they are excluded from the main rollout-comparison interpretation.
- Without completed manual audit labels, the final report intentionally downgrades the recommendation to a non-promotion-ready conclusion.
