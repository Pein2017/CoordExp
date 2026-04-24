# Grep Seeds

Use these to get a broad context sweep without guessing.

## Project-Wide Background

```bash
rg -n "stage2|Channel-B|loss_dead_anchor_suppression|clean-prefix|rollout|runtime-architecture-refactor-program|stage2_coordination" docs progress openspec configs src
rg -n "full_idea|near_dup|symptom|diagnosis|audit|canonical|supersedes|artifact" progress
rg -n "Requirement:|loss_dead_anchor_suppression|stage2-ab-training|rollout-matching-sft|teacher-forcing|runtime-critical refactors|stable behavior" openspec/specs openspec/changes
```

## Stage-2 / Channel-B

```bash
rg -n "stage2_coordination|stage2_two_channel|stage2_rollout_aligned|rollout_aligned_targets|rollout_aligned_evaluator|accepted_objects_clean|duplicate_bursts|loss_dead_anchor_suppression|dead_anchor|closure_supervision" src docs openspec tests
rg -n "stage2_vllm_server|rollout_runtime|server_gpus|train_gpus|stage2_ab.pipeline|rollout_matching.pipeline" src scripts configs docs openspec
```

## Eval / Metrics

```bash
rg -n "eval/detection/|eval/parsing/|eval/runtime/|eval_det_|rollout/mAP|rollout/f1|bbox_AP|proxy_eval_bundle_summary" docs openspec progress src tests
rg -n "duplicate_control|metrics_guarded|per_image_guarded|duplicate_guard_report|gt_vs_pred_scored_guarded|loss_dead_anchor_suppression|run_metadata|pipeline_manifest" docs openspec tests src
rg -n "raw-text|coord-token|norm1000|pred_coord_mode|bbox_format|confidence_postop|constant-score|cxcy_logw_logh|cxcywh" docs progress src configs tests
```

## Runtime / Artifacts

```bash
rg -n "src/bootstrap|pipeline_manifest|run_metadata|trainer_setup|write_infer_summary|evaluate_and_save_outputs|build_infer_summary_payload|resolve_infer_artifact_paths" src docs openspec tests
rg -n "resolved_config.json|resolved_config.path|runtime_env.json|effective_runtime.json|experiment_manifest.json|run_metadata.json|summary.json|metrics.json|timing_summary.json|gt_vs_pred.jsonl|gt_vs_pred_scored.jsonl" docs progress src tests openspec
rg -n "val200|limit=200|first 200|full-val|full val|throughput|GPU|launch shape|kept / total" progress/benchmarks progress/diagnostics
```
