# Grep Seeds

Use these to get a broad context sweep without guessing.

## Project-Wide

```bash
rg -n "stage1_detection_teacher_forcing|DetectionScene|DetectionSupervisionView|MetricEvent|DiagnosticEvent|DetectionAssignment|CorrectionEvent" docs src tests configs
rg -n "stage2_rollout_correction|stage2_coordination|stage2_rollout_runtime|rollout_runtime|rollout_aligned_targets|rollout_aligned_evaluator|stage2_vllm_server" docs openspec src scripts configs tests
rg -n "runtime-architecture-refactor-program|pipeline_manifest|run_metadata|trainer_setup|resolved_config.json|effective_runtime.json" docs openspec src tests
rg -n "contract|jsonl|geometry|packing" docs/data src/datasets src/detection tests
```

## Eval / Metrics

```bash
rg -n "infer|backend|artifacts|orchestration|confidence|metrics|Oracle-K|proxy_eval_bundle" docs/eval docs/training src scripts tests
rg -n "duplicate_control|metrics_guarded|per_image_guarded|duplicate_guard_report|gt_vs_pred_scored_guarded|run_metadata|pipeline_manifest" docs openspec tests src
rg -n "raw-text|coord-token|norm1000|pred_coord_mode|bbox_format|confidence_postop|constant-score|cxcy_logw_logh|cxcywh" docs progress src configs tests
```

## Evidence / History

```bash
rg -n "val200|limit=200|first 200|full-val|full val|throughput|GPU|launch shape|kept / total" progress/benchmarks progress/diagnostics
rg -n "full_idea|near_dup|symptom|diagnosis|audit|canonical|supersedes|artifact" progress
```
