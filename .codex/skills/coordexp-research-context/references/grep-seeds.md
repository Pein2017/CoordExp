# Grep Seeds

Use these to get a broad context sweep without guessing.

## Project-Wide Background

```bash
rg -n "stage2|Channel-B|loss_dead_anchor_suppression|clean-prefix|rollout|runtime-architecture-refactor-program" docs progress openspec configs
rg -n "full_idea|near_dup|symptom|diagnosis|audit" progress
rg -n "Requirement:|loss_dead_anchor_suppression|stage2-ab-training|rollout-matching-sft|teacher-forcing|runtime-critical refactors" openspec/specs openspec/changes
```

## Stage-2 / Channel-B

```bash
rg -n "stage2_two_channel|stage2_rollout_aligned|rollout_aligned_targets|rollout_aligned_evaluator|accepted_objects_clean|duplicate_bursts|loss_dead_anchor_suppression|dead_anchor|closure_supervision" src docs openspec tests
rg -n "stage2_vllm_server|rollout_runtime|server_gpus|train_gpus|stage2_ab.pipeline|rollout_matching.pipeline" src scripts configs docs openspec
```

## Eval / Metrics

```bash
rg -n "eval/detection/|eval/parsing/|eval/runtime/|eval_det_|rollout/mAP|rollout/f1" docs openspec progress src tests
rg -n "dup/|stage2_ab/channel_b/dup/|loss_dead_anchor_suppression|run_metadata|pipeline_manifest" docs openspec tests src
```

## Runtime / Artifacts

```bash
rg -n "src/bootstrap|pipeline_manifest|run_metadata|trainer_setup|write_infer_summary|evaluate_and_save_outputs" src docs openspec tests
rg -n "resolved_config.json|resolved_config.path|runtime_env.json|run_metadata.json|summary.json|metrics.json|gt_vs_pred.jsonl" docs src tests openspec
```
