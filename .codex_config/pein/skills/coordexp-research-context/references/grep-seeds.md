# Grep Seeds

Use these to get a broad context sweep without guessing.

## Project-Wide Background

```bash
rg -n "stage2|Channel-B|duplicate_ul|clean-prefix|rollout" docs progress openspec configs
rg -n "full_idea|near_dup|symptom|diagnosis|audit" progress
rg -n "Requirement:|duplicate_ul|stage2-ab-training|teacher-forcing" openspec/specs openspec/changes
```

## Stage-2 / Channel-B

```bash
rg -n "stage2_two_channel|accepted_objects_clean|duplicate_bursts|duplicate_ul|closure_supervision" src docs openspec tests
rg -n "duplicate_iou_threshold|stage2_ab.pipeline|rollout_matching.pipeline" configs docs openspec
```

## Eval / Metrics

```bash
rg -n "eval_rollout/|eval_det_|rollout/mAP|rollout/f1" docs openspec progress
rg -n "dup/|stage2_ab/channel_b/dup/|closure_supervision/N_drop" docs openspec tests
```
