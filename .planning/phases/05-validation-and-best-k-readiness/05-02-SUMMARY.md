---
phase: 05-validation-and-best-k-readiness
plan: 02
status: completed
subsystem: runtime-smoke
tags: [stage2, smoke, pseudo_positive, vllm, runtime]
requires:
  - 05-01
provides:
  - Full-pipeline enabled `K=4` pseudo-positive smoke evidence
  - Runtime proof that Channel-B rollout execution reaches the new triage metrics
affects: [phase-05-plan-03]
tech-stack:
  added: []
  patterns:
    - Use the repo-native combined vLLM server + learner launcher for server-mode smoke
key-files:
  created: []
  modified:
    - configs/stage2_two_channel/smoke/b_majority_coco1024_pseudo_positive_4steps.yaml
requirements-completed: [VAL-02]
completed: 2026-03-22
verification:
  - "`server_gpus=2 train_gpus=3,4,5,6 config=configs/stage2_two_channel/smoke/b_majority_coco1024_pseudo_positive_4steps.yaml wait_timeout=120 conda run -n ms bash scripts/train_stage2.sh`"
---

# Phase 5 Plan 02 Summary

**Completed: full-pipeline pseudo-positive smoke reached and finished the enabled `K=4` runtime path**

## Accomplishments
- Switched the smoke profile to the requested stage-1 checkpoint and reduced the
  dataset limits to `train_sample_limit=32` and `val_sample_limit=4`.
- Added an explicit dedicated rollout-server endpoint to the smoke YAML so the
  run no longer depended on an implicit default port.
- Ran the enabled pseudo-positive smoke through the repo-native combined
  launcher, which booted a dedicated vLLM rollout server and then executed the
  learner via `src.sft`.
- Completed the run to `global_step/max_steps=4/4`.

## Evidence
- Run directory:
  `output/stage2_ab/smoke/b_majority_coco1024/pseudo_positive_4steps/smoke_b_majority_coco1024_pseudo_positive_4steps/v5-20260322-170102`
- Final-step runtime evidence from `logging.jsonl`:
  - `stage2/channel_b=1.0`
  - `stage2/raw_rollouts=8.0`
  - `train/triage/pseudo_positive_selected_count=2.0`
  - `train/triage/pseudo_positive_selected_support_rate_num=5.0`
  - `train/triage/pseudo_positive_selected_support_rate_den=6.0`
  - `train/triage/recovered_ground_truth_count=7.0`
  - `stage2_ab/channel_b/dup/N_duplicate_bursts=0.0`
  - `train/optimization/loss_dead_anchor_suppression=0.0`
- Monitor-dump evidence from `step_000002.json`:
  - `valid_explorer_count=3`
  - `anchor_support_counts=[2, 3]`
  - `anchor_support_rates=[0.666..., 1.0]`
  - `pseudo_positive_anchor_indices=[0, 1]`

## Notes
- Earlier direct `src.sft` attempts were still useful because they exposed two
  concrete environment issues:
  - worktree-relative checkpoint resolution
  - missing / incompatible external rollout server
- The successful smoke path resolved both by using:
  - an absolute local checkpoint path
  - a dedicated launcher-managed local rollout server on a free GPU
- No explorer-preparation abort was triggered in this successful smoke. That
  failure telemetry remains covered by the implementation and regression path
  rather than by this happy-path runtime run.

---
*Phase: 05-validation-and-best-k-readiness*
*Completed: 2026-03-22*
