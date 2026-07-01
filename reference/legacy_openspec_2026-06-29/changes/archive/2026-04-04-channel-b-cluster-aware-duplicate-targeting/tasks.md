## 1. OpenSpec And Contract Completion

- [x] 1.1 Update the existing change-local delta specs for
      [stage2-ab-training/spec.md](/data/home/xiaoyan/AIteam/data/CoordExp/openspec/changes/channel-b-cluster-aware-duplicate-targeting/specs/stage2-ab-training/spec.md),
      [teacher-forcing-objective-pipeline/spec.md](/data/home/xiaoyan/AIteam/data/CoordExp/openspec/changes/channel-b-cluster-aware-duplicate-targeting/specs/teacher-forcing-objective-pipeline/spec.md),
      and
      [trainer-metrics-components/spec.md](/data/home/xiaoyan/AIteam/data/CoordExp/openspec/changes/channel-b-cluster-aware-duplicate-targeting/specs/trainer-metrics-components/spec.md)
      so they describe the new pre-match duplicate-control contract, the
      unchanged UL payload shape, and the normalized metric names.
- [x] 1.2 Add change-local delta specs for
      [inference-pipeline/spec.md](/data/home/xiaoyan/AIteam/data/CoordExp/openspec/specs/inference-pipeline/spec.md)
      and
      [detection-evaluator/spec.md](/data/home/xiaoyan/AIteam/data/CoordExp/openspec/specs/detection-evaluator/spec.md)
      covering guarded artifact emission, raw-vs-guarded reporting, and the
      offline duplicate-control post-op contract.

## 2. Shared Duplicate-Control Core

- [x] 2.1 Add a repo-owned shared duplicate-control runtime helper under
      [src/common/](/data/home/xiaoyan/AIteam/data/CoordExp/src/common),
      expected as
      [duplicate_control.py](/data/home/xiaoyan/AIteam/data/CoordExp/src/common/duplicate_control.py),
      that owns:
      - duplicate feature extraction for bbox rollout objects,
      - the canonical pair predicate ported from
        [small_object_duplication_study.py](/data/home/xiaoyan/AIteam/data/CoordExp/src/analysis/small_object_duplication_study.py),
      - connected-component building,
      - conservative crowd-safety exemptions,
      - deterministic survivor ranking,
      - policy decision records,
      - and normalized duplicate-control metrics.
- [x] 2.2 Reuse or extend
      [semantic_desc.py](/data/home/xiaoyan/AIteam/data/CoordExp/src/common/semantic_desc.py),
      [schemas.py](/data/home/xiaoyan/AIteam/data/CoordExp/src/common/schemas.py),
      and
      [prediction_parsing.py](/data/home/xiaoyan/AIteam/data/CoordExp/src/common/prediction_parsing.py)
      only as needed so infer/eval/train consume one normalized object contract.

## 3. Stage-2 Training Refactor

- [x] 3.1 Rewrite
      [build_channel_b_rollout_view()](/data/home/xiaoyan/AIteam/data/CoordExp/src/trainers/stage2_two_channel/rollout_views.py)
      so it no longer performs legacy sequential dedup and instead returns raw
      anchor/explorer objects plus duplicate-control evidence and normalized raw
      duplicate metrics.
- [x] 3.2 Replace
      [_compute_duplicate_diagnostics()](/data/home/xiaoyan/AIteam/data/CoordExp/src/trainers/stage2_two_channel/target_builder.py)
      and
      [_sequential_dedup_bbox_objects()](/data/home/xiaoyan/AIteam/data/CoordExp/src/trainers/stage2_two_channel/target_builder.py)
      with shared duplicate-control consumption; update
      [_build_channel_b_triage()](/data/home/xiaoyan/AIteam/data/CoordExp/src/trainers/stage2_two_channel/target_builder.py),
      [_build_channel_b_supervision_targets()](/data/home/xiaoyan/AIteam/data/CoordExp/src/trainers/stage2_two_channel/target_builder.py),
      and
      [_build_duplicate_burst_unlikelihood_targets()](/data/home/xiaoyan/AIteam/data/CoordExp/src/trainers/stage2_two_channel/target_builder.py)
      to consume duplicate-control decisions rather than legacy
      `duplicate_bursts_by_boundary`.
- [x] 3.3 Update
      [stage2_two_channel.py](/data/home/xiaoyan/AIteam/data/CoordExp/src/trainers/stage2_two_channel.py)
      to aggregate normalized duplicate-control gauges/counters and persist the
      new policy metadata into Channel-B segment meta without changing the UL
      payload shape consumed by
      [loss_duplicate_burst_unlikelihood.py](/data/home/xiaoyan/AIteam/data/CoordExp/src/trainers/teacher_forcing/modules/loss_duplicate_burst_unlikelihood.py).
- [x] 3.4 Update
      [resolve_stage2_ab_metric_spec()](/data/home/xiaoyan/AIteam/data/CoordExp/src/trainers/stage2_coordination.py)
      so the normalized duplicate-control gauges reduce as weighted means and
      the normalized counters remain additive.
- [x] 3.5 Refactor
      [schema.py](/data/home/xiaoyan/AIteam/data/CoordExp/src/config/schema.py)
      and
      [loader.py](/data/home/xiaoyan/AIteam/data/CoordExp/src/config/loader.py)
      to expose only the minimal typed training surface for duplicate-control
      under `stage2_ab.channel_b`.

## 4. Offline Inference / Evaluation Guard

- [x] 4.1 Apply the shared duplicate-control policy in
      [detection.py](/data/home/xiaoyan/AIteam/data/CoordExp/src/eval/detection.py)
      and
      [orchestration.py](/data/home/xiaoyan/AIteam/data/CoordExp/src/eval/orchestration.py)
      so offline evaluation emits:
      - raw metrics,
      - guarded metrics,
      - a guarded JSONL artifact,
      - and a duplicate-control report.
- [x] 4.2 Update
      [artifacts.py](/data/home/xiaoyan/AIteam/data/CoordExp/src/eval/artifacts.py),
      [pipeline.py](/data/home/xiaoyan/AIteam/data/CoordExp/src/infer/pipeline.py),
      [artifacts.py](/data/home/xiaoyan/AIteam/data/CoordExp/src/infer/artifacts.py),
      and
      [scripts/evaluate_detection.py](/data/home/xiaoyan/AIteam/data/CoordExp/scripts/evaluate_detection.py)
      only as needed so raw and guarded artifact paths are resolved and
      persisted consistently without adding new CLI flags.

## 5. Docs, Names, And Metrics Normalization

- [x] 5.1 Update
      [METRICS.md](/data/home/xiaoyan/AIteam/data/CoordExp/docs/training/METRICS.md),
      [STAGE2_RUNBOOK.md](/data/home/xiaoyan/AIteam/data/CoordExp/docs/training/STAGE2_RUNBOOK.md),
      [WORKFLOW.md](/data/home/xiaoyan/AIteam/data/CoordExp/docs/eval/WORKFLOW.md),
      and
      [ARTIFACTS.md](/data/home/xiaoyan/AIteam/data/CoordExp/docs/ARTIFACTS.md)
      to document the normalized duplicate-control names, guarded evaluator
      artifacts, and the “raw plus guarded” reporting rule.
- [x] 5.2 Remove legacy sequential duplicate terminology from user-facing docs,
      metric names, and config names unless a surviving key still literally
      refers to the unchanged UL consumer contract.

## 6. Tests And Verification

- [x] 6.1 Extend
      [tests/test_stage2_ab_training.py](/data/home/xiaoyan/AIteam/data/CoordExp/tests/test_stage2_ab_training.py)
      with deterministic fixtures for:
      - non-sequential same-description chain clusters,
      - tight local duplicate-collapse,
      - spatially spread same-description crowd exemptions,
      - explorer-supported exemptions,
      - and collapsed UL multiplicity by `(boundary, token_id)`.
- [x] 6.2 Extend
      [tests/test_stage2_pending_metrics_aggregation.py](/data/home/xiaoyan/AIteam/data/CoordExp/tests/test_stage2_pending_metrics_aggregation.py)
      so the normalized duplicate-control gauges/counters reduce correctly:
      - `dup/raw/*` gauges finalize as weighted means,
      - `dup/raw/*_count` and `stage2_ab/channel_b/dup/N_*` remain additive,
      - and `N_duplicate_burst_unlikelihood_skipped_no_divergence` remains the
        one canonical additive UL-skip key.
- [x] 6.3 Add or extend offline-eval coverage in
      [tests/test_detection_eval_output_parity.py](/data/home/xiaoyan/AIteam/data/CoordExp/tests/test_detection_eval_output_parity.py),
      [tests/test_detection_eval_ingestion_diagnostics.py](/data/home/xiaoyan/AIteam/data/CoordExp/tests/test_detection_eval_ingestion_diagnostics.py),
      and related evaluator tests to assert:
      - guarded artifact emission,
      - raw-vs-guarded metrics coexistence,
      - deterministic duplicate-control report contents,
      - and deterministic guarded naming for
        `gt_vs_pred_scored_guarded.jsonl` and
        `matches@{iou_thr}_guarded.jsonl` where applicable.
- [x] 6.4 Run targeted tests:
      - `conda run -n ms python -m pytest tests/test_stage2_ab_training.py -q`
      - `conda run -n ms python -m pytest tests/test_stage2_pending_metrics_aggregation.py -q`
      - `conda run -n ms python -m pytest tests/test_detection_eval_output_parity.py -q`
- [x] 6.5 Run a `stage2_two_channel` smoke with one canonical smoke config and
      compare:
      - raw duplicate-control gauges,
      - suppression counters,
      - UL boundary counts,
      - raw eval metrics,
      - guarded eval metrics,
      - and reproducibility artifacts across repeated runs.
      To-do when dedicated GPUs are available:
      - rerun the targeted top-2 suspicious-duplication smoke slice with
        `num_rollouts: 2` and `max_new_tokens: 256`
      - confirm whether the first explorer rollout completes and whether the
        resulting run emits `stage2_ab/channel_b/dup/N_*` counters on the
        duplication-prone subset
      - if the targeted slice still proves too slow under HF rollouts, repeat
        the same smoke under a freer multi-GPU window rather than changing the
        canonical decode budget
      - 2026-04-04 rerun used canonical smoke
        `configs/stage2_two_channel/smoke/b_majority_coco1024_pseudo_positive_tiny_1step.yaml`
        via the existing top-2 subset override
        `temp/b_majority_coco1024_pseudo_positive_tiny_1step_dup_subset.yaml`
      - exact command:
        `ROOT_IMAGE_DIR=/data/CoordExp/public_data/coco/rescale_32_1024_bbox_max60 PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 conda run -n ms python -m src.sft --config temp/b_majority_coco1024_pseudo_positive_tiny_1step_dup_subset.yaml`
      - exit status: `0`
      - run dir:
        `output/stage2_ab/smoke/b_majority_coco1024_pseudo_positive_tiny_1step_dup_subset/smoke_tiny_1step-k4-pseudo_positive-dup_subset/v7-20260404-082030`
      - key artifacts:
        `logging.jsonl`, `monitor_dumps/step_000001.json`,
        `resolved_config.json`, `pipeline_manifest.json`,
        `runtime_env.json`, `run_metadata.json`
      - first explorer rollout completed under the requested decode slice:
        `rollout/explorer/gen_new_tokens_mean = 256.0`
      - duplicate-control counters were emitted on the subset:
        `stage2_ab/channel_b/dup/N_raw_bbox_valid = 10.0`,
        `N_clean_accepted = 10.0`,
        `N_clusters_total = 0.0`,
        `N_clusters_suppressed = 0.0`,
        `N_objects_suppressed = 0.0`,
        `N_ul_boundaries = 0.0`
      - raw gauges from `logging.jsonl`:
        `dup/raw/max_desc_count = 9.0`,
        `dup/raw/saturation_rate = 0.2`,
        `dup/raw/duplicate_like_max_cluster_size = 1.0`,
        `dup/raw/desc_entropy = 0.32508297`
      - repeated-run artifact root now contains `v0` through `v7`, each with
        the standard reproducibility sidecars; this rerun added `v7`
      - 2026-04-04 expanded suspicious-slice pre-run used the full
        8-sample subset under the Stage-2 server launcher:
        `server_gpus=0,1,2,3,4,5 train_gpus=6,7 config=temp/b_majority_coco1024_pseudo_positive_vllm_8gpu_dup_prerun.yaml conda run -n ms bash scripts/train_stage2.sh`
      - config prep needed two launcher-specific fixes before the run could
        start cleanly:
        - use the contract-compliant subset JSONL
          `temp/duplication_subset_seed/suspicious_duplication_subset_only/subset/monitor_subset.coord.jsonl`
          instead of the `*_abs.coord.jsonl` variant because
          `scripts/train_stage2.sh` preflight enforces relative image paths
        - expose `subset/images -> /data/CoordExp/public_data/coco/rescale_32_1024_bbox_max60/images`
          so JSONL-relative image checks pass under launcher preflight
      - first full-slice run dir:
        `output/stage2_ab/prerun/b_majority_coco1024_pseudo_positive_vllm_8gpu_dup_subset/prerun_8gpu-k4-pseudo_positive-suspicious_duplication_subset/v1-20260404-163059`
      - `v1` exited successfully and consumed the full suspicious subset with
        `train/samples_total = 8.0`, but the single step realized
        `stage2/channel_a = 1.0`, `stage2/channel_b = 0.0`, and
        `stage2_ab/b_ratio_realized = 0.0`, so it did **not** exercise the
        duplication path despite the authored `b_ratio: 0.85`
      - follow-up rerun forced `stage2_ab.schedule.b_ratio: 1.0` in
        `temp/b_majority_coco1024_pseudo_positive_vllm_8gpu_dup_prerun.yaml`
        and wrote:
        `output/stage2_ab/prerun/b_majority_coco1024_pseudo_positive_vllm_8gpu_dup_subset/prerun_8gpu-k4-pseudo_positive-suspicious_duplication_subset/v2-20260404-163403`
      - `v2` resolved config confirms `b_ratio = 1.0`, `max_steps = 1`, and
        the same full 8-sample suspicious subset for both train and val
      - `v2` produced a Channel-B monitor dump before manual interruption:
        `monitor_dumps/step_000001.json` reports
        `meta.stage2_channel = "B"`,
        `meta.selection = "suspicious_duplication"`,
        `meta.candidate_count = 4`,
        and `rollout_backend = "vllm"`
      - aggregated from `v2/monitor_dumps/step_000001.json` across the 4
        sampled suspicious records in that first Channel-B step:
        `raw_bbox_valid = 222`,
        `clean_accepted = 102`,
        `clusters_total = 19`,
        `clusters_suppressed = 14`,
        `clusters_exempt = 5`,
        `objects_suppressed = 120`,
        `duplicate_burst_unlikelihood_boundary_count = 8`
      - per-sample highlights from `v2`:
        - base_idx `0`: `raw_bbox_valid = 123`, `clean_accepted = 29`,
          `clusters_suppressed = 3`, `clusters_exempt = 2`,
          `objects_suppressed = 94`,
          `near_iou90_pairs_same_desc_count = 457`,
          `max_desc_count = 68.0`
        - base_idx `5`: `raw_bbox_valid = 56`, `clean_accepted = 36`,
          `clusters_suppressed = 9`, `objects_suppressed = 20`,
          `max_desc_count = 31.0`
        - base_idx `6`: `raw_bbox_valid = 30`, `clean_accepted = 24`,
          `clusters_suppressed = 2`, `objects_suppressed = 6`,
          `max_desc_count = 28.0`
        - base_idx `2`: `raw_bbox_valid = 13`, `clean_accepted = 13`,
          `clusters_total = 0`, `objects_suppressed = 0`
      - `v2` was interrupted after the first Channel-B step because the
        launcher stack did not reach a clean closeout quickly enough to justify
        pinning all 8 GPUs once the needed duplicate-control evidence had been
        captured; no `logging.jsonl` closeout was written for `v2`
      - next clean rerun should keep `b_ratio: 1.0` for the suspicious-slice
        verification and disable the step-1 eval phase so the launcher can
        close promptly after the first Channel-B step
      - 2026-04-04 clean verification rerun completed successfully as:
        `output/stage2_ab/prerun/b_majority_coco1024_pseudo_positive_vllm_8gpu_dup_subset/prerun_8gpu-k4-pseudo_positive-suspicious_duplication_subset/v3-20260404-164849`
      - `v3` resolved config confirms the intended final verification surface:
        - `stage2_ab.schedule.b_ratio = 1.0`
        - `training.eval_strategy = "no"`
        - `training.do_eval = false`
        - `training.max_steps = 1`
        - `training.effective_batch_size = 8`
        - train/val both use
          `temp/duplication_subset_seed/suspicious_duplication_subset_only/subset/monitor_subset.coord.jsonl`
      - `v3` launcher metadata in `run_metadata.json` confirms the canonical
        8-GPU Stage-2 split:
        `COORDEXP_STAGE2_LAUNCHER=scripts/train_stage2.sh`,
        `COORDEXP_STAGE2_SERVER_GPUS=0,1,2,3,4,5`,
        `COORDEXP_STAGE2_LEARNER_GPUS=6,7`
      - `v3/logging.jsonl` shows a clean completed Channel-B step over the full
        suspicious subset:
        - `stage2/raw_rollouts = 8.0`
        - `stage2_ab/channel_b/invalid_rollout = 0.0`
        - `train/samples_total = 8.0`
        - `dup/raw/max_desc_count = 44.875`
        - `dup/raw/saturation_rate = 0.19856464`
        - `dup/raw/duplicate_like_max_cluster_size = 33.375`
        - `dup/raw/near_iou90_pairs_same_desc_count = 539.0`
        - `stage2_ab/channel_b/dup/N_raw_bbox_valid = 520.0`
        - `stage2_ab/channel_b/dup/N_clean_accepted = 346.0`
        - `stage2_ab/channel_b/dup/N_clusters_total = 36.0`
        - `stage2_ab/channel_b/dup/N_clusters_exempt = 7.0`
        - `stage2_ab/channel_b/dup/N_clusters_suppressed = 29.0`
        - `stage2_ab/channel_b/dup/N_objects_suppressed = 174.0`
        - `stage2_ab/channel_b/dup/N_ul_boundaries = 20.0`
        - `stage2_ab/channel_b/dup/N_duplicate_burst_unlikelihood_skipped_no_divergence = 0.0`
        - `diag/duplicate_burst/num_terms = 54.0`
        - `diag/duplicate_burst/num_ul_boundaries = 18.0`
        - `train/optimization/loss_duplicate_burst_unlikelihood = 0.17865312`
      - `v3/monitor_dumps/step_000001.json` confirms the trainer selected the
        suspicious-duplication Channel-B path:
        `meta.phase = "train"`,
        `meta.stage2_channel = "B"`,
        `meta.selection = "suspicious_duplication"`,
        `meta.candidate_count = 4`,
        `meta.rollout_backend = "vllm"`
      - per-sample duplicate-control behavior in the clean `v3` dump remained
        deterministic and consistent with the earlier interrupted `v2` evidence:
        - base_idx `0`: `clusters_total = 5`, `clusters_suppressed = 3`,
          `clusters_exempt = 2`, `objects_suppressed = 94`,
          `raw_bbox_valid = 123`, `clean_accepted = 29`,
          `duplicate_like_max_cluster_size = 57.0`,
          `max_desc_count = 68.0`, `UL boundaries = 3`
        - base_idx `5`: `clusters_total = 11`, `clusters_suppressed = 9`,
          `clusters_exempt = 2`, `objects_suppressed = 20`,
          `raw_bbox_valid = 56`, `clean_accepted = 36`,
          `duplicate_like_max_cluster_size = 10.0`,
          `max_desc_count = 31.0`, `UL boundaries = 4`
        - base_idx `6`: `clusters_total = 3`, `clusters_suppressed = 2`,
          `clusters_exempt = 1`, `objects_suppressed = 6`,
          `raw_bbox_valid = 30`, `clean_accepted = 24`,
          `duplicate_like_max_cluster_size = 20.0`,
          `max_desc_count = 28.0`, `UL boundaries = 1`
        - base_idx `2`: `clusters_total = 0`, `clusters_suppressed = 0`,
          `objects_suppressed = 0`, `raw_bbox_valid = 13`,
          `clean_accepted = 13`, `duplicate_like_max_cluster_size = 1.0`,
          `max_desc_count = 3.0`, `UL boundaries = 0`
