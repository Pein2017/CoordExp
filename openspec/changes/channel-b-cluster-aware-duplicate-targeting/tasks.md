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
- [ ] 6.5 Run a `stage2_two_channel` smoke with one canonical smoke config and
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
