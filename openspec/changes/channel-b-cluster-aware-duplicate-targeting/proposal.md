## Why

CoordExp is currently fighting the same pathology in two disconnected ways:

- Stage-2 Channel-B applies a narrow, sequential same-description duplicate
  suppression path during training.
- Offline evaluation relies on ad hoc post-op duplicate stripping experiments to
  understand how much benchmark loss is coming from catastrophic re-enumeration.

That split is not working well enough. The current training-side path is too
narrow, and the current evaluation-side guard is not a canonical contract. The
result is a repeatable `duplicate-collapse` failure mode where the model
saturates the prediction budget with local same-description repeats and misses
other objects.

The trigger for this change is strong empirical evidence from the full-val
LVIS-proxy run in
[output/infer/coco1024_val200_lvis_proxy_merged_2b](/data/home/xiaoyan/AIteam/data/CoordExp/output/infer/coco1024_val200_lvis_proxy_merged_2b).
The temporary duplicate-strip experiment at
[dup_postop_eval.py](/data/home/xiaoyan/AIteam/data/CoordExp/temp/dup_postop_eval.py)
removed tight same-description local duplicate clusters from
[gt_vs_pred_scored.jsonl](/data/home/xiaoyan/AIteam/data/CoordExp/output/infer/coco1024_val200_lvis_proxy_merged_2b/gt_vs_pred_scored.jsonl)
and improved full-val `coco_real bbox_AP` from `0.389928` to `0.397880` in
[comparison.json](/data/home/xiaoyan/AIteam/data/CoordExp/temp/dup_postop_eval/comparison.json),
while `bbox_AR100` fell slightly. That trade-off matches the chosen objective
for this change: reduce catastrophic duplicate-collapse even if recall drops a
bit.

Additional historical evidence matters here:

- the failure already appears under greedy decode,
- the strongest current read is a local duplicate-collapse attractor rather than
  a generic global preference for duplicate objects,
- the repo already has an evidence-backed duplicate-like relation in
  [small_object_duplication_study.py](/data/home/xiaoyan/AIteam/data/CoordExp/src/analysis/small_object_duplication_study.py)
  that is stronger than the current sequential high-IoU heuristic.

This change therefore proposes one canonical duplicate-control policy shared
across Stage-2 training and offline evaluation, instead of continuing to treat
training suppression and offline post-op guarding as unrelated mechanisms.

## What Changes

- Replace the legacy sequential duplicate-burst path in
  `stage2_two_channel` with a canonical duplicate-control policy that operates
  on rollout objects using the duplicate-like relation proven out in
  [small_object_duplication_study.py](/data/home/xiaoyan/AIteam/data/CoordExp/src/analysis/small_object_duplication_study.py).
- Apply duplicate-control after anchor and explorer views are available but
  before GT matching, so duplicate collapse is suppressed as the core symptom
  rather than as a late clean-up step.
- Keep `loss_duplicate_burst_unlikelihood` as the canonical B-only suppression
  module and preserve its single-forward payload shape, but redefine the
  metadata producer so non-exempt non-survivors disappear from the positive
  prefix and become UL targets.
- Introduce a shared training/evaluation duplicate policy:
  - training uses it to suppress duplicate-collapse before GT matching,
  - offline evaluation applies the same policy as a post-op guard and reports
    both raw and guarded metrics/artifacts.
- Normalize duplicate-control metric names, artifact names, and typed knobs
  around the new policy rather than preserving legacy sequential terminology or
  backward-compat aliases.
- Keep malformed-output handling separate from this change. Malformed parsing
  remains owned by parse/salvage and invalid-rollout policy; this change covers
  duplicate-collapse semantics only.

## Capabilities

### New Capabilities

- None.

### Modified Capabilities

- `stage2-ab-training`: Channel-B duplicate handling changes from sequential
  duplicate bursts to a canonical pre-match duplicate-control policy that uses
  anchor plus explorer evidence and removes non-exempt duplicates from the
  positive prefix.
- `teacher-forcing-objective-pipeline`: `loss_duplicate_burst_unlikelihood`
  remains canonical, but its producer now emits duplicate-control decisions from
  the new shared policy while preserving the current `(boundary, rel_pos,
  token_id)` payload contract.
- `trainer-metrics-components`: duplicate-collapse metrics are normalized around
  the new duplicate-control policy and explicitly distinguish raw pathology from
  suppression decisions.
- `inference-pipeline`: offline inference/eval flow gains canonical guarded
  duplicate-control artifacts alongside raw artifacts without requiring new CLI
  flags.
- `detection-evaluator`: evaluator applies the shared duplicate-control policy
  as an offline guard and reports both raw and guarded metrics/artifacts.

## Impact

- Affected training/runtime code is expected in:
  - [rollout_views.py](/data/home/xiaoyan/AIteam/data/CoordExp/src/trainers/stage2_two_channel/rollout_views.py)
  - [target_builder.py](/data/home/xiaoyan/AIteam/data/CoordExp/src/trainers/stage2_two_channel/target_builder.py)
  - [stage2_two_channel.py](/data/home/xiaoyan/AIteam/data/CoordExp/src/trainers/stage2_two_channel.py)
  - [stage2_coordination.py](/data/home/xiaoyan/AIteam/data/CoordExp/src/trainers/stage2_coordination.py)
  - a new shared duplicate-control core under
    [src/common/](/data/home/xiaoyan/AIteam/data/CoordExp/src/common)
- Affected config/schema/docs are expected in:
  - [schema.py](/data/home/xiaoyan/AIteam/data/CoordExp/src/config/schema.py)
  - [loader.py](/data/home/xiaoyan/AIteam/data/CoordExp/src/config/loader.py)
  - [METRICS.md](/data/home/xiaoyan/AIteam/data/CoordExp/docs/training/METRICS.md)
  - [STAGE2_RUNBOOK.md](/data/home/xiaoyan/AIteam/data/CoordExp/docs/training/STAGE2_RUNBOOK.md)
  - [WORKFLOW.md](/data/home/xiaoyan/AIteam/data/CoordExp/docs/eval/WORKFLOW.md)
- Affected offline evaluation/inference code is expected in:
  - [pipeline.py](/data/home/xiaoyan/AIteam/data/CoordExp/src/infer/pipeline.py)
  - [artifacts.py](/data/home/xiaoyan/AIteam/data/CoordExp/src/infer/artifacts.py)
  - [detection.py](/data/home/xiaoyan/AIteam/data/CoordExp/src/eval/detection.py)
  - [orchestration.py](/data/home/xiaoyan/AIteam/data/CoordExp/src/eval/orchestration.py)
  - [artifacts.py](/data/home/xiaoyan/AIteam/data/CoordExp/src/eval/artifacts.py)
- No new CLI flags are introduced.
- `stage2_rollout_aligned` is explicitly out of scope for the first landing.
- Backward compatibility is not a goal for this refactor. Legacy sequential
  duplicate handling, legacy metric aliases, and compatibility shims may be
  removed rather than preserved.
- Primary acceptance posture:
  - reduce catastrophic duplicate-collapse on hard scenes,
  - improve or stabilize raw `AP` / `F1-ish`,
  - allow some recall loss,
  - always report both raw and guarded offline metrics.
- Background evidence to preserve for handoff:
  - offline study script:
    [dup_postop_eval.py](/data/home/xiaoyan/AIteam/data/CoordExp/temp/dup_postop_eval.py)
  - filtered scored predictions:
    [gt_vs_pred_scored.dedup.jsonl](/data/home/xiaoyan/AIteam/data/CoordExp/temp/dup_postop_eval/gt_vs_pred_scored.dedup.jsonl)
  - strip summary:
    [dedup_summary.json](/data/home/xiaoyan/AIteam/data/CoordExp/temp/dup_postop_eval/dedup_summary.json)
  - metric deltas:
    [comparison.json](/data/home/xiaoyan/AIteam/data/CoordExp/temp/dup_postop_eval/comparison.json)
