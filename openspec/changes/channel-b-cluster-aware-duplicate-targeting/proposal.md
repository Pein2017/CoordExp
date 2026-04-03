## Why

Full-val LVIS-proxy inference exposed a repeatable `duplication collapse` failure mode where Channel-B style outputs saturate the prediction budget with local same-description repeats and miss other objects. The current Stage-2 duplicate suppression path is intentionally narrow, so this is the right time to upgrade the canonical duplicate-target generator without changing the public Stage-2 training surface.

The immediate trigger for this change is an evidence-backed offline post-op detector study on the full-val run in [output/infer/coco1024_val200_lvis_proxy_merged_2b](/data/home/xiaoyan/AIteam/data/CoordExp/output/infer/coco1024_val200_lvis_proxy_merged_2b). A temporary duplicate-strip experiment at [dup_postop_eval.py](/data/home/xiaoyan/AIteam/data/CoordExp/temp/dup_postop_eval.py) removed tight same-description local duplicate clusters from [gt_vs_pred_scored.jsonl](/data/home/xiaoyan/AIteam/data/CoordExp/output/infer/coco1024_val200_lvis_proxy_merged_2b/gt_vs_pred_scored.jsonl) and improved full-val `coco_real bbox_AP` from `0.389928` to `0.397880` in [comparison.json](/data/home/xiaoyan/AIteam/data/CoordExp/temp/dup_postop_eval/comparison.json). That result is not the final training solution, but it is strong enough to justify moving the same detector idea earlier into Channel-B target construction.

## What Changes

- Upgrade Channel-B duplicate-target discovery from sequential same-description high-IoU duplicate bursts to deterministic cluster-aware duplicate-like grouping on rollout objects.
- Preserve the canonical `loss_duplicate_burst_unlikelihood` module name, B-only routing, empty config contract, and single-forward Channel-B realization.
- Redefine the runtime metadata consumed by duplicate unlikelihood so it represents cluster-aware duplicate continuations projected onto the post-triage clean prefix rather than only sequential duplicate bursts.
- Add crowd-safety rules so real crowded objects are less likely to be penalized as duplicates when they are spatially spread or explorer-supported.
- Extend duplicate-collapse diagnostics so runs can distinguish heavy local duplicate clusters from ordinary crowded scenes and verify that the new targeting path is active.

## Capabilities

### New Capabilities

- None.

### Modified Capabilities

- `stage2-ab-training`: Channel-B rollout preparation and duplicate suppression semantics change from sequential duplicate bursts to cluster-aware duplicate targeting while preserving the anchor-rooted single-forward contract.
- `teacher-forcing-objective-pipeline`: `loss_duplicate_burst_unlikelihood` remains canonical, but its consumed runtime metadata changes from sequential-burst first-divergence targets to cluster-aware duplicate continuation targets.
- `trainer-metrics-components`: duplicate-collapse diagnostics expand to expose cluster-aware metrics needed to validate suppression behavior and distinguish collapse from real crowding.

## Impact

- Affected code is expected in [target_builder.py](/data/home/xiaoyan/AIteam/data/CoordExp/src/trainers/stage2_two_channel/target_builder.py), [rollout_views.py](/data/home/xiaoyan/AIteam/data/CoordExp/src/trainers/stage2_two_channel/rollout_views.py), [stage2_two_channel.py](/data/home/xiaoyan/AIteam/data/CoordExp/src/trainers/stage2_two_channel.py), and duplicate-analysis helpers currently living under [src/analysis/](/data/home/xiaoyan/AIteam/data/CoordExp/src/analysis).
- No new CLI flags are introduced, and the canonical Stage-2 YAML objective surface stays intact.
- Training behavior changes in a correctness- and eval-validity-relevant way because Channel-B will generate different duplicate-unlikelihood targets on dense scenes.
- Reproducibility impact is positive if the new detector remains deterministic and fully covered by Stage-2 smoke tests plus downstream duplicate-metric checks.
- Background evidence to preserve for handoff:
  - offline study script: [dup_postop_eval.py](/data/home/xiaoyan/AIteam/data/CoordExp/temp/dup_postop_eval.py)
  - filtered scored predictions: [gt_vs_pred_scored.dedup.jsonl](/data/home/xiaoyan/AIteam/data/CoordExp/temp/dup_postop_eval/gt_vs_pred_scored.dedup.jsonl)
  - strip summary: [dedup_summary.json](/data/home/xiaoyan/AIteam/data/CoordExp/temp/dup_postop_eval/dedup_summary.json)
  - metric deltas: [comparison.json](/data/home/xiaoyan/AIteam/data/CoordExp/temp/dup_postop_eval/comparison.json)
