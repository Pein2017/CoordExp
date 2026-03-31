## Why

The current 2B comparison between the Stage-1 `original` checkpoint and the Stage-2 `a_only` checkpoint suggests that both models already have strong semantic objectness, while the remaining gap is concentrated in missed rollout objects, crowded-scene separability, and prefix-dependent continuation behavior. We need a reproducible offline study that isolates why FN persists under fixed checkpoints, without conflating those questions with ongoing training changes.

This study should be treated as a direct extension of the authority-first experiment style established in [2026-03-16-unmatched-proposal-verifier-ablation](/data/home/xiaoyan/AIteam/data/CoordExp/openspec/changes/archive/2026-03-16-unmatched-proposal-verifier-ablation/proposal.md): staged evidence, explicit validity gates, and a small manually auditable benchmark before drawing a strong conclusion.

## What Changes

- Add a fixed-checkpoint rollout-factor analysis workflow for offline COCO-1024 `train` and `val` JSONL splits, optimized for intensive ablations on small fixed hard-case subsets rather than one broad full-split sweep.
- Define a deterministic bootstrap selector that ranks a frozen candidate pool per dataset split and derives two canonical hard-case subsets, `Hard-32` and `Hard-16`, where `Hard-16` is the top-16 prefix subset of `Hard-32` within the same split.
- Define a reproducible full factor matrix that compares `original` and `a_only` under controlled decode, prefix, and sequence-length conditions on the same split-matched `Hard-16` and `Hard-32` subsets.
- Add study support for explicit prefix-order interventions, including `oracle_gt_prefix_train_order`, `oracle_gt_prefix_random_order`, `self_prefix`, plus switched/broken continuation variants for targeted stress tests.
- Add stochastic union-of-`K` coverage analysis to distinguish “never emitted” objects from “sometimes emitted but not chosen” objects.
- Add a rollout-health gate, inspired by the archived unmatched-verifier study, so cells with parser collapse, empty outputs, or unusable rollout populations are kept visible but excluded from main causal attribution.
- Add a continuation-partition contract so prefix-injected objects are normatively separated from continuation output before `prefix_sensitive_miss` is assigned.
- Add a normative GT-object identity contract for per-GT recovery tables and review overlays.
- Add sequence-length / EOS-bias diagnostics that compare emitted object-count and token-count behavior against training-style object-list priors.
- Freeze the initial canonical checkpoint pair as:
  - `original = output/stage1_2b/coco_bbox_max60-hard_ce_soft_ce_w1_gate_merged-1332`
  - `a_only = output/stage2_ab/2b_1024/a_only_iter1/merged_ckpt-900`
- Freeze the initial canonical dataset sources as:
  - `train = public_data/coco/rescale_32_1024_bbox_max60/train.coord.jsonl`
  - `val = public_data/coco/rescale_32_1024_bbox_max60/val.coord.jsonl`
- Preserve `a_only` provenance from `output/stage2_ab/2b_1024/a_only_iter1/epoch_2-eff_size_64-n_softctx_iter_1-a_only/v0-20260309-102351/config_source.yaml` in the initial manifests and reports.
- Add study artifacts and summary reports that separate deterministic baseline, sampled coverage, prefix sensitivity, and length-sensitivity evidence instead of collapsing them into one score.
- Standardize multi-GPU execution for fixed-checkpoint offline studies using four local GPUs without requiring training, with cell/shard provenance that stays joinable at the per-image and per-object level.
- Define the canonical study as a staged experiment program so design and implementation can proceed in a controlled order instead of one unbounded factor sweep.

## Capabilities

### New Capabilities
- `rollout-fn-factor-analysis-study`: A reproducible offline study workflow for fixed-checkpoint rollout diagnosis across checkpoint, decoding, prefix, and sequence-length factors.

### Modified Capabilities
- None.

## Impact

- Affected code is expected to live primarily under `configs/analysis/`, `scripts/analysis/`, and `src/analysis/`.
- The workflow will reuse existing inference/parity/evaluation surfaces rather than changing training behavior.
- The study will consume fixed checkpoints plus offline `train` / `val` JSONL only; it does not require additional training runs.
- The study is intentionally `hf`-only for the first version so prefix continuation and length behavior stay easy to interpret.
- Main systems impacted: checkpoint manifesting, subset definition, inference backend orchestration, rollout collection, evaluation aggregation, and GT-vs-Pred audit artifacts.
