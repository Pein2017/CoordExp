thread_id: 019d614c-c72a-7ef2-b63d-fb4742478e63
updated_at: 2026-04-06T06:01:17+00:00
rollout_path: /data/home/xiaoyan/AIteam/data/CoordExp/.codex/sessions/2026/04/06/rollout-2026-04-06T05-38-34-019d614c-c72a-7ef2-b63d-fb4742478e63.jsonl
cwd: /data/home/xiaoyan/AIteam/data/CoordExp
git_branch: main

# Stage-2 proxy-dataset config audit, duplication-hook check, and loss-weight review

Rollout context: The user wanted a new Stage-2 production config for training from a 2B Stage-1 checkpoint using pseudo labels and the latest duplication mechanism, then asked for an audit of how Stage-2 training behaves with the proxy dataset and what loss modules/weights are enabled. The repo was inspected in `/data/home/xiaoyan/AIteam/data/CoordExp`. The investigation focused on the current `configs/stage2_two_channel/prod/` tree, the active OpenSpec deltas for proxy supervision and cluster-aware duplicate targeting, and the runtime code paths that actually consume proxy metadata.

## Task 1: Prepare a Stage-2 prod config for the 2B proxy run

Outcome: success

Preference signals:
- The user said: "I refer to my checkpoint `output/stage1_2b/coco_bbox_max60-coco80-desc_first-1024-lvis_proxy-merged`. I want to use it to train the `stage-2` with pseudo labels and enable the latest `duplication` hook mechanism in the openspec." -> this indicates they want the new config anchored to that exact checkpoint and the newest OpenSpec-described duplicate mechanism, not an ad hoc old recipe.
- The user added: "I also need to `train` on the `lvis extended` proxy dataset, like `configs/stage1/lvis_bbox_max60_1024.yaml`" -> this indicates they want the merged proxy dataset family, not pure COCO or pure LVIS.

Key steps:
- Traced the stage-2 operator surface through `docs/AGENT_INDEX.md`, `docs/SYSTEM_OVERVIEW.md`, `docs/IMPLEMENTATION_MAP.md`, and `docs/training/STAGE2_RUNBOOK.md` to confirm `configs/stage2_two_channel/` is the active Stage-2 tree.
- Read the active OpenSpec change `channel-b-cluster-aware-duplicate-targeting`, which describes the latest duplicate-targeting semantics as cluster-aware, pre-triage duplicate-like grouping feeding the existing `loss_duplicate_burst_unlikelihood` module.
- Checked the existing production profiles in `configs/stage2_two_channel/prod/` and identified the best inheritance base as `ab_mixed_coco1024_bmajority_channel_b_pseudo_positive.yaml`.
- Created a new profile: `configs/stage2_two_channel/prod/2b-ab_mixed_coco1024_lvis_proxy_channel_b_pseudo_positive_dup_targeting.yaml`.
- The new profile points to `output/stage1_2b/coco_bbox_max60-coco80-desc_first-1024-lvis_proxy-merged` and uses the LVIS-proxy dataset JSONLs under `public_data/coco/rescale_32_1024_bbox_max60_lvis_proxy/`.

Reusable knowledge:
- The merged proxy dataset family for this rollout is `public_data/coco/rescale_32_1024_bbox_max60_lvis_proxy/`, not the pure LVIS dataset under `public_data/lvis/...`.
- The Stage-2 pseudo-positive prod recipe already carries the canonical duplicate module `loss_duplicate_burst_unlikelihood` in the expected position, so the config can inherit that instead of re-declaring the full pipeline.

Failures and how to do differently:
- The first draft of the new config was incomplete for the user's intent because it only swapped the checkpoint and JSONL family; the audit later showed that the current Stage-2 runtime does not yet actually apply proxy-tier weighting, so the new config is only a launch scaffold until that code path exists.

References:
- [1] New config added: `configs/stage2_two_channel/prod/2b-ab_mixed_coco1024_lvis_proxy_channel_b_pseudo_positive_dup_targeting.yaml`
- [2] Existing inheritance base: `configs/stage2_two_channel/prod/ab_mixed_coco1024_bmajority_channel_b_pseudo_positive.yaml`
- [3] Proxy dataset summary: `public_data/coco/rescale_32_1024_bbox_max60_lvis_proxy/train.coord.summary.json` (shows `accepted_proxy_counts: plausible=63215, strict=41440`, `metadata_namespace: coordexp_proxy_supervision`)
- [4] New config contents include:
  - `model: output/stage1_2b/coco_bbox_max60-coco80-desc_first-1024-lvis_proxy-merged`
  - `train_jsonl: public_data/coco/rescale_32_1024_bbox_max60_lvis_proxy/train.coord.jsonl`
  - `val_jsonl: public_data/coco/rescale_32_1024_bbox_max60_lvis_proxy/val.coord.jsonl`

## Task 2: Audit proxy-dataset Stage-2 behavior and loss-weight decisions

Outcome: partial

Preference signals:
- The user said: "I have never tried on the `stage-2` training with `proxy dataset` yet. Please help me audit the training mechanism, loss weights decisions" -> this indicates they want a concrete mechanism/weight audit before launching the run, not just a config dump.
- Later the user asked: "Please list out all the `losses` I'll enable and their weights." -> this shows they want the answer in explicit module/weight form.
- The user then asked whether `adjacent_repulsion` is duplicate with the latest `duplicate_burst_unlikelihood`, whether they can keep only the latest one, and whether LVIS-extended objects should get different weights -> this indicates they want a distinction between duplicate suppression and coordinate regularization, plus a clear answer on proxy object weighting.

Key steps:
- Inspected the Stage-2 runtime path in `src/sft.py` and found that `stage2_two_channel` / `stage2_rollout_aligned` bypass the batch-extras collator, so the proxy supervision enricher is not used there.
- Checked `src/trainers/teacher_forcing/module_registry.py` and confirmed the Stage-2 objective config allowlists do not include `object_weight_mode` for the current runtime modules, despite the active OpenSpec delta expecting metadata-driven weighting.
- Verified that the only code paths currently using `proxy_desc_token_weights` / `proxy_coord_token_weights` are collator and metrics plumbing, not Stage-2 objective execution.
- Read the Stage-2 pseudo-positive prod config and listed the enabled objectives and weights:
  - `token_ce` weight `1.0`
    - `desc_ce_weight: 1.0`
    - `rollout_fn_desc_weight: 1.5`
    - `rollout_global_prefix_struct_ce_weight: 1.0`
  - `loss_duplicate_burst_unlikelihood` weight `2.0`
    - `config: {}`
  - `bbox_geo` weight `1.0`
    - `smoothl1_weight: 1.0`
    - `ciou_weight: 0.5`
  - `bbox_size_aux` weight `1.0`
    - `log_wh_weight: 0.05`
    - `oversize_penalty_weight: 0.0`
    - `oversize_area_frac_threshold: null`
    - `oversize_log_w_threshold: null`
    - `oversize_log_h_threshold: null`
    - `eps: 1e-6`
  - `coord_reg` weight `1.0`
    - `coord_ce_weight: 0.02`
    - `coord_gate_weight: 1.0`
    - `text_gate_weight: 0.1`
    - `soft_ce_weight: 0.1`
    - `w1_weight: 0.02`
    - `temperature: 1.0`
    - `target_sigma: 2.0`
    - `target_truncate: 8`
    - `adjacent_repulsion_weight: 0.0`
    - `adjacent_repulsion_filter_mode: same_desc`
    - `adjacent_repulsion_margin_ratio: 0.05`
    - `adjacent_repulsion_copy_margin: 0.8`
- Also noted the Channel-B knobs inherited from that config:
  - `pseudo_positive.coord_weight: 0.3`
  - `recovered_ground_truth_weight_multiplier: 3.0`
  - `triage_posterior.num_rollouts: 4`
- Compared those weights to the proxy dataset composition and found the merged proxy dataset is heavily skewed toward `plausible` proxies, which means treating all objects as hard GT would over-strengthen the noisiest part of the proxy set.

Reusable knowledge:
- For the current repo state, `stage2_two_channel` does not yet consume the proxy-supervision metadata path that the OpenSpec delta defines; proxy objects will not receive tier-specific soft weights unless the Stage-2 code is extended.
- The current Stage-2 proxy JSONL still carries metadata under `coordexp_proxy_supervision`, and the data artifact itself reports `plausible` and `strict` counts, but those weights are not yet applied by the Stage-2 trainer path.
- `adjacent_repulsion` is a coord-distribution regularizer inside `coord_reg`, not the duplicate-targeting mechanism; the latest duplicate mechanism is `loss_duplicate_burst_unlikelihood`.
- The active duplicate-targeting code path in `src/trainers/stage2_two_channel/target_builder.py` is still sequential same-description IoU dedup + duplicate-burst target construction, not the cluster-aware version described by the active OpenSpec delta.

Failures and how to do differently:
- The user’s proxy-dataset Stage-2 audit could not be fully green-lit because the code path the OpenSpec expects is not wired yet: Stage-2 bypasses the proxy collator, and the module registry does not yet admit the `object_weight_mode` contract.
- The duplicate-targeting mechanism is not yet the newer cluster-aware version in runtime; the code still uses `_sequential_dedup_bbox_objects(...)` and old duplicate-burst metrics, so the “latest spec” behavior is not what this run would exercise.
- Because of that, the safe recommendation was to keep `loss_duplicate_burst_unlikelihood`, keep `adjacent_repulsion_weight: 0.0`, and postpone launching the proxy-dataset Stage-2 run until proxy-tier weighting is implemented.

References:
- [1] Stage-2 bypass of proxy collator: `src/sft.py:2360` (`stage2_two_channel` / `stage2_rollout_aligned` set `data_collator = base_collator`)
- [2] Proxy batch-extras plumbing exists but is diagnostics-side: `src/data_collators/batch_extras_collator.py:26-92`, `src/trainers/batch_extras.py:15-16`, `src/trainers/metrics/mixins.py:517-518` and `706`
- [3] Stage-2 module allowlists do not yet include `object_weight_mode`: `src/trainers/teacher_forcing/module_registry.py:72-197`
- [4] Pseudo-positive prod config with inherited loss weights: `configs/stage2_two_channel/prod/ab_mixed_coco1024_bmajority_channel_b_pseudo_positive.yaml:16-88`
- [5] Current duplicate implementation remains sequential: `src/trainers/stage2_two_channel/target_builder.py:213-251`, `476-542`, `1165-1273`
- [6] Active OpenSpec delta for proxy weighting: `openspec/changes/add-lvis-coco-proxy-supervision/specs/stage2-ab-training/spec.md:5-19`
- [7] Active OpenSpec delta for cluster-aware duplicate targeting: `openspec/changes/channel-b-cluster-aware-duplicate-targeting/specs/stage2-ab-training/spec.md:22-43`, `openspec/changes/channel-b-cluster-aware-duplicate-targeting/specs/teacher-forcing-objective-pipeline/spec.md:13-20`
- [8] The direct answer given in the rollout: `adjacent_repulsion` is not duplicate suppression; `loss_duplicate_burst_unlikelihood` is the canonical duplicate module; LVIS-extended proxy objects should have different weights, but the code still needs proxy-weight plumbing for Stage-2 to honor that.
