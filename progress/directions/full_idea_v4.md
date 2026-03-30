---
title: Full Idea v4
status: active
scope: stage1-stage2
topics: [stage1, stage2, pseudo-positive, pseudo-label, clean-prefix, triage-posterior, k4]
supersedes: progress/directions/full_idea_v3.md
references:
  - docs/PROJECT_CONTEXT.md
  - docs/training/STAGE1_OBJECTIVE.md
  - docs/training/STAGE2_RUNBOOK.md
  - docs/training/METRICS.md
  - openspec/specs/stage2-ab-training/spec.md
  - progress/directions/full_idea_v3.md
  - progress/directions/stage2_clean_prefix_v2.md
  - progress/directions/stage2_emish_set_supervision_v1.md
---

# Full Idea v4

This note is the new research-history summary for the current Stage-1 plus Stage-2 direction.

Read it after the stable docs/spec layer when you want one place that explains:

- what the latest Stage-1 pipeline is actually responsible for,
- what the latest Stage-2 pseudo-positive pipeline is actually doing,
- why `K=4` now matters,
- and how the current contract differs from:
  - [full_idea_v3.md](full_idea_v3.md),
  - [stage2_clean_prefix_v2.md](stage2_clean_prefix_v2.md),
  - [stage2_emish_set_supervision_v1.md](stage2_emish_set_supervision_v1.md).

## Status

- The current normative behavior lives in:
  - [Project Context](../../docs/PROJECT_CONTEXT.md)
  - [Stage-1 Objective](../../docs/training/STAGE1_OBJECTIVE.md)
  - [Stage-2 Runbook](../../docs/training/STAGE2_RUNBOOK.md)
  - [Stage-2 Metrics](../../docs/training/METRICS.md)
  - [Stage-2 AB spec](../../openspec/specs/stage2-ab-training/spec.md)
- This file is intentionally non-normative.
- Its job is to explain why the current repo ended up with:
  - a stronger Stage-1 foundation surface,
  - a typed Stage-2 pseudo-positive mechanism,
  - and an authored `K=4` multi-view Channel-B profile.

## Executive Summary

The simplest way to think about v4 is:

1. Stage-1 remains the language-of-boxes foundation.
2. Stage-2 remains the rollout-time correction layer.
3. The old vague "pseudo-label later maybe" idea has now become a concrete Stage-2 `pseudo_positive` contract.
4. `K=4` is not "large-K RL"; it is a small multi-view triage setup:
   - `1` deterministic anchor rollout,
   - `3` stochastic explorer rollouts.
5. The current pseudo-label mechanism is not a union-over-explorers rebuild.
   It is anchor-centric:
   - unmatched anchor clean objects collect explorer support,
   - high-support candidates can be promoted,
   - overlapping candidates are clustered,
   - cluster losers are demoted back to shielded context,
   - explorer-only non-GT-backed objects are still not promoted into prefix positives by default.

In one sentence:

> Keep Stage-1 strong and teacher-forced, keep Stage-2 clean-prefix and one-forward, and make pseudo-labeling a small, typed, anchor-centric, support-weighted `K=4` extension rather than a free-form rebuilt pseudo-target.

## The Latest Stage-1 Pipeline

### What Stage-1 is for now

Stage-1 is still the baseline teacher-forced SFT layer.

Its job is to teach:

- the CoordJSON output protocol,
- the coord-token vocabulary and calibration,
- stable geometry-aware serialization,
- and packing-friendly large-scale supervised training.

Current handles:

- docs:
  - [Stage-1 Objective](../../docs/training/STAGE1_OBJECTIVE.md)
- config:
  - [configs/stage1/sft_base.yaml](../../configs/stage1/sft_base.yaml)
  - [configs/stage1/profiles/4b/coord_soft_ce_gate_coco80_desc_first.yaml](../../configs/stage1/profiles/4b/coord_soft_ce_gate_coco80_desc_first.yaml)
  - [configs/stage1/lvis_bbox_max60_1024.yaml](../../configs/stage1/lvis_bbox_max60_1024.yaml)
- code:
  - [src/sft.py](../../src/sft.py)

### What Stage-1 now optimizes

Current Stage-1 is stronger than the older "pure coord-distribution only" historical framing.

The active Stage-1 surface is:

- base CE on non-coordinate tokens,
- coord-only `coord_soft_ce_w1` supervision:
  - optional hard coord CE,
  - soft CE,
  - W1,
  - coord-vocab gate,
- optional decoded-box `bbox_geo`,
- optional `bbox_size_aux`,
- optional coord-offset adapter.

This matters because the latest Stage-1 is no longer just a token-format warmup.
It is the checkpoint-quality foundation that Stage-2 can build from.

Concrete handles:

- docs:
  - [Stage-1 Objective](../../docs/training/STAGE1_OBJECTIVE.md)
- configs:
  - [configs/stage1/_shared/coord_soft_ce_gate_4b.yaml](../../configs/stage1/_shared/coord_soft_ce_gate_4b.yaml)
  - [configs/stage1/_shared/coord_soft_ce_gate_2b.yaml](../../configs/stage1/_shared/coord_soft_ce_gate_2b.yaml)
  - [configs/stage1/smoke/lvis_bbox_max60_1024.yaml](../../configs/stage1/smoke/lvis_bbox_max60_1024.yaml)
- tests:
  - [tests/test_coord_softce_w1_loss.py](../../tests/test_coord_softce_w1_loss.py)
  - [tests/test_stage1_metric_key_parity.py](../../tests/test_stage1_metric_key_parity.py)
  - [tests/test_stage1_static_packing_runtime_config.py](../../tests/test_stage1_static_packing_runtime_config.py)

### What Stage-1 is not

Stage-1 is not where the pseudo-label or `K=4` logic lives.

There is no rollout-view triage loop in Stage-1.
There is no `pseudo_positive` contract in Stage-1.
There is no anchor/explorer logic in Stage-1.

That separation is part of the current design maturity:

- Stage-1 teaches the model how to speak boxes.
- Stage-2 teaches the model how to behave under its own rollout-time set decisions.

The separation is visible in the fact that Stage-2 rejects legacy Stage-1-style objective authoring as the active Channel-B surface and instead uses `stage2_ab.pipeline.objective[]`.

## The Latest Stage-2 Pipeline

### The active mental model

The live Stage-2 contract is not just "duplicate UL plus clean prefix" anymore.

It is now:

- still clean-prefix,
- still one-forward,
- still anchor-first,
- but extended with a typed pseudo-positive mechanism.

Current authoritative references:

- [Stage-2 Runbook](../../docs/training/STAGE2_RUNBOOK.md)
- [Stage-2 Metrics](../../docs/training/METRICS.md)
- [Stage-2 AB spec](../../openspec/specs/stage2-ab-training/spec.md)

### Channel split in the current repo

The active Stage-2 split is:

- Channel-A:
  - GT-anchored, teacher-forced stabilization path,
  - geometry and text/coord objective modules declared through `stage2_ab.pipeline.objective[]`.
- Channel-B:
  - rollout-time set/cardinality correction path,
  - anchor/explorer triage,
  - clean-prefix reconstruction,
  - FN injection,
  - duplicate UL,
  - pseudo-positive partial promotion.

This is materially more operational than the older notes because the supported knobs now live in a typed config schema and the trainer/runtime paths are explicitly organized around packed, step-budgeted Channel-B execution.

Concrete handles:

- configs:
  - [configs/stage2_two_channel/prod/ab_mixed.yaml](../../configs/stage2_two_channel/prod/ab_mixed.yaml)
  - [configs/stage2_two_channel/prod/ab_mixed_coco1024_bmajority_channel_b_pseudo_positive.yaml](../../configs/stage2_two_channel/prod/ab_mixed_coco1024_bmajority_channel_b_pseudo_positive.yaml)
- code:
  - [src/trainers/stage2_two_channel.py](../../src/trainers/stage2_two_channel.py)
  - [src/trainers/stage2_two_channel/target_builder.py](../../src/trainers/stage2_two_channel/target_builder.py)
  - [src/trainers/stage2_two_channel/coordination.py](../../src/trainers/stage2_two_channel/coordination.py)
  - [src/trainers/stage2_two_channel/executors.py](../../src/trainers/stage2_two_channel/executors.py)
  - [src/config/schema.py](../../src/config/schema.py)

## What `K=4` Means Now

`K=4` in the current repo means total rollout views, not explorer count.

Operationally:

- `1` anchor rollout,
- `3` explorer rollouts.

This is the authored pseudo-positive profile in:

- [configs/stage2_two_channel/prod/ab_mixed_coco1024_bmajority_channel_b_pseudo_positive.yaml](../../configs/stage2_two_channel/prod/ab_mixed_coco1024_bmajority_channel_b_pseudo_positive.yaml)

with:

- `stage2_ab.channel_b.pseudo_positive.enabled: true`
- `stage2_ab.channel_b.pseudo_positive.coord_weight: 0.3`
- `stage2_ab.channel_b.triage_posterior.num_rollouts: 4`
- `explorer_temperature: 0.7`
- `explorer_top_p: 0.95`
- `explorer_top_k: 64`
- `unlabeled_consistent_iou_threshold: 0.9`
- `recovered_ground_truth_weight_multiplier: 3.0`

The schema makes this behavior explicit:

- when `pseudo_positive.enabled=false`, default Channel-B total views stay at `2`,
- when `pseudo_positive.enabled=true`, the default authored total view count becomes `4`.

This is the key difference from v3:

- v3 treated `K=2` as the intended default and warned against immediately hardening unlabeled-consistent clusters,
- v4 still keeps the promotion cautious, but it now has a real multi-view operational contract.

## The New Pseudo-Label Mechanism

The current repo uses the term `pseudo_positive`, not a broad unrestricted pseudo-label union.

That naming matters.

The mechanism is:

1. Start from anchor clean objects.
2. Look only at unmatched anchor clean objects as promotion candidates.
3. Measure support from explorer views.
4. Keep explorer support as evidence, not as an alternate teacher trajectory.
5. Promote only a subset of supported anchors.
6. Keep overlapping losers visible as shielded context rather than turning them into fully positive duplicates.

In the current implementation, the triage builder computes:

- `anchor_support_counts`
- `anchor_support_rates`
- `pseudo_positive_candidate_indices`
- `pseudo_positive_anchor_indices`
- `pseudo_positive_cluster_demoted_indices`
- `recovered_gt_support_counts`
- `valid_explorer_count`

See:

- [src/trainers/stage2_two_channel/target_builder.py](../../src/trainers/stage2_two_channel/target_builder.py)

The current implementation detail is stronger than the high-level docs:

- pseudo-positive candidates require meaningful multi-view support,
- then candidates are overlap-clustered,
- then each cluster chooses one winner,
- non-winning supported candidates are demoted back to shielded context.

This is the most important conceptual shift from older versions:

- v2 said "neutral unmatched is not duplicate-certified."
- v3 said "high-confidence unlabeled-consistent clusters may later become small soft positives."
- v4 says "some supported unmatched anchor objects now do become explicit partial positives, but only through a typed, anchor-centric, support-driven, cluster-aware mechanism."

## The Current Channel-B Supervision Taxonomy

The current clean-prefix teacher-forced forward distinguishes more cases than the older notes did.

### 1. Matched clean objects

These stay the classical positive clean-prefix subset.

They receive:

- positive geometry / coord supervision,
- positive structure supervision,
- no extra desc-positive widening beyond the current contract.

### 2. FN injections

Recovered or still-missed GT objects are appended into the clean `objects[]` container.

They receive:

- positive structure,
- positive desc CE,
- positive geometry / coord supervision,
- and possibly extra weight when they are recovered by explorer support.

### 3. Selected pseudo-positive anchors

These are retained unmatched anchor clean objects that survived support thresholding and overlap clustering.

They receive:

- global prefix structure CE,
- positive coord / bbox supervision with the configured pseudo-positive weight.

### 4. Support-positive shielded anchors

These are retained unmatched anchor objects with non-zero explorer support that were not selected as the cluster winner.

They remain prefix-visible context.

In the current implementation they may receive:

- global prefix structure CE,
- partial coord / bbox supervision scaled by support rate.

### 5. Cluster-demoted pseudo-positive candidates

These are supported candidates that lost overlap clustering.

They remain visible as context but are intentionally prevented from becoming another full positive duplicate.

They get:

- structure-only prefix supervision,
- no positive coord / bbox ownership.

### 6. Dead anchors

These are unmatched anchor objects with no sufficient support or policy-forced death.

They do not remain positive owners in the clean prefix.

This explicit taxonomy is what turns the old "pseudo-label consistency" idea into a reproducible contract.

## Other New Features Worth Calling Out

The current repo changes are not just pseudo-positive plus `K=4`.

There are several other important upgrades around that core idea.

### Typed Stage-2 config surfaces

The current config contract is explicit and fail-fast:

- `stage2_ab.channel_b.pseudo_positive`
- `stage2_ab.channel_b.triage_posterior`

Unknown keys fail fast.
Several older knobs are explicitly removed.

This is one of the biggest practical improvements over v1/v2/v3, which were still partly design-language-first.

Concrete handle:

- [src/config/schema.py](../../src/config/schema.py)

### Current objective surface is module-based

The active Stage-2 prod recipe declares objectives through:

- `token_ce`
- `loss_duplicate_burst_unlikelihood`
- `bbox_geo`
- `bbox_size_aux`
- `coord_reg`

inside `stage2_ab.pipeline.objective[]`.

That is a cleaner and more reproducible surface than older free-form descriptions.

### Stronger geometry continuation variants

The current production tree includes hardened variants:

- [ab_mixed_coco1024_bmajority_channel_b_pseudo_positive_hardened_spiky_coord.yaml](../../configs/stage2_two_channel/prod/ab_mixed_coco1024_bmajority_channel_b_pseudo_positive_hardened_spiky_coord.yaml)
- [ab_mixed_coco1024_bmajority_channel_b_pseudo_positive_hardened_spiky_coord-from_stage1.yaml](../../configs/stage2_two_channel/prod/ab_mixed_coco1024_bmajority_channel_b_pseudo_positive_hardened_spiky_coord-from_stage1.yaml)

These show that the current direction is not only about selection logic.
It is also actively tuning the geometry side of the positive supervision surface and explicitly supporting Stage-1-to-Stage-2 continuation.

### Better metric decomposition

The current Stage-2 metrics expose the pseudo-positive slices separately:

- `train/triage/unlabeled_consistent_count`
- `train/triage/pseudo_positive_candidate_count`
- `train/triage/pseudo_positive_selected_count`
- `train/triage/pseudo_positive_cluster_demoted_count`
- support-rate numerators / denominators
- `stage2/raw_rollouts`

This is much closer to a paper-ready diagnostic surface than the older note-only era.

## Comparison With Older Versions

### v1: EM-ish set supervision

Main idea:

- Channel-A iterative soft self-context,
- Channel-B rollout -> parse -> Hungarian -> FN append,
- FP-neutral handling because unmatched predictions may be unlabeled positives.

What v1 got right:

- keeping Channel-B FP-neutral,
- treating Stage-2 as rollout-aware rather than pure GT replay,
- not adding a separate detection head.

What v1 lacked relative to v4:

- no typed pseudo-positive config surface,
- no explicit `K=4` authored profile,
- no support-weighted shielded-anchor supervision taxonomy,
- no cluster-demoted bucket as a first-class contract.

### v2: clean-prefix duplicate correction

Main idea:

- generic unmatched accepted predictions stay neutral,
- duplicate-certified continuations get targeted negative supervision,
- positive CE should run under a clean deduplicated prefix.

What v2 got right:

- the clean-prefix formulation,
- the distinction between generic extras and self-collision duplicates,
- preserving room for unlabeled positives.

What v2 lacked relative to v4:

- it still treated "neutral unmatched" as the main non-GT category,
- it did not yet operationalize support-based partial promotion,
- it did not yet turn cross-rollout evidence into typed pseudo-positive supervision.

### v3: K=2 triage-posterior bridge

Main idea:

- `K=2` anchor/explorer rollouts,
- cross-rollout cluster-level triage,
- classify hypotheses as GT-backed, unlabeled-consistent, or dead,
- keep pseudo-label promotion cautious and mostly future-facing.

What v3 got right:

- anchor/explorer framing,
- support-style reasoning over clusters,
- explicit warning not to harden unlabeled-consistent clusters too early.

What v3 lacked relative to v4:

- `K=2` was still the intended default,
- pseudo-label promotion was still mostly conceptual,
- the supported implementation surface was not yet the current typed one,
- the final promoted-vs-shielded-vs-demoted split was not yet fully codified.

### v4: current repo direction

Main idea:

- keep Stage-1 strong and separate,
- keep Stage-2 clean-prefix and one-forward,
- keep anchor-first triage,
- make pseudo-labeling real but still cautious:
  - typed,
  - support-driven,
  - anchor-centric,
  - overlap-cluster-aware,
  - `K=4`,
  - metric-visible,
  - test-covered.

## Why This Is Better Than "Just Make Pseudo Labels Hard"

The current direction avoids two easy mistakes.

### Mistake 1: promote explorer-only objects directly

The current contract does not do this.

Promotion starts from unmatched anchor clean objects.
Explorer evidence is support evidence, not an independent positive teacher trajectory.

### Mistake 2: treat all supported unmatched objects as equally positive

The current contract does not do this either.

It now distinguishes:

- selected winners,
- retained supported shielded anchors,
- cluster-demoted losers.

That is the key safety valve against turning support gathering into another duplicate factory.

## Practical Handles

If someone wants to inspect the current v4 stack directly, the smallest useful set is:

- Stage-1:
  - [docs/training/STAGE1_OBJECTIVE.md](../../docs/training/STAGE1_OBJECTIVE.md)
  - [configs/stage1/sft_base.yaml](../../configs/stage1/sft_base.yaml)
  - [src/sft.py](../../src/sft.py)
- Stage-2:
  - [docs/training/STAGE2_RUNBOOK.md](../../docs/training/STAGE2_RUNBOOK.md)
  - [docs/training/METRICS.md](../../docs/training/METRICS.md)
  - [openspec/specs/stage2-ab-training/spec.md](../../openspec/specs/stage2-ab-training/spec.md)
  - [configs/stage2_two_channel/prod/ab_mixed_coco1024_bmajority_channel_b_pseudo_positive.yaml](../../configs/stage2_two_channel/prod/ab_mixed_coco1024_bmajority_channel_b_pseudo_positive.yaml)
  - [src/config/schema.py](../../src/config/schema.py)
  - [src/trainers/stage2_two_channel/target_builder.py](../../src/trainers/stage2_two_channel/target_builder.py)
  - [src/trainers/stage2_two_channel.py](../../src/trainers/stage2_two_channel.py)

## Verification

Recommended verification path for the current v4 direction:

- Stage-1 objective and runtime contract:
  - [tests/test_coord_softce_w1_loss.py](../../tests/test_coord_softce_w1_loss.py)
  - [tests/test_stage1_metric_key_parity.py](../../tests/test_stage1_metric_key_parity.py)
  - [tests/test_stage1_static_packing_runtime_config.py](../../tests/test_stage1_static_packing_runtime_config.py)
- Stage-2 config contract:
  - [tests/test_stage2_ab_config_contract.py](../../tests/test_stage2_ab_config_contract.py)
  - [tests/test_stage2_ab_profile_leaf_contract.py](../../tests/test_stage2_ab_profile_leaf_contract.py)
- Stage-2 runtime behavior:
  - [tests/test_stage2_ab_training.py](../../tests/test_stage2_ab_training.py)

The most important live metrics are:

- `stage2/raw_rollouts`
- `train/triage/unlabeled_consistent_count`
- `train/triage/pseudo_positive_candidate_count`
- `train/triage/pseudo_positive_selected_count`
- `train/triage/pseudo_positive_cluster_demoted_count`
- support-rate numerators / denominators

## One-Sentence Summary

Keep Stage-1 as the packed, teacher-forced language-of-boxes foundation; keep Stage-2 as the clean-prefix rollout-correction layer; and treat pseudo-labeling as a typed `K=4` anchor-centric pseudo-positive mechanism with selected winners, support-weighted shielded anchors, and cluster-demoted losers rather than a blanket hard pseudo-label rewrite.
