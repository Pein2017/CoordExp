## Context

Today the Channel-B data flow is:

`rollout tokens -> bounded parse/salvage -> bbox-valid filtering -> sequential dedup -> Hungarian match + triage -> clean-prefix edit -> duplicate first-divergence target build -> one teacher-forced forward -> duplicate unlikelihood`

The narrow point is the pre-triage duplicate discovery step. Channel-B currently treats duplicates as sequential same-description boxes that overlap an earlier kept object above `duplicate_iou_threshold`, then synthesizes first-divergence bad-token targets from those sequential duplicate continuations. That keeps the contract simple, but it under-targets the heavy duplicate-collapse pattern we just observed offline: large local same-description clusters near the prediction ceiling that crowd out other objects. The offline post-op experiment in [comparison.json](/data/home/xiaoyan/AIteam/data/CoordExp/temp/dup_postop_eval/comparison.json) showed that removing those clusters improved full-val `coco_real bbox_AP` from `0.3899` to `0.3979`, which is enough evidence to strengthen the training signal.

Constraints:

- Keep the canonical Channel-B one-forward contract and anchor-rooted edited target.
- Keep `loss_duplicate_burst_unlikelihood` as the canonical B-only suppression module.
- Keep `loss_duplicate_burst_unlikelihood.config` empty in v1 unless we deliberately open a new surface later.
- Preserve geometry invariants and deterministic behavior.
- Avoid new CLI flags and prefer repo-owned helpers over ad hoc experiment code.

### Empirical Background

The design is motivated by a concrete detector experiment that has already been run on saved inference artifacts rather than by a purely speculative heuristic.

Observed failure pattern:

- Worst-valid full-val scenes showed `duplication collapse`, not just ordinary crowding.
- Several images were saturated at `127-128` predictions while still missing `30+` ground-truth objects.
- Visual review in [contact_sheet.png](/data/home/xiaoyan/AIteam/data/CoordExp/temp/poor_recall_vis/review/contact_sheet.png) and the per-image renders under [temp/poor_recall_vis/review](/data/home/xiaoyan/AIteam/data/CoordExp/temp/poor_recall_vis/review) showed dense same-description bursts like repeated `book`, `cup`, `boat`, `train`, and `broccoli` boxes.

Offline detector shape:

- The temporary detector in [dup_postop_eval.py](/data/home/xiaoyan/AIteam/data/CoordExp/temp/dup_postop_eval.py) built same-description local duplicate clusters using:
  - high overlap,
  - center-distance proximity,
  - image-level gates such as prediction saturation and dominant-description ratio.
- It kept one anchor prediction per tight cluster and stripped lower-value local repeats.
- It did not use rollout explorer support because it operated purely on saved inference predictions.

Measured effect on the full-val run:

- `coco_real bbox_AP`: `0.389928 -> 0.397880` (`+0.007952`)
- `bbox_AP50`: `0.537521 -> 0.549229`
- `bbox_AP75`: `0.417916 -> 0.426416`
- `bbox_AR100`: `0.529065 -> 0.525611`
- `F1-ish@0.50`: `0.597773 -> 0.629771`
- predictions before/after stripping from [dedup_summary.json](/data/home/xiaoyan/AIteam/data/CoordExp/temp/dup_postop_eval/dedup_summary.json):
  - `45643 -> 39638`
  - `6005` stripped predictions across `654` records

Important interpretation:

- The offline detector improved precision enough to raise AP, but it also increased FN modestly, which is why `AR100` dropped slightly.
- Because it runs after generation, it cannot recover objects the model never emitted.
- That is the main reason to reuse the detector idea in Channel-B training rather than stopping at post-op filtering.

## Goals / Non-Goals

**Goals:**

- Detect duplicate-like Channel-B rollout objects using a cluster-aware relation that is stronger than sequential high-IoU matching.
- Reuse that detector to generate better duplicate-unlikelihood targets without changing the Stage-2 objective list or requiring a second teacher-forced forward.
- Distinguish heavy duplication collapse from real crowded scenes using deterministic spatial and rollout-support cues.
- Emit enough diagnostics to verify activation, coverage, and failure modes in smoke runs and short ablations.

**Non-Goals:**

- Introduce a new objective module name or replace `loss_duplicate_burst_unlikelihood` with a different pipeline surface.
- Depend on post-hoc confidence scoring during training.
- Recover objects that were never emitted by the rollout.
- Add a user-facing decoding knob or a new authored config tree in this change.

## Decisions

### 1. Move the change to duplicate-target construction, not the unlikelihood module

We will keep [loss_duplicate_burst_unlikelihood.py](/data/home/xiaoyan/AIteam/data/CoordExp/src/trainers/teacher_forcing/modules/loss_duplicate_burst_unlikelihood.py) intact as the canonical B-only consumer of `(boundary, rel_pos, token_id)` targets. The change will happen upstream in the Channel-B rollout preparation path, primarily in [rollout_views.py](/data/home/xiaoyan/AIteam/data/CoordExp/src/trainers/stage2_two_channel/rollout_views.py) and [target_builder.py](/data/home/xiaoyan/AIteam/data/CoordExp/src/trainers/stage2_two_channel/target_builder.py).

Why:

- The current module contract is already codified in schema and OpenSpec.
- Preserving the module interface avoids a broader pipeline migration.
- The detector is naturally object-level, while the module is just a token-level consumer of prepared metadata.

Alternative considered:

- Replace the module with a new cluster-aware suppression objective.
  Rejected for this change because it would expand the public objective surface and require a larger contract migration than the behavior gain justifies.

### 2. Define duplicates as same-description duplicate-like clusters, not sequential pair hits

We will construct an undirected duplicate-like graph on parsed bbox objects before triage. Two objects belong to the same duplicate-like component when they share normalized description and are either:

- above a deterministic IoU threshold, or
- within a deterministic center-distance radius scaled by object size

This reuses the proven relation from [small_object_duplication_study.py](/data/home/xiaoyan/AIteam/data/CoordExp/src/analysis/small_object_duplication_study.py) rather than inventing a new duplicate predicate.

Within each connected component:

- one deterministic survivor is kept on the clean anchor surface,
- the remaining members become duplicate-candidate continuations for unlikelihood target generation.

Why:

- The sequential heuristic misses non-adjacent and bursty local repeats.
- Connected components are a better match for the collapse pattern seen in dense scenes.
- A deterministic survivor keeps the clean-prefix edit contract stable.
- The offline detector already demonstrated that this style of grouping can improve benchmark AP on the saved full-val predictions.

Alternative considered:

- Plain NMS-style suppression with no description matching.
  Rejected because it confuses real crowded objects from different classes with duplicate collapse.

### 3. Use crowd-safety guards before turning duplicate candidates into UL targets

Not every same-description cluster should create suppression. The target builder will only materialize duplicate-unlikelihood targets for components that are sufficiently collapse-like. The initial v1 guards are:

- clusters must be spatially tight, not just same-description,
- spread-out same-description rows or grids should not be treated as local duplicate collapse,
- explorer-supported same-description objects should bias toward “real crowded object” rather than “duplicate collapse,”
- the logic must remain deterministic and based only on rollout-visible state.

Why:

- The user’s concern is valid: crowded scenes with many `chair`, `book`, or `apple` objects must not be over-penalized.
- Channel-B already has explorer views, which provide extra evidence not available in offline post-op scoring.
- This is the main way the training version can improve on the temporary post-op detector rather than merely copying it.

Alternative considered:

- Penalize every non-survivor in every same-description component.
  Rejected because it would over-suppress real crowded scenes and change recall behavior too aggressively.

### 4. Preserve first-divergence token semantics, but broaden the metadata meaning

The metadata passed into duplicate unlikelihood will remain first-divergence targets projected onto the post-triage clean prefix. What changes is the meaning of the source duplicates:

- old: sequential duplicate-burst continuations
- new: cluster-aware duplicate-like continuations

Why:

- This preserves the single-forward training path.
- It keeps schema and module plumbing stable.
- It makes the minimal contract change needed to upgrade behavior.

Alternative considered:

- Emit multiple bad tokens or weighted targets per duplicate component.
  Deferred. That is promising, but it changes the objective payload shape and should be evaluated after the cluster-aware v1 target builder lands.

### 5. Add cluster-aware diagnostics to the canonical metrics surface

In addition to the existing duplicate gauges and counters, we will emit a small set of cluster-aware diagnostics, likely including:

- `dup/duplicate_like_max_cluster_size`
- `dup/desc_entropy`
- `stage2_ab/channel_b/dup/N_duplicate_like_clusters`

Why:

- We need direct evidence that the new detector is active on collapse-heavy scenes.
- These metrics help separate “many objects” from “many local duplicate clusters.”

Alternative considered:

- Reuse only the existing `max_desc_count` and `near_iou90_pairs_*` metrics.
  Rejected because they are informative but not sufficient to distinguish collapse shape from legitimate crowding.

## Risks / Trade-offs

- [Risk] Real crowded scenes may still be partially over-suppressed. → Mitigation: keep v1 guards conservative, test on dense-scene smokes, and validate duplicate metrics together with recall-oriented eval.
- [Risk] The detector may be too weak and fail to expand target coverage meaningfully. → Mitigation: compare target counts and cluster metrics against the current sequential path on the same smoke runs.
- [Risk] Reusing analysis logic directly could import non-training-safe assumptions or data formats. → Mitigation: port only the minimal geometric relation into repo-owned Stage-2 helpers instead of importing experiment scripts wholesale.
- [Risk] Different duplicate-target semantics can change training trajectories enough to complicate baseline comparisons. → Mitigation: keep the config surface unchanged and call out the change explicitly in run provenance and OpenSpec.

## Migration Plan

1. Land the spec and implementation behind the existing canonical Stage-2 path with no CLI changes.
2. Update Channel-B rollout preparation to emit cluster-aware duplicate metadata and diagnostics.
3. Add smoke tests and deterministic fixture coverage for:
   - tight duplicate collapse
   - spread-out real crowded same-description objects
   - explorer-supported same-description objects
4. Run a short Stage-2 smoke and compare duplicate-target counts and cluster diagnostics against the current baseline.
5. Run a short inference/eval validation to confirm duplicate-collapse metrics improve and no geometry contract regresses.

Rollback is straightforward because the implementation seam is local: restore the old sequential duplicate-target generator while leaving the loss module and config surface untouched.

## Open Questions

- Should explorer support act only as a crowd-safety exemption, or also help choose the cluster survivor?
- Do we want the first landed version to emit new canonical metrics immediately, or should those metrics be added in the same patch only if the implementation remains small?

## Handoff Notes

Artifacts and observations worth preserving for the next implementer:

- Full-val baseline eval:
  - [eval_coco_real/metrics.json](/data/home/xiaoyan/AIteam/data/CoordExp/output/infer/coco1024_val200_lvis_proxy_merged_2b/eval_coco_real/metrics.json)
- Offline duplicate-strip experiment:
  - [dup_postop_eval.py](/data/home/xiaoyan/AIteam/data/CoordExp/temp/dup_postop_eval.py)
  - [comparison.json](/data/home/xiaoyan/AIteam/data/CoordExp/temp/dup_postop_eval/comparison.json)
  - [dedup_summary.json](/data/home/xiaoyan/AIteam/data/CoordExp/temp/dup_postop_eval/dedup_summary.json)
- Worst-scene visualization pack:
  - [poorest_fn_summary.json](/data/home/xiaoyan/AIteam/data/CoordExp/temp/poor_recall_vis/poorest_fn_summary.json)
  - [contact_sheet.png](/data/home/xiaoyan/AIteam/data/CoordExp/temp/poor_recall_vis/review/contact_sheet.png)

Key handoff conclusion:

- The detector idea is already empirically justified.
- The likely highest-leverage implementation is not to replace the unlikelihood module, but to replace the duplicate-target generator feeding it.
