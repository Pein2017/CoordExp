## Why

The canonical Stage-2 Channel-B contract is intentionally conservative.

- `matched_clean` remains the only clean-prefix positive object subset.
- `recovered GT` stays on the existing FN-injection path with higher weight.
- unmatched anchor objects are either kept neutral as `shielded_anchor` or dropped as `dead_anchor`.
- dead-anchor suppression is currently tied to duplicate-like alternate continuations.

That conservatism protects the model from noisy supervision, but it can also harm learning in dense scenes and incomplete-GT scenes. Some unmatched rollout objects appear to correspond to real visible objects that simply are not labeled in GT. Blanket neutral treatment for those trusted unmatched objects can teach the model to stay silent on real objects.

The first implementation-ready goal is still simple:

- spend extra rollout compute on better consistency evidence,
- replace harmful blanket neutral handling only for a trusted unmatched subset,
- keep the final teacher-forced target clean,
- and avoid shortcut collapse such as enumeration bursts, duplicate bursts, or oversized entangled boxes.

## What Changes

- Add an explicit opt-in config surface under `stage2_ab.channel_b.pseudo_positive`:
  - `enabled: false` by default,
  - `coord_weight: 0.5` as the first implementation-ready default when enabled.
- Preserve legacy canonical `K=2` behavior when `stage2_ab.channel_b.pseudo_positive.enabled=false`.
- When `stage2_ab.channel_b.pseudo_positive.enabled=true`, support arbitrary `stage2_ab.channel_b.triage_posterior.num_rollouts >= 2` total rollout views:
  - `1` deterministic anchor rollout
  - `num_rollouts - 1` stochastic explorer rollouts using the same decode profile with different seeds
  - `num_rollouts=4` is the default pseudo-positive implementation profile for this change, but the contract must stay valid for future `best-K` ablations
- Keep the final teacher-forced target anchored on the edited anchor clean sequence rather than rebuilding a union ordering.
- Replace blanket unmatched-neutral handling only for trusted unmatched anchor objects that have enough explorer support rate.
- Keep v1 geometry-first:
  - use explorer support ratios from cross-rollout bbox association,
  - define GT conflict using the existing unlabeled-consistency IoU threshold against GT-backed anchor objects,
  - do not require semantic-desc agreement for pseudo-positive selection in v1,
  - do not add pseudo-positive desc CE in v1.
- Keep rollout support reproducible:
  - malformed anchor preparation drops the sample instead of using the empty-prefix fallback,
  - each explorer view must complete the standard accepted-clean preparation path,
  - empty explorer accepted-clean output still counts as a valid zero-support view,
  - partial per-view failure must raise and abort the current optimizer step rather than silently shrinking the support denominator.
- Keep recovered-GT semantics narrow under arbitrary `K`:
  - any explorer GT hit is sufficient to recover a missed GT object,
  - multiple explorer GT hits collapse to one FN-injection object,
  - `recovered_ground_truth_weight_multiplier` applies once per recovered GT object in v1,
  - recovered-GT explorer support counts and support rates are recorded in per-sample triage metadata using `recovered_support_rate = recovered_support_count / valid_explorer_count`.
- Add a conservative anchor-side overlap guard before promotion:
  - pseudo-positive candidate overlap clusters are connected components of the undirected anchor graph with edges at `IoU >= duplicate_iou_threshold`,
  - at most one pseudo-positive object may be selected from each such cluster,
  - ties resolve by higher explorer support rate, then earlier anchor order,
  - non-winning cluster members fall back to `shielded_anchor`.
- Keep negative-side behavior simple:
  - dead anchors stay out of the final target,
  - only duplicate-like dead branches receive explicit negative suppression,
  - duplicate-like dead means same normalized desc plus high-IoU overlap to an earlier kept anchor object within the same local continuation boundary group,
  - no full-object negative CE is introduced for dead anchors in v1.
- Reuse the existing decoded-box harness:
  - `bbox_geo`
  - `coord_reg`
  - optional light `bbox_size_aux`
- Pin pseudo-positive target geometry to the selected anchor object's own canonical coordinates.
- Keep `token_ce` narrow in v1:
  - matched-prefix structure CE remains matched-only,
  - FN-injection desc CE remains FN-only,
  - pseudo-positive objects do not create new desc CE or matched-prefix structure CE targets.
- Add explicit observability for auditability:
  - per-sample Channel-B triage metadata with `valid_explorer_count`,
  - per-anchor explorer support counts and support rates,
  - pseudo-positive selected anchor indices,
  - per-explorer `dead_explorer_indices_by_view`,
  - recovered-GT support counts and support rates,
  - exact additive metrics in the existing `train/triage/*` namespace:
    - `train/triage/pseudo_positive_candidate_count`
    - `train/triage/pseudo_positive_subthreshold_count`
    - `train/triage/pseudo_positive_selected_count`
    - `train/triage/pseudo_positive_cluster_demoted_count`
    - `train/triage/anchor_preparation_dropped_count`
    - `train/triage/pseudo_positive_support_rate_num`
    - `train/triage/pseudo_positive_support_rate_den`
    - `train/triage/pseudo_positive_selected_support_rate_num`
    - `train/triage/pseudo_positive_selected_support_rate_den`
    - `train/triage/recovered_ground_truth_rate_num`
    - `train/triage/recovered_ground_truth_rate_den`
  - compatibility semantics:
    - `train/triage/unlabeled_consistent_count` stays as the total shielded-anchor count
    - legacy `rollout/explorer/*` metrics become mean-over-valid-explorer-view summaries under arbitrary `K`
    - `stage2/raw_rollouts` continues to count total rollout trajectories across anchor plus all explorers
    - explorer-preparation aborts are reported through failure telemetry / ablation reporting, not a normal finalized-step metric
- Strip version suffixes from YAML knobs and code-facing variable names:
  - keep `v1` only in change/capability naming,
  - use `stage2_ab.channel_b.pseudo_positive.*` in config and code-facing terminology.

## Recommended First Version

The recommended implementation-ready first version is:

- keep canonical legacy Channel-B unchanged when `stage2_ab.channel_b.pseudo_positive.enabled=false` and activate the experiment only with `stage2_ab.channel_b.pseudo_positive.enabled=true`,
- support arbitrary `K` when enabled, while using `K=4` as the default pseudo-positive implementation profile:
  - `1` greedy anchor
  - `K-1` explorers with the same moderate decode profile and different seeds
- evaluate pseudo-positive candidates only on the anchor side
- associate each explorer to the anchor view with the existing one-to-one IoU rule
- compute both `explorer_support_count` and `explorer_support_rate` for each unmatched anchor object
- keep the per-explorer geometry gate strict
- require both `explorer_support_count >= 2` and `explorer_support_rate >= 2/3` for pseudo-positive promotion so `K=2` remains a no-promotion control instead of a one-hit promotion regime
- do not add semantic gating for v1

The recommended v1 selection rule is:

- support count `0` -> `dead_anchor`
- support count `> 0` but failing either threshold -> `shielded_anchor` only, kept as context but not positively supervised
- support count `>= 2` and support rate `>= 2/3` -> pseudo-positive candidate subset
- within duplicate-like anchor overlap clusters, keep only one selected pseudo-positive candidate:
  - highest explorer support rate wins,
  - ties break by earlier anchor order,
  - remaining candidates fall back to `shielded_anchor`

The recommended v1 supervision rule is:

- `matched_clean` -> existing positive coord supervision + matched-prefix structure CE
- `fn_injection` -> existing positive coord supervision + desc CE
- `pseudo_positive` -> positive coord supervision only, at `coord_weight=0.5`
- `coord_weight` operates in the existing per-bbox-group weight space and scales only `bbox_geo`, `coord_reg`, and `bbox_size_aux` contributions for pseudo-positive objects
- pseudo-positive bbox/coord supervision targets come from the selected anchor object's own canonical coordinates
- `dead_anchor` -> excluded from the final target; only duplicate-like dead branches get explicit branch-entry suppression

The recommended dead-anchor rule is:

- all dead anchors are dropped from the final teacher-forced target,
- dead anchors do not receive positive supervision,
- dead anchors do not receive full-object negative CE,
- only duplicate-like dead alternate continuations produce explicit suppression targets at the first divergent branch token.

## Assumptions

- Hallucination is currently rare enough under conservative SFT / early Stage-2 behavior that a trusted unmatched subset can be positively supervised without immediate collapse.
- Extra rollout compute is more valuable when spent on repeated consistency evidence than on broader semantic gating in v1.
- Geometry repeatability is a more reliable first-stage trust signal than desc consistency for this task.
- A `support_count >= 2` plus `support_rate >= 2/3` rule is conservative enough to replace blanket neutral handling without becoming "all unmatched become positives."
- The enabled arbitrary-`K` path may fail fast on malformed/missing explorer views rather than trying to reinterpret support rates over a smaller denominator.

## Non-Blocking Follow-Ups

- Whether `support_rate = 1.0` should later receive a slightly larger coord weight than lower but still qualifying pseudo-positive support rates
- Whether the shield-only bucket should later split into finer sub-cases such as single-hit support versus multi-hit-but-below-rate support
- Whether duplicate-like dead suppression should later extend beyond same-desc duplicate-style branches
- Whether the current explorer decode profile is still optimal once `K` becomes a tuned ablation dimension

## Risks To Validity

- Even with `support_count >= 2` and `support_rate >= 2/3`, some pseudo-positive objects may still be unlabeled noise rather than real objects.
- If the explorer decode is too aggressive, support ratios may become noisy and less meaningful.
- If the pseudo-positive threshold is too conservative, recall gains may be too small to measure.
- If per-anchor overlap clustering is too aggressive, some legitimate adjacent same-class objects may be demoted back to shield-only.
- If duplicate-like dead suppression is too weak, duplicate bursts may remain.
- If duplicate-like dead suppression is too strong, recall may regress in crowded scenes.

## Required Evidence

- Manual audit of anchor unmatched objects bucketed by explorer support evidence:
  - `support_count = 0`
  - `support_count > 0` but below promotion thresholds
  - `support_count >= 2` and `support_rate >= 2/3`
- Short-horizon training comparison against the current blanket-neutral baseline on:
  - dense-scene recall
  - hallucination rate
  - duplicate burst rate
  - enumeration-style overproduction
  - oversized / entangled box frequency
- Evidence that `support_count >= 2` and `support_rate >= 2/3` is a better first pseudo-positive threshold than broader lower-rate or single-hit promotion
- Evidence that duplicate-like dead branch suppression improves behavior over drop-only dead handling before expanding any broader negative-side design
- Evidence that support-rate observability remains comparable across at least two different `num_rollouts` settings in a `best-K` ablation sweep
- Evidence that recovered-GT rate metrics and explorer-abort metrics are reported alongside pseudo-positive metrics so `K`-dependent confounds remain visible

## Capabilities

### New Capabilities

- `channel-b-lightweight-pseudopositive-v1`: an opt-in arbitrary-`K` training-time Channel-B path that replaces blanket neutral treatment on a trusted unmatched anchor subset with lower-weight coord supervision while keeping dead anchors out of the final target.

### Modified Capabilities

- `stage2-ab-training`: modify the strict `stage2_ab.channel_b` allowlist, the canonical `K=2` Channel-B contract, the grouped rollout knobs, the invalid-rollout fallback behavior, and the matched-only `bbox_size_aux` wording so the opt-in `stage2_ab.channel_b.pseudo_positive` contract is unambiguous when enabled.
- `teacher-forcing-unified-loss-registry`: modify rollout-context semantics so selected pseudo-positive anchors are exempt from blanket FP-neutral handling only for coord-side supervision, while shielded anchors remain neutral.
- `trainer-metrics-components`: extend canonical Channel-B triage metrics with exact pseudo-positive candidate, selection, demotion, anchor-drop, and support-rate numerator/denominator keys.

## Impact

- Immediate impact is still proposal/spec only.
- The expected first implementation surface is intentionally narrow and likely centered on:
  - `src/config/schema.py`
  - `src/trainers/stage2_two_channel.py`
  - `src/trainers/stage2_two_channel/target_builder.py`
  - `src/trainers/teacher_forcing/modules/loss_dead_anchor_suppression.py`
- The first implementation also needs targeted test updates because current config and trainer assumptions are still hard-wired to canonical `K=2` behavior.
- This first version intentionally avoids:
  - semantic-desc gating for pseudo-positive selection
  - pseudo-positive desc CE
  - pseudo-positive matched-prefix structure CE
  - full-object negative CE for dead anchors
  - blanket promotion of all unmatched objects
