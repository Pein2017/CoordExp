## Context

The canonical Stage-2 Channel-B contract in main is intentionally conservative:

- one deterministic anchor rollout plus one stochastic explorer rollout,
- one edited anchor target,
- `matched_clean` remains the only clean-prefix positive object subset,
- recovered GT stays on the FN-injection path,
- `shielded_anchor` remains neutral context,
- and only duplicate-like dead continuations receive local negative suppression.

That baseline is a good default, but it also creates the exact learning concern behind this change:

> in dense or incompletely labeled scenes, some anchor-side unmatched objects are likely real objects, yet the current blanket neutral handling teaches the model neither to keep them nor to localize them more confidently.

The user’s desired v1 remains intentionally narrow:

- do not redesign the whole Channel-B trainer,
- do not broaden semantic-desc supervision,
- do not create a second teacher-forced branch,
- do not globally penalize all unmatched objects,
- and do not invent a new standalone oversize-negative subsystem.

Instead, use more rollout evidence to conservatively identify a trusted unmatched subset and apply only lighter coord-side positive supervision there, while keeping the existing one-forward edited-anchor training shape.

The main implementation authority remains:

- `src/config/schema.py`
- `src/trainers/stage2_two_channel.py`
- `src/trainers/stage2_two_channel/target_builder.py`
- `src/trainers/stage2_two_channel/rollout_views.py`
- `src/trainers/rollout_matching/matching.py`
- `src/trainers/teacher_forcing/modules/token_ce.py`
- `src/trainers/teacher_forcing/modules/bbox_geo.py`
- `src/trainers/teacher_forcing/modules/bbox_size_aux.py`
- `src/trainers/teacher_forcing/modules/coord_reg.py`
- `src/trainers/teacher_forcing/modules/loss_dead_anchor_suppression.py`

## Goals / Non-Goals

**Goals**

- Keep legacy canonical Channel-B available unchanged when disabled and introduce this behavior only through an explicit opt-in config surface.
- Use `K=4` as the default pseudo-positive rollout profile for implementation planning and authored YAML examples.
- Support arbitrary total rollout count `K = stage2_ab.channel_b.triage_posterior.num_rollouts` when enabled:
  - `1` deterministic anchor
  - `K-1` stochastic explorers
- Preserve the single edited-anchor target and single teacher-forced forward contract.
- Promote only a conservative subset of unmatched anchor objects to pseudo-positive status.
- Express pseudo-positive voting in support-rate terms so the contract scales cleanly across future `best-K` ablations, while still requiring a minimum absolute evidence floor.
- Keep pseudo-positive supervision coord-only:
  - `bbox_geo`
  - `coord_reg`
  - optional `bbox_size_aux`
- Preserve matched-prefix structure CE as matched-only and FN desc CE as FN-only.
- Keep dead-anchor suppression narrow and boundary-local.
- Make the experiment auditable through per-sample triage metadata and aggregate rate-ready metrics.

**Non-Goals**

- No semantic-desc gating for v1 pseudo-positive selection.
- No pseudo-positive desc CE.
- No pseudo-positive matched-prefix structure CE.
- No full-object negative CE for dead anchors.
- No explorer-only pseudo-positive path in v1.
- No removal of the legacy `K=2` compatibility / control contract.
- No new CLI flags; this remains YAML/config-first.

## Decisions

### 1) Make pseudo-positive v1 opt-in, default the enabled profile to K=4, and retain legacy K=2 compatibility

This change adds a typed config block under `stage2_ab.channel_b.pseudo_positive`:

- `enabled`
- `coord_weight`

Legacy disabled behavior remains the canonical `K=2` Channel-B contract. The new behavior activates only when:

- `stage2_ab.channel_b.pseudo_positive.enabled=true`

When enabled, total rollout count is determined by:

- `stage2_ab.channel_b.triage_posterior.num_rollouts`

The contract accepts any integer `>= 2`, with `4` as the default pseudo-positive implementation profile for this change.

Why:

- this keeps the current Stage-2 contract stable for all existing configs,
- makes the experimental scope explicit,
- and leaves room for `best-K` ablations without another schema redesign.

### 2) Spend extra compute on repeated explorer evidence, not on a second teacher trajectory

The enabled runtime uses:

- `1` anchor rollout for ordering and target editing,
- `K-1` explorers for evidence only.

The explorers all share one authored decode profile and differ only by deterministic stochastic identity derived from:

- the sample rollout seed base,
- the explorer ordinal.

The final positive target still comes only from the edited anchor sequence.

Why:

- repeated explorers buy us a support-rate estimate instead of a binary anchor/explorer agreement test,
- while preserving the one-forward teacher-forced training contract.

Rejected alternatives:

- `K` fully exploratory rollouts with no anchor,
- a union ordering over anchor and explorer objects,
- a second teacher-forced explorer pass.

### 3) Trusted unmatched selection stays geometry-first and anchor-centric

Pseudo-positive candidates come only from unmatched anchor clean objects.

For each explorer view:

- associate explorer objects to anchor objects using the existing deterministic one-to-one max-IoU rule,
- apply the existing `unlabeled_consistent_iou_threshold`,
- only count support when the explorer counterpart is also unmatched,
- and reject support when the anchor object conflicts with a GT-backed anchor object.

Conflict remains geometry-first:

- an unmatched anchor conflicts with a GT-backed anchor when IoU to any GT-backed anchor object is at least `unlabeled_consistent_iou_threshold`.

For each unmatched anchor object, compute:

- `support_count`
- `support_rate = support_count / valid_explorer_count`

Why:

- it is the lowest-risk extension of the current contract,
- it reuses existing rollout-matching machinery,
- and it scales cleanly as `K` changes.

### 4) Recovered GT remains any-hit collapsed FN injection

Under enabled pseudo-positive behavior, recovered GT is defined conservatively:

- anchor misses GT,
- at least one explorer matches that GT,
- multiple explorer hits collapse to one recovered FN object,
- `recovered_ground_truth_weight_multiplier` is applied once.

Recovered GT support count and `recovered_support_rate = recovered_support_count / valid_explorer_count` are recorded in per-sample triage metadata for auditability.

Why:

- this preserves continuity with the current recovered-FN contract,
- and avoids inventing a second weighting or aggregation scheme for v1.

### 5) Bucket unmatched anchor objects by support rate, then apply overlap clustering

Bucket assignment before overlap clustering:

- support count `0` -> `dead_anchor`
- support count `> 0` but failing the promotion rule -> `shielded_anchor`
- pseudo-positive promotion rule:
  - `support_count >= 2`
  - `support_rate >= 2/3`

Pseudo-positive candidates are then clustered using connected components of the undirected anchor-side graph with edges:

- `IoU >= duplicate_iou_threshold`

Within each component:

- keep exactly one pseudo-positive winner,
- choose the highest support rate,
- break ties by earlier anchor order,
- demote the remaining component members back to `shielded_anchor`.

Why:

- this prevents multiple near-identical unmatched anchors from all being promoted,
- keeps `K=2` as a no-promotion control instead of a one-hit auto-promotion regime,
- and still keeps non-promoted but partially supported anchors visible as context rather than converting them all to dead objects.

### 6) Pseudo-positive supervision is limited to existing coord-side group weights

Selected pseudo-positive anchors:

- remain in the final edited anchor prefix,
- reuse the current bbox/coord group path,
- share one per-group weight `coord_weight`,
- use the selected anchor object's own canonical coordinates as the bbox/coord target source,
- and scale only:
  - `bbox_geo`
  - `coord_reg`
  - `bbox_size_aux`

They do not create:

- desc CE,
- matched-prefix structure CE,
- new flat objective-module weights.

Why:

- this is the smallest practical positive-side extension,
- it maps directly onto the existing `Stage2BBoxGroup.weight` style carrier,
- and it avoids turning uncertain objects into full text-supervised positives.

### 7) Dead anchors are dropped from the final target and only duplicate-like dead branches get explicit suppression

All dead anchors are excluded from the final edited target.

Only duplicate-like dead anchors are eligible for explicit suppression. In v1 that means:

- the dead anchor belongs to the same local continuation boundary group as an earlier kept anchor object,
- it overlaps that earlier kept anchor at `IoU >= duplicate_iou_threshold`,
- and it shares the same normalized description under the existing duplicate-style normalization rule.

Those duplicate-like dead anchors may generate first-divergent bad-token suppression targets for the existing:

- `loss_dead_anchor_suppression`

module.

Non-duplicate dead anchors are simply dropped.

Why:

- this preserves the clean target,
- keeps the local negative mechanism narrow,
- and avoids punishing all unmatched objects as if GT were complete.

### 8) Explorer-view failure is a hard reproducibility error

If an explorer rollout fails before accepted-clean preparation completes, the trainer must:

- raise,
- and abort the current optimizer step.

It must not:

- silently drop the sample,
- reinterpret support rates over fewer explorers,
- or retry with a different denominator.

Zero accepted-clean explorer output is still valid and counts as zero support.

If the anchor rollout fails before accepted-clean preparation completes under the enabled pseudo-positive contract:

- the sample is dropped from Channel-B training for that step,
- the canonical empty-prefix fallback is not used,
- and the drop is counted explicitly in triage metrics.

Why:

- support-rate semantics are only meaningful if the denominator stays fixed,
- and silent explorer shrinkage would make the experiment hard to reproduce.

### 9) Observability must be rate-ready and K-comparable

The canonical audit carrier for this change is per-sample Channel-B triage metadata, not optional monitor dumps.

Required fields include:

- `valid_explorer_count`
- per-anchor explorer support counts
- per-anchor explorer support rates
- `pseudo_positive_anchor_indices`
- `dead_explorer_indices_by_view`
- per-recovered-GT explorer support counts
- per-recovered-GT explorer support rates
- mean-over-valid-explorer-view semantics for legacy singular explorer metrics such as `rollout/explorer/*`

Aggregate metrics should also expose:

- pseudo-positive candidate count
- pseudo-positive sub-threshold count
- pseudo-positive selected count
- pseudo-positive cluster-demoted count
- anchor-preparation dropped count
- pseudo-positive support-rate numerators / denominators
- selected-pseudo-positive support-rate numerators / denominators
- recovered-GT rate numerators / denominators

The canonical aggregate namespace for the new metrics stays in the existing `train/triage/*` family:

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

Why:

- monitor payloads are optional,
- per-sample metadata must be present whenever the feature is enabled,
- numerator / denominator metrics stay comparable across different `K` settings,
- `train/triage/unlabeled_consistent_count` can remain backward-compatible as the total shielded-anchor count,
- and explorer-preparation aborts need failure-telemetry reporting rather than ordinary finalized step metrics because the step fails fast.

## Data Flow

### Enabled pseudo-positive v1 per-sample flow

`raw sample`
`-> anchor rollout`
`-> explorer rollout #1 ... explorer rollout #(K-1)`
`-> per-view accepted-clean preparation`
`-> anchor/explorer association per explorer`
`-> support counting + support-rate projection + recovered-GT detection`
`-> support-rate buckets`
`-> anchor overlap clustering for pseudo-positive candidates`
`-> final edited anchor target`
`-> one teacher-forced forward`
`-> bucketed positive/negative loss projection`

### Output buckets after triage

- `matched_clean`
  - positive coord supervision
  - matched-prefix structure CE
- `fn_injection`
  - positive coord supervision
  - FN desc CE
- `pseudo_positive`
  - positive coord supervision only
- `shielded_anchor`
  - context only
- `dead_anchor`
  - removed from target
  - duplicate-like subset may create dead-branch suppression targets

## Implementation Notes

### Expected code surfaces

- `src/config/schema.py`
  - add typed `pseudo_positive`
  - require `num_rollouts >= 2` when enabled
- `src/trainers/stage2_two_channel.py`
  - schedule `1 + (K-1)` rollout generation
  - ensure deterministic per-explorer identities
  - drop malformed anchor-preparation samples
  - fail fast on malformed explorer prep
  - preserve and document mean-over-valid-explorer-view semantics for legacy singular explorer metrics
- `src/trainers/stage2_two_channel/target_builder.py`
  - aggregate support counts across `K-1` explorers
  - project support rates from the fixed denominator
  - collapse recovered GT any-hit semantics
  - apply support-rate buckets with a `support_count >= 2` floor for promotion
  - connected-component overlap clustering
  - pseudo-positive bbox-group weighting
  - pseudo-positive anchor-coordinate target source
  - boundary-local duplicate-like dead suppression filtering
- `src/trainers/teacher_forcing/modules/loss_dead_anchor_suppression.py`
  - preserve current first-divergence mechanism
  - consume only filtered duplicate-like dead targets
- `src/trainers/stage2_two_channel/types.py`
  - extend triage metadata for support counts, support rates, valid explorer count, selected pseudo-positive indices, and `dead_explorer_indices_by_view`

### Risky edges to keep explicit during implementation

- current schema hard-rejects `num_rollouts != 2`, so the opt-in condition must be generalized carefully
- current trainer logic assumes exactly one explorer, so support aggregation must be added without silently changing canonical `K=2` behavior
- legacy singular explorer observability must gain explicit aggregate semantics instead of being silently reused with multi-explorer data
- overlap clustering must stay deterministic across ranks and devices
- dead-duplicate filtering must stay boundary-local so it aligns with existing suppression target construction
- YAML knobs and code-facing variable names should stay versionless; keep `v1` only in change/capability naming
- support-rate metrics must remain mathematically well-defined for every enabled sample, which is why explorer prep failure aborts instead of shrinking the denominator
- recovered-GT observability must stay explicit during `best-K` ablations because larger `K` changes both pseudo-positive voting and any-hit recovered-FN opportunities

## Verification Plan

- Validate the OpenSpec change:
  - `openspec validate study-channel-b-pseudopositive-promotion --type change --strict --json --no-interactive`
- Config-layer verification:
  - reject unknown `stage2_ab.channel_b.pseudo_positive.*` keys
  - reject enabled pseudo-positive mode when `num_rollouts < 2`
  - preserve canonical behavior when disabled
- Target-builder verification:
  - support-rate bucket assignment at multiple `K` values
  - no pseudo-positive promotion at `K=2` unless the contract is intentionally widened later
  - any-hit recovered GT collapse
  - connected-component pseudo-positive clustering
  - duplicate-like dead-anchor filtering aligned to boundary groups
- Training-path verification:
  - one clean teacher-forced forward only
  - pseudo-positive weights reach only bbox/coord losses
  - no pseudo-positive desc CE or matched-prefix structure CE
  - legacy `rollout/explorer/*` metrics remain mean-over-valid-explorer-view explorer summaries under arbitrary `K`
- Smoke/eval verification after implementation:
  - dense-scene qualitative recall
  - hallucination rate
  - duplicate burst rate
  - enumeration-like overproduction
  - oversized / entangled box frequency
  - at least one small `best-K` ablation comparing two enabled `num_rollouts` settings with rate-based metrics
  - recovered-GT rate reporting and explorer-abort reporting alongside pseudo-positive metrics
