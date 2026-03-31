## ADDED Requirements

### Requirement: Channel-B lightweight pseudo-positive v1 is an opt-in extension with a default K=4 profile and legacy K=2 compatibility
The system SHALL preserve the canonical Stage-2 Channel-B `K=2` contract as the legacy compatibility path and SHALL activate lightweight pseudo-positive v1 only when `stage2_ab.channel_b.pseudo_positive.enabled=true`.

Normative behavior:

- `stage2_ab.channel_b.pseudo_positive` MUST be a typed mapping,
- the mapping MUST accept only:
  - `enabled`
  - `coord_weight`
- `enabled` MUST default to `false`,
- `coord_weight` MUST default to `0.5`,
- `coord_weight` MUST be finite and satisfy `0.0 < coord_weight < 1.0`,
- `coord_weight` MUST live in the existing per-bbox-group weight space used by Channel-B coord-supervised object groups,
- when `stage2_ab.channel_b.pseudo_positive.enabled=false`, the canonical Stage-2 Channel-B `K=2` anchor-plus-one-explorer contract remains in force unchanged,
- when `stage2_ab.channel_b.pseudo_positive.enabled=true`, `stage2_ab.channel_b.triage_posterior.num_rollouts` MUST be an integer greater than or equal to `2`,
- repo-authored pseudo-positive profiles and implementation planning artifacts SHOULD use `num_rollouts=4` as the default rollout count, but the contract MUST remain valid for arbitrary `num_rollouts >= 2` so `best-K` ablations do not require a new spec rewrite.

#### Scenario: Disabled pseudo-positive v1 preserves canonical K=2 behavior
- **WHEN** `stage2_ab.channel_b.pseudo_positive.enabled=false`
- **THEN** the trainer uses the canonical `K=2` Channel-B contract
- **AND** it does not apply pseudo-positive v1 bucketing or weighting.

#### Scenario: Repo-authored pseudo-positive v1 defaults to K=4
- **WHEN** the repo authors a default pseudo-positive-enabled profile or planning baseline for this change
- **THEN** it uses `stage2_ab.channel_b.triage_posterior.num_rollouts=4`
- **AND** the same contract still permits other values greater than or equal to `2` for ablation.

#### Scenario: Enabled pseudo-positive v1 accepts arbitrary K at or above two total rollouts
- **WHEN** `stage2_ab.channel_b.pseudo_positive.enabled=true`
- **THEN** config loading rejects any `stage2_ab.channel_b.triage_posterior.num_rollouts` value below `2`
- **AND** the rollout contract uses one anchor view plus `num_rollouts - 1` explorer views.

### Requirement: Pseudo-positive v1 uses one anchor rollout and K-minus-one valid explorer rollouts
When lightweight pseudo-positive v1 is enabled, the system SHALL build pseudo-positive evidence from one deterministic anchor rollout and `num_rollouts - 1` stochastic explorer rollouts.

Normative behavior:

- the anchor rollout MUST remain the only rollout used for target ordering and target editing,
- the explorer rollouts MUST share one authored decode profile and differ only by seed or equivalent stochastic replay identity,
- the explorer rollouts MUST use deterministic per-explorer stochastic identities derived from the sample's rollout seed base and explorer ordinal,
- each rollout MUST independently reuse the existing bounded salvage + strict record acceptance + bbox-valid filtering + sequential dedup + Hungarian matching path,
- if the anchor view fails to complete accepted-clean preparation, the sample MUST be dropped from Channel-B training for the step,
- anchor-preparation failure MUST NOT use the canonical empty-prefix fallback in the enabled pseudo-positive contract,
- each explorer view MUST either complete that standard accepted-clean preparation path or raise and abort the current optimizer step rather than silently reducing the support denominator,
- an explorer view with zero accepted-clean objects still counts as a valid explorer view with zero support evidence,
- per-sample `valid_explorer_count` MUST equal `num_rollouts - 1` whenever the enabled pseudo-positive sample reaches triage,
- the final teacher-forced target MUST still be built from the edited anchor clean sequence rather than from a union ordering across rollouts,
- all enabled losses for the sample MUST still be computed from one merged teacher-forced forward on that edited target.

#### Scenario: Enabled pseudo-positive v1 keeps one edited anchor target
- **WHEN** lightweight pseudo-positive v1 is enabled for a Channel-B sample
- **THEN** the trainer generates one anchor rollout and `num_rollouts - 1` explorer rollouts
- **AND** derives one final edited target from the anchor clean sequence
- **AND** computes all enabled losses from one teacher-forced forward on that target.

### Requirement: Trusted unmatched voting is geometry-first, anchor-centric, and deterministic
When lightweight pseudo-positive v1 is enabled, the system SHALL determine trusted unmatched support only from unmatched anchor objects using deterministic geometry-first association to each explorer view.

Normative behavior:

- trusted unmatched selection MUST begin from anchor clean objects that are unmatched to GT on the anchor side,
- each explorer view MUST be associated to the anchor view independently using the existing one-to-one max-IoU rule and the existing `unlabeled_consistent_iou_threshold`,
- an explorer view contributes support to an anchor object only when:
  - the anchor object has an associated explorer counterpart in that view,
  - the associated explorer counterpart is not GT-matched in that view,
  - the anchor object does not conflict with any GT-backed anchor object on the anchor side,
- an anchor object conflicts with a GT-backed anchor object when its IoU to any GT-backed anchor object is greater than or equal to `unlabeled_consistent_iou_threshold`,
- each unmatched anchor object MUST expose:
  - `support_count`
  - `support_rate = support_count / valid_explorer_count`,
- the enabled pseudo-positive contract MUST use `support_rate` as the canonical scaling surface while also requiring an absolute evidence floor of `support_count >= 2` for pseudo-positive promotion,
- semantic-desc agreement MUST NOT be required for pseudo-positive selection in v1,
- explorer-only non-GT-backed objects MUST NOT become pseudo-positive objects in v1.

#### Scenario: Support rate is computed only from unmatched non-conflicting anchor objects
- **WHEN** an unmatched anchor object has associated explorer counterparts across the valid explorer rollouts
- **THEN** only explorer counterparts that are also unmatched and whose anchor object does not conflict with any GT-backed anchor object contribute to that anchor object's `support_count`
- **AND** `support_rate` is computed as `support_count / valid_explorer_count`.

### Requirement: Recovered GT under pseudo-positive v1 remains a collapsed FN-injection path
When lightweight pseudo-positive v1 is enabled, recovered GT behavior SHALL remain on the existing FN-injection path rather than becoming a second teacher trajectory.

Normative behavior:

- a GT object is `recovered_fn` when it is missed by anchor accepted-clean matching and matched by at least one explorer accepted-clean view,
- multiple explorer hits for the same recovered GT object MUST collapse to one FN-injection object,
- the configured `recovered_ground_truth_weight_multiplier` MUST apply once per recovered GT object,
- recovered GT support count and `recovered_support_rate = recovered_support_count / valid_explorer_count` MUST be recorded in per-sample triage metadata,
- recovered GT objects MUST NOT create a second teacher-forced trajectory or explorer-prefix distillation path.

#### Scenario: Multiple explorer GT hits collapse to one recovered FN object
- **WHEN** two or more explorer rollouts match the same GT object that anchor missed
- **THEN** the trainer appends one recovered FN object through the normal FN-injection path
- **AND** applies `recovered_ground_truth_weight_multiplier` once for that object.

### Requirement: Unmatched anchor objects are bucketed by explorer support rate with overlap-cluster tie-breaking
When lightweight pseudo-positive v1 is enabled, the system SHALL map unmatched anchor objects to dead, shield-only, or pseudo-positive buckets using explorer support rate and an anchor-side overlap-cluster guard.

Normative behavior:

- unmatched anchor objects with `support_count = 0` MUST map to `dead_anchor`,
- unmatched anchor objects with `support_count > 0` that do not satisfy the pseudo-positive rule MUST map to `shielded_anchor`,
- unmatched anchor objects MUST become pseudo-positive candidates only when both:
  - `support_count >= 2`
  - `support_rate >= 2/3`,
- pseudo-positive candidates MUST be clustered as connected components of the undirected anchor-side overlap graph whose edges connect candidate pairs with IoU greater than or equal to `duplicate_iou_threshold`,
- at most one pseudo-positive object MUST be selected from each such overlap cluster,
- the selected pseudo-positive object in a cluster MUST be the candidate with the highest `support_rate`,
- ties inside a cluster MUST be broken by earlier anchor order,
- pseudo-positive candidates that lose cluster tie-breaking MUST fall back to `shielded_anchor`.

#### Scenario: Positive but sub-threshold or low-evidence support remains shield-only
- **WHEN** an unmatched anchor object has `support_count > 0`
- **AND** it does not satisfy both `support_count >= 2` and `support_rate >= 2/3`
- **THEN** it remains in the `shielded_anchor` bucket
- **AND** it is not selected for pseudo-positive positive supervision.

#### Scenario: Threshold-meeting candidates collapse to one selected object inside an overlap cluster
- **WHEN** two unmatched anchor objects both reach pseudo-positive candidacy under the `support_count >= 2` and `support_rate >= 2/3` rule
- **AND** they overlap each other at or above `duplicate_iou_threshold`
- **THEN** only one of them is selected as pseudo-positive
- **AND** the non-winning object falls back to `shielded_anchor`.

### Requirement: Selected pseudo-positive anchors stay in the edited anchor prefix and share the global rollout-prefix structure CE surface
When lightweight pseudo-positive v1 is enabled, selected pseudo-positive anchor objects SHALL remain in the final edited anchor sequence, SHALL receive weighted coord-side positive supervision, and SHALL share the same global rollout-prefix structure CE surface as other retained prefix objects when that token-ce knob is enabled.

Normative behavior:

- selected pseudo-positive anchor objects MUST remain in the final edited anchor prefix in anchor order,
- selected pseudo-positive anchor objects MUST contribute weighted positive supervision only through the existing decoded-box and coord paths,
- selected pseudo-positive anchor objects MUST use `coord_weight` as their shared coord-group weight in v1,
- selected pseudo-positive anchor objects MUST use their own retained anchor coordinate bins as the bbox/coord target source,
- `coord_weight` MUST scale pseudo-positive contributions only for `bbox_geo`, `coord_reg`, and `bbox_size_aux`,
- selected pseudo-positive anchor objects MUST NOT create new desc CE targets,
- selected pseudo-positive anchor objects MAY participate in the same global rollout-prefix structure CE surface as other retained prefix objects when `token_ce.config.rollout_global_prefix_struct_ce_weight > 0`,
- if `bbox_size_aux` is enabled for Channel-B, selected pseudo-positive anchor objects MUST reuse the same decoded-box auxiliary path as other coord-supervised groups,
- ordinary matched-clean and FN-injection supervision behavior MUST remain unchanged.

#### Scenario: Pseudo-positive object receives coord supervision and shared prefix structure CE but no desc CE
- **WHEN** an anchor object is selected as pseudo-positive in lightweight pseudo-positive v1
- **THEN** it remains in the final edited anchor prefix
- **AND** it contributes weighted bbox/coord supervision
- **AND** it may participate in the shared global rollout-prefix structure CE surface
- **BUT** it creates no desc CE.

### Requirement: Channel-B loss application remains bucketed under one teacher-forced forward
When lightweight pseudo-positive v1 is enabled, the system SHALL preserve the one-forward training contract while applying different loss surfaces to matched-clean, FN-injection, pseudo-positive, shield-only, and dead-anchor buckets.

Normative behavior:

- matched-clean objects MUST remain eligible for matched-prefix coord supervision,
- retained prefix objects MAY participate in global rollout-prefix structure CE when `token_ce.config.rollout_global_prefix_struct_ce_weight > 0`,
- FN-injection objects MUST remain eligible for tail coord supervision and FN desc CE,
- pseudo-positive objects MUST remain eligible for positive coord-supervised losses and the shared global rollout-prefix structure CE surface in v1,
- shield-only objects MUST remain outside positive desc, bbox, and coord supervision and MAY participate only in the shared global rollout-prefix structure CE surface,
- dead-anchor objects MUST remain outside positive CE, bbox, and coord supervision,
- global rollout-prefix structure CE MUST be controlled by a single typed token-ce config knob rather than per-bucket prefix knobs,
- all enabled loss terms for all buckets MUST be derived from the same clean edited-target teacher-forced forward.

#### Scenario: Bucketed loss computation still uses one clean forward
- **WHEN** a pseudo-positive v1 sample contains matched-clean, FN-injection, pseudo-positive, shield-only, and dead-anchor objects
- **THEN** the trainer computes all enabled losses from one clean edited-target forward
- **AND** it does not run a second teacher-forced forward for dead-anchor handling.

### Requirement: Dead anchors stay out of the final target and only duplicate-like dead branches produce suppression targets
When lightweight pseudo-positive v1 is enabled, the system SHALL keep dead anchors out of the final edited target and SHALL narrow explicit negative suppression to duplicate-like dead alternate continuations only.

Normative behavior:

- all dead anchors MUST be excluded from the final edited target sequence,
- dead anchors MUST NOT be reinserted into the final teacher-forced target,
- dead anchors MUST NOT receive full-object negative CE,
- a dead anchor is duplicate-like in v1 only when it belongs to the same local continuation boundary group as an earlier kept anchor object, overlaps that earlier kept anchor object at or above `duplicate_iou_threshold`, and has the same normalized description under the existing duplicate-style normalization rule,
- duplicate-like dead anchors MAY remain in dead-anchor bookkeeping,
- duplicate-like dead anchors MUST create first-divergent bad-token suppression targets only when the dead-anchor suppression objective module is enabled,
- dead anchors that are not duplicate-like MUST NOT create `dead_anchor_suppression_targets`,
- the first-divergent bad-token suppression targets for duplicate-like dead anchors MUST be consumed from the same clean teacher-forced forward logits rather than from a second forward.

#### Scenario: Duplicate-like dead branch is penalized without a second forward
- **WHEN** a dead anchor belongs to the same local continuation boundary group as an earlier kept anchor object
- **AND** overlaps that earlier kept anchor object at or above `duplicate_iou_threshold`
- **AND** shares the same normalized description
- **THEN** it is eligible to create a first-divergent dead-branch suppression target
- **AND** that suppression is applied on the logits from the single clean teacher-forced forward.

#### Scenario: Non-duplicate dead anchor is dropped without explicit suppression
- **WHEN** a dead anchor does not satisfy the duplicate-like predicate
- **THEN** it is excluded from the final target
- **AND** it creates no full-object negative CE
- **AND** it creates no `dead_anchor_suppression_targets`.

### Requirement: Pseudo-positive v1 emits auditable support and bucket observability
When lightweight pseudo-positive v1 is enabled, the system SHALL expose enough observability to audit support counting, pseudo-positive promotion, and recovered-GT evidence.

Normative behavior:

- per-sample Channel-B triage metadata MUST include:
  - `valid_explorer_count`,
  - per-anchor explorer support counts,
  - per-anchor explorer support rates,
  - `pseudo_positive_anchor_indices`,
  - `dead_explorer_indices_by_view`,
  - per-recovered-GT explorer support counts,
  - per-recovered-GT explorer support rates,
- in arbitrary-`K` mode, singular explorer-local metadata carriers such as `dead_explorer_indices` MUST NOT be reused as if they described one merged explorer view,
- monitor payloads MAY mirror those fields when monitoring is enabled,
- aggregate count-like metrics MUST include:
  - `train/triage/pseudo_positive_candidate_count`
  - `train/triage/pseudo_positive_subthreshold_count`
  - `train/triage/pseudo_positive_selected_count`
  - `train/triage/pseudo_positive_cluster_demoted_count`
  - `train/triage/anchor_preparation_dropped_count`
- aggregate numerator / denominator metrics MUST include:
  - `train/triage/pseudo_positive_support_rate_num`
  - `train/triage/pseudo_positive_support_rate_den`
  - `train/triage/pseudo_positive_selected_support_rate_num`
  - `train/triage/pseudo_positive_selected_support_rate_den`
  - `train/triage/recovered_ground_truth_rate_num`
  - `train/triage/recovered_ground_truth_rate_den`
- `train/triage/unlabeled_consistent_count` MUST remain the canonical total shielded-anchor count and MUST equal `train/triage/pseudo_positive_subthreshold_count + train/triage/pseudo_positive_cluster_demoted_count` in pseudo-positive-enabled runs,
- `best-K` ablation artifacts MUST report the recovered-GT rate metrics alongside pseudo-positive support-rate metrics because `num_rollouts` co-varies both mechanisms,
- explorer-preparation aborts MUST be surfaced through failure telemetry or ablation reporting rather than through a finalized-step `train/triage/*` metric.

#### Scenario: Audit artifacts show how pseudo-positive objects were selected
- **WHEN** pseudo-positive v1 training emits triage metadata or monitoring artifacts
- **THEN** an auditor can reconstruct both the support numerator and the support rate for each promoted anchor object
- **AND** can distinguish pseudo-positive support-rate evidence from recovered-GT support-rate evidence across different `num_rollouts` settings.

### Requirement: Pseudo-positive v1 remains inference-feedback gated
When lightweight pseudo-positive v1 is enabled, the system SHALL treat the experiment as conservative and feedback-seeking rather than as a fully trusted contract replacement.

Normative behavior:

- the experiment SHOULD monitor hallucination rate, duplicate burst rate, enumeration-style overproduction, oversized or entangled box frequency, and dense-scene recall,
- materially worse hallucination, duplicate behavior, enumeration collapse, or oversized-box behavior SHOULD be treated as a stop signal,
- broader semantic gating and broader dead-negative design remain deferred unless these failures clearly emerge.

#### Scenario: Duplicate burst regression blocks widening
- **WHEN** lightweight pseudo-positive v1 materially increases duplicate bursts or related shortcut failures in inference feedback
- **THEN** the operator treats that as a stop signal
- **AND** does not widen the pseudo-positive contract without further redesign.
