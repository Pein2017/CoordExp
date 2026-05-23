# channel-b-lightweight-pseudopositive-v1 Specification

## Purpose

Define the opt-in lightweight pseudo-positive Channel-B extension that keeps the
legacy `K=2` contract available while enabling arbitrary-`K` multi-view support
with a repo-default `K=4` profile.

## Requirements

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
- when `stage2_ab.channel_b.pseudo_positive.enabled=false`, the canonical Stage-2 Channel-B clean-prefix contract uses `K=2` peer rollout attempts,
- when `stage2_ab.channel_b.pseudo_positive.enabled=true`, `stage2_ab.channel_b.triage_posterior.num_rollouts` MUST be an integer greater than or equal to `4`,
- repo-authored pseudo-positive profiles and implementation planning artifacts SHOULD use `num_rollouts=4` as the default rollout count.

#### Scenario: Disabled pseudo-positive v1 preserves canonical K=2 behavior
- **WHEN** `stage2_ab.channel_b.pseudo_positive.enabled=false`
- **THEN** the trainer uses the canonical `K=2` Channel-B contract
- **AND** it does not apply pseudo-positive v1 bucketing or weighting.

#### Scenario: Repo-authored pseudo-positive v1 defaults to K=4
- **WHEN** the repo authors a default pseudo-positive-enabled profile or planning baseline for this change
- **THEN** it uses `stage2_ab.channel_b.triage_posterior.num_rollouts=4`
- **AND** enabled pseudo-positive ablations still require values greater than or equal to `4`.

#### Scenario: Enabled pseudo-positive v1 requires at least four total rollouts
- **WHEN** `stage2_ab.channel_b.pseudo_positive.enabled=true`
- **THEN** config loading rejects any `stage2_ab.channel_b.triage_posterior.num_rollouts` value below `4`
- **AND** the rollout contract uses `num_rollouts` peer attempts with no
  privileged role split.

### Requirement: Pseudo-positive v1 uses K peer rollout attempts
When lightweight pseudo-positive v1 is enabled, the system SHALL build
pseudo-positive evidence from `num_rollouts` independent peer rollout attempts.
The attempts MAY use different ordinal temperatures, but no attempt is globally
privileged by ordinal, temperature bucket, or decode request.

Normative behavior:

- each rollout attempt MUST be eligible to become the current training attempt
  and therefore own one teacher-forced target segment,
- `stage2_ab.channel_b.triage_posterior.rollout_temperatures`, when authored,
  MUST provide either one broadcast temperature or exactly `num_rollouts`
  ordinal temperatures,
- rollout attempts MUST use deterministic per-attempt stochastic identities
  derived from the sample's rollout seed base and attempt ordinal,
- each rollout MUST independently reuse the existing bounded salvage + strict record acceptance + bbox-valid filtering + sequential dedup + configured Stage-2 assignment path,
- each rollout attempt MUST either complete that standard accepted-clean preparation path or raise/drop according to the configured invalid-rollout policy rather than silently reducing the support denominator,
- a rollout attempt with zero accepted-clean objects still counts as valid zero-support evidence,
- for each current attempt, peer-support denominator MUST equal `num_rollouts - 1`,
- the final teacher-forced target for a segment MUST be built from that current attempt's edited clean sequence rather than from a union ordering across rollouts,
- all enabled losses for the segment MUST still be computed from one merged teacher-forced forward on that edited target.

#### Scenario: Enabled pseudo-positive v1 keeps one edited current-attempt target
- **WHEN** lightweight pseudo-positive v1 is enabled for a Channel-B sample
- **THEN** the trainer generates `num_rollouts` peer attempts
- **AND** derives one edited target per valid current attempt
- **AND** computes all enabled losses from one teacher-forced forward on that target.

### Requirement: Trusted unmatched voting is geometry-first, desc-consistent, and deterministic
When lightweight pseudo-positive v1 is enabled, the system SHALL determine trusted unmatched support for the current attempt using deterministic geometry-first association to each peer attempt.

Normative behavior:

- trusted unmatched selection MUST begin from current-attempt clean objects that are unmatched to GT in that attempt,
- each peer attempt MUST be associated to the current attempt independently using the existing one-to-one max-IoU rule and the existing `unlabeled_consistent_iou_threshold`,
- a peer attempt contributes support to a current-attempt object only when:
  - the current object has an associated peer counterpart in that attempt,
  - the associated peer counterpart is not GT-matched in that attempt,
  - the current object does not conflict with any GT-backed object in the current attempt,
  - the current object and peer counterpart have the same normalized description,
- a current object conflicts with a GT-backed object when its IoU to any GT-backed current-attempt object is greater than or equal to `unlabeled_consistent_iou_threshold`,
- each unmatched current-attempt object MUST expose:
  - `support_count`
  - `support_rate = support_count / (num_rollouts - 1)`,
- pseudo-positive promotion MUST require `support_count == num_rollouts - 1`
  and `support_rate == 1.0`,
- peer-only non-GT-backed objects MUST NOT become pseudo-positive objects unless
  they satisfy the same condition when that peer is evaluated as the current attempt.

#### Scenario: Support rate is computed only from unmatched non-conflicting current-attempt objects
- **WHEN** an unmatched current-attempt object has associated counterparts across the valid peer attempts
- **THEN** only peer counterparts that are also unmatched, desc-consistent, and whose current object does not conflict with any GT-backed current-attempt object contribute to that object's `support_count`
- **AND** `support_rate` is computed as `support_count / (num_rollouts - 1)`.

### Requirement: Recovered GT under pseudo-positive v1 remains a collapsed FN-injection path
When lightweight pseudo-positive v1 is enabled, recovered GT behavior SHALL remain on the existing FN-injection path rather than becoming a second teacher trajectory.

Normative behavior:

- a GT object is `recovered_fn` for the current attempt when it is missed by current-attempt accepted-clean matching and matched by at least one peer attempt,
- multiple peer hits for the same recovered GT object MUST collapse to one FN-injection object,
- the configured `recovered_ground_truth_weight_multiplier` MUST apply once per recovered GT object,
- recovered GT support count and `recovered_support_rate = recovered_support_count / (num_rollouts - 1)` MUST be recorded in per-sample triage metadata,
- recovered GT objects MUST NOT create a second teacher-forced trajectory or peer-prefix distillation path.

#### Scenario: Multiple peer GT hits collapse to one recovered FN object
- **WHEN** two or more peer attempts match the same GT object that the current attempt missed
- **THEN** the trainer appends one recovered FN object through the normal FN-injection path
- **AND** applies `recovered_ground_truth_weight_multiplier` once for that object.

### Requirement: Unmatched current-attempt objects are bucketed by peer support rate with overlap-cluster tie-breaking
When lightweight pseudo-positive v1 is enabled, the system SHALL map unmatched
current-attempt objects to dead, shield-only, or pseudo-positive buckets using
peer support rate and a current-attempt overlap-cluster guard.

Normative behavior:

- unmatched current-attempt objects with `support_count = 0` MUST map to the dead bucket,
- unmatched current-attempt objects with `support_count > 0` that do not satisfy the pseudo-positive rule MUST map to the shield-only bucket,
- unmatched current-attempt objects MUST become pseudo-positive candidates only when:
  - `num_rollouts >= 4`
  - `support_count == num_rollouts - 1`
  - `support_rate == 1.0`,
- pseudo-positive candidates MUST be clustered as connected components of the undirected current-attempt overlap graph whose edges connect candidate pairs with IoU greater than or equal to the configured duplicate-control IoU threshold,
- at most one pseudo-positive object MUST be selected from each such overlap cluster,
- the selected pseudo-positive object in a cluster MUST be the candidate with the highest `support_rate`,
- ties inside a cluster MUST be broken by earlier current-attempt order,
- pseudo-positive candidates that lose cluster tie-breaking MUST fall back to
  the shield-only bucket.

#### Scenario: Positive but sub-threshold or low-evidence support remains shield-only
- **WHEN** an unmatched current-attempt object has `support_count > 0`
- **AND** it does not satisfy full K-of-K peer consensus
- **THEN** it remains in the shield-only bucket
- **AND** it is not selected for pseudo-positive positive supervision.

#### Scenario: Threshold-meeting candidates collapse to one selected object inside an overlap cluster
- **WHEN** two unmatched current-attempt objects both reach pseudo-positive candidacy under the K-of-K rule
- **AND** they overlap each other at or above the configured duplicate-control IoU threshold
- **THEN** only one of them is selected as pseudo-positive
- **AND** the non-winning object falls back to the shield-only bucket.

### Requirement: Selected pseudo-positive objects stay in the edited current-attempt prefix and share the global rollout-prefix structure CE surface
When lightweight pseudo-positive v1 is enabled, selected pseudo-positive
current-attempt objects SHALL remain in that attempt's final edited sequence,
SHALL receive weighted coord-side positive supervision, and SHALL share the
same global rollout-prefix structure CE surface as other retained prefix objects
when that token-ce knob is enabled.

Normative behavior:

- selected pseudo-positive current-attempt objects MUST remain in the final edited current-attempt prefix in attempt order,
- selected pseudo-positive current-attempt objects MUST NOT create removed decoded-box or coord-regularizer objective terms,
- selected pseudo-positive current-attempt objects MUST use `coord_weight` as their shared coord-group weight in v1,
- selected pseudo-positive current-attempt objects MUST use their own retained coordinate bins as the bbox/coord target source,
- `coord_weight` is retained only as historical config context and MUST NOT
  activate removed `bbox_geo`, `coord_reg`, or `bbox_size_aux` modules,
- selected pseudo-positive current-attempt objects MUST NOT create new desc CE targets,
- selected pseudo-positive current-attempt objects MAY participate in the same global rollout-prefix structure CE surface as other retained prefix objects when `token_ce.config.rollout_global_prefix_struct_ce_weight > 0`,
- `bbox_size_aux` MUST remain unavailable for Channel-B pseudo-positive supervision,
- ordinary matched-clean and FN-injection supervision behavior MUST remain unchanged.

#### Scenario: Pseudo-positive object receives coord supervision and shared prefix structure CE but no desc CE
- **WHEN** a current-attempt object is selected as pseudo-positive in lightweight pseudo-positive v1
- **THEN** it remains in the final edited current-attempt prefix
- **AND** it contributes weighted bbox/coord supervision
- **AND** it may participate in the shared global rollout-prefix structure CE surface
- **BUT** it creates no desc CE.

### Requirement: Channel-B loss application remains bucketed under one teacher-forced forward
When lightweight pseudo-positive v1 is enabled, the system SHALL preserve the one-forward training contract while applying different loss surfaces to matched-clean, FN-injection, pseudo-positive, shield-only, and dead-current buckets.

Normative behavior:

- matched-clean objects MUST remain eligible for matched-prefix coord supervision,
- retained prefix objects MAY participate in global rollout-prefix structure CE when `token_ce.config.rollout_global_prefix_struct_ce_weight > 0`,
- FN-injection objects MUST remain eligible for tail coord supervision and FN desc CE,
- pseudo-positive objects MUST remain eligible for positive coord-supervised losses and the shared global rollout-prefix structure CE surface in v1,
- shield-only objects MUST remain outside positive desc, bbox, and coord supervision and MAY participate only in the shared global rollout-prefix structure CE surface,
- dead-current objects MUST remain outside positive CE, bbox, and coord supervision,
- global rollout-prefix structure CE MUST be controlled by a single typed token-ce config knob rather than per-bucket prefix knobs,
- all enabled loss terms for all buckets MUST be derived from the same clean edited-target teacher-forced forward.

#### Scenario: Bucketed loss computation still uses one clean forward
- **WHEN** a pseudo-positive v1 sample contains matched-clean, FN-injection, pseudo-positive, shield-only, and dead-current objects
- **THEN** the trainer computes all enabled losses from one clean edited-target forward
- **AND** it does not run a second teacher-forced forward for dead-current handling.

### Requirement: Dead current-attempt objects stay out of the final target and duplicate-like branches remain diagnostic-only
When lightweight pseudo-positive v1 is enabled, the system SHALL keep dead
current-attempt objects out of the final edited target. Duplicate-like dead
branches MAY remain only in duplicate-control diagnostics and bookkeeping after
Task 1A objective cleanup.

Normative behavior:

- all dead current-attempt objects MUST be excluded from the final edited target sequence,
- dead current-attempt objects MUST NOT be reinserted into the final teacher-forced target,
- dead current-attempt objects MUST NOT receive full-object negative CE,
- a dead current-attempt object is duplicate-like in v1 only when it belongs to the same local continuation boundary group as an earlier kept current-attempt object, overlaps that earlier kept current-attempt object at or above the configured duplicate-control IoU threshold, and has the same normalized description under the existing duplicate-style normalization rule,
- duplicate-like dead current-attempt objects MAY remain in duplicate-control diagnostics and dead-object bookkeeping,
- duplicate-like dead current-attempt objects MAY record first-divergence metadata, but that metadata is diagnostic-only,
- duplicate-like dead current-attempt objects MUST NOT create live objective terms, positive/negative supervision, training target distributions, or loss-consumed suppression targets,
- dead current-attempt objects that are not duplicate-like MUST NOT create duplicate-control first-divergence diagnostic metadata.

#### Scenario: Duplicate-like dead branch is recorded without supervision
- **WHEN** a dead current-attempt object belongs to the same local continuation boundary group as an earlier kept current-attempt object
- **AND** overlaps that earlier kept current-attempt object at or above the configured duplicate-control IoU threshold
- **AND** shares the same normalized description
- **THEN** it may be recorded in duplicate-control diagnostics and bookkeeping
- **AND** any first-divergence metadata is diagnostic-only
- **AND** it creates no live objective terms, positive/negative supervision, training target distributions, or loss-consumed suppression targets.

#### Scenario: Non-duplicate dead current-attempt object is dropped without explicit suppression
- **WHEN** a dead current-attempt object does not satisfy the duplicate-like predicate
- **THEN** it is excluded from the final target
- **AND** it creates no full-object negative CE
- **AND** it creates no duplicate-control first-divergence diagnostic metadata.

### Requirement: Pseudo-positive v1 emits auditable support and bucket observability
When lightweight pseudo-positive v1 is enabled, the system SHALL expose enough observability to audit support counting, pseudo-positive promotion, and recovered-GT evidence.

Normative behavior:

- per-sample Channel-B triage metadata MUST include:
  - `valid_peer_count`,
  - per-current-attempt peer support counts,
  - per-current-attempt peer support rates,
  - `pseudo_positive_current_indices`,
  - `dead_peer_indices_by_view`,
  - per-recovered-GT peer support counts,
  - per-recovered-GT peer support rates,
- in arbitrary-`K` mode, singular peer-local metadata carriers MUST NOT be
  reused as if they described one merged peer view,
- monitor payloads MAY mirror those fields when monitoring is enabled,
- aggregate count-like metrics MUST include:
  - `train/triage/pseudo_positive_candidate_count`
  - `train/triage/pseudo_positive_subthreshold_count`
  - `train/triage/pseudo_positive_selected_count`
  - `train/triage/pseudo_positive_cluster_demoted_count`
  - `train/triage/current_preparation_dropped_count`
- aggregate numerator / denominator metrics MUST include:
  - `train/triage/pseudo_positive_support_rate_num`
  - `train/triage/pseudo_positive_support_rate_den`
  - `train/triage/pseudo_positive_selected_support_rate_num`
  - `train/triage/pseudo_positive_selected_support_rate_den`
  - `train/triage/recovered_ground_truth_rate_num`
  - `train/triage/recovered_ground_truth_rate_den`
- `train/triage/shield_only_count` MUST remain the canonical total shield-only
  count and MUST equal `train/triage/pseudo_positive_subthreshold_count +
  train/triage/pseudo_positive_cluster_demoted_count` in
  pseudo-positive-enabled runs,
- `best-K` ablation artifacts MUST report the recovered-GT rate metrics alongside pseudo-positive support-rate metrics because `num_rollouts` co-varies both mechanisms,
- current-attempt preparation aborts MUST be surfaced through failure telemetry
  or ablation reporting rather than through a finalized-step `train/triage/*`
  metric.

#### Scenario: Audit artifacts show how pseudo-positive objects were selected
- **WHEN** pseudo-positive v1 training emits triage metadata or monitoring artifacts
- **THEN** an auditor can reconstruct both the support numerator and the support rate for each promoted current-attempt object
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
