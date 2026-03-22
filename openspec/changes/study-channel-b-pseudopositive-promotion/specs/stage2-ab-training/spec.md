# stage2-ab-training Specification (Delta)

## MODIFIED Requirements

### Requirement: Stage-2 Two-Channel module configs are strict and typed
Stage-2 Two-Channel SHALL validate module `config` payloads and `stage2_ab.channel_b` payloads strictly so experiments are reproducible and fail fast on schema drift.

Normative behavior:

- `loss_dead_anchor_suppression.config` MUST be an empty mapping in v1.
- `token_ce.config` no longer accepts any legacy invalid-structure amplification knob for Channel-B.
- `stage2_ab.channel_b` MUST accept only:
  - `duplicate_iou_threshold`
  - `pseudo_positive`
  - `triage_posterior`
  - `producer_wait_timeout_s`
  - `ddp_phase_timeout_s`
- `stage2_ab.channel_b.pseudo_positive` MUST be a typed mapping.
- `stage2_ab.channel_b.pseudo_positive` MUST accept only:
  - `enabled`
  - `coord_weight`
- `stage2_ab.channel_b.pseudo_positive.enabled` MUST default to `false`.
- `stage2_ab.channel_b.pseudo_positive.coord_weight` MUST default to `0.5`.
- `stage2_ab.channel_b.pseudo_positive.coord_weight` MUST satisfy `0.0 < coord_weight < 1.0`.
- Unknown keys in a module `config` or in `stage2_ab.channel_b` MUST fail fast with actionable diagnostics.

#### Scenario: Unknown pseudo-positive key fails fast
- **WHEN** a Stage-2 AB config includes an unknown key under `stage2_ab.channel_b.pseudo_positive`
- **THEN** configuration parsing fails fast
- **AND** the error indicates the full dotted path of the unknown key.

### Requirement: Stage-2 Two-Channel adheres to the unified loss registry contract
Stage-2 Two-Channel training SHALL implement loss naming and masking semantics per the `teacher-forcing-unified-loss-registry`
capability.

Normative behavior:

- Stage-2 Two-Channel MUST build token-type masks and object-subset masks according to the registry contexts:
  - Channel-A uses `context=gt` for CE and bbox/coord supervision.
  - Channel-B uses `context=rollout` with EOS-enforced semantics.
- when `stage2_ab.channel_b.pseudo_positive.enabled=false`, Channel-B rollout context remains FP-neutral outside matched-clean and FN-injection supervision,
- when `stage2_ab.channel_b.pseudo_positive.enabled=true`, only the selected `pseudo_positive` anchor subset is exempt from blanket FP-neutral handling, and that subset remains limited to coord-side supervision,
- when the module pipeline is enabled, objective/diagnostics modules MUST emit metric keys consistent with the registry’s canonical component names.

#### Scenario: Enabled pseudo-positive does not broaden text-side rollout supervision
- **WHEN** `stage2_ab.channel_b.pseudo_positive.enabled=true`
- **THEN** only the selected pseudo-positive anchor subset is removed from blanket FP-neutral coord masking
- **AND** the same step does not add pseudo-positive desc CE or matched-prefix structure CE.

### Requirement: Stage-2 AB Channel-B uses anchor-rooted rollout triage with legacy K=2 compatibility and default K=4 pseudo-positive evidence
When `custom.trainer_variant: stage2_two_channel`, the canonical Channel-B contract SHALL build its clean teacher-forced target from rollout evidence rooted in the anchor clean sequence.

Normative behavior:

- when `stage2_ab.channel_b.pseudo_positive.enabled=false`, the canonical Channel-B contract uses exactly two rollout views:
  - one anchor rollout using greedy / deterministic decoding,
  - one explorer rollout using stochastic decoding configured under `stage2_ab.channel_b.triage_posterior`,
- when `stage2_ab.channel_b.pseudo_positive.enabled=true`, the opt-in pseudo-positive contract uses exactly `stage2_ab.channel_b.triage_posterior.num_rollouts` total rollout views:
  - one anchor rollout using greedy / deterministic decoding,
  - `num_rollouts - 1` explorer rollouts using the shared stochastic decode profile configured under `stage2_ab.channel_b.triage_posterior`,
  - repo-authored default pseudo-positive profiles SHOULD set `num_rollouts` to `4`,
- each rollout MUST independently reuse the existing bounded salvage + strict record acceptance + bbox-valid filtering + sequential dedup + Hungarian matching path,
- GT-backed semantics MUST inherit the existing Channel-B accepted-clean Hungarian + gating contract,
- the final positive target MUST be built by editing the anchor clean sequence rather than rebuilding a union order,
- explorer-only non-GT-backed objects MUST be treated as dead by default,
- a GT hit found only on one or more explorer views MUST project to `recovered_fn`, not to anchor retention,
- when `stage2_ab.channel_b.pseudo_positive.enabled=true` and the anchor view fails to complete accepted-clean preparation, the sample MUST be dropped from Channel-B training for that step,
- the enabled pseudo-positive contract MUST NOT use the canonical empty-prefix fallback for malformed anchor preparation.

#### Scenario: Enabled pseudo-positive drops malformed anchor samples
- **WHEN** `stage2_ab.channel_b.pseudo_positive.enabled=true`
- **AND** the anchor rollout does not complete accepted-clean preparation for a sample
- **THEN** that sample is dropped from Channel-B training for the step
- **AND** the trainer does not fall back to the empty-prefix FN-only path for that sample.

### Requirement: Channel-B invalid rollouts fall back deterministically (no silent skips)
When Channel-B is selected, rollout-preparation failure handling SHALL remain deterministic and explicit.

Normative behavior:

- when `stage2_ab.channel_b.pseudo_positive.enabled=false`, the canonical invalid-rollout fallback remains unchanged:
  - mark the rollout invalid for that sample,
  - fall back to the canonical empty prefix `{\"objects\": [`,
  - treat the rollout as containing zero valid predicted objects,
  - append all GT objects as FN and continue training that sample,
- when `stage2_ab.channel_b.pseudo_positive.enabled=true` and the anchor view fails to complete accepted-clean preparation, the sample MUST be dropped from Channel-B training for that step,
- when `stage2_ab.channel_b.pseudo_positive.enabled=true` and any explorer view fails to complete accepted-clean preparation, the trainer MUST raise and abort the current optimizer step,
- the enabled pseudo-positive contract MUST NOT silently reinterpret support rates over fewer than `num_rollouts - 1` valid explorer views.

#### Scenario: Enabled pseudo-positive does not use anchor empty-prefix fallback
- **WHEN** `stage2_ab.channel_b.pseudo_positive.enabled=true`
- **AND** the anchor rollout cannot be recovered into accepted-clean preparation
- **THEN** the sample is dropped from Channel-B training for the step
- **AND** the empty-prefix fallback is not used for that sample.

### Requirement: Stage-2 AB Channel-B v3-specific knobs are typed and grouped
The Stage-2 AB config SHALL expose v3-specific rollout knobs under `stage2_ab.channel_b.triage_posterior`.

Normative behavior:

- `stage2_ab.channel_b.triage_posterior` MUST be a typed mapping,
- the mapping MUST accept only:
  - `num_rollouts`
  - `explorer_temperature`
  - `explorer_top_p`
  - `explorer_top_k`
  - `unlabeled_consistent_iou_threshold`
  - `recovered_ground_truth_weight_multiplier`
- when `stage2_ab.channel_b.pseudo_positive.enabled=false`, `num_rollouts` MUST equal `2`,
- when `stage2_ab.channel_b.pseudo_positive.enabled=true`, `num_rollouts` MUST be an integer greater than or equal to `2`,
- unknown keys under `stage2_ab.channel_b.triage_posterior` MUST fail fast.

#### Scenario: Enabled pseudo-positive accepts arbitrary K at or above two total rollouts
- **WHEN** `stage2_ab.channel_b.pseudo_positive.enabled=true`
- **THEN** config loading fails fast unless `stage2_ab.channel_b.triage_posterior.num_rollouts >= 2`.

### Requirement: Stage-2 AB arbitrary-K explorer observability remains deterministic and aggregate
When `stage2_ab.channel_b.pseudo_positive.enabled=true`, the Stage-2 AB trainer SHALL keep legacy single-explorer observability surfaces meaningful by defining them as deterministic aggregate explorer summaries.

Normative behavior:

- aggregate explorer metric families such as `rollout/explorer/*` MUST represent deterministic summaries across valid explorer views rather than one arbitrarily chosen explorer view,
- `rollout/explorer/pred_objects`, `rollout/explorer/valid_pred_objects`, `rollout/explorer/parse_truncated_rate`, `rollout/explorer/gen_new_tokens_mean`, `rollout/explorer/gen_new_tokens_p90`, `rollout/explorer/near_iou90_any`, and `rollout/explorer/near_iou90_same` MUST use mean-over-valid-explorer-view aggregation so their semantics remain comparable across different `num_rollouts` values,
- explorer decode-profile observability such as `rollout/explorer/temperature`, `rollout/explorer/do_sample`, `rollout/explorer/top_p`, `rollout/explorer/top_k`, and `explorer_decode_mode` MUST report the shared explorer decode profile used for all explorer views,
- `stage2/raw_rollouts` MUST continue to count the total raw rollout trajectories produced across the anchor rollout plus all explorer rollouts,
- `rollout/parse_truncated_rate` MUST continue to represent the parse-truncated ratio over those total raw rollouts,
- per-sample explorer-local metadata that cannot be losslessly merged, such as dead explorer index lists, MUST move to explorer-indexed carriers such as `dead_explorer_indices_by_view` rather than overloading the legacy singular field.

#### Scenario: Enabled arbitrary-K keeps explorer metrics aggregate rather than arbitrary
- **WHEN** `stage2_ab.channel_b.pseudo_positive.enabled=true`
- **AND** `stage2_ab.channel_b.triage_posterior.num_rollouts > 2`
- **THEN** legacy `rollout/explorer/*` metrics still have a defined deterministic meaning
- **AND** explorer-local metadata is emitted through explorer-indexed carriers rather than singular merged fields.

### Requirement: Recovered GT objects stay on the FN injection path with higher weight
The canonical Channel-B contract SHALL treat recovered GT objects as weighted FN injections, not as a second teacher trajectory.

Normative behavior:

- `recovered GT` means “missed in anchor accepted-clean matching and hit in at least one explorer accepted-clean matching view,”
- recovered GT objects MUST remain on the same FN injection path used by ordinary FN objects,
- the configured `recovered_ground_truth_weight_multiplier` MUST increase their desc+geo+coord supervision weight relative to ordinary FN objects,
- recovered-prefix distillation MUST NOT be part of the contract.

#### Scenario: Recovered GT object uses weighted FN injection
- **WHEN** a GT object is missed in anchor and hit in at least one explorer view
- **THEN** it is appended through the normal FN-injection path
- **AND** it receives the configured recovered-FN positive weight
- **AND** no separate explore-prefix teacher-forced pass is created.

### Requirement: Channel-B v3 uses deterministic one-to-one anchor/explorer association
The Channel-B contract SHALL associate anchor accepted objects to explorer accepted objects deterministically before projecting triage actions.

Normative behavior:

- candidate cross-rollout pairs MUST be scored by IoU,
- only pairs with `IoU >= unlabeled_consistent_iou_threshold` are eligible,
- the chosen association MUST be one-to-one and maximize IoU,
- if multiple assignments achieve the same maximum total IoU, the chosen assignment MUST be the one whose sorted pair list `[(anchor_index, explorer_index), ...]` is lexicographically smallest,
- when `stage2_ab.channel_b.pseudo_positive.enabled=true`, the same deterministic one-to-one rule MUST be applied independently between the anchor view and each explorer view,
- support voting in the enabled pseudo-positive contract MUST be computed from support ratios whose numerator is the number of supporting explorer views and whose denominator is the sample's `valid_explorer_count`, and pseudo-positive promotion MUST still require an absolute evidence floor of `support_count >= 2`.

#### Scenario: Enabled pseudo-positive uses independent deterministic association and ratio-based voting per explorer
- **WHEN** `stage2_ab.channel_b.pseudo_positive.enabled=true`
- **THEN** each explorer view is associated to the anchor view independently under the canonical one-to-one IoU rule
- **AND** support ratios are computed from those independent deterministic associations using the sample's `valid_explorer_count`.

### Requirement: Channel-B v3 uses one merged teacher-forced forward
The Channel-B contract SHALL realize `L(clean_anchor) + L(explore-derived corrections)` through one merged teacher-forced forward on the edited anchor target.

Normative behavior:

- the trainer MUST run one teacher-forced forward on the final edited target,
- positive, weighted-FN, pseudo-positive coord, and dead-anchor UL terms MUST be derived from that same forward,
- the trainer MUST NOT require a second explore teacher-forced payload.

#### Scenario: Single-forward target realization with pseudo-positive enabled
- **WHEN** `stage2_ab.channel_b.pseudo_positive.enabled=true`
- **THEN** all loss terms are derived from a single teacher-forced forward over the edited anchor target
- **AND** no second teacher-forced explore payload is required.

### Requirement: Stage-2 AB can add decoded-box size auxiliaries through `bbox_size_aux`
Stage-2 AB SHALL support optional decoded-box size auxiliaries on the existing
coord-supervised decoded-box path without changing bbox parameterization or decode format.

Normative behavior:

- when `bbox_size_aux.config.log_wh_weight > 0`, the trainer MUST add log-width/log-height supervision on canonicalized decoded boxes for every Channel-B coord-supervised object group,
- when `bbox_size_aux.config.oversize_penalty_weight > 0`, the trainer MAY add the thresholded oversize penalty on decoded boxes for the same coord-supervised groups,
- when `stage2_ab.channel_b.pseudo_positive.enabled=true`, selected pseudo-positive objects MAY participate in `bbox_size_aux` through the same coord-supervised group path as matched-clean and FN-injection objects,
- pseudo-positive `bbox_size_aux` targets MUST use the selected anchor object's own canonical coordinate bins rather than explorer-consensus geometry,
- Channel-A and Channel-B applicability MUST remain controlled by the authored `channels` field on the `bbox_size_aux` module entry,
- `bbox_size_aux` MUST remain separate from `bbox_geo` in the authored pipeline so the new size loss is an independently removable plugin module,
- `bbox_size_aux` MUST consume the current four bbox coord slots in the existing `xyxy` order rather than introducing a new bbox expression.

#### Scenario: Pseudo-positive size aux uses anchor geometry
- **WHEN** `stage2_ab.channel_b.pseudo_positive.enabled=true`
- **AND** a selected pseudo-positive anchor object participates in `bbox_size_aux`
- **THEN** its target decoded box is derived from that anchor object's own canonical coordinates
- **AND** no explorer-consensus geometry target is constructed for `bbox_size_aux`.
