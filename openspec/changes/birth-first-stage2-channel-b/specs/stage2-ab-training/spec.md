## MODIFIED Requirements

### Requirement: Stage-2 Two-Channel module configs are strict and typed
Stage-2 Two-Channel SHALL validate module `config` payloads and `stage2_ab.channel_b` payloads strictly so experiments are reproducible and fail fast on schema drift.

Normative behavior:
- `loss_duplicate_burst_unlikelihood.config` MUST be an empty mapping in v1.
- `token_ce.config` no longer accepts any legacy invalid-structure amplification knob for Channel-B.
- `stage2_ab.channel_b` MUST continue to accept the existing canonical key family plus the new `birth_first` mapping:
  - `insertion_order`
  - `duplicate_control`
  - `triage_posterior`
  - `producer_wait_timeout_s`
  - `ddp_phase_timeout_s`
  - `invalid_rollout_policy`
  - `pseudo_positive`
  - `birth_first`
- `stage2_ab.channel_b.pseudo_positive` MUST be a typed mapping and MUST accept only:
  - `enabled`
  - `coord_weight`
- `stage2_ab.channel_b.birth_first` MUST be a typed mapping and MUST accept only:
  - `enabled`
  - `continue_over_eos_weight`
  - `continue_over_eos_margin`
- `stage2_ab.channel_b.triage_posterior` MUST accept only:
  - `num_rollouts`
  - `explorer_temperature`
  - `explorer_top_p`
  - `explorer_top_k`
  - `unlabeled_consistent_iou_threshold`
  - `recovered_ground_truth_weight_multiplier`
- `stage2_ab.channel_b.duplicate_control` MUST remain unchanged by this change and MUST continue to accept only:
  - `iou_threshold`
  - `center_radius_scale`
- when `stage2_ab.channel_b.birth_first.enabled=true`:
  - `stage2_ab.channel_b.pseudo_positive.enabled` MUST be `false`,
  - `stage2_ab.channel_b.triage_posterior.num_rollouts` MUST be `2`,
  - `stage2_ab.pipeline.objective[name=token_ce].config.rollout_global_prefix_struct_ce_weight` MUST be greater than `0`
- Unknown keys in a module `config` or in `stage2_ab.channel_b` MUST fail fast with actionable diagnostics.

#### Scenario: Non-empty loss_duplicate_burst_unlikelihood.config fails fast
- **WHEN** `stage2_ab.pipeline.objective[*].name=loss_duplicate_burst_unlikelihood`
- **AND** its `config` mapping contains any key
- **THEN** configuration parsing fails fast
- **AND** the error indicates `loss_duplicate_burst_unlikelihood.config` must be empty for v1.

#### Scenario: Birth-first mode rejects multi-view pseudo-positive configuration
- **WHEN** a Stage-2 AB config sets `stage2_ab.channel_b.birth_first.enabled: true`
- **AND** either `stage2_ab.channel_b.pseudo_positive.enabled: true` or `stage2_ab.channel_b.triage_posterior.num_rollouts != 2`
- **THEN** configuration parsing fails fast
- **AND** the error explains that birth-first mode is the anchor-plus-one-explorer decision profile.

#### Scenario: Birth-first mode requires non-zero rollout prefix structure weight
- **WHEN** a Stage-2 AB config sets `stage2_ab.channel_b.birth_first.enabled: true`
- **AND** `stage2_ab.pipeline.objective[name=token_ce].config.rollout_global_prefix_struct_ce_weight <= 0`
- **THEN** configuration parsing fails fast
- **AND** the error explains that birth-first mode requires active rollout-prefix structure CE.

### Requirement: Hybrid objective preserves Channel-A anchoring and uses clean-prefix Channel-B supervision
The Stage-2 AB trainer MUST compute a hybrid objective with:

Channel-A:
- **Token CE anchor at the GT teacher-forced forward**:
  - CE on non-coord tokens MUST be computed from the GT-context logits.
  - Coord tokens MUST NOT contribute to CE, to avoid double-supervision.
- **Geometry + distribution regularizers from the same single-pass Channel-A
  logits**:
  - Geometry losses and any distribution-level losses MUST be computed from the
    supported single-pass Channel-A logits, not from a deprecated final
    self-context pass.

Channel-B:
- **Clean-prefix positive supervision**:
  - the positive teacher-forced prefix MUST be canonical serialization of `accepted_objects_clean`,
  - matched clean prefix objects MUST receive structure-only CE,
  - retained unmatched anchor objects MUST be partitioned into support-positive retained shields versus neutral retained shields using the current anchor/explorer association contract,
  - when `stage2_ab.channel_b.birth_first.enabled=false`, generic unmatched clean extras MAY remain in the clean prefix as context but MUST remain neutral,
  - when `stage2_ab.channel_b.birth_first.enabled=true`, support-positive retained shields MUST remain in the clean prefix and MUST receive structure-only CE through the shared rollout-prefix structure surface,
  - neutral retained shields MAY remain in the clean prefix as context but MUST remain outside positive desc CE and outside positive geometry/coord supervision,
  - FN objects MUST be appended to the clean target and receive structure+desc CE,
  - recovered GT objects MUST stay on that same FN path with the configured recovered-GT weight multiplier.
- **FP-neutral geometry**:
  - geometry losses MUST be computed for matched clean prefix objects and FN-injected objects,
  - when `stage2_ab.channel_b.birth_first.enabled=true`, support-positive retained shields MUST NOT receive positive geometry/coord supervision,
  - when `stage2_ab.channel_b.birth_first.enabled=false`, pseudo-positive-selected or support-weighted retained-shield behavior MAY still contribute positive geometry/coord supervision as defined by the active pseudo-positive contract,
  - neutral retained shields MUST NOT receive geometric gradients.
- **Birth-first continue calibration**:
  - when `stage2_ab.channel_b.birth_first.enabled=true` and `stage2_ab.channel_b.birth_first.continue_over_eos_weight > 0`, recovered-GT boundaries MUST contribute a local continue-over-EOS margin term under the same one-forward Channel-B target realization,
  - the margin MUST compare the next-token log-probability of `EOS` against the first canonical continuation token of the recovered object serialization as defined by the unified loss registry requirement for `continue_over_eos_margin`.
- **Duplicate-ul supervision**:
  - duplicate-certified continuations MUST be removed from the positive clean prefix,
  - duplicate UL MUST target the first true LCP-divergence token relative to the clean continuation at the same clean boundary,
  - same-boundary duplicates that share the same divergence token MUST collapse to one UL term.
- **Closure supervision stays on**:
  - the outermost JSON closure `}` and `<|im_end|>` MUST remain CE-supervised,
  - if closure-marker bookkeeping becomes ambiguous after the clean target is built, the sample MUST stay on the deterministic FN-tail fallback path rather than being dropped.

Configurable desc supervision (both channels):
- Desc CE weights MUST be expressed via the declared pipeline module configs:
  - Channel-A: `stage2_ab.pipeline.objective[name=token_ce].config.desc_ce_weight`
  - Channel-B (FN tail): `stage2_ab.pipeline.objective[name=token_ce].config.rollout_fn_desc_weight`

Bbox geometry losses (both channels) are computed from coord distributions:
- The trainer MUST decode coordinates from coord-token distributions via CoordExp expectation decoding (not argmax):
  - Let bins `k ∈ {0..999}` correspond to the coord-token sub-vocabulary.
  - Let `p(k)` be the softmax distribution over these 1000 bins for a coord slot, taken from the standard causal shift:
    - For a coord token at input position `p`, `p(k)` MUST be computed from logits at position `p-1` (consistent with CE).
  - The decoded normalized coordinate MUST be: `c_hat = Σ_k p(k) * (k/999)` in `[0, 1]`.
- The trainer MUST compute bbox losses in normalized coordinate space `[0, 1]`:
  - GT bbox ints in `[0, 999]` MUST be converted to floats by dividing by `999`.
  - Predicted bbox coords MUST be the decoded normalized floats from `c_hat` above.
- Geometry loss MUST use logits from:
  - the single Channel-A GT-anchor forward, and
  - the Channel-B clean-prefix teacher-forced logits.

Loss form (normative):
- The trainer MUST use SmoothL1 (Huber) + CIoU as the bbox regression terms.
- The trainer MUST NOT use GIoU in Stage-2 AB.

Numerical stability (normative):
- The trainer MUST canonicalize predicted boxes before CIoU:
  - `(x1,x2) := (min(x1,x2), max(x1,x2))`, `(y1,y2) := (min(y1,y2), max(y1,y2))`.
- The trainer MUST ensure the geometry losses do not produce NaNs/Infs, including early training when predictions are degenerate.

#### Scenario: Birth-first support-positive retained anchor gets structure-only positive supervision
- **GIVEN** `stage2_ab.channel_b.birth_first.enabled: true`
- **AND** an unmatched anchor object is retained with non-zero explorer support
- **WHEN** Channel-B CE masks are built for the clean-prefix target
- **THEN** that retained anchor participates in rollout-prefix structure CE
- **AND** it contributes no extra desc CE
- **AND** it contributes no positive geometry/coord supervision.

#### Scenario: Recovered boundary adds continue-over-EOS calibration without a second forward
- **GIVEN** `stage2_ab.channel_b.birth_first.enabled: true`
- **AND** `stage2_ab.channel_b.birth_first.continue_over_eos_weight > 0`
- **AND** a GT object is missed in anchor and recovered by the explorer
- **WHEN** Channel-B realizes the final edited target
- **THEN** the recovered object still uses the ordinary weighted FN-injection path
- **AND** the same one-forward realization emits one local continue-over-EOS calibration term for that recovered boundary
- **AND** no second teacher-forced payload is created.

Birth-first retained-anchor bucket contract:

| Bucket | Definition | Struct CE | Desc CE | Positive geo/coord | Duplicate UL target |
| --- | --- | --- | --- | --- | --- |
| `matched_clean` | clean accepted anchor object matched to GT | yes | no | yes | no |
| `support_positive_shielded` | retained unmatched anchor object with a one-to-one associated explorer object satisfying `IoU >= unlabeled_consistent_iou_threshold` | yes in birth-first | no | no in birth-first | no |
| `neutral_shielded` | retained unmatched anchor object without such associated explorer support | context-only | no | no | no |
| `pseudo_positive_selected` | promoted unmatched anchor object under non-birth-first pseudo-positive mode | yes | no | yes under pseudo-positive contract | no |
| `cluster_demoted` | non-birth-first pseudo-positive candidate demoted after overlap clustering | context-only | no | no | no |
| `duplicate` | duplicate-certified continuation removed from the clean prefix | no | no | no | yes |
| `recovered_gt` / `fn` | GT object injected through the FN path | yes | yes | yes | no |

### Requirement: Recovered GT objects stay on the FN injection path with higher weight
The canonical birth-first contract SHALL treat recovered GT objects as weighted FN injections, not as a second teacher trajectory.

Normative behavior:
- `recovered GT` means “missed in anchor accepted-clean matching and hit in explorer accepted-clean matching,”
- recovered GT objects MUST remain on the same FN injection path used by ordinary FN objects,
- the configured `recovered_ground_truth_weight_multiplier` MUST increase their desc+geo+coord supervision weight relative to ordinary FN objects,
- when `stage2_ab.channel_b.birth_first.enabled=true` and `stage2_ab.channel_b.birth_first.continue_over_eos_weight > 0`, each recovered GT object MUST also create one local continue-over-EOS margin target at the clean boundary immediately before its injected continuation,
- recovered-prefix distillation MUST NOT be part of the canonical contract,
- no separate explore-prefix teacher-forced pass is introduced in this change.

#### Scenario: Recovered GT object uses weighted FN injection and local continue calibration
- **WHEN** a GT object is missed in anchor and hit in explorer
- **AND** birth-first continue-over-EOS is enabled
- **THEN** it is appended through the normal FN-injection path
- **AND** it receives the configured recovered-FN positive weight
- **AND** it contributes one local continue-over-EOS margin target at its recovery boundary
- **AND** no separate explore-prefix teacher-forced pass is created.

### Requirement: Retained unmatched anchor objects remain prefix-visible with explicit pseudo-positive supervision subsets
Retained unmatched anchor objects MAY remain in the clean prefix, but the Channel-B contract SHALL distinguish between selected pseudo-positive anchors, support-positive retained shields, neutral retained shields, and cluster-demoted shielded anchors.

Normative behavior:
- retained unmatched anchor objects MAY participate in global rollout-prefix struct masks when `token_ce.config.rollout_global_prefix_struct_ce_weight > 0`,
- when `stage2_ab.channel_b.birth_first.enabled=false`, selected pseudo-positive anchors MUST receive positive bbox/coord supervision using their retained anchor coordinates and the configured pseudo-positive weight,
- when `stage2_ab.channel_b.birth_first.enabled=false`, support-positive retained shields that are not cluster-demoted MAY receive support-rate-weighted bbox/coord supervision,
- `support_positive_shielded` means the retained unmatched anchor object has a one-to-one associated explorer object satisfying `IoU >= unlabeled_consistent_iou_threshold` under the canonical anchor/explorer association,
- `neutral_shielded` means no such associated explorer object exists,
- `cluster_demoted` remains a pseudo-positive-only subset and MUST NOT be introduced as a separate birth-first positive bucket,
- when `stage2_ab.channel_b.birth_first.enabled=true`, support-positive retained shields MUST participate in the global rollout-prefix structure CE surface,
- when `stage2_ab.channel_b.birth_first.enabled=true`, support-positive retained shields MUST remain outside positive bbox/coord supervision groups and MUST NOT create extra positive desc targets,
- cluster-demoted pseudo-positive candidates and other neutral retained shields MUST stay outside bbox/coord supervision groups,
- retained unmatched anchor objects MUST NOT create extra positive desc targets,
- retained unmatched anchor objects MAY remain visible in the final clean prefix as context.

#### Scenario: Support-positive retained shield is structure-first in birth-first mode
- **WHEN** `stage2_ab.channel_b.birth_first.enabled=true`
- **AND** an unmatched anchor object is retained with non-zero explorer support
- **THEN** it may remain in the edited clean prefix
- **AND** it participates in the global rollout-prefix structure CE surface
- **AND** it contributes no positive bbox/coord supervision
- **AND** it contributes no extra positive desc CE.

#### Scenario: Neutral retained shield stays context-only in birth-first mode
- **WHEN** `stage2_ab.channel_b.birth_first.enabled=true`
- **AND** a retained unmatched anchor object has no positive explorer support
- **THEN** it may remain in the edited clean prefix as context
- **AND** it contributes no positive geo/coord supervision
- **AND** it contributes no extra positive desc CE
- **AND** it does not create duplicate-ul positives.
