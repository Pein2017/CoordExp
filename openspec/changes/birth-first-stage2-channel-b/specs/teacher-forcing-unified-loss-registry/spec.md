## ADDED Requirements

### Requirement: Continue-over-EOS calibration is a canonical boundary-local rollout component
The unified loss registry SHALL support `continue_over_eos_margin` as an optional canonical rollout-text component for birth-first Channel-B calibration.

Normative behavior:
- `continue_over_eos_margin` MUST apply only in `context=rollout`.
- Eligible targets MUST be recovered-GT boundaries where:
  - the anchor misses a GT object,
  - the explorer recovers that GT object,
  - `stage2_ab.channel_b.birth_first.enabled=true`,
  - and `stage2_ab.channel_b.birth_first.continue_over_eos_weight > 0`.
- For each eligible boundary `b`, let `c1` be the first canonical continuation token of the recovered object continuation under the same clean boundary context.
- Let `s_cont = log p(c1 | b)` and `s_eos = log p(EOS | b)` from the same next-token distribution at boundary `b`.
- With margin `m = stage2_ab.channel_b.birth_first.continue_over_eos_margin`, the per-boundary term MUST be `max(0, m - (s_cont - s_eos))`.
- The component MUST emit at most one term per recovered boundary.
- The component MUST aggregate as a mean over eligible recovered boundaries for the current optimizer step.
- If no eligible recovered boundary exists for a step, the component MAY emit `0.0` and MUST NOT fail the step.
- This change MUST NOT introduce a new authored objective-pipeline module for `continue_over_eos_margin`; the component MUST project through the existing rollout-text surface owned by the current Channel-B text objective machinery.

#### Scenario: One recovered boundary yields one continue-over-EOS term
- **WHEN** a Stage-2 Channel-B sample has one recovered GT boundary under birth-first mode
- **THEN** the rollout-context registry includes one `continue_over_eos_margin` contribution for that boundary
- **AND** the contribution is computed without a second teacher-forced forward.

#### Scenario: No recovered boundary produces zero contribution without failure
- **WHEN** a Stage-2 Channel-B step has no recovered GT boundary eligible for birth-first stop calibration
- **THEN** `continue_over_eos_margin` contributes `0.0` for that step
- **AND** the absence of eligible boundaries does not invalidate additive projection or optimizer-step reduction.

## MODIFIED Requirements

### Requirement: Channel-B rollout context is explicit, triage-aware, and EOS-enforced
For Stage-2 Channel-B (`context=rollout`), the rollout-context contract SHALL be defined over the clean accepted sequence rather than the raw rollout prefix.

Normative rollout object subsets:
- `matched_clean`: clean accepted objects matched to GT.
- `pseudo_positive_selected`: unmatched clean accepted anchor objects promoted by pseudo-positive triage.
- `support_positive_shielded`: unmatched clean accepted anchor objects retained in the prefix with positive explorer support under the current association contract.
- `neutral_shielded`: unmatched clean accepted anchor objects retained in the prefix as context without positive birth evidence.
- `duplicate`: duplicate-certified objects removed from the positive clean prefix.
- `recovered_fn`: GT objects injected into the same top-level `objects[]` container because the explorer hit them and the anchor missed them.
- `fn`: remaining GT objects injected into the same top-level `objects[]` container for supervision.

Normative behavior:
- `duplicate` objects MUST NOT appear in the positive teacher-forced prefix.
- retained clean-prefix objects MAY receive global rollout-prefix structure supervision as defined by the Channel-B contract.
- `matched_clean` objects receive positive geometry/coord supervision as defined by the Channel-B contract.
- `pseudo_positive_selected` objects receive positive geometry/coord supervision using their retained anchor coordinates and the configured pseudo-positive weight when pseudo-positive mode is active.
- `support_positive_shielded` objects:
  - MUST receive structure-only positive supervision in birth-first mode,
  - MAY receive support-rate-weighted geometry/coord supervision only under the non-birth-first pseudo-positive contract,
  - MUST remain outside extra desc-positive supervision in both modes.
- `neutral_shielded` objects MAY remain in the clean prefix as context but MUST remain outside positive geometry/coord supervision, continue-over-EOS positives, duplicate-ul positives, and extra desc-positive supervision.
- `recovered_fn` objects remain positively supervised on the FN path and MAY also drive `continue_over_eos_margin` when birth-first mode is enabled.
- `fn` objects remain positively supervised.
- Closure / EOS remain supervised.

#### Scenario: Duplicate-certified objects are removed from the positive prefix
- **WHEN** a rollout object is classified as `duplicate`
- **THEN** it does not contribute to the positive teacher-forced prefix
- **AND** it is represented only through duplicate-ul supervision and diagnostics.

#### Scenario: Support-positive shielded object is structure-only in birth-first mode
- **WHEN** `context=rollout` is built under birth-first mode
- **AND** a retained unmatched anchor object has positive explorer support
- **THEN** it is classified as `support_positive_shielded`
- **AND** it participates in rollout-prefix structure supervision
- **AND** it contributes no extra desc-positive supervision
- **AND** it contributes no positive geometry/coord supervision.
