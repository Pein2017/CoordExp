# teacher-forcing-unified-loss-registry Specification (Delta)

## MODIFIED Requirements

### Requirement: Channel-B rollout context is defined over anchor-edited triage subsets
For Stage-2 Channel-B (`context=rollout`), the rollout-context contract SHALL be defined over the edited anchor clean sequence produced by the v3 triage stage.

Normative rollout object subsets:

- `anchor_gt_backed`: anchor objects retained as GT-backed
- `shielded_anchor`: anchor objects retained as neutral context
- `dead_anchor`: anchor objects removed from the positive prefix
- `dead_explorer`: explorer-only or unretained explorer evidence used for diagnostics only
- `fn`: GT objects injected through the FN tail
- `recovered_fn`: injected FN objects that were missed in anchor and hit in explorer

Normative behavior:

- `dead_anchor` objects MUST NOT appear in the positive teacher-forced prefix,
- `shielded_anchor` objects MAY remain in the clean prefix as context but MUST remain neutral,
- `anchor_gt_backed` objects continue to follow the current matched-prefix policy (structure supervision plus matched geo/coord path; no new matched-desc CE is implied by this change),
- `recovered_fn` objects remain positively supervised through the FN path with higher desc+geo+coord weight,
- closure / EOS remain supervised.

#### Scenario: Shielded anchor object remains context-only
- **WHEN** an anchor object is classified as shielded
- **THEN** it may remain in the edited clean prefix
- **AND** it contributes no positive CE/geo/coord terms.

### Requirement: loss_dead_anchor_suppression is first-divergence suppression for dead anchor-side continuations
The unified loss registry SHALL treat `loss_dead_anchor_suppression` as the canonical first-divergence local suppression objective for dead anchor-side continuations in v3.

Normative behavior:

- the continuation source is any anchor-side continuation classified as `dead_anchor` by the v3 triage stage,
- the target token remains the first true LCP divergence token relative to the canonical edited clean continuation,
- explorer-only dead objects MUST NOT create a separate explorer-side suppression branch in v1.

#### Scenario: Explorer-only dead object does not create a second suppression branch
- **WHEN** an object exists only in the explorer rollout and is classified as dead
- **THEN** it does not create a separate explore-side UL path in the canonical v1 contract
- **AND** only anchor-side dead continuations may produce `loss_dead_anchor_suppression` targets.

## ADDED Requirements

### Requirement: Recovered FN weighting is part of rollout-context supervision
The rollout-context contract SHALL support higher positive weighting for recovered FN objects while keeping them on the canonical FN tail.

Normative behavior:

- `recovered_fn` objects MUST be identifiable in the rollout-context metadata,
- positive CE/geo/coord terms derived from those objects MUST use the configured recovered-FN weight,
- ordinary FN objects MUST remain supported with their baseline positive weight.

#### Scenario: Recovered FN weight changes only recovered tail objects
- **WHEN** a rollout-context sample contains both ordinary FN objects and recovered FN objects
- **THEN** only the recovered subset uses the higher recovered-FN positive weight across CE/geo/coord
- **AND** ordinary FN objects continue to use the baseline FN positive weight.
