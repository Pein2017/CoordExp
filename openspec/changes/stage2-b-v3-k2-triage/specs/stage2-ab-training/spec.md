# stage2-ab-training Specification (Delta)

## MODIFIED Requirements

### Requirement: Stage-2 AB Channel-B uses the canonical K=2 anchor/explorer triage contract
When `custom.trainer_variant: stage2_two_channel`, the canonical Channel-B contract SHALL build its clean teacher-forced target from two rollout views:

- one anchor rollout using greedy / deterministic decoding,
- one explorer rollout using stochastic decoding configured under `stage2_ab.channel_b.v3_k2`.

Normative behavior:

- each rollout MUST independently reuse the existing bounded salvage + strict record acceptance + bbox-valid filtering + sequential dedup + Hungarian matching path,
- GT-backed semantics MUST inherit the existing Channel-B accepted-clean Hungarian + gating contract,
- the final positive target MUST be built by editing the **anchor** clean sequence rather than rebuilding a union order,
- explorer-only non-GT-backed objects MUST be treated as dead by default in v1,
- a GT hit found only on the explorer side MUST project to `recovered_fn`, not to anchor retention.

#### Scenario: Channel-B builds the final target from the anchor clean sequence
- **GIVEN** anchor and explorer rollouts were both produced for a Channel-B sample
- **WHEN** the trainer constructs the teacher-forced target
- **THEN** it starts from the anchor clean accepted sequence
- **AND** it preserves anchor order for retained objects
- **AND** it does not rebuild a union ordering over anchor and explorer objects.

#### Scenario: Explorer-only GT hit does not keep a bad anchor object positive
- **GIVEN** an anchor/explorer pair-or-singleton record where the anchor side misses GT and the explorer side matches GT
- **WHEN** the trainer projects triage evidence into training actions
- **THEN** the outcome is `recovered_fn`
- **AND** the bad anchor object is not kept as an anchor GT-backed positive.

### Requirement: Stage-2 AB Channel-B v3-specific knobs are typed and grouped
The Stage-2 AB config SHALL expose v3-specific K=2 rollout knobs under `stage2_ab.channel_b.v3_k2`.

Normative behavior:

- `stage2_ab.channel_b.v3_k2` MUST be a typed mapping,
- the mapping MUST accept only:
  - `explorer_temperature`
  - `explorer_top_p`
  - `explorer_top_k`
  - `consistent_iou_threshold`
  - `recovered_fn_weight`
- unknown keys under `stage2_ab.channel_b.v3_k2` MUST fail fast.

#### Scenario: Unknown v3_k2 key fails fast
- **WHEN** a Stage-2 AB config includes an unknown key under `stage2_ab.channel_b.v3_k2`
- **THEN** config loading fails fast with the full dotted path.

### Requirement: Recovered GT objects stay on the FN injection path with higher weight
The canonical v1 v3 contract SHALL treat recovered GT objects as weighted FN injections, not as a second teacher trajectory.

Normative behavior:

- `recovered GT` means “missed in anchor accepted-clean matching and hit in explorer accepted-clean matching,”
- recovered GT objects MUST remain on the same FN injection path used by ordinary FN objects,
- the configured `recovered_fn_weight` MUST increase their desc+geo+coord supervision weight relative to ordinary FN objects,
- recovered-prefix distillation MUST NOT be part of the canonical v1 contract.

#### Scenario: Recovered GT object uses weighted FN injection
- **WHEN** a GT object is missed in anchor and hit in explorer
- **THEN** it is appended through the normal FN-injection path
- **AND** it receives the configured recovered-FN positive weight
- **AND** no separate explore-prefix teacher-forced pass is created.

## ADDED Requirements

### Requirement: Channel-B v3 uses deterministic one-to-one anchor/explorer association
The canonical v1 v3 contract SHALL associate anchor and explorer accepted objects deterministically before projecting triage actions.

Normative behavior:

- candidate cross-rollout pairs MUST be scored by IoU,
- only pairs with `IoU >= consistent_iou_threshold` are eligible,
- the chosen association MUST be one-to-one and maximize IoU,
- if multiple assignments achieve the same maximum total IoU, the chosen assignment MUST be the one whose sorted pair list `[(anchor_index, explorer_index), ...]` is lexicographically smallest.

#### Scenario: Crowded-scene association is stable under tie conditions
- **WHEN** two eligible anchor/explorer candidate pairs have identical IoU
- **THEN** the selected association is resolved by the canonical lexicographic assignment tie-break rule rather than container ordering or hash iteration.

### Requirement: Channel-B v3 uses one merged teacher-forced forward
The canonical v1 v3 contract SHALL realize `L(clean_anchor) + L(explore-derived corrections)` through one merged teacher-forced forward on the edited anchor target.

Normative behavior:

- the trainer MUST run one teacher-forced forward on the final edited target,
- positive, weighted-FN, and dead-anchor UL terms MUST be derived from that same forward,
- the trainer MUST NOT require a second explore teacher-forced payload in the canonical v1 contract.

#### Scenario: Single-forward v3 target realization
- **WHEN** a Channel-B v3 sample is prepared
- **THEN** all loss terms are derived from a single teacher-forced forward over the edited anchor target
- **AND** no second teacher-forced explore payload is required.

### Requirement: Shielded anchor objects remain neutral context
Anchor objects triaged as shielded MAY remain in the clean prefix, but they MUST remain neutral with respect to positive supervision.

Normative behavior:

- shielded anchor objects MUST stay outside matched-prefix struct masks,
- shielded anchor objects MUST stay outside bbox/coord supervision groups,
- shielded anchor objects MUST NOT create extra positive desc targets,
- shielded anchor objects MAY remain visible in the final clean prefix as context.

#### Scenario: Shielded anchor object stays in prefix but produces no positive supervision
- **WHEN** an anchor object is classified as shielded
- **THEN** it may remain in the edited clean prefix
- **AND** it contributes no positive CE, bbox, or coord supervision.
