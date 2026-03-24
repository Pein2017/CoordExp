# teacher-forcing-unified-loss-registry Specification (Delta)

## MODIFIED Requirements

### Requirement: Channel-B rollout context is FP-neutral and EOS-enforced
For Stage-2 Channel-B (`context=rollout`), the rollout-context contract SHALL be defined over the anchor-edited clean sequence rather than the raw rollout prefix.

Normative rollout object subsets:

- `matched_clean`: anchor clean objects matched to GT.
- `pseudo_positive`: selected unmatched anchor objects promoted for coord-positive supervision while remaining desc-neutral.
- `shielded_anchor`: unmatched anchor objects kept as coord-neutral context.
- `dead_anchor`: unmatched anchor objects removed from the positive prefix.
- `fn`: GT objects injected into the same top-level `objects[]` container for supervision.
- `recovered_fn`: injected FN objects that were missed in anchor and hit in at least one explorer view.

Normative behavior:

- `dead_anchor` objects MUST NOT appear in the positive teacher-forced prefix,
- retained prefix objects MAY receive global rollout-prefix structure supervision as defined by the Channel-B contract,
- `matched_clean` objects receive positive geometry/coord supervision as defined by the Channel-B contract,
- `pseudo_positive` objects MUST remain in the edited anchor prefix and MUST be exempt from blanket FP-neutral handling only for positive coord-side supervision,
- `pseudo_positive` objects MUST use their own retained anchor coordinates as the target source for bbox/coord supervision,
- `pseudo_positive` objects MUST NOT create desc CE,
- `shielded_anchor` objects MAY remain in the clean prefix as context but MUST remain outside coord supervision and desc supervision,
- `fn` and `recovered_fn` objects remain positively supervised through the FN tail,
- closure / EOS remain supervised.

#### Scenario: Pseudo-positive object uses anchor geometry and no new desc CE
- **WHEN** an unmatched anchor object is selected as `pseudo_positive`
- **THEN** it remains in the edited anchor prefix
- **AND** its bbox/coord supervision target is derived from that anchor object's own canonical coordinates
- **AND** it contributes no desc CE
- **AND** it may still participate in global rollout-prefix structure CE.

### Requirement: Geometry loss (`geo`) uses canonicalized boxes and a stable decomposition
The unified loss registry SHALL treat selected pseudo-positive Channel-B anchors as part of the rollout geometry-supervised set when the pseudo-positive contract is enabled.

Normative behavior:

- when `stage2_ab.channel_b.pseudo_positive.enabled=false`, Stage-2 Channel-B rollout `geo` semantics remain unchanged and aggregate over `matched_clean + fn`,
- when `stage2_ab.channel_b.pseudo_positive.enabled=true`, Stage-2 Channel-B rollout `geo` semantics MUST aggregate over `matched_clean + pseudo_positive + fn`,
- `shielded_anchor`, `dead_anchor`, and explorer-only non-GT-backed objects MUST remain outside the rollout `geo` supervised set,
- selected pseudo-positive anchors MUST use their retained anchor coordinates as the rollout `geo` target source.

#### Scenario: Enabled pseudo-positive expands the rollout geometry-supervised set only on the coord side
- **WHEN** `stage2_ab.channel_b.pseudo_positive.enabled=true`
- **THEN** Channel-B rollout `geo` includes `pseudo_positive` objects alongside `matched_clean` and `fn`
- **AND** shielded or dead unmatched anchors remain outside rollout `geo` supervision.

### Requirement: Gate terms respect context-specific masking and pseudo-positive coord-positive participation
The unified loss registry SHALL keep pseudo-positive spans coord-positive, desc-neutral, and aligned with the shared global rollout-prefix structure CE surface when coord regularization gate terms are enabled.

Normative behavior:

- in `context=rollout`, selected `pseudo_positive` anchors MUST contribute to coord-side `coord_reg` sub-terms only through the same positive coord-supervised bbox-group surface used by `matched_clean` and `fn`,
- selected `pseudo_positive` anchors MAY contribute to `coord_gate` where that sub-term is defined on coord-supervised positions,
- selected `pseudo_positive` anchors MUST NOT contribute to `text_gate` or desc-side supervision,
- shielded and dead unmatched anchors MUST remain outside both `coord_gate` and `text_gate`.

#### Scenario: Pseudo-positive spans are coord-positive and desc-neutral under coord regularization
- **WHEN** `stage2_ab.channel_b.pseudo_positive.enabled=true`
- **AND** coord-regularization gate terms are enabled for Channel-B rollout context
- **THEN** selected pseudo-positive anchors may contribute to coord-side gate terms
- **AND** they do not contribute to `text_gate` or desc-side supervision beyond the shared global rollout-prefix structure CE surface.

### Requirement: Rollout-context semantics are explicit, auditable, and coherent across trainers
The unified loss registry SHALL treat anchor-edited rollout semantics as the canonical Channel-B rollout contract.

Normative behavior:

- Channel-B positive masks are built from the clean teacher-forced target, not the raw rollout prefix,
- shielded unmatched extras MAY participate in the global rollout-prefix struct masks when that token-ce weight is enabled, but MUST stay outside desc supervision and positive bbox/coord groups,
- selected pseudo-positive anchors MUST populate positive bbox/coord groups and may participate in the shared global rollout-prefix struct masks,
- duplicate-like dead-anchor suppression MUST remain boundary-local and explicit rather than encoded through hidden token-ce behavior.

#### Scenario: Shielded and pseudo-positive unmatched anchors diverge cleanly
- **WHEN** a rollout-context sample contains both `shielded_anchor` and `pseudo_positive` unmatched anchor objects
- **THEN** the shielded subset remains context-only
- **AND** only the pseudo-positive subset enters positive bbox/coord supervision
- **AND** both retained subsets may still share the global rollout-prefix structure CE surface.
