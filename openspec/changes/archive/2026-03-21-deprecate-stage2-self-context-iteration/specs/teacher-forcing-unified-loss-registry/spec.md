# teacher-forcing-unified-loss-registry Specification (Delta)

## MODIFIED Requirements

### Requirement: Canonical contexts, token types, and loss component names are shared
The system SHALL define canonical teacher-forcing contexts and token types that
are used consistently across Stage-2 two-channel and rollout-aligned training.

Normative contexts:

- `gt`: pure GT teacher forcing.
- `rollout`: Channel-B clean-prefix teacher-forced logits under clean accepted
  prefix + FN injection.

Normative token types (mutually exclusive):

- `struct`: JSON syntax + keys/punctuation and other non-desc, non-coord
  content.
- `desc`: free-text tokens inside an object’s `desc` value span.
- `coord`: coord-vocabulary tokens `<|coord_k|>`.
- `eos`: end token `<|im_end|>` (Qwen3-VL).

Normative minimum canonical loss component names (metrics use these names when
emitted):

- `struct_ce`
- `desc_ce`
- `loss_dead_anchor_suppression`
- `geo`
- `coord_reg`

NOTE (logging contract):

- These are canonical registry component names (often surfaced as
  `loss/<component>` keys inside pipeline-internal metrics).
- Trainers MAY choose to omit raw component keys from the training log to
  reduce redundancy and instead emit only objective-weighted provenance keys
  under `loss/<provenance>/<atom>` (see `trainer-metrics-components`).

#### Scenario: Stage-2 two-channel no longer defines self_context as a registry context
- **WHEN** registry contexts are enumerated for active Stage-2 training
- **THEN** the supported contexts include `gt` and `rollout`
- **AND** `self_context` is not part of the active context contract.

### Requirement: Gate terms respect context-specific masking (FP-neutral and desc-disabled spans)
Gate terms MUST respect the same masking semantics as CE for the current
context.

Normative behavior:

- In `context=rollout`:
  - FP spans MUST NOT contribute to `coord_gate` or `text_gate`.
  - Desc-disabled behavior MUST be respected (if `desc` supervision is
    masked/disabled, `text_gate` MUST NOT apply to those spans).
- In `context=gt`:
  - Gate terms MUST follow the supervised GT token positions for the current
    objective configuration.
  - Implementations MAY ignore `type=eos` when the corresponding gate term is
    not defined on EOS positions.

#### Scenario: GT-context text gating follows ordinary supervised positions
- **WHEN** Stage-2 Channel-A computes gate terms under `context=gt`
- **THEN** gate masking follows the supervised GT token positions
- **AND** no struct-only self-context masking rule is applied.

### Requirement: Loss component names and contexts are canonical and shared
The system SHALL define canonical loss component names and context types that
are used consistently across Stage-1 and Stage-2 code paths.

Normative contexts:

- `gt`: pure GT teacher forcing logits and targets (Stage-1; also the Channel-A
  supervision surface for Stage-2 two-channel).
- `rollout`: Channel-B clean-prefix teacher-forced logits under clean accepted
  prefix + FN injection.

Normative loss component names (minimum set; can be extended):

- `struct_ce`: token cross entropy on structure tokens, including EOS
  enforcement (EOS is a distinct token type but its CE contribution is
  accounted under `struct_ce`).
- `desc_ce`: token cross entropy on description tokens.
- `loss_dead_anchor_suppression`: duplicate-certified unlikelihood over
  clean-boundary divergence tokens in Channel-B rollout context.
- `coord_token_ce`: token cross entropy on coord vocabulary tokens (optional;
  typically GT context only).
- `coord_reg`: coord-subspace regularizers computed from
  logits/probabilities (optional; includes distribution/ordinal terms on coord
  positions and vocab-partition gate terms).
- `geo`: bbox-level geometry loss computed on decoded boxes.

Normative behavior:

- A single implementation of the above components MUST be reused across
  stages/channels (no duplicated definitions).
- Module pipelines and trainers MUST use these stable canonical component names
  for registry identity and objective semantics.
- Stage-2 two-channel MUST NOT introduce a separate `self_context` registry
  context for Channel-A.

#### Scenario: Shared naming excludes self_context aliases
- **WHEN** Stage-2 training logs or registry payloads name contexts
- **THEN** Channel-A uses `gt`
- **AND** the naming contract does not rely on `self_context` aliases.

### Requirement: Geometry decode follows the fixed expectation path
The system SHALL use the fixed expectation-decode path for Stage-2 geometry in
active training.

Normative behavior:

- Stage-2 two-channel MUST NOT expose `stage2_ab.coord_decode_mode`.
- Stage-2 rollout-aligned MUST NOT expose `rollout_matching.coord_decode_mode`.
- Active Stage-2 geometry decode uses the expectation path for coord-subspace
  logits.

#### Scenario: Deprecated coord decode toggles are absent
- **WHEN** active Stage-2 geometry config surfaces are enumerated
- **THEN** `stage2_ab.coord_decode_mode` and
  `rollout_matching.coord_decode_mode` are not supported authored keys
- **AND** geometry decode follows the fixed expectation path.

### Requirement: Geometry loss (`geo`) uses canonicalized boxes and a stable decomposition
The system SHALL define geometry loss (`geo`) on decoded continuous boxes in a
way that is:

- stable under near-degenerate boxes,
- compatible with packing (segment-local indices),
- compatible with Channel-B FP-neutral masking.

Normative behavior:

- The system MUST decode 4 coords per box from coord-subspace logits using the
  fixed expectation decode path.
- The system MUST canonicalize decoded boxes before applying geometry loss:
  - `x_lo = min(x1, x2)`, `x_hi = max(x1, x2)`
  - `y_lo = min(y1, y2)`, `y_hi = max(y1, y2)`
  - enforce non-zero size with an `eps` floor where required for CIoU-like
    terms.
- The system MUST implement `geo` as a weighted sum of:
  - SmoothL1 (Huber) on `(x_lo,y_lo,x_hi,y_hi)`,
  - CIoU on the same canonicalized box representation.
- The system MUST aggregate `geo` as a mean over the supervised object set for
  the current context:
  - Stage-2 Channel-A `gt`: identity-aligned GT objects,
  - Stage-2 Channel-B `rollout`: `matched_clean` + `fn` objects (`duplicate`
    and `unmatched_clean` excluded).

#### Scenario: Channel-A geometry loss aggregates over GT objects only
- **WHEN** Stage-2 two-channel computes Channel-A geometry loss
- **THEN** `geo` aggregates over identity-aligned GT objects
- **AND** no final-pass self-context object set is consulted.

## REMOVED Requirements

### Requirement: Channel-A CE anchoring and self-context geometry are separate contexts
This requirement is removed. Stage-2 two-channel no longer uses a separate
final-pass `self_context` Channel-A surface for geometry or optional
format/closure stabilization.

#### Scenario: (removed) Channel-A CE uses A1 while geometry uses self_context
- **WHEN** Stage-2 Channel-A executes
- **THEN** both CE and bbox/coord supervision now come from the supported GT
  context rather than from split GT vs self-context surfaces.

## ADDED Requirements

### Requirement: Stage-2 Channel-A uses GT context only
The unified registry SHALL treat Stage-2 two-channel Channel-A as a single
GT-context supervision surface.

Normative behavior:

- Channel-A token CE, bbox geometry, and coord regularizer supervision MUST use
  `context=gt`.
- Final-pass/self-context-only Channel-A stabilizers are unsupported.
- Channel-A routing MUST NOT depend on `n_softctx_iter` or any equivalent final
  self-context pass count.

#### Scenario: Channel-A geometry uses GT context
- **WHEN** Stage-2 two-channel computes Channel-A bbox or coord supervision
- **THEN** the registry context is `gt`
- **AND** no `self_context` context is constructed.
