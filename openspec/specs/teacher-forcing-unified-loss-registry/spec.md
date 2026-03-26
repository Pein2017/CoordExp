# teacher-forcing-unified-loss-registry Specification

## Purpose
Define the canonical teacher-forcing contexts, token types, and registry-level loss component naming shared across Stage-2 two-channel and rollout-aligned training.
## Requirements
### Requirement: Canonical contexts, token types, and loss component names are shared
The system SHALL define canonical teacher-forcing contexts and token types that are used consistently across Stage-2 two-channel and rollout-aligned training.

Normative contexts:
- `gt`: pure GT teacher forcing.
- `rollout`: Channel-B clean-prefix teacher-forced logits under clean accepted prefix + FN injection.

Normative token types (mutually exclusive):
- `struct`: JSON syntax + keys/punctuation and other non-desc, non-coord content.
- `desc`: free-text tokens inside an object’s `desc` value span.
- `coord`: coord-vocabulary tokens `<|coord_k|>`.
- `eos`: end token `<|im_end|>` (Qwen3-VL).

Normative minimum canonical loss component names (metrics use these names when emitted):
- `struct_ce`
- `desc_ce`
- `loss_duplicate_burst_unlikelihood`
- `geo`
- `coord_reg`

NOTE (logging contract):
- These are canonical registry component names (often surfaced as `loss/<component>` keys inside pipeline-internal metrics).
- Trainers MAY choose to omit raw component keys from the training log to reduce redundancy and instead emit only objective-weighted provenance keys under `loss/<provenance>/<atom>` (see `trainer-metrics-components`).

#### Scenario: Stage-2 two-channel no longer defines self_context as a registry context
- **WHEN** registry contexts are enumerated for active Stage-2 training
- **THEN** the supported contexts include `gt` and `rollout`
- **AND** `self_context` is not part of the active context contract.

#### Scenario: loss_duplicate_burst_unlikelihood is a canonical registry loss component name
- **WHEN** the clean-prefix Channel-B objective is reported through the unified registry
- **THEN** the duplicate-unlikelihood component is identified canonically as `loss_duplicate_burst_unlikelihood`
- **AND** it is not folded into `struct_ce` or `desc_ce`.

### Requirement: Gate terms are logit-derived and require no new heads
The registry SHALL support two complementary vocab-partition gate sub-terms inside `coord_reg`:
- `coord_gate`: applied at `type=coord` positions, encourages high coord-vocab probability mass.
- `text_gate`: applied at `type=struct|desc` positions, encourages low coord-vocab probability mass (i.e., high non-coord mass).

Normative behavior:
- Gate computation MUST be derived from token logits and MUST NOT require adding new model heads.
- Let `S_all(t) = logsumexp(logits_full[t, :])` and `S_coord(t) = logsumexp(logits_full[t, coord_vocab_ids])`.
  Define `p_coord(t) = exp(S_coord(t) - S_all(t))`.
- `coord_gate` per token: `L = -log(p_coord(t) + eps)`.
- `text_gate` per token: `L = -log(1 - p_coord(t) + eps)`.
- `eps` MUST be a small positive constant to avoid `log(0)` and MUST be applied after clamping the argument into `(0,1)`.

#### Scenario: Text gate increases when coord mass is high at text positions
- **WHEN** a text-position token has `p_coord(t)` close to `1.0`
- **THEN** `text_gate(t)` is large and positive
- **AND** decreasing `p_coord(t)` decreases `text_gate(t)`.

### Requirement: Gate terms respect context-specific masking (FP-neutral and desc-disabled spans)
Gate terms MUST respect the same masking semantics as CE for the current context.

Normative behavior:
- In `context=rollout`:
  - FP spans MUST NOT contribute to `coord_gate` or `text_gate`.
  - Desc-disabled behavior MUST be respected (if `desc` supervision is masked/disabled, `text_gate` MUST NOT apply to those spans).
- In `context=gt`:
  - Gate terms MUST follow the supervised GT token positions for the current
    objective configuration.
  - Implementations MAY ignore `type=eos` when the corresponding gate term is
    not defined on EOS positions.

#### Scenario: GT-context text gating follows ordinary supervised positions
- **WHEN** Stage-2 Channel-A computes gate terms under `context=gt`
- **THEN** gate masking follows the supervised GT token positions
- **AND** no struct-only self-context masking rule is applied.

### Requirement: Canonical loss scalars are mean-like and scale-invariant
All canonical component scalars (registry metrics `loss/<component>` when emitted) MUST be mean-like values that do not scale with packing length or token counts.

Normative behavior:
- `struct_ce` and `desc_ce` MUST be computed as weighted means over supervised tokens:
  - numerator: `sum_t (w_t * CE_t)`
  - denominator: `sum_t w_t` (clamped to a positive epsilon).
- `geo` MUST be a mean over supervised objects/boxes in the relevant context.
- `coord_reg` sub-terms MUST be means over their contributing positions.

#### Scenario: Packed sequences do not change mean-like loss scale
- **WHEN** two packed forwards contain different numbers of supervised tokens but identical per-token distributions
- **THEN** the reported `loss/struct_ce` and `loss/desc_ce` are comparable as mean-like scalars.

### Requirement: Loss component names and contexts are canonical and shared
The system SHALL define canonical loss component names and context types that are used consistently across Stage-1 and
Stage-2 code paths.

Normative contexts:
- `gt`: pure GT teacher forcing logits and targets (Stage-1; also the Channel-A
  supervision surface for Stage-2 two-channel).
- `rollout`: Channel-B clean-prefix teacher-forced logits under clean accepted prefix + FN injection.

Normative loss component names (minimum set; can be extended):
- `struct_ce`: token cross entropy on structure tokens, including EOS enforcement (EOS is a distinct token type but its
  CE contribution is accounted under `struct_ce`).
- `desc_ce`: token cross entropy on description tokens.
- `loss_duplicate_burst_unlikelihood`: duplicate-certified unlikelihood over clean-boundary divergence tokens in Channel-B rollout context.
- `coord_token_ce`: token cross entropy on coord vocabulary tokens (optional; typically GT context only).
- `coord_reg`: coord-subspace regularizers computed from logits/probabilities (optional; includes distribution/ordinal
  terms on coord positions and vocab-partition gate terms).
- `geo`: bbox-level geometry loss computed on decoded boxes.

Normative behavior:
- A single implementation of the above components MUST be reused across stages/channels (no duplicated definitions).
- Module pipelines and trainers MUST use these stable canonical component names for registry identity and objective semantics.
- Public training logs for registry-defined objective modules MUST follow the canonical metric emission contract in
  `trainer-metrics-components` (for example `loss/<provenance>/<atom>` objective atoms), rather than inventing trainer-specific aliases.
- This change MUST update `docs/training/METRICS.md` and `docs/training/STAGE2_RUNBOOK.md` to reflect the
  canonical metric key contract introduced by the unified registry/pipeline.
- Stage-2 two-channel MUST NOT introduce a separate `self_context` registry
  context for Channel-A.

#### Scenario: Shared naming excludes self_context aliases
- **WHEN** Stage-2 training logs or registry payloads name contexts
- **THEN** Channel-A uses `gt`
- **AND** the naming contract does not rely on `self_context` aliases.

### Requirement: Canonical loss scalars are mean-like (scale-invariant)
To make runs comparable across packing, batch sizing, and grad-accum settings, the system SHALL treat all canonical
`loss/<component>` scalars as **mean-like** values (not raw sums).

Normative behavior:
- `struct_ce`, `desc_ce`, and `coord_token_ce` MUST be computed as a weighted mean over supervised tokens:
  - numerator: `sum_t (w_t * CE_t)`,
  - denominator: `sum_t w_t` (clamped to a positive epsilon to avoid NaNs),
  - where `w_t` reflects token-type membership and explicit config weights (e.g., `desc_ce_weight`).
- `geo` MUST be a mean over supervised objects/boxes in the relevant context.
- `coord_reg` sub-terms MUST be means over the contributing token positions (coord positions for distribution terms;
  token-type appropriate positions for gate terms).
- Any sum/count values needed to form a mean (e.g., numerators/denominators) MUST either:
  - remain internal-only (not emitted), or
  - be emitted only under explicit counter-like names (suffix `*_sum`, `*_count`, `*_num`, `*_den`), never as
    `loss/<component>`.

#### Scenario: Loss scalars do not scale with token count
- **GIVEN** two packed forwards with different numbers of segments/tokens
- **WHEN** `loss/struct_ce` is logged
- **THEN** it is comparable as a per-token mean-like scalar (not proportional to total supervised tokens).

### Requirement: Token-type partition is explicit and deterministic
The system SHALL partition supervised tokens into mutually exclusive token types.

Normative token types:
- `struct`: JSON syntax + keys/punctuation and other non-desc, non-coord content.
- `desc`: free-text tokens inside an object’s `desc` value span.
- `coord`: coord-vocabulary tokens `<|coord_k|>`.
- `eos`: end token `<|im_end|>` (Qwen3-VL); treated as its own type.

Normative behavior:
- Token-type assignment MUST be deterministic for a given tokenizer + encoded sequence + meta/spans.
- EOS MUST NOT be double-counted as both `struct` and `eos`.

#### Scenario: EOS token is counted exactly once
- **WHEN** token-type masks are built for a sequence containing `<|im_end|>`
- **THEN** the EOS token receives `type=eos`
- **AND** it receives `type!=struct` (no double counting).

### Requirement: Channel-B rollout context is explicit, triage-aware, and EOS-enforced
For Stage-2 Channel-B (`context=rollout`), the rollout-context contract SHALL be defined over the clean accepted sequence rather than the raw rollout prefix.

Normative rollout object subsets:
- `matched_clean`: clean accepted objects matched to GT.
- `pseudo_positive_selected`: unmatched clean accepted anchor objects promoted by pseudo-positive triage.
- `shielded_clean`: unmatched clean accepted anchor objects retained in the prefix as context.
- `duplicate`: duplicate-certified objects removed from the positive clean prefix.
- `fn`: GT objects injected into the same top-level `objects[]` container for supervision.

Normative behavior:
- `duplicate` objects MUST NOT appear in the positive teacher-forced prefix.
- retained clean-prefix objects MAY receive global rollout-prefix structure supervision as defined by the Channel-B contract.
- `matched_clean` objects receive positive geometry/coord supervision as defined by the Channel-B contract.
- `pseudo_positive_selected` objects receive positive geometry/coord supervision using their retained anchor coordinates and the configured pseudo-positive weight.
- support-positive `shielded_clean` objects that are not cluster-demoted MAY receive support-rate-weighted geometry/coord supervision under pseudo-positive mode.
- cluster-demoted or otherwise neutral `shielded_clean` objects MAY remain in the clean prefix as context but MUST remain outside positive geometry/coord supervision, duplicate-ul positives, and extra desc-positive supervision.
- `fn` objects remain positively supervised.
- Closure / EOS remain supervised.

#### Scenario: Duplicate-certified objects are removed from the positive prefix
- **WHEN** a rollout object is classified as `duplicate`
- **THEN** it does not contribute to the positive teacher-forced prefix
- **AND** it is represented only through duplicate-ul supervision and diagnostics.

### Requirement: Rollout-context semantics are explicit, auditable, and coherent across trainers
The unified loss registry SHALL treat clean-prefix rollout semantics as the canonical Channel-B rollout contract.

Normative behavior:
- Channel-B positive masks are built from the clean teacher-forced target, not the raw rollout prefix.
- Neutral `shielded_clean` extras MAY participate in the global rollout-prefix struct masks when that token-ce weight is enabled, but MUST stay outside coord supervision groups and duplicate-ul positives.
- Duplicate-ul supervision MUST be boundary-local and explicit rather than encoded through hidden token-ce behavior.

#### Scenario: Cluster-demoted unmatched clean extras remain coord-neutral while keeping shared prefix structure supervision
- **WHEN** a retained unmatched clean object is cluster-demoted or otherwise left neutral after pseudo-positive triage
- **THEN** it may remain in the clean prefix as context
- **AND** it contributes no positive geo/coord or duplicate-ul target
- **AND** it may still participate in global rollout-prefix structure CE.

### Requirement: Duplicate UL is boundary-local and LCP-defined
The unified loss registry SHALL define duplicate unlikelihood as a boundary-local objective over canonical clean vs duplicate continuations.

Normative behavior:
- For each clean boundary `b`, define `clean_continuation(b)` from the canonical clean teacher-forced target.
- For each duplicate attached to boundary `b`, define `duplicate_continuation(b, dup)` as canonical serialization of that duplicate object at boundary `b`, followed by the same canonical clean suffix.
- The target token is the first true divergence token of `duplicate_continuation(b, dup)` relative to `clean_continuation(b)`.
- Duplicate-ul aggregation is one unit term per unique divergence token per boundary.
- If no safe divergence token exists for a continuation, that continuation is skipped and counted in diagnostics.
- This deduplicated-per-boundary aggregation is intentional: the canonical v1 contract does not sum one UL term per duplicate object when multiple duplicates encode the same divergence token at the same clean boundary.

#### Scenario: Same-class-next-object cases do not blindly suppress the first desc token
- **WHEN** a duplicate continuation shares a non-empty token prefix with the clean continuation
- **THEN** duplicate-ul targets the first true LCP-divergence token
- **AND** it does not blindly suppress the first desc token.

#### Scenario: Unsafe or unavailable divergence token is skipped and counted
- **WHEN** a duplicate continuation yields no safe divergence token relative to the clean continuation
- **THEN** duplicate-ul does not contribute a loss term for that continuation
- **AND** the skipped continuation is counted in the corresponding duplicate-ul diagnostics counter.

### Requirement: Stage-2 Channel-A uses GT context only
The unified registry SHALL treat Stage-2 two-channel Channel-A as a single
GT-context supervision surface.

Normative behavior:
- Channel-A token CE, bbox geometry, and coord regularizer supervision MUST use
  `context=gt`.
- Final-pass/self_context-only Channel-A stabilizers are unsupported.
- Channel-A routing MUST NOT depend on `n_softctx_iter` or any equivalent final
  self-context pass count.

#### Scenario: Channel-A geometry uses GT context
- **WHEN** Stage-2 two-channel computes Channel-A bbox or coord supervision
- **THEN** the registry context is `gt`
- **AND** no `self_context` context is constructed.

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
The system SHALL define geometry loss (`geo`) on decoded continuous boxes in a way that is:
- stable under near-degenerate boxes,
- compatible with packing (segment-local indices),
- compatible with Channel-B FP-neutral masking.

Normative behavior:
- The system MUST decode 4 coords per box from coord-subspace logits using the
  fixed expectation decode path.
- The system MUST canonicalize decoded boxes before applying IoU-based losses:
  - `x_lo = min(x1, x2)`, `x_hi = max(x1, x2)`
  - `y_lo = min(y1, y2)`, `y_hi = max(y1, y2)`
  - enforce non-zero size with an `eps` floor where required for CIoU-like terms.
- The system MUST implement `geo` as a weighted sum of:
  - SmoothL1 (Huber) on `(x_lo,y_lo,x_hi,y_hi)`,
  - CIoU on the same canonicalized box representation.
- The system MUST aggregate `geo` as a mean over the supervised object set for the current context:
  - Stage-2 Channel-A `gt`: identity-aligned GT objects,
  - Stage-2 Channel-B `rollout`: `matched_clean` + `fn` objects (`duplicate` and `unmatched_clean` excluded).

#### Scenario: Channel-A geometry loss aggregates over GT objects only
- **WHEN** Stage-2 two-channel computes Channel-A geometry loss
- **THEN** `geo` aggregates over identity-aligned GT objects
- **AND** no final-pass self-context object set is consulted.

### Requirement: Coord regularizer loss (`coord_reg`) is explicit and strictly configured
The system SHALL treat coord regularization (`coord_reg`) as an optional component computed directly from logits and/or
coord-subspace probabilities, and its configuration MUST be explicit and strict (no silent enablement via ad-hoc trainer
code paths).

Normative behavior:
- `coord_reg` MAY include sub-terms that operate on:
  - coord-subspace probabilities at coord positions (`p_{t,k} = softmax(s_t / τ)` over coord bins `k ∈ [0,999]`), and/or
  - vocab-partition “gate” probabilities `p_coord(t)` at any supervised token position (coord vs non-coord).
- The set of enabled `coord_reg` sub-terms MUST be explicit via typed configuration (e.g., non-zero weights on
  `soft_ce`, `w1`, `entropy`, `coord_gate`, `text_gate`, `expected_l1`, `expected_huber`).
- Unknown `coord_reg` sub-term keys (or unknown module config keys for the module that implements `coord_reg`)
  MUST fail fast.
- The default behavior SHOULD keep `coord_reg` disabled unless explicitly enabled (Stage-2 geometry is the primary
  calibration signal; distribution-shape regularizers are hyperparameter-sensitive).

#### Scenario: Unknown coord_reg config fails fast
- **WHEN** a config enables `coord_reg` but contains an unknown sub-term/config key
- **THEN** initialization fails fast with actionable guidance listing supported keys.

#### Definition: Vocab-partition gate terms (`coord_gate`, `text_gate`)
To stabilize decoding and support ST/expectation bridges, the registry SHALL support two complementary gate sub-terms
inside `coord_reg`:
- `coord_gate`: applied at `type=coord` positions, encourages the model to place high probability mass on the coord
  vocab subset.
- `text_gate`: applied at `type=struct|desc` positions, encourages the model to place low probability mass on the coord
  vocab subset (i.e., high mass on the non-coord complement).

Normative behavior:
- Gate computation MUST NOT require adding new model heads; it MUST be derived from the token logits.
- Let `S_coord(t) = logsumexp(logits[t, coord_vocab_ids])` and `S_text(t) = logsumexp(logits[t, noncoord_vocab_ids])`.
  Define `p_coord(t) = exp(S_coord) / (exp(S_coord) + exp(S_text))`.
- `coord_gate` loss per token (coord positions): `L = -log(p_coord(t) + eps)`.
- `text_gate` loss per token (text positions): `L = -log(1 - p_coord(t) + eps)`.
- Gate losses MUST respect the same masking semantics as CE for the current context:
  - FP-neutral in `context=rollout` (FP spans excluded),
  - matched-prefix `desc` excluded where `CE_desc=0`,
  - closure/EOS supervision remains unchanged (gate terms MAY ignore `type=eos`).

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

### Requirement: `bbox_size_aux` is a separate optional decoded-box loss component
The unified loss registry SHALL treat bbox size supervision as a separate
optional loss component `bbox_size_aux`, not as an implicit expansion of `geo`.

Normative behavior:

- `bbox_size_aux` MUST consume canonicalized decoded boxes in the current
  `xyxy` / top-left-bottom-right contract,
- `bbox_size_aux` MUST preserve the current public bbox expression
  `bbox_2d: [x1, y1, x2, y2]` and the current coord-token `0..999` contract,
- `bbox_size_aux` MAY include:
  - matched `bbox_log_wh`,
  - thresholded `bbox_oversize`,
- `bbox_size_aux` MUST be mean-like over the supervised object set for the
  current context,
- the oversize term MUST remain opt-in and MUST NOT define a default small-box
  prior,
- the supervised object subsets MUST match the existing matched bbox contract
  for the active context.

#### Scenario: Matched size auxiliaries vanish on exact match
- **WHEN** predicted and target boxes are identical after canonicalization
- **THEN** the matched `bbox_log_wh` auxiliary is near zero
- **AND** the result does not depend on original corner ordering.

#### Scenario: Public bbox slot order remains unchanged
- **WHEN** `bbox_size_aux` is enabled
- **THEN** the public bbox expression remains `bbox_2d: [x1, y1, x2, y2]`
- **AND** internal canonicalization does not redefine the serialized slot order.
