# teacher-forcing-unified-loss-registry Specification (Delta)

## Purpose
Define the *code-level internal contract* for a unified teacher-forcing loss system shared across:
- Stage-1 SFT (GT teacher forcing; static packing),
- Stage-2 Channel-A (GT CE anchor + self-context geometry),
- Stage-2 Channel-B (rollout context; FP-neutral + EOS-enforced).
- Rollout-matching SFT (rollout prefix + FN append; teacher-forced update under rollout context).

This capability is intentionally an internal contract:
- it constrains **naming**, **mask semantics**, and **context types** used in code,
- it enables modular loss/diagnostics composition without redefining loss math in multiple trainers,
- it does not require changing upstream HF model files.

## ADDED Requirements

### Requirement: Loss component names and contexts are canonical and shared
The system SHALL define canonical loss component names and context types that are used consistently across Stage-1 and
Stage-2 code paths.

Normative contexts:
- `gt`: pure GT teacher forcing logits/targets (Stage-1; also the CE anchor for Stage-2 Channel-A).
- `self_context`: Channel-A final-iteration logits under soft/ST coord-slot self-context.
- `rollout`: Channel-B one-pass teacher-forced logits under rollout-prefix + FN injection.

Normative loss component names (minimum set; can be extended):
- `struct_ce`: token cross entropy on structure tokens, including EOS enforcement (EOS is a distinct token type but its
  CE contribution is accounted under `struct_ce`).
- `desc_ce`: token cross entropy on description tokens.
- `coord_token_ce`: token cross entropy on coord vocabulary tokens (optional; typically GT context only).
- `coord_reg`: coord-subspace regularizers computed from logits/probabilities (optional; includes distribution/ordinal
  terms on coord positions and vocab-partition gate terms).
- `geo`: bbox-level geometry loss computed on decoded boxes.

Normative behavior:
- A single implementation of the above components MUST be reused across stages/channels (no duplicated definitions).
- Any module pipeline or trainer mixin that emits per-step metrics MUST use stable metric key prefixes derived from the
  canonical component names (e.g., `loss/geo`, `loss/struct_ce`).
- Trainers MUST emit only canonical `loss/<component>` keys for registry-defined loss components; trainer-specific loss
  aliases MUST NOT be emitted.
- This change MUST update `docs/training/METRICS_LOSSES.md` and `docs/training/STAGE2_RUNBOOK.md` to reflect the
  canonical metric key contract introduced by the unified registry/pipeline.

#### Scenario: Shared naming prevents silent drift
- **GIVEN** Stage-1 and Stage-2 both report bbox geometry loss
- **WHEN** metrics are logged
- **THEN** both code paths emit the same canonical key prefix for that component (e.g., `loss/geo/*`).

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


### Requirement: Channel-B rollout context is FP-neutral and EOS-enforced
For Stage-2 Channel-B (`context=rollout`), the system SHALL apply FP-neutral masking while enforcing EOS/closure
supervision.

Definitions (rollout context object subsets; stop-grad alignment via parse/match):
- `matched`: predicted prefix objects accepted into ValidMatched.
- `fp`: predicted prefix objects not matched / dropped-invalid treated as FP.
- `fn`: GT objects injected into the same top-level `objects[]` container for supervision.

Normative behavior (rollout context masks):
- FP spans MUST receive zero CE weight and MUST NOT contribute to geometry/coord distribution losses.
- Matched prefix objects MUST receive **struct-only** CE (desc CE weight = 0).
- FN injected objects MUST receive struct+desc CE (desc CE weight = 1 by default, configurable via desc weight).
- The outermost JSON closure `}` and `<|im_end|>` MUST remain supervised (EOS-enforced), even when other spans are
  masked (this prevents stop-neutral regressions).

#### Scenario: FP-neutral does not disable EOS supervision
- **GIVEN** a rollout prefix with at least one FP object span
- **WHEN** rollout-context CE masks are built
- **THEN** FP spans are masked out
- **AND** the top-level closure token(s) remain supervised.

### Requirement: Rollout-context semantics are explicit, auditable, and coherent across trainers
The system SHALL make rollout-context supervision semantics explicit and auditable so that:
- Stage-2 Channel-B and rollout-matching SFT use a coherent rollout-context masking contract by default, and
- any ablations are expressible via typed configuration (not trainer-specific forks).

Normative behavior:
- The unified loss registry MUST provide a single rollout-context mask builder that enforces the
  `progress/full_idea.md` rollout semantics:
  - FP-neutral masking: FP spans have zero CE weight and are excluded from geometry/coord-dist losses,
  - matched prefix objects: struct-only CE (desc weight = 0),
  - FN injected objects: struct CE + desc CE (desc supervised by default, configurable via an explicit weight),
  - closure/EOS: supervised (EOS-enforced).
- Stage-2 Channel-B and rollout-matching SFT MUST both consume this same rollout-context mask builder (no duplicated
  rollout-context masking logic in trainer code).
- Any deviations for ablations (e.g., `fn_desc_weight=0`, `matched_prefix_struct_weight=0`) MUST be expressed via
  explicit, typed weights and MUST be logged as part of pipeline identity so runs are auditable.


### Requirement: Channel-A CE anchoring and self-context geometry are separate contexts
For Stage-2 Channel-A, the system SHALL treat:
- CE anchor logits as `context=gt` (A1 logits), and
- self-context logits as `context=self_context` (final iteration).

Normative behavior:
- Channel-A **desc CE** MUST be computed from `context=gt` logits (to prevent format drift under self-conditioning).
- Channel-A **struct CE** MUST be computed from `context=gt` logits as the primary CE anchor.
- Channel-A MAY additionally compute a small-weight **self-context format/closure CE stabilizer** from
  `context=self_context` logits restricted to token types `struct` and `eos` only:
  - `type=desc` tokens MUST have CE weight `0` in `context=self_context`,
  - `type=coord` tokens MUST have CE weight `0` in `context=self_context`,
  - `type=struct|eos` tokens remain supervised.
  This stabilizer MUST be controlled by an explicit typed weight in the objective pipeline (e.g.,
  `token_ce.config.self_context_struct_ce_weight`) and MUST be recorded in pipeline identity so runs are auditable.
- Channel-A geometry (`geo`) MUST be computed from `context=self_context` logits (to train under self-conditioned coord
  context).

#### Scenario: Channel-A CE uses A1 logits
- **WHEN** Stage-2 Channel-A executes with `n_softctx_iter >= 2`
- **THEN** the CE anchor loss uses A1 logits (context `gt`)
- **AND** geometry uses final-iteration logits (context `self_context`).


### Requirement: Straight-Through (ST) bridge modes are configurable
The system SHALL support ST modes for:
1) coord-slot self-context embeddings (Channel-A), and
2) geometry coord decode (Channel-A and Channel-B).

Normative config knobs (YAML-driven and typed):
- `coord_ctx_embed_mode`: `soft|st|hard`
- `coord_decode_mode`: `exp|st`

Normative key paths (this repo; required for implementers):
- Stage-2 two-channel (`custom.trainer_variant: stage2_two_channel`):
  - `stage2_ab.coord_ctx_embed_mode`
  - `stage2_ab.coord_decode_mode`
- Stage-2 rollout-aligned (`custom.trainer_variant: stage2_rollout_aligned`):
  - `rollout_matching.coord_decode_mode`

Normative semantics:
- `soft` uses expected embedding / expectation decode.
- `st` uses hard forward values (argmax) and soft backward gradients (expectation path).
- `hard` is debug/inference-only; it MAY be supported but MUST NOT be the default for training.

Normative ST identity (informative but required semantics):
- Let `stopgrad(·)` denote detach/stop-gradient.
- ST outputs MUST be expressible as `y = y_hard + (y_soft - stopgrad(y_soft))` so:
  - forward evaluates the hard path, and
  - backward follows the soft path gradients.

#### Scenario: ST/exp defaults are explicit and override-able
- **GIVEN** current defaults (e.g., `stage2_ab.coord_ctx_embed_mode=st`, `stage2_ab.coord_decode_mode=exp`)
- **WHEN** training starts without explicit ST keys
- **THEN** behavior matches the current implementation
- **AND** users can override embedding/decode modes by setting the ST keys in YAML.


### Requirement: Geometry loss (`geo`) uses canonicalized boxes and a stable decomposition
The system SHALL define geometry loss (`geo`) on decoded continuous boxes in a way that is:
- stable under near-degenerate boxes,
- compatible with packing (segment-local indices),
- compatible with Channel-B FP-neutral masking.

Normative behavior:
- The system MUST decode 4 coords per box from coord-subspace logits using `coord_decode_mode`:
  - `exp`: expectation decode (CoordExp / soft expectation),
  - `st`: ST decode (hard argmax forward + expectation grad).
- The system MUST canonicalize decoded boxes before applying IoU-based losses:
  - `x_lo = min(x1, x2)`, `x_hi = max(x1, x2)`
  - `y_lo = min(y1, y2)`, `y_hi = max(y1, y2)`
  - enforce non-zero size with an `eps` floor (to prevent NaNs in CIoU-like terms).
- The system MUST implement `geo` as a weighted sum of:
  - SmoothL1 (Huber) on `(x_lo,y_lo,x_hi,y_hi)` and
  - CIoU on the same canonicalized box representation.
- The system MUST aggregate `geo` as a mean over the supervised object set for the current context:
  - Stage-2 Channel-A `self_context`: identity-aligned GT objects,
  - Stage-2 Channel-B `rollout`: `matched` + `fn` objects (FP excluded).

#### Scenario: FP objects do not contribute to geometry loss
- **WHEN** Stage-2 Channel-B runs with FP objects present in the rollout prefix
- **THEN** FP objects contribute `0` to `geo`
- **AND** only `matched` and `fn` objects contribute to `geo`.


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
