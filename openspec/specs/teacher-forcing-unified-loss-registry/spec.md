# teacher-forcing-unified-loss-registry Specification

## Purpose
Define the canonical teacher-forcing contexts, token types, and registry-level loss component naming shared across Stage-2 two-channel and rollout-aligned training.

## Requirements
### Requirement: Canonical contexts, token types, and loss component names are shared
The system SHALL define canonical teacher-forcing contexts and token types that are used consistently across Stage-2 two-channel and rollout-aligned training.

Normative contexts:
- `gt`: pure GT teacher forcing (CE anchor).
- `self_context`: Channel-A final-iteration logits under coord-slot self-context.
- `rollout`: Channel-B teacher-forced logits under rollout-prefix + FN injection.

Normative token types (mutually exclusive):
- `struct`: JSON syntax + keys/punctuation and other non-desc, non-coord content.
- `desc`: free-text tokens inside an object’s `desc` value span.
- `coord`: coord-vocabulary tokens `<|coord_k|>`.
- `eos`: end token `<|im_end|>` (Qwen3-VL).

Normative minimum canonical loss component names (metrics use these names when emitted):
- `struct_ce`
- `desc_ce`
- `geo`
- `coord_reg`

NOTE (logging contract):
- These are canonical registry component names (often surfaced as `loss/<component>` keys inside pipeline-internal metrics).
- Trainers MAY choose to omit raw component keys from the training log to reduce redundancy and instead emit only objective-weighted provenance keys under `loss/<provenance>/<atom>` (see `trainer-metrics-components`).

#### Scenario: EOS token is assigned exactly once
- **WHEN** token-type masks are built for a sequence containing `<|im_end|>`
- **THEN** the EOS token receives `type=eos`
- **AND** it does not receive any other token type.

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
- In `context=self_context`:
  - `text_gate` MUST apply only to supervised non-coord text positions (struct/EOS by default).
  - Gate terms MAY ignore `type=eos`.

#### Scenario: FP objects do not contribute to text gate
- **WHEN** a rollout-context teacher-forced target contains FP spans
- **THEN** `text_gate` is computed with FP spans excluded
- **AND** changing only FP spans cannot change the computed `text_gate` scalar.

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
