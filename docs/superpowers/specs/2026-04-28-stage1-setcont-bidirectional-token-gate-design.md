# Stage-1 Set-Continuation Bidirectional Token Gate Design

> Archived / superseded on 2026-05-02.
> Historical provenance only for the pre-refactor Stage-1 set-continuation family.
> Do not use this file as an execution source.
> Active execution sources:
> - `docs/superpowers/specs/2026-05-02-training-infra-template-mode-refactor-design.md`
> - `docs/superpowers/plans/2026-05-02-training-infra-template-mode-refactor.md`

## Problem

The Stage-1 set-continuation objective currently scores coord-token labels under
coord-vocabulary normalization while eval-time rollout samples from the full
vocabulary. That is useful for coordinate-bin ranking but can hide full-vocab
probability leakage at coordinate slots. The reverse leakage is also possible:
coord-token mass can appear at schema, description, separator, or boundary
positions. Both failures are token-type control problems rather than stopping
problems.

## Goal

Add a lightweight bidirectional gate that preserves the repaired
set-continuation objective while restoring slot-type pressure:

```text
coord slot       -> penalize non-coord vocabulary mass
non-coord slot   -> penalize coord vocabulary mass
```

The mechanism must be validated independently of production mAP.

## Non-Goals

- Do not strengthen close supervision as part of this change.
- Do not relax strict eval parsing.
- Do not re-enable ordinary one-sequence Stage-1 loss mixins.
- Do not require `custom.coord_soft_ce_w1.enabled=true`.
- Do not introduce grammar-constrained decoding in this change.

## Mechanism

Each encoded branch already contains objective and token-type masks. The gate
uses label-coordinate masks before the standard next-token shift:

```text
coord_gate_label_mask =
  objective_label_mask AND coord_label_mask

text_gate_label_mask =
  objective_label_mask
  AND NOT coord_label_mask
  AND supervised label
  AND NOT special/eos/im_end/end_of_text/padding
```

After the same suffix crop and next-token shift used by candidate scoring:

```text
p_coord(t) = sum softmax(logits_full(t) / temperature)[coord_token_ids]

loss/coord_gate = mean(-log(p_coord(t)) over coord_gate positions)
loss/text_gate  = mean(-log(1 - p_coord(t)) over text_gate positions)
```

Reduction contract:

- Each gate loss is a token mean over all contributing scored objective-branch
  positions for one sample.
- The sample gate loss is added once per objective-contributing sample with the
  same sample denominator policy as `loss/candidate_balanced`.
- Exact all-candidate scoring and cap-8 fallback change only which branch tokens
  are observed, not the scalar gate weight per sample.
- If PEM threshold loss is enabled for an ablation, the gate still applies after
  the PEM margin is satisfied because it controls token type, not stop behavior
  or observed-GT probability mass.

The sample loss adds weighted gate terms:

```text
loss =
  loss/candidate_balanced
  + coord_gate_weight * loss/coord_gate
  + text_gate_weight * loss/text_gate
  + existing structural terms
```

## Config Shape

```yaml
custom:
  stage1_set_continuation:
    bidirectional_token_gate:
      enabled: true
      coord_gate_weight: 0.5
      text_gate_weight: 0.1
      temperature: 1.0
      scope: objective_tokens
```

Only `scope: objective_tokens` is valid in v1.

## Validation Checkpoints

1. Token-type assignment
   - Coord labels map only to coord gate.
   - Schema, description, comma/close boundaries, and object keys map only to
     text gate.
   - Prefix-only tokens and special stop tokens map to neither gate.

2. Loss masking alignment
   - Every gate row corresponds to the logits position that predicts the masked
     label token.
   - Supervised-suffix cropping yields the same gate loss as full logits.

3. Logits span
   - Gate logits rows are drawn from the same branch forward as candidate
     scoring.
   - Prefix logits are excluded even when physically returned by the model.

4. Vocabulary scope
   - Coord ids are exactly the configured 1000 coord-token ids.
   - Missing, duplicate, or out-of-vocab coord ids fail fast.

5. Runtime equivalence
   - Retained-graph serial scoring and smart-batched exact scoring produce the
     same gate losses, token counts, candidate scores, and gradients on
     deterministic fixtures.

6. Smoke gate
   - Real tokenizer/chat-template smoke logs finite gate losses.
   - Parse hygiene does not regress.
   - Coord-slot coord mass is high or moves upward.
   - Text-slot coord mass remains low or moves downward.

## Metrics

Compact metrics:

```text
loss/coord_gate
loss/text_gate
gate/coord_slot_coord_mass_mean
gate/text_slot_coord_mass_mean
gate/coord_tokens_count
gate/text_tokens_count
```

These keys are emitted only when `stage1_set_continuation` is active and the
gate is enabled.

## Main Risk

The largest risk is mask drift: using `candidate_object_label_mask` instead of
`objective_label_mask` would miss schema opener and post-candidate boundaries.
The second largest risk is a suffix-crop off-by-one that trains the wrong token.
Both risks must be red-tested before implementation.
