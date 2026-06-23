---
type: idea
title: Ledger Auxiliary Loss Draft
description: Initial v0 proposal for a CE plus ledger-only auxiliary loss in Stage-1 autoregressive detection teacher forcing.
tags: [stage1, compact-detection, teacher-forcing, auxiliary-loss, visual-pooling, diagnostics]
state: active
updated: 2026-06-23
---

# Ledger Auxiliary Loss Draft

## Background

The model uses a full-wrapper autoregressive detection template:

```text
<object_ref_start>{desc}<object_ref_end><box_start>x1 y1 x2 y2<box_end>
<object_ref_start>{desc}<object_ref_end><box_start>x1 y1 x2 y2<box_end>
...
```

The current full-wrapper baseline improves grammar and parse stability, but
still shows low recall and duplication bursts. The first experiment should not
start with rollout-level RL/GRPO or token-level duplication unlikelihood. The
goal is to test whether the model can learn an internal object-inventory or
ledger state during a normal teacher-forced forward pass.

## Core Idea

Train with CE plus a ledger-only auxiliary loss.

At every inventory state position, especially prompt end before any object is
emitted and each `<box_end>` token after an object row is completed, the hidden
state should be able to answer, for each GT object in the image:

> Has this object already been emitted by the current prefix?

This is a training-only auxiliary loss. It should not change inference,
decoding, templates, tokenizer behavior, or model architecture at inference
time.

## Target Construction

For each image with `N` GT objects and `K` generated object rows, build
object-wise covered labels:

```text
state 0, prompt end:
  covered set = {}

state 1, after row 1 <box_end>:
  covered set = {object_1}

state 2, after row 2 <box_end>:
  covered set = {object_1, object_2}

...

state K, after row K <box_end>:
  covered set = {object_1, ..., object_K}
```

For every state `k` and object `i`:

```text
target y[k, i] = 1 if object i is already emitted in the prefix at state k,
                 else 0
```

## Requested V0 Shape

- Locate inventory positions using prompt end and row `<box_end>` positions.
- Reuse existing row metadata when possible; otherwise add robust collator
  annotation for detection span token positions.
- Build object embeddings from projected visual tokens that enter the LLM.
- Detach visual tokens for the auxiliary loss by default.
- Pool visual tokens by bbox overlap or token centers inside each bbox, with a
  nearest-top-k fallback for tiny boxes.
- Add small trainable state and object projection heads.
- Score state-object pairs with a temperature-scaled dot product.
- Use `BCEWithLogitsLoss` averaged over states and objects.
- Support full `K x N` scoring by default for debugging, with optional object
  sampling later.
- Integrate as `total_loss = CE_loss + lambda_ledger * ledger_loss`.
- Use a small default `lambda_ledger`, approximately `0.02` to `0.05`.
- Do not add commit loss, flip loss, stable loss, no-return unlikelihood,
  frontier head, visual current-vs-covered margin, or router loss in v0.

## Requested Diagnostics

Log:

- `loss/ledger`
- `loss/ce`
- `ledger_auc` if easy
- `ledger_accuracy` at threshold `0.5`
- `covered_score_mean`
- `uncovered_score_mean`
- `covered_minus_uncovered_margin`
- metrics bucketed by object count if easy
- metrics bucketed by same-class crowded examples if metadata exists

Mechanism checks should ask:

- Does ledger AUC improve?
- Does covered/uncovered separation improve as object count grows?
- Do crowded same-class examples improve?
- Does recall improve without merely reducing duplicate count?
- Does duplication decrease naturally because the inventory state becomes
  clearer?

## Safety Constraints

- Do not modify inference.
- Do not change decoding.
- Do not change the template for v0.
- Keep auxiliary heads training-only.
- Detach visual object embeddings by default.
- Make config flags for enabling/disabling the auxiliary loss.
- Be robust to images with zero objects or missing token positions.
- Add debug dumps for the first few batches showing token positions, object
  order, covered labels per state, object embedding shape, and ledger logits
  shape.
