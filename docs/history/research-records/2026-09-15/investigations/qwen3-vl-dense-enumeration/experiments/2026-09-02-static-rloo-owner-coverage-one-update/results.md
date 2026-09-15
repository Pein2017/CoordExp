---
title: Static RLOO Owner-Coverage One-Update Results
description: One exact eight-rank static RLOO update leaves IoU50 train owner coverage unchanged while improving two owners across the IoU80 boundary.
type: investigation
role: research-result
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: complete_for_unit
unit_id: 2026-09-02-static-rloo-owner-coverage-one-update
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-09-02
---

# Static RLOO owner-coverage one-update result

## Decision

- Mechanical status: **valid**, including fresh unmerged cold readback.
- Registered scientific disposition:
  **`NO_TRAIN_GAIN_STATIC_RLOO_STOP`**.
- Secondary observation: **positive train-side geometry movement**.

One frozen K4 complete-trajectory RLOO update leaves the decision-owning
natural-greedy IoU50 annotated-owner count unchanged at `53 / 67 -> 53 / 67`.
The same update retains the exact IoU50 and IoU60 owner sets, improves matched
mean IoU from `0.8589287` to `0.8629771`, and moves IoU80 coverage from
`38 / 67` to `40 / 67` with two gains and no loss.  The registered stop is
therefore correct for the fixed IoU50-count candidate, but “no train gain” must
not be generalized to “no behavioral learning.”

## Immutable evidence

- update receipt:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-02-static-rloo-owner-coverage-one-update/receipt.json`,
  SHA-256
  `fd6c4004ab3781c32dc16beb77d279efd6cb73cee0c6309ec1662d96c4a77659`;
- cold-read receipt:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-02-static-rloo-owner-coverage-one-update/cold-read-receipt.json`,
  SHA-256
  `b5d8ec481289f2fedf0e39eb3ee1d5e3a1380b58fa3bc28dc02949e8aad29182`;
- evaluation receipt:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-02-static-rloo-owner-coverage-one-update/evaluation.json`,
  SHA-256
  `50abf61fb63eb1ecc43b30d7ded73b758845e3a4715d952ef7b9275fa17b1718`;
- natural scored rows:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-02-static-rloo-owner-coverage-one-update/natural-eval/qwen3-vl-2b-static-rloo-owner-coverage-panel8/gt_vs_pred.jsonl`,
  SHA-256
  `a1a5c99ea3a213e17569ded2824994ccf58ac83a4d55e26f8a65426bc3edf427`;
- update and natural-decode logs: SHA-256
  `2b4dfbbb11e9e496c82fdd4dad6f2387bcc1d78b9a31726468ee468b55e4ad93`
  and
  `adff7c43d3837837e86e88c0cd5947f804be131e5453db486d9e53adf5ed96b0`.

The saved artifact remains one shared rank-16 DoRA adapter over the universal
base.  Its fingerprint is
`9cbfde447f9164875180cd8bbe6122d329ceb49d732882f66cb9b03e7caadc1d`;
the `adapter_model.safetensors` SHA-256 is
`0096976cec21eb90a78844e72037314254b893d4a1d493dad2a539e65dd7494e`.
No merged checkpoint, per-image residual, optimizer state, or external memory
was written.

## Update mechanics

Eight ranks replayed four frozen complete actions each.  The run performed 32
forwards, 32 backwards, and exactly one fresh AdamW step.  Raw gradient norm
was `7.499629`; clipping reduced it to `1.0`; realized adapter-delta norm was
`0.0105915`; and clipped-gradient dot realized-update was `-0.00477169`, the
expected descent sign.  All 588 registered language-DoRA tensors changed and
no frozen tensor acquired a gradient.

Rank-zero update time was `151.75 s`; peak allocated GPU memory was `25.37 GB`
and peak host RSS was `12.18 GB`.  Fresh cold composition found all 588 saved
tensors exactly equal to their materialized values, zero runtime casts, and
`merged_adapters=[]`.

The first launch failed after all backwards but before `optimizer.step` because
the receipt compared nested schedule lists through a Python set.  It published
no adapter.  That immutable `MECHANICAL_INVALID` receipt has SHA-256
`52f0068daa77a90c740a7e3fc6f8b155f660dc849ec54b1ec225a90d6d6d7817`.
The one-line schedule-hash repair then restarted from exact C.  A later cold
read initially encountered an extra, non-contracted `requires_grad` assertion;
it wrote no receipt and performed no mutation.  Removing only that assertion
restored the specified inference gate.

## Natural train behavior

| threshold | C owners | candidate owners | gains | losses |
|---|---:|---:|---:|---:|
| IoU50 | 53 | 53 | 0 | 0 |
| IoU60 | 53 | 53 | 0 | 0 |
| IoU80 | 38 | 40 | 2 | 0 |

The two IoU80 gains are owner `381996:coco_ann:1504757` and owner
`474979:coco_ann:1308570`.  Five of eight decoded rows are byte-identical to C.
The other three keep the same descriptions, prediction counts, ordering
lengths, and terminal EOS while changing only coordinate tokens:

- image `381996`: one coordinate token changes;
- image `401735`: two coordinate tokens change;
- image `474979`: thirteen coordinate tokens change.

All eight rows end naturally at `im_end`; there are no invalid or dropped
predictions.  Prediction count remains 67 and the one duplicate candidate is
unchanged.  Fourteen valid unmatched predictions remain unknown.  Two natural
ordering violations are legal monitor events.

## Interpretation and next boundary

**Observation:** a small shared internal update driven only by relative
complete-trajectory IoU50 owner coverage changes natural greedy output and
improves localization on this selected training panel without changing its
owner identities or row count.

**Inference:** the score-function direction is not an optimizer no-op and can
carry useful geometry information into ordinary unmerged DoRA.  Because the
reward did not distinguish localization quality above IoU50, the IoU80 gain is
an indirect correlated effect, not proof that this estimator reliably
optimizes high-IoU localization.

**Not established:** an IoU50 owner-count gain, repeatability, wider-train or
image-disjoint improvement, true precision/F1/mAP, or robustness to missing
labels.  The frozen unit therefore stops without an internal seed or dose
sweep.  A later train-first successor may use this geometry signal, but it must
be recorded separately rather than retroactively changing this decision.
