---
title: Smoke A and Learning-Rate Dose Screen Results
description: Source-parity, one-step optimization, and ordinary greedy-rollout evidence for the first own-prefix calibration fixture.
type: investigation-result
role: evidence
authority: non_normative_research
status: complete
evidence_status: smoke_a_complete_real_smoke_b_required
updated: 2026-07-20
---

# Smoke A and Learning-Rate Dose Screen Results

## Decision

The implementation and one-event training path are executable, but this is not
enough evidence for the 256-image screen.

Two local treatments produced real ordinary-rollout changes:

1. the coordinate-boundary package at learning rate `2e-5`; and
2. a low-dose full-row candidate preference at learning rate `5e-6`.

The second treatment must no longer be described as entity-only supervision.
In the selected same-category event, the uncovered and covered people share
the same wrapper and description tokens. Their physical owners become
distinguishable only through their coordinates. The treatment therefore ranks
one complete candidate row above another and trains phrase and geometry
together.

The next gate is a genuinely reviewed 8-to-16-state Smoke B with held-out
events. No 256-image job is authorized by this result.

## Frozen Evidence Identity

The source is the geometry-sorted, description-first, pure-cross-entropy plus
token-type-gate Qwen3-VL 2B Weight-Decomposed Low-Rank Adaptation checkpoint at
step `4,887`:

```text
/data/CoordExp/.worktrees/CoordExp-swift/outputs/prod/coordexp_swift/
qwen3_vl_2b_desc_first_geo_sorted_pure_ce_typegate_dora_r16a32_llm_12000_
accelerate8_ebs24_8epoch_warmup0p1/checkpoints/step-4887/
```

The source-composition fingerprint is:

```text
ec308f8f5a26f5425eadeecf6fde5114d10dbf8e31de0b1fa0cc78abab8c3f2c
```

The immutable historical two-event bank is:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-07-20-own-prefix-entity-transition-and-coordinate-boundary-calibration-training-screen/
smoke-a-combined-v4/state-bank/
```

Its identities are:

| Artifact | SHA-256 |
| --- | --- |
| `manifest.json` | `09e5e53c32fa8822c7d9e430d1dc20f33e72adf432716dbaae3756511c58fbda` |
| `records.jsonl` | `52a027cb01b4fc6be98422100c06812568a121f3f33fd577519a1553c2cc13af` |
| Bank identity | `36fe931faa1c421817c3183287e1573eba30f83fe36720dcb2262c5a5f9475e9` |

This bank records the executed Smoke A evidence but is no longer admissible
under the hardened geometry-trust rule: its same-category positive owner is
resolved through four coordinate tokens whose geometry was marked ambiguous.
Current code intentionally refuses to load it for new training. Reproduction
of the historical run is pinned to producing revision
`fbceb6288ddcfa04365f19b321a2f3da5649dd9c`; Smoke B must use a newly reviewed
bank. The artifact and the completed historical observation are unchanged.

The exact-prefix source replay is:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-07-20-own-prefix-entity-transition-and-coordinate-boundary-calibration-training-screen/
smoke-a-combined-v4/source-parity/exact-prefix-replay-image7816-seed12.json
```

Its SHA-256 is
`113f78350969c5f7a35d42f6763126a9c553e15cea1361d1bce6d913ecc27447`.
The exact greedy harmful row and sampled seed-`12` row were reproduced from the
source checkpoint without decode-and-retokenize reconstruction.

## Trainable Surface

Every run passed source reconstruction and trainable-surface checks:

- Vision Tower frozen;
- multimodal MLP aligner frozen;
- selected-token embedding delta loaded once and frozen;
- language-tower Weight-Decomposed Low-Rank Adaptation only;
- `196` matched language linear modules;
- `588` trainable low-rank tensors;
- 32-bit floating-point research-loss math; and
- finite gradients followed by one applied optimizer update.

## Local Margin Movement

All declared one-step updates moved their training margin in the intended
direction.

| Profile | Learning rate | Transition margin, before to after | Coordinate margin, before to after |
| --- | ---: | ---: | ---: |
| coordinate boundary | `1e-5` | not applicable | `-0.2384` to `+1.6449` |
| coordinate boundary | `2e-5` | not applicable | `-0.2384` to `+3.1864` |
| candidate row | `5e-6` | `-1.2836` to `+3.0300` | not applicable |
| candidate row | `1e-5` | `-1.2836` to `+5.4585` | not applicable |
| candidate row | `2e-5` | `-1.2836` to `+9.6989` | not applicable |
| joint | `5e-6` | `-1.2836` to `+2.4662` | `-0.2384` to `+0.2616` |
| joint | `1e-5` | `-1.2836` to `+5.6365` | `-0.2384` to `+0.7318` |
| joint | `2e-5` | `-1.2836` to `+9.4159` | `-0.2384` to `+1.4818` |

The earlier `5e-5` implementation smoke moved the margins farther, but its
ordinary rollout exposed strong overshoot. It is retained only as a dose
reference.

## Ordinary Greedy Rollout

All runs used repetition penalty `1.0`, greedy decoding, the normal Qwen3-VL
forward path, and no state bank or controller at inference time. Every run had
zero parser failures and natural image-end termination on both images.

Image `7816` is the dense-person transition case. The source emitted ten rows,
but its fifth and sixth person rows both mapped to covered owner `205108`; it
missed nearby owner `211764`. Image `9400` is the coordinate case. Its first
source box was `[874, 98, 999, 263]` for trusted owner `1211112`, with
intersection over union `0.6128`.

The following entity counts use class-aware maximum-cardinality matching at
intersection over union at least `0.5`, followed by maximum total intersection
over union. The critical right-person cluster on image `7816` was also checked
directly against owners `205108` and `211764`; the reported rescue is not a
same-class assignment artifact.

| Profile | Learning rate | Image 7816 rows | Unique owners in fixed row budget | Owner `211764` | Owner `205108` | Image 9400 first-box intersection over union |
| --- | ---: | ---: | ---: | --- | --- | ---: |
| source | none | 10 | 9 | missed | duplicated | `0.6128` |
| coordinate boundary | `1e-5` | 10 | 9 | missed | duplicated | `0.9135` |
| coordinate boundary | `2e-5` | 10 | 10 | hit once, `0.5897` | hit once, `0.7603` | `0.9135` |
| candidate row | `5e-6` | 10 | 10 | hit once, `0.6376` | hit once, `0.7109` | `0.6128` |
| candidate row | `1e-5` | 10 | 7 | missed | suppressed | source-like |
| candidate row | `2e-5` | 11 | 8 overall, 7 in first 10 | hit once, about `0.62` | suppressed | source-like |
| joint | `5e-6` | 10 | 10 | hit once, `0.6376` | hit once, `0.7109` | `0.6128` |
| joint | `1e-5` | 10 | 7 | missed | suppressed | `0.9135` |
| joint | `2e-5` | 10 | 8 | hit once | suppressed | `0.9135` |

The transition-only and joint `5e-6` prediction files are byte-identical, with
SHA-256
`a23be43dedb91d541a99736faa12118655eb7343d382c85f63d74998724ff356`.

The two decisive image-`7816` row pairs are:

| Run | Covered-person row | Neighbor row |
| --- | --- | --- |
| source | `[877,132,927,432]` | `[886,145,935,432]`, another owner-`205108` box |
| coordinate `2e-5` | `[882,129,927,421]` | `[894,137,931,213]`, owner `211764` |
| candidate row `5e-6` | `[882,129,927,374]` | `[894,134,935,218]`, owner `211764` |

## Interpretation

### Result One: the coordinate training package changed rollout composition

The coordinate-boundary event was trained on another image, yet its `2e-5`
checkpoint both corrected image `9400` and split two previously duplicate-like
rows on image `7816` into two physical owners. This proves only that the whole
coordinate-boundary training package changed later rollout composition in
this one untrained image.

It is not yet evidence of a general coverage mechanism. The training case and
the rescued person are both small people near the top-right image boundary, so
a transferable local extent prior remains a strong alternative explanation.
The treatment also includes the small rollout-site token-type gate and updates
language-tower Weight-Decomposed Low-Rank Adaptation parameters globally. A
gate-only control is still required before attributing the rollout change to
the coordinate loss itself.

### Result Two: low-dose complete-row preference can repair one branch

At `5e-6`, ranking a model-discovered uncovered-object row above the actual
covered duplicate redirected the native rollout from owner `205108` to owner
`211764` without adding rows or losing unique owners. This is a real behavioral
result, not only an offline margin change.

However, the positive and harmful rows share all wrapper and description
tokens. Only their coordinates distinguish the two same-category owners. The
objective therefore did not train an owner identity independently of geometry.
The selected positive geometry was marked ambiguous, and the useful output is
close to that sampled geometry. The one-event result cannot establish generic
commitment, coverage tracking, or cross-image generalization.

Future use of this treatment is renamed **same-prefix uncovered-versus-covered
candidate-row preference**. Every scored positive or harmful row whose owner
resolution includes coordinate tokens must have trusted geometry for that
exact sampled path. A reviewer-corrected box may define the separate
coordinate-boundary target, but it cannot silently replace an untrusted
candidate row.

### Result Three: dose is part of the mechanism

For the complete-row preference, `5e-6` was useful while `1e-5` already
damaged neighboring rows and `2e-5` created a local short-box cluster. The
correct conclusion is not that the preference direction is useless; it is that
this one-event gradient has a very narrow safe dose and cannot be scaled from a
single fixture.

## Visual Evidence

- [Source versus coordinate `2e-5`, image 7816](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-20-own-prefix-entity-transition-and-coordinate-boundary-calibration-training-screen/smoke-a-combined-v4/dose-v1/visual-review/source-vs-coordinate-2e-5/0000_coco2017_val_000000007816_prediction_comparison.png)
- [Source versus candidate-row `2e-5`, image 7816](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-20-own-prefix-entity-transition-and-coordinate-boundary-calibration-training-screen/smoke-a-combined-v4/dose-v1/visual-review/source-vs-transition-2e-5/0000_coco2017_val_000000007816_prediction_comparison.png)
- [Source versus coordinate `2e-5`, image 9400](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-20-own-prefix-entity-transition-and-coordinate-boundary-calibration-training-screen/smoke-a-combined-v4/dose-v1/visual-review/source-vs-coordinate-2e-5-image9400/0000_coco2017_val_000000009400_prediction_comparison.png)

The canonical comparison renderer in these files shows only the fixture anchor
annotation. Its displayed true-positive and false-positive labels are not
full-image metrics; the figures are used only to inspect geometry.

## Promotion Gate

Proceed only to a real 8-to-16-state Smoke B:

- coordinate-boundary package at `1e-5` and `2e-5`;
- same-prefix uncovered-versus-covered candidate-row preference at `5e-6`;
- no joint arm until the two single treatments establish safe behavior across
  multiple events;
- one small gate-only control;
- diverse `x1`, `y1`, `x2`, and `y2` cases across location, category, scale,
  and crowding;
- image-grouped training and evaluation-only events frozen before dose choice;
  and
- all 12 blind dense images excluded from collection, review, tuning, and arm
  selection.

The 256-image screen remains blocked until this gate passes.
