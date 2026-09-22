---
title: Static RLOO Train-248 Breadth Read Results
description: The unchanged eight-image static-RLOO adapter has positive net owner coverage at IoU50, IoU60, and IoU80 across the registered 248-image train cohort.
type: investigation
role: research-result
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: complete_for_unit
unit_id: 2026-09-02-static-rloo-owner-coverage-train248-read
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-09-02
---

# Static RLOO train-248 breadth result

## Outcome

- Mechanics: **`MECHANICALLY_VALID`**.
- Scientific summary: **`TRAIN_WIDE_POSITIVE_MIXED_EXCHANGE`**.
- Next action: one image-disjoint read of this same unchanged adapter.

The one-update shared DoRA improves natural-greedy annotated-owner coverage at
all three reported thresholds across the registered 248-image training cohort:

| threshold | C | candidate | gains | losses | net |
|---|---:|---:|---:|---:|---:|
| IoU50 | 1,259 | 1,260 | 3 | 2 | **+1** |
| IoU60 | 1,186 | 1,188 | 6 | 4 | **+2** |
| IoU80 | 896 | 899 | 8 | 5 | **+3** |

Across the three thresholds this is `3,341 -> 3,347`, or six additional
owner-threshold hits.  The eight optimization images contribute two of those
hits; the other 240 images contribute four.  In particular, the eight-image
panel had zero IoU50/60 change, so the full-cohort `+1/+2` at those thresholds
is necessarily outside-panel transfer.  This is the first evidence from this
static-RLOO candidate that its learned direction is useful beyond the images
whose trajectories produced the gradient.

## Authoritative evidence

The authoritative analysis is
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-02-static-rloo-owner-coverage-train248-read/analysis-v2.json`,
SHA-256
`1f519f55d9dbcc93fc63f2e5faffec1c8fec043c731c98ab4478553191c04219`.
It binds the exact C and candidate rows, candidate manifest, config, adapter,
threshold comparisons, panel decomposition, and natural-behavior counters.
It supersedes the preserved first draft, which mislabeled the inference
summary's raw `scoreable_prediction_count` as the matcher's prediction count;
owner metrics and the scientific result were unchanged.

- candidate rows: SHA-256
  `966a674ab428217b401bf71913cf1bc9498f83a16a0a30960e940ec0975c8344`;
- candidate run manifest: SHA-256
  `f04b5501b6e4951fd17d3f05637c1325a569f372507cd70dd6da6ef6965079a6`;
- canonical summary: SHA-256
  `95fd660776ef1df02499629b14109b42987a3b7deeb3d17543634525aaa56a4e`;
- IoU50/60/80 comparisons: SHA-256
  `0c34ac536be731133053befe8c17160fd8c6ac6702af733cfb28e060c493660e`,
  `8965a2e1f2259cb6e27a69099f53900f8a906e91dbe87ae3fece1095e6304ece`,
  and
  `cc0141dab6a819716a269bc47250e909f29414914ed32622613e5bbde220c561`;
- eight-GPU inference log: SHA-256
  `93cff328016c335d9050c8e18215705c2b8edcf465ee20b7a0c9c81b8c031729`.

The repository's native controller split and canonically merged the 248 rows
over eight GPUs.  All rank-local and merged artifact families validate.  The
slowest rank decoded in `570.04 s`; aggregate capacity was `51.05` generated
tokens/s.  Peak allocated memory was `14.92 GB` per rank.

## Behavior and debt

The update is sparse but not panel-local: 174 rows are byte-identical to C, 60
change coordinates without changing descriptions or row count, and 14 change
row structure or semantics.  Both arms have exactly 245 natural `im_end` rows
and the same three pre-existing length caps; no stop reason changes.

The positive coverage vector is not uniform owner preservation:

- IoU50 has three gains and two losses;
- duplicate candidates improve `49 -> 44`;
- matcher-visible predictions decrease `2,226 -> 2,220`;
- invalid predictions increase `2 -> 3`;
- parser-dropped spans increase `845 -> 849`, entirely on two already-capped
  dense rows.

At IoU50, the gained categories are mouse, bottle, and cup; the losses are one
book and one handbag.  Visual inspection supports the user's annotation-quality
distinction: the lost book belongs to an already length-capped image with six
small book annotations whose physical boundaries are difficult to verify, so
it is weak debt evidence.  The handbag in image `544655` is visually
identifiable and is not covered by that exception, so the second loss remains
real preservation debt.  Gains also vary in reliability: the mouse is clear;
the bottle is small; the cup is in a dense restaurant scene.  The registered
counts remain unchanged rather than being manually relabeled.

## Interpretation

**Observation:** one gradient from only eight images produces positive net
IoU50/60/80 owner coverage over 248 train images, including positive net
movement outside its optimization panel.

**Inference:** static complete-trajectory credit is not merely memorizing the
eight selected rows.  It has learned a shared internal direction with bounded
train-wide transfer.  The mixed gains and losses show that binary IoU50 reward
does not yet provide clean owner preservation or a monotone localization
objective.

**Not established:** image-disjoint generalization, repeatability, full-scene
precision, F1/mAP under missing labels, or production readiness.  The next
shortest discriminator is therefore one image-disjoint natural read of this
exact adapter, not a larger optimizer, QP distillation stack, or multi-step RL
launch.
