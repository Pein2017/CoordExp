---
title: Human-13 Row-Contrast and Geometry-Preservation Successor Results
type: investigation
role: results
authority: non_normative_research
unit_id: 2026-08-13-human13-row-contrast-geometry-preservation-successor
topic: qwen3-vl-dense-enumeration
status: complete_bounded_same_panel_hold
updated: 2026-08-13
---

# Human-13 Row-Contrast and Geometry-Preservation Successor Results

## Disposition

`COMPLETE_BOUNDED_SAME_PANEL_HOLD`.

The successor is mechanically complete, but neither R1 nor R2 is a Pareto
improvement over historical A4 at exposure two. R1/R2 recover one additional
K-hit owner and two additional incidental K-miss owners, and reduce malformed
rows sharply, but they lose one additional Source owner and increase duplicate
burden. The duplicate failure is localized to image 14038 and remains a
free-running book-box attractor despite complete-row contrast at the twelve
frozen training states.

R2 did not exercise gradient projection: the accumulated R1 and aggregate
Source-G coordinate-watch gradients had positive dot products at both updates.
R2 therefore supplies no evidence that projection preserves owners. No
checkpoint is promoted, and no extra dose or post-result update was run.

This remains an overfit-only same-panel mechanism result. It says nothing about
validation or transfer.

## Evidence identity

Artifact root:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-08-13-human13-row-contrast-geometry-preservation-successor
```

- Sealed Human-13 manifest SHA-256:
  `a8f88716c1227054ab29dc698f89462c9369c47c8d6415de3783c0937f60a6fb`.
- Successor ledger: `78` aligned duplicate events, SHA-256
  `73dc8507060a316bf4660135e5e3aae3a9343e3ea37ab13e61258eee7eca1d5f`.
  Training used the twelve sealed-manifest events; the other 66 events remain
  diagnostic only.
- Canonical final analysis: `analysis/final-analysis.json`, SHA-256
  `653b04821bcb91c77c3cd93f6b1ec62ab78109331b99c00882b5c68d495a6ee7`.
- R1/R2 training receipts: SHA-256
  `639d1333821521aebfd0d5574294d0536032968592d8917931889dadc52d2b7b`
  and
  `fad480c32163b2ab1e03ded32afa199d93162f874492dbb9f806e41dcf7127ec`.
- R1 evaluation receipts at exposures one/two: SHA-256
  `e393062fd855ad8dcc47b7c45d6f72b1d56c6f3bfa28f2ea016efdb063d5cc26`
  and
  `3207d71f443e01a51be3490cc49b02912fdb09501576b8c32e0f856c71c5a89b`.
- R2 evaluation receipts at exposures one/two: SHA-256
  `d05ba000efa689a99479c6194439a4a0f0aed45f90ee3e07ae5aba56ae0dc6e3`
  and
  `da2d280c1ea1f97c922cca57f3681a824785ff1fb60a1e0ed0e569e2b211936c`.

Every decision-owning output uses original-prompt HF fp32/SDPA, physical batch
one, greedy decoding, repetition penalty `1.0`, natural `im_end`, and the
sealed cardinality-first maximum-total-IoU matcher.

## Pooled outcome

`H+`, `G-`, and `M+` count owner identities. Duplicate rows are chronological,
class-agnostic prediction-to-prediction IoU `>0.95` and receive no owner credit.

| Arm | Exposure | Unique | H+ | G- | M+ | Rows | Tok | Dup | Unmatched | Malformed | Cap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen Source | 0 | 173 | 0 | 0 | 0 | 255 | 2360 | 9 | 62 | 11 | 0 |
| Historical A4 | 2 | 181 | 8 | 5 | 5 | 311 | 2863 | 32 | 84 | 14 | 0 |
| R1 | 1 | 178 | 4 | 2 | 3 | 310 | 2855 | 35 | 86 | 11 | 0 |
| R2 | 1 | 178 | 4 | 2 | 3 | 310 | 2855 | 35 | 86 | 11 | 0 |
| R1 | 2 | 183 | 9 | 6 | 7 | 312 | 2872 | 40 | 86 | 3 | 0 |
| R2 | 2 | 183 | 9 | 6 | 7 | 312 | 2872 | 39 | 87 | 3 | 0 |

At exposure one, R1 and R2 have identical pooled metrics. Their image-14038
token traces are not byte-identical, so this is not an exact reproducibility
claim. At exposure two they retain the same owner tuple while differing by one
duplicate/unmatched classification. Because projection was a no-op, none of
these small differences is projection evidence.

## What happened

### Rectangle validity improved geometry burden, not owner selection

The malformed distribution is concentrated rather than panel-wide. Source has
all eleven malformed rows on image 14038. Historical A4@2 has twelve on image
14038 and two on image 7511. R1/R2@2 have only one on image 14038 and two on
image 7511; all thirteen images still terminate naturally, with no cap stops or
accepted invalid rows.

This is a strong bundled signal that the rectangle-valid `x2/y2` gate changes
the intended failure surface. It is not an isolated causal estimate because
R1 also changes duplicate loss. More importantly, the former malformed mass
does not become clean set coverage: image 14038 still emits 97 accepted rows,
of which 39--40 are duplicates and 42--43 are unmatched. Geometry validity and
owner enumeration are distinct problems.

### Complete-row contrast at twelve static states does not control rollout duplicates

All successor duplicates occur on image 14038: `35` at exposure one and
`39--40` at exposure two. The other twelve images have none. The trajectory is
a long sequence of repeated or near-repeated `book` rows after the initial
trusted rows. The row-contrast objective was finite and decreased from about
`0.174` to `0.143--0.151`, but it supervised only the twelve exact frozen
decision prefixes. Once greedy enters a new downstream prefix, that negative
state and its nearby coordinate aliases are outside the static objective.

This does not show that row-level unlikelihood is intrinsically ineffective.
It shows that upgrading y2-only unlikelihood to full-row contrast without
refreshing negative-state support is insufficient against this autoregressive
attractor. Increasing the coefficient on the same twelve states is not the
evidence-backed next move.

### Aggregate G-coordinate compatibility does not imply owner retention

R2's raw pre-projection dot products were `+46.6901` and `+41.3277`; the
projection coefficient was zero at both updates. Nevertheless R1/R2 lose six
Source owners at exposure two. An aggregate coordinate-watch direction can be
non-adverse while individual owners conflict, and coordinate CE does not guard
description choice, row order, stopping, or the discrete IoU-threshold event.

The result rules out interpreting this one aggregate watch vector as a
sufficient owner-retention certificate. It does not test an actively projected
negative-dot batch and therefore does not rule out gradient projection in
general.

### The extra recall is real but exchanged

Relative to A4@2, R1@2 moves `H+8 -> H+9` and `M+5 -> M+7`, but also
`G-5 -> G-6`. The largest local change is image 14038: R1@2 reaches fourteen
unique owners with `H+2/G-1/M+4`, versus A4@2's thirteen with
`H+1/G-0/M+3`. Other G losses occur on images 2299, 7511, 10707, 13923, and
16228. The pooled unique-owner increase therefore combines useful recovery,
incidental K-miss recovery, and owner exchange; it is not safe consolidation.

## Mechanical and compute receipts

- The real image-14038 vertical slice completed nine no-padding packs, one
  update, checkpoint write/read, full-panel HF evaluation, and analyzer. It
  used `100321` packed tokens, `31.57` training seconds, and about `9.46 GB`
  peak allocated memory. Its pooled evaluation was `H+3/G-3/M+3`; on image
  14038 it retained all nine Source owners but emitted 37 duplicate rows.
- Each full exposure has `55` no-padding packs and `624922` packed tokens.
  R1 completed two updates in `380.14 s` at `9.90 GB` peak; R2 completed in
  `422.91 s` at `9.97 GB` peak, including nine watch forwards per update.
- The four thirteen-image HF evaluations took `224.60`, `221.14`, `229.89`,
  and `221.60 s`. Packing removes padding but does not reuse image/prefix
  computation.
- Both arms used fresh Source model and AdamW state, independent world-size-one
  roots, and checkpoints one/two were structurally read back before inference.

## Supported

1. Rectangle-valid greedy-site supervision is a promising small component for
   suppressing the measured malformed-rectangle phenotype.
2. A4-style native K-union consolidation remains capable of moving additional
   H owners into greedy at two exposures, but the gain still comes with G
   exchange and heavy image-local duplication.
3. Complete-row contrast is only as useful as its conditional-state coverage;
   twelve static states do not control newly reached autoregressive prefixes.
4. Aggregate gradient compatibility with a G-coordinate watch is not an
   empirical owner-retention guarantee.

## Ruled out for this route

- Promoting R1 or R2 as a safe K-union-to-greedy recipe.
- Claiming that full-coordinate duplicate unlikelihood alone eliminates
  duplicate rollouts.
- Claiming projection efficacy from R2; it never activated.
- Extending the same checkpoints or static objective beyond exposure two.
- Treating lower malformed burden as owner-set success.

## Unresolved and next decision

The next useful discriminator, if separately authorized, is not more epochs.
It is a fresh-Source contrast that keeps the rectangle gate but changes exactly
one missing condition:

1. refresh duplicate negatives from the latest natural rollout at a declared
   next-batch boundary, or train a state-general anti-repeat relation rather
   than twelve exact prefixes; and
2. replace the single aggregate G watch with owner-level constraints or direct
   preserved greedy-margin evidence if preservation remains the target.

Those are new estimands and are not authorized by this completed unit. K-miss
supervision, external bridges, online same-batch update/decode, long dosing,
and checkpoint promotion remain out of scope.

## Claim boundary

This adaptive thirteen-image laboratory can establish only same-panel behavior
under the exact bound surfaces. It cannot establish generalization,
population prevalence, production safety, duplicate elimination, full-set
mastery, or architecture sufficiency.
