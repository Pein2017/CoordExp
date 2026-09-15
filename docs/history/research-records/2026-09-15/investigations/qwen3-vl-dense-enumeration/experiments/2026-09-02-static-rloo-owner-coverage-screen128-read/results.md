---
title: Static RLOO Image-Disjoint Screen-128 Results
description: The train-positive static-RLOO adapter loses annotated-owner counts on the image-disjoint screen while slightly improving matched localization geometry.
type: investigation
role: research-result
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: complete_for_unit
unit_id: 2026-09-02-static-rloo-owner-coverage-screen128-read
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-09-02
---

# Static RLOO image-disjoint screen-128 result

## Outcome

- Mechanics: **`MECHANICALLY_VALID_WITH_ONE_CANDIDATE_LENGTH_CAP`**.
- Scientific summary:
  **`IMAGE_DISJOINT_COUNT_NEGATIVE_GEOMETRY_POSITIVE_WITH_STRUCTURAL_DEBT`**.
- The previously verified train-248 gain remains a positive result; this
  screen read shows that the same update does not yet improve image-disjoint
  annotated-owner counts.
- Stop this exact adapter after its registered read.  The next experiment is a
  separately frozen train-first successor from C, not more steps on this
  candidate.

| threshold | C | candidate | gains | losses | net | matched-IoU mean |
|---|---:|---:|---:|---:|---:|---:|
| IoU50 | 630 | 624 | 5 | 11 | **-6** | 0.854850 -> 0.856647 |
| IoU60 | 598 | 594 | 5 | 9 | **-4** | 0.870911 -> 0.871994 |
| IoU80 | 461 | 459 | 2 | 4 | **-2** | 0.916676 -> 0.917160 |

Across the three thresholds the annotated owner-threshold aggregate is
`1,689 -> 1,677`, or `-12`.  The common-owner IoU delta is also slightly
positive at every threshold (`+0.000066`, `+0.000480`, and `+0.000198`), so
the geometry movement is not solely survivor selection.  These tiny geometry
increments do not offset the owner-count losses.

## Authoritative evidence

The machine-readable receipt is
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-02-static-rloo-owner-coverage-screen128-read/analysis.json`,
SHA-256
`40969403dce0ef56bddd2710e51e2062231be47cb5f74c1d564d92d44c57481e`.
It binds the exact C and candidate rows, candidate manifest, inference config,
adapter fingerprint, three threshold comparisons, row-change classes,
ordering monitors, and the cap sensitivity.

- candidate rows: SHA-256
  `bece0971ca8c6989ca59dd15d2526802b646cded17ac77c2827293bc2a75acde`;
- candidate run manifest: SHA-256
  `ba70eaac0692fdbf86a024aadf2da1182a63991ee68fad3b9787bfe9e7d60d01`;
- candidate summary: SHA-256
  `ec6d3fa6d7fd41b1726c02f5766387d82afa99dae8daf477eee1abd34d260851`;
- IoU50/60/80 comparisons: SHA-256
  `0b0f679187aa8b12ecd15e8537d9b4e1f66c19d354ad1730b2054244ff92d40f`,
  `6a1959d04a84bc0511a15a40511a643d47a625c8193691ee4f2b076b804d2e7e`,
  and
  `97f84fd8526ac1a1655e48b7f3576d9879fc5bf56344f2142848ee63a7a2836f`.

Both canonical artifact families validate.  The candidate completed all 128
rows over eight GPUs with `127` natural `im_end` stops and one length cap.
Peak allocated memory was `14.91 GB` per rank, and the slowest rank decoded in
`380.31 s`.

## Natural behavior and cap sensitivity

The change is sparse: `95/128` rows are byte-identical, 25 change only parsed
coordinates, and eight change structure or semantics.  Matcher-visible
predictions decrease `1,387 -> 1,381`, duplicate candidates improve
`52 -> 47`, and invalid predictions remain `1 -> 1`.  Natural ordering
violations move from 40 events on 21 images to 42 on 22 images; ordering is a
monitor, not a gate.

The new cap occurs on image `59571`: C emits 86 valid predictions and natural
EOS, while the candidate emits 123 valid predictions, accumulates 214 dropped
spans, and reaches the 3,084-token limit.  This is genuine repetition and
termination debt.  It does **not** explain the count losses:

| 127-row sensitivity | C | candidate | net |
|---|---:|---:|---:|
| IoU50 | 620 | 614 | -6 |
| IoU60 | 590 | 585 | -5 |
| IoU80 | 457 | 455 | -2 |

The cap row itself is tied at IoU50 and IoU80 and gives the candidate one
extra IoU60 owner.  Excluding it therefore leaves IoU50/80 unchanged and makes
IoU60 one count worse.

Visual inspection was used only to characterize difficulty, never to replace
registered labels.  Image `182967` is a dense ski slope with small and
overlapping people/skis; image `39654` is a dense fruit-market scene; image
`59571` is a cluttered kitchen/store scene.  Annotation ambiguity is plausible
in parts of these scenes but is not established, and no count was manually
changed.  The screen losses therefore remain registered as observed.

## Interpretation

**Observation:** one eight-image static-RLOO update improves train-248 owner
counts at IoU50/60/80 by `+1/+2/+3`, yet moves the image-disjoint screen by
`-6/-4/-2`.  On both cohorts it tends to improve localization geometry while
exchanging some owner identities.

**Inference:** the complete-trajectory update contains a reusable geometry
signal, but binary IoU50 count credit is too coarse to produce reliable
image-disjoint owner coverage from this one update.  This supports testing a
more informative train-side trajectory reward from C before introducing QP
projection, a critic, PPO, or multi-step infrastructure.

**Not established:** repeatability, population generalization, full-scene
precision, F1, or mAP.  Under partial annotations, unmatched valid predictions
remain unknown; full precision-derived metrics require a complete or
adjudicated evaluation surface.
