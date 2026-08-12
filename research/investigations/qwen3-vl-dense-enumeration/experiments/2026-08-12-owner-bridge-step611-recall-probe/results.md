---
title: Owner Bridge Step-611 Natural-Decode Recall Probe Results
description: Bridge-bound 13-image primary screen and supplementary val200 evidence for the interrupted step-611 checkpoint.
type: investigation
role: research-result
authority: user-authorized-checkpoint-probe
architecture_promotion_status: not_promoted
training_promotion_status: not_promoted
unit_id: 2026-08-12-owner-bridge-step611-recall-probe
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified_behavioral_hold
updated: 2026-08-12
---

# Owner Bridge Step-611 Recall Probe Results

## Decision

**HOLD.** Step-611 proves that the permanent OwnerBridge composition can load,
route, latch, write through complete rows, reroute, and terminate under native
greedy decoding. It does not show useful dense-object recall. The dominant
observed failure is under-enumeration: every sequence terminates naturally even
though every final bridge route selects a non-null atom and applies a saturated
admission residual. The smaller number of emitted rows also has nontrivial
grounding and geometry error.

No source checkpoint was run. These results establish absolute behavior for
step-611, not a relative improvement over Source or another tuned checkpoint.

## Bound composition

- model identity fingerprint:
  `d5fa28b0aed8107013832e104f6cfe5a1a041c4e9557637ff23ec61812bc59db`
- complete composition fingerprint:
  `2d011d5fa30ad20344c0f1f7e9c680a51a7ddc01e1e0afc77570606ad66e3e88`
- OwnerBridge payload fingerprint:
  `c74a93ce0a7c09c590fb320d1e71a1abc3c2201ada040355a4024c89b22f316d`
- adapter fingerprint:
  `b1ba294b6a87b751e2ecc99e40cbe035ed240492dcd7f9b452a9c01ee377e881`
- selected-embedding fingerprint:
  `63d6784a8f51b830646f53972cff0f755325f1f2be1bd3869356cb61ce330252`
- decode: dynamic HF, BF16, FlashAttention-2, batch size 2, native greedy,
  `max_new_tokens=512`, `repetition_penalty=1.0`.

The adapter marker requires the bridge and selected-embedding payloads; bridge
omission is a composition error rather than an adapter-only inference mode.

## Primary human-refined 13-image screen

The exact published panel was first attempted directly and failed before model
load because its authored object order violates the checkpoint's
`geo_sorted_xy` template contract. The one repaired input preserves the same
13 image ids, all 392 GT objects as multisets, and all non-object metadata; it
only sorts objects by `(x1, y1)` and adjusts relative image paths for the new
directory depth. This is a technical input repair, not a model-result retry.

The reporting contract requires the legacy-12 and image `2299` slices before a
pooled statistic:

| Slice | Images | GT | Valid predictions | Dropped raw rows | Empty valid output | class+IoU50 matches | class+IoU75 matches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Legacy human-refined 12 | 12 | 346 | 23 | 5 | 1 | 7 | 4 |
| Image 2299 | 1 | 46 | 0 | 1 | 1 | 0 | 0 |
| Pooled 13, secondary only | 13 | 392 | 23 | 6 | 2 | 7 | 4 |

The class-aware diagnostic matches use greedy one-to-one matching in the shared
0-999 coordinate space: GT `bbox` against prediction `coord_bins`. Pooled
micro-recall is `7/392 = 0.01786` at IoU 0.50 and `4/392 = 0.01020` at IoU
0.75. The canonical evaluator is intentionally non-benchmark on this selected
panel; its diagnostic `bbox_AR100` is `0.0134129` and `bbox_AP` is `0.0137757`.

Image `2299` is already decisive: the model emits one `person` row with
`[999, 281, 991, 796]`, which is rejected because `x1 > x2`, then stops. It
recovers none of the 46 human-refined owners.

The independent 13-image run took `25.37 s`. The first twelve rows agree
exactly with their val200 occurrences; the sole physical-batch-size-one final
row differs only in two nearby coordinate tokens. The decision is invariant to
that BF16 batch-shape difference.

## Bridge lifecycle on the 13 images

| Receipt | Count |
| --- | ---: |
| valid lifecycle rows | 13/13 |
| routes | 42 |
| latches / opener writes / box-end writes | 29 / 29 / 29 |
| total row writes | 272 |
| admission residual clipped at 0.03 cap | 42/42 |
| row writes clipped at 0.10 cap | 91/272 |
| null route ranked first | 0/42 |
| final route selected a non-null atom | 13/13 |
| native EOS termination | 13/13 |
| bridge terminal errors | 0 |

The 29 latches equal the 29 raw emitted rows (`23` valid + `6` dropped), and
the 42 routes equal one initial route per image plus one reroute after every
raw row. The bridge is therefore active and reconciled, not silently bypassed.

The critical seam is after the final row. All 13 final routes select a non-null
atom and apply the maximum allowed admission residual, but native greedy still
chooses EOS. The implementation makes this possible by design: routing stores a
pending atom and changes a residual state, but only a subsequently generated
row opener latches that atom; the bridge never forces or owns token choice.

## Supplementary val200 prevalence

The val200 run completed before the user corrected the probe order. It is kept
as supplementary evidence only; the 13-image screen was sufficient to stop
scale-up and identify the same failure family.

- canonical `bbox_AR100 = 0.05231925`, `bbox_AP = 0.04917330`,
  `bbox_AP50 = 0.07551875`, `bbox_AP75 = 0.04597845`;
- size recall: small `0.004329`, medium `0.010643`, large `0.071894`;
- 1,600 GT, 331 valid predictions, 49 dropped raw rows, 8 rows with no valid
  prediction, zero truncation, and 200/200 native `im_end` stops;
- corrected class-aware matching in shared 0-999 coordinates gives 91 IoU50
  matches (`0.056875` micro-recall) and 53 IoU75 matches (`0.033125`);
- IoU50 micro-recall by GT density is `0.1894` on 1-4 GT images, `0.0648` on
  5-9 GT images, and `0.02574` on 10+ GT images;
- 380 raw rows equal 331 valid plus 49 dropped; only one exact repeated valid
  prediction occurs, so duplicate explosion is not the primary failure;
- lifecycle is valid on 200/200 rows: 580 routes, 380 latches, 3,573 writes,
  580/580 clipped admissions, no null-top route, and every final route is
  non-null before native EOS.

The average valid output is 1.655 rows per image against 8 GT objects per
image. `bbox_AR1` (`0.05229548`) is almost identical to `bbox_AR100`, another
sign that extra allowed detections do not unlock meaningful coverage.

## What step-611 brought

1. **A real, load-bearing bridge composition.** Adapter, selected embeddings,
   and bridge are identity-bound and execute end to end.
2. **A working route/latch/write lifecycle once a row opens.** Every emitted
   raw row is accounted for by one latch, opener write, box-end write, and
   reroute.
3. **No demonstrated useful recall increase.** Without a source arm no relative
   delta is estimable, and absolute recall on the decision-bearing dense panel
   is too low to qualify as useful.
4. **A concrete training-to-decode mismatch.** The step-611 presentation-end
   teacher-forced eval reports `acc_top1=0.6221`, but natural decode frequently
   stops after one to four rows while the router still strongly rejects null.
5. **Two separable remaining failures.** Admission does not reliably turn a
   non-null route into another opener; among admitted rows, owner grounding and
   box geometry are also imperfect. The 13-image screen alone contains 6/29
   invalid raw rows and only 7 IoU50 matches.

## Claim boundary and next discriminator

The evidence rejects “the bridge was missing” and “duplicate explosion is the
main reason recall is low.” It supports a route-to-admission realization gap,
plus secondary grounding/geometry weakness. It does not distinguish whether
the admission cap is too small, the learned admission direction is wrong, or
the boundary objective is miscalibrated for on-policy states.

The next shortest discriminator, if authorized, is not another broad eval. Use
the same 13 images and inspect the post-final-route opener-versus-EOS logits and
the admission residual dose/direction at the final boundary, with no source arm
needed initially. Stop if the native opener margin does not respond coherently
to a small signed dose ladder; otherwise test whether an admitted opener
preserves the routed atom through one complete row. No further training is
authorized by this result.

## Evidence roots

- primary 13-image run:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-12-owner-bridge-step611-recall-probe/artifacts/inference/owner-bridge-step611-recall-bridge-human13-repair1/`
- supplementary val200 run:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-12-owner-bridge-step611-recall-probe/artifacts/inference/owner-bridge-step611-recall-bridge-val200-repair1/`
- preserved direct-panel technical failure:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-12-owner-bridge-step611-recall-probe/artifacts/inference/owner-bridge-step611-recall-bridge-human13/`
- preserved val200 relative-path technical failure:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-12-owner-bridge-step611-recall-probe/artifacts/inference/owner-bridge-step611-recall-bridge-val200/`
