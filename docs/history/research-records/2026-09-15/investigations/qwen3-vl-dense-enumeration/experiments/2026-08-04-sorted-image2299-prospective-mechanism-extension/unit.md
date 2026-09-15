---
title: Sorted Image 2299 Prospective Mechanism Extension
description: Applies the frozen owner-accessibility and native-prefix mechanism chain to image 2299 as a separately reported prospective confirmation slice.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: authorized_within_goal
unit_id: 2026-08-04-sorted-image2299-prospective-mechanism-extension
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified_with_calibration_stop
updated: 2026-08-04
---

# Sorted Image 2299 Prospective Mechanism Extension

## Decision and claim boundary

This unit applies the already completed sorted-checkpoint false-negative
mechanism chain to exactly one new image, COCO val image `2299`, using the
versioned input admitted by
[`2026-08-04-sorted-prospective-13-image-panel-admission`](../2026-08-04-sorted-prospective-13-image-panel-admission/unit.md).
The image contains `46` refined physical owners (`38 person`, `8 tie`). It is
reported as its own prospective slice. It never changes the completed
legacy-12 denominators, calibration, discovery rule, or conclusions.

The primary question is not whether image `2299` improves a pooled 13-image
metric. It is:

> When a large-owner, extremely dense, mostly single-category image is scored
> with the frozen legacy owner-accessibility rule, what fraction of native
> false negatives retains tested category-field support at owner geometry, and
> which later native-prefix channel prevents those supported owners from being
> emitted?

The unit is prospective only with respect to the owner-accessibility
disposition and its downstream channel ladder. Image `2299` appeared in older
July mechanism probes, so it is not a globally naive image. This unit must not
present it as an independent population sample or use it to invent a new
mechanism rule.

## Frozen sources

| Source | Frozen role |
| --- | --- |
| 13-image admission artifact | `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-04-sorted-prospective-13-image-panel-admission/evaluation-inputs/human-refined-13.coord.jsonl` |
| Panel SHA-256 | `01086b139fa23983697492fdb535b5154429277803e8f12b243f9a031d1451f8` |
| Image-2299 authority-row SHA-256 | `ce19853c74a595f22cc183ce450e561f2da3216e54a1e499cfbca1be7e1c425b` |
| Image bytes SHA-256 | `cd7199a37188c9ac6481520175866cbd78fa6fba35f4290bb0b60c78afcb2df3` |
| Prospective native reference config | `configs/coordexp_infras/infer/qwen3_vl_2b_desc_first_geo_sorted_step4887_human_refined13_hf_fp32_rp1p0.yaml`; this owns the prospective-13 input/run identity used by the native reference contract |
| S1 scorer runtime leaf | `configs/coordexp_infras/infer/qwen3_vl_2b_desc_first_geo_sorted_step4887_human_refined12_hf_fp32_rp1p0.yaml`; the S1 receipt binds this legacy leaf while its plan separately seals the admitted 13-image panel and exact image-2299 authority row. A literal diff shows that model/runtime/generation fields are identical; only comments, `run.name`, and `data.input_jsonl` differ. |
| Legacy census run | `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-03-sorted-owner-accessibility-phenotype-census/20260803T065743Z/` |
| Frozen support calibration content digest | `9dd6d7646fc55db6155124dc4bbfa46642b32b006d39758bd2e24d1ca97058c5` |
| Frozen support calibration file SHA-256 | `bbe0ccb5141e659c20b8df984d7fbf50e9c8ab96f0b1b860a1664b29a2dd12ba` |
| Frozen discovery rule content digest | `45bbe07065670a4291ed7d874fc2a8ca15caee79adf69e3f5fc78ab2e43101f8` |

Only the refined 46-owner row is authoritative. The official/val200
22-owner row for the same JPEG is a different ground-truth version and is
forbidden in this unit.

## Prospective prediction

The legacy-12 owner census found, among `202` eligible native false-negative
owners:

- `114/202` (`56.4%`) resolved tested localization support;
- `72/202` (`35.6%`) persistent no tested localization support; and
- `16/202` (`7.9%`) ambiguity-bound flips.

The persistent cohort was substantially smaller on the image plane than the
resolved cohort: median normalized area `0.000424` versus `0.001781`.
Image `2299` is deliberately a pressure test of that association: its median
owner normalized area is approximately `0.0247`, median minimum dimension is
approximately `110 px`, only `3/46` owners have minimum dimension below
`16 px`, and it nevertheless contains strong same-description crowding.

Before any new score is read, this unit freezes the directional prediction:

> If apparent owner scale is the leading discriminator for tested local
> support, the native-FN cohort on image `2299` should contain very few
> `persistent_no_tested_localization_support` owners; any such owners should
> concentrate among the smallest ties. A material persistent share among the
> large persons weakens the scale explanation and strengthens image-level
> routing, same-description competition, or interface-state alternatives.

The prediction is a single-image stress test, not an iid binomial population
claim. Exact binomial numbers may be shown only as a descriptive reference;
the 46 owners share one image encoding and one native trajectory.

## Stages

### S0: exact native greedy and strict matching

Obtain one geometry-sorted step-4887 native greedy rollout at `rp=1.0`, HF
fp32, baseline visual-token budget, and a nonbinding `max_new_tokens=3084`.
The rollout must stop naturally and must be matched one-to-one against the
46-owner refined row. Emit owner and prediction ledgers with stable IDs.

A historical artifact may be reused only after a dedicated admission proves
the exact checkpoint/adapter/embedding delta, runtime, prompt/template,
image bytes, refined ground truth, generation policy, token trace, parser
rows, and natural stop. A shorter configured horizon is admissible only when
the trace proves the model emitted `im_end` far before that horizon.

### S1: frozen owner-accessibility census

Build the same native self-prefix registry and the same score-independent
17-role candidate bank used by the legacy census. Score category-field support
at owner geometry and the separate proposal surface for all 46 owners at all
native contexts. Import the frozen legacy calibration; do not recompute a
threshold, extend the bank after seeing scores, or fit a new cohort rule.

The primary visual-support disposition remains:

- `resolved_tested_localization_support`;
- `persistent_no_tested_localization_support`; or
- ambiguity-bound flip.

The estimand is category-field support at owner geometry after a forced
category query. It is not per-owner proposal probability and must never be
named as one.

### S2: frozen native-prefix reachability ladder

For every newly resolved native FN, reproduce the predecessor's owner-level
ladder over native prefixes only:

- root / row-boundary / terminal support;
- before-or-at versus after-frontier support;
- continue-versus-stop gate;
- category route rank one / top three;
- same-category physical-owner rank one;
- covered-other versus uncovered-other competitor; and
- the favorable top-three and exact crossing-boundary cohorts.

Forced-continue rows remain excluded.

### S3: crossing release and realization, conditional

Run the inherited crossing-boundary release/realization protocol only if image
`2299` contains at least `8` frozen-definition crossing-favorable owners.
If fewer are available, report the count and stop this branch. If opened,
classify the existing three branches without retuning:

- displaced by another owner;
- release lost after the crossing boundary; or
- row realization failed after release.

The geometric relation stratification is CPU-only downstream of this capture
and is bundled with it. The invalid neutral-row insertion control is closed
and must not be repeated.

### S4: input-side spatial-position diagnostic

The sorted checkpoint's target order is exactly `(y1, x1)` raster order, so
owner position and due-row index are confounded after any row is emitted. The
only clean input-side position readout is therefore the root context.

Using the frozen legacy root-context rows, then applying the same readout to
image `2299`, report category-field support as a function of normalized owner
center, Manhattan/raster distance to the visual-token-block end, normalized
area, and same-category crowding. Position is descriptive and must not be
interpreted as a RoPE/attention cause. Image `2299` is useful here because its
owners are comparatively scale-homogeneous and quadrant-balanced.

## Frozen gates and stop rules

1. **Authority gate:** panel/file/image/calibration identities and the
   `46 = 38 person + 8 tie` owner composition must match exactly.
2. **Native-rollout gate:** the rollout must stop naturally and be
   nontruncated. If native FN `< 15`, publish S0 and stop as underpowered.
   With `15-19` FNs, later proportions are descriptive only. With `>=20`, the
   frozen directional stress test is considered materially powered, while
   still respecting one-image dependence.
3. **Calibration-transfer gate:** this gate owns a validity decision only when
   image `2299` supplies at least `10` strict native-TP controls. In that case,
   at least `80%` must have frozen-rule support at their exact native due
   boundary; below the floor, report the arm as calibration-nontransferring
   and do not interpret FN dispositions. With fewer than `10` strict native
   TPs, report `calibration_transfer_underpowered`: retain continuous rows and
   show frozen-rule FN dispositions descriptively, but do not turn the single
   image's direction into a formal pass/fail of the scale prediction. The
   matcher threshold must not be relaxed to manufacture calibration controls.
4. **Crossing gate:** both the resolved-support FN cohort and the exact
   crossing-favorable cohort must contain at least `8` owners before S3 opens.
5. **Reporting gate:** state the new `2299` slice first and the unchanged
   legacy-12 slice second. A pooled 392-owner line is optional and descriptive
   only. It cannot replace either slice.

The full-canvas token-budget intervention and the matched-length neutral-row
control are closed predecessor lines and are explicitly outside this unit.

## Primary outputs

- exact native rollout admission and one-to-one owner/prediction ledgers;
- one row per owner with native TP/FN, apparent scale, frozen support
  disposition, root support, best native support context, and ambiguity bound;
- supported-FN reachability rows with source context IDs and channel ranks;
- conditional crossing/geometry rows if their gate opens;
- root-position tables and score-independent visual atlases;
- a two-slice report preserving all legacy denominators unchanged; and
- a self-sealed receipt with exact input/output/source digests.

## Not claimed

- No population prevalence from one image.
- No causal claim that image position, raster distance, object scale,
  crowding, or prefix state causes a miss.
- No new calibration, owner phenotype, threshold, candidate bank, or matcher.
- No architecture, loss, training, or production promotion.
- No use of pooled 13-image numbers to rewrite legacy results.
