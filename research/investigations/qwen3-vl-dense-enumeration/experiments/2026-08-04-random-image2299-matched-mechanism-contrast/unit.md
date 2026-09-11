---
title: Random Image 2299 Matched Mechanism Contrast
description: Repeats the image-2299 native and owner-accessibility capture with the original random-order step-4887 checkpoint under the sorted slice's frozen inputs and scorer.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: authorized
unit_id: 2026-08-04-random-image2299-matched-mechanism-contrast
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: complete
updated: 2026-08-04
---

# Random Image 2299 Matched Mechanism Contrast

## Question

For the refined image `2299` row containing `46 = 38 person + 8 tie`
physical owners, does the original random-order step-4887 checkpoint expose a
materially different native coverage and conditional coordinate landscape from
the completed geometry-sorted step-4887 checkpoint?

This is a matched checkpoint contrast, not a new prevalence estimate. The
image, prompt, HF fp32 runtime, `rp=1.0` greedy policy, `3084` horizon, strict
one-to-one matcher, canonical category query, score-independent 17-role box
bank, and likelihood readout remain fixed. Only the adapter and selected-token
embedding delta change from sorted step-4887 to random step-4887.

## Decision-bearing observations

1. **Native owner accounting.** Report random TP/FN/unmatched/invalid rows and
   natural stop, then the exact sorted-to-random retained/gained/lost owner
   sets.
2. **Matched root landscape.** At the empty assistant history, compare every
   owner's exact-box score, best local candidate score, local concentration,
   category-population peak lift, owner-relative candidate rank, and margin.
   This is the cleanest checkpoint contrast because the textual history is
   identical.
3. **Native self-prefix landscape.** Repeat the frozen census over the random
   checkpoint's own natural greedy prefixes. These rows measure the combined
   effect of checkpoint and self-selected history; they are not a pure
   checkpoint intervention.
4. **Conditional greedy realization.** Preserve the free four-coordinate
   greedy sidecar after the forced category query. Separate finite teacher-
   forced support from actual coordinate argmax realization.

## Strongest alternatives

- Any apparent random advantage may be entirely explained by a different
  native trajectory rather than a stronger root representation.
- The sorted checkpoint's frozen support thresholds may not transfer to the
  random checkpoint, just as they failed their preregistered transfer gate on
  sorted image `2299` itself.
- A finite 17-role candidate bank can miss a real mode; `persistent` under this
  bank is not proof that the owner can never be generated.
- Sequence likelihood over four coordinate tokens is conditional geometry
  support after a forced category query, not the marginal probability of
  proposing that physical owner from the vocabulary action space.

## Frozen inputs and runtime

| Item | Frozen value |
| --- | --- |
| Ground truth | `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-04-sorted-prospective-13-image-panel-admission/evaluation-inputs/human-refined-13.coord.jsonl` |
| Panel SHA-256 | `01086b139fa23983697492fdb535b5154429277803e8f12b243f9a031d1451f8` |
| Image-2299 authority row SHA-256 | `ce19853c74a595f22cc183ce450e561f2da3216e54a1e499cfbca1be7e1c425b` |
| Image bytes SHA-256 | `cd7199a37188c9ac6481520175866cbd78fa6fba35f4290bb0b60c78afcb2df3` |
| Random config | `configs/coordexp_infras/infer/qwen3_vl_2b_desc_first_random_step4887_human_refined13_hf_fp32_rp1p0.yaml` |
| Policy | HF fp32, greedy, temperature `0`, top-p `1`, repetition penalty `1.0`, seed `0` |
| Horizon | `max_new_tokens=3084`; native run must stop with `im_end` before the horizon |
| Sorted reference | `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-04-sorted-image2299-prospective-mechanism-extension/20260804T105647Z/` |

The official COCO 22-owner row for image `2299` is forbidden.

## Analysis policy

The completed sorted slice failed its frozen calibration-transfer gate at
`14/19`. Therefore this unit does not promote the legacy sorted thresholds to
a random-checkpoint classifier. It reports:

- continuous landscape measurements as primary;
- random native-TP transfer controls;
- the sorted frozen thresholds only as explicitly labeled descriptive
  sensitivity; and
- threshold-sweep stability where a binary comparison is useful.

No random/sorted pooled recoverability fraction is created.

## Stop rule

Stop after one exact native rollout, the full frozen candidate/context capture,
and CPU matched analysis. Do not open downstream forced-continuation,
crossing-boundary, high-resolution, sampling, training, or architecture work
unless this contrast leaves a concrete unresolved discriminator.

If the random native rollout is truncated, malformed, or fails exact runtime
identity, do not substitute a historical approximate artifact. If the random
native-TP controls fail threshold transfer, retain continuous evidence and
withhold formal binary FN dispositions.

The observed `rp=1.0` full-canvas repetition opened one narrow post-hoc
decode-only discriminator: repeat the same random checkpoint, image, prompt,
greedy policy, and horizon with the checkpoint's usual production
`rp=1.10`. This does not enter the matched checkpoint contrast or recalibrate
the frozen support rule; it only tests whether the repeated coordinate basin is
processor-sensitive.

## Claim boundary

- One image cannot establish population prevalence.
- Root-context differences are matched checkpoint evidence; later self-prefix
  differences also include each checkpoint's endogenous trajectory.
- Tested local support is not visual-tower decodability, owner proposal
  probability, or proof of future rollout reachability.
- No architecture, objective, training recipe, or production behavior is
  promoted by this unit.
