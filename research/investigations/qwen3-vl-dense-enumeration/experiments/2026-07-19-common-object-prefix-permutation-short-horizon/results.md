---
title: Common Physical Objects under Different Prefix Permutations and a Short Future Horizon Results
description: A paired four-row probe finds that earlier prefix order changes later coordinates and physical-owner routes even when the covered object set and final one or two rows are identical.
type: investigation
role: research-result
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: complete_for_unit
unit_id: 2026-07-19-common-object-prefix-permutation-short-horizon
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified_bounded
conclusion_status: order_sensitive_route_and_geometry_state_supported
updated: 2026-07-19
---

# Common Physical Objects under Different Prefix Permutations and a Short Future Horizon Results

## Verdict

The geometry-sorted pure-cross-entropy checkpoint does not reduce a detection
prefix to an order-invariant set of already emitted objects. With the image,
prompt, exact complete-row multiset, covered physical-object set, row count,
and final one or two rows held fixed, changing only earlier row order changes
the next four generated rows in `173 / 216` paired trajectories. The strict
physical-owner sequence changes in `116 / 216`, and the final strict unique
owner set changes in `99 / 216`.

This is not mainly a syntax or terminal-token failure. The boolean four-row
completion result changes in only `8 / 216` pairs. Three additional pairs are
incomplete on both sides but generate different row counts. At aligned row
positions, `316` differences preserve the class description but change
coordinates, `158` change both description and coordinates, and none change
only the description while preserving identical coordinates. These
coordinate-span differences must not automatically be called localization
drift: in a same-class scene they can encode a different physical owner.

The strongest bounded interpretation is:

> Earlier serialized history remains part of the model's effective state. It
> changes later object routes and which same-class geometry basin is selected.
> A common suffix does not erase this influence.

This is not evidence that Qwen3-VL has no useful commit-like behavior, and it
does not prove that an explicit covered-set memory is required. It shows that
the native state learned by this checkpoint mixes covered-object information,
recent order, traversal habit, and phrase-to-geometry decisions. It also shows
why identical serialized trajectories are the wrong training target: `17` of
the `116` strict-owner sequence changes still finish with the same strict
unique-owner set within four rows.

## Evidence Scope

- Primary model: Qwen3-VL 2 billion parameters with the description-first,
  geometry-sorted, pure-cross-entropy plus token-type-gate
  Weight-Decomposed Low-Rank Adaptation checkpoint at step `4,887`.
- Runtime: Hugging Face, full-model 32-bit floating point, repetition penalty
  `1.0`.
- Images: `18380`, `9400`, `9590`, and `19109`.
- Promoted image-depth cases: five; image `19109` contributes depths six and
  ten.
- Prefixes: six or ten exact complete rows.
- Comparisons: `24`; one predefined comparison was omitted because it produced
  the exact same row order as another arm.
- Paired continuations per comparison: one greedy trajectory and eight
  sampled trajectories with seeds `101` through `108`, temperature `0.4`, and
  top-p `0.95`.
- Future horizon: up to four complete generated rows, with exact generated
  token identifiers appended after every accepted row.
- Total paired trajectories: `216`, comprising `24` greedy and `192` sampled
  pairs.
- Resolved inference configuration fingerprint:
  `6af0ff8b638a4464cfb993e74d0f2fa2fb87789f81b83f0f7fd851e62d861df5`.

The five cases were selected because the one-row screen already showed an
order-sensitive result. These counts therefore measure the behavior of an
activated case panel; they are not population prevalence estimates. Sampling
seeds and prefix permutations are repeated measurements, not independent
images.

Generated rows were matched against each image's full artifact-local COCO-80
ledger, rather than only against the six or ten prefix entities. That ledger
is still incomplete and its strict owner requires same-category intersection
over union of at least `0.5`. A strict-set difference can therefore reflect a
real physical switch, a shifted or partial box around the same entity, or a
true but unannotated entity. It is an upper bound on physical coverage
instability, not a final estimate.

## Aggregate Results

| Paired outcome | Count | Meaning |
|---|---:|---|
| Any generated row text changes | `173 / 216` | Earlier order usually changes some part of the four-row trajectory. |
| Strict physical-owner sequence changes | `116 / 216` | More than coordinate jitter alone is present. |
| Final strict unique-owner set changes | `99 / 216` | Many changed routes do not reconverge under strict matching within four rows. |
| Strict unique-owner count changes | `66 / 216` | Short-horizon matched coverage quantity often changes. |
| Owner sequence changes but final strict set agrees | `17 / 216` | Benign route reordering or reconvergence also exists. |
| Four-row completion boolean changes | `8 / 216` | Early termination is real but not the dominant effect. |
| Completion boolean or generated-row count changes | `11 / 216` | Includes three pairs incomplete on both sides at different lengths. |

No alternative order is uniformly better. Summed across the repeated paired
comparisons, canonical trajectories contain `499` strict unique-owner hits and
alternative trajectories contain `488`; the reverse-with-final-two arm is
exactly balanced at `104` versus `104`. Individual changed routes help some
trajectories and hurt others.

Across the `864` aligned row positions:

| Row-level relation | Count |
|---|---:|
| Exact same row | `387` |
| Same description, different coordinates | `316` |
| Different description and coordinates | `158` |
| Different description, identical coordinates | `0` |
| One side has no corresponding row | `3` |

The absence of description-only changes is not a universal law. In this panel,
however, it shows that the effect is not merely a class-token preference shift.
It does not by itself distinguish altered physical-owner selection from altered
extent for the same owner.

### Geometry conditional on a stable owner

The independent audit resolves that ambiguity for the first generated row.
Among the `67` Stage 2 pairs whose first rows strictly match the same physical
owner, `60` boxes are exactly identical and `66` have prediction-to-prediction
intersection over union of at least `0.90`; the remaining pair is still at
least `0.50`. Stage 1 has the same pattern: among `85` same-owner pairs, `69`
boxes are exact, `80` are at least `0.90`, and `84` are at least `0.75`.

Therefore the stronger reading is:

> Prefix order often changes which object or same-class geometry basin wins.
> Once the same strict physical owner wins, its box is usually transcribed
> almost identically.

This makes object selection and phrase-to-instance binding a more plausible
first treatment surface than generic coordinate smoothing.

## Dependence on Perturbation and History Distance

| Alternative prefix order | Paired trajectories | Strict final-set changes | Completion or row-count changes |
|---|---:|---:|---:|
| Historical-random relative order, final two rows fixed | `45` | `17` | `3` |
| Reverse earlier rows, final two rows fixed | `45` | `22` | `3` |
| Reverse earlier rows, final one row fixed | `45` | `31` | `2` |
| One adjacent swap before the final two rows | `36` | `10` | `2` |
| Seeded shuffle, final two rows fixed | `45` | `19` | `1` |

The adjacent-swap result matters: even a mild one-pair order change alters the
strict final set in `10 / 36` pairs. Extreme off-policy prefix corruption
therefore cannot explain the whole effect. It remains a valid alternative for
the larger reverse and shuffle interventions because prefix likelihood was not
measured in this pilot.

The reverse comparison is more sensitive with a final one-row suffix
(`31 / 45`) than with a final two-row suffix (`22 / 45`). This is consistent
with a recency contribution. It is not a clean memory-decay estimate because
the final-one condition also reverses one additional earlier row.

## Heterogeneity across Images

| Image and depth | Pairs | Row-text changes | Strict final-set changes | Completion or row-count changes | Bounded reading |
|---|---:|---:|---:|---:|---|
| `18380`, depth 6 | `45` | `34` | `18` | `0` | Contains both reconvergence and changes among real people and table objects. |
| `9400`, depth 10 | `45` | `29` | `8` | `1` | Most stable case; many changes are geometry-only. |
| `9590`, depth 10 | `45` | `35` | `19` | `10` | Crowded tabletop routes are sensitive and account for most length changes. |
| `19109`, depth 6 | `36` | `30` | `17` | `0` | Dense person and motorcycle choices change, with difficult geometry. |
| `19109`, depth 10 | `45` | `45` | `37` | `0` | Strongest strict-set and covered-owner recurrence sensitivity, but also the least reliable strict geometry matching. |

The selected enlarged-image reviews establish that several switches are
between real neighboring entities. They also show that many automatic
`unmatched` outcomes are shifted, narrow, oversized, or partial boxes around a
real semantic region. The six reviewed paired trajectories provide three
different kinds of evidence:

1. Image `18380` contains a benign route change: the person and cup appear in
   a different order, but both trajectories recover the same strict unique
   owner set within the horizon.
2. Image `9590` contains both directions of value change. Under one adjacent
   swap, the first two strict owners agree, but the canonical path later emits
   a spoon that clearly corresponds to a real remaining table object while
   the alternative path terminates. Under another reversal, the alternative
   path reaches an additional valid cup rather than merely extending the
   sequence with unsupported output.
3. Images `9400` and `19109` show real route switches among nearby computers,
   people, and motorcycles. The dense motorcycle scene also makes exact
   instance identity difficult: several unmatched boxes land on physically
   plausible motorcycle regions but are shifted or mixed across neighbors.
   Its covered-owner recurrence is therefore credible artifact-level evidence,
   but not every individual recurrence is a publication-grade duplicate claim.

The robust claim is therefore that prefix order changes the emitted route and
the selected geometry basin, sometimes harmlessly and sometimes with a real
short-horizon coverage consequence. The exact population rate at which it
changes true physical-object coverage remains unresolved.

## Covered-Object Recurrence

The most constructive failure occurs in image `19109`. Across unique arm
executions, images `18380`, `9400`, and `9590` have no strict recurrence of a
frozen-prefix owner. Image `19109` has `10` such recurrences at depth six and
`34` at depth ten.

At depth ten, the fixed-final-two arms all end with the same two exact complete
rows, yet their strict covered-prefix recurrence counts over `36` generated
row attempts per arm are:

| Prefix order | Covered-prefix recurrences |
|---|---:|
| Current geometry-sorted order | `9` |
| Historical-random relative order | `4` |
| Reverse earlier rows | `6` |
| One adjacent swap | `4` |
| Seeded shuffle | `8` |

The recurrent owners are mainly motorcycles `gt_0017` and `gt_0021`. Some
reappear at generated row three and again at row four. This provides a lower
bound on influence distance: earlier order remains behaviorally active after
two byte-identical suffix rows and additional generated rows. Dense overlapping
motorcycles also make strict instance matching difficult, so enlarged-crop
review is required before treating every recurrence as a publication-grade
physical duplicate.

## What This Unit Supports

1. **The covered set alone is not a sufficient native state description.**
   Prefixes with the same objects and common suffix produce different future
   trajectories.
2. **The effect can survive two identical final rows.** It is attenuated by a
   longer common suffix, not erased.
3. **The effect is not limited to extreme permutations.** One adjacent swap can
   change the future strict set.
4. **Raw differences concentrate in coordinate spans, but stable-owner geometry
   is usually stable.** The key unresolved boundary is selection or
   phrase-to-instance binding, not generic box noise.
5. **Some route freedom is harmless.** Different owner orders can reconverge to
   the same strict set within four rows.
6. **Generic terminal or malformed collapse is not the main explanation.**
   Nearly all paired trajectories preserve the ability to emit complete rows.
7. **Native covered-owner suppression is not durably reliable.** In image
   `19109`, previously covered motorcycles can re-enter the generated route
   several rows later, and earlier order changes the recurrence rate.

## What This Unit Does Not Establish

- that an explicit covered-set carrier, object slot, query, or detector is
  necessary;
- that all strict-set changes are changes in true physical discovery;
- that arbitrary prefix permutation should be a training augmentation;
- that one canonical geometry-sorted sequence is the correct target;
- that the random-order training policy is better or worse;
- a population effect size across COCO; or
- the internal layer or attention route that carries the order-sensitive
  state.

The newly launched matched random-order run has not produced a checkpoint.
Its exact replication remains pending. The historical random checkpoint was
used only to construct one relative-order arm and is not a matched model
comparison.

## Training and Architecture Decision

Do not promote a slot module, explicit covered-set memory, permutation-
invariance loss, or long-horizon training objective from this result alone.
Also do not train the model to reproduce one exact next row: the result contains
both useful alternate routes and harmful short-horizon divergence.

The next decision should locate where the divergence first appears. At the
first shared next-row boundary, and at later row boundaries in image `19109`,
score complete candidate rows for:

1. a real uncovered object later reached by one trajectory;
2. the other trajectory's selected object;
3. a previously covered object, especially recurrent motorcycles in image
   `19109`; and
4. terminal output.

Report phrase-token score and coordinate-token score separately.

- If the useful uncovered row already loses under one frozen prefix, the first
  training screen should directly prefer that valid row over a repeated row,
  terminal output, or habitual alternative at the same self-rollout state.
- If class-description scores remain similar while complete coordinate scores
  choose different same-class instances, treatment should first target
  phrase-to-instance binding rather than generic coordinate smoothing.
- If first-boundary candidate scores remain similar and divergence emerges
  only after generated rows are appended, treatment should target rollout-state
  robustness or future unique coverage rather than immediate ranking.

This score probe is smaller and more discriminating than immediately training
a ledger, controller, or general order-invariance objective. If a covered row
genuinely regains margin over verified uncovered rows, the first small training
screen should rank any verified uncovered complete row above covered or
repeated rows and above terminal output while verified objects remain. Its
success measure must be three-to-four-row unique physical coverage, not
agreement with one prescribed next object.

## Artifacts

Stage 1 one-row screen:

`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-19-common-object-prefix-permutation-short-horizon/stage1-screen-sorted-pure-cross-entropy-step4887-v1/`

Stage 1 selected visual comparisons:

`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-19-common-object-prefix-permutation-short-horizon/stage1-screen-sorted-pure-cross-entropy-step4887-v1/visual-review-selected-switches-v1/`

Stage 2 four-row artifacts:

`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-19-common-object-prefix-permutation-short-horizon/stage2-four-row-sorted-pure-cross-entropy-step4887-v1/`

Stage 2 selected trajectory sheets and their selection manifest:

`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-19-common-object-prefix-permutation-short-horizon/stage2-four-row-sorted-pure-cross-entropy-step4887-v1/visual-review-selected-trajectories-v1/`

Stage 2 checksums:

- image `18380`, depth 6:
  `a09b45b1a17fabef0186534a1e05ee3a20e24e3e8a71ab744b59e54bdc898560`;
- image `9400`, depth 10:
  `b3f70bdb9a53d8e3749d2c5f65614a8271873ec850b324507aac2958cd635941`;
- image `9590`, depth 10:
  `a32b43ff2c789224faf9a81539248304ea98866da8fdedc719f6a96ed6ee525d`;
- image `19109`, depths 6 and 10:
  `85eee1709618cb20025c93204ff549dfef6ab19651638022926cfd02c9214070`.

Experiment-local implementation:

- `scripts/research/build_common_object_prefix_permutation_cases.py`;
- `scripts/research/run_same_covered_set_prefix_order_probe.py`;
- `scripts/research/summarize_same_covered_set_prefix_order_probe.py`.
