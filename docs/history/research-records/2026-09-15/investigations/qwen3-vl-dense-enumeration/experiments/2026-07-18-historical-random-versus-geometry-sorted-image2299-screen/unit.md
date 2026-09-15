---
title: Historical Random versus Geometry-Sorted Image 2299 Next-Row Screen
description: A bounded comparison of the matched checkpoint-3668 adapters on one densely relabeled image under identical prompt, prefix, candidate rows, precision, and decoding conditions.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: complete
unit_id: 2026-07-18-historical-random-versus-geometry-sorted-image2299-screen
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-07-18
---

# Historical Random versus Geometry-Sorted Image 2299 Next-Row Screen

Execution is complete. See [results](results.md).

## Question

Given the same densely relabeled image, exact compact prompt, already emitted
rows, and candidate object rows, do the historical random-order and
geometry-sorted pure cross-entropy adapters assign materially different
probability to the next physical person?

This unit asks what changed under the two historical supervision policies. It
does not assume that either policy learned a ledger, an explicit instance
pointer, or an order-independent traversal algorithm.

## Benchmark adapters

Random-order adapter:

```text
/data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/compact_full_fullobj_random_sft_bsz16_4epoch_tokenrows_v2/compact-full-fullobj-random-sft-bsz16-4epoch-tokenrows-v2/v1-20260601-062428/checkpoint-3668
```

Geometry-sorted adapter:

```text
/data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/compact_full_fullobj_sorted_sft_bsz16_4epoch_tokenrows_v2/compact-full-fullobj-sorted-sft-bsz16-4epoch-tokenrows-v2/v1-20260601-062429/checkpoint-3668
```

Both use the Qwen3 Vision-Language 2-billion-parameter base, four training
epochs, effective batch size 128, Low-Rank Adaptation rank 8 and alpha 32, the
same compact coordinate-token row schema, and one training seed. Their known
controlled training difference is the realized target-row ordering policy.
The compact no-separator prompt text is independent of that policy.

This pair is historical and must not be compared as a matched pair with the
current Weight-Decomposed Low-Rank Adaptation step-4887 checkpoints.

## Frozen image and object authority

- Image identifier: `2299`.
- Data record: `/data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000/val.coord.jsonl`, line 28.
- Image size: `1216 x 736` pixels.
- Human relabel: 38 `person` instances and 8 `tie` instances.
- Relabel record SHA-256 checksum:
  `ce19853c74a595f22cc183ce450e561f2da3216e54a1e499cfbca1be7e1c425b`.
- Source image SHA-256 checksum:
  `cd7199a37188c9ac6481520175866cbd78fa6fba35f4290bb0b60c78afcb2df3`.

Person ranks are zero-based ranks among the 38 `person` annotations in their
serialized geometry-sorted order. They are not global indices among all 46
annotations. The shared parent contains person ranks 0 and 1. The four frozen
treatments are:

| Person-only rank | Global serialized index | Bounding box | Annotation identifier |
|---:|---:|---|---:|
| 2 | 2 | `[625, 86, 711, 356]` | `-2` |
| 3 | 3 | `[331, 87, 423, 369]` | `-23` |
| 4 | 4 | `[708, 91, 784, 368]` | `-9` |
| 14 | 19 | `[639, 266, 734, 514]` | `-6` |

This explicit identity contract prevents confusion between person-only rank 14
and global serialized index 14, which is person-only rank 9. These are forced
teacher rows, so the result is a descriptive cross-adapter comparison, not
evidence about a naturally reached historical checkpoint state.

## Competing explanations

1. **Different exact-row transition distribution.** The adapters rank
   not-yet-emitted canonical rows differently under the identical forced state.
2. **Different teacher-forced coordinate likelihood.** Both favor `person`,
   but the observed exact-row contrast enters at one or more coordinate slots.
3. **Only a scale or confidence difference.** Candidate ordering is similar,
   while all row probabilities are sharper or flatter in one adapter.
4. **No stable difference.** Historical aggregate metrics came from later
   rollout drift, malformed rows, or one-seed variation rather than a reliable
   immediate next-row rule.

## Stage One: complete-row likelihood screen

Run eight independent arms in full-model 32-bit floating point:

```text
2 adapters x 4 treatment owners (person ranks 2, 3, 4, and 14)
```

Every arm uses the same image bytes, prompt token identifiers, parent rows,
candidate rows, and scoring code. The four treatment owners match the completed
step-4887 image-2299 panel, but their canonical relabel rows are rebuilt for
this historical comparison; no step-4887 donor token sequence is reused.
Owner rank 2 is the canonical next row after parent ranks 0 and 1. Owners 3, 4,
and 14 intentionally skip earlier rows and are challenge states increasingly
outside the geometry-sorted adapter's training trajectory. They are interpreted
separately rather than pooled as exchangeable states.

At every state, teacher-score all 46 frozen relabel rows -- 38 `person` and 8
`tie` rows -- plus the terminal token. Retain:

- unnormalized complete-row log probability;
- token-count-normalized complete-row log probability;
- description-span log probability;
- each of the four coordinate-token log probabilities;
- terminal-token log probability;
- rank of the just-emitted owner;
- candidate-restricted probability assigned to emitted versus not-yet-emitted
  people;
- candidate-restricted probability assigned to `person` versus `tie` rows;
- adapter-to-adapter rank correlation over all 38 people.

The complete row is primary. Coordinate decomposition explains where a
difference enters the teacher-forced row but does not establish whether the
underlying cause is object discovery, instance choice, or coordinate binding.
Because the 46 rows are not exhaustive of all possible model continuations,
their normalized values are always called candidate-restricted probability,
not total model probability mass.

## Stage Two: conditional paired sampling

Run paired low-temperature seeds only if Stage One finds a clear exact-row
contrast worth testing behaviorally. Interpret the canonical owner-2 arm and
the three skipped-prefix challenge arms separately. Use temperature `0.4`,
top-p `0.95`, repetition penalty `1.0`, and a nine-token maximum horizon.
Sample exactly one successor row and stop.

Stage Two asks whether the likelihood contrast changes physical-owner sampling.
It is not launched merely because one isolated coordinate logit differs.

## Required controls

- Exact model base, adapter path, adapter checksum, prompt tokens, prefix
  tokens, image checksum, precision, and device must be recorded per arm.
- The historical `coord_offset_adapter` must be installed and its saved
  coordinate-row tensors verified after loading; a loader that merely accepts
  the Low-Rank Adaptation weights is insufficient.
- Candidate rows come only from the frozen human relabel.
- Previously emitted owners remain in the candidate set so suppression can be
  measured rather than assumed.
- Raw scores and generations are retained before matching or filtering.
- Physical owner matching uses the human relabel and reports ambiguous matches
  separately.
- Terminal is compared with the row-start next-token probability. A one-token
  terminal score must not be compared with a length-normalized seven-token row
  score.

## Decision rules

- **Canonical-prefix training-policy signal:** an adapter contrast appears in
  owner rank 2, survives complete-row scoring, and changes paired successor
  ownership without increasing malformed rows.
- **Skipped-prefix robustness signal:** owners 3, 4, or 14 differ from the
  canonical arm in a way consistent with their increasing deviation from the
  geometry-sorted training trajectory.
- **Teacher-forced coordinate-likelihood signal:** description behavior is
  similar but one or more coordinate slots produce the exact-row contrast.
- **Confidence-only phenotype:** candidate ranks remain highly correlated and
  top candidates remain stable, conditional entropy changes, and sampling
  ownership does not change despite an approximately affine score-scale
  difference.
- **No promoted difference:** a contrast is isolated to one challenge state,
  exists only in one coordinate token without a complete-row effect, or
  disappears in full-model 32-bit floating point.

Any positive result remains evidence about one historical seed and one forced
state. It can justify a small held-out screen, not new architecture or training.

## Existing evidence not to repeat

The prior 64-state smoke already found mixed checkpoint signatures under the
common compact prompt. Its verified artifact root is:

```text
/data/CoordExp/outputs/analysis/autoreg_object_rollout/sorted_random_no_newline_phenotype/fullobj_purece_random_vs_sorted_ckpt3668_phase_a3_2_smoke_retry2
```

This unit contributes the newly relabeled dense image, complete-row candidate
distribution, full-model 32-bit execution, and direct connection to the recent
image-2299 transition result.

## Artifact root

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-18-historical-random-versus-geometry-sorted-image2299-screen
```
