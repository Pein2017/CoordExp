# FP is a mixture of repetition, missing annotation and detection errors

Status: **lead-accepted bounded visual audit; not replacement ground truth**.
The user ruled that prediction-to-previous-prediction IoU>.95 directly
classifies duplication. That rule owns all484 strict-repeat FP; no additional
visual confirmation is required. The active visual audit covers only the
remaining non-repeat predictions.

## Exact population versus visual sample

The accepted round1 train256 run has1955 GT objects,1262 matched predictions
and1094 unmatched predictions at global category-consistent IoU50.

| Exact structural category | FP count | Visual sample |
|---|---:|---:|
| Strict geometric repeat, preceding prediction IoU>.95 | 484 | Not required; automatic rule |
| Non-repeat, same-category GT IoU>=.1 | 269 | 20 |
| Remaining non-repeat, other-category GT IoU>=.5 | 14 | All14 |
| Remaining non-repeat, weak GT relation | 327 | 20 |
| Total | 1094 | 54 non-repeat cases |

Strict repeats are44.24% of all FP. Independently, four capped images contain
511 FP (46.71%); these sets overlap and must not be added. FP occur on130 of
the256 images. This is prediction-weighted burden, not the frequency of an
error type across images or a claim that those many distinct entities exist.

The visual sample is fixed-seed20260909, sampled without replacement inside
the three non-repeat strata:54 predictions from39 images. An initial64-item
packet included10 strict-repeat examples; those were excluded from the active
visual denominator after the user's ruling, without replacing any non-repeat
sample. Extra capped-image-only checks were cancelled. The original packet
is preserved provenance, not the current54-case reading scope.

## Visual review results

Each active case was inspected in full-image context and a pixel-preserving
crop. The crop's raw panel remains unobstructed; a separate panel draws the
target box and nearby GT. These are model-generated visual judgments, with
uncertainty retained, not a certified full-image annotation exercise.

| Primary explanation in the54 reviewed non-repeat FP | Count |
|---|---:|
| Visible separate instance not covered by the current GT | 14 |
| Localization/extent error on a visible instance | 7 |
| Category error on a visible entity | 5 |
| One box groups multiple instances | 5 |
| Annotation extent, grouping or depiction-scope ambiguity | 6 |
| Visually repeated instance despite pair IoU<=.95 | 3 |
| Unresolved from the available pixels/semantics | 14 |
| Confirmed unsupported-entity hallucination | 0 |

These are **sample counts, not whole-population percentages**. Sampling
fractions differ sharply between strata. In particular, all14 high-other-
category-overlap cases were inspected, whereas only20/327 weak-GT-relation
cases were sampled. The machine summary retains design-weighted exploratory
point estimates for later analysis, but they do not account for reviewer error
and are not adopted as precise population prevalence here.

The zero in the last row does not establish zero hallucination rate. Fourteen
cases remain unresolved, and most non-repeat predictions were not visually
reviewed. Here unsupported-entity hallucination means no corresponding entity
is visually supported in the target region; wrong-category naming of a real
entity is counted separately. A broader definition of hallucination would
require a different, explicitly stated counting rule.

Independent axes matter:46/54 cases have a supported single or multiple visual
entity;8 remain uncertain on entity existence. Only20/54 boxes were judged
acceptable geometrically. A visible object is therefore **not** permission
to reinforce its current category and box unchanged. Category judgments are
32 correct,7 incorrect and15 uncertain; these differ from the primary-reason
counts because an item can have both redundancy and category problems.

## Concrete inspected examples

- **Missing current GT coverage:** `540107:p15` encloses a visibly distinct
  small elephant, with no GT overlap. `152252:p3` encloses a separate seated
  boy in a group photograph, while available GT boxes cover other people.
  These are not visually empty regions. [Elephant crop](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-09-fp-visual-distribution/crops/540107_p015.png), [person crop](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-09-fp-visual-distribution/crops/152252_p003.png).
- **Real entity, wrong class:** `315195:p11` says cup but encloses a visibly
  crushed plastic bottle. Its any-category GT IoU is0.852; matching failure
  here does not mean the region lacks an object. [Crop](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-09-fp-visual-distribution/crops/315195_p011.png).
- **Extent:** `64010:p24` covers a real black car but includes substantial
  umbrella/foreground and adjacent-car area; GT IoU is0.426. Occlusion and
  visible-versus-inferred extent still deserve care. [Crop](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-09-fp-visual-distribution/crops/64010_p024.png).
- **Several instances in one box:** `502725:p36` spans portions of two
  neighboring overlapping knives rather than one complete knife. [Crop](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-09-fp-visual-distribution/crops/502725_p036.png).
- **Annotation granularity:** `274509:p28` targets one narrow book-spine
  region inside a wider book annotation that encloses adjacent books; the
  target is also vertically tight. A mismatch is not proof that the book
  was invented. [Crop](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-09-fp-visual-distribution/crops/274509_p028.png).

“Current GT” refers to the retained processed annotation set. This audit does
not establish whether absent instances were missing in the original COCO
annotations, filtered during preparation, or outside an annotation-scope rule.
Likewise, objects depicted inside photographs or paintings require scope care;
visual support is not a claim that a physical object exists in the photographed
room at that location.

## Lead corrections and verification

Three independent visual readers covered17 disjoint cases each; root covered
the remaining3. Root additionally checked11 reader cases against original
pixels/crops, for14 root-inspected non-repeat cases total. Five explicit
adjudications are preserved separately rather than overwriting reader outputs:

- Glass labeled `cup` in GT versus a predicted wine glass: its support/stem is
  occluded, so GT alone cannot prove a category error.
- A predicted motorcycle overlaps a person annotation, but the crop resembles
  a rider on a two-wheeler; the vehicle category is not visually decidable.
- A spoon/knife disagreement exposes only a handle; the working end is hidden.
- A chair box intersects seated legs/support at low resolution; multiple
  visible chairs were not established.
- The table example's serving-board/table extent is ambiguous; objects resting
  on a table do not themselves make it a multi-instance table box.

Four primary labels became uncertain; one retained its grouping/scope label
with corrected independent axes. This guards against using GT disagreement
itself as supposed visual proof of error. No further review loop was opened.

Checks passed: exact54 IDs and disjoint ownership, valid label enums, actual
context/crop paths, retained uncertainty, correct sampling weights summing to
610 non-repeat FP, unchanged raw/case source hashes, and all39 selected image
hashes matching the executed image plan. No GPU or new inference was used.

## Bounded implications for the proposed learning direction

1. Keep the strict duplication rule separate; it need not consume visual-review
   effort or be treated as hundreds of independent newly hallucinated objects.
2. Do not assign a uniform negative reward to every annotation-relative FP.
   Real unannotated instances and annotation-granularity differences exist in
   the inspected sample, alongside genuine localization and category errors.
3. Entity existence, class, localization and single-instance binding need
   distinct supervision semantics. Even a visually real candidate is not an
   automatically correct box/label target.
4. This audit supplies no negative label for unresolved cases and no license
   to self-distill all unmatched predictions as truth. Recovering missing GT
   while retaining prior TP remains a separate training/outcome question.

No loss, reward, official GT, model architecture, inference code or shared
skill was changed. The requested FP check is complete; modeling/Pro reasoning
can now use this bounded evidence rather than equating FP with hallucination.

## Artifacts and reproduction

- [Final review summary](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-09-fp-visual-distribution/final-v2/review-summary.json), SHA256 `a679521017a9d889315706a653ef9d4dc5d0c98416aa73bcff498ad4f8d411a6`.
- [All54 judgments, original opinions and image links](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-09-fp-visual-distribution/final-v2/reviewed-cases.json), SHA256 `7ece21dfd32fee13539e1e9c5fc4757d7d8c040ac2fecf1c11b48b8d67c7790a`.
- [Lead acceptance](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-09-fp-visual-distribution/lead-acceptance.json), [active scope](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-09-fp-visual-distribution/active-review-scope.json) and [full1094-FP inventory](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-09-fp-visual-distribution/inventory.jsonl).
- [Native overview manifest](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-09-fp-visual-distribution/native-overviews/manifest.json); overview matching colors are reference-only and do not replace global matcher membership.

The frozen preparation script generated the initial packet. To recompute only
the accepted reduction, use an absent output directory:

```bash
PYTHONPATH=/data/CoordExp/.worktrees/research-probes python \
  /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-09-fp-visual-distribution/reduce_visual_reviews.py \
  --output-dir <absent-output-directory> \
  --overrides /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-09-fp-visual-distribution/lead-overrides.jsonl
```
