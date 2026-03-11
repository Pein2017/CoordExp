# GT Overlap Threshold Search on COCO1024 Val

Date: 2026-03-11

## Question

Given duplicate-heavy rollouts after Channel-B UL, what IoU or CIoU threshold is safe to use for a semantics-agnostic duplicate punishment rule?

Target dataset:

- `public_data/coco/rescale_32_1024_bbox_max60/val.coord.jsonl`

Relevant training handles:

- current duplicate gate uses IoU only:
  - `stage2_ab.channel_b.duplicate_iou_threshold`
  - default `0.90`
  - see `docs/training/STAGE2_RUNBOOK.md`
  - see `configs/stage2_two_channel/base.yaml`
- current IoU implementation for duplicate filtering:
  - `src/trainers/stage2_two_channel.py::_bbox_iou_norm1000_xyxy`
- CIoU formula used elsewhere in training:
  - `src/trainers/teacher_forcing/geometry.py::bbox_smoothl1_ciou_loss`

## Method

- Stream the full validation JSONL.
- Parse every GT `bbox_2d` in norm1000 space.
- For each image, compute pairwise GT-vs-GT overlap for all object pairs.
- Report:
  - all-pair maxima
  - different-desc maxima
  - same-desc maxima
  - threshold sweep counts for IoU and CIoU
- Treat CIoU as the score `ciou`, not the training loss `1 - ciou`.

Study artifact:

- summary JSON:
  - `output/analysis/gt_overlap_threshold_search_20260311/summary.json`
- top overlap CSVs:
  - `output/analysis/gt_overlap_threshold_search_20260311/top_diff_desc_pairs_by_iou.csv`
  - `output/analysis/gt_overlap_threshold_search_20260311/top_diff_desc_pairs_by_ciou.csv`

## Coverage

- images: `4951`
- GT bbox objects: `36273`
- within-image GT pairs: `243225`
- same-desc GT pairs: `85904`
- different-desc GT pairs: `157321`

## Main Result

There is no aggressive semantics-agnostic threshold that is perfectly safe on this validation GT.

Observed maxima:

- max IoU over all GT pairs: `0.9982394366`
- max CIoU over all GT pairs: `0.9982392473`
- max IoU over different-desc GT pairs: `0.9982394366`
- max CIoU over different-desc GT pairs: `0.9982392473`
- max IoU over same-desc GT pairs: `0.9526840950`
- max CIoU over same-desc GT pairs: `0.9525317346`

Implication:

- If you punish duplicates across all semantics only by overlap, then a truly zero-collision threshold must be set above `0.99824`.
- In practice that means `0.999` or higher.
- A threshold like `0.95` or `0.97` is not safe against GT collisions.

## Threshold Sweep

Different-desc GT collisions at each threshold:

| threshold | diff-desc IoU pairs >= thr | diff-desc CIoU pairs >= thr |
| --- | ---: | ---: |
| `0.80` | `236` | `236` |
| `0.85` | `192` | `190` |
| `0.90` | `145` | `142` |
| `0.92` | `102` | `101` |
| `0.95` | `55` | `53` |
| `0.97` | `21` | `21` |
| `0.98` | `7` | `7` |
| `0.99` | `2` | `2` |

All-pair GT collisions at each threshold:

| threshold | all-pair IoU pairs >= thr | all-pair CIoU pairs >= thr |
| --- | ---: | ---: |
| `0.80` | `245` | `245` |
| `0.85` | `198` | `196` |
| `0.90` | `150` | `147` |
| `0.92` | `106` | `105` |
| `0.95` | `57` | `55` |
| `0.97` | `21` | `21` |
| `0.98` | `7` | `7` |
| `0.99` | `2` | `2` |

Same-desc GT collisions at each threshold:

| threshold | same-desc IoU pairs >= thr | same-desc CIoU pairs >= thr |
| --- | ---: | ---: |
| `0.80` | `9` | `9` |
| `0.85` | `6` | `6` |
| `0.90` | `5` | `5` |
| `0.92` | `4` | `4` |
| `0.95` | `2` | `2` |
| `0.97` | `0` | `0` |
| `0.98` | `0` | `0` |
| `0.99` | `0` | `0` |

## Top GT Collisions

Top different-desc collisions are not rare only because of crowded same-class scenes. The val GT itself contains near-identical cross-class boxes.

Top examples by IoU:

1. `IoU=0.998239`, `CIoU=0.998239`
   - image: `images/val2017/000000183246.jpg`
   - pair: `truck` vs `car`
   - boxes: `(431,0,999,999)` vs `(432,0,999,999)`
2. `IoU=0.990356`, `CIoU=0.990355`
   - image: `images/val2017/000000205514.jpg`
   - pair: `chair` vs `couch`
3. `IoU=0.988844`, `CIoU=0.988833`
   - image: `images/val2017/000000100582.jpg`
   - pair: `pizza` vs `dining table`
4. `IoU=0.986974`, `CIoU=0.986953`
   - image: `images/val2017/000000039769.jpg`
   - pair: `couch` vs `bed`
5. `IoU=0.986004`, `CIoU=0.985996`
   - image: `images/val2017/000000578500.jpg`
   - pair: `chair` vs `couch`

Top same-desc collisions by IoU:

1. `IoU=0.952684`, `CIoU=0.952532`
   - image: `images/val2017/000000277051.jpg`
   - pair: `bird` vs `bird`
2. `IoU=0.951116`, `CIoU=0.950960`
   - image: `images/val2017/000000043314.jpg`
   - pair: `person` vs `person`
3. `IoU=0.941415`, `CIoU=0.941249`
   - image: `images/val2017/000000313182.jpg`
   - pair: `person` vs `person`

This matters because a semantics-agnostic threshold below about `0.953` will definitely suppress some legitimate same-class GT pairs, and a threshold below about `0.9983` will suppress some legitimate different-class GT pairs too.

## Recommendation

Use two reference thresholds rather than one:

1. `safe_hard_threshold = 0.999`
   - Use this if the rule is hard, semantics-agnostic, and you want zero GT collisions on this validation set.
   - This is the only truly safe global reference found in the data.
2. `practical_high_precision_threshold = 0.99`
   - Use this if you want a strong penalty band that is still extremely selective.
   - Risk remains: `2` different-desc GT pairs in val still exceed this threshold.

Do not expect CIoU to solve the safety problem by itself:

- In the high-overlap regime, IoU and CIoU tails are almost identical here.
- Example:
  - at `0.90`: IoU collisions `145`, CIoU collisions `142`
  - at `0.95`: IoU collisions `55`, CIoU collisions `53`
  - at `0.99`: both `2`

Practical interpretation:

- If you want a hard semantics-agnostic duplicate punishment, prefer IoU and set the reference near `0.99` to `0.999`.
- If you need stronger recall on duplicate suppression than `0.99` provides, then a pure overlap threshold is not enough; you will need either:
  - an allowlist / exception path for legitimate GT-overlap patterns, or
  - a soft penalty that ramps up after `0.95` and becomes hard only near `0.99+`.

## Suggested Next Step

If this will be used to modify Channel-B training, the cleanest next experiment is:

- keep the current exact-desc duplicate path for the hard gate
- add a semantics-agnostic soft penalty band starting around `0.95`
- reserve hard semantics-agnostic punishment for `IoU or CIoU >= 0.99`

That preserves safety against GT collisions while still pushing down the obvious narrow-region hallucinated stacks seen in recent UL-heavy rollouts.
