# Stage-2 (Channel-A Only) Visual Audit: Coord Token Drift, Duplicate Boxes, and A1 Coord-Loss Decision (2026-02-25)

This note analyzes the Stage-2 **Channel-A only** experiment primarily through **monitor dump visualizations** (GT vs Pred side-by-side), with the goal of:

- diagnosing *coordinate token loss / coordinate-decoding* failure modes that show up in rollouts,
- interpreting the observed “near-duplicate” predictions in crowded scenes (books / people), especially under **incomplete annotations**, and
- giving an evidence-backed recommendation on whether to introduce **any coord loss in A1** (the GT teacher-forced anchor forward), and if so, *which form* (hard `coord_token_ce` vs stronger `soft_ce_weight`).

## 0) Artifact + configs (repro handles)

- Run artifact (logs/ckpts/monitor dumps):
  - `output/stage2_ab/coco_bbox_max60/a_only/epoch_2-eff_size_64-n_softctx_iter_2-/v0-20260224-134747`
- Training config:
  - `configs/stage2_two_channel/prod/desc_first_a_only.yaml`
- Key knobs (resolved config excerpt; see `resolved_config.json` in the run dir):
  - `stage2_ab.schedule.b_ratio = 0.0` (A-only)
  - `stage2_ab.n_softctx_iter = 2` (iterative soft/ST self-context)
  - `stage2_ab.coord_ctx_embed_mode = "st"`
  - `stage2_ab.coord_decode_mode = "exp"` (CoordExp expectation decode for geometry loss)
  - `custom.coord_soft_ce_w1.enabled = true`
  - `custom.coord_soft_ce_w1.ce_weight = 0.0` (**no hard coord token CE anywhere**)
  - `custom.coord_soft_ce_w1.soft_ce_weight = 0.1`, `w1_weight = 0.1`

## 1) What the monitor-dump images are telling us (high signal)

Across crowded scenes, the dominant failure pattern is **not** “slightly-wrong boxes”.
It is **mode collapse / duplication and granularity mismatch** under rollout:

- **Duplicate predictions (same class, almost same box)**: repeated “book” boxes with very high overlap.
- **Group-box predictions**: very wide “person” boxes spanning a row of a crowd (a “group” concept), which cannot match any single GT person box.
- **Annotation incompleteness amplifies the apparent FP rate**: many “FP” boxes are plausibly real objects that simply are not labeled (classic COCO crowd/book-shelf situations).

This matters because:
- `eval/rollout/f1` (set-matching style) punishes extra predictions in the GT label space, so **unlabeled-but-real** objects look like FP.
- COCO `mAP` can move differently depending on whether the model is improving localization on the labeled subset vs increasing duplicate “extra” boxes.

## 2) Case studies (crowded)

### 2.1 Bookshelf (crowded small objects + duplicates)

Visualization:
- `output/stage2_ab/coco_bbox_max60/a_only/epoch_2-eff_size_64-n_softctx_iter_2-/v0-20260224-134747/monitor_dumps/vis_step_000900_v2/step_000900_s08_base000002.png`

What it shows:
- GT contains multiple **book** instances on the shelf.
- Pred emits **many more** book boxes, but a large fraction are:
  - near-duplicates (almost identical boxes repeated), and/or
  - row-level / segment-level boxes (granularity mismatch vs per-book GT).

Quantifying “near-duplicate” (from `monitor_dumps/step_000900.json`, computed on pixel boxes):
- Pred has 43 boxes (38 are `book`).
- Among predicted boxes:
  - **35 pairs** have IoU > **0.90**
  - **8 pairs** have IoU > **0.95**
  - **21** predicted boxes have at least one neighbor with IoU > **0.90**
- Within the `book` class alone:
  - **35 pairs** have IoU > **0.90**

Interpretation:
- This looks like *token-level repetition / collapse* rather than “just hard localization”.
- In practice it creates both:
  - **FP** (duplicate boxes), and
  - **FN** (because duplicates don’t match distinct GT book instances).

### 2.2 Sports court / stadium (crowded people + group boxes)

Visualization:
- `output/stage2_ab/coco_bbox_max60/a_only/epoch_2-eff_size_64-n_softctx_iter_2-/v0-20260224-134747/monitor_dumps/vis_step_000900_v2/step_000900_s06_base000354.png`

What it shows:
- GT labels some visible people.
- Pred emits many plausible “person” detections in the stands and audience areas.
- However, a noticeable subset are **very wide, thin horizontal boxes** spanning many individuals (a “crowd strip”), which are:
  - not matchable to any single person GT box,
  - very likely to be counted as FP, and
  - consistent with a “group-box” decoding mode (granularity mismatch).

Quantifying “near-duplicate” here (same computation as above):
- True duplicates are **not** the primary issue in this sample:
  - only a few pairs exceed IoU > 0.5
- The dominant issue is **granularity** (group boxes) + **annotation incompleteness**.

### 2.3 “Long-flat” group boxes (anti-atomic style)

Observation:
- Some predictions are **very long and very thin** boxes (large width, tiny height), often covering *multiple instances* of the same semantic class.
- These are qualitatively different from “unlabeled FP”: they are not plausible atomic instances, and they directly create both:
  - **FP** (the group box itself), and
  - **FN** (because multiple GT instances remain unmatched).

Concrete example:
- In `step_000900_s06_base000354.png`, two predicted `person` boxes span large horizontal strips of the audience.
- Geometry sanity checks on the parsed coord tokens (pixel-space after `norm1000→px`):
  - aspect ratios are ~**13:1** (extremely flat),
  - each has best IoU with any GT `person` box of only ~**0.04–0.05** (not “slightly off”; it is the wrong style).

Interpretation:
- This is best understood as an **instance granularity / atomicity failure**:
  - the model sometimes chooses a “cover the crowd” box rather than emitting per-person boxes.
- This failure mode is only weakly connected to *coord distribution quality*.
  - Even perfect coord calibration does not guarantee the model will *choose* an atomic decomposition under rollout.

Why this matters for evaluation:
- Under `f1ish_pred_scope=annotated` (the current default), `person` is inside the GT label space in these images, so these long-flat boxes are counted as FP rather than being “ignored as open-vocab”.
- COCO `mAP` is less directly sensitive to these “extra” group boxes if there are still some good matches on labeled instances.

### 2.4 Simple “missed clear object” (FN without label ambiguity)

Visualization:
- `output/stage2_ab/coco_bbox_max60/a_only/epoch_2-eff_size_64-n_softctx_iter_2-/v0-20260224-134747/monitor_dumps/vis_step_000900_v2/step_000900_s05_base000211.png`

What it shows:
- Pred localizes the large / salient objects correctly (no obvious “unlabeled FP” argument here).
- There is a **single clear FN** (`handbag`) that should be learnable (visually distinct, not “crowd ambiguity”).

Interpretation:
- This is closer to a **recall / enumeration** issue than a coordinate-calibration issue.
- Improving this likely requires “generate the extra object” pressure (rollout alignment / stop control) more than sharper coord bins.

### 2.5 Crowded street scene (mixed FP/FN; unlabeled FP plausible)

Visualization:
- `output/stage2_ab/coco_bbox_max60/a_only/epoch_2-eff_size_64-n_softctx_iter_2-/v0-20260224-134747/monitor_dumps/vis_step_000900_v2/step_000900_s07_base000047.png`

What it shows:
- Many instances of the same class (`person`) in a cluttered scene.
- A subset of “FP” boxes are plausibly real people/objects but may be **unlabeled** (classic COCO incompleteness).
- There are still meaningful **FNs** (missed persons / small accessories) that can be learned.

Interpretation:
- A purely label-space F1 metric can over-penalize the model for “detecting too much” here, while mAP may still improve if the model ranks and localizes the labeled subset better.

### 2.6 Far-distance / “rule-metric” ambiguity (label vs unlabel both plausible)

Visualization:
- `output/stage2_ab/coco_bbox_max60/a_only/epoch_2-eff_size_64-n_softctx_iter_2-/v0-20260224-134747/monitor_dumps/vis_step_000900_v2/step_000900_s03_base000017.png`

What it shows:
- A large salient object is localized well (bus).
- Small “person” boxes inside/around the bus are ambiguous: both “label” and “unlabel” are defensible depending on annotation and evaluation policy.
- Even in this ambiguity regime, the **FNs are still meaningful** to learn if we want “detect every instance” behavior.

## 3) What this implies about coord supervision / “coord token loss”

Even if the *distributional* coord losses (SoftCE/W1) trend down, these visuals indicate two rollout-facing problems:

1) **Discrete emission stability** (token-level): under greedy rollout, the model repeatedly emits nearly the same coord token sequences for many objects of the same class.

2) **Instance granularity mismatch**: the model sometimes prefers a “cover the crowd/books row” box rather than per-instance boxes.

Only (1) is plausibly improved by adding stronger coord supervision in A1.
(2) is more about prompt semantics, dataset labeling, and on-policy (rollout) learning signals.

## 4) A1 coord loss: recommendation (evidence-based, tied to visuals)

### 4.1 What we want A1 coord loss to fix

From the bookshelf case, the highest-leverage target is:
- reduce **near-duplicate coord emissions** for repeated instances (e.g., books),
- improve per-instance **disambiguation** so that distinct GT objects don’t collapse onto the same coords.

### 4.2 Recommendation

**Recommendation (primary): add a small coord distribution anchor on A1, but keep hard `coord_token_ce` off initially.**

Rationale:
- The failure looks like *rollout repetition/collapse*, and we do not want to overfit to hard coord bins too early.
- A1 already anchors struct+desc CE. Adding an A1 *distributional* coord term (SoftCE/W1) makes the **first-iterate coord beliefs** less noisy, which should also improve the quality of ST context embeddings fed into the final iteration in `n_softctx_iter=2`.

Concretely:
- Keep `custom.coord_soft_ce_w1.ce_weight = 0.0` initially.
- Increase **A1-applied** SoftCE slightly (target: more stable coord beliefs early), e.g.:
  - `soft_ce_weight: 0.1 → 0.2` (trial)
  - keep `w1_weight` at `0.1` or `0.2` (trial)

**Recommendation (secondary / if duplicates persist): enable a *tiny* hard coord CE in A1 only.**

If the “nearly identical book boxes” pattern persists after stronger SoftCE/W1:
- add a very small hard CE (token-level) term in A1 only, e.g. `coord_token_ce` weight `~0.01`,
  explicitly treating it as a “tie-breaker” to stabilize argmax emissions, not the main coord objective.

### 4.3 What not to expect from A1 coord loss

A1 coord loss will **not** solve:
- unlabeled-object “FPs” in crowded scenes (books/stands),
- group-box semantics (“one box for many people”),
- over-generation of object entries under rollout (that’s an on-policy / decoding / Channel-B surface).

Those need either:
- Channel-B rollout-aligned supervision (FP-neutral), and/or
- inference-time dedup / constraints (IoU-based drop / NMS-like filter), and/or
- prompt / dataset adjustments.

### 4.4 Discouraging “group boxes” and encouraging atomic boxes (actionable options)

From the visuals, **two** rollout-facing issues are worth addressing directly:

1) **Near-duplicate boxes** (bookshelf) → “same instance repeated”
2) **Long-flat group boxes** (sports crowd) → “multiple instances merged”

Practical mitigation options (ordered by intrusiveness):

- **(Option A) Prompt-level atomicity constraint (lowest risk; config-only).**
  - Add explicit language to `custom.user_prompt`:
    - “Each bbox_2d must correspond to exactly one object instance.”
    - “Do not output a bbox that covers multiple people/books.”
    - “If you cannot localize a single instance, omit it rather than drawing a crowd-strip box.”
  - This is the fastest lever to test and often works surprisingly well for LLM-style detectors.

- **(Option B) Inference-time filters / dedup (easy, but changes behavior/metrics).**
  - **Class-conditional “flat box” filter** (recommended to start with `person` only):
    - Drop predicted `person` boxes with extreme aspect ratio (e.g., `w/h > 8`) *and* large size (e.g., area fraction > 1%).
    - This specifically targets the crowd-strip artifact without harming typical person shapes.
  - **Within-class IoU dedup** (NMS-like):
    - For each class, suppress boxes whose IoU > 0.9 with a higher-priority box.
    - This directly targets the bookshelf duplication.

- **(Option C) Training-time shaping (higher effort; may require Stage-2B or new regularizers).**
  - Channel-A teacher forcing does not directly “see” extra rollout boxes, so it cannot strongly penalize group boxes/duplicates that occur only in rollout.
  - If Option A/B are insufficient, the most principled fix is to reintroduce a small amount of on-policy signal (Channel-B) or add a new unlabeled-friendly prior (shape prior / repulsion), which should go through an OpenSpec change.

## 5) How to verify improvements (visual-first)

Use the same monitor-dump visualization and look for **qualitative changes**:

1) Bookshelf sample:
   - fewer near-identical `book` boxes (duplicates drop sharply),
   - more per-book tight boxes that match GT,
   - fewer FN(book) and fewer FP(book).

2) Stadium sample:
   - fewer “crowd strip” boxes labeled as `person`,
   - more per-person boxes in the regions where GT labels exist.
   - fewer extreme-aspect `person` boxes (e.g., `w/h > 8` and area > 1% image)

Suggested quantitative add-on (optional, easy sanity check):
- compute `#pairs(pred_iou>0.9)` within class `book` / `person` for the monitor-dump samples
  to objectively track “duplicate collapse”.

## 6) Repro commands (visualization)

Render a dump (single JSON):

```bash
PYTHONPATH=. conda run -n ms python -P vis_tools/vis_monitor_dump_gt_vs_pred.py \
  --monitor_json output/stage2_ab/coco_bbox_max60/a_only/epoch_2-eff_size_64-n_softctx_iter_2-/v0-20260224-134747/monitor_dumps/step_000900.json \
  --save_dir output/stage2_ab/coco_bbox_max60/a_only/epoch_2-eff_size_64-n_softctx_iter_2-/v0-20260224-134747/monitor_dumps/vis_step_000900_v2 \
  --limit 0
```

Render a directory (all `step_*.json`):

```bash
PYTHONPATH=. conda run -n ms python -P vis_tools/vis_monitor_dump_gt_vs_pred.py \
  --monitor_json output/stage2_ab/coco_bbox_max60/a_only/epoch_2-eff_size_64-n_softctx_iter_2-/v0-20260224-134747/monitor_dumps \
  --save_dir output/stage2_ab/coco_bbox_max60/a_only/epoch_2-eff_size_64-n_softctx_iter_2-/v0-20260224-134747/monitor_dumps/vis_all \
  --limit 0
```
