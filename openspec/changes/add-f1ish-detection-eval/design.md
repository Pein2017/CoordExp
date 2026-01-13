# Design: F1-ish detection evaluator (CoordExp)

This change adds a complementary evaluation mode intended for research iteration and EM-ish training loops.

## Goals
- Provide a **parsable, deterministic** metric that matches the modeling assumptions:
  - prediction is a **set** of objects, order not meaningful
  - greedy decoding has **no reliable score calibration**
  - open-vocab desc strings require **semantic tolerance**
- Produce artifacts that can be reused by Stage-2 EM-ish training:
  - per-image matched pairs (pred ↔ gt) with token/geometry info

## Non-goals
- Replacing COCOeval entirely (we keep it for comparability).
- Designing a DETR-style head or confidence estimator.

## Data flow
Input:
- `pred.jsonl` (pixel-space objects), with inline `gt` and `pred` arrays.

Pipeline:
1) Load JSONL → validate + drop invalid geometries (existing behavior).
2) For each image:
   - Build per-object normalized representation:
     - Always have `bbox` (pixel-space).
     - If polygon exists: keep `segmentation` polygon.
     - If bbox-only: generate a rectangle polygon segmentation.
     - **Line geometries are excluded** from F1-ish matching/metrics (consistent with the current evaluator docs/behavior).
3) Select the evaluated prediction set (open-vocab / partial-GT friendly):
   - Default "annotated-object recovery" behavior ignores predictions whose `desc` is not semantically close to any GT `desc` in the image (so extra open-vocab objects don't become FP).
   - A strict mode can count all predictions as FP ("all predictions" scope).
4) Compute location-first matching:
   - Build candidate pairs with IoU.
   - Greedy 1:1 assignment (highest IoU first), gated by `iou_thr`, with explicit deterministic tie-breaking.
5) For matched pairs:
   - Compute semantic similarity between `pred_desc` and `gt_desc`.
   - Determine `sem_ok` using exact match or embedding cosine similarity ≥ `semantic_thr`.
6) Aggregate metrics:
   - Localization-only counts: TP/FP/FN → precision/recall/F1.
   - Semantic-on-matched: semantic accuracy, plus optional strict “full” F1 variant.

Outputs:
- `metrics.json` includes:
  - existing COCO-style metrics (optional)
  - `f1ish@{iou_thr}_*` metrics (optional), where `{iou_thr}` is formatted with two decimals (e.g. `0.30`, `0.50`).
- `per_image.json` includes per-image F1-ish counts (and optionally match summaries) under a stable `f1ish` field keyed by `{iou_thr}`.
- `matches.jsonl` includes matched pairs for a primary IoU threshold (see defaults), and additional `matches@{iou_thr}.jsonl` files MAY be emitted when multiple IoU thresholds are requested.

## Location matching details
### IoU type selection (auto)
- If **either object** has a polygon geometry, compute **segmentation IoU**.
- Otherwise, compute **bbox IoU**.

### Line geometries
Line geometries are excluded from F1-ish matching and metrics:
- They SHALL NOT generate candidate pairs.
- They SHALL NOT contribute to `tp_loc` / `fp_loc` / `fn_loc`.
- They MAY appear in diagnostics as skipped/invalid objects, but are not part of the evaluated sets.

### bbox ↔ poly support
Always support matching bbox ↔ poly:
- Convert bbox `[x1,y1,x2,y2]` into a rectangle polygon segmentation
  `[x1,y1, x2,y1, x2,y2, x1,y2]`.
- Then compute segmentation IoU (mask IoU) when polygon exists on either side.

Implementation note:
- Use `pycocotools.mask` utilities:
  - polygon → RLE via `frPyObjects`
  - rectangle polygon → RLE via `frPyObjects`
  - IoU via `maskUtils.iou`

### Greedy assignment
Greedy is the default due to simplicity/debuggability:
1) Compute all candidate pairs with IoU ≥ `iou_thr`.
2) Sort pairs by `(iou desc, pred_idx asc, gt_idx asc)` to break ties deterministically.
3) Iterate sorted pairs; accept if both pred and gt are unmatched.

Determinism requirements:
- Images MUST be processed in `image_id` ascending (JSONL index order).
- `matches.jsonl` MUST be emitted in that same `image_id` ascending order.
- Within an image, matched pairs MUST be emitted in the greedy-acceptance order, which is deterministic given the tie-break rule.

Optional future extension:
- Hungarian assignment as a flag for crowded scenes, with the same IoU gating.

## Semantic scoring
Semantic evaluation is only applied **after** location matching.
- The evaluator MUST NOT rewrite predicted desc into `unknown`.
- Semantic correctness can be:
  - exact string match (fast, strict), OR
  - embedding similarity ≥ `semantic_thr` (tolerant to synonyms).

Embedding model:
- HuggingFace encoder (default: `sentence-transformers/all-MiniLM-L6-v2`)
- Cache embeddings per unique string within an eval run.
- Device selection:
  - default `auto` chooses CUDA when available.

## Metrics
Report both **localization-only** and **localization+semantic** views.

Localization-only (micro + macro, per IoU threshold):
- `tp_loc`: matched by IoU
- `fp_loc`: unmatched evaluated predictions (hallucinations/extras within scope)
- `fn_loc`: unmatched GT (missing)
- `precision_loc_micro`, `recall_loc_micro`, `f1_loc_micro` computed from global sums over all images.
- `precision_loc_macro`, `recall_loc_macro`, `f1_loc_macro` computed as the unweighted mean over images of per-image precision/recall/F1.

Macro-over-images definition:
- For each image, compute:
  - `precision_i = tp_i / (tp_i + fp_i)` if `(tp_i + fp_i) > 0`, else `1.0`.
  - `recall_i = tp_i / (tp_i + fn_i)` if `(tp_i + fn_i) > 0`, else `1.0`.
  - `f1_i = 2 * precision_i * recall_i / (precision_i + recall_i)` if `(precision_i + recall_i) > 0`, else `0.0`.
- Then `macro_*` is the mean of the per-image values.

This convention yields:
- empty GT + empty pred → precision=1, recall=1, F1=1 (perfect).
- empty pred + non-empty GT → precision=1, recall=0, F1=0 (all missing).
- non-empty pred + empty GT → precision=0, recall=1, F1=0 (all hallucination).

Semantic-on-matched:
- `matched_sem_ok`, `matched_sem_bad`
- `sem_acc_on_matched`

Optional strict combined score (useful when you want “end task success”):
- Treat matched-but-wrong-sem as an error term in both FP and FN:
  - `tp_full = matched_sem_ok`
  - `fp_full = fp_loc + matched_sem_bad`
  - `fn_full = fn_loc + matched_sem_bad`
  - `f1_full`

## Defaults (initial recommendation)
- `f1ish_iou_thrs = [0.3, 0.5]` with `0.5` treated as the primary threshold for artifact names (`matches.jsonl` corresponds to `0.5`, and `matches@0.30.jsonl` corresponds to `0.3`).
- `semantic_thr = 0.6` (based on early mapping scores: synonyms ≳ 0.64, many spurious matches ≲ 0.50)
