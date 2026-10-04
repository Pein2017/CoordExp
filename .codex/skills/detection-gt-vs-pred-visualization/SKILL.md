---
name: detection-gt-vs-pred-visualization
description: Inspect CoordExp detection predictions from scored artifacts or saved rollout JSONL with explicit GT labels. Use for missed GT, repeated boxes, invalid geometry, or before/after comparisons, with focused crops and actual image viewing.
---

# Detection visual inspection

Use the maintained CLI `scripts/visualize_detection.py` and APIs
`src.vis.render_gt_vs_prediction` / `render_prediction_comparison` from the
current checkout. The implementation owners are `src/vis/`. Read its current
CLI help when an option or input schema is uncertain. Existing saved outputs
suffice: visualization does not require inference or a model load.

## Bind the input

| Input | Required contract |
| --- | --- |
| Scored artifacts (default) | Matching `gt_vs_pred.jsonl` and `gt_vs_pred_scored.jsonl`; supply their directory or the scored file. |
| Rule-stability native rollout JSONL | Explicit `--input-format rollout --labels-json <full-labels.json>`; original records with producer `kind=rule_stability`, `engine=native`, image/token identities, text, and aligned log probabilities. |

Do not guess coordinate units or silently recognize arbitrary JSONL formats.
The supported native rollouts join full labels by image ID with image identity/dimensions
checked. Cropped or scaled acquisition needs its own explicit adapter.
GT and native coordinate tokens use norm1000; conversion is
`round(bin * image_extent / 1000)`. Scored prediction `bbox` is already pixels;
its `coord_bins` is metadata, not another pixel conversion.

For rule-stability per-image saved records, assemble JSONL without changing
the original records or pretending they are scored inference artifacts:

```bash
python -m probes.rule_stability.visualization \
  --run-dir <native-primary-B-01> --version 16 \
  --image-id 351017 --image-id 7511 --output <fresh-rollouts.jsonl>
```

This adapter binds raw records and saved analysis. Reuse their original action
positions; generic text parsing cannot invent token positions. Preserve raw
generated order, GT owner IDs, invalid rows, and malformed/censored spans.
In native rollout mode, `P46` means raw prediction order 46, not the 47th valid
prediction. Scored mode retains its existing prediction-list indices; inspect
any original row identity carried in object metadata separately.

## Inspect, then focus

1. Select a bounded set tied to the question: missed GT, a repeated pair, or an
   invalid row. Bind source version/checkpoint, image ID, GT owner/raw prediction
   IDs, and any saved metric being cited.
2. Render and **open the full-image PNG with `view_image`** (or the environment's
   image viewer). Inspect the source image too when overlays obscure it.
3. For each difficult region, select a crop in **original image pixels** with
   enough surrounding context. Render the crop from source pixels, then
   **open the cropped PNG**. Enlarging an already downsampled overview loses
   detail. Artifact existence or a dense overview alone is not visual review.
4. Use `--focus-gt N` / `--focus-pred N` (repeatable), or `--focus-region X1 Y1
   X2 Y2`. A region selects objects whose centers are inside it; explicit IDs
   and the region are combined. Selected boxes are solid and labeled; other
   boxes are thin, dashed, and faded by `--context-alpha` (default 0.18).
   Select exact IDs when overlapping boxes would all fall inside a region.
5. Check the manifest's identities, original coordinates, view options, and
   full object pool. Cropping/focus changes presentation, not matching or counts.
   Keep the uncropped view alongside local views; crop absence is not absence
   from the full prediction sequence.

```bash
python scripts/visualize_detection.py gt-vs-pred \
  --input-format rollout --run-dir <rollouts.jsonl> \
  --labels-json <full-labels.json> --row-id 7511 \
  --crop 720 465 850 550 --focus-gt 21 --focus-pred 46 --focus-pred 48 \
  --context-alpha 0.12 --out-dir <fresh-crop-dir>

python scripts/visualize_detection.py compare \
  --input-format rollout --labels-json <full-labels.json> \
  --left-run-dir <B0.jsonl> --right-run-dir <B16.jsonl> \
  --left-label B0 --right-label B16 --row-id 351017 \
  --crop 200 40 570 470 --focus-gt 34 --left-focus-pred 1 \
  --out-dir <fresh-comparison-dir>
```

For an overview omit crop/focus flags. Compare requires the same selected
ordered image IDs and GT. Explicitly focused GT remains visible even when
matched; default comparison hides matched GT. A one-sided prediction focus
fades context on both sides. Each sample gets its own PNG; same-image side-by-side
panels are supported. Choose fresh output directories to preserve earlier views.

## Interpret and report

- Invalid native geometry remains visible as directed endpoints/segments with
  its original coordinates; never sort corners into a valid rectangle. Invalid
  rows are excluded from box matching. Malformed/censored tails stay in metadata
  without invented boxes. Scored-artifact geometry remains strict.
- Renderer colors use class-aware greedy pixel IoU matching at 0.5. Purple
  duplicate hints use same-category pairwise IoU (default 0.30). These differ
  from research assignment and chronological duplicate-event definitions;
  retain saved research results separately. Faint dashed boxes mean context,
  not a second duplicate decision.
- Separate visible entity, category, localization, and output-sequence claims.
  Unmatched prediction does not establish hallucination; annotation miss does
  not by itself establish a physical miss. Visual symptoms do not establish
  the model's internal cause.

Report what was actually opened and observed, with source versions, row/owner/
raw prediction IDs, crop coordinates, PNG and manifest links, and remaining
uncertainty. Give a few readable examples instead of reproducing the full pool;
keep that pool accessible through the manifest and original records.
