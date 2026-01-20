# detection-evaluator Specification

## Purpose
TBD - created by archiving change add-detection-evaluator. Update Purpose after archive.
## Requirements
### Requirement: Ingestion and validation
- The evaluator SHALL accept ground-truth JSONL that follows `docs/DATA_JSONL_CONTRACT.md` and predictions either as parsed JSONL (e.g., from `vis_tools/vis_coordexp.py`) or raw generation text parsed with the shared utility.
- For records with multiple images, the evaluator SHALL use only the first image entry (index 0) to match current generation tooling, count the skip as `multi_image_ignored`, and continue.
- Width/height from GT SHALL be the source of truth; conflicting width/height in predictions SHALL be ignored and counted as `size_mismatch`.
- Entries missing width/height, containing multiple geometries, degenerate/zero-area shapes, or with coords outside 0–999 SHALL be dropped and counted (e.g., `invalid_geometry`, `invalid_coord`, `missing_size`).

#### Scenario: Malformed prediction is counted and skipped
- GIVEN a prediction object with both `bbox_2d` and `poly`
- WHEN the evaluator ingests the predictions JSONL
- THEN that object is dropped, increments an "invalid_geometry" counter, and does not appear in the COCO outputs.

#### Scenario: Multi-image record handled deterministically
- GIVEN a JSONL record with `images: ["a.jpg", "b.jpg"]`
- WHEN the evaluator ingests the record
- THEN it evaluates only `a.jpg`, logs one `multi_image_ignored`, and proceeds without failure.

### Requirement: Parsing and coordinate handling
- The evaluator SHALL reuse the shared coord-processing module (via `src/common/geometry`/`src/common/schemas`) used by inference/visualization, supporting coord tokens or ints in 0–999 with one geometry per object.
- Supported geometries SHALL be limited to `bbox_2d` and `poly`.
- Any object containing `line` or `line_points` SHALL be treated as invalid geometry and dropped with an `invalid_geometry`-style counter.

#### Scenario: Line geometry is rejected
- GIVEN a prediction object that contains `line`
- WHEN the evaluator ingests the predictions JSONL
- THEN that object is dropped, increments an `invalid_geometry` counter, and does not contribute to matches/metrics.

### Requirement: Category mapping and unknown handling
The evaluator SHALL support an `unknown_policy` configuration with modes:
- `bucket`: map unknown desc to a synthetic `unknown` category id (COCO export only),
- `drop`: drop unknown desc predictions (COCO export only),
- `semantic`: keep predicted desc unchanged, but for COCO export MAY map unknown predicted desc to the nearest GT-desc category by embedding similarity.

If `unknown_policy=semantic` is requested and the embedding model cannot be loaded, the evaluator SHALL fail loudly with a clear error message (it SHALL NOT silently degrade into bucketing).

#### Scenario: Semantic unknown handling fails loudly when model is missing
- **GIVEN** `unknown_policy=semantic` and the configured semantic model is not available in local cache and cannot be downloaded
- **WHEN** the evaluator is executed
- **THEN** it raises a runtime error describing how to disable semantic mapping or provide the model.

### Requirement: COCO artifacts and scoring modes
- The evaluator SHALL emit `coco_gt.json` and `coco_preds.json` with images, annotations, categories, and predictions using xywh bbox format; segmentation is included when polygons are present.
- Scores SHALL be set to a constant 1.0 for all predictions (greedy decoding without confidence); any provided `score` fields in predictions SHALL be ignored. When scores tie, export order SHALL remain stable using input order.
- Image ids in COCO export SHALL be derived deterministically from the JSONL index (0-based) to avoid collisions across shards.

#### Scenario: Stable ordering under constant scores
- GIVEN multiple predictions with identical scores
- WHEN exported
- THEN they retain their input order in `coco_preds.json`, ensuring deterministic evaluation.

#### Scenario: Pred export with polygons
- GIVEN polygon predictions and GT polygons for an image
- WHEN the evaluator exports COCO files
- THEN `coco_preds.json` contains both `bbox` and `segmentation` entries and can be evaluated with COCOeval `segm` metrics.

### Requirement: Metrics and diagnostics
In addition to COCOeval metrics, the evaluator SHALL support an F1-ish set-matching evaluation mode that reports:
- per-image and global counts of `matched` (TP), `missing` (FN), and `hallucination` (FP),
- micro- and macro-averaged precision/recall/F1 for localization (macro-over-images),
- semantic accuracy over matched pairs, and MAY report a strict combined F1 that penalizes matched-but-wrong-semantic pairs.

In open-vocabulary / partially-annotated settings, F1-ish SHALL support an "annotated-object recovery" interpretation:
- Predictions outside the per-image GT label space MAY be excluded from evaluation (not counted as `hallucination`).
- When excluded predictions are supported, the evaluator SHALL report the total/evaluated/ignored prediction counts per IoU threshold using the keys:
  - `f1ish@{iou_thr}_pred_total` (all predictions),
  - `f1ish@{iou_thr}_pred_eval` (predictions considered for FP/TP),
  - `f1ish@{iou_thr}_pred_ignored` (predictions excluded from FP/TP).

Macro definition (required for reproducibility):
- **Micro** precision/recall/F1 SHALL be computed from global sums across all images.
- **Macro** precision/recall/F1 SHALL be computed as the unweighted mean over images of per-image precision/recall/F1.
- Per-image precision/recall are defined with a deterministic empty-set convention:
  - `precision_i = tp_i / (tp_i + fp_i)` if `(tp_i + fp_i) > 0`, else `1.0`.
  - `recall_i = tp_i / (tp_i + fn_i)` if `(tp_i + fn_i) > 0`, else `1.0`.
  - `f1_i = 2 * precision_i * recall_i / (precision_i + recall_i)` if `(precision_i + recall_i) > 0`, else `0.0`.
- Metrics keys for F1-ish SHALL be namespaced by IoU threshold using the prefix `f1ish@{iou_thr}_`, where `{iou_thr}` is formatted with two decimals (e.g. `0.30`, `0.50`).
- `per_image.json` SHALL include per-threshold per-image counts under a stable `f1ish` field keyed by `{iou_thr}`.

#### Scenario: F1-ish metrics are reported
- **GIVEN** valid GT/pred objects for an image
- **WHEN** F1-ish evaluation mode runs
- **THEN** `metrics.json` includes `f1ish@{iou_thr}_*` keys and `per_image.json` includes `matched/missing/hallucination` counts for each requested `iou_thr`.

### Requirement: CLI, configuration, and outputs
- The evaluator SHALL provide a CLI (`scripts/evaluate_detection.py` or `python -m src.eval.detection`) that accepts GT/pred paths, cat map, score mode, IoU thresholds, polygon/segm toggle, unknown policy, strict-parse, and output directory.
- A reproducible YAML config template SHALL be added under `configs/eval/detection.yaml` and may be passed to the CLI.
- The evaluator SHALL emit `metrics.json`, `per_class.csv` (per-class AP), `per_image.json` (matches/unmatched with reasons), and MAY emit overlay PNG/HTML samples for top-K FP/FN when enabled (default disabled).
- The default training loop remains unchanged; an eval hook MAY be enabled explicitly to call the same evaluator offline.

#### Scenario: CLI run produces artifacts
- GIVEN GT and prediction JSONL files and an output directory
- WHEN `python -m src.eval.detection --gt_jsonl ... --pred_jsonl ... --out_dir ...` is executed
- THEN the output directory contains `metrics.json`, `per_class.csv`, and `per_image.json` (and overlays if enabled).

### Requirement: Tests and fixtures
- The change SHALL include a small fixture dataset (2–3 images) with deterministic predictions and a CI smoke test that exercises parsing, category mapping, COCOeval (AP50 threshold), and robustness counters.

#### Scenario: CI smoke test
- GIVEN the fixture GT/pred files
- WHEN the evaluator test runs
- THEN it completes without errors and asserts AP50 exceeds a small threshold (e.g., >0) while verifying invalid/empty counters are zero for the fixture.

### Requirement: Location-first greedy matching
In F1-ish evaluation, the evaluator SHALL match predicted objects to GT objects per image using a greedy 1:1 assignment:
- Compute candidate pairs with IoU ≥ `iou_thr`,
- Sort pairs by IoU descending, breaking ties deterministically by `(pred_idx asc, gt_idx asc)`,
- Accept a pair only if both pred and GT are unmatched,
- Remaining GT are `missing` and remaining *evaluated* preds are `hallucination`.

Additionally:
- Objects with `line` geometry SHALL be excluded from F1-ish matching and SHALL NOT contribute to `matched/missing/hallucination` counts.
- The evaluator SHALL process images in `image_id` ascending (JSONL index order) to keep artifacts reproducible.

#### Scenario: Greedy matching produces 1:1 pairs
- **GIVEN** two GT objects and three predicted objects with overlapping boxes
- **WHEN** greedy matching runs
- **THEN** at most two pairs are matched and the extra prediction is counted as hallucination.

### Requirement: Segmentation IoU when polygons exist
For F1-ish matching, if either the GT or prediction object has polygon geometry, the evaluator SHALL compute location similarity using segmentation IoU (mask IoU). It SHALL always support bbox ↔ poly matching by representing bboxes as rectangle segmentations.

#### Scenario: bbox prediction matches poly GT
- **GIVEN** a GT polygon and a predicted bbox covering the same region
- **WHEN** F1-ish matching runs
- **THEN** the pair is eligible for matching using segmentation IoU.

#### Scenario: poly prediction matches bbox GT
- **GIVEN** a GT bbox and a predicted polygon covering the same region
- **WHEN** F1-ish matching runs
- **THEN** the pair is eligible for matching using segmentation IoU.

### Requirement: Semantic scoring after location match
After location matching, the evaluator SHALL score semantic correctness on matched pairs using:
- exact string match, OR
- embedding similarity ≥ `semantic_thr`.

The evaluator SHALL NOT rewrite predicted desc strings for the purpose of F1-ish scoring.

#### Scenario: Near-synonym counts as semantic-correct
- **GIVEN** a matched pair where GT desc is `chair` and predicted desc is `armchair`
- **WHEN** semantic scoring runs with `semantic_thr` below the similarity between those strings
- **THEN** the match is counted as semantic-correct.

### Requirement: Match artifact export
When F1-ish evaluation runs, the evaluator SHALL emit a `matches.jsonl` artifact that records, per image:
- stable join keys (`image_id` as the JSONL index, and `file_name` when available),
- matched pred/gt indices (GT indices index into the post-validation per-image GT array; prediction indices index into the original per-image prediction array),
- location IoU,
- predicted and GT desc strings,
- semantic similarity score and pass/fail flag.

If the evaluator supports excluding predictions outside the GT label space (see "annotated-object recovery"), it SHALL additionally record:
- `pred_scope` (e.g. `annotated` or `all`),
- `pred_count` / `pred_count_eval` / `pred_count_ignored`,
- `ignored_pred_indices` listing prediction indices that were excluded from FP/TP counting.

If multiple IoU thresholds are requested for F1-ish, the evaluator SHALL:
- treat `0.50` as the primary threshold when present (otherwise use the largest requested threshold),
- write the primary threshold results to `matches.jsonl`,
- and write additional thresholds to `matches@{iou_thr}.jsonl`, where `{iou_thr}` is formatted with two decimals (e.g. `0.30`).

#### Scenario: Matches file supports EM-ish workflows
- **GIVEN** an evaluation run on `pred.jsonl`
- **WHEN** F1-ish evaluation completes
- **THEN** `matches.jsonl` exists and contains enough information to reuse the pred↔gt pairing in a training loop.

### Requirement: CLI supports selecting metric families
The evaluator CLI SHALL provide a flag to select evaluation families:
- COCO metrics only,
- F1-ish only,
- or both in one run.

#### Scenario: Both metric families can be produced
- **GIVEN** `--metrics both`
- **WHEN** the evaluator runs
- **THEN** it writes COCO artifacts and also writes F1-ish metrics/artifacts in the same output directory.

### Requirement: F1-ish prediction scope (annotated-object recovery)
In F1-ish evaluation, the evaluator SHALL support controlling which predictions are counted toward FP/TP:
- `annotated`: only predictions whose desc is semantically close to any GT desc in the image are evaluated; other predictions are ignored (not counted as FP).
- `all`: all predictions are evaluated (strict; extra predictions are counted as FP).

Semantic closeness SHALL use the same embedding model and `semantic_thr` as F1-ish semantic scoring, falling back to exact string match when embeddings are unavailable.

#### Scenario: Predictions outside GT label space are ignored
- **GIVEN** a GT set containing only the label `chair`
- **AND** predictions contain `chair` and `table` with valid boxes
- **WHEN** F1-ish runs with prediction scope `annotated`
- **THEN** the `table` prediction is ignored and does not contribute to `hallucination` (FP).

