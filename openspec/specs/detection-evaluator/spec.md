# detection-evaluator Specification

## Purpose
Define the detection evaluation contract for CoordExp, including the JSONL artifact schema it consumes, description matching policy, and reported metrics.
## Requirements
### Requirement: Ingestion and validation
For the unified pipeline workflow, the evaluator SHALL treat the pipeline artifact `gt_vs_pred.jsonl` (containing embedded `gt` and `pred` per sample) as the primary evaluation input.

The public CLI/pipeline integration SHALL NOT expose a separate-GT mode (`gt_jsonl` separate from predictions); GT MUST be embedded inline per record.

Coordinate handling:
- If a record contains `coord_mode: "pixel"`, the evaluator SHALL interpret `gt` and `pred` `points` as pixel-space coordinates and SHALL NOT denormalize again.
- If a record contains `coord_mode: "norm1000"`, the evaluator SHALL denormalize using per-record `width` and `height` and then clamp/round.
- Records missing `width` or `height` (or with null width/height) SHALL be skipped (counted) because denormalization and validation are undefined.

#### Scenario: Pixel-ready artifact is evaluated without rescaling
- **GIVEN** a `gt_vs_pred.jsonl` record with `coord_mode: "pixel"` and valid `width`/`height`
- **WHEN** the evaluator ingests the record
- **THEN** it evaluates `gt` and `pred` using the provided pixel coordinates without denormalization.

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
- Coordinates SHALL be converted according to `coord_mode`: when `norm1000`, denormalize using per-image width/height, clamp to bounds, and round; when `pixel`, only clamp/round applies to avoid double scaling. Missing width/height SHALL cause the object to be dropped and counted. If invalid/degenerate, the evaluator SHALL drop the object, increment a counter, and retain the raw geometry in per-image diagnostics.
- Polygons retain segmentation and also expose a bbox; bbox GT SHALL be given a minimal quadrilateral segmentation when segm evaluation is enabled so bbox–poly predictions can be paired; degenerate geometries are dropped and counted.

#### Scenario: Coord-token prediction is denormalized correctly
- GIVEN a prediction with `bbox_2d: ['<|coord_10|>', '<|coord_20|>', '<|coord_200|>', '<|coord_220|>']`, `coord_mode="norm1000"`, and width=1000, height=800
- WHEN parsed by the evaluator
- THEN it produces a pixel bbox [10, 16, 200, 176] (rounded) and uses that bbox in COCO artifacts.

#### Scenario: Polygon prediction pairs with bbox GT
- GIVEN a GT bbox and a prediction polygon overlapping the same region with `coord_mode="norm1000"`
- WHEN the evaluator derives a bbox/segmentation via the shared helper
- THEN the polygon prediction is eligible for IoU matching against the bbox GT instead of being discarded for geometry mismatch.

#### Scenario: Line geometry is rejected
- GIVEN a prediction object that contains `line`
- WHEN the evaluator ingests the predictions JSONL
- THEN that object is dropped, increments an `invalid_geometry` counter, and does not contribute to matches/metrics.

### Requirement: Semantic description matching
The evaluator SHALL always run description matching via `sentence-transformers/all-MiniLM-L6-v2` when deriving COCO annotations. Predictions whose normalized descriptions are not mapped with cosine similarity ≥ `semantic_threshold` SHALL be dropped (counted in `unknown_dropped`) rather than assigned to synthetic categories.

Legacy configuration keys `unknown_policy` and `semantic_fallback` are unsupported and MUST fail fast if present (no backward/legacy support).

If the semantic encoder cannot be loaded (missing from the HuggingFace cache and download is not possible), the evaluator SHALL fail loudly with a clear error message (it SHALL NOT silently degrade into bucketed or dropped defaults).

#### Scenario: Description matching fails when the encoder is unavailable
- **GIVEN** any evaluation run and `sentence-transformers/all-MiniLM-L6-v2` cannot be loaded from caches or downloads
- **WHEN** the evaluator starts mapping descriptions
- **THEN** it raises a runtime error describing that the encoder is mandatory for evaluation and advising the user to ensure the model is cached/downloadable.

#### Scenario: Deprecated keys fail fast
- **WHEN** evaluation config includes `unknown_policy` or `semantic_fallback`
- **THEN** evaluation fails fast with an actionable error describing that these keys are unsupported and must be removed.

### Requirement: COCO artifacts and scoring modes
When COCO artifacts/metrics are requested, the evaluator SHALL export COCO artifacts and compute metrics in a score-aware manner as specified below.

This requirement applies when COCO artifacts/metrics are requested. Runs that compute only non-COCO metrics (e.g., f1ish-only) MAY accept unscored artifacts.

Milestone scope note:
- `confidence-postop` currently computes confidence for `bbox_2d` only. As a result, `gt_vs_pred_scored.jsonl` will contain only bbox predictions (polygon predictions are dropped as unscorable).
- Therefore, COCO evaluation over `gt_vs_pred_scored.jsonl` SHOULD be bbox-only (do not run segm metrics) until polygon confidence is supported.

- The evaluator SHALL emit `coco_gt.json` and `coco_preds.json` with images, annotations, categories, and predictions using xywh bbox format; segmentation is included when polygons are present.
- Scores SHALL be taken from each prediction object’s `score` field and exported as the COCO `score` for ranking (AP/mAP).
- Each prediction object MUST include a finite numeric `score` satisfying `0.0 <= score <= 1.0`. Missing, non-numeric, `NaN`, infinite, or out-of-range scores MUST fail fast with actionable diagnostics (record index + object index).
- Input records MUST include `pred_score_source` (string, non-empty) and `pred_score_version` (int). This change intentionally rejects unscored artifacts for COCO evaluation.
- When scores tie, export order SHALL remain stable using input order.
- Image ids in COCO export SHALL be derived deterministically from the JSONL index (0-based) to avoid collisions across shards.
- If `coco_preds.json` is empty, evaluator metrics MUST still include the standard aggregate keys for active IoU families and set them explicitly to `0.0` (rather than omitting metrics/nulls).

#### Scenario: COCO export honors prediction scores
- **GIVEN** a prediction record with two bbox predictions with `score=0.9` and `score=0.1`
- **WHEN** the evaluator exports COCO predictions
- **THEN** `coco_preds.json` contains those same score values for the corresponding exported predictions.

#### Scenario: Missing score fails fast
- **GIVEN** a prediction object missing the `score` field (or with a non-numeric score)
- **WHEN** the evaluator attempts to export COCO predictions
- **THEN** evaluation terminates with a clear error identifying the offending record and object index.

#### Scenario: Unscored artifact is rejected for COCO export
- **GIVEN** an input JSONL record missing `pred_score_source` / `pred_score_version`
- **WHEN** the evaluator attempts to export COCO predictions
- **THEN** evaluation terminates with a clear error explaining that scored artifacts are mandatory for COCO evaluation and the input is not score-provenanced.

#### Scenario: Empty scored predictions produce explicit zero COCO metrics
- **GIVEN** a COCO evaluation run where no predictions survive scoring/export
- **WHEN** the evaluator writes metrics
- **THEN** standard aggregate metrics (e.g., `bbox_AP`, `bbox_AP50`, `bbox_AR100`) are present and equal `0.0`.

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

#### Scenario: Metrics summary persisted
- GIVEN a valid GT/pred pair
- WHEN the evaluator runs
- THEN it writes `metrics.json` containing the COCO summary metrics and the robustness counters.

### Requirement: CLI, configuration, and outputs
The evaluator SHALL support a YAML config template under `configs/eval/` and SHOULD accept `--config` to run evaluation reproducibly.

If both CLI flags and YAML are provided, CLI flags SHALL override YAML values, and the evaluator SHALL log the resolved configuration.

#### Scenario: Evaluate via YAML config
- **GIVEN** `configs/eval/detection.yaml` and a prediction JSONL artifact
- **WHEN** the user runs evaluator with `--config configs/eval/detection.yaml`
- **THEN** it produces `metrics.json`, `per_class.csv`, `per_image.json` under the configured output directory.

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

### Requirement: Pipeline integration
The evaluator SHALL be callable as a stage from the unified inference pipeline runner, using the same resolved run directory and artifact conventions.

#### Scenario: Pipeline stage calls evaluator
- **GIVEN** a unified pipeline run directory containing `gt_vs_pred.jsonl`
- **WHEN** the pipeline runner executes the eval stage
- **THEN** evaluation outputs are written under the run directory (or a deterministic subdirectory) without requiring additional user inputs.

### Requirement: Category mapping and unknown handling
The evaluator SHALL support an `unknown_policy` configuration with modes:
- `bucket`: map unknown desc to a synthetic `unknown` category id (COCO export only),
- `drop`: drop unknown desc predictions (COCO export only),
- `semantic`: keep predicted desc unchanged, but for COCO export MAY map unknown predicted desc to the nearest GT-desc category by embedding similarity.

If `unknown_policy=semantic` is requested and the embedding model cannot be loaded, the evaluator SHALL fail loudly with a clear error message (it SHALL NOT silently degrade into bucketing).

#### Scenario: Synonym without alias falls to unknown
- GIVEN GT contains "traffic light" and a prediction uses desc "stoplight"
- WHEN exported without alias/fuzzy mapping
- THEN the prediction is assigned to category `unknown` unless `unknown` dropping is enabled, in which case it is dropped.

#### Scenario: Semantic unknown handling fails loudly when model is missing
- **GIVEN** `unknown_policy=semantic` and the configured semantic model is not available in local cache and cannot be downloaded
- **WHEN** the evaluator is executed
- **THEN** it raises a runtime error describing how to disable semantic mapping or provide the model.

### Requirement: Evaluator ingestion diagnostics are path-and-line explicit
Detection-evaluator SHALL provide path-and-line explicit diagnostics for malformed JSONL ingestion failures.
Diagnostics MUST identify source file and 1-based line number for parse failures.
Diagnostics SHOULD include a clipped payload snippet for rapid operator triage.

#### Scenario: Malformed JSONL line reports precise source context
- **GIVEN** an input artifact containing malformed JSON on one line
- **WHEN** evaluator ingestion parses the file
- **THEN** diagnostics identify the source path and 1-based line number for the malformed record
- **AND** diagnostics include a clipped snippet of the malformed payload.

### Requirement: Evaluator reuses shared coordinate and geometry helpers
Detection-evaluator SHALL reuse shared coordinate/geometry helper contracts for conversion and validation, rather than maintaining parallel helper implementations.
For overlapping active deltas, helper-level strictness/diagnostic defaults are authoritative in `2026-02-11-src-ambiguity-cleanup`; this change MUST remain consistent with that contract.
This requirement SHALL preserve existing evaluation metric intent and artifact compatibility.

#### Scenario: Shared helper reuse preserves evaluation eligibility behavior
- **GIVEN** bbox/poly mixed geometry records
- **WHEN** evaluator processes coordinates through shared helpers
- **THEN** match eligibility decisions remain consistent with canonical helper behavior.

### Requirement: Evaluation artifact and metric schema parity is preserved during refactor
Detection-evaluator SHALL preserve existing evaluation artifact schema (`metrics.json`, per-image outputs, match artifacts where enabled) and existing metric naming conventions during internal refactor.

#### Scenario: Refactored evaluator produces schema-compatible outputs
- **GIVEN** the same evaluator inputs and settings
- **WHEN** evaluation runs before and after refactor
- **THEN** output artifact schema and stable metric key names remain compatible for downstream consumers.

### Requirement: Evaluator strict-parse behavior is config-driven and bounded
Evaluator ingestion strictness SHALL be controlled by `eval.strict_parse` (default `false`).
- `eval.strict_parse=true`: fail fast on first malformed/non-object JSONL record.
- `eval.strict_parse=false`: warn+skip malformed records deterministically with bounded diagnostics (`warn_limit=5`, `max_snippet_len=200`).

For overlapping helper-consolidation deltas, this change MUST stay consistent with `2026-02-11-src-ambiguity-cleanup` as the authoritative strict-parse helper contract.

#### Scenario: Strict and non-strict parse modes remain deterministic
- **GIVEN** one malformed JSONL record in evaluator input
- **WHEN** `eval.strict_parse=true`
- **THEN** evaluation fails immediately with explicit path+line diagnostics
- **AND WHEN** `eval.strict_parse=false`
- **THEN** evaluator emits bounded warnings and skips the malformed record deterministically.

### Requirement: Semantic encoder implementation is shared across training and evaluator
Semantic description normalization and sentence-embedding computation used for evaluator description mapping and Stage-2 semantic gating/monitoring SHALL use the same canonical implementation (normalization rules, mean pooling, and L2 normalization).
The evaluator MUST NOT carry a separate parallel encoder implementation that could drift.

#### Scenario: Normalization rules are consistent across surfaces
- **GIVEN** two descriptions that differ only by punctuation/whitespace (e.g., `Armchair/Chair (Wood)` and `armchair chair wood`)
- **WHEN** training gating and evaluation normalize descriptions
- **THEN** they produce the same normalized description string.

### Requirement: Evaluation JSONL ingestion diagnostics are centralized
When the evaluator loads `gt_vs_pred.jsonl`, JSON parsing diagnostics (path + 1-based line number + clipped snippet) SHALL be implemented via a shared helper so parsing/warning behavior is consistent.
Strict mode MUST fail fast on malformed records; non-strict mode MUST warn a bounded number of times and skip malformed records deterministically.
The governing config key is `eval.strict_parse` (default `false`).
Bounded diagnostics defaults are normative: `warn_limit=5`, `max_snippet_len=200`.

#### Scenario: Malformed JSONL line is reported with path and line number
- **GIVEN** a `gt_vs_pred.jsonl` containing a malformed JSON line at line 3
- **WHEN** the evaluator loads records in non-strict mode
- **THEN** it emits a warning containing the file path and `:3`
- **AND** it skips the malformed record.

#### Scenario: Strict parse mode fails on first malformed JSONL record
- **GIVEN** a `gt_vs_pred.jsonl` containing malformed JSON
- **WHEN** the evaluator loads records with `eval.strict_parse=true`
- **THEN** it fails immediately with explicit path+line diagnostics
- **AND** it does not continue with partial record ingestion.

### Requirement: Image-path resolution helper is shared
Evaluator surfaces that resolve image paths (e.g., overlay rendering) SHALL delegate to shared image-path resolution helpers rather than implementing ad-hoc base-dir logic.

#### Scenario: Relative image path resolves deterministically
- **GIVEN** an image field `images/foo.jpg` and an explicit base directory
- **WHEN** the evaluator resolves the image path via the shared helper
- **THEN** it deterministically resolves to `<base_dir>/images/foo.jpg` (absolute path).

