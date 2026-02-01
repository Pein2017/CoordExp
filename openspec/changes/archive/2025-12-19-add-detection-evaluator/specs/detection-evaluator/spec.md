## ADDED Requirements

### Requirement: Ingestion and validation
- The evaluator SHALL accept ground-truth JSONL that follows `docs/data/JSONL_CONTRACT.md` and predictions either as parsed JSONL (e.g., from `vis_tools/vis_coordexp.py`) or raw generation text parsed with the shared utility.
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
- The evaluator SHALL reuse the coord-token parsing/normalization logic from `vis_tools/vis_coordexp.py` (coord tokens or ints 0–999, one geometry per object).
- Coordinates SHALL be denormalized to pixel space using per-image width/height, clamped to bounds, and rounded to the nearest integer before export; when a pixel-space flag is set, only clamp/round applies.
- Lines SHALL be converted to their tight bbox for detection; polygons retain segmentation and also expose a bbox; degenerate geometries are dropped and counted.

#### Scenario: Coord-token prediction is denormalized correctly
- GIVEN a prediction with `bbox_2d: ['<|coord_10|>', '<|coord_20|>', '<|coord_200|>', '<|coord_220|>']` and width=1000, height=800
- WHEN parsed by the evaluator
- THEN it produces a pixel bbox [10, 16, 200, 176] (rounded) and uses that bbox in COCO artifacts.

### Requirement: Category mapping and unknown handling
- The evaluator SHALL map description strings to category ids using exact string matches against a vocab derived from GT desc strings; no alias map or fuzzy matching is required (alias-free stance).
- Unknown/mismatched/empty desc SHALL be bucketed into an `unknown` category by default; a config flag MAY switch to dropping unknowns.

#### Scenario: Synonym without alias falls to unknown
- GIVEN GT contains "traffic light" and a prediction uses desc "stoplight"
- WHEN exported without alias/fuzzy mapping
- THEN the prediction is assigned to category `unknown` unless `unknown` dropping is enabled, in which case it is dropped.

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
- The evaluator SHALL run COCOeval (bbox, and segm when polygons exist) and report AP@[0.50:0.95], AP50, AP75, APs/m/l, AR1/10/100, and per-class AP.
- It SHALL also report robustness counters: invalid-JSON/geometry rate (including degenerate), empty-prediction rate, unknown-desc rate, size mismatches, and multi-image skips; optional class-agnostic IoU recall and open-vocab recall (ignoring class) MAY be enabled.
- Default behavior is soft-drop with counters; a `--strict-parse` flag MAY abort on first parse/validation error.

#### Scenario: Metrics summary persisted
- GIVEN a valid GT/pred pair
- WHEN the evaluator runs
- THEN it writes `metrics.json` containing the COCO summary metrics and the robustness counters.

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
