# Design: Detection Evaluator (CoordExp)

## Scope
- Offline evaluator converting CoordExp-style JSONL (single-image per record) into COCO artifacts and reporting detection metrics. Default training loop remains unchanged; add an optional training-time eval hook and a standalone CLI.
- Support bbox and polygon geometries; lines are converted to bbox (no segm). Add a TODO hook for polygon GIoU alongside COCOeval.

## Inputs and Flow
1) **Ground truth:** CoordExp JSONL (one image, width/height, single geometry, desc). Treated as clean and authoritative for image size.
2) **Predictions:** JSONL from `vis_tools/vis_coordexp.py` (or raw generations parsed identically). Assumed to reference the same images/width/height as the GT JSONL; conflicts use GT and increment a `size_mismatch` counter. A flag indicates whether predictions are normalized 0–999 or already pixel-space to avoid double scaling.
3) **Config/CLI:** Paths, IoU thresholds (COCO defaults), segm toggle, output dir (overwrites silently), overlay toggle/count (default off), inference knobs (temperature, repetition_penalty) aligned with `vis_tools/vis_coordexp.py` when running generation + eval together.

## Parsing & Coord Handling
- Reuse parsing helpers from `vis_tools/vis_coordexp.py`; accept coord tokens or ints in [0,999].
- One geometry per object; multiple geometries → drop object with warning. Degenerate/zero-area geometry is dropped and counted. Lines are converted to their tight bbox (no segm) for detection.
- Denormalize with record width/height (assumed correct), clamp, round-to-nearest-int before export. If a flag declares preds are already pixel-space, skip denorm and only clamp/round.
- Malformed/truncated objects are dropped individually; other objects in the record remain.

## Category & Scoring
- Exact string match only; no alias map or fuzzy matching (alias-free stance). Non-identical desc strings are distinct categories. Text is assumed English; no additional normalization beyond lowercase/strip.
- Unknown handling: bucket unmapped/empty desc into `unknown` (do not drop) by default; dropping is optional.
- Scores: constant 1.0 for all predictions (greedy decoding has no reliable confidence); any provided score fields are ignored. Ties resolved deterministically by input order.

## COCO Export & Metrics
- Emit `coco_gt.json` and `coco_preds.json` (xywh). Include segmentation for polygons; lines are already converted to bbox-only.
- Run COCOeval for bbox and segm (when polygons exist). Add TODO placeholder to compute polygon GIoU in the future.
- Diagnostics: counts of dropped/invalid objects (including degenerate), size mismatches, raw parse errors, unknown-desc rate. No separate alias/fuzzy handling.

## CLI, Logging & Integration
- Primary entry: `scripts/evaluate_detection.py` (alias module run acceptable). Overwrites output directory without prompting. Overlay debug renders are off by default; when enabled, reuse the drawing logic from `vis_tools` and emit top-K FP/FN (default 8–16) PNGs.
- Outputs: `metrics.json`, `per_class.csv`, `per_image.json`, optional overlays, including raw error reasons.
- Training integration: provide an eval hook usable from `evaluation_step` that logs metrics to TensorBoard under `eval/*`; also support standalone checkpoint+JSONL evaluation via the CLI. Inference/eval expected to run on GPU; CPU not supported.
- Inference knobs: expose temperature and repetition_penalty to mirror `vis_tools/vis_coordexp.py` behavior; other generation defaults unchanged.

## Edge Cases & Assumptions
- Single-image intent: if a record contains multiple images, only index 0 is evaluated and the event is counted (`multi_image_ignored`).
- Width/height mismatches are unexpected; if detected, warn but proceed using GT sizes.
- No open-vocab recall or aliasing; strict string equality governs categories.
- Output directory overwrite is allowed silently.

## Open Questions / Ambiguities to Settle Before Implementation
- **Pixel-space flag default:** Should predictions be assumed normalized (0–999) unless `--pixel-space` is passed, or auto-detected by value range?
- **Unknown handling policy in metrics:** Are unknown-bucket detections included in COCO evaluation (as their own category) or excluded from AP and reported only in counters?
- **Lines → bbox behavior:** Any cases where lines should be excluded entirely (e.g., when representing scanlines rather than objects)?
- **Polygons plus bbox GT:** If GT supplies both polygon and bbox, do we export both or prefer polygon-only for segm while reusing bbox for detection?
- **Score passthrough contract:** If a `score` field exists, is it guaranteed in [0,1]? Should we clamp/validate or fail?
- **Open-vocab recall metrics:** Should class-agnostic IoU recall be on by default or opt-in to avoid extra compute?
- **Overlay generation scope:** Do we need HTML alongside PNG, and what is the default sample count (K)?
- **Training-time hook cadence:** Should the hook run every eval epoch or only on explicit trigger to avoid slowing training?
- **Concurrency/sharding:** How should deterministic image ids be coordinated when evaluating shards independently and later merging?
