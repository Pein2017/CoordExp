## Context
- Inference is split across `scripts/run_infer.py` (coord-only, norm1000 outputs) and `vis_tools/vis_coordexp.py` (mixed schema, embedded scaling); evaluation consumes legacy fields and infers coordinate modes heuristically.
- Requirement: a single inference surface for coord-token and pure-text checkpoints that outputs absolute-pixel geometries with preserved types (`bbox_2d`, `poly`), fixed scores, deterministic behavior, per-sample error reporting, and explicit mode; generation params come from CLI flags. Consumers (vis/eval) must be updated; no legacy dual-schema fallback.

## Goals / Non-Goals
- Goals: unify generation, parsing, scaling, and output schema; enforce mode-specific scaling; preserve raw geometry; integrate error handling/counters and seeding; update evaluator/vis to consume the new schema; cut legacy mixed-format outputs; ensure a simple flow where (1) inference centralizes scaling/validation, (2) outputs are pixel-ready JSONL for both gt/pred, and (3) downstream visualizer/evaluator load `pred.jsonl` without additional normalization.
- Non-goals: new model architectures, batching/multi-image support, confidence calibration, or line-specific metrics (line tolerated structurally only).

## Decisions (pre-implementation)
- Capability name: `inference-engine`.
- Change ID: `add-centralized-inference-engine`.
- Output schema: per-sample JSON with `gt` and `pred` arrays of objects `{type, points, desc, score=1.0}` in absolute pixels; auxiliary `raw_output`; `errors` list per sample; include `width/height`, `image`, `mode`. No legacy `predictions`/norm fields.
- Mode handling: `--mode` required; no auto-detect. Coord mode enforces 0-999 on GT/preds and denorms both; text mode keeps GT pixels, denorms preds when norm/tokens, allows pixel preds; mismatch GT/mode recorded as per-sample error and preds skipped.
- Polygon handling: keep polygons as-is; evaluate via COCO segmentation/mask IoU derived from polygon vertices (single ring, clamped, non-degenerate). Lines carried through and clamped but excluded from metrics.
- Generation config: CLI flags only (temperature, top_p, max_new_tokens, repetition_penalty, seed, device, limit, out).
- Failure policy: on missing size/invalid coords/odd points/mode-GT mismatch, record error, empty preds, continue; run-level succeeds.
- Determinism: seed torch + cuda and pass a seeded `torch.Generator` into `model.generate`.
- Limit: truncate after N samples consistently for both GT consumption and output count.
- Counters: aggregate invalid_json/geometry/coord/size_mismatch/empty_pred/mode_gt_mismatch etc.; emit summary JSON (e.g., `pred.summary.json`) alongside `pred.jsonl` with counters map, mode, total samples processed/emitted, and distinct error codes.
- Consumers: update evaluator and vis tool in this change to read the new schema (types in {`bbox_2d`,`poly`,`line`}; polygons via COCO segm; lines rendered/ignored for metrics) and drop legacy parsing.
 - End-to-end flow: inference handles all norm/denorm and mode logic; the emitted JSONL is already pixel-space for gt/pred; downstream tools perform straightforward rendering/evaluation with no extra scaling.

## Alternatives Considered
- Auto-detect mode from GT tokens: rejected to avoid silent mis-scaling.
- Emitting legacy norm fields for backward compatibility: rejected; instead, consumers are updated in the same change.
- Custom polygon IoU (analytic) instead of COCO segm/mask IoU: rejected for this change to stay compatible with existing COCOeval flow and reduce effort.

## Risks / Mitigations
- Risk: Downstream tools break on schema change. Mitigation: update evaluator/vis within this change and document the new schema.
- Risk: Mask rasterization (COCO segm) differences vs analytic polygon IoU. Mitigation: document that COCO mask IoU is used, clamp polygons, and add tests to verify expected behavior.
- Risk: Performance overhead from per-sample validation. Mitigation: keep validation lightweight and summarize counters once.

## Open Questions (resolved via user)
- Keep raw geometry types with desc: yes.
- Scores fixed at 1.0: yes.
- Lines not needed for metrics: yes, tolerate structurally only.
