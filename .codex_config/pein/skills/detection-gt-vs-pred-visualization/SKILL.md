---
name: detection-gt-vs-pred-visualization
description: Render, modify, or debug detection-style GT/pred visualizations in CoordExp when the user asks to draw objects on an input image, inspect an input-image-plus-expected-output-object-list pairing, review evaluator/monitor GT-vs-Pred scenes, or compare multiple detection outputs. Reuse the shared canonical visualization pipeline and exact GT-vs-Pred contract instead of building a bespoke renderer or ad hoc schema.
---

# Detection GT/Pred Visualization

Reuse the existing shared visualization stack.
Do not introduce a new renderer unless the current pipeline cannot express the requested figure.

## Primary Entry Points

- single-run render:
  - `src/infer/vis.py`
  - `vis_tools/vis_coordexp.py`
  - `scripts/run_vis.sh`
- canonical normalization + shared review renderer:
  - `src/vis/gt_vs_pred.py`
- comparison composition:
  - `src/vis/comparison.py`
  - `scripts/analysis/rollout_backend_bench/vis_rollout_backend_compare.py`
- evaluator integration:
  - `src/eval/detection.py`

## Workflow

1. Identify the input family:
   - raw `gt_vs_pred.jsonl` or `gt_vs_pred_scored.jsonl`
   - canonical `vis_resources/gt_vs_pred.jsonl`
   - evaluator-selected scenes or precomputed matching payloads
   - comparison members such as `pred_hf.jsonl` and `pred_vllm.jsonl`
   - ad hoc `image + expected object list` requests
2. Reuse the shared path:
   - single-run artifact render: call `src.infer.vis.render_vis_from_jsonl(...)`
   - programmatic or ad hoc scene: call `materialize_gt_vs_pred_vis_resource(...)` then `render_gt_vs_pred_review(...)`
   - comparison scene: call `compose_comparison_scenes_from_jsonls(...)` or the compare script
3. Keep input and output paths explicit.
4. Fail fast on contract violations; do not hide missing fields with renderer-local fallback logic.

## Canonical Contract

- canonical top-level fields:
  - `schema_version`
  - `source_kind`
  - `record_idx`
  - `image`
  - `width`
  - `height`
  - `coord_mode: "pixel"`
  - `gt`
  - `pred`
- canonical object fields:
  - `index`
  - `desc`
  - `bbox_2d`
- shared review requires canonical `matching`:
  - `pred_index_domain: canonical_pred_index`
  - `gt_index_domain: canonical_gt_index`
  - `matched_pairs`
  - `fn_gt_indices`
  - `fp_pred_indices`

## Precaution

- Reuse shared geometry helpers.
  - `src.common.geometry.denorm_and_clamp`
  - `src.common.geometry.bbox_from_points`
  - `src.common.geometry.object_geometry.extract_single_geometry`
- Never add renderer-local `norm1000` or coord-token inverse scaling.
- Preserve prediction order exactly as emitted by the source artifact.
- Preserve explicit prediction indices when the source provides them; matching may refer to stable non-dense indices.
- Canonicalize GT ordering deterministically through the shared adapter; let it remap source-local GT indices into `canonical_gt_index`.
- Do not overwrite raw `<run_dir>/gt_vs_pred.jsonl`; derived canonical resources belong under `<run_dir>/vis_resources/gt_vs_pred.jsonl`.
- Shared GT-vs-Pred review rendering requires canonical matching. Materialize or normalize matching before rendering; do not recompute matching inside the renderer as a fallback.
- Keep the default review semantics unchanged:
  - `1x2` layout
  - GT left, Pred right
  - GT green
  - FN orange
  - matched Pred green
  - FP Pred red
  - labels focus on `FN` and `FP` objects by default
- For comparison scenes, compose multiple canonical single-view members and verify exact GT equivalence (`width`, `height`, canonical `gt`) before drawing.

## Ad Hoc Image + Object List Requests

- If the user provides only an image path and an expected output object list, first build a tiny detection-style record with:
  - `image`
  - `width`
  - `height`
  - `gt` or `pred`
- Then send that record through the shared canonical path instead of drawing boxes directly in bespoke code.

## Avoid

- Do not build a new matplotlib or PIL overlay path if `src/vis/` or `src/infer/vis.py` already covers the request.
- Do not invent a new compare-only per-object schema.
- Do not change colors, panel order, or matching semantics unless the user explicitly requests a different figure style and that change does not violate the repo contract.

## Quick References

- contract:
  - `openspec/changes/unify-gt-vs-pred-visualization/specs/gt-vs-pred-visualization/spec.md`
- implementation:
  - `src/vis/gt_vs_pred.py`
  - `src/vis/comparison.py`
  - `src/infer/vis.py`
  - `src/eval/detection.py`
