---
title: Visualization Tools Index for Detection and Rollout Inspection
status: completed-diagnostic
scope: visualization-tools
topics: [visualization, overlays, rollout, eval, diagnostics]
references:
  - docs/eval/WORKFLOW.md
  - scripts/README.md
---

# Visualization Tools Index for Detection and Rollout Inspection

Date: 2026-03-11

This note inventories the most relevant visualization tools currently available in the CoordExp codebase for:

- GT vs prediction overlay inspection
- rollout failure / monitor-dump inspection
- backend-to-backend rollout comparison
- run-to-run comparison around detection outputs

The goal is to provide a short “open these first” index instead of raw grep output.

---

## 1) Primary Overlay Tool: `scripts/run_vis.sh`

Best for:

- quick GT-vs-pred overlay rendering from a `gt_vs_pred.jsonl`
- browsing inference outputs after a standard run
- manual sanity-check of predictions on real images

Entry points:

- `scripts/run_vis.sh`
- `vis_tools/vis_coordexp.py`
- `src/infer/vis.py`

What it does:

- materializes `vis_resources/gt_vs_pred.jsonl`
- resolves the image path
- renders the shared `1x2` GT-left / Pred-right reviewer
- uses error-focused labels by default (`FN` / `FP`)
- writes one PNG per sample

Key code handles:

- `scripts/run_vis.sh`
- `vis_tools/vis_coordexp.py`
- `src/infer.vis/render_vis_from_jsonl`

Useful command pattern:

```bash
pred_jsonl=output/.../gt_vs_pred.jsonl \
save_dir=output/.../vis \
root_image_dir=public_data/coco/rescale_32_1024_bbox_max60 \
bash scripts/run_vis.sh
```

Why this is the default:

- dependency-light
- works directly on the core inference artifact
- easiest tool for “show me what GT and prediction look like”

---

## 2) Eval Overlay Path: `scripts/evaluate_detection.py --overlay`

Best for:

- rendering overlays during evaluation
- sampling images directly from the evaluated artifact without a separate vis pass
- attaching overlays to the same output directory as eval metrics

Entry points:

- `scripts/evaluate_detection.py`
- `src/eval/detection.py`

What it does:

- runs standard evaluation
- materializes the canonical visualization sidecar
- optionally renders overlay PNGs into `<eval_out>/overlays/`
- reuses the same shared `1x2` reviewer semantics as `run_vis.sh`

Key code handles:

- `src/vis.gt_vs_pred/materialize_eval_gt_vs_pred_vis_resource`
- `src/eval.detection/evaluate_and_save`

Relevant config keys:

- `overlay: true`
- `overlay_k: <N>`

When to prefer this over `run_vis.sh`:

- when you are already running eval
- when you want overlays colocated with metrics
- when you want only a small sampled subset instead of all records

---

## 3) Monitor Dump Visualizer: `vis_tools/vis_monitor_dump_gt_vs_pred.py`

Best for:

- inspecting Stage-2 monitor dumps
- reviewing matched / unmatched GT and pred objects from rollout monitoring
- understanding aggregate FP / FN class patterns in dumped monitor samples

Entry point:

- `vis_tools/vis_monitor_dump_gt_vs_pred.py`

What it does:

- reads one monitor dump JSON or a directory of monitor dumps
- renders GT / pred paired visualizations
- produces a `class_summary.json` with aggregate FP / FN class counts

Key code handles:

- `vis_tools.vis_monitor_dump_gt_vs_pred/main`
- `vis_tools.vis_monitor_dump_gt_vs_pred/_render_pair`
- `vis_tools.vis_monitor_dump_gt_vs_pred/_draw_class_summary`

Why it matters:

- this is the most specialized visualization tool for Stage-2 debugging
- it is better than generic overlays when you need rollout-monitor semantics

---

## 4) Backend Comparison Visualizer: `scripts/analysis/rollout_backend_bench/vis_rollout_backend_compare.py`

Best for:

- comparing two rollout sources on the same image
- especially HF vs vLLM rollout comparisons
- seeing GT, one prediction set, another prediction set, and a shared class legend side-by-side

Entry point:

- `scripts/analysis/rollout_backend_bench/vis_rollout_backend_compare.py`

Key code handle:

- `scripts/analysis/rollout_backend_bench/vis_rollout_backend_compare.py:draw_compare`

What it does:

- opens the image once
- renders a 4-panel figure:
  - GT
  - HF rollout
  - vLLM rollout
  - legend with counts

Why it is useful:

- best existing code path for “same image, compare two different rollout sources”
- easy to adapt if you later want “baseline vs Oracle sample A vs Oracle sample B”

---

## 5) Comparison / Triage Tool: `scripts/analysis/compare_detection_runs.py`

Best for:

- artifact-level comparison between runs
- summarizing token traces, `gt_vs_pred`, confidence outputs, and eval metrics
- triaging which runs are worth visual inspection

Entry point:

- `scripts/analysis/compare_detection_runs.py`

What it is not:

- it is not primarily an image renderer

Why it still belongs here:

- it helps decide *which* runs or samples to visualize next
- it is a good pre-pass before using the rendering tools above

---

## 6) Rollout Stability Plot Tool: `scripts/analysis/report_rollout_stability.py`

Best for:

- writing plot-ready `gt-vs-pred` rows
- detecting failure-heavy rollouts before manual inspection

Entry point:

- `scripts/analysis/report_rollout_stability.py`

Why it is relevant:

- not a direct PNG overlay tool
- but useful as a sample-selection tool for later visualization

---

## 7) Current Generated Review Bundle

Not a codebase tool itself, but a generated artifact you can inspect immediately:

- `output/analysis/heavy_fp_rollout_review_20260311/ul_res_1024/`
- `output/analysis/heavy_fp_rollout_review_20260311/ul_res_1024_v2/`

These were created from the codebase’s rendering primitives to review heavy-FP rollout cases.

Each folder includes:

- PNG visualizations
- `README.md`
- `manifest.json`

This is the fastest place to look for the current heavy-FP manual audit.

---

## 8) Recommended Read Order

If your goal is “I want to inspect suspicious predictions on images”:

1. `scripts/run_vis.sh`
2. `vis_tools/vis_coordexp.py`
3. `src/infer/vis.py`

If your goal is “I want to inspect Stage-2 rollout / monitor failures”:

1. `vis_tools/vis_monitor_dump_gt_vs_pred.py`
2. `progress/diagnostics/stage2_ul_capture_highres1024_2026-03-09.md`

If your goal is “I want side-by-side rollout comparison”:

1. `scripts/analysis/rollout_backend_bench/vis_rollout_backend_compare.py`

If your goal is “I want to choose which runs / samples deserve rendering”:

1. `scripts/analysis/compare_detection_runs.py`
2. `scripts/analysis/report_rollout_stability.py`

---

## 9) Bottom Line

The most relevant visualization stack in this repo is:

- `scripts/run_vis.sh` -> `vis_tools/vis_coordexp.py` -> `src/infer/vis.py`

for general GT-vs-pred overlays,

and:

- `vis_tools/vis_monitor_dump_gt_vs_pred.py`

for Stage-2 rollout-monitor inspection.

For pairwise rollout comparison, the best existing reusable handle is:

- `scripts/analysis/rollout_backend_bench/vis_rollout_backend_compare.py`
