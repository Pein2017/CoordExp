# Detection Evaluator (CoordExp)

Offline evaluator to compute COCO-style metrics and/or an F1-ish set-matching metric from CoordExp JSONL.

## Inputs
- `pred_jsonl`: Pixel-space predictions produced by `scripts/run_infer.py` (unified engine); **must include inline `gt`** per line.
- Geometry objects live under `gt` / `pred` arrays with canonical fields `{type, points, desc}` in base artifacts.  
  For COCO/mAP, use the scored artifact (`gt_vs_pred_scored.jsonl`) where kept predictions also include `score` plus per-record score provenance keys (`pred_score_source`, `pred_score_version`).
- Width/height are taken from the inline GT; size mismatches are counted but do not abort.
- JSONL parse strictness is config-driven:
  - `eval.strict_parse=true`: fail fast on the first malformed/non-object record.
  - `eval.strict_parse=false` (default): warn+skip deterministically with bounded diagnostics (`warn_limit=5`, `max_snippet_len=200`).

## Behavior
- Pixel-ready consumption: no coord-mode inference/denorm is performed; polygons are exported directly as COCO `segmentation` (mask IoU), bboxes as `bbox`. Line geometries are rejected and counted as invalid.
- One geometry per object; degenerate/invalid entries are dropped with counters and recorded per-image for diagnostics.
- Categories use exact GT desc strings. Predicted `desc` values that are not in the GT category set are **semantically mapped** to the closest GT desc (excluding synthetic `unknown`) using a sentence-transformer encoder; if the best match is below threshold, the prediction is **dropped** (map-or-drop; no bucket/drop fallbacks).
  - `--unknown-policy` and `--semantic-fallback` are **deprecated/ignored** (retained only for back-compat; the evaluator will warn once).
  - When semantic mapping runs, the evaluator writes `semantic_desc_report.json` to the output directory for inspection.
- COCO scoring always honors each kept prediction’s `pred[*].score`; missing/non-numeric/non-finite/out-of-range scores are contract violations and fail fast with record/object indices.
- For COCO runs, unscored legacy artifacts are rejected: each record must include non-empty `pred_score_source` and integer `pred_score_version`.
- COCOeval runs for bbox and segm (when polygons exist). TODO: polygon GIoU hook.
  - Milestone note: `confidence-postop` v1 scores `bbox_2d` only and drops unscorable predictions in the scored artifact. For scored COCO runs, prefer bbox-only evaluation (`use_segm: false`) until polygon confidence is added.
- Optional F1-ish mode runs greedy 1:1 matching by IoU, then reports set-level counts (matched / missing / hallucination) and semantic-on-matched correctness (exact or embedding similarity).
  - By default (`--f1ish-pred-scope annotated`), predictions whose `desc` is **not semantically close to any GT `desc` in the image** are **ignored** (not counted as FP). This makes F1-ish behave like “how well do we recover annotated objects” on partially-annotated / open-vocab settings.
  - Use `--f1ish-pred-scope all` for strict counting that penalizes any extra predictions as FP.
- GPU is optional. Only semantic matching needs extra compute; set `--semantic-device cpu` (default: `auto`) if you want to force CPU.

## CLI
YAML-first (recommended for reproducibility):
```
PYTHONPATH=. conda run -n ms python scripts/evaluate_detection.py \
  --config configs/eval/detection.yaml
```

Legacy flags (still supported during transition):
```
PYTHONPATH=. conda run -n ms python scripts/evaluate_detection.py \
  --pred_jsonl <path> \
  --out_dir eval_out \
  [--metrics coco|f1ish|both] \
  [--semantic-model <hf-id>] [--semantic-threshold 0.6] [--semantic-device auto] \
  [--f1ish-iou-thrs 0.3 0.5] [--f1ish-pred-scope annotated|all] \
  [--strict-parse] [--no-segm] [--iou-thrs 0.5 0.75] \
  [--overlay --overlay-k 12]
```
Artifacts (always): `metrics.json` (metrics + counters), `per_image.json`, optional `overlays/` when enabled.

Artifacts (when COCO is enabled via `--metrics coco|both`): `per_class.csv`, `coco_gt.json`, `coco_preds.json`.

Artifacts (when F1-ish is enabled via `--metrics f1ish|both`):
- `matches.jsonl` for the primary IoU threshold (0.5 if present in `--f1ish-iou-thrs`, else the max threshold).
- Optional `matches@<thr>.jsonl` for additional IoU thresholds (formatted with two decimals, e.g. `matches@0.30.jsonl`).
- `per_image.json` gains a stable `f1ish` field keyed by IoU threshold strings (e.g. `"0.50"`), containing per-image TP/FP/FN and semantic-on-matched counts.

## Staged workflow

The base unified artifact is `gt_vs_pred.jsonl` (pixel-space, with inline GT).

1) Run inference to produce `gt_vs_pred.jsonl`:

```bash
PYTHONPATH=. conda run -n ms python scripts/run_infer.py \
  --gt_jsonl <data.jsonl> \
  --model_checkpoint <ckpt> \
  --mode auto \
  --out output/infer/<run_name>/gt_vs_pred.jsonl \
  --summary output/infer/<run_name>/summary.json
```

2) Run confidence post-op (CPU-only) to produce:
- `pred_confidence.jsonl`,
- `gt_vs_pred_scored.jsonl`,
- `confidence_postop_summary.json`.

```bash
PYTHONPATH=. conda run -n ms python scripts/postop_confidence.py \
  --config configs/postop/confidence.yaml
```

3) Evaluate:
- COCO (`metrics: coco|both`): use `gt_vs_pred_scored.jsonl`.
- f1ish-only (`metrics: f1ish`): base `gt_vs_pred.jsonl` is allowed.

```bash
PYTHONPATH=. conda run -n ms python scripts/evaluate_detection.py \
  --pred_jsonl output/infer/<run_name>/gt_vs_pred_scored.jsonl \
  --out_dir output/infer/<run_name>/eval \
  --metrics coco
```

4) Visualize (consumes inline GT from the base artifact):

```bash
PYTHONPATH=. conda run -n ms python vis_tools/vis_coordexp.py \
  --pred_jsonl output/infer/<run_name>/gt_vs_pred.jsonl \
  --save_dir output/infer/<run_name>/vis \
  --limit 20 \
  --root_image_dir <image_root>
```

Tip: you can also run infer+eval+vis from a single YAML pipeline config:

```bash
PYTHONPATH=. conda run -n ms python scripts/run_infer.py --config configs/infer/pipeline.yaml
```

When pipeline eval requests COCO metrics, configure `artifacts.gt_vs_pred_scored_jsonl`
and run confidence post-op first; otherwise pipeline eval fails fast by design.

Tip: for benchmarkable runs (like the ones driven by `scripts/run_infer_eval.sh`), prefer the dedicated configs under `configs/bench/` instead of re-typing long env/flag lists. For example:

```bash
PYTHONPATH=. conda run -n ms python scripts/run_infer.py \
  --config configs/bench/a_only_ckpt_6064_infer_eval.yaml
```

These configs already pin `run.name`, `output_dir`, checkpoints, generation, and eval knobs so the inferred artifacts stay consistent and the `resolved_config.json` captures the full run metadata without having to repeat dozens of terminal arguments.

## Validation checklist (unified pipeline)

This is a lightweight checklist intended for paper-ready runs and to prevent
silent contract drift between infer/eval/vis stages.

1) Artifact layout (deterministic)
- Confirm `<run_dir>/gt_vs_pred.jsonl` and `<run_dir>/summary.json` exist.
- For score-aware COCO runs, confirm `<run_dir>/gt_vs_pred_scored.jsonl`,
  `<run_dir>/pred_confidence.jsonl`, and `<run_dir>/confidence_postop_summary.json` exist.
- If you ran the YAML pipeline, confirm `<run_dir>/resolved_config.json` exists
  (this records the effective config/stages/artifacts for reproducibility).
- Treat `resolved_config.json` compatibility as:
  - stable top-level keys: `schema_version`, `config_path`, `root_image_dir`, `root_image_dir_source`, `stages`, `artifacts`,
  - redacted `cfg` snapshot as diagnostics-only (opaque; not a stable contract surface).
- If `eval` ran, confirm `<run_dir>/eval/metrics.json` exists.
- If `vis` ran, confirm `<run_dir>/vis/vis_0000.png` exists (or more, depending on limit).

2) Schema sanity (quick spot-check)
- `gt_vs_pred.jsonl` lines are JSON dicts with canonical keys: `image`, `width`, `height`,
  `gt` (list), `pred` (list), and `errors` (list).
- Geometry is pixel-ready:
  - `coord_mode` is `"pixel"` (or `null` for records without geometries),
  - `gt[*].points` / `pred[*].points` are flat numeric lists,
  - bbox points are length 4; polygon points are even-length and >= 6.

3) Determinism + provenance
- Prefer setting `infer.generation.seed` (or `--seed` in legacy mode) for best-effort
  determinism under the HF backend.
- For throughput, you can batch decoding:
  - HF backend: set `infer.generation.batch_size` (>1) to use batched `model.generate()`.
  - vLLM backend: set `infer.generation.batch_size` and (optionally) `infer.backend.client_concurrency`
    to issue multiple requests concurrently (server-side batching still depends on your vLLM server settings).
- Record the backend choice and generation config in run artifacts:
  - YAML pipeline: `<run_dir>/resolved_config.json`
  - Legacy: `<run_dir>/summary.json` + your shell logs.
- For HF attention fallback auditing, use exact `summary.json` fields:
  - `backend.attn_implementation_requested`
  - `backend.attn_implementation_selected`
- Note: `infer.backend.type=vllm` does not guarantee byte-identical outputs; treat it as
  schema-stable, not token-stable.

4) Metric sanity (small subset)
- Run with a small limit (e.g., 10) and check that:
  - COCO metrics are present when enabled, and are finite numbers,
  - counters (invalid_json / invalid_geometry / etc.) are not exploding,
  - overlays render when enabled (optional).

## Comparison recipe (limit=10): YAML pipeline vs legacy wrapper

Goal: verify that the unified YAML pipeline produces the same contract and
comparable metrics as the legacy wrapper scripts on a tiny subset.

1) Legacy wrapper (flag-only inference + evaluator)

```bash
CKPT=<ckpt_path> \
GT_JSONL=<gt_jsonl_path> \
OUTPUT_BASE_DIR=output/infer/compare_legacy \
MODE=auto \
LIMIT=10 \
SEED=42 \
scripts/run_infer_eval.sh
```

2) YAML pipeline (infer + eval + vis)

Create a small one-off config (recommended to keep experiments reproducible):

```bash
cp configs/infer/pipeline.yaml temp/compare_infer.yaml
# Edit temp/compare_infer.yaml:
# - run.name: compare_yaml
# - infer.gt_jsonl / infer.model_checkpoint
# - infer.limit: 10
# - infer.generation.seed: 42
PYTHONPATH=. conda run -n ms python scripts/run_infer.py --config temp/compare_infer.yaml
```

3) Compare outputs
- Line counts should match `limit`:
  - `wc -l output/infer/compare_legacy/gt_vs_pred.jsonl`
  - `wc -l output/infer/compare_yaml/gt_vs_pred.jsonl`
- Compare summary + metrics for large drift:
  - `cat .../summary.json`
  - `cat .../eval/metrics.json`
- If using HF backend + fixed seed, the JSONL should be *very* similar; if using vLLM,
  expect different raw text but stable schema and reasonable metric proximity.

## Training hook
Use `src.callbacks.DetectionEvalCallback` with pre-generated prediction JSONL:
```python
from src.callbacks import DetectionEvalCallback

callback = DetectionEvalCallback(
    gt_jsonl="path/to/gt.jsonl",
    pred_jsonl="path/to/preds.jsonl",  # generated separately
    out_dir="eval_out",
    mode="text",
)
trainer.add_callback(callback)
```
Metrics are logged under `eval_det_*` keys; callback skips if CUDA is unavailable.
