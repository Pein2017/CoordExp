# Stage-2 (AB) Channel-A Only — Inference/Eval Notes (2026-02-01)

This note summarizes what we observed when benchmarking **Channel-A only** Stage-2 checkpoints on LVIS bbox-only validation, focusing on:

- **Stability** under autoregressive inference (empty/invalid/degenerate outputs).
- **Duplication** as *exactly repeated* `(desc, bbox)` predictions (IoU==1.0 / identical integer pixel bbox).
- How **generation knobs** (notably `repetition_penalty`) affect the above.

## Checkpoints

Benchmarked (HF backend):

- `output/stage2_ab/a_only_ckpt_6064`  (\"A only\", intended `n_softctx_iter≈2`)
- `output/stage2_ab/a_only_iter_1_ckpt_6064` (\"A only iter=1\", intended `n_softctx_iter≈1`)

Related configs:

- `configs/stage2_ab/prod/a_only.yaml`
- `configs/stage2_ab/prod/a_only_iter-1.yaml`

> Note: `configs/stage2_ab/prod/a_only_iter-1.yaml` currently ends with an invalid YAML fragment (`: 1`).
> Treat any “iter=1” conclusions as “this checkpoint was produced from an iter=1 intention” unless the training run logs confirm the actual `n_softctx_iter`.

## Dataset / evaluation

- GT JSONL: `public_data/lvis/rescale_32_768_bbox_max60/val.bbox_only.max60.coord.jsonl`
- Mode: `coord` (coord tokens found; `pred_coord_mode=auto`)
- Evaluation: COCO-style bbox metrics written under `output/bench/*/eval/metrics.json`

## Run A (older run): `limit=1000`, `repetition_penalty=1.05`

This run existed before the 2026-02-01 rerun and was analyzed from the then-current `output/bench/*` artifacts.
The directories were later overwritten by Run B, so the numbers here are recorded from the earlier analysis.

Generation settings (from the earlier `summary.json`):

- `temperature=0.1`, `top_p=1.0`, `max_new_tokens=2048`, `repetition_penalty=1.05`, `seed=42`
- `limit=1000`

### Key observations (Run A)

1) **`n_softctx_iter≈2` was more stable than `≈1`** on degenerate/invalid geometry:
   - “invalid_geometry” (bench summary): `5` vs `38` (iter=2 vs iter=1).
   - In practice, these were dominated by boxes that collapse to **zero width/height** after bin→pixel rounding.

2) **Catastrophic exact-duplication loops existed**, i.e. the model repeats the *same bbox* many times for the same `desc`.
   - Examples seen in saved visualizations:
     - `pipe ×63` (one exact bbox repeated 63 times)
     - `spectacles ×58`
     - `cup ×45`

3) Duplication vs “overprediction”:
   - Overprediction (many objects / partially labeled LVIS) can be acceptable *if boxes are distinct*.
   - The failures above were **not** that; they are **generation loops** producing repeated `(desc, bbox)`.

Artifacts:

- Visualizations from Run A (exact-dup focused) were saved under `temp/bench_vis/exact_dup/`.

## Run B (2026-02-01 rerun): `limit=100`, `repetition_penalty=1.1`

This is the current content of:

- `output/bench/a_only_ckpt_6064/*`
- `output/bench/a_only_iter_1_ckpt_6064/*`

Generation settings:

- `temperature=0.1`, `top_p=0.9`, `max_new_tokens=2048`, `repetition_penalty=1.1`, `seed=42`
- `limit=100`

### Stability summary (Run B)

From `output/bench/*/summary.json` + `gt_vs_pred.jsonl`:

- `a_only_ckpt_6064`: `empty_pred=1`, `invalid_geometry=0`
- `a_only_iter_1_ckpt_6064`: `empty_pred=1`, `invalid_geometry=2` (appears as 2 `degenerate` drops)

### Duplication summary (Run B)

Using **strict exact-duplication** (identical integer pixel bboxes for the same `desc`):

- `a_only_ckpt_6064`:
  - `dup_exact = 0 / 522 preds` (**0.00%**), `0/100` images affected
- `a_only_iter_1_ckpt_6064`:
  - `dup_exact = 10 / 463 preds` (**2.16%**), `1/100` images affected
  - Worst case: `images/train2017/000000001204.jpg` had `lanyard` repeated with `max_repeat=×11`

Near-duplication (useful secondary metric): IoU>0.95 within a `desc`.

- Both runs showed low near-dup rates (~2%), with a single worst image each.

### Metrics snapshot (Run B; 100 samples, noisy)

COCO bbox AP from `output/bench/*/eval/metrics.json`:

- `a_only_ckpt_6064`: `bbox_AP ≈ 0.2496`
- `a_only_iter_1_ckpt_6064`: `bbox_AP ≈ 0.2241`

Because this is `limit=100` and `top_p=0.9`, treat this as directional only.

## Interpretation / conclusion so far

1) **Exact bbox repetition is a decoding instability**, not a dataset labeling artifact.
   - When you see `×40~×60` repeats of *the same bbox*, it is not “crowded instances”; it is a generation loop.

2) Increasing `repetition_penalty` to `1.1` (with `top_p=0.9`) **greatly reduces catastrophic exact-duplication**.
   - On the 100-sample rerun, `a_only` had `dup_exact=0`.

3) Channel-A iteration (intended `n_softctx_iter=2` vs `1`) still appears to help **geometry stability**
   under autoregressive inference, but the training-side config needs verification due to the YAML issue.

## Recommended next experiments

To keep comparisons clean and paper-ready:

1) **Verify `n_softctx_iter` in training**
   - Fix `configs/stage2_ab/prod/a_only_iter-1.yaml` and/or record the resolved config in the run artifact.

2) Run the planned ablation from `progress/full_idea.md`:
   - `n_softctx_iter ∈ {1, 2, 3}` with *identical* inference settings.

3) Decoding sweep for stability vs quality:
   - `repetition_penalty ∈ {1.05, 1.10, 1.15}`
   - (Optional) keep `top_p=1.0` and `temperature=0.0` for deterministic eval, then measure robustness with mild sampling separately.

4) Always report (in addition to AP/AR):
   - `empty_pred`, `degenerate/invalid_geometry`
   - `dup_exact` rate (IoU==1 / identical bbox) and “near-dup” rate (IoU>0.95)
   - preds/image tail (p99/max) to catch rare meltdowns.

