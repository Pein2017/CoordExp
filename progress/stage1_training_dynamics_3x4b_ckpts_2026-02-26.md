# Stage-1 Training Dynamics (4B): soft_ce_only vs pure_ce_only vs soft_ce_hard_ce_mixed (2026-02-26)

This note analyzes *training-time dynamics* (teacher-forcing metrics) for three **4B-sized** stage-1 runs whose merged checkpoints were evaluated in:

- [progress/stage1_coco80_eval_4b_ckpts_768_vs_1024_2026-02-26.md](/data/home/xiaoyan/AIteam/data/CoordExp/progress/stage1_coco80_eval_4b_ckpts_768_vs_1024_2026-02-26.md)

The question: why do these checkpoints differ so much on COCO80 detection metrics, even though many training scalars look broadly similar?

## 1) Data source (what we can and cannot audit)

For the two older stage-1 runs (`soft_ce_only`, `pure_ce_only`), the full training run directories under `output/stage1/...` are not present (only merged checkpoints remain). Therefore:

- We **can** audit **TensorBoard scalar logs** under `tb/`.
- We **cannot** audit the exact `resolved_config.json` / `config_source.yaml` used at that time from the run directories.

For the mixed run (`soft_ce_hard_ce_mixed`), we do have a full training run directory under `output/stage1_2b/...` including `resolved_config.json` and `logging.jsonl`, but for fair cross-run comparisons below we stick to **TB scalars**, since that exists for all three.

### TensorBoard event files used

- `soft_ce_only`:
  - `tb/stage1/coco_bbox_max60-coco80-desc_first/events.out.tfevents.1771479835.lupeian-2-15-1771140724-default0-0.1115982.0`
- `pure_ce_only`:
  - `tb/stage1/coco_bbox_max60-coco80-desc_first-pure_ce/events.out.tfevents.1771838172.lupeian-2-15-1771140724-default0-0.3763771.0`
- `soft_ce_hard_ce_mixed`:
  - `tb/stage1_2b/coco_bbox_max60-hard_ce_softce_w1_gate/events.out.tfevents.1772025838.lupeian-2-15-1771140724-default0-0.134679.0`

## 2) “End of training” scalar summary (teacher-forcing)

Notes:
- Values below are the **last** `eval/*` scalar logged in each event file.
- `eval_coord_*` uses `eval/coord_softce_w1/*` when available; otherwise it falls back to `eval/coord_diag/*`.
- Loss scales are not directly comparable across objectives; prefer the coord diagnostics and accuracy-like metrics.
- Caution: `eval/text_token_acc` is *not* reliably comparable across these three logs. In this codebase, `text_token_acc` excludes coord tokens only when the batch has token-types or when the trainer can infer a coord mask; pure-CE runs may lack those hooks, causing `text_token_acc` to include coord tokens and look artificially low.

| model | max_train_step | max_eval_step | eval_coord_ce | eval_coord_soft_ce | eval_coord_w1 | eval_coord_gate | eval_p_gt_mean | eval_acc_top5 | eval_expected_bin_mae | eval_coord_vocab_mass | eval_margin_mean | eval_flip_coord_to_noncoord |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| soft_ce_only | 1830 | 1832 | 0 | 3.5260 | 0.0803 | 0.0026 | 0.0655 | 0.4085 | 20.8764 | 0.9995 | 0.8338 | 0 |
| pure_ce_only | 1930 | 1932 | NA | 4.8058 | 0.0881 | 0.0066 | 0.1234 | 0.5448 | 20.9029 | 0.9988 | 1.0985 | 0 |
| soft_ce_hard_ce_mixed | 1340 | 1344 | 3.4371 | 3.6293 | 0.0296 | 0.0029 | 0.0930 | 0.4782 | 21.3715 | 0.9987 | 1.1014 | 8.4831e-05 |

## 3) Key training dynamics (what actually differs)

### 3.1 Coord loss composition differs materially

The cleanest “signature” difference visible in TB:

- `soft_ce_only`: `eval/coord_softce_w1/ce = 0`
  - Interpretable as “no hard-CE term at coord positions” (soft targets + W1 + gate only).
- `soft_ce_hard_ce_mixed`: `eval/coord_softce_w1/ce ~= 3.44`
  - Interpretable as “hard-CE term at coord positions is ON, in addition to softCE/W1/gate”.
- `pure_ce_only`: no `coord_softce_w1/*` block (it still logs `coord_diag/*`).

This aligns with the intuition:
- “soft-only” is good at shaping distributions but may be weaker for **discrete bin selection under autoregressive sampling**.
- “pure-only” is strong at discrete selection (`p_gt_mean`, `acc_top5`) but does not keep the distribution aligned with a soft target (see next section).
- “mixed” gets both: an anchor (`ce`) plus distribution shaping (`soft_ce`, `w1`) plus leakage control (`gate`).

### 3.2 pure_ce_only’s soft-target diagnostic improves early, then worsens later

Looking at `eval/coord_diag/soft_ce` for **pure_ce_only** (sampled at common eval checkpoints):

- step ~40: `8.19`
- step ~200: `4.21`
- step ~400: `4.19`
- step ~800: `4.56`
- step ~1200: `4.72`
- step ~1600: `4.81`
- final (1932): `4.81`

This is qualitatively different from:

- `soft_ce_only`: `eval/coord_softce_w1/soft_ce` decreases and then stabilizes around `~3.53`.
- `soft_ce_hard_ce_mixed`: `eval/coord_softce_w1/soft_ce` decreases and stabilizes around `~3.63`.

Interpretation:
- Under pure CE, the model becomes increasingly “delta-like” on coord bins as training proceeds, which can *increase* the soft-target CE diagnostic after an initial improvement.
- That effect can matter at inference time because our decoding is discrete and our post-op scoring uses confidence, so calibration/shape effects can bite even if token-level accuracy is fine.

### 3.3 All three learn “coord vocab mass confinement” very quickly

`eval/coord_diag/coord_vocab_mass` rises from very low values at step ~40 to ~`0.99+` by step ~200 for all runs.

- soft_ce_only starts at `0.34` (step 40) and reaches `0.9985` by step 200.
- pure_ce_only starts lowest at `0.0565` (step 40) and reaches `0.9916` by step 200.
- mixed starts at `0.192` (step 40) and reaches `0.9959` by step 200.

So “coord-vocab leakage” is probably *not* the main explanation of the large COCO mAP delta at the end.

### 3.4 Expected-bin MAE converges to ~21 bins for all three (teacher forcing)

`eval/coord_diag/expected_bin_mae` ends up extremely similar:

- soft_ce_only: `20.88`
- pure_ce_only: `20.90`
- mixed: `21.37`

This is the strongest evidence that the huge detection gap is **not** well-predicted by standard teacher-forcing coord diagnostics alone.

## 4) Why mAP can still differ a lot if teacher-forcing metrics look close

COCO detection evaluation is based on **free autoregressive generation** (rollouts), not teacher forcing.

Small differences in per-token correctness can compound into large differences in:

- “How long the model keeps generating objects before terminating”
- “How many objects it emits per image” (recall)
- “Whether it stays on the expected output schema under sampling”

This is consistent with the evaluation artifacts already observed:

- Mixed produced materially more predicted objects (`pred_total`) than pure/soft in the 200-sample bench, and longer token traces in dumps.

In other words: **training-time token metrics are necessary but not sufficient**; the gap is likely coming from rollout stability and recall behavior.

## 5) Practical hypotheses (testable)

If we want to reduce surprise and make the behavior predictable, the most actionable hypotheses are:

1. **Adding coord hard-CE to softCE is the main win**
   - Mixed has `coord_softce_w1/ce` ON; soft_ce_only has it OFF.
2. **pure_ce_only might be “best earlier” than the final step**
   - Because `coord_diag/soft_ce` improves early then degrades; an earlier checkpoint might decode better.
3. **Teacher-forcing coord diagnostics are not capturing generation-side failure modes**
   - Consider adding a lightweight periodic *decode-based* validation metric (even on a tiny slice) for stage-1.
