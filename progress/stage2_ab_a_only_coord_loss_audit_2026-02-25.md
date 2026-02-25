# Stage-2 AB (Channel-A only) — Coord Loss Dynamics + “A1 coord loss?” Decision Note (2026-02-25)

This note analyzes the **Stage-2 AB “Channel‑A only”** experiment to diagnose **coord-token loss behavior** and to inform the design decision of whether we should introduce **any explicit coord loss on A1** (the **first** GT/teacher‑forced forward pass).

Primary artifact (this note is scoped to this run only):
- Run dir: `output/stage2_ab/coco_bbox_max60/a_only/epoch_2-eff_size_64-n_softctx_iter_2-/v0-20260224-134747`
- TB event file: `tb/stage2_ab/coco_bbox_max60/a_only/events.out.tfevents.1771940909.k8s-worker02.1247404.0`
- Monitor dumps: `output/stage2_ab/coco_bbox_max60/a_only/epoch_2-eff_size_64-n_softctx_iter_2-/v0-20260224-134747/monitor_dumps/step_*.json`
- Training config: `configs/stage2_two_channel/prod/desc_first_a_only.yaml`

Repo provenance (from `run_metadata.json`):
- git: `main` @ `4d2c6d3142ec043013fa7302e286950774ce10df` (clean)
- seed: `17`
- trainer: `custom.trainer_variant=stage2_two_channel`

---

## 0) Contract recap: where A1 vs self‑context losses live (spec + code)

OpenSpec is authoritative. For Stage‑2 AB Channel‑A:
- **Token CE anchor at A1**: CE on **non‑coord** tokens uses logits from the **first forward** `z^(0)`; coord tokens are masked out.
- **Geometry + any coord distribution losses** are computed from the **final softctx iteration logits** `z^(n_softctx_iter-1)` (not A1).

Spec reference:
- `openspec/specs/stage2-ab-training/spec.md` (see the “Hybrid objective preserves…” requirement, esp. “Token CE anchor at A1” + “Geometry + distribution regularizers from final softctx logits”).

Implementation pointers (Stage‑2 AB “two channel” trainer):
- Channel‑A `n_softctx_iter` loop + A1 logit capture: `src/trainers/stage2_two_channel.py` (`Stage2ABTrainingTrainer.compute_loss`, softctx loop)
- CE masking skips coord tokens entirely: `src/trainers/teacher_forcing/modules/token_ce.py` (`run_token_ce_module`, `if tok_id in coord_id_set: continue`)

Implication for the design question:
- “Add coord loss in A1” is **not a config tweak** in the current contract; it is a **contract / implementation change** (OpenSpec‑governed).
- What *is* a config knob today:
  - `stage2_ab.coord_ce_weight` → enables **hard coord-token CE** on the supervised coord slots, computed from **final** softctx logits (not A1).
  - `custom.coord_soft_ce_w1.soft_ce_weight` / `w1_weight` → controls the **coord_dist** regularizer used in the `coord_reg` term (again, from **final** logits).

---

## 1) Run configuration summary (what this experiment is actually doing)

Key knobs from `configs/stage2_two_channel/prod/desc_first_a_only.yaml` (and `resolved_config.json`):
- Channel routing: `stage2_ab.schedule.b_ratio: 0.0` (Channel‑A only)
- Soft self‑context: `stage2_ab.n_softctx_iter: 2`
- Grad semantics: `stage2_ab.softctx_grad_mode: unroll`
- Coord slot embedding mode: `stage2_ab.coord_ctx_embed_mode: st` (ST embedding)
- Geometry decode: `stage2_ab.coord_decode_mode: exp` (CoordExp expectation decode)
- Bbox loss weights: `bbox_smoothl1_weight: 2.0`, `bbox_ciou_weight: 0.2` (CIoU on)
- Coord distribution regularizer (Stage‑1‑style):
  - `custom.coord_soft_ce_w1.soft_ce_weight: 0.1`
  - `custom.coord_soft_ce_w1.w1_weight: 0.1`
  - `custom.coord_soft_ce_w1.gate_weight: 1.0`
  - `custom.coord_soft_ce_w1.target_sigma: 2.0`, `target_truncate: 8`, `temperature: 1.0`
- Hard coord-token CE: **off** (`stage2_ab.coord_ce_weight: 0.0`, `loss/coord_token_ce` logs as 0)

Eval cadence:
- `training.eval_steps: 300`
- monitor dump every 300 steps (enabled)

Run length note:
- This artifact contains monitor dumps up to step **1500** and train logs through ~**1600/3662** steps, so this is an **early‑to‑mid training** snapshot (not a completed 2‑epoch run).

---

## 2) What the “coord token loss” scalars mean in this run

This config logs several coord‑related scalars; the important distinction:
- `train/loss/coord_token_ce`: **hard** CE over the 1000 coord bins (disabled here → 0.0).
- `train/loss/coord_soft_ce`: soft‑label CE over coord bins (Gaussian target, sigma=2, truncate=8), computed per supervised coord slot.
- `train/loss/coord_w1`: W1‑like ordinal regularizer over bins, per supervised coord slot.
- `train/loss/coord_gate`: coord‑vocab gate penalty (discourages probability mass leaking to non‑coord vocab).
- `train/loss/coord_reg`: weighted sum of the above “coord_dist” terms (per spec it is computed from **final** softctx logits).
- `train/coord_diag/*`: diagnostics from the same supervised coord logits (also from **final** softctx logits):
  - `p_gt_mean`: mean probability assigned to the GT bin
  - `acc_top5`: whether the GT bin is in top‑5
  - `expected_bin_mae`: MAE of the expected bin index `E[k]` vs GT (bin space)
  - `coord_vocab_mass_mean`: fraction of probability mass inside the coord‑token slice

Practical implication:
- If you saw a “coord loss” curve in TB moving, it’s **almost certainly** `coord_soft_ce`/`coord_w1`/`coord_reg` (since `coord_token_ce` is identically 0 in this run).

---

## 3) Training dynamics: what actually improved vs what stayed flat

### 3.1 Coord distribution quality (train‑side)

From `logging.jsonl` + TB scalars:
- `train/coord_diag/expected_bin_mae` improved meaningfully over the run window (and at eval checkpoints):
  - mean(near step 300): **~42.49 bins**
  - mean(near step 1500): **~36.66 bins**
- `train/coord_diag/expected_bin_abs_err_p90` also improved:
  - mean(near step 300): **~121.28 bins**
  - mean(near step 1500): **~100.31 bins**

Meanwhile, peakedness proxies did **not** improve much:
- `train/coord_diag/acc_top5`: ~0.33 → ~0.35 (small increase)
- `train/coord_diag/p_gt_mean`: ~0.048 → ~0.050 (basically flat at eval checkpoints)
- `train/coord_diag/margin_mean` trends downward overall in the raw train trace (consistent with a non‑delta optimum under soft targets).

Interpretation:
- The **expected coordinate** is getting closer to GT (good for CoordExp geometry loss),
- but the **discrete** distribution is not getting dramatically sharper (not necessarily bad, but relevant if we care about greedy discrete coord tokens).

### 3.2 Why the coord loss curve “fluctuates”

Empirically, `loss/coord_soft_ce` is moderately correlated with how many coord tokens are present in the batch:
- Pearson corr(`coord_diag/coord_tokens`, `loss/coord_soft_ce`) ≈ **0.52** in this run.

This is consistent with “batch difficulty” effects under packing:
- more objects → more coord slots → harder distribution shaping → higher soft CE/W1 in that batch
- so the curve is expected to be noisy even if the underlying model is improving.

### 3.3 Gate behavior (type leakage control)

Good news for parseability:
- `coord_diag/coord_vocab_mass_mean` stays extremely high and trends up slightly (~0.9983 → ~0.9998).
- `loss/coord_gate` drops by ~O(10×), indicating the gate penalty is becoming easier to satisfy.

This is consistent with the earlier diagnosis that coord‑vocab “type leakage” can be a major format/parseability failure mode (see `progress/stage2_ab_channel_a_only_coord_gate_diagnosis_2026-02-21.md`).

---

## 4) Eval dynamics: f1 vs mAP (and what divergence would mean)

### 4.1 What we actually see in *this* artifact

At eval steps (from TB scalars / `monitor_dumps/step_*.json`):

| step | mAP | f1 | precision | recall | pred_objects | matched | fp_total | fn_total |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 300 | 0.3425 | 0.6209 | 0.5973 | 0.6464 | 3958 | 2364 | 1594 | 1293 |
| 600 | 0.3376 | 0.6306 | 0.6172 | 0.6445 | 3819 | 2357 | 1462 | 1300 |
| 900 | 0.3429 | 0.6316 | 0.6138 | 0.6505 | 3876 | 2379 | 1497 | 1278 |
| 1200 | 0.3347 | 0.6432 | 0.6416 | 0.6448 | 3675 | 2358 | 1317 | 1299 |
| 1500 | 0.3351 | 0.6460 | 0.6376 | 0.6546 | 3755 | 2394 | 1361 | 1263 |

So for this run segment:
- `eval/rollout/f1` **increases** steadily.
- `eval/rollout/mAP` is **roughly flat** (slight down‑drift after step 900, but within small absolute range).

This does **not** match the “f1 down while mAP up” pattern described in the prompt; if you saw that in TB, it may have been:
- a different run (`a_only_em_detach` history), and/or
- smoothing artifacts, and/or
- different metric tags (e.g., a separate f1-ish thresholded metric).

### 4.2 If f1 decreases while mAP increases: likely mechanisms + how to verify

In general, divergence is plausible because the metrics reward different behaviors:
- `mAP` (COCO) rewards **localization quality across IoU thresholds** and is sensitive to high‑IoU improvements.
- `f1` here is a **matching‑based** summary that is sensitive to **object count calibration** (FP/FN trade) and gating behavior.

Common patterns that yield “f1 ↓, mAP ↑”:
1) **Sharper boxes but worse count calibration**: localization improves (mAP ↑) while over/under‑predicting objects increases FP or FN (f1 ↓).
2) **Gating/parse filtering changes**: stricter gating rejects more predictions (precision may rise, recall falls → f1 ↓) while the remaining boxes are higher quality (mAP ↑).
3) **Semantics vs geometry decouple**: if category/desc mapping improves localization AP but matching rejects on semantic thresholds, f1 can drop even if mAP rises.

How to check quickly (actionable):
- Compare `eval/rollout/pred_objects`, `fp_total`, `fn_total` vs step.
- Inspect `eval/rollout/gating_rejections` and parse drop counters (`parse_dropped_invalid`, `parse_dropped_ambiguous`).
- Track a localization proxy like `eval/rollout/matched_maskiou_mean` alongside mAP.

### 4.3 Monitor dumps: qualitative rollout patterns (greedy decode)

We inspected the 10‑sample monitor dumps at steps **300/600/900/1200/1500** (`monitor_dumps/step_*.json`).

High‑level: **no format corruption** in the monitored samples.
- `parse_invalid_rollout=false` for all monitored samples in all 5 dumps.
- `parse_dropped_invalid=0`, `parse_dropped_ambiguous=0`, `parse_truncated=false` across dumps.

Monitor subset summary (10 samples each; *not* the full eval set):

| step | mean_gt_objects | mean_pred_objects | mean_matched | mean_f1 | invalid/trunc |
|---:|---:|---:|---:|---:|---:|
| 300 | 10.0 | 11.2 | 5.4 | 0.739 | 0 / 0 |
| 600 | 10.0 | 11.6 | 5.4 | 0.717 | 0 / 0 |
| 900 | 10.0 | 13.3 | 5.4 | 0.700 | 0 / 0 |
| 1200 | 10.0 | 10.9 | 5.3 | 0.709 | 0 / 0 |
| 1500 | 10.0 | 10.2 | 5.1 | 0.716 | 0 / 0 |

The dominant observed failure pattern is **duplicate enumeration** (many near‑duplicate boxes for the same class), especially on:

1) **Base idx = 2 (bookshelf)** — persistent “book spam” across dumps
   - GT objects: 17 total (**13× book**, 2× potted plant, chair, bed).
   - Predictions: **25–43 objects**, with **20–38 books**, but only **4–5 matched** objects total.
   - Example (step 900):
     - `gt=17`, `pred=43`, `matched=5`, `precision≈0.116`, `recall≈0.294`, `f1≈0.167`
     - class histogram: `book=38`, `potted plant=2`, plus a few others.
     - the predicted **non-book** large objects match well (bed/chair/plants with bbox IoU ~0.75–0.96),
       but the book boxes are often **coarse** (covering big shelf regions) and/or **near‑duplicates**.
     - near‑duplicate severity: among the 38 predicted `book` boxes at step 900, there are **35 pairs with IoU ≥ 0.9**
       (8 pairs with IoU ≥ 0.95), indicating repeated overlapping detections.
   - Ordering: even in this failure case, the sequence is almost perfectly sorted by the requested `(y1, x1)` anchor
     (only 1 ordering violation over 43 predicted objects), so the main issue is **dedup + fine localization**, not ordering.

2) **Crowded “person” scenes** — many GT persons; model under-matches and duplicates
   - Example (step 900, base idx 354):
     - `gt=29`, `pred=33`, `matched=11`, `f1≈0.355`
     - histogram dominated by `person` (23 persons predicted).

Interpretation:
- Channel‑A only training is currently **stable/parseable** (gate works; no wrong‑arity failures in monitored samples).
- The remaining rollout weaknesses look like **set-level behavior**:
  - duplicate boxes (needs dedup / stronger stop/continue control), and
  - small/thin object localization (books) where discrete box IoU gating is harsh.
- These patterns do **not** point to “coord‑token CE in A1” as the first lever; they point to:
  - stronger discrete coord‑token sharpness (small hard CE on final logits, or stronger softCE), and/or
  - more set‑alignment help (Channel‑B, or decode‑time/post‑processing constraints if acceptable).

---

## 5) Answering the research questions (evidence‑backed)

### RQ1 (Primary): Should we introduce *any* coord loss in A1?

**Recommendation (for the current `unroll`, `n_softctx_iter=2` regime): do *not* introduce a new A1‑only coord loss term yet.**

Why (evidence + contract alignment):
- The **current Stage‑2 AB contract** (OpenSpec) explicitly anchors CE at A1 (non‑coord only) and computes geo + coord_dist losses from the **final** logits.
- In this run, coord_dist + geo supervision already yields a clear improvement in **expected‑bin error** and stable eval metrics, without requiring A1‑coord CE.
- Because `softctx_grad_mode=unroll`, gradients from the final iteration flow back through the coord‑slot context update (A2), so A1 coord logits are not “completely unsupervised” in practice.

What I would change *before* touching A1:
1) If the goal is **sharper discrete coord tokens**, first try a small increase in the existing **coord_dist** pressure (still on final logits):
   - bump `custom.coord_soft_ce_w1.soft_ce_weight` (e.g., `0.1 → 0.2`) while holding `w1_weight` fixed.
2) If you specifically want **delta‑like** coord bins, prefer a *small* `stage2_ab.coord_ce_weight` trial (e.g., 0.01–0.05) **on final logits**, with close monitoring of:
   - `coord_diag/p_gt_mean`, `coord_diag/acc_top5`
   - eval parseability counters + `eval/rollout/mAP`

What would justify revisiting “A1 coord loss” as a spec change:
- Evidence that the **A1 coord distributions** (iteration 0) drift or become unreliable in a way that harms the self‑context update (A2).
  - Today we do **not log A1‑specific coord diagnostics** separately, so we can’t confirm this from the current artifact.
  - A minimal next logging addition (if needed) would be to log `coord_diag/*` for `logits_a1` vs final logits as separate tag namespaces (diagnostics‑only change).

### RQ2 (Secondary): Is A2 coord supervision still “drifted” / suboptimal?

On this artifact, there is **no strong evidence** that the A2 self‑context update is drifting in a way that hurts evaluation:
- train expected‑bin errors improve monotonically at eval checkpoints,
- coord vocab mass is extremely high and improving,
- eval strict parse drops are 0 in the sampled monitor dump (at least for the 5 checkpoints),
- f1 improves over time.

What *is* potentially suboptimal (and consistent with the soft‑target optimum):
- peakedness proxies (`p_gt_mean`, `margin_mean`) don’t improve much, suggesting distributions may remain broad.
- if we care about **greedy discrete coord token accuracy**, this is where we may want sharper dist shaping (via `soft_ce_weight`, temperature, or (carefully) a small hard CE term).

### RQ3: Why does f1 decrease while mAP increases? What does it indicate?

For this run segment: it’s **not happening** (f1 ↑, mAP ~flat).

In general, if you observe “f1 ↓, mAP ↑”, it often indicates:
- the model is improving **localization precision** (helping mAP at higher IoU thresholds),
- while simultaneously becoming worse at **count/coverage calibration** (more FP or FN), hurting f1.

That is usually a *behavioral / decoding* issue more than a pure coordinate regression issue:
- check for over‑long object lists (runaway continuation),
- check gating thresholds and parse rejection rates,
- check whether the model is emitting many near‑duplicate boxes (duplicates hurt f1 strongly).

---

## 6) Concrete next actions (config‑first)

If the goal is to decide “coord loss in A1?” with minimal churn, I’d run **two short A/Bs** (same seed, same max_steps, same eval cadence):

1) **Increase soft coord_dist** (sharper distribution supervision, still ordinal‑aware):
   - `custom.coord_soft_ce_w1.soft_ce_weight: 0.2` (from 0.1)
   - keep `w1_weight: 0.1`
   - watch `coord_diag/expected_bin_mae` vs `coord_diag/acc_top5` and eval `mAP`/`f1`

2) **Enable small hard coord CE (final logits)** (test if discrete accuracy matters):
   - `stage2_ab.coord_ce_weight: 0.02` (start small)
   - keep soft CE as-is (0.1) initially
   - watch whether `p_gt_mean`/`acc_top5` improve without causing parseability regressions

Verification checklist for each A/B:
- Eval: `eval/rollout/mAP`, `eval/rollout/f1`, `eval/rollout/parse_dropped_*`, `eval/rollout/coco_counter_degenerate`
- Train: `coord_diag/expected_bin_mae`, `coord_diag/p_gt_mean`, `coord_diag/acc_top5`, `coord_diag/coord_vocab_mass_mean`, `loss/coord_gate`
- Qualitative: skim `monitor_dumps/step_*.json` samples for bbox array integrity (`wrong_arity`) + runaway repetition.

If either A/B suggests that improvements require anchoring A1 directly, that’s the point where an **OpenSpec change proposal** (“optional A1 coord_dist anchor”) becomes justified.

---

## References (repo paths)

- Run artifacts:
  - `output/stage2_ab/coco_bbox_max60/a_only/epoch_2-eff_size_64-n_softctx_iter_2-/v0-20260224-134747/logging.jsonl`
  - `output/stage2_ab/coco_bbox_max60/a_only/epoch_2-eff_size_64-n_softctx_iter_2-/v0-20260224-134747/monitor_dumps/`
  - `tb/stage2_ab/coco_bbox_max60/a_only/`
- Config:
  - `configs/stage2_two_channel/prod/desc_first_a_only.yaml`
- Design/contract:
  - `progress/full_idea.md` (Sections 0.2–0.4, 6.2, 7.1)
  - `openspec/specs/stage2-ab-training/spec.md`
- Implementation:
  - `src/trainers/stage2_two_channel.py` (Channel‑A softctx loop, coord losses, logging)
  - `src/trainers/teacher_forcing/modules/token_ce.py` (coord tokens masked out of CE)
  - `src/trainers/teacher_forcing/objective_pipeline.py` (module weighting + logging)
  - Prior related notes:
    - `progress/stage2_ab_channel_a_only_coord_gate_diagnosis_2026-02-21.md`
    - `progress/stage2_ab_softctx_discretization_vs_stage1_bbox_losses_2026-02-22.md`
