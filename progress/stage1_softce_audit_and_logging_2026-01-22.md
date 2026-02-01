# Stage-1 SoftCE (W1) Audit: Loss Dynamics, Logging Alignment, and Efficiency Notes (2026-01-22)

This note captures the key findings from auditing CoordExp stage-1 pretraining (soft cross-entropy over coord bins, optional W1 regularizer), with emphasis on:
- why earlier train/eval losses looked misaligned,
- what we changed to make metrics comparable,
- what the distributional metrics are telling us,
- where compute is actually going (softCE vs hardCE vs gate), and
- what to try next (config-first).

## 1) What stage-1 is optimizing (and what to expect)

Stage-1 in this stack is primarily a *coord-token distribution shaping* phase:
- Instead of treating a coordinate token as a single correct class (hard CE), we supervise a *distribution* over the 0..999 coord bins (soft CE), typically a truncated Gaussian centered at the ground-truth bin.
- Optionally add a Wasserstein-1 (W1) style term to encourage the predicted distribution to place probability mass close to the target mean (distribution-level geometry).
- A separate "gate" term (logsumexp-based) is used to discourage probability leaking from the coord-vocab slice to the rest of the tokenizer vocab.

Expected patterns:
- `coord_token_acc` / `eval_coord_token_acc` (top-1) can decrease while distributional losses still improve, because the optimum is not necessarily a delta peak.
- Train vs eval loss *should* be in the same ballpark if computed on the same token subsets and scaled consistently.

## 2) Root cause: why earlier train loss was ~7-8x eval loss

Observed: in older runs, `train/loss` was about `gradient_accumulation_steps` times larger than `eval_loss`.

Cause:
- In the transformers Trainer path used by ms-swift + recent transformers, when `model_accepts_loss_kwargs=True` and `num_items_in_batch` is provided, Trainer intentionally *skips* dividing by `gradient_accumulation_steps`.
- That makes the backprop magnitude fine, but the logged `loss` becomes scaled by the current grad-accum steps.

Why this felt confusing vs `../Qwen3-VL`:
- Different trainer versions / codepaths can diverge here. Some setups divide by grad-accum for logging, others do not depending on the `num_items_in_batch` path and `model_accepts_loss_kwargs` behavior.
- Net effect: you can have correct training but misleading logs.

## 3) Fix: make train loss comparable to eval loss

We added a small mixin that rescales the *train* loss by the current gradient accumulation steps in the same gating scenario where HF would otherwise skip it.

Outcome:
- Train/eval loss ratio at eval steps becomes ~1.0 (in the v2 run it was ~0.94), which is the expected magnitude alignment.

Practical impact:
- Makes it possible to compare `loss` / `eval_loss` directly.
- Enables fair comparisons across different `gradient_accumulation_steps` settings.

## 4) Train/eval metric alignment improvements

To help fair comparisons we added/logged consistent components in both train and eval:
- `base_ce/loss` and `base_ce/noncoord_tokens` to make clear what part of the supervision is "standard CE".
- runtime scalars in eval too (e.g., learning rate and accumulation info), so you can line up train/eval records without guessing.

We also added config knobs to reduce metric clutter and cost:
- `custom.token_type_metrics.log_top5`
- `custom.token_type_metrics.coord_monitor_mass`
- `custom.token_type_metrics.coord_monitor_mass_max_tokens` (caps number of coord tokens used for mass diagnostics).

## 5) Distributional metrics: what to watch

Given top-1 coord accuracy can degrade, focus on distributional signals:
- `eval_coord_softce_w1/soft_ce` should decrease (primary learning signal for soft targets).
- `eval_coord_softce_w1/w1` should usually decrease slowly; if it is flat or tiny, W1 may be redundant.
- `eval_coord_softce_w1/p_gt_mean` is the average probability assigned to the *ground-truth bin*. It can go down if the model spreads mass more widely (which can be OK), but if it collapses while soft_ce does not improve, that suggests diffusion/under-confidence.
- `eval_coord_softce_w1/coord_vocab_mass` measures how much probability stays inside the coord-vocab slice. If it trends down, leakage is increasing.
- `eval_coord_softce_w1/gate` tracks the penalty discouraging leakage.

In the v2 run:
- soft_ce showed a strong decreasing trend (good).
- gate increased and coord_vocab_mass decreased modestly (leakage trend worth monitoring).
- p_gt_mean and acc_top5 decreased; not necessarily a problem if the distribution remains unimodal and centered, but it can indicate flattening.

## 6) Efficiency: softCE vs hardCE vs gate

We microbenchmarked loss-side overhead (A100, loss-only, K=1000 coord bins):
- hard CE: ~0.04 ms
- softCE+W1: ~0.79 ms
- gate term (full vocab V ~ 150k logsumexp-style): ~6.0 ms

Key takeaway:
- If you are worried about loss compute overhead, the "gate" is the dominant cost, not softCE itself.
- End-to-end training is still likely dominated by the model forward/backward over long packed sequences, but if you are chasing perf, gate is the first knob to profile.

## 7) What `target_truncate` does (and what it does NOT do)

`target_truncate` truncates the *target distribution shape* around the ground-truth mean (e.g., only bins within +/- truncate get non-negligible target probability). It affects supervision sharpness.

Important:
- In the current implementation, changing `target_truncate` does not materially reduce compute, because the target is still constructed over all 1000 bins (it is mainly masking/renormalization).
- So truncate is mostly a learning-shape knob, not a speed knob.

## 8) SoftCE+W1 vs hardCE: when is W1 helpful?

Hypothesis (to validate):
- W1 may be redundant if the soft target already encodes geometric proximity (Gaussian over bins) and the model learns unimodal distributions.
- If `w1_weight` is non-zero but `w1` loss barely changes or does not correlate with improved decoding metrics, it may be safe to drop W1.

Config-first A/B proposed:
- baseline: `configs/dlora/sft_stage1_softce_w1.yaml`
- no W1: `configs/dlora/sft_stage1_softce_w1_no_w1.yaml` (w1_weight=0)

## 9) Temperature / sigma / truncate tuning (and size-adaptive intuition)

General:
- Lower temperature (<1) sharpens predicted distributions (can increase p_gt_mean, improve localization, but risks overconfidence / poorer calibration).
- Smaller target_sigma sharpens the target distribution (stronger pressure to concentrate near GT).
- target_truncate controls how far from GT bins get non-zero-ish target mass.

Your proposed experiment:
- temperature=0.9, target_sigma=1.5 is a reasonable sharper-target trial.
- For truncate: 12 is not obviously necessary; 8 often already covers most Gaussian mass when sigma is ~1.5-2.0.

Object-size-adaptive noise allowance:
- The intuition is sound: small objects need tighter coordinate tolerance, large objects can tolerate broader distributions.
- Implementing size-adaptive sigma is a *new capability* (needs object size metadata at the token level and consistent mapping from object -> coord tokens). If pursued, this should go through OpenSpec change governance.

## 10) Decoding perspective: expectation/median vs argmax

Expectation decode can outperform argmax when distributions are:
- unimodal,
- approximately symmetric,
- well-centered.

Median decode (or top-k mean) is more robust to tails and mild multimodality.

Training implication:
- If you plan to decode by expectation/median later, the goal is to avoid strong multimodality in the coord token distributions.
- That suggests monitoring distribution diagnostics and potentially adding (future) unimodality/entropy/variance controls, but that would be a capability change.

## 11) Next steps (practical, config-first)

1) Short A/B runs (same seed, same max_steps) to isolate knobs:
- baseline softCE+W1
- temperature=0.9 / sigma=1.5 / truncate=8: `configs/dlora/sft_stage1_softce_w1_t0p9_s1p5_tr8.yaml`
- no W1
- optional no-gate (risky): `configs/dlora/sft_stage1_softce_w1_no_gate.yaml`

2) Compare speed:
- tokens/sec or steps/sec (trainer logs) + GPU util

3) Compare distributional quality:
- soft_ce, p_gt_mean, margin, coord_vocab_mass, gate

4) Downstream validation:
- run existing detection/grounding evaluators on checkpoints from each variant (ensure checkpoints are saved).

---

## References (repo paths)
- Stage-1 run logs: `output/1-20/stage1_softce_w1/.../logging.jsonl`
- Stage-1 configs: `configs/dlora/sft_stage1_softce_w1*.yaml`
- Trainer entry: `src/sft.py`
- Metrics/loss mixins: `src/metrics/dataset_metrics.py`
- Metrics doc: `docs/training/METRICS_LOSSES.md`
- Stage-1 idea doc: `progress/pretrain/first_stage.md`
