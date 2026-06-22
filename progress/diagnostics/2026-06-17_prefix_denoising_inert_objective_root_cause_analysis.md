# Prefix-Denoising SFT: Root-Cause Analysis of bbox Degradation

Date: 2026-06-17
Scope: worktree `/data/CoordExp/.worktrees/geometry-aware-denoising-sft`, branch `codex/prefix-denoising-sft`
Method: code read (builder/loss/forward) + training-log diagnostics + decode-config + cross-run AP comparison. No new training launched.
Companion: `progress/diagnostics/2026-06-16_prefix_denoising_axis_sort_repair_negative_result.md` (axis-sort repair negative result).

---

## Diagnosis (headline)

The bbox degradation is **not** caused by the noise/clean pair objective corrupting coordinates. Two stronger facts dominate:

1. **The geometry-aware denoising objective is inert.** Throughout training, the noisy-branch CE equals the clean-branch CE to ~4 decimals, and the local-window KL is ~0 from step 0. The model's coordinate prediction is essentially **insensitive to the preceding coordinate tokens**, so feeding a noised coordinate prefix changes nothing. The run reduces to a plain hard-CE LoRA SFT that pays 2× forward cost for no signal.

2. **The "degradation" is measured against a non-comparable baseline.** The 0.35+/0.42 reference is a **4-epoch (ckpt-1332), SoftCE-coordinate, full-merged** model; the prefix-denoising run is a **2-epoch (ckpt-450), LoRA-r16, hard-CE** model. The gap is overwhelmingly a *recipe* gap (SoftCE vs hard-CE, 3× steps, full-FT vs LoRA), not a *denoising harm*. No matched control (hard-CE LoRA 2-epoch, denoising OFF) exists.

So the experiment did not test what it intended to test. The premise — that the noisy bbox prefix teaches geometry-aware self-correction — never engaged the model, and the negative result is confounded by baseline mismatch. This is therefore **not** evidence against geometry-aware denoising as a concept; it is evidence the implementation's core assumption is false for this architecture/format.

---

## Evidence

### A. The noisy branch carries no information (objective is inert)

Training log: `outputs/stage1_2b/.../v25-20260615-124531/logging.jsonl`

- Per-step `clean_full/loss/ce` vs `noisy_full/loss/ce` are identical to ~4 decimals across all 450 steps. Examples: 12.5215 vs 12.5206 (start) … 2.7062 vs 2.7066 (end). The noisy-vs-clean delta is ~4e-4 nats (~0.01% of the loss).
- Local-window KL `raw` ≈ 5e-4, `weighted` ≈ 0.0000 throughout (weight 0.05). The student (noisy) distribution already matches the teacher (clean) distribution from step 0.
- Last eval, per coord slot, teacher(clean) ≡ student(noisy) GT-conditional prob:
  - x1 0.0896 vs 0.0897 · y1 0.0678 vs 0.0679 · x2 0.1455 vs 0.1453 · y2 0.1108 vs 0.1103.

Interpretation: ±8%-magnitude perturbation of the previously emitted coordinate tokens does **not** move the next-coordinate distribution. The model grounds each coordinate in the **image/structure**, not in the emitted coordinate prefix. The thing the method tries to regularize (autoregressive coordinate self-conditioning) is largely **absent** here, so the noisy branch is a near-duplicate of the clean branch and the KL has nothing to pull.

### B. Teacher-forced coordinate distributions are intrinsically broad (hard-CE signature)

- `teacher_support_mass` ≈ 0.137: under teacher forcing on **clean** inputs, only ~14% of coordinate probability mass lands within ±8 bins of GT. ~86% of mass is outside a sub-1%-of-image window.
- `teacher_top1_is_gt` ≈ 0.11 (and slightly **decreasing** over training, 0.1235 → 0.1136): the clean-branch argmax coordinate bin exactly matches GT only ~11% of the time.

This is the expected behavior of a plain hard-CE coordinate head (broad, weakly localized), and it is *independent* of denoising.

### C. The architecture of the loss forbids the obvious "cheating" shortcut

`src/trainers/metrics/prefix_denoising.py:_run_isolated_prefix_denoising_forwards` runs clean and noisy as **two separate forward passes in separate batch rows** (clean at `index`, noisy at `sample_count+index`). The noisy branch physically cannot attend to clean coord tokens. So the inertness is genuine model behavior, not a leakage shortcut that collapses the task.

### D. Noise is valid and mild (user's premise confirmed)

`construct_valid_norm1000_bbox_noise` (geometry.py) applies a coherent center-shift (≤8% of side) + uniform scale (±8%), forces all 4 coords to change, and validates the result is a positive-area in-range box. `_validate_clean_noisy_alignment` enforces that clean and noisy differ **only** at the 4 coord positions and never collide. So the noisy bbox is a valid, mildly-perturbed box — exactly as assumed. Validity was never the issue.

### E. The reference baseline is not comparable

| Run (decode rp≈1.10, low temp) | objective | epochs/steps | adapter | bbox_AP | AP50 | AP75 | APl | APm | APs | AR100 |
|---|---|---|---|---|---|---|---|---|---|---|
| `core6_base1332` (reference) | hard_ce + **SoftCE w1 gate** | epoch 4 / **ckpt-1332** | **merged-full** | **0.423** | 0.561 | 0.467 | 0.499 | 0.291 | 0.126 | 0.433 |
| prefix-denoising ckpt450 (raw) | hard CE + **inert** denoise | epoch 2 / ckpt-450 | LoRA r16 DoRA | 0.146 | 0.256 | 0.137 | 0.202 | 0.001 | 0.0 | 0.172 |

Reference: `outputs/output_remote/stage1_2b/coco_bbox_max60-hard_ce_soft_ce_w1_gate/epoch_4-from-base-2B/v0-.../checkpoint-1332-merged-full`.

The reference differs on **four** axes at once (SoftCE coordinate loss, 3× steps, 2× epochs, full-merge vs LoRA). Attributing the gap to denoising is unsupported.

### F. The failure profile is a hard-CE/localization story, not a denoising story

- prefix-denoising AP is **entirely large-object**: APl 0.20 but **APm ≈ 0.001, APs = 0.0**. The model cannot localize small/medium objects at all — the direct consequence of a broad coordinate head (Evidence B). The reference (SoftCE) recovers APm 0.29 / APs 0.13.
- Recall-dominant collapse: recall (f1@0.30 micro) 0.575 → 0.282; AR100 0.433 → 0.17.
- Strict-parse summary (`summary.json`): of 200 images, **empty_pred 64**, invalid_geometry 56, wrong_coord_arity 9. A third of images yielded no parseable box. The axis-sort repair recovered *materialization* (empty_pred 64→1) but barely moved AP (0.118→0.146) — consistent with the companion negative result and with "localization, not parsing, is the ceiling."

### G. Decode-side confound

Diagnostic eval used **`repetition_penalty = 1.1`** under greedy decode (`resolved_config.json`/`summary.json`). Repetition penalty applies across the whole vocab including `<|coord_*|>` tokens, biasing the model away from already-used coordinate bins toward unused/extreme bins (edge saturation) and inflating arity/length pathologies once a sequence drifts off-distribution. The sibling baseline sweep shows decode params alone move AP50 by ~0.18 on a fixed model (rp110_t020 0.561 → rp122_t050 0.381). The denoising eval used the *good* rp setting, so this does not explain the gap vs the reference — but it does suppress the absolute number and likely amplifies the edge-saturation/arity tails seen in the companion doc.

---

## Probe Scope

- Train: COCO `rescale_32_1024_bbox_max60`, sorted compact-full, marker-delimited (`row_separator: none`), LoRA r16 DoRA, freeze ViT+aligner, 2 epochs, eff. batch 128, ckpt-450. KL weight 0.05, window_radius 8, num_objects_per_image 1. Noise center_shift_frac 0.08, scale [0.92,1.08].
- Eval: val200 (limit 200), free greedy decode, temp 0, rp 1.1, max_new_tokens 3084, `marker_delimited_strict` parse (+ axis-sort-repair diagnostics in companion doc).
- Diagnostics source: training `logging.jsonl` (train + 5 eval points), infer `summary.json` / `metrics*.json`, cross-run AP table.
- Not run here: a matched denoising-OFF control; SoftCE-on prefix-denoising; a coord-token-aware decode (rp=1.0 on coord tokens). These are the recommended next steps.

---

## Symptom Taxonomy

- train improves, free-rollout poor → **exposure/off-policy gap** (teacher-forced token acc plateaus ~0.50, coord support reasonable-ish; free decode collapses to empty/edge/degenerate). The denoising method was meant to address exactly this and did not engage.
- recall down, small/medium AP ≈ 0 → **under-localization + under-generation** from a broad hard-CE coordinate head, amplified by rp=1.1 and greedy off-distribution drift.
- "noisy ≡ clean" CE and KL≈0 → **inert objective / mechanism-not-engaged**, not corruption.
- huge AP gap vs reference → **baseline/recipe mismatch (SoftCE + 3× steps + full-FT)**, an eval-validity problem, not a model-degradation problem.

Classification: primarily **objective-not-engaged + baseline mismatch (eval validity)**, layered on a genuine **model limitation** (hard-CE localization) and an **exposure-bias** free-rollout gap. There is no evidence of an implementation bug in the noise/alignment path, optimization instability, or stale/wrong adapter.

---

## Likely Root Cause (ranked)

1. **Wrong premise → inert mechanism.** This compact-full coordinate model predicts each coordinate from visual grounding and structural position, not from the previously emitted coordinate *tokens*. Perturbing those tokens (the noisy branch) is a no-op (Evidence A). The denoising signal — both the noisy CE and the local KL — collapses to redundant clean CE. The method cannot help because there is no autoregressive coordinate dependency to denoise.
2. **Non-comparable baseline (eval validity).** The 0.35+/0.42 target comes from a SoftCE, 4-epoch, full-merged model; the run is hard-CE, 2-epoch, LoRA. The bulk of the measured gap is recipe, not denoising (Evidence E/F).
3. **Underlying limitation the run never fixed: hard-CE coordinate localization is broad** (support_mass 0.137, top1_is_gt 0.11; APm≈0, APs=0). This, not denoising, is what holds AP down (Evidence B/F).
4. **Free-decode exposure bias + rp=1.1** convert a broad coordinate head into empty/edge-saturated/wrong-arity rows (Evidence F/G). Recall-dominant collapse.

Secondary (considered, ruled out or minor):
- *Clean-coord attention leakage shortcut* — ruled out; forwards are isolated (Evidence C).
- *Noisy pairs actively teaching bad transitions / anti-anchoring* — not happening; the model already ignores the prefix, so the gradient delta is ~0.01% (Evidence A). The hypothesis is real in principle but moot here.
- *0.5/0.5 branch weighting halving the clean signal* — since noisy≈clean, the summed gradient ≈ full clean; near-neutral. Costs 2× compute, not accuracy.
- *Local-window KL is blind to the tails* — true and worth fixing if KL is ever revived: only ~14% of coord mass is inside the ±8-bin window, so the KL can never see or suppress edge-saturation mass. But with weight ~0 effective, it is currently irrelevant.
- *Noise infeasibility skips biasing the training set* — the `noise_infeasible_4coord_changed` / `degenerate_gt_bbox` skips drop some tiny/edge boxes from the hybrid set; a possible small-object data-coverage bias worth a counter check, but not the main driver.

---

## Corrective Strategies

Separate "fix the measured symptom" from "make geometry-aware denoising actually work."

### Make the comparison valid (do this first — cheap, decisive)
- **Run the matched control**: hard-CE LoRA r16, 2 epochs, compact-full sorted marker, denoising **fully disabled** (single clean branch), same decode. If its AP ≈ the prefix-denoising AP, the denoising add-on is confirmed neutral and the report's causal claim is retracted. This is the single highest-value next probe.
- Stop comparing to `base1332`. Either re-evaluate `base1332` is irrelevant, or build a like-for-like reference (same objective/steps/adapter) for any future denoising claim.

### Lift the real ceiling (coordinate localization)
- **Re-introduce SoftCE / Gaussian coordinate loss** (the project's known strong localizer; the reference uses `soft_ce_w1_gate`). The inert-denoising run is essentially "hard-CE only," which is exactly the configuration SoftCE was designed to beat. Keep slot-wise readout (x1/y1/x2/y2 separate) per the Stage-1 coordinate-locality guidance.
- **Train longer / larger adapter** to the reference budget (≥4 epochs, or full-FT) before drawing objective conclusions.

### Make denoising engage (if the idea is to be salvaged)
The mechanism failed because the perturbation lives in a channel the model ignores. To create a real denoising gradient, the perturbation must affect a quantity the model conditions on:
- **Self-conditioning / scheduled-sampling on the model's own decode**, not on synthetically perturbed *teacher tokens*: roll the model out, then supervise it to recover the clean target from *its own* drifted prefix. This injects noise the model actually produces and uses, addressing the exposure-bias gap the teacher-forced denoising could not. (This is the Stage-2 rollout-aware direction the repo already treats as first-class.)
- **Larger / structured noise that the next-coordinate genuinely depends on** — e.g., perturb x1 and require x2 to track the *clean* width relative to the *noisy* x1 (relative-encoding denoising), so the target actually depends on the perturbed prefix. Only worthwhile if the model is shown to use relative coordinate context (it currently does not).
- **If KL is revived**: make it global over the coord vocabulary (or a much wider window), not a ±8-bin local window that sees ~14% of mass and is blind to the edge-saturation tails.

### De-confound the decode for any localization claim
- Evaluate with **`repetition_penalty = 1.0`** (or coord-token-exempt rep penalty) and a sane `max_new_tokens`. Report raw vs guarded AP and the empty/invalid/arity counters every time. The companion doc's "materialization vs localization" split should be standard.

---

## Verification

Already verified in this analysis:
- Objective inertness: `logging.jsonl` ce_clean≡ce_noisy and KL≈0 across train + 5 eval points (Evidence A).
- Forward isolation: code read of `_run_isolated_prefix_denoising_forwards` (Evidence C).
- Noise validity: code read of `construct_valid_norm1000_bbox_noise` + `_validate_clean_noisy_alignment` (Evidence D).
- Baseline non-comparability: reference checkpoint path = `...hard_ce_soft_ce_w1_gate/epoch_4.../checkpoint-1332-merged-full` (Evidence E).
- Failure profile: APm≈0.001, APs=0.0, recall collapse, empty_pred 64 (Evidence F).
- Decode confound: `repetition_penalty=1.1`, rp/temp AP sensitivity in sibling sweep (Evidence G).

Recommended confirming probes (not yet run; gate before any large run per repo rules):
1. Matched denoising-OFF control (settles causal attribution).
2. SoftCE-on rerun at matched budget (settles whether the ceiling is the coordinate objective).
3. rp=1.0 re-eval of the existing ckpt-450 (isolates decode-driven edge saturation).

---

## Confidence / Verdict

- **High confidence**: the denoising objective is inert in this run; the noisy/clean pair learning is not what degraded bboxes; the reported baseline is non-comparable; the AP ceiling is a hard-CE localization problem (small/medium-object AP ≈ 0).
- **Medium confidence**: a matched hard-CE-only LoRA control would land close to the prefix-denoising AP (i.e., near-zero denoising effect). Needs the control run to confirm.
- **Verdict**: This is a **negative result about the implementation's premise, not about geometry-aware denoising**. The model does not autoregressively condition coordinates on previously emitted coordinate tokens, so prefix-token denoising has nothing to act on. Re-run with a matched control + SoftCE before drawing any conclusion about the concept; if the idea is pursued, move the noise into a channel the model actually uses (self-rollout / relative-encoding), and de-confound the decode (rp=1.0).
