# Stage-2 (AB) Channel-A Only — COCO Parseability Degradation Diagnosis + Coord-Vocab Gate (2026-02-21)

Last updated: 2026-02-22

This note records an ongoing investigation into a serious **Stage-2 AB Channel-A only** failure mode on **COCO bbox-only** where rollout outputs become increasingly **unparseable / format-incorrect** as training proceeds (many invalid predictions dropped during eval).

It also documents the most likely algorithmic mechanism discovered so far (coord-token "type" leakage), the mitigation implemented (`coord_gate_weight`), and the rerun plan to verify whether the degradation disappears.

Status: **Coord gate implemented; em_detach CE-grad bug fixed; awaiting rerun confirmation.**

---

## Update (2026-02-22): `em_detach` was confounded by a CE-grad implementation bug

We found a **P0 implementation bug** in Channel-A `softctx_grad_mode=em_detach` that explains why `em_detach` runs showed rapidly increasing parse drops and format/key corruption even after adding coord-vocab gating.

Mechanism:

- Channel-A CE is intentionally anchored to the **A1** forward logits (`logits_a1`).
- However, in `em_detach` we were running the **A1 forward under `torch.no_grad()`** (to save memory on early iterations).
- That makes `logits_a1` **non-differentiable**, so **`loss/ce` contributes no gradient** in `em_detach`.

Observed symptoms (from COCO A-only `em_detach` runs):

- `loss/ce` explodes over training while coord losses remain roughly stable.
- Generated record keys drift from `"bbox_2d"` into invalid variants like `"bbox_0d"`, `"bbox_3d"`, `"bbox_4d"`, and `"bbox_d"`.
- Strict parsing drops these as `unexpected_keys`, driving `eval_rollout/parse_dropped_invalid` up and depressing recall (FN skyrockets due to fewer valid objects).

Fix:

- `src/trainers/stage2_ab_training.py` now enables grad on `it==0` (A1) **and** the final iter in `em_detach`, while keeping middle iters `no_grad`.
- Commit: `88a0873` (`stage2_ab: fix em_detach CE by enabling grad on A1`)

---

## 1) Symptom

During Stage-2 AB training in **Channel-A only** mode (self-context expectation / softctx iterations), rollouts progressively regress:

- More generated outputs fail strict parsing (invalid objects dropped).
- Many samples end up as `empty_pred` after dropping invalid objects.
- Overall rollout eval quality regresses even if earlier checkpoints were stable.

This is observed as training advances (e.g., around ckpt-2400 vs ckpt-300).

---

## 2) Affected runs / evidence sources

### 2.1 Inference/Eval artifacts showing degradation across training

- `output/stage2_ab/coco_bbox_max60/analysis/infer_ckpt-300_hf_b64_n100_gpu0`
- `output/stage2_ab/coco_bbox_max60/analysis/infer_ckpt-2400_hf_b64_n100_gpu1`

### 2.2 Original training log

- `output/stage2_ab/coco_bbox_max60/a_only/geometry_first_merged_ckpt1832_ep4/v1-20260220-144354/logging.jsonl`

### 2.3 Retrain with `em_detach` (still degraded / worse)

Artifacts:

- `output/stage2_ab/coco_bbox_max60/a_only_em_detach/desc_first_merged_ckpt1832_ep2/v0-20260221-113516`
- `output/stage2_ab/coco_bbox_max60/a_only_em_detach/geometry_first_merged_ckpt1832_ep2/v0-20260221-113950`

Interpretation (updated): these `em_detach` runs were later found to be **confounded by the CE-grad bug** (see Update section),
so they do **not** provide a fair comparison between `unroll` vs `em_detach` until rerun with `88a0873` or later.

---

## 3) Key observed failure mechanism: coord-slot token contamination

From raw debug dumps (see Section 5), many parse failures were caused by **non-coordinate tokens appearing inside bbox arrays**, e.g. producing:

- plain digits (`"85"`) or other text tokens in place of `<|coord_...|>`
- even non-ASCII tokens (CJK numerals were observed in dumps)

This yields strict-parse drops with reason patterns like:

- `wrong_arity` (bbox array is no longer 4 coord tokens)
- cascading to `empty_pred` (all objects dropped)

This is consistent with a model that has learned to put significant probability mass on the *general text vocab* at positions that are supposed to be restricted to the *coord vocab*.

---

## 4) Most likely root cause (algorithmic): coord-sliced losses without a coord-vocab gate

### 4.1 What we were (effectively) optimizing

Stage-2 AB uses self-context expectation decoding of coords, and the implementation computes many coord-related losses using **coord-vocab-sliced logits**:

- Build `coord_logits = logits_full[..., coord_token_ids]`
- Apply coord CE / expectation decode / entropy regularizers on `coord_logits`

This is fine for learning the *distribution within coord tokens* (which bin), but it can be blind to a crucial failure:

> If the model starts preferring non-coord tokens at coord slots (digits/words/etc), a coord-sliced objective can remain "happy" because it renormalizes within the coord vocab and never sees the competition from the rest of the vocab.

I.e., the training signal can say "coord bin 241 is correct" while the full-vocab model is actually putting most probability on `"85"` or `"book"` at that position.

### 4.2 Why this causes parseability collapse

At generation time, decoding is done from the **full vocabulary**, so even a small drift where non-coord tokens become competitive at coord slots can surface as:

- wrong token types in bbox arrays
- strict parse failures
- dropped objects and empty predictions
- downstream set-matching/eval collapse

This neatly matches the observed `wrong_arity` + `empty_pred` pattern.

### 4.3 Why this could differ from older infra / LVIS runs

This issue may not have been visible previously because:

- stage-1 training had a different loss mix (and/or already included a coord-vocab "type" gate)
- dataset/prompt differences (LVIS vs COCO) changed how easy it is for the model to "cheat" with digits/text tokens
- infrastructure changes altered which positions receive CE, or how label masking is applied

This is still under investigation; the immediate mitigation focuses on the symptom mechanism itself.

---

## 5) Mitigation implemented: `stage2_ab.coord_gate_weight` (coord-vocab gate loss)

We added a **coord-vocab gate loss** to Stage-2 AB training, controlled by:

- `stage2_ab.coord_gate_weight` (default: `1.0` in Stage-2 configs)

Implementation details:

- Function: `src/trainers/losses/coord_soft_ce_w1.py:coord_vocab_gate_loss`
- Integrated into Stage-2 AB: `src/trainers/stage2_ab_training.py`

Gate loss (per token) is:

`gate = logsumexp(full_vocab_logits/T) - logsumexp(coord_vocab_logits/T)`

This equals:

- `gate = -log( sum_{i in coord_vocab} softmax(full_vocab_logits/T)[i] )`

So it directly encourages:

- **high total probability mass on coord tokens at coord-supervised slots**

This is intentionally one-way: it penalizes "text leaks into coord slots", not the reverse.

Added logging scalars:

- `loss/coord_gate`
- `coord_diag/coord_vocab_mass_mean` (mean coord-vocab probability mass at coord slots)

Config rollout:

- Enabled by default in `configs/stage2_ab/base.yaml`
- Enabled by default in all `configs/stage2_ab/prod/*.yaml` and `configs/stage2_ab/smoke/*.yaml`

Commits (for traceability):

- `6e72e1b`: stage2_ab monitor dumps (eval-time raw debug)
- `63615f5`: stage2_ab coord-vocab gate loss + default `coord_gate_weight: 1.0`

---

## 6) Debugging aid implemented: raw parse-failure monitor dumps

To make parse failures actionable, we turned on eval-time dumps for rollout-matching monitoring:

- Dump directory: `<run_dir>/monitor_dumps/`
- Dumps include (per eval step): **10 samples**, **no truncation** (`max_text_chars: 0`)
- Goal: capture the exact raw generation that caused parse drops (and the reason)

This is enabled via config blocks under `configs/stage2_ab/prod/*.yaml` (and can be extended similarly).

---

## 7) Rerun plan (verification)

To verify the gate fix, rerun short experiments (at least to steps 300 and 600):

1) Channel-A only baseline:
   - `configs/stage2_ab/prod/geo_first_a_only.yaml`  (typically `softctx_grad_mode=unroll`)

2) Channel-A only with detached E-step:
   - `configs/stage2_ab/prod/geo_first_a_only_em_detach.yaml`

Keep everything else identical (seed, batch size, dataset version, decoding settings during eval) for a fair comparison.

### What to check

Primary success criteria (P0):

- `eval_rollout/parse_dropped_invalid` drops substantially vs prior runs
- `eval_rollout/sample_valid_pred_rate` stays high over training
- `monitor_dumps/step_*.json` shows bbox arrays with **only `<|coord_k|>` tokens** at coord slots

Secondary:

- `coord_diag/coord_vocab_mass_mean` should increase and stay near 1.0
- `loss/coord_gate` should be non-zero early and then reduce as the model learns to gate correctly

If parseability is fixed but performance still regresses, re-rank suspects:

- truncation / termination (rollout length growth)
- text/format token supervision (CE masking on structure tokens)
- prompt/template changes or stop-condition regressions

---

## 8) Open questions / next suspected failure modes (if gate is not enough)

If `coord_gate_weight` does not fix degradation, likely next culprits:

- **Text/format CE coverage bug**: format tokens not supervised (masking regression) leading to broken JSON.
- **Decode stopping/termination regressions**: the model learns to over-generate objects until `max_new_tokens`, causing truncation and JSON errors.
- **Coord token id mapping / offset regression**: coord ids mismatch between tokenizer/model/parser. (Less likely given the observed symptom is token-type leakage rather than wrong bins.)
- **Parser strictness drift**: eval parser contract changed vs training/inference output format.

For each, monitor dumps + drop-reason histograms should make it clear what dominates.

---

## 9) Pending: attach rerun results

When reruns complete, update this note with:

- run dirs (output + tb)
- a small table of `eval_rollout/*` deltas across steps (300/600/1200)
- a few minimal examples from monitor dumps demonstrating success/failure
