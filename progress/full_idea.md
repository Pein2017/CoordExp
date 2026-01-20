# CoordExp + EM-ish Set-Supervision for V-LLM Detection (Qwen3-VL, norm1000)

> Goal: Train a pretrained V-LLM (e.g., Qwen3-VL) to output **open-vocabulary object descriptions + norm1000 boxes** in a structured text format, while enabling **continuous geometric gradients** (L1/GIoU) to flow back into the LM head via **CoordExp** and reducing order sensitivity via **Hungarian matching** in an **EM-ish** training loop—**without adding a DETR-style detection head**.

---

## 0. Core Design Principles

### 0.1 Constraints
- Base model: pretrained V-LLM (Qwen3-VL).
- Coordinate representation: existing `<|coord_k|>` tokens with `k ∈ {0..999}` (norm1000; 1000 bins, 0 = near edge, 999 = just inside far edge).
- Do NOT introduce a separate detection head or DETR-like query decoder.
- Training should remain compatible with standard SFT infrastructure (teacher forcing, token CE), with additional losses computed from LM logits.

### 0.2 What “EM-ish” means here
- **E-step (latent alignment):** model generates a hypothesis output (rollout). Parse it into a set of predicted objects. Use Hungarian matching to align predictions to GT objects (latent correspondence).
- **M-step (parameter update):** update model parameters using (i) standard SFT on reordered GT for generation/format, and (ii) differentiable geometric losses computed from the model’s logits under the model’s own rollout context (“self-context forward”).

### 0.3 Why “self-context forward”
Hungarian matching is computed from the model’s rollout output. To make the geometric gradient consistent with the same context that produced that rollout, we recompute logits using teacher forcing on the rollout tokens (`ŷ`) and apply GT-based geometric supervision there.

This avoids supervising coordinates under a mismatched context (pure GT teacher forcing) and behaves like a policy-improvement loop, but stays fully differentiable (no REINFORCE/PPO).

---

## 1. Output Format / Token Protocol

### 1.1 Canonical object record format
Each object is encoded as a fixed-structure token subsequence:

```

<obj> <desc> ...desc tokens... </desc> <box> x1 y1 x2 y2 </box> </obj>

```

- `x1 y1 x2 y2` are each a single `<|coord_k|>` token (k=0..999).
- Multiple objects are concatenated:
```

<bos>  OBJ_1  OBJ_2  ...  OBJ_N  <eos>

```

### 1.2 Parsing invariants
- Records must be parseable deterministically:
  - Find `<obj> ... </obj>` segments.
  - Inside each segment, find `<desc> ... </desc>` (text) and `<box> ... </box>` (4 coord tokens).
- During training and evaluation, only parse objects that satisfy:
  - exactly 4 coord tokens inside `<box> ... </box>`
  - optional additional formatting constraints as needed.

---

## 2. CoordExp: Differentiable Coordinate Decode from LM Logits

### 2.1 Coordinate token subset
- `V_coord = { <|coord_0|>, <|coord_1|>, ..., <|coord_999|> }`
- `K = 1000`

### 2.2 From logits to probability over coord tokens
At a coordinate position `t`, LM head produces logits `z_t ∈ R^{|V|}` over the full vocab.

Gather coord-subspace logits:
- `s_{t,k} = z_t[coord_k]` for `k=0..999`

Softmax (temperature τ):
- `p_{t,k} = exp(s_{t,k}/τ) / Σ_m exp(s_{t,m}/τ)`

### 2.3 Expectation decode (continuous coord)
Map `k -> k/1000`:
- `φ(k) = k/1000 ∈ [0,1]`

Expectation:
- `ĉ_t = Σ_k p_{t,k} * φ(k)`

For an object `i`, apply to 4 positions:
- `b̂_i = (x̂1_i, ŷ1_i, x̂2_i, ŷ2_i)` in `[0,1]^4`

### 2.4 Key gradient identity (for documentation/analysis)
- `∂ĉ_t/∂s_{t,k} = (1/τ) * p_{t,k} * (φ(k) - ĉ_t)`
So geometric losses on `ĉ_t` backprop to coord logits smoothly.

### 2.5 Practical knobs (optional but recommended)
- Temperature schedule (anneal τ down during training).
- Optional entropy regularization on coord distributions to prevent multi-peak “mean collapse”.
- Optional top-k expectation (mask to k highest probs before softmax) if multi-modal instability appears.

---

## 3. Dataset / Supervision Format

### 3.1 Data sources and tasks

We target **open-vocabulary detection and grounding** on top of existing public datasets:

- Detection / instance segmentation: COCO, LVIS, Objects365, etc.
- Referring expressions / grounding: RefCOCO, RefCOCO+, RefCOCOg, etc.

All sources are converted into a **shared JSONL contract** defined in `docs/DATA_JSONL_CONTRACT.md`. Conceptually, each record corresponds to one or more images plus a set of annotated objects.

### 3.2 JSONL data contract (high level)

Each JSONL line follows the global contract:

- `images` (list of str, required): relative image paths.
- `objects` (list, required): one entry per annotated object.
- `width`, `height` (int, required): image size in pixels.
- Optional fields: `summary` (single-line description) and `metadata` (provenance, dataset tags, etc.).

Each object has:

- `desc` (str, required): open-vocabulary description or class phrase.
- Exactly one geometry field (required, mutually exclusive):
  - `bbox_2d`: `[x1, y1, x2, y2]` rectangular box.
  - `poly`: flat `[x1, y1, ..., xn, yn]` polygon.
  - Note: `line` / `line_points` polyline geometry was removed from the runtime contract and is no longer supported.

Coordinates can be **pixel-space floats** or **pre-tokenized coord tokens** (`"<|coord_k|>"`, `k ∈ [0, 999]`). When using coord tokens in the JSONL, we keep `width`/`height` and set `custom.coord_tokens.skip_bbox_norm: true` in configs to avoid double normalization.

For intuition, a single-image shorthand (omitting some optional fields) looks like:

```jsonc
{
  "images": ["path/to/image.jpg"],
  "width": 640,
  "height": 480,
  "objects": [
    {
      "desc": "a red cup on the table",
      "bbox_2d": [x1, y1, x2, y2]  // pixel or coord-tokenized values
    }
  ]
}
```

For early CoordExp experiments, **axis-aligned bounding boxes** (`bbox_2d`) are the main target; polygons are supported by the contract but used later.

### 3.3 Supervision view

For each (single) image in a record we can think in terms of an abstract GT object set:
- `G = { (d_j, b_j) }_{j=1..M}`
  - `d_j`: description tokens for object j (open-vocab phrase/caption) derived from `desc`.
  - `b_j`: GT box in normalized coordinates `[0,1]` or norm1000 integers (after optional preprocessing).

When we choose to supervise in coord-token space, we convert normalized coordinates to discrete bins:
- `coord_bin(c) = round(1000 * c)` with clamp to `[0, 999]`
- Use `<|coord_coord_bin|>` as the GT coord token.

---

## 4. Full Training Pipeline (Multi-Stage)

### Overview
We separate training into two conceptual stages:
- **Stage-1:** Pure SFT / format stabilization / coord token adaptation.
- **Stage-2:** EM-ish loop with rollout + matching + GT-supervised geometric calibration under self-context.

---

## 5. Stage-1: Standard SFT to “learn the language of boxes”

### 5.1 Inputs
- image + prompt (task instruction)
- GT token sequence `y_GT` constructed by concatenating all GT objects in a chosen canonical order (e.g., left-to-right, top-to-bottom; any fixed rule is fine for Stage-1)

### 5.2 Forward
Teacher forcing on GT sequence:
- logits at each step `z_t = f_θ(image, prompt, y_<t)`

### 5.3 Loss
- Base: full-token cross entropy on GT:
  - `L_CE_all = -Σ_t log p_θ(y_t | image, prompt, y_<t)`
- Optional: extra weight on coord token positions (coord-focused CE):
  - `L_CE_coord = -Σ_{t∈coord_positions} log p_θ(y_t | ...)`
- Optional (light): CoordExp-L1 on coord positions to introduce continuous geometry early:
  - decode `ĉ_t` from logits at coord positions
  - `L_L1_coord = Σ_{t∈coord_positions} |ĉ_t - c_gt|`
- Stage-1 total:
  - `L_stage1 = L_CE_all + λ_coordCE * L_CE_coord + λ_L1 * L_L1_coord`

### 5.4 Purpose
- Make the model reliably output the structured format.
- Make coord token prediction stable and aligned with norm1000 vocabulary.
- Build a baseline that already works with greedy decoding.

---

## 6. Stage-2: EM-ish Training Loop (Rollout → Match → Update)

### 6.1 Per-sample loop (single image)

#### Step A — Rollout (E-step hypothesis, no grad)
1) Run autoregressive generation using current model θ:
   - `ŷ = Rollout(f_θ, image, prompt)` (greedy or beam)
2) Parse rollout into predicted objects:
   - `Ô = Parse(ŷ)` returns a list:
     - `Ô = { (d̂_i, b̂_i_disc, idx_i) }_{i=1..N}`
   Where:
  - `d̂_i`: decoded description string/tokens from `<desc> ... </desc>`
  - `b̂_i_disc`: discrete box from the 4 coord tokens (ints 0..999, or normalized to [0,1])
   - `idx_i`: token indices for the 4 coord positions in the rollout sequence (used later)

> Notes:
> - If parsing fails, optionally fall back to Stage-1 SFT only for that sample.

#### Step B — Hungarian matching (E-step correspondence, no grad)
3) Build cost matrix `C ∈ R^{N×M}` between predicted objects `i` and GT objects `j`:
   - Geometry cost: `L_geo_disc(b̂_i_disc, b_j_gt)` (can be L1 on norm1000 + IoU-based term)
   - Optional semantic cost: `D_sem(d̂_i, d_j_gt)` (use frozen text encoder; keep small early)
   - `C_{i,j} = λ_geo * L_geo_disc + λ_sem * D_sem`
4) Run Hungarian to get assignment:
   - `σ = Hungarian(C)`
   - Produces a set of matched pairs `(i -> j)`.

5) Optional gating for stability:
   - Only accept a match if (IoU_disc(b̂_i_disc, b_j_gt) > iou_thr) or (L1 < thr).
   - Let `ValidMatched = { i | match accepted }`.

> Purpose:
> - Avoid wrong matches producing wrong geometric gradients.

---

## 7. Stage-2 Updates (M-step): Two complementary supervision paths

Stage-2 consists of **two backprop-able losses** that target different failure modes:

### 7.1 Path A: Geometric calibration under self-context (CoordExp + L1/GIoU)

#### Step C — Self-context forward (with grad)
6) Run a teacher-forced forward pass using **the rollout tokens** as context:
   - Input tokens: `ŷ_{<t}` (rollout prefix)
   - Compute logits for all positions:
     - `z_t^self = f_θ(image, prompt, ŷ_<t)`
   - This is standard teacher forcing, but the sequence is `ŷ` (model’s own outputs), not GT.

#### Step D — CoordExp decode on coord positions
7) For each predicted object `i` and each of its 4 coord token positions `t ∈ idx_i`:
   - Gather coord logits: `s_{t,k} = z_t^self[coord_k]`
   - Softmax → `p_{t,k}`
   - Expectation → `ĉ_t`
8) Assemble continuous predicted box:
   - `b̂_i_cont = (ĉ_{t_x1}, ĉ_{t_y1}, ĉ_{t_x2}, ĉ_{t_y2})`

#### Step E — Geometric loss using matched GT (GT supervision)
9) For each accepted match `(i -> j)` with `i ∈ ValidMatched`:
   - `ℓ_geo(i,j) = α * || b̂_i_cont - b_j_gt ||_1 + β * (1 - GIoU(b̂_i_cont, b_j_gt))`
10) Sum:
   - `L_geo_self = Σ_{(i->j), i∈ValidMatched} ℓ_geo(i,j)`

> Key property:
> - Context tokens are `ŷ` (self context).
> - Supervision target is GT box `b_j_gt`.
> - No coordinate pseudo-labeling from `ŷ`.

---

### 7.2 Path B: Generation / format / recall control via reordered GT SFT

This path handles:
- Missing objects (not generated in rollout, thus no coord positions exist)
- Extra objects (hallucinations)
- Caption correctness and formatting robustness

#### Step F — Reorder GT according to matching
11) Build a “reordered GT sequence” `y_GT_reordered`:
   - The first `N` GT slots are arranged to correspond to predicted object order:
     - if predicted object `i` matched to GT `j`, place GT object `j` at position `i`
   - For GT objects that were not matched (misses), append them as “completion objects” at the end (optional, controlled by policy).
   - For predicted objects that were unmatched (extras), optionally enforce early stop behaviors (see below).

#### Step G — Standard SFT (with grad)
12) Teacher forcing on `y_GT_reordered`:
   - `z_t^gt = f_θ(image, prompt, y_GT_reordered_<t)`
13) Compute CE losses:
   - `L_CE_text`: CE on description tokens + structure tokens
   - `L_CE_coord`: optional CE on coord tokens (discrete supervision)
   - `L_CE_eos`: optional targeted CE encouraging `<eos>` at end-of-sequence when appropriate

> Optional handling policies:
> - **Missing GT objects:** append as extra GT object records so CE teaches the model to include them.
> - **Extra predicted objects:** encourage termination after expected objects:
>   - simplest: ensure GT ends with `<eos>`; the model is penalized if it keeps generating.
>   - optionally: add a rule that if rollout generates too many objects, the reordered GT truncates earlier and increases `<eos>` weight.

---

## 8. Stage-2 Total Objective

Combine both paths:

- `L_stage2 = L_CE(y_GT_reordered) + λ_geo * L_geo_self + λ_coordCE * L_CE_coord + λ_ent * L_entropy(optional)`

Where:
- `L_CE(y_GT_reordered)` includes text + structure + optional EOS weighting.
- `L_geo_self` provides continuous geometric gradients under self context.
- `L_CE_coord` sharpens coord distributions and reduces multi-peak ambiguity.
- `L_entropy` (optional) regularizes coord distributions.

---

## 9. Inference Pipeline (Deployment)

Given image + prompt:
1) Autoregressively generate sequence `y_pred` (greedy/beam).
2) Parse objects:
   - extract `<desc>` text
   - extract `<|coord_k|>` tokens for x1,y1,x2,y2
3) Convert coord tokens to box values:
   - discrete: `k/1000`
   - optional: if you want continuous decode at inference, you can compute CoordExp from logits during generation, but simplest is discrete token readout.
4) Output:
   - `{ desc_i, box_i }` for each object.

> Note: training uses CoordExp to improve gradients; inference can still be discrete token decoding.

---

## 10. Implementation Checklist (What must exist)

### 10.1 Token / formatting utilities
- Template builder: GT objects → token sequence (`y_GT`)
- Parser: predicted sequence → list of objects + coord token indices
- Coord vocab index list: indices of `<|coord_0|>.. <|coord_999|>` in tokenizer

### 10.2 Matching utilities
- Cost matrix builder:
  - geometry cost from discrete coords (fast IoU/L1)
  - optional semantic cost from frozen text embedding model
- Hungarian solver + match gating

### 10.3 CoordExp module
- Given logits `z_t`, gather coord logits, compute softmax, expectation
- Return continuous scalar in [0,1] (or [0,999] if you prefer working in index space)

### 10.4 Loss module
- `L_geo_self`: L1 + GIoU over continuous boxes from CoordExp
- `L_CE`: standard token CE (with optional position-wise weights)
- optional entropy regularizer and/or top-k expectation

---

## 11. Multi-Stage Roadmap (High-Level)

### Stage-1 (SFT baseline)
- Ensure robust structured output and coord token usage.
- Verify: parsing success rate, basic mAP/IoU with discrete coords.

### Stage-2 (EM-ish improvements)
- Add rollout + matching + self-context geometric calibration.
- Use gating and curriculum:
  - early: geometry-only matching, strict gating
  - later: relax gating; optionally add semantic term
- Evaluate:
  - localization improvement (IoU distribution, GIoU loss reduction)
  - robustness to permutation/order
  - reduced sensitivity to quantization boundary cases

### Optional Stage-3 (light rollout consistency)
- Low-frequency additional rollouts with the same machinery.
- Focus on structure validity and geometric calibration; avoid heavy RL.

---

## 12. Minimal Working Example (Two objects: “black cat” and “yellow dog”)

### GT sequence
```

<bos>
<obj><desc> black cat </desc><box> <coord_110> <coord_310> <coord_410> <coord_705> </box></obj>
<obj><desc> yellow dog </desc><box> <coord_520> <coord_285> <coord_890> <coord_660> </box></obj>
<eos>
```

### Rollout (model output)

```
<bos>
<obj><desc> black cat </desc><box> <coord_120> <coord_300> <coord_420> <coord_700> </box></obj>
<obj><desc> yellow dog </desc><box> <coord_500> <coord_280> <coord_880> <coord_650> </box></obj>
<eos>
```

### Matching

* predicted #1 → GT(cat)
* predicted #2 → GT(dog)

### Self-context geometric update

* teacher force on rollout tokens to compute logits at coord positions
* CoordExp decode continuous boxes
* apply L1+GIoU to matched GT boxes
* backprop to LM head coord logits and upstream θ

### Reordered-GT SFT update

* reorder GT to follow predicted order (if needed)
* standard CE to improve captions/format, fix misses/extras

---

## 13. Notes on “No extra detection head”

This pipeline:

* uses the pretrained V-LLM exactly as-is
* relies on:

  * structured token protocol
  * CoordExp differentiable decoding from LM logits
  * Hungarian matching as a stop-grad alignment step
  * SFT-compatible losses
* avoids DETR query decoders and separate regression heads.

---

## 14. Expected Failure Modes & Built-in Stabilizers

### 14.1 Wrong matching early

* Use geometry-only matching initially.
* Use strict gating (IoU/L1 thresholds) to avoid bad gradients.

### 14.2 Missing objects

* Self-context geo loss cannot fix missing outputs.
* Use reordered-GT SFT with “completion append” to teach recall.

### 14.3 Extra objects

* Penalize continuation after GT end via EOS weighting.
* Optionally truncate GT length and emphasize EOS when rollout over-generates.

### 14.4 Multi-peak coord distributions (“mean collapse”)

* Use coord CE auxiliary.
* Temperature anneal.
* Optional entropy regularization / top-k expectation.

---

## 15. Summary

This document specifies a complete end-to-end pipeline for training a pretrained V-LLM (Qwen3-VL) to perform open-vocab detection using structured text output with norm1000 coordinates, leveraging:

* CoordExp for differentiable geometry gradients into LM logits,
* Hungarian matching for set-level alignment,
* EM-ish rollout + update loops,
* and standard SFT for caption/format/recall control,
  all without introducing new detection heads or DETR-style architectures.
