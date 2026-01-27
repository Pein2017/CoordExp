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
- We separate **continuous / smooth geometry calibration** from **discrete / set-level correction**:
  - **Channel-A (hot path):** no autoregressive rollout; do **N× forward** (iterative soft self-context) to approximate “self-context” *only on coord slots* using soft coord embeddings (differentiable, fast).
  - **Channel-B (cold path):** occasional autoregressive rollout + Parse + Hungarian matching to handle **permutation / missing / extra / malformed JSON** and to ground “true self-context” learning.

Under this view:
- **E-step (latent alignment):** performed only on Channel-B steps via rollout → set → Hungarian (stop-grad correspondence).
- **M-step (parameter update):** always uses SFT-compatible losses; geometry losses can be applied under:
  - Channel-A: “soft self-context proxy” (no sampling)
  - Channel-B: “true self-context” (teacher-forced on rollout tokens)

### 0.3 Why “self-context forward”
Hungarian matching is computed from the model’s rollout output. To make the geometric gradient consistent with the same context that produced that rollout, we recompute logits using teacher forcing on the rollout tokens (`ŷ`) and apply GT-based geometric supervision there.

This avoids supervising coordinates under a mismatched context (pure GT teacher forcing) and behaves like a policy-improvement loop, but stays fully differentiable (no REINFORCE/PPO).

**Update (practical):** in large V-LLMs, “true self-context forward” is expensive because it requires autoregressive rollout.
We therefore introduce a **soft self-context proxy** that replaces only `<|coord_*|>` token embeddings by their expected embedding
under the model’s own coord distribution, enabling a *non-sampling* approximation that is compatible with Qwen3’s tied weights.

**Iterative variant (Channel-A):** the same soft self-context proxy can be run as an *iteration* (no rollout) to reduce “GT leakage / lag” and move closer to on-policy self-conditioning.

- Let `e^(m)` denote the (soft) embeddings used at coord slots in iteration `m` (text/struct tokens remain teacher-forced under GT embeddings).
- Define a mapping `e^(m+1) = F_θ(e^(m))` where `F_θ` is: one full forward under `e^(m)` → extract coord-slot distributions → form expected coord embeddings.
  - For coord slot `t`: `p_t^(m) = softmax(s_t^(m) / τ_A)` and `ē_t^(m) = Σ_k p_{t,k}^(m) * E_coord[k]`.
- Using `n_softctx_iter = N` total forwards corresponds to `N-1` applications of `F_θ` (Jacobi-style fixed-point intuition).
- Larger `N` is *closer* to rollout/self-context (because the coord context is repeatedly updated from the model’s own belief), but it is still not equivalent to autoregressive rollout because it is soft (expectation), non-sampled, and keeps the GT text/structure scaffold.
- In practice, bbox chains are short (4 coords). `N=2` is often sufficient; `N=3` is a useful ablation to test whether the teacher-forcing vs rollout gap shrinks further.

---

## 1. Output Format / Token Protocol

### 1.1 Canonical object record format (JSON-only, dense mode)
CoordExp uses JSON-only assistant outputs (no wrapper tags like `<obj>`/`<box>`). In dense mode, the assistant emits a single top-level JSON object mapping `"object_1"`, `"object_2"`, ... to per-object payloads. This payload is derived from the dataset JSONL record contract (`docs/DATA_JSONL_CONTRACT.md` / `docs/DATA_AND_DATASETS.md`), which stores objects as a list; the template enumerates that list and assigns stable `object_i` keys.

Each `object_i` value is a JSON object with:
- `desc` (string, required): open-vocabulary description / class phrase.
- Exactly one geometry field (required, mutually exclusive):
  - `bbox_2d`: `[x1, y1, x2, y2]`
  - `poly`: polygon vertices (>= 3 points). In text it may appear as a flat list `[x1, y1, ...]` or as paired points `[[x1, y1], ...]` as long as it flattens to an even-length list. JSONL records may also include `poly_points` as metadata, but the assistant output does not need it (and strict parsing may drop unexpected keys).

Coordinate representation (norm1000):
- Coord-token mode (recommended for Stage-2): each coordinate is a string `"<|coord_k|>"` where `k ∈ [0, 999]`.
- Numeric mode: coordinates are integers in `[0, 999]` (pre-normalized offline; runtime normalization is disabled).

Example (coord-token mode, bbox only):

```json
{
  "object_1": {
    "desc": "black cat",
    "bbox_2d": ["<|coord_110|>", "<|coord_310|>", "<|coord_410|>", "<|coord_705|>"]
  },
  "object_2": {
    "desc": "yellow dog",
    "bbox_2d": ["<|coord_520|>", "<|coord_285|>", "<|coord_890|>", "<|coord_660|>"]
  }
}
```

### 1.2 Parsing invariants
- The response MUST contain a top-level JSON object.
- Valid objects are entries whose keys match `object_<N>` (1-indexed).
- Each object MUST contain:
  - a non-empty `desc`, and
  - exactly one geometry field: `bbox_2d` or `poly`.
- Objects SHOULD NOT contain extra keys beyond `desc` + the geometry field (strict parsing may drop them).
- Geometry arity rules (after flattening nested lists, if any):
  - `bbox_2d` must contain exactly 4 coordinates.
  - `poly` must flatten to an even-length list with length >= 6.
- In coord-token mode, geometry values MUST be coord-token strings `"<|coord_k|>"` with `k ∈ [0, 999]`. Malformed objects are dropped (no repair).

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
Map `k -> k/999`:
- `φ(k) = k/999 ∈ [0,1]`

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
- `coord_bin(c) = round(999 * c)` with clamp to `[0, 999]`
- Use `<|coord_k|>` as the GT coord token, where `k = coord_bin(c)` (clamped to [0,999]).

---

## 4. Full Training Pipeline (Multi-Stage)

### Overview
We separate training into two conceptual stages:
- **Stage-1:** Pure SFT / format stabilization / coord token adaptation.
- **Stage-2:** two-channel mixture:
  - **Channel-A (default):** N× forward iterative soft self-context (no rollout), with default n_softctx_iter=2; n_softctx_iter=1 degenerates to pure teacher-forcing.
  - **Channel-B (sparse):** rollout + matching + GT-supervised geometric calibration under true self-context + reordered-GT SFT for set-level correction.

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

### 6.1 Two-channel schedule (recommended)
We run Stage-2 as a **mixture** over steps / samples:
- **Channel-A (hot path, e.g., 95% steps):** no rollout; always available and fast.
- **Channel-B (cold path, e.g., 5% steps or 1/P batches):** true rollout + Hungarian to fix discrete failure modes.

Rationale:
- In practice, “hard CE on coord tokens” is already strong in Stage-1/SFT, so Stage-2 should not assume coord-distribution losses will dominate.
- The main remaining instability is typically **self-context generation** (long JSON, permutation, missing/extras/format).
  Channel-B targets that directly; Channel-A provides a cheap, smooth geometry signal to stabilize coord decoding and reduce drift.

Practical suggestion: default Channel-A uses `n_softctx_iter=2`, and run an ablation `1 vs 2 vs 3` to quantify the teacher-forcing vs self-context gap (still no rollout). Optionally, increase iterations for samples/objects with high coord entropy or low `max_prob` (a heuristic; not required).

---

### 6.2 Channel-A (hot): Iterative soft self-context via N× forward (no rollout)

Key idea: approximate “self-context” only where it matters most (coord slots), without sampling.
We keep all **text/struct** tokens teacher-forced as GT, but feed **coord tokens** as *soft embeddings*
derived from the model’s own coord distribution.

Definitions / hyperparameters:

- `n_softctx_iter` : total number of full-sequence forwards in Channel-A (`>= 1`).
  - `n_softctx_iter=1`: only Step A1 (pure teacher forcing), a degenerate / control variant (no soft self-context).
  - `n_softctx_iter=2`: the original Channel-A (A1 → A2 → A3 → A4).
  - `n_softctx_iter>2`: iterate the A2 ↔ A3 loop multiple times (fixed-point / Jacobi-style intuition).
- Iteration indexing: we use `m = 0..n_softctx_iter-1`, where `m=0` corresponds to the teacher-forced GT forward in Step A1. The **final iteration** logits used for CoordExp decode and geometric loss are `z^(n_softctx_iter-1)`.
- (Optional) `softctx_damping_alpha` : damping coefficient for coord-slot embedding updates:
  - `E_new = (1-α) * E_old + α * ē`, default `α=1`.
- (Optional) Adaptive iterations: run extra iterations only for low-confidence coord slots (e.g., high entropy or low `max_prob`). This is a suggestion; keep the default deterministic.

#### Step A1 — Teacher-forced GT forward (with grad)
1) Teacher force on GT (or reordered-GT if you already have one) to get logits:
   - `z_t^gt = f_θ(image, prompt, y_GT_<t)`

2) Identify coord positions `t ∈ coord_positions` from labels/template (packing-safe). For each coord position `t`:
   - `s_t^(0) ∈ R^K`, `s_{t,k}^(0) = z_t^gt[coord_k]`
   - `p_t^(0) = softmax(s_t^(0) / τ_A)` over `k=0..999`

> Note (stability): early in Stage-2, **detach** the coord distribution used to build soft embeddings:
> - `p_t^(m,stop) = stopgrad(p_t^(m))`
> This keeps Channel-A “EM-ish”: the soft context is treated like an E-step artifact, while gradients come from the (final) softctx forward + the usual SFT CE.

#### Step A2 — Build soft coord embeddings (no sampling)
3) Let `E_coord ∈ R^{K×d}` be the embedding table restricted to `<|coord_k|>`.
   Construct the expected coord embedding at each coord slot:
   - `ē_t^(m) = Σ_k p_{t,k}^(m,stop) * E_coord[k]`

4) Prepare the initial embedding tensor (full sequence) for iterative soft self-context:
   - Start from standard embeddings of `y_GT` (or from the model’s embedding lookup): `emb^0`.
   - For coord positions, you have two valid initializations:
     - (A) **Start from GT** (recommended): keep coord slots as GT in `emb^0`, and only start replacing after the first update.
     - (B) Start from `ē^(0)`: immediately replace coord slots in `emb^0` by `ē_t^(0)` before the first softctx forward.
   - Keep attention mask unchanged.

This is cheap because `K=1000` and coord slots are sparse compared to total tokens.

#### Step A3 — Iterative soft self-context forwards (with grad)
5) Run a Jacobi-style “soft self-conditioning” iteration using full-sequence forwards (NOT token-by-token autoregressive generation):

```text
# Inputs: GT tokens y_GT, coord_positions, E_coord, n_softctx_iter >= 1
# Output: final logits z^(n_softctx_iter-1)

# (Forward #0) already computed in A1:
z^(0) := z^gt
Compute p^(0) from z^(0) at coord_positions
Compute ē^(0) from p^(0)

emb^0 := Embed(y_GT)               # full GT embeddings
Optional: replace coord slots with ē^(0)  # init variant (B); default is (A) start-from-GT

for m in 1 .. (n_softctx_iter - 1):
    z^(m) := f_θ(image, prompt, inputs_embeds=emb^(m-1))   # full forward
    Compute p^(m) from z^(m) at coord_positions
    Compute ē^(m) from p^(m)
    emb^(m) := UpdateCoordSlots(emb^(m-1), ē^(m), damping_alpha=α)  # α=1 by default

Use final logits z^(n_softctx_iter-1) for CoordExp decode + losses
```

Interpretation: this is a fixed-point iteration on coord-slot context, `e^(m+1)=F_θ(e^(m))`, that repeatedly refreshes coord embeddings from the model’s own belief while keeping the GT text/structure scaffold. Increasing `n_softctx_iter` pushes the effect of self-conditioning further along the coord chain without sampling.

#### Step A4 — Geometry + (optional) coord sharpening losses
6) Default: decode continuous coords only from the **final** logits `z^(n_softctx_iter-1)` (i.e., `z_t^self`, Section 7.1) at coord slots via CoordExp, and assemble boxes per object.
7) Apply geometry loss against GT boxes **by identity alignment** (no Hungarian in Channel-A):
   - `L_geo_soft = Σ_obj [ α||b̂_obj^cont - b_obj^gt||_1 + β(1-GIoU(...)) ]`

8) (Optional) add a *light* coord-distribution sharpening term computed from the final logits:
   - `L_coordCE_soft = -Σ_{t∈coord_positions} log p_θ(y_t^gt | ... , softctx)`
   - or your existing (softCE/W1/gate) but keep it small; treat as regularizer, not the main driver.

Optional enhancement (not required): supervise intermediate iterates as well:
- `L_geo_soft_total = Σ_m w_m * L_geo_soft^(m)` with `w_m` increasing over `m` (late iterates matter more).

Channel-A per-sample loss:
- `L_A = L_CE(y_GT) + λ_geo^A L_geo_soft + λ_coord^A L_coord_sharp(optional)`

> Why this is “unified” with rollout-matching:
> - Channel-A trains geometry under a form of self-conditioning (but without sampling).
> - Channel-B trains under true self-conditioning (rollout tokens), but sparsely.
> Together they approximate “train on what you infer” while keeping throughput high.

Why `N×` helps (intuition, bbox case):
- `N=2` mainly constrains `y1` under a soft (model-derived) `x1` context instead of the GT `x1`.
- `N=3` further constrains `x2/y2` under a context where `y1` (and partially `x2`) is closer to on-policy soft self-conditioning.
- This is not “sampling error accumulation”; it targets “GT leakage / lag” that grows with chain length. One more iteration can reduce the lag without performing autoregressive rollouts.

---

## 7. Stage-2 Updates (M-step): Two complementary supervision paths

Stage-2 consists of **two complementary paths** (Channel-A hot path + Channel-B cold path) that target different failure modes:

### 7.1 Channel-A (hot path): Geometric calibration under soft self-context (CoordExp + L1/GIoU)

Channel-A uses the **N× forward** construction from Section 6.2:
- First forward: teacher-forced GT to obtain `p_t^(0)` over coord bins (and initial expected embeddings `ē^(0)`).
- Subsequent forward(s): replace coord token embeddings by expected embeddings and forward again, optionally iterating (`n_softctx_iter >= 1`).

We denote the final-iteration logits as:
- `z_t^self = z_t^(n_softctx_iter-1)`
- Default: all CoordExp decodes, `L_geo_soft`, and any coord sharpening terms use `z_t^self` (final iteration logits) only.
  (Optional, not default): you may also supervise intermediate iterates via a weighted sum across `m`, but keep the default single-shot on the final iteration for simplicity.

#### Step D — CoordExp decode on coord positions
7) For each GT object `i` and each of its 4 coord token positions `t` in the template:
   - Gather coord logits: `s_{t,k} = z_t^self[coord_k]`
   - Softmax → `p_{t,k}`
   - Expectation → `ĉ_t`
8) Assemble continuous predicted box:
   - `b̂_i_cont = (ĉ_{t_x1}, ĉ_{t_y1}, ĉ_{t_x2}, ĉ_{t_y2})`

#### Step E — Geometric loss using GT (identity alignment)
9) For Channel-A, object indices come from the GT template, so we use direct alignment:
   - `ℓ_geo(i) = α * || b̂_i_cont - b_i_gt ||_1 + β * (1 - GIoU(b̂_i_cont, b_i_gt))`
10) Sum:
   - `L_geo_soft = Σ_i ℓ_geo(i)`

> Key properties:
> - Context is **partially self-conditioned** (coord embeddings come from model belief).
> - Supervision target is GT box.
> - No sampling; stable gradients; throughput-friendly.

---

### 7.2 Channel-B (cold path): Rollout-matching for set-level / discrete correction

Channel-B is the original EM-ish loop, run sparsely to correct:
- permutation/order mismatch,
- missing GT objects / extra predicted objects,
- malformed JSON / broken segments,
- true “self-context degeneration” that Channel-A cannot expose.

#### Step B0 — Rollout (E-step hypothesis, no grad)
1) Run autoregressive generation using current model θ:
   - `ŷ = Rollout(f_θ, image, prompt)` (greedy / low-T sampling / beam)
2) Parse rollout into predicted objects:
   - `Ô = Parse(ŷ)` returns:
     - `Ô = { (d̂_i, b̂_i_disc, idx_i) }_{i=1..N}`

> Notes:
> - If parsing fails, fall back to Channel-A only for that sample, and log it as “invalid-rollout”.

#### Step B1 — Hungarian matching (E-step correspondence, no grad)
3) Build cost matrix `C ∈ R^{N×M}` between predicted objects `i` and GT objects `j`:
   - Geometry cost: `L_geo_disc(b̂_i_disc, b_j_gt)` (L1 on norm1000 + IoU term)
   - Optional semantic cost: `D_sem(d̂_i, d_j_gt)` (frozen encoder; keep small early)
   - `C_{i,j} = λ_geo * L_geo_disc + λ_sem * D_sem`
4) Hungarian:
   - `σ = Hungarian(C)` → matched pairs `(i -> j)`
5) Gating:
   - accept only if IoU/L1 passes thresholds
   - `ValidMatched = { i | match accepted }`

#### Step B2 — True self-context forward (with grad)
6) Teacher force on rollout tokens to recompute logits:
   - `z_t^roll = f_θ(image, prompt, ŷ_<t)`

7) CoordExp decode on coord positions `t ∈ idx_i`, assemble `b̂_i_cont`.
8) Apply geo loss on matched GT targets:
   - `L_geo_roll = Σ_{(i->j), i∈ValidMatched} [ α||b̂_i_cont - b_j_gt||_1 + β(1-GIoU) ]`

#### Step B3 — Reordered-GT SFT (with grad)
9) Build `y_GT_reordered` following predicted order + append missing GT objects (see Section 7.3).
10) Standard SFT on `y_GT_reordered`:
   - `L_CE_text`, `L_CE_coord(optional)`, `L_CE_eos(optional)`

Channel-B per-sample loss:
- `L_B = L_CE(y_GT_reordered) + λ_geo^B L_geo_roll + λ_coord^B L_coord_sharp(optional)`

---

### 7.3 Path B (within Channel-B): Generation / format / recall control via reordered GT SFT

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

Combine both channels:

- Stage-2 is a mixture objective:
  - `L_stage2 = (1-ρ) * L_A + ρ * L_B`
  - where `ρ` is the Channel-B frequency/probability (small, e.g. 0.05).

Expanded:
- `L_A = L_CE(y_GT) + λ_geo^A * L_geo_soft + λ_coord^A * L_coord_sharp(optional)`
- `L_B = L_CE(y_GT_reordered) + λ_geo^B * L_geo_roll + λ_coord^B * L_coord_sharp(optional)`

Where:
- Channel-A geo term improves coord calibration cheaply under soft self-conditioning.
- Channel-B terms fix true self-context + set-level discrete problems (order/missing/extras/format).

Practical defaults (based on current evidence that hard-CE is not weak):
- Keep `L_CE` as the anchor in both channels.
- Treat coord-distribution losses (softCE/W1/gate/entropy/top-k) as optional regularizers, not mandatory.

---

## 9. Inference Pipeline (Deployment)

Given image + prompt:
1) Autoregressively generate sequence `y_pred` (greedy/beam).
2) Parse objects:
   - parse the JSON dict and read each object's `desc` string
   - parse `bbox_2d` as 4 coord-token strings `"<|coord_k|>"` (x1,y1,x2,y2)
3) Convert coord tokens to box values:
   - discrete: `k/999`
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

### 10.1.1 (New) Soft self-context utilities (Channel-A)
- A packing-safe way to get `coord_positions` (from labels/template).
- `E_coord` gather: embedding rows for coord token IDs (K=1000).
- `n_softctx_iter` config + loop logic (support `>= 1` full forwards).
- Build `inputs_embeds` and an update function where coord positions are replaced by `ē_t^(m) = Σ_k p_{t,k}^(m) * E_coord[k]` and iterated.
- Control whether `p_t^(m)` is stop-grad (recommended early) via config flag.
- (Optional) damping `softctx_damping_alpha` for `E_new = (1-α)E_old + α ē` to improve stability.
- (Optional) entropy / `max_prob`-based extra-iteration trigger (per-sample or per-object), still no-rollout (documented as a heuristic, not required).
- Each iteration is a **full forward** using `inputs_embeds` (NOT token-by-token generation); KV-cache reuse is an optional optimization and not required for correctness.
- Model forward must support `inputs_embeds` (Qwen3-VL does; ensure your trainer path passes it correctly).

### 10.2 Matching utilities
- Cost matrix builder:
  - geometry cost from discrete coords (fast IoU/L1)
  - optional semantic cost from frozen text embedding model
- Hungarian solver + match gating

### 10.3 CoordExp module
- Given logits `z_t`, gather coord logits, compute softmax, expectation
- Return continuous scalar in [0,1] (or [0,999] if you prefer working in index space)

### 10.4 Loss module
- `L_geo_soft`: Channel-A L1 + GIoU over continuous boxes from CoordExp under soft self-context
- `L_geo_roll`: Channel-B L1 + GIoU over continuous boxes from CoordExp under rollout self-context (matched via Hungarian)
- `L_CE`: standard token CE (with optional position-wise weights)
- optional coord-distribution regularizers (entropy, top-k expectation, softCE/W1, etc.)

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

**Update:** run Stage-2 as Channel-A/Channel-B mixture:
- Channel-A provides high-throughput, stable geometry calibration without rollout.
- Channel-B provides sparse but essential “true self-context + set correction”.

### Optional Stage-3 (light rollout consistency)
- Low-frequency additional rollouts with the same machinery.
- Focus on structure validity and geometric calibration; avoid heavy RL.

---

## 12. Minimal Working Example (Two objects: “black cat” and “yellow dog”)

### GT (JSON-only assistant payload; bbox only)
```json
{
  "object_1": {
    "desc": "black cat",
    "bbox_2d": ["<|coord_110|>", "<|coord_310|>", "<|coord_410|>", "<|coord_705|>"]
  },
  "object_2": {
    "desc": "yellow dog",
    "bbox_2d": ["<|coord_520|>", "<|coord_285|>", "<|coord_890|>", "<|coord_660|>"]
  }
}
```

### Rollout (model output; JSON-only, bbox only)
```json
{
  "object_1": {
    "desc": "black cat",
    "bbox_2d": ["<|coord_120|>", "<|coord_300|>", "<|coord_420|>", "<|coord_700|>"]
  },
  "object_2": {
    "desc": "yellow dog",
    "bbox_2d": ["<|coord_500|>", "<|coord_280|>", "<|coord_880|>", "<|coord_650|>"]
  }
}
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
