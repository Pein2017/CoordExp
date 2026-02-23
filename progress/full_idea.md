# CoordExp + EM-ish Set-Supervision for V-LLM Detection (Qwen3-VL, norm1000)

> Goal: Train a pretrained V-LLM (e.g., Qwen3-VL) to output **open-vocabulary object descriptions + norm1000 boxes** in a structured text format, while enabling **continuous geometric gradients** (SmoothL1 + CIoU) to flow back into the LM head via **CoordExp** and reducing order sensitivity via **Hungarian matching** in an **EM-ish** training loop—**without adding a DETR-style detection head**.

---

## 0. Core Design Principles

### 0.1 Constraints
- Base model: pretrained V-LLM (Qwen3-VL).
- Coordinate representation: existing `<|coord_k|>` tokens with `k ∈ {0..999}` (norm1000; 1000 bins, 0 corresponds to 0.0 and 999 corresponds to 1.0).
- Do NOT introduce a separate detection head or DETR-like query decoder.
- Training should remain compatible with standard SFT infrastructure (teacher forcing, token CE), with additional losses computed from LM logits.

### 0.2 What “EM-ish” means here
- We separate **continuous / smooth geometry calibration** from **discrete / set-level alignment**, but we intentionally do **NOT** use token-level CE to suppress extra (unmatched) predicted objects in Channel-B.
  - **Channel-A (hot path):** no autoregressive rollout; do **N× forward** (iterative soft self-context) to approximate “self-context” *only on coord slots* using soft/ST coord embeddings (differentiable, fast; Section 2.6). Channel-A remains responsible for normal SFT behaviors (including end-of-sequence / closure supervision).
  - **Channel-B (cold path):** occasional autoregressive rollout + Parse + Hungarian matching to define a stop-grad **set alignment** between predicted objects and GT objects, then run a **Unified one-pass teacher-forced forward** (rollout prefix + FN injection inside the same JSON dict) for geometry/text supervision under true self-context.

Under this view:
- **E-step (latent alignment):** performed only on Channel-B steps via rollout → parse → set → Hungarian (stop-grad correspondence). This defines `(pred_i → gt_j)` matches, FN, and FP sets.
- **M-step (parameter update):** always uses SFT-compatible objectives; geometry losses are applied under:
  - Channel-A: “soft self-context proxy” (no sampling)
  - Channel-B: “true self-context” (teacher-forced on rollout tokens), FP-neutral, with geometry gradients on **ValidMatched + FN-injected** entries and no geometry gradients on FP entries.

**Important (Channel-A gradient semantics):** Channel-A supports two modes:
- **A-UNROLL (default):** allow gradients through the soft self-context loop (no detach). This enables cross-step / cross-coord credit assignment. After Stage-1 pretraining this is typically stable.
- **A-EM (stable fallback):** detach the coord distribution used to build soft embeddings (EM-ish). This approximates on-policy context while avoiding unstable feedback gradients.
Default is `A-UNROLL`; fallback is `A-EM` when training stability requires it (Section 6.2).

### 0.3 Why “self-context forward”
Hungarian matching is computed from the model’s rollout output. To make the geometric gradient consistent with the same context that produced that rollout, we recompute logits using teacher forcing on the rollout tokens (`ŷ`) and apply GT-based geometric supervision there.

This avoids supervising coordinates under a mismatched context (pure GT teacher forcing) and behaves like a policy-improvement loop, but stays fully differentiable (no REINFORCE/PPO).

**Update (practical):** in large V-LLMs, “true self-context forward” is expensive because it requires autoregressive rollout.
We therefore introduce a **soft self-context proxy** that replaces only `<|coord_*|>` token embeddings by their expected embedding
under the model’s own coord distribution, enabling a *non-sampling* approximation that is compatible with Qwen3’s tied weights.

**Iterative variant (Channel-A):** the same soft self-context proxy can be run as an *iteration* (no rollout) to reduce “GT leakage / lag” and move closer to on-policy self-conditioning.

This iterative proxy can be run in:
- **UNROLL mode:** keep the loop differentiable for credit assignment.
- **EM mode:** detach soft context to stabilize training.

- Let `e^(m)` denote the (soft) embeddings used at coord slots in iteration `m` (text/struct tokens remain teacher-forced under GT embeddings).
- Define a mapping `e^(m+1) = F_θ(e^(m))` where `F_θ` is: one full forward under `e^(m)` → extract coord-slot distributions → form expected coord embeddings.
  - For coord slot `t`: `p_t^(m) = softmax(s_t^(m) / τ_A)` and `e_soft,t^(m) = Σ_k p_{t,k}^(m) * E_coord[k]`.
- Using `n_softctx_iter = N` total forwards corresponds to `N-1` applications of `F_θ` (Jacobi-style fixed-point intuition).
- Larger `N` is *closer* to rollout/self-context (because the coord context is repeatedly updated from the model’s own belief), but it is still not equivalent to autoregressive rollout because it is soft (expectation), non-sampled, and keeps the GT text/structure scaffold.
- In practice, bbox chains are short (4 coords). `N=2` is the default in this document.

### 0.4 Channel-B “FP-neutral” principle (central)
In real detection data, unmatched predicted objects (FP under Hungarian) may correspond to **unlabeled** true objects. Therefore Channel-B must be:
- **FP-neutral:** no CE / geometry gradients on FP object spans, and
- **EOS-enforced (NOT stop-neutral):** the model must still be trained to close the top-level container and emit EOS.

The previous stop-neutral variant is removed from this design due to observed rollout-length inflation without reliable TP-hit gains.

All loss naming and mask rules for Stage-1 / Channel-A / Channel-B are defined once in **Unified Loss Registry (Deduped)** (Section 5.5). In particular, Channel-B uses the `context=rollout` mask spec: FP-neutral on FP spans, struct-only CE on matched prefix objects, struct+desc CE on FN-injected objects, and **EOS enforced** on the outermost `}` and `<|im_end|>`.

---

## 1. Output Format / Token Protocol

### 1.1 Canonical object record format (JSON-only, dense mode)
CoordExp uses JSON-only assistant outputs (no wrapper tags like `<obj>`/`<box>`). In dense mode, the assistant emits a single top-level CoordJSON object:

```json
{"objects": [{...}, {...}]}
```

Each `objects[]` record is a JSON object with:
- `desc` (string, required): open-vocabulary description / class phrase.
- Exactly one geometry field (required, mutually exclusive):
  - `bbox_2d`: `[x1, y1, x2, y2]`
  - `poly`: flat polygon list `[x1, y1, x2, y2, ...]` with even length and >= 6 values. JSONL may include `poly_points` metadata, but assistant output does not emit metadata keys.

Coordinate representation (norm1000):
- Raw JSONL stores coord tokens as quoted strings `"<|coord_k|>"`.
- Assistant CoordJSON uses **bare** CoordTok literals `<|coord_k|>` where `k ∈ [0, 999]`.
- Internally, assistant CoordJSON is transpiled to strict JSON with integer geometry bins before matching/eval/loss.

Example (coord-token mode, bbox only):

```json
{
  "objects": [
    {
      "desc": "black cat",
      "bbox_2d": [<|coord_110|>, <|coord_310|>, <|coord_410|>, <|coord_705|>]
    },
    {
      "desc": "yellow dog",
      "bbox_2d": [<|coord_520|>, <|coord_285|>, <|coord_890|>, <|coord_660|>]
    }
  ]
}
```

Optional: geometry-first object payload order

Default remains `desc` then `bbox_2d` (desc-first).

An optional ablation order is `bbox_2d`/`poly` then `desc` (geometry-first). This can reduce early long-text noise on geometry decoding and make coordinates available earlier for parsing, while desc-first usually gives stronger early semantic anchoring in dense scenes.

Optional geometry-first example:

```json
{
  "objects": [
    {
      "bbox_2d": [<|coord_110|>, <|coord_310|>, <|coord_410|>, <|coord_705|>],
      "desc": "black cat near sofa"
    }
  ]
}
```

### 1.2 Parsing invariants
- The response MUST contain a top-level JSON object.
- The top-level object MUST contain exactly one key: `objects`.
- `objects` MUST be an array; each record MUST contain:
  - a non-empty `desc`, and
  - exactly one geometry field: `bbox_2d` or `poly`.
- Records MUST NOT contain extra keys beyond `desc` + geometry.
- Geometry arity rules (after flattening nested lists, if any):
  - `bbox_2d` must contain exactly 4 coordinates.
  - `poly` must be a flat even-length list with length >= 6.
- In CoordJSON mode, geometry values MUST be bare coord-token literals `<|coord_k|>` with `k ∈ [0, 999]`.

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
  - Closed-interval convention: `φ(0)=0.0` and `φ(999)=1.0`.

Expectation:
- `ĉ_t = Σ_k p_{t,k} * φ(k)`

For an object `i`, apply to 4 positions:
- `b̂_i = (x̂1_i, ŷ1_i, x̂2_i, ŷ2_i)` in `[0,1]^4`

### 2.4 Key gradient identity (for documentation/analysis)
- `∂ĉ_t/∂s_{t,k} = (1/τ) * p_{t,k} * (φ(k) - ĉ_t)`
So geometric losses on `ĉ_t` backprop to coord logits smoothly.

### 2.5 Future extensions (off by default)
- Temperature scheduling and distribution-shape controls (e.g., entropy/top-k) are future extensions; default pipeline keeps this path disabled.

### 2.6 Straight-Through (ST) Bridge for Discrete Tokens

CoordExp (Section 2.3) and Channel-A soft self-context (Section 6.2) are differentiable but can introduce a **soft-vs-hard mismatch**:
- Inference and strict parsing operate on **hard discrete coord tokens**.
- Training often uses **soft expected coord embeddings** and **expectation-decoded boxes**.

We therefore introduce a **Straight-Through (ST) bridge** that enables **hard forward / soft backward** behavior at two insertion points:
1) **Coord-slot context embeddings** used for self-context (`coord_ctx_embed_mode`, Channel-A Step A2).
2) **Coord decode** used for geometry loss (`coord_decode_mode`, Channel-A Step A4 and Channel-B geometry decode).

Throughout this section, `stopgrad(·)` denotes stop-gradient / detach.

#### 2.6.1 ST identity (general)
Define:
- `y = a + (b - stopgrad(b))`

Then:
- Forward: `y = a`
- Backward: `∇y = ∇b`

#### 2.6.2 ST-Embedding (coord-slot self-context)
Given coord-subspace logits `s` and temperature `τ`, form a distribution:
- `p = softmax(s / τ)`
- `k* = argmax_k p_k`

Let `E[k]` be the embedding table row for `<|coord_k|>`:
- `e_soft = Σ_k p_k E[k]`
- `e_hard = E[k*]`
- `e_st = e_hard + (e_soft - stopgrad(e_soft))`

Interpretation:
- forward uses `e_hard` (tokenization/inference-consistent),
- backward uses the gradient of `e_soft` (differentiable).

#### 2.6.3 ST-Coord decode (geometry loss on “inference boxes”)
Let coord values be `v_k = k/999 ∈ [0,1]`. Using the same `p` and `k*`:
- `x_soft = Σ_k p_k v_k`  (CoordExp expectation, Section 2.3)
- `x_hard = v_{k*}`       (hard argmax decode)
- `x_st = x_hard + (x_soft - stopgrad(x_soft))`

Interpretation:
- forward evaluates geometry on a hard-decoded coordinate (`x_hard`),
- backward uses the CoordExp expectation gradient via `x_soft` (Section 2.4).

#### 2.6.4 Relationship to `softctx_grad_mode` (unroll vs detach)
ST and unroll address different issues and are intentionally orthogonal:
- **ST (`coord_ctx_embed_mode=st`, `coord_decode_mode=st`)** reduces hard/soft mismatch by using hard values in the forward path while keeping soft gradients.
- **UNROLL (`softctx_grad_mode=unroll`)** controls whether gradients are allowed to flow through the self-conditioning loop, enabling credit assignment across coord slots / iterations; `em_detach` is a stability fallback when that coupling is too strong.

---

## 3. Dataset / Supervision Format

### 3.1 Data sources and tasks

We target **open-vocabulary detection and grounding** on top of existing public datasets:

- Detection / instance segmentation: COCO, LVIS, Objects365, etc.
- Referring expressions / grounding: RefCOCO, RefCOCO+, RefCOCOg, etc.

All sources are converted into a **shared JSONL contract** defined in `docs/data/JSONL_CONTRACT.md`. Conceptually, each record corresponds to one or more images plus a set of annotated objects.

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
  - **Channel-B (sparse):** rollout + strict parse + Hungarian + FN injection into the same JSON dict + one-pass teacher-forced update for set-level correction under true self-context.

---

## 5. Stage-1: Standard SFT to “learn the language of boxes”

### 5.1 Inputs
- image + prompt (task instruction)
- GT token sequence `y_GT` constructed by concatenating all GT objects in a chosen canonical order (e.g., left-to-right, top-to-bottom; any fixed rule is fine for Stage-1)

### 5.2 Forward
Teacher forcing on GT sequence:
- logits at each step `z_t = f_θ(image, prompt, y_<t)`

### 5.3 Loss
Stage-1 is pure GT teacher forcing (`context=gt`), and uses the **Unified Loss Registry** (Section 5.5).

Enabled components (typical):
- Token CE on GT:
  - `L_struct_ce + L_desc_ce` (includes EOS `<|im_end|>` under `L_struct_ce`).
  - Optional: `L_coord_token_ce` (hard CE on GT coord tokens), optionally upweighted (coord-focused CE).
- Optional (light): `L_coord_dist` on GT coord positions (pick one mode; default off in this document).
- Geometry: `L_geo` is typically `0` (or very small) in Stage-1; Stage-2 is the intended geometry calibration stage.

Stage-1 total (registry view):
- `L_stage1 = L_struct_ce + L_desc_ce + λ_coord_ce * L_coord_token_ce + λ_coord_dist * L_coord_dist`  (all masks from `context=gt`).

### 5.4 Purpose
- Make the model reliably output the structured format.
- Make coord token prediction stable and aligned with norm1000 vocabulary.
- Build a baseline that already works with greedy decoding.

---

## 5.5 Unified Loss Registry (Deduped)

This section is the **single source of truth** for:
- Loss component names + definitions (Stage-1 / Stage-2 Channel-A / Channel-B).
- Mask/weight rules (token-type × object-subset × context).
- Non-gradient policies that affect assignment (Hungarian/gating), but are not backpropagated.

All later sections should **only reference** this registry (no re-definitions of SmoothL1/CIoU, CoordExp decode, or CE mask rules elsewhere).

### 5.5.1 Token types, object subsets, contexts

**Token types** (mutually exclusive; assigned via GT template metadata or strict parse spans):
- `struct`: JSON syntax + keys/punctuation (e.g., `{ } [ ] : ,`, quotes, keys like `"objects"`, `"desc"`, `"bbox_2d"`, and whitespace), excluding `desc` content and coord tokens.
- `desc`: the free-text tokens inside an object’s `desc` string value.
- `coord`: coord vocabulary tokens `<|coord_k|>` for `k ∈ [0,999]`.
- `eos`: end token `<|im_end|>` (Qwen3-VL). `eos` is a **separate token type** (not `struct`): enforce `w_struct(t)=0` on `<|im_end|>` and use `w_eos(t)` for EOS supervision (counted inside `L_struct_ce`).

**Object subsets** (Channel-B only; stop-grad from rollout → parse → Hungarian(+gating)):
- `matched`: predicted objects accepted into `ValidMatched`.
- `fp`: predicted objects in the rollout prefix not in `ValidMatched` (plus dropped-invalid spans treated as FP for masking).
- `fn`: GT objects not matched; rendered and injected into `y_in` inside `{"objects":[...]}`.

**Context types** (where logits/targets come from):
1) `gt` (GT context): pure teacher forcing on the GT sequence `y_GT` (Stage-1; also the CE anchor in Channel-A).
2) `self_context` (soft/st self-context): Channel-A final-iteration logits under coord-slot self-conditioning (no rollout).
3) `rollout` (rollout context): Channel-B one-pass teacher forcing on `y_in` = rollout prefix + FN injection.

### 5.5.2 Loss components (backprop; unified names)

Let per-token cross entropy be `CE_t = -log p_θ(y_t | ·)`. Let `w_type(t) ∈ [0,1]` be masks/weights defined in Section 5.5.4.

#### (1) Token CE losses
- `L_struct_ce`: structure-token CE **including EOS**:
  - `L_struct_ce = Σ_t (w_struct(t) + w_eos(t)) * CE_t`
  - **Hard constraint (no EOS double counting):** token types are mutually exclusive; for the EOS token `<|im_end|>`, force `w_struct(t)=0` and use only `w_eos(t)=1`.
- `L_desc_ce`: description-token CE:
  - `L_desc_ce = Σ_t w_desc(t) * CE_t`
- `L_coord_token_ce` (optional): hard CE on coord tokens (GT bin):
  - `L_coord_token_ce = Σ_t w_coord(t) * CE_t`

#### (2) Coordinate distribution / ordinal losses (optional; do not blindly stack)
`L_coord_dist` is computed from coord-subspace probabilities `p_{t,k}` (Section 2.2) at selected coord positions.

Choose **one** primary mode (and optionally one stabilizer):
- **Option A (recommended default when enabling): DFL / soft-label CE**
  - Given continuous GT coord `c_gt ∈ [0,1]`, let `u = 999*c_gt`, `g0=floor(u)`, `g1=min(g0+1,999)`, `α=u-g0`.
  - Soft target: `q[g0]=1-α`, `q[g1]=α`, else 0.
  - `L_coord_dist_A = Σ_t Σ_k q_{t,k} * (-log p_{t,k})`
- **Option B (heavy / hyperparam-sensitive): W1/EMD or cost-sensitive CE**
  - Example (expected absolute error): `Σ_t Σ_k p_{t,k} * |v_k - c_gt|`, where `v_k = k/999`.
- **Option C (stability only): entropy penalty / τ anneal**
  - Use only as a stabilizer; do not mix A+B+C without evidence.

#### (3) Geometry regression (bbox; coupled four-coord supervision)
`L_geo` is box-level geometry loss computed on decoded continuous boxes:

1) **Decode per-coordinate continuous values** from coord logits using `coord_decode_mode`:
   - `exp`: CoordExp expectation `ĉ` (Section 2.3).
   - `st`: ST coord decode `x_st` (Section 2.6.3; hard forward + soft grad).
2) **Assemble boxes** `b_pred = (x1,y1,x2,y2)` and canonicalize (mandatory; prevents NaNs):
   - `x_lo = min(x1, x2)`, `x_hi = max(x1, x2)`
   - `y_lo = min(y1, y2)`, `y_hi = max(y1, y2)`
   - enforce non-zero size: `x_hi = max(x_hi, x_lo + eps)`, `y_hi = max(y_hi, y_lo + eps)`
3) **Apply SmoothL1 + CIoU**:
   - `L_geo = Σ_obj [ λ_huber * SmoothL1(b_pred - b_gt; δ) + λ_ciou * CIoU(b_pred, b_gt) ]`

### 5.5.3 Policy / alignment rules (no backprop; gradient hygiene)
- `coord_gate`: geometric match gating (cost/IoU thresholds) applied after Hungarian; determines `ValidMatched`, `FP`, `FN`.
- `text_gate` (optional): semantic gate/cost term for matching; off by default.

These are stop-grad policies: they affect sample allocation/subsets but do not backprop.

### 5.5.4 Unified Mask / Weight Spec (very important; defined only here)

We define token-type masks/weights per context. Implementation-wise you can build separate per-type masks and then sum them into the loss components above.

#### (A) GT context (`context=gt`): Stage-1 and Channel-A CE anchor
- All GT tokens are supervised normally, by token type (mutually exclusive):
  - `w_struct(t)=1` on `struct` tokens; `w_desc(t)=1` on `desc` tokens; `w_eos(t)=1` on `<|im_end|>` (and `w_struct(t)=0` on EOS).
  - `w_coord(t)=1` if using `L_coord_token_ce`, else 0
- Optional coord upweight (coord-focused CE): multiply `w_coord` by a constant; do not introduce a separate loss name for this.

#### (B) self_context (`context=self_context`): Channel-A geometry/dist path
- Token CE is **not** taken from self_context logits (CE anchor stays in GT context).
- `L_geo` and (optional) `L_coord_dist` are computed from the final-iteration logits at GT coord positions, identity-aligned to GT objects (no matching).

#### (C) rollout (`context=rollout`): Channel-B one-pass (FP-neutral + EOS-enforced)
Let `ValidMatched`, `FP`, `FN` come from rollout→parse→Hungarian(+gating). Let `span(obj)` denote token spans for objects in `y_in` (prefix or injected).

**FP-neutral (hard rule):**
- For `obj ∈ FP`: set all token-type masks inside `span(obj)` to 0, and exclude that object from `L_geo` / `L_coord_dist`.

**Matched prefix objects (accepted matches):**
- For `obj ∈ ValidMatched`, inside `span(obj)`:
  - `w_struct=1`, `w_desc=0`, `w_coord=0`

**FN injected objects:**
- For injected `obj ∈ FN`, inside `span(obj)`:
  - `w_struct=1`, `w_desc=1`, `w_coord=0` (default)

**EOS-enforced (this document’s requirement; Channel-B is NOT stop-neutral):**
- Top-level closure is `]}`: keep `w_struct=1` on both the top-level array close `]` and the **outermost top-level** JSON closing brace `}` in `y_in`.
- Keep `w_eos=1` on the EOS token `<|im_end|>` in `y_in` (Channel-B must ensure `y_in` ends with `<|im_end|>`; Step B3).
- FP-neutral wins: do **not** override FP-span zeros. If these tokens fall inside an FP span (rare), they remain masked 0.
- The outermost `}` must be located by a brace-stack scan (ignoring braces inside quoted strings); never by “last `}` token” heuristics.

This spec keeps Channel-B FP-neutral while still forcing proper closure/termination (prevents infinite-rollout length inflation).

### 5.5.5 Legacy name mapping (one place only)

This document previously used multiple names for the same concepts. The unified registry names above are canonical; the mapping is:

- `L_CE_all` / `L_CE_A1` / `L_CE_struct` / `L_CE_desc_FN` → `L_struct_ce` + `L_desc_ce` (+ optional `L_coord_token_ce`) with context-specific masks.
- `L_CE_coord` → coord-token weight multiplier inside `L_coord_token_ce` (GT context).
- `L_E-L1_coord` / `L_dist` → `L_coord_dist` (Option B for expected-L1; Option A for DFL).
- `L_geo_soft` / `L_geo_matched` / `L_geo_FN` / `L_geo_total` → `L_geo` instantiated on different contexts/subsets:
  - Channel-A self_context (identity-aligned GT objects),
  - Channel-B rollout (ValidMatched + FN injected, FP excluded).

---

## 6. Stage-2: EM-ish Training Loop (Rollout → Match → Update)

### 6.1 Two-channel schedule (recommended)
We run Stage-2 as a **mixture** over steps / samples:
- **Channel-A (hot path, e.g., 95% steps):** no rollout; always available and fast.
- **Channel-B (cold path, e.g., 5% steps or 1/P batches):** true rollout + Hungarian + FN injection + one-pass teacher-forced update to fix discrete/set-level failure modes.

Rationale:
- In practice, “hard CE on coord tokens” is already strong in Stage-1/SFT, so Stage-2 should not assume coord-distribution losses will dominate.
- The main remaining instability is typically **self-context generation** (long JSON, permutation, missing/extras/format).
  Channel-B targets that directly; Channel-A provides a cheap, smooth geometry signal to stabilize coord decoding and reduce drift.

Default Stage-2 settings:
- Channel-A: `n_softctx_iter=2`, `softctx_grad_mode=unroll`, `coord_ctx_embed_mode=st` (default), `coord_decode_mode=exp` (default; can optionally transition to `st` once coord distributions are sufficiently sharp).
- Channel-A fallback when unstable: `softctx_grad_mode=em_detach`.
- Channel-B: Unified one-pass update (rollout prefix + FN injection + single teacher-forced forward), with `context=rollout` masks from the Unified Loss Registry (FP-neutral + EOS-enforced).

---

### 6.1.1 Implementation Form: Strict Step-Level Routing (方案 A) + Async Actor-Learner
Implementation is **step-level routing** with strict separation per optimizer update:
- **Routing granularity:** A/B are chosen per optimizer step (one accumulation window). A step is **all A** or **all B**; never mix A/B microbatches within the same optimizer update.
- **Data sources:** Channel-A uses the main dataloader (hot path, always available). Channel-B consumes from a `rollout_queue` (cold path), optionally via a `b_ready_queue` after preprocess.
- **Async actor-learner (optional):** actor rolls out with parameters `θ_{t-Δ}`, learner updates `θ_t`. Each rollout item carries `ver`; learner only consumes items with `ver >= current_ver - version_window`. Use `queue_limit` to bound off-policy gap and memory.
- **DDP safety:** step kind must be identical across all ranks; rank0 decides and broadcasts step_kind to avoid mismatched collectives or hang.
- **Gradient accumulation:** step_kind is locked for the entire accumulation window (`grad_accum_steps`); it cannot change mid-window.
- **Routing rule:** if `B_queue` has enough **fresh** items to cover the full optimizer update (`batch_size_B` across all microbatches), run a **B-step**; otherwise run an **A-step**. A is the fallback when B is insufficient.

This is the enforced implementation form for Stage-2 (方案 A).

### 6.2 Channel-A (hot): Iterative soft self-context via N× forward (no rollout)

Key idea: approximate “self-context” only where it matters most (coord slots), without sampling.
We keep all **text/struct** tokens teacher-forced as GT, but feed **coord slots** as *soft or ST context embeddings*
derived from the model’s own coord distribution.

Multi-object note (intended behavior):
- For sequences with multiple objects, we run soft self-context on the full sequence under the standard causal mask (no object-wise isolation).
- Later objects are allowed to condition on earlier objects’ coord slots (this matches inference for a single JSON sequence).
- With `softctx_grad_mode=unroll`, geometry losses from later objects may backprop through soft coord embeddings into earlier coord slots (cross-object credit assignment). This coupling is expected; if unstable, switch to `em_detach`.

Definitions / hyperparameters:

- `softctx_grad_mode` : gradient semantics for the soft self-context loop.
  - `unroll` (default): keep the loop differentiable (no detach), enabling credit assignment across coord slots / iterations.
  - `em_detach` (fallback): detach the coord distribution used to build soft embeddings (EM-ish) if training is unstable.

- `n_softctx_iter` : total number of full-sequence forwards in Channel-A (`>= 1`).
  - default: `n_softctx_iter=2`.
  - fallback: `n_softctx_iter=1` (pure teacher forcing) when needed for stability/debug.
- Iteration indexing: we use `m = 0..n_softctx_iter-1`, where `m=0` corresponds to the teacher-forced GT forward in Step A1. The **final iteration** logits used for CoordExp decode and geometric loss are `z^(n_softctx_iter-1)`.

- `coord_ctx_embed_mode` : how to construct coord-slot **context embeddings** in Step A2.
  - `soft`: expected embedding `e_soft = Σ_k p_k E[k]` (original behavior).
  - `st`: ST-Embedding `e_st` (Section 2.6.2; default for Channel-A).
  - `hard`: argmax embedding `E[argmax p]` (debug/inference only; not recommended for training).

- `coord_decode_mode` : how to decode coords for geometry loss.
  - `exp`: CoordExp expectation `ĉ` (Section 2.3; default).
  - `st`: ST coord decode `x_st` (Section 2.6.3; hard forward + soft grad).

#### Step A1 — Teacher-forced GT forward (with grad)
1) Teacher force on GT to get logits:
   - `z_t^gt = f_θ(image, prompt, y_GT_<t)`

2) Identify coord positions `t ∈ coord_positions` from labels/template (packing-safe). For each coord position `t`:
   - gather coord-subspace logits and compute `p_t^(0)` as in Section 2.2 (using Channel-A temperature `τ_A`)

> Note (gradient semantics):
> - Default: `softctx_grad_mode=unroll`, use `p_t^(m,ctx) = p_t^(m)`.
> - Fallback: `softctx_grad_mode=em_detach`, use `p_t^(m,ctx) = stopgrad(p_t^(m))`.

#### Step A2 — Build coord-slot context embeddings (soft or ST; no sampling)
3) Let `E_coord ∈ R^{K×d}` be the embedding table restricted to `<|coord_k|>`.
   First construct the expected (soft) embedding at each coord slot:
   - `e_soft,t^(m) = Σ_k p_{t,k}^(m,ctx) * E_coord[k]`  (use `p^(m,ctx)` per `softctx_grad_mode`)

   Then choose the actual context embedding `e_ctx` per `coord_ctx_embed_mode`:
   - `soft`: `e_ctx = e_soft`
   - `st`: `e_ctx = ST_Embed(p^(m,ctx), E_coord)` (Section 2.6.2)
   - `hard`: `e_ctx = E_coord[argmax_k p_{t,k}^(m,ctx)]`

4) Prepare the initial embedding tensor (full sequence) for iterative soft self-context:
   - Start from standard embeddings of `y_GT` (or from the model’s embedding lookup): `emb^0`.
   - For coord positions, you have two valid initializations:
     - (B) **Start from `e_ctx^(0)` (recommended default):** replace coord slots in `emb^0` by `e_ctx,t^(0)` before the first softctx forward. This makes `n_softctx_iter=2` actually run one self-context forward for the geometry loss.
     - (A) Start from GT (stable fallback): keep coord slots as GT in `emb^0`, and only start replacing after the first update. Note this effectively shifts the soft-context effect by one iteration (e.g., `n_softctx_iter=2` becomes almost pure teacher forcing).
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
Compute p^(0,ctx) from p^(0) per softctx_grad_mode
Compute e_ctx^(0) from p^(0,ctx) per coord_ctx_embed_mode

emb^0 := Embed(y_GT)               # full GT embeddings
emb^0 := UpdateCoordSlots(emb^0, e_ctx^(0))  # default init: start-from-e_ctx^(0)
Optional (fallback): keep GT coord slots for the first softctx forward (init=A); this shifts the effect by one iteration

for m in 1 .. (n_softctx_iter - 1):
    z^(m) := f_θ(image, prompt, inputs_embeds=emb^(m-1))   # full forward
    Compute p^(m) from z^(m) at coord_positions
    Compute p^(m,ctx) from p^(m) per softctx_grad_mode
    Compute e_ctx^(m) from p^(m,ctx) per coord_ctx_embed_mode
    emb^(m) := UpdateCoordSlots(emb^(m-1), e_ctx^(m))

Use final logits z^(n_softctx_iter-1) for coord distributions + losses (coord decode per Section 2; `L_geo` per Section 5.5)
```

Interpretation: this is a fixed-point iteration on coord-slot context, `e^(m+1)=F_θ(e^(m))`, that repeatedly refreshes coord embeddings from the model’s own belief while keeping the GT text/structure scaffold. Increasing `n_softctx_iter` pushes the effect of self-conditioning further along the coord chain without sampling.

#### Step A4 — Geometry loss
6) Decode continuous coords from the **final** logits `z^(n_softctx_iter-1)` using `coord_decode_mode` (Section 2.3 / Section 2.6.3), and assemble boxes per object.
7) Apply geometry loss against GT boxes **by identity alignment** (no Hungarian in Channel-A) using `L_geo` from the Unified Loss Registry (Section 5.5; includes canonicalization + SmoothL1+CIoU).
8) Optional (use sparingly): apply `L_coord_dist` on coord distributions if evidence requires it (Section 5.5.2).

Channel-A per-sample objective (registry view):
- CE anchor (GT context, Step A1): `L_struct_ce + L_desc_ce (+ optional L_coord_token_ce)` with `context=gt` masks.
- Geometry/dist (self_context, final logits): `λ_geo^A * L_geo (+ optional λ_coord_dist^A * L_coord_dist)` with `context=self_context`.

> Why this is “unified” with rollout-matching:
> - Channel-A trains geometry under a form of self-conditioning (but without sampling).
> - Channel-B trains under true self-conditioning (rollout tokens), but sparsely.
> Together they approximate “train on what you infer” while keeping throughput high.

Why `N×` helps (intuition, bbox case):
- View `N×` as **propagation depth** along the bbox coord chain under a Jacobi-style fixed-point iteration.
- For bbox = `(x1 → y1 → x2 → y2)` (4-step chain), the soft self-context effect propagates by ~1 step per extra iteration:
  - `N=1`: pure teacher forcing (control).
  - `N=2`: `y1` sees a soft `x1` context (model-derived).
  - `N=3`: `x2` sees a context where `x1/y1` are more on-policy (soft-refreshed).
  - `N=4`: `y2` sees a context where `x1/y1/x2` have been soft-refreshed.
- This targets “GT leakage / lag” without autoregressive sampling; stability/credit assignment depends on `softctx_grad_mode` (`unroll` vs `em_detach`).
- Note: this interpretation assumes the default init (start-from-`e_ctx^(0)`). If you use init=A (start-from-GT), the self-context effect is delayed by one iteration (so you may need `N+1` to get the same propagation depth).

---

## 7. Stage-2 Updates (M-step): Two complementary supervision paths

Stage-2 consists of **two complementary paths** (Channel-A hot path + Channel-B cold path) that target different failure modes:

### 7.1 Channel-A (hot path): Geometric calibration under soft/ST self-context (box-level L_geo + optional L_coord_dist)

Channel-A uses the **N× forward** construction from Section 6.2 and instantiates the **Unified Loss Registry** (Section 5.5) under two contexts:
- `context=gt` (CE anchor, Step A1): compute `L_struct_ce + L_desc_ce (+ optional L_coord_token_ce)` on the teacher-forced GT forward.
- `context=self_context` (geometry, final iter): from the final-iteration logits `z^(n_softctx_iter-1)`, decode coords per `coord_decode_mode` (Section 2.3 / Section 2.6.3) and apply `L_geo` by identity alignment (no Hungarian in Channel-A).
- Optional (rare): add a small `L_coord_dist` on final-iteration coord distributions if evidence requires it (Section 5.5.2; do not blindly stack modes).

> Key properties:
> - Context is **partially self-conditioned** (coord slots only; GT text/struct scaffold stays teacher-forced).
> - Supervision target is GT geometry; identity alignment is exact in Channel-A.
> - No sampling; throughput-friendly. Use `coord_ctx_embed_mode=st` to reduce soft-vs-hard mismatch.

---

### 7.2 Channel-B (cold path): Rollout-matching for set-level / discrete correction

Channel-B is a single Unified one-pass pipeline, run sparsely to correct set-level errors under true rollout context.

#### Step B0 — Rollout (E-step hypothesis, no grad)
1) Run autoregressive generation using current model θ:
   - `ŷ_full = Rollout(f_θ, image, prompt)` (greedy / low-T sampling / beam)

#### Step B1 — Strict parse + validate (no grad)
2) Parse `ŷ_full` into candidate objects and apply strict instance-level validation (drop invalid objects; do not repair).
3) Keep valid predicted objects:
   - `Ô = { (d̂_i, b̂_i_disc, span_i, desc_span_i, coord_pos_i) }`
   - `span_i`: token span of object `i` in prefix (for FP full-mask).
   - `desc_span_i`: token span of `desc` for object `i` (so matched objects can keep desc tokens masked while keeping struct tokens supervised; Section 5.5.4, `context=rollout`).
   - `coord_pos_i`: coord input/logit index mapping for object `i`.
   - Tokens belonging to dropped-invalid instances are treated as FP spans for loss masking (`loss=0`).

#### Step B2 — Hungarian + gating (no grad)
4) Build cost matrix on valid predicted objects vs GT objects, run Hungarian, apply gating, and build:
   - `ValidMatched` (accepted matched predicted objects),
   - `FP` (valid predicted objects not in `ValidMatched`),
   - `FN` (GT objects not matched).

#### Step B3 — Build `y_in` by FN append inside `{"objects": [...]}` (BUG-1 + BUG-2 fix)
5) Construct the one-pass teacher-forced target sequence `y_in` from `ŷ_full`:
   - Do **not** construct `y_in` as `ŷ_full + RenderGT(FN)`: rollout usually already contains top-level closures and `<|im_end|>`, so direct concatenation places FN entries outside the container.
   - Suffix-trim rollout text to an append-ready prefix that ends inside the `objects` array (`{"objects": [` or `{"objects": [{...}`).
   - Append FN records as additional array elements in GT canonical order.
   - If prefix already contains at least one retained object, prepend `, ` before the first FN record.
   - Close container with `]}` and preserve suffix tokens after container closure (including `<|im_end|>`), which remain supervised by CE.
   - **Hard rule (EOS presence):** if the constructed `y_in` does **not** end with `<|im_end|>`, explicitly append `<|im_end|>` (EOS-enforced remains on).

#### Step B4 — One-pass teacher-forced forward + loss
7) Single forward on the constructed sequence:
   - `z^B = f_θ(image, prompt, y_in_<t)`

8) Build token-type CE masks per the **Unified Loss Registry** (Section 5.5), `context=rollout`:
- FP-neutral on FP object spans (all token losses 0; FP excluded from geometry/dist).
- Matched prefix objects: struct-only token CE; desc/coord are masked.
- FN injected objects: struct+desc token CE; coord is masked by default.
- EOS-enforced (NOT stop-neutral): keep CE=1 on the top-level closure `]}` (struct) and on the `<|im_end|>` token in `y_in` (FP-neutral wins if there is a conflict).

9) Geometry losses from the same `z^B`:
- Decode coords per `coord_decode_mode` (Section 2.3 / Section 2.6.3).
- Apply `L_geo` (Section 5.5) on `ValidMatched` and FN-injected objects (FP excluded).
- Optional strategy: FN-focused Channel-B (downweight matched geometry) is allowed but not required by this design.

10) Channel-B objective (registry view):
- `L_B = (L_struct_ce + L_desc_ce) + λ_geo^B * L_geo` with `context=rollout` masks/subsets.

Concise pseudocode (Unified Channel-B):

```python
def channel_b_unified(sample, model):
    y_hat_full = rollout(sample, model)  # no grad
    parsed = strict_parse_and_validate(y_hat_full)
    valid_matched, fp_ids, fn_ids = hungarian_with_gating(parsed, sample.gt_objects)

    y_prefix = trim_to_append_ready_objects_prefix(y_hat_full)  # ends inside {"objects": [...]
    fn_entries = render_fn_array_entries(sample.gt_objects, fn_ids, order="gt_canonical")
    comma = ", " if prefix_has_any_object_entry(y_prefix) and fn_entries else ""
    y_in = y_prefix + comma + fn_entries + "]}"
    if not y_in.endswith("<|im_end|>"):
        y_in = y_in + "<|im_end|>"

    ce_mask = zeros_like(y_in)  # default all-off
    ce_mask = apply_matched_mask(ce_mask, parsed, struct_on=True, desc_on=False, coord_on=False)
    ce_mask = apply_fp_full_zero_mask(ce_mask, parsed, fp_ids)  # Strategy A
    ce_mask = apply_fn_mask(ce_mask, y_in, fn_entries, struct_on=True, desc_on=True, coord_on=False)
    ce_mask = apply_eos_enforced_mask(ce_mask, y_in)  # EOS-enforced (not stop-neutral); do not override FP-span zeros

    logits = teacher_forced_forward(model, sample, y_in)
    L_struct_ce, L_desc_ce = compute_ce_terms(logits, y_in, ce_mask)
    geo_matched = geo_from_coord_logits(logits, parsed, valid_matched, source="prefix", coord_decode_mode=coord_decode_mode)
    geo_fn = geo_from_coord_logits(logits, y_in, fn_entries, source="injected_fn", coord_decode_mode=coord_decode_mode)
    L_geo = mean_or_zero(geo_matched) + mean_or_zero(geo_fn)
    return L_struct_ce + L_desc_ce + lambda_geo_B * L_geo
```

---

## 8. Stage-2 Total Objective

Stage-2 uses the same **Unified Loss Registry** (Section 5.5) instantiated under three contexts.

### 8.1 Contexts (registry instantiation)

1) **GT context** (`context=gt`; pure GT teacher forcing; Stage-1 style)
- Enabled: `L_struct_ce + L_desc_ce + (optional L_coord_token_ce or L_coord_dist)`.
- Geometry: `L_geo` is typically off (or very small) here.
- Masks: `context=gt` (Section 5.5.4A).

2) **self_context** (`context=self_context`; Channel-A N× self-context proxy; no rollout)
- CE anchor stays in GT context (Step A1): `L_struct_ce + L_desc_ce (+ optional L_coord_token_ce)`.
- Geometry comes from the final self_context logits: `λ_geo^A * L_geo`.
- Optional: `λ_coord_dist^A * L_coord_dist` (use sparingly).
- Masks: CE uses `context=gt`; geo/dist use `context=self_context` (Section 5.5.4B).

3) **rollout context** (`context=rollout`; Channel-B rollout prefix + FN injection; true self-context)
- FP-neutral: FP spans are fully masked; FP excluded from `L_geo` / `L_coord_dist`.
- Matched prefix: struct-only CE.
- FN injected: struct+desc CE.
- **EOS-enforced (NOT stop-neutral):** outermost `}` and `<|im_end|>` must be supervised.
- Enabled: `(L_struct_ce + L_desc_ce) + λ_geo^B * L_geo` (desc is effectively FN-only via masks).
- Masks: `context=rollout` (Section 5.5.4C).

### 8.2 Mixture (step-level routing)

- Stage-2 is a mixture objective:
  - `L_stage2 = (1-ρ) * L_A + ρ * L_B`
  - where `ρ` is the Channel-B frequency/probability (small, e.g. 0.05).
  - **Implementation note:** in code, the mixture is approximated by the step-level router, so the realized `ρ_hat` is determined by queue availability and routing decisions.

Expanded (registry view):
- `L_A = (L_struct_ce + L_desc_ce + λ_coord_ce * L_coord_token_ce)_{context=gt} + λ_geo^A * L_geo_{context=self_context} + λ_coord_dist^A * L_coord_dist_{context=self_context}`
- `L_B = (L_struct_ce + L_desc_ce)_{context=rollout} + λ_geo^B * L_geo_{context=rollout}`

Equivalently:
- `L_stage2 = (1-ρ) * L(self-context + GT anchor) + ρ * L(rollout)`

Optional strategy (allowed, not required): **FN-focused Channel-B**
- Downweight the `ValidMatched` contribution inside `L_geo_{context=rollout}` (FN geometry/text stays on). This can reduce overfitting to noisy matches early.

---

## 9. Inference Pipeline (Deployment)

Given image + prompt:
1) Autoregressively generate sequence `y_pred` (greedy/beam).
2) Parse objects:
   - parse CoordJSON `{"objects": [...]}` and read each record `desc`
   - parse `bbox_2d` as 4 bare coord tokens `<|coord_k|>` (x1,y1,x2,y2)
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
- Ability to record object spans and coord logit indices during strict parse on rollout prefix (`y_prefix`) with explicit input-position vs logit-position mapping (causal-shift aware).
- Ability to keep an append-ready `{"objects": [` prefix and rebuild `y_in` by appending FN records as array elements.
- Ability to map each object to coord logit positions (shift-aware) for both prefix parsed objects and FN injected objects.
- Loss-mask builder that implements the Unified Loss Registry `context=rollout` masks (Section 5.5.4C): FP-neutral on FP spans and **EOS-enforced** on the outermost `}` and `<|im_end|>`.

### 10.1.1 (New) Soft self-context utilities (Channel-A)
- A packing-safe way to get `coord_positions` (from labels/template).
- `E_coord` gather: embedding rows for coord token IDs (K=1000).
- `n_softctx_iter` config + loop logic (support `>= 1` full forwards).
- Build `inputs_embeds` and an update function where coord input positions are replaced by `e_ctx` computed from `p^(m,ctx)` and `E_coord` per `coord_ctx_embed_mode` (`soft` uses `Σ pE`; `st` uses ST-Embedding, Section 2.6.2).
- Control gradient semantics via `softctx_grad_mode` (unroll default, `em_detach` fallback).
- Terminology note: "coord-slot input embeddings" means the per-position rows of `inputs_embeds` for positions whose *input token* is a coord token (`<|coord_k|>`). This is distinct from `E_coord` (the coord-token embedding table). In code you may want to track both:
  - coord *input* positions (for embedding replacement), and
  - coord *output/logit* positions (for reading coord distributions and computing losses),
  because many LM implementations use a 1-token shift between inputs and the logits that predict the next token.
- Each iteration is a **full forward** using `inputs_embeds` (NOT token-by-token generation); KV-cache reuse is an optional optimization and not required for correctness.
- Model forward must support `inputs_embeds` (Qwen3-VL does; ensure your trainer path passes it correctly).

### 10.1.2 Unified Channel-B construction utilities
- Prefix scanner that returns an append-ready cut inside top-level `{"objects": [...]}`.
- Comma insertion logic for FN append (`insert ', ' iff prefix already has retained entries and FN is non-empty`).
- Span/index exporter for matched prefix objects, FP objects, and injected FN objects.

### 10.2 Matching utilities
- Cost matrix builder:
  - geometry cost from discrete coords (fast IoU/L1)
  - semantic cost from frozen text embedding model is a future extension (off by default)
- Hungarian solver + match gating

### 10.3 CoordExp module
- Given logits `z_t`, compute coord-subspace distribution (Section 2.2) and decode per `coord_decode_mode`:
  - `exp`: expectation `ĉ` (Section 2.3)
  - `st`: ST decode `x_st` (Section 2.6.3)

### 10.4 Loss module
- Implement Unified Loss Registry components (Section 5.5):
  - Token CE: `L_struct_ce`, `L_desc_ce`, optional `L_coord_token_ce` (plus token-type masks/weights).
  - Optional coord distribution supervision: `L_coord_dist` (pick one mode at a time; Section 5.5.2).
  - Geometry: `L_geo` (canonicalize + SmoothL1+CIoU; Section 5.5.2).
- Channel-specific instantiation:
  - Channel-A: `L_geo` from `context=self_context` logits, identity-aligned to GT objects.
  - Channel-B: `L_geo` from `context=rollout` logits on `ValidMatched` + FN-injected objects (FP excluded).
- Future extension (off by default): `poly` geometry support.

### 10.5 Step Router / Scheduler (方案 A)
- step_kind broadcast across ranks (rank0 decides, all ranks follow)
- grad_accum window must lock a single step_kind
- B_queue freshness filter and drop policy for stale items

### 10.6 Rollout Queue + Versioning
- actor weight sync policy (every K steps or drift-trigger)
- `ver` written into each rollout item
- `queue_limit` enforced; drop oldest when full

### 10.7 Channel-B Preprocess Worker
- parse rollout -> valid objects
- Hungarian + gating -> matches / ValidMatched
- build sets -> `ValidMatched / FP / FN`
- build append-ready prefix + append FN entries into top-level `objects[]` (with comma rules)
- export span/index metadata for matched-prefix coords, FP spans, and FN-injected coords
- build token-type CE masks per Unified Loss Registry `context=rollout` (FP-neutral + EOS-enforced; Section 5.5.4C)

### 10.8 Diagnostics
- queue length (raw + ready), `ver` lag stats
- parse success rate, `N_valid_pred`, `N_drop_invalid` (by reason)
- `|ValidMatched|`, `|FP|`, `|FN|`
- realized A/B ratio over a window

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
  - localization improvement (IoU distribution, CIoU loss reduction)
  - robustness to permutation/order
  - reduced sensitivity to quantization boundary cases

**Update:** run Stage-2 as Channel-A/Channel-B mixture:
- Channel-A provides high-throughput, stable geometry calibration without rollout.
- Channel-B provides sparse but essential “true self-context + set correction”.

---

## 12. Minimal Working Example (Two objects: “black cat” and “yellow dog”)

### GT (JSON-only assistant payload; bbox only)
```json
{
  "objects": [
    {
      "desc": "black cat",
      "bbox_2d": [<|coord_110|>, <|coord_310|>, <|coord_410|>, <|coord_705|>]
    },
    {
      "desc": "yellow dog",
      "bbox_2d": [<|coord_520|>, <|coord_285|>, <|coord_890|>, <|coord_660|>]
    }
  ]
}
```

### Rollout (model output; JSON-only, bbox only)
```json
{
  "objects": [
    {
      "desc": "black cat",
      "bbox_2d": [<|coord_120|>, <|coord_300|>, <|coord_420|>, <|coord_700|>]
    },
    {
      "desc": "yellow dog",
      "bbox_2d": [<|coord_500|>, <|coord_280|>, <|coord_880|>, <|coord_650|>]
    }
  ]
}
```

### Matching

* predicted #1 → GT(cat)
* predicted #2 → GT(dog)

### Unified one-pass Channel-B update

* rollout -> strict parse -> Hungarian/gating to get `ValidMatched / FP / FN`
* append FN records into the same top-level `objects[]` container
* run one teacher-forced forward on injected sequence `y_in`
* apply token-type CE masks per Unified Loss Registry `context=rollout` (Section 5.5.4C): FP-neutral on FP spans and **EOS-enforced** on top-level `}` + `<|im_end|>`
* decode coords per `coord_decode_mode` (Section 2.3 / 2.6.3) and apply `L_geo` on `ValidMatched` + FN geometry from the same logits (Section 5.5)

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
* Use Unified Channel-B FN injection (inside the same JSON dict) to teach recall.

### 14.3 Extra objects

* In many detection datasets, “FP under Hungarian” may be **unlabeled true objects**. Therefore Channel-B should be **FP-neutral**:
  - no geometric loss on unmatched predicted objects,
  - no token CE on FP spans (Strategy A full-mask),
  - **EOS-enforced (not stop-neutral):** top-level `}` and `<|im_end|>` remain supervised (Section 5.5.4C).

### 14.4 Multi-peak coord distributions (“mean collapse”)

* Default pipeline keeps distribution-shape regularizers off.
* If box-level `L_geo` becomes unstable early, first use `softctx_grad_mode=em_detach`; distribution-shape controls are future extensions.

### 14.5 Cross-object gradient coupling (UNROLL mode)

* In `softctx_grad_mode=unroll`, later objects' geometry losses may backprop through soft coord embeddings into earlier objects' coord slots (cross-object credit assignment).
* Pros: encourages set-level consistency (e.g., fewer duplicate/overlapping boxes) and can improve global coherence.
* Cons: may introduce order bias or instability for long sequences / many objects.
* Mitigations: use `softctx_grad_mode=em_detach`, reduce `n_softctx_iter`, and use gradient clipping.

---

## 15. Summary

This document specifies a complete end-to-end pipeline for training a pretrained V-LLM (Qwen3-VL) to perform open-vocab detection using structured text output with norm1000 coordinates, leveraging:

* CoordExp for differentiable geometry gradients into LM logits,
* Hungarian matching for set-level alignment,
* EM-ish rollout + update loops,
* and standard SFT for caption/format/recall control,
  all without introducing new detection heads or DETR-style architectures.
