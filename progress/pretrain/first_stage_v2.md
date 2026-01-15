# Phase 1 / Stage-1 Training (v2): Distribution Supervision for `<|coord_*|>` Tokens

This note describes the **current Stage‑1 (SFT) training architecture** used in CoordExp.

- **Legacy reference:** `progress/pretrain/first_stage_v1.md` (kept for historical context)
- **Current spec anchor:** OpenSpec change `2026-01-14-update-stage1-softce-w1`
- **Code entrypoint:** `src/sft.py`

The key update in **v2** is that Stage‑1 is trained with a **single-forward loss**:

$$
\textbf{total\_loss} \;=\; L_{\text{text/struct CE}} \;+\; \lambda_{ce}L_{\text{coord softCE}} \;+\; \lambda_{w1}L_{\text{coord W1}} \;+\; \lambda_{gate}L_{\text{coord gate}}
$$

with:
- **Text/structure tokens:** standard full‑vocab CE (native Qwen3‑VL / ms‑swift path)
- **Coord tokens (`<|coord_k|>`):** distribution losses (`softCE + W1`) + a coord‑vocab gate penalty
- **No decoded coordinates** (no expectation/argmax/median) and **no box/polygon continuous losses** in Stage‑1

---

## 1) Data + Output Contract (unchanged)

Stage‑1 uses the project’s **JSON-only assistant schema** (no wrapper tokens like `<obj_start>`).

- Data schema (authoritative): `docs/DATA_JSONL_CONTRACT.md`
- Preprocessing pipeline (authoritative): `docs/DATA_PREPROCESSING_PIPELINE.md`

At a high level, each record contains:
- `images`: image paths
- `width` / `height`
- `objects`: list of objects where each has:
  - `desc` (string)
  - exactly one geometry field: `bbox_2d` **or** `poly` **or** `line`

In **coord-token mode**, geometry values are serialized as strings like `"<|coord_12|>"` so that the tokenizer sees them as special tokens.

Assistant output (dense mode) is a single JSON object:
```json
{"object_1": {"desc": "...", "bbox_2d": ["<|coord_12|>", "<|coord_34|>", "<|coord_56|>", "<|coord_78|>"]}, "...": "..."}
```

---

## 2) Stage‑1 Pipeline (data → tokens → forward → losses → logs)

### 2.1 Dataset + ordering
Stage‑1 is teacher-forced SFT: we supervise the assistant tokens produced by the template builder.

Object ordering is controlled by YAML:
- `custom.object_ordering: sorted` (default): sort objects by top-left (`y1`, then `x1`)
- `custom.object_ordering: random`: shuffle per sample (ablation)

Implementation reference: `src/datasets/dense_caption.py` (see object ordering in `__getitem__`).

### 2.2 Template encode
We rely on the model’s tokenizer/template to produce:
- `input_ids`
- `labels` (with `-100` for non-supervised positions, including image tokens)
- `attention_mask`

No additional “object span” parser is required for Stage‑1 loss computation in v2; coord positions are derived from **labels**.

### 2.3 Collator
The collator may attach optional metadata (packing-safe):
- `token_types` (for aggregate token-type metrics)
- instability monitor metadata (optional)

Implementation reference: `src/data_collators/dataset_metrics.py`

### 2.4 Trainer mixins (single forward)
We use trainer mixins to keep behavior local and avoid editing upstream model internals.

- Base CE is computed by the underlying ms-swift trainer (native path).
- We optionally inject coord distribution losses from `outputs.logits` after the forward.

Implementation reference:
- `src/metrics/dataset_metrics.py` (`CoordSoftCEW1LossMixin`)

---

## 3) Loss Architecture (v2 / Scheme A)

### 3.1 Definitions
- Vocabulary size: $V$
- Coord sub-vocabulary (ordered bins): $V_{\text{coord}}$, size $K=1000$
- Temperature: $\tau > 0$
- Coord label bin index at a coord position: $k^*\in\{0,\dots,K-1\}$

Coord token IDs are obtained from the tokenizer, and we build an ID→bin index map:
- `coord_id_map[vocab_id] = bin_index` for vocab IDs that correspond to `<|coord_k|>`
- all non-coord vocab IDs map to `-1`

This is how we identify coord-supervised positions:
- **MUST** use labels/teacher forcing positions (not model predictions)

### 3.2 Base CE on non‑coord tokens only
We compute the model’s native CE loss, but we **mask coord-token targets** to ignore index (`-100`) first:

- Let `labels_orig` be the original labels from the template.
- Let `labels_masked = mask_coord_targets(labels_orig)` where any position whose target is a coord token is set to `-100`.
- Feed `labels_masked` into the underlying trainer CE computation.

Effect: coord slots do **not** contribute to full‑vocab CE (no double-supervision).

### 3.3 Coord distribution losses from the same logits
Using the *same forward pass* logits, at positions where the **original** target is a coord token:

1) **Coord-vocab gate (slice logits)**
   - Full vocab logits: $z \in \mathbb{R}^{V}$
   - Coord logits: $s = z[V_{\text{coord}}] \in \mathbb{R}^{K}$ (ordered 0..999)

2) **Pred distribution**
   $$
   p = \text{softmax}(s/\tau)
   $$

3) **Target soft label (unimodal Gaussian over bins)**
Build a unimodal soft target $q$ centered at $k^*$ using a Gaussian kernel (optionally truncated to a finite radius in bins).

4) **Soft cross-entropy**
   $$
   L_{\text{softCE}}(p,q) = -\sum_{k=0}^{K-1} q_k \log p_k
   $$

5) **1D Wasserstein-1 via CDF**
Let $CDF_p(i)=\sum_{k\le i}p_k$ and similarly $CDF_q$. Then:
   $$
   L_{W1}(p,q) = \sum_{i=0}^{K-1} |CDF_p(i)-CDF_q(i)|
   $$
(optionally normalized by $K$ so the scale is “fraction of coord range”.)

6) **Coord-vocab gate loss (mass leakage penalty)**
We penalize probability mass assigned to non‑coord vocab at coord positions:
   $$
   \text{mass}_{coord} = \sum_{i\in V_{coord}} \text{softmax}(z/\tau)_i
   \qquad
   L_{gate} = -\log(\text{mass}_{coord})
   $$
Numerically implemented as `logsumexp(all/T) - logsumexp(coord/T)`.

7) **Total loss**
$$
\textbf{total\_loss} \;=\; L_{\text{text/struct CE}}
\;+\; \lambda_{ce}L_{\text{coord softCE}}
\;+\; \lambda_{w1}L_{\text{coord W1}}
\;+\; \lambda_{gate}L_{\text{coord gate}}
$$

Implementation references:
- Loss utilities: `src/coord_tokens/soft_ce_w1.py`
- Trainer integration + gate term: `src/metrics/dataset_metrics.py`

---

## 4) Config Surface (YAML-first)

Stage‑1 runs are configured via YAML (no hyperparam CLI flags).

Minimal knobs:
```yaml
custom:
  coord_tokens:
    enabled: true
  coord_soft_ce_w1:
    enabled: true
    soft_ce_weight: 1.0
    w1_weight: 1.0
    gate_weight: 1.0
    temperature: 1.0
    target_sigma: 2.0
    target_truncate: 16
```

Example config to start from:
- `configs/dlora/sft_stage1_softce_w1.yaml`

Run command:
```bash
PYTHONPATH=. /root/miniconda3/envs/ms/bin/python -m src.sft --config configs/dlora/sft_stage1_softce_w1.yaml
```

---

## 5) Logging / Diagnostics (Stage‑1)

We keep Stage‑1 training diagnostics focused on:
- aggregate token-level accuracy for **text/structure**
- distribution-loss stability for coord tokens

Coord distribution logs (train/eval):
- `coord_softce_w1/loss`
- `coord_softce_w1/soft_ce`
- `coord_softce_w1/w1`
- `coord_softce_w1/gate`
- `coord_softce_w1/coord_vocab_mass` (mean coord-vocab probability mass)
- `coord_softce_w1/coord_tokens` (count of supervised coord tokens in the batch)

Optional token-type metrics (aggregate only):
- enable `custom.token_type_metrics.enabled: true`

Detection metrics (mAP, IoU-based) are **offline** evaluation concerns:
- see `docs/detection_evaluator.md`

---

## 6) Differences vs. `first_stage_v1.md` (legacy)

### 6.1 Loss design
**Legacy (v1):**
- Explicit loss ablation groups:
  - “coord CE only” vs “CoordExp + L1/GIoU” vs “hybrid”
- Required **decoded continuous coordinates** (expectation / top‑k expectation) to compute L1/GIoU.

**Now (v2):**
- **One Stage‑1 recipe** (Scheme A): CE on non‑coord tokens + distribution losses on coord tokens.
- **No decoding** (no expectation / argmax / median), therefore:
  - no L1 regression loss,
  - no bbox GIoU,
  - no poly mask IoU rasterization loss,
  - no poly smoothness/curvature regularization.

### 6.2 What defines coord positions
**Legacy (v1):**
- Often framed as requiring a “sequence parser” to find coord spans per object for losses/metrics.

**Now (v2):**
- Coord positions are identified strictly from **teacher-forced labels** via a coord-token ID map.
- This is robust to packing and does not depend on model predictions.

### 6.3 Guardrails / calibration
**Legacy (v1):**
- Full-vocab CE on coord tokens could implicitly keep mass on coord vocab, but it also caused double objectives with geometric losses.

**Now (v2):**
- We intentionally remove full-vocab CE at coord positions, so we add an explicit **coord-vocab gate penalty** to prevent coord slots from behaving like text slots.

### 6.4 Config + code surface
**Legacy (v1):**
- Multiple flags and hyperparameters for geometric losses (e.g., L1/GIoU weights, top‑k ratios, rasterization settings).

**Now (v2):**
- A single config block: `custom.coord_soft_ce_w1`
- Legacy blocks are removed and treated as configuration errors:
  - `custom.coord_loss` (removed)
  - decoded-coordinate “expectation metrics” (removed)

---

## 7) Practical Stage‑1 checklist (paper-ready)

1) Pick a base checkpoint (expanded vocab with coord tokens).
2) Use coord-token JSONLs produced by the preprocessing pipeline.
3) Start with a short packing length for stability, then scale up:
   - packing guide: `docs/PACKING_MODE_GUIDE.md`
4) Monitor:
   - `token_acc` / token-type metrics for structure correctness
   - `coord_softce_w1/*` for distribution stability and leakage (gate mass)
5) Run offline evaluation only after Stage‑1 is stable (parse rate, no NaNs).

---

## 8) Pointers

- Stage‑1 loss implementation: `src/metrics/dataset_metrics.py`
- Coord distribution helpers: `src/coord_tokens/soft_ce_w1.py`
- Config schema: `src/config/schema.py`
- Suggested Stage‑1 YAML: `configs/dlora/sft_stage1_softce_w1.yaml`
- Data contract: `docs/DATA_JSONL_CONTRACT.md`
- Preprocessing pipeline: `docs/DATA_PREPROCESSING_PIPELINE.md`
- Offline evaluator: `docs/detection_evaluator.md`
