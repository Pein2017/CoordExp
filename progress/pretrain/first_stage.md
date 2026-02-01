# Phase 1 / Stage-1 Training: Coord-Token SFT (Distribution Supervision)

This note describes the **current Stage-1 (baseline) SFT training** used in CoordExp to teach the model to:
- emit **JSON-only** structured detection outputs, and
- emit coordinate tokens `<|coord_k|>` reliably (k in `0..999`).

Anchors:
- **OpenSpec**: change `2026-01-14-update-stage1-softce-w1`
- **Entry point**: `src/sft.py`
- **Stage-1 source-of-truth strategy**: `configs/dlora/sft_stage1_softce_w1.yaml`

The Stage-1 recipe is a **single-forward loss**:

$$
\textbf{total\_loss} \;=\; L_{\text{text/struct CE}}
\;+\; \lambda_{ce}L_{\text{coord softCE}}
\;+\; \lambda_{w1}L_{\text{coord W1}}
\;+\; \lambda_{gate}L_{\text{coord gate}}
$$

with:
- **Text/structure tokens**: standard full-vocab CE (ms-swift / Qwen3-VL path)
- **Coord tokens (`<|coord_k|>`)**: distribution losses (`softCE + W1`) + a coord-vocab gate penalty
- **No decoded coordinates** (no expectation/argmax/median) and **no continuous box/polygon losses** in Stage-1

---

## 1) Data + Output Contract

Stage-1 uses the project’s **JSON-only assistant schema** (no wrapper tokens like `<obj_start>`).

- Data schema (authoritative): `docs/data/JSONL_CONTRACT.md`
- Preprocessing pipeline (authoritative): `docs/DATA_PREPROCESSING_PIPELINE.md`

At a high level, each record contains:
- `images`: image paths
- `width` / `height`
- `objects`: list of objects where each has:
  - `desc` (string)
  - exactly one geometry field: `bbox_2d` **or** `poly` **or** `line`

In **coord-token mode**, geometry values are serialized as strings like `"<|coord_12|>"` so the tokenizer treats them as special tokens.

Assistant output (dense mode) is a single JSON object, e.g.:
```json
{"object_1": {"desc": "...", "bbox_2d": ["<|coord_12|>", "<|coord_34|>", "<|coord_56|>", "<|coord_78|>"]}, "...": "..."}
```

Geometry guardrails:
- Training path disables runtime image resizing (see `configs/base.yaml` `template.max_pixels`); do **not** drop/reorder coordinates.
- Use shared geometry utilities for any conversions (`src/datasets/geometry.py`, `src/common/geometry/coord_utils.py`).

---

## 2) Stage-1 Pipeline (data -> tokens -> forward -> losses -> logs)

### 2.1 Dataset + ordering (reproducibility)
Stage-1 is teacher-forced SFT: we supervise the assistant tokens produced by the template builder.

Object ordering is controlled by YAML:
- `custom.object_ordering: sorted` (default): sort objects by top-left (`y1`, then `x1`)
- `custom.object_ordering: random`: shuffle per sample (ablation)

Implementation reference: `src/datasets/dense_caption.py` (object ordering in `__getitem__`).

### 2.2 Template encode (no span parser)
We rely on the tokenizer/template to produce:
- `input_ids`
- `labels` (with `-100` for non-supervised positions, including image tokens)
- `attention_mask`

No additional “object span” parser is required for Stage-1 losses: **coord-supervised positions are derived from teacher-forced `labels`**.

### 2.3 Collator (packing-safe metadata)
The collator may attach optional metadata (packing-safe):
- `token_types` (for aggregate token-type metrics)
- instability monitor metadata (optional)

Implementation reference: `src/data_collators/dataset_metrics.py`

### 2.4 Trainer mixins (single forward)
We use trainer mixins to keep behavior local and avoid editing upstream model internals.

- Base CE is computed by the underlying ms-swift trainer.
- We optionally inject coord distribution losses from `outputs.logits` after the same forward.

Implementation reference:
- `src/metrics/dataset_metrics.py` (`CoordSoftCEW1LossMixin`)

---

## 3) Loss Architecture (Scheme A)

### 3.1 Definitions
- Vocabulary size: $V$
- Coord sub-vocabulary (ordered bins): $V_{\text{coord}}$, size $K=1000$
- Temperature: $\tau > 0$
- Coord label bin index at a coord position: $k^*\in\{0,\dots,K-1\}$

Coord token IDs are obtained from the tokenizer and we build an ID -> bin index map:
- `coord_id_map[vocab_id] = bin_index` for vocab IDs that correspond to `<|coord_k|>`
- all non-coord vocab IDs map to `-1`

**Coord positions MUST be identified from labels/teacher forcing** (not model predictions). This stays correct under packing.

### 3.2 Base CE on non-coord tokens only (mask coord targets)
We compute native CE loss, but **mask coord-token targets** to ignore index (`-100`) first:

- Let `labels_orig` be the original labels from the template.
- Let `labels_masked = mask_coord_targets(labels_orig)` where any position whose target is a coord token is set to `-100`.
- Feed `labels_masked` into the underlying trainer CE computation.

Effect: coord slots do **not** contribute to full-vocab CE (no double-supervision).

### 3.3 Coord distribution losses (softCE + W1 + gate) from the same logits
Using the same forward logits, at positions where the **original** target is a coord token:

1) **Coord-vocab slice**
   - Full vocab logits: $z \in \mathbb{R}^{V}$
   - Coord logits: $s = z[V_{\text{coord}}] \in \mathbb{R}^{K}$ (ordered 0..999)

2) **Pred distribution**
   $$
   p = \text{softmax}(s/\tau)
   $$

3) **Target soft label**
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
We penalize probability mass assigned to non-coord vocab at coord positions:
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

## 4) Stage-1 Config Surface (source of truth)

Stage-1 runs are configured via YAML (avoid ad-hoc CLI flags).

The **current source-of-truth strategy** is: `configs/dlora/sft_stage1_softce_w1.yaml`.
Key points from that config:

- **Coord-token mode is enabled** (via `configs/dlora/sft_base.yaml` defaults):
  - `custom.coord_tokens.enabled: true`
  - `custom.coord_tokens.skip_bbox_norm: true` (avoid double-normalization when JSONLs already contain `<|coord_k|>`)
- **Packing is on** and is the main efficiency lever:
  - `training.packing: true`
  - `training.packing_buffer: 512`
  - `training.packing_min_fill_ratio: 0.7`
  - `training.packing_drop_last: true`
  - `training.eval_packing: true`
- **Optimizer buckets** are used to isolate coordinate-token adaptation:
  - `training.optimizer: multimodal_coord_offset`
  - separate `learning_rate`, `vit_lr`, `aligner_lr` (see YAML)
- **Coord offset adapter** trains only the newly added coord-token ID range:
  - `custom.coord_offset.enabled: true`
  - `custom.coord_offset.ids: { start: 151670, end: 152669 }`  # `<|coord_0|>`..`<|coord_999|>`
- **Distribution loss weights** (defaults from YAML):
  - `custom.coord_soft_ce_w1.soft_ce_weight: 1.0`
  - `custom.coord_soft_ce_w1.w1_weight: 3.0`
  - `custom.coord_soft_ce_w1.gate_weight: 5.0`
  - `custom.coord_soft_ce_w1.temperature: 0.9`
  - `custom.coord_soft_ce_w1.target_sigma: 1.5`
  - `custom.coord_soft_ce_w1.target_truncate: 8`
- **Token-type metrics** are enabled in this run config (useful for auditing structure vs coord behavior):
  - `custom.token_type_metrics.enabled: true`

Run command:
```bash
conda run -n ms python -m src.sft --config configs/dlora/sft_stage1_softce_w1.yaml
```

---

## 5) Logging / Diagnostics

Stage-1 diagnostics focus on:
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

## 6) Legacy Notes (archived Phase-1 draft; included here)

This section is preserved for historical context only. It is **NOT** the current Stage-1 baseline (Sections 1-5).

### 6.1 Legacy objective (bounding boxes only)
Earlier Phase-1 notes framed Stage-1 as “coord token pretraining + loss ablations”:

1. Integrate `<|coord_k|>` tokens (`k = 0..999`) into tokenizer + LM head (1000 bins).
2. Define a structured output that interleaves free-form descriptions with coord tokens.
3. Compare coordinate supervision variants:
   - **Group A**: full-vocab CE on all tokens (including coord tokens)
   - **Group B**: decode continuous coords (CoordExp) + L1/GIoU, no CE on coord tokens
   - **Group C**: hybrid (CE + decoded-geo losses)

Stage-1 today intentionally avoids decoded-geometry objectives; those ideas belong in Stage-2+ experiments if needed.

### 6.2 Legacy ordering assumption
Legacy Phase-1 notes used a fixed ordering rule to avoid mixing “loss design” with “ordering issues”:
- Primary key: `y1` ascending (top to bottom)
- Secondary key: `x1` ascending (left to right)

This matches today’s default `custom.object_ordering: sorted`.

### 6.3 Legacy loss ablation sketch (not the current baseline)
The legacy draft described token-level subsets:
- `coord_positions`: positions whose target is `<|coord_k|>`
- `desc_positions`: description + structure positions

and defined a geometric loss per box (after decoding continuous coordinates):

[
\ell_{\text{coord}}(b̂_i, b_i) =
\alpha \lVert b̂_i - b_i \rVert_1 + \beta (1 - \mathrm{GIoU}(b̂_i, b_i)).
]

Then two main baselines were outlined:

**Group Coord-CE (coord full-vocab CE only)**
[
L_{\text{Coord-CE}} = L_{\text{text-desc}} + \lambda_{\text{coord-CE}} L_{\text{coord-CE}}.
]

**Group Coord-CE+Geo (coord CE + decoded CoordExp + L1/GIoU)**
[
L_{\text{Coord-CE+Geo}} =
L_{\text{text-desc}}
+ \lambda_{\text{coord-CE}} L_{\text{coord-CE}}
+ \lambda_{\text{coord}} L_{\text{coord}}.
]

These are not used in today’s Stage-1 recipe, which is purely distributional at coord slots (softCE + W1 + gate).

### 6.4 Legacy implementation checklist (still conceptually useful)
The older draft called out:
- loss injection without editing upstream model forward
- config-driven ablations (rather than ad-hoc scripts)
- modularity for future phases (matching, multi-order, polygon support)

---

## 7) Pointers

- Stage-1 loss implementation: `src/metrics/dataset_metrics.py`
- Coord distribution helpers: `src/coord_tokens/soft_ce_w1.py`
- Config schema: `src/config/schema.py`
- Stage-1 YAML: `configs/dlora/sft_stage1_softce_w1.yaml`
- Data contract: `docs/data/JSONL_CONTRACT.md`
- Preprocessing pipeline: `docs/DATA_PREPROCESSING_PIPELINE.md`
- Offline evaluator: `docs/detection_evaluator.md`
