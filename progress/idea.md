## Project Overview: CoordExp-Det on Qwen3-VL

### 1. High-Level Goal

This project aims to turn a vision-language large model (V-LLM), specifically **Qwen3-VL**, into an **open-vocabulary, set-structured object detector** without adding any extra detection head or using reinforcement learning.

Instead, we:

* Extend the tokenizer with **discrete coordinate tokens** (`<|coord_0|> ... <|coord_999|>`).
* Define a **forward mapping from coordinate-token probabilities to continuous box coordinates** (CoordExp: coordinate expectation).
* Apply **continuous geometric losses (L1 + GIoU)** directly on the LM head outputs.
* Gradually introduce **set-level supervision** (Hungarian matching, multi-order training, etc.) on top of the standard autoregressive SFT framework.

The long-term goal is to support both **bounding boxes** and **polygons**, and to reuse the same unified language interface for dense captioning, detection, and grounding-like tasks.

---

### 2. Environment and Constraints

* **Base model**: Qwen3-VL family (vision-language transformer).
* **Input coordinate convention**: Qwen3-VL uses a **1000-bin integer grid** “norm1000” with indices **0–999** (0 = edge, 999 = just inside the far edge).
  We will align `<|coord_k|>` to this convention.
* **Trainable components**:

  * Token embedding matrix (for newly added special tokens, especially coordinate tokens).
  * LM head (output projection for the new tokens).
  * Either full fine-tuning or parameter-efficient fine-tuning (e.g., dLoRA) can be used.
* **Frozen components**:

  * Core transformer layers, attention, FFNs, vision encoder, and vision-text alignment modules should remain structurally unchanged.
  * We do not introduce a separate detection head or CNN/heatmap branch.

---

### 3. Data and Tasks

We target **open-vocabulary detection and grounding** on top of existing public datasets:

* Detection / instance segmentation: **COCO**, **LVIS**, **Objects365**, etc.
* Referring expressions / grounding: **RefCOCO**, **RefCOCO+**, **RefCOCOg**, etc.

All datasets are transformed into a shared intermediate format per image:

```jsonc
{
  "image": "path/to/image.jpg",
  "objects": [
    {
      "desc": "a red cup on the table",     // free-form description
      "bbox": [x1, y1, x2, y2],           // normalized to [0, 1]
      "poly": [x1, y1, ..., xn, yn]       // optional, can be null or omitted
    },
    ...
  ]
}
```

For early stages, **bounding boxes** are the main target; polygons are considered a later extension.

---

### 4. Core Design Ideas

#### 4.1 Coordinate Tokens and CoordExp

* The vocabulary is split into:

  * `V_text`: regular text tokens.
  * `V_coord = { <|coord_0|>, ..., <|coord_999|> }`: coordinate tokens (1000 bins).

* Mapping from tokens to continuous coordinates:

  [
  \phi(\langle\text{coord}_k\rangle) = \frac{k}{1000}, \quad k \in {0,\dots, 999}.
  ]

* For a coordinate position (e.g., x1) with logits `z ∈ R^{|V|}`, we restrict to `V_coord`, apply a softmax (with temperature τ if needed), and decode a continuous coordinate via expectation:

  [
  p_k = \frac{\exp(z_k / \tau)}{\sum_{k' \in V_{\text{coord}}}\exp(z_{k'}/\tau)}, \quad
  \hat{c} = \sum_{k \in V_{\text{coord}}} \phi(k), p_k.
  ]

* We apply this to x1, y1, x2, y2 to obtain a predicted box:

  [
  \hat{b} = (\hat{x}_1, \hat{y}_1, \hat{x}_2, \hat{y}_2).
  ]

* Invalid boxes (e.g., x1 > x2) are corrected with simple post-processing (clamp to [0,1], swap coordinates, or use a different parameterization like center + width/height).

This mechanism provides a **differentiable bridge** from discrete token probabilities to continuous coordinates, enabling **L1 + GIoU** style geometric supervision directly on the LM head.

---

#### 4.2 Object Subsequence Encoding

Each detected object is encoded as a structured subsequence inside the autoregressive output, for example:

```text
<obj_start>
  [text tokens for object description]
  <|coord_x1|> <|coord_y1|> <|coord_x2|> <|coord_y2|>
<obj_end>
```

Multiple objects are concatenated into a full sequence, using a **consistent ordering rule** (e.g., top-to-bottom, then left-to-right, based on (y1, x1)). This ordering is important for teacher forcing in early stages, while later we will introduce **set-based training** to relax this ordering constraint.

---

#### 4.3 Set-Level Supervision and Hungarian Matching (Later Stages)

At the set level:

* The model output is parsed into a predicted set:
  [
  \hat{G} = {\hat{g}_i = (\hat{d}_i, \hat{b}*i)}*{i=1}^N.
  ]
* Ground-truth objects:
  [
  G = {g_j = (d_j, b_j)}_{j=1}^M.
  ]
* A cost matrix combines geometric and semantic distances:
  [
  C_{ij} = \lambda_{\text{geo}} L_{\text{coord}}(\hat{b}*i, b_j)
  + \lambda*{\text{sem}} D_{\text{sem}}(\hat{d}_i, d_j).
  ]
* We run **Hungarian matching** (or equivalent) to obtain an order-free matching between predicted and ground-truth objects, and use this for geometric supervision.

This makes the supervision **set-structured and order-invariant** at the object level, while still operating within the standard autoregressive SFT training loop.

---

#### 4.4 Unified Loss

The general training loss combines:

* Token-level text loss (cross-entropy) for natural language tokens (and optionally structural tokens).
* Continuous geometric loss for boxes produced via CoordExp.

Example form:

[
L_{\text{total}} = L_{\text{text}} + \lambda_{\text{coord}} L_{\text{coord}},
]

where:

* ( L_{\text{text}} ) is an autoregressive cross-entropy over selected positions.
* ( L_{\text{coord}} ) is a sum of L1 + (1 − GIoU) over matched box pairs.

Later, we will also compare different combinations:

* pure token CE for coordinates vs.
* CoordExp + geometric loss vs.
* hybrid CE + geometric loss on coordinate positions.

---

### 5. Polygon Extension (Future Phase)

Polygons are planned as a later phase:

* Option 1: reuse `V_coord` and output more coordinate tokens to represent polygon vertices.
* Option 2: introduce a dedicated `V_poly_coord` and polygon-specific structural tokens (`<poly_start>`, `<poly_end>`, etc.).
* Vertices are truncated or simplified to a maximum number (e.g., 8–16 points) using standard simplification (e.g., RDP).
* Polygon loss may approximate IoU at mask level (e.g., via rasterization at low resolution) or a differentiable polygon IoU.

The overall architecture remains the same: **all geometric supervision flows through LM head logits via CoordExp-style decoding**.

---

### 6. Out of Scope / Non-Goals

* No additional detection heads or CNN heatmap branches.
* No large-scale RL pipelines (PPO, REINFORCE with IoU rewards, etc.) in the main method.
* No changes to Qwen3-VL’s internal transformer or vision backbone topology.

We focus on what can be achieved by **vocabulary design, sequence format, and loss functions** on top of a frozen architecture.

---

### 7. Roadmap and Development Stages

We plan several incremental stages:

1. **Stage 1 – Coord Token Pretraining & Loss Ablations (current focus)**

   * Introduce `<|coord_k|>` tokens and structured output format.
   * Train only with fixed object ordering, no Hungarian yet.
   * Perform systematic ablations:

     * Standard CE-only baseline (discrete coordinates).
     * CoordExp + L1+GIoU with no CE on coordinates.
     * Hybrid CE + geometric loss on coordinates.

2. **Stage 2 – Set-Level Training with Hungarian Matching**

   * Switch from index-based (`i ↔ i`) supervision to set-based matching.
   * Keep ordering fixed in the teacher-forced sequence, but geometric loss becomes order-invariant at the object set level.

3. **Stage 3 – Multi-Order / Monte Carlo Training**

   * For each image, sample multiple permutations of object order, train on all, and average the loss to approximate order-agnostic risk.

4. **Stage 4 – Free-Generation Consistency (Optional)**

   * Occasionally run the model in free-generation mode (no teacher forcing), parse outputs as sets, and enforce geometric consistency with the ground truth.

5. **Stage 5 – Polygon Integration**

   * Extend the same CoordExp idea and set-level training to polygon vertices and mask-like supervision.

Throughout all stages, a global principle is:

> **Design everything to support fine-grained ablations, so we can toggle each component and isolate its effect.**
