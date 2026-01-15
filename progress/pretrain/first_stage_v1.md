## Phase 1 Spec: Coord Token Pretraining and Loss Ablations

### 1. Objective

Phase 1 focuses on **pretraining Qwen3-VL with coordinate tokens** and running **fine-grained loss ablations** for bounding-box prediction only.

Main goals:

1. Integrate `<|coord_k|>` tokens (`k = 0..999`) into the tokenizer and LM head (1000 bins).
2. Define a structured output format that interleaves free-form object descriptions with coordinate tokens.
3. Implement and compare multiple loss variants:

   * **Group A**: standard cross-entropy (CE-only) for all tokens, including coordinates.
   * **Group B**: CoordExp + L1+GIoU for coordinates, no CE on coordinate tokens.
   * **Group C**: hybrid CE + CoordExp + L1+GIoU on coordinate positions.
4. Run controlled experiments under identical settings to understand the benefit of **continuous geometric supervision** vs. discrete token CE.

At this phase we **do not** use Hungarian matching or multi-order sampling; the object order is fixed and supervision is index-based (`i ↔ i`).

---

### 2. Assumptions and Preconditions

* A Qwen3-VL checkpoint is available and can be fine-tuned (full or dLoRA).
* The model and tokenizer can be modified to add new tokens (coordinate tokens and structural markers like `<obj_start>`, `<obj_end>` if needed).
* A detection dataset can be loaded and converted to an intermediate format with:

  * image path,
  * list of objects with text descriptions and bounding boxes normalized to [0,1].

---

### 3. Data Format and Preprocessing

#### 3.1 Unified Sample Format

Each training sample should be logically represented as:

```jsonc
{
  "image": "path/to/image.jpg",
  "objects": [
    {
      "desc": "description of object i",   // free-form English phrase
      "bbox": [x1, y1, x2, y2]            // normalized to [0, 1]
    },
    ...
  ]
}
```

For Phase 1:

* Only **bounding boxes** are used.
* The number of objects per image can be capped (e.g., max 30) to keep sequences reasonably short.

#### 3.2 Object Ordering

Phase 1 uses a fixed deterministic ordering rule to avoid mixing “loss design” with “ordering issues”:

* Primary key: **y1 ascending** (top to bottom).
* Secondary key: **x1 ascending** (left to right).

This yields a “top-left → bottom-right” scan order.

This ordering is applied:

* When sorting the `objects` list.
* When constructing the sequence of object subsequences.

---

### 4. Vocabulary and Tokenization Changes

#### 4.1 Coordinate Tokens

Add the following tokens to the tokenizer and LM head:

* `<|coord_0|>`, `<|coord_1|>`, ..., `<|coord_999|>` (1000 bins)

Mapping:

* `k` in `[0, 999]` → `<|coord_k|>`
* Numeric value in [0,1) ↔ coordinate index `round(c * 1000)` (clamped to 0–999) for teacher-forced GT.

#### 4.2 Structural Tokens

We **do not** introduce custom wrapper tokens like `<obj_start>` / `<obj_end>` in Phase 1.

Object boundaries are instead determined by the **sequence template** itself (e.g., JSON-like keys such as `"object_1"`, `"object_2"`, or simple line breaks between objects), and all such structural characters are treated as regular text tokens supervised with standard CE.

---

### 5. Sequence Format and Prompting

#### 5.1 Prompt Template (Example)

A simple, consistent prompt for detection-style training:

```text
USER: Describe all objects in the image with their bounding boxes.
ASSISTANT: <generated sequence>
```

For Phase 1, we can keep the prompt fixed and short, focusing on making the model output well-structured object subsequences.

#### 5.2 Target Sequence Format

For a sorted list of objects `[obj_1, ..., obj_M]`, the target sequence (assistant side) for training might look like a simple per-line format without extra wrapper tokens:

```text
desc_1_tokens <|coord_x1_1|> <|coord_y1_1|> <|coord_x2_1|> <|coord_y2_1|>
desc_2_tokens <|coord_x1_2|> <|coord_y1_2|> <|coord_x2_2|> <|coord_y2_2|>
...
```

where:

* `desc_i_tokens` are the tokenized description from `objects[i].desc`.
* Coordinate tokens are obtained by quantizing the normalized coordinates:

  * `coord_x1_i_idx = round(x1_i * 1000)` → `<|coord_{coord_x1_i_idx}|>` (clamped to 0–999)

For Phase 1 we assume:

* The number and order of object subsequences are given by the dataset (after sorting).
* There is a 1-to-1 alignment between the i-th predicted object subsequence and the i-th ground-truth object.

---

### 6. Loss Variants for Ablation

The global principle here is to **focus ablations on coordinate supervision**. We keep standard CE on text/format tokens as baseline SFT, and vary only how we supervise the **coordinate positions**.

#### 6.1 Common Notation

* Let `T` be the total number of tokens in the target sequence.
* Let `coord_positions` be the subset of token indices corresponding to `<|coord_k|>` tokens.
* Let `desc_positions` be the token indices corresponding to description tokens (and optionally structural tokens).
* Let `b_i` be the i-th ground-truth box in [0,1]^4.
* Let `b̂_i` be the predicted box obtained via CoordExp over the four coordinate positions for object i.

Geometric loss per box:

[
\ell_{\text{coord}}(b̂_i, b_i) =
\alpha \lVert b̂_i - b_i \rVert_1 + \beta (1 - \mathrm{GIoU}(b̂_i, b_i)).
]

Total geometric loss:

[
L_{\text{coord}} = \sum_{i=1}^M \ell_{\text{coord}}(b̂_i, b_i).
]

We always include a **text/format CE** term:

[
L_{\text{text-desc}} =
-\sum_{t \in \text{desc_positions}} \log p_\theta(y_t | x, y_{<t}).
]

and ablate what happens on `coord_positions`.

---

#### 6.2 Group Coord-CE: Pure Coordinate CE (No Geometric Loss)

**Purpose**: baseline where coordinates are supervised **only as discrete tokens**; no auxiliary geometric loss is applied.

* Text loss (always on):
  [
  L_{\text{text-desc}} =
  -\sum_{t \in \text{desc_positions}} \log p_\theta(y_t | x, y_{<t}).
  ]
* Coordinate CE:
  [
  L_{\text{coord-CE}} =
  -\sum_{t \in \text{coord_positions}} \log p_\theta(y_t | x, y_{<t}).
  ]
* No CoordExp-based geometric loss:
  [
  L_{\text{coord}} = 0.
  ]
* Total loss:
  [
  L_{\text{Coord-CE}} = L_{\text{text-desc}} + \lambda_{\text{coord-CE}} L_{\text{coord-CE}}.
  ]

Flags:

* `use_coord_ce = True`
* `use_coordexp_loss = False`
* `use_geom_loss = False`

This corresponds to the **“pure coord with CE”** setting.

---

#### 6.3 Group Coord-CE+Geo: Coordinate CE + L1+GIoU

**Purpose**: test whether adding continuous geometric supervision on top of discrete coord CE improves localization.

* Text loss (same as above):
  [
  L_{\text{text-desc}} =
  -\sum_{t \in \text{desc_positions}} \log p_\theta(y_t | x, y_{<t}).
  ]
* Coordinate CE (same definition as Group Coord-CE):
  [
  L_{\text{coord-CE}} =
  -\sum_{t \in \text{coord_positions}} \log p_\theta(y_t | x, y_{<t}).
  ]
* CoordExp geometric loss:
  1. Restrict logits at each coord position to `V_coord`.
  2. Softmax → `p_k`.
  3. CoordExp decode `b̂_i` for each object.
  4. Compute:
     [
     L_{\text{coord}} = \sum_{i=1}^M \ell_{\text{coord}}(b̂_i, b_i).
     ]
* Total loss:
  [
  L_{\text{Coord-CE+Geo}} =
  L_{\text{text-desc}}
  + \lambda_{\text{coord-CE}} L_{\text{coord-CE}}
  + \lambda_{\text{coord}} L_{\text{coord}}.
  ]

Flags:

* `use_coord_ce = True`
* `use_coordexp_loss = True`
* `use_geom_loss = True`

This corresponds to **“pure coord with CE + L1-loss + GIoU loss”**: coordinates receive both discrete and continuous supervision.

---

### 7. Implementation Requirements

The Phase 1 implementation should provide:

1. **Configurable loss flags**

   * CLI / config fields for:

     * `use_coord_ce` (bool)
     * `use_coordexp_loss` (bool)
     * `use_geom_loss` (bool)
     * `lambda_coord`, `lambda_coord_ce`, `alpha_l1`, `beta_giou`
2. **Reusable CoordExp module**

   * Given:

     * logits tensor of shape `[batch, seq_len, vocab_size]`,
     * a list of coordinate positions per object,
   * Return:

     * decoded continuous boxes `b̂_i` for geometric loss.
3. **Reusable sequence parser**

   * From target sequence token indices and metadata:

     * identify:

       * `coord_positions`
       * `desc_positions`
       * object segmentation (per-object coord spans in the linearized sequence)
     * map between object index and coordinate token positions.
4. **Training loop changes**

   * Inject additional loss terms without modifying the core Qwen3-VL forward function.
   * Support multiple experiment groups in a clean way (e.g., one unified training script with config-driven behavior).

---

### 8. Experiment Design and Metrics

#### 8.1 Fixed Experimental Setup

To make ablations comparable:

* Use the same:

  * dataset split (e.g., COCO train subset, COCO val for evaluation),
  * image preprocessing,
  * prompts,
  * optimizer type and learning rate schedule,
  * batch size,
  * training steps or epochs.

For each of Group A, B, C, only loss flags should change.

#### 8.2 Evaluation Metrics

* Standard detection metrics:

  * COCO-style mAP@[0.5:0.95], AP@0.5, AP@0.75, AP_small, AP_medium, AP_large.
* Training diagnostics:

  * `train_total_loss`, `train_text_loss`, `train_coord_loss`, `train_coord_ce_loss` (where applicable).
  * Validation loss components.
  * Optionally: distribution of predicted vs. ground-truth coordinates, histogram of IoU.

---

### 9. Deliverables for Phase 1

The Phase 1 implementation should deliver:

1. Updated tokenizer and model loading code that supports `<|coord_k|>` tokens and structured object subsequences.
2. Data preprocessing utilities that:

   * normalize boxes to [0,1],
   * quantize them to `0..999`,
   * build consistent target sequences with fixed object order.
3. A set of training configurations for:

   * Group A (CE-only),
   * Group B (Geom-only on coordinates),
   * Group C (Hybrid).
4. Evaluation scripts that:

   * decode generated sequences into object sets,
   * map coordinate tokens to [0,1] coords via argmax or CoordExp,
   * run COCO-style evaluation and log results.
5. Logging and plotting hooks that make it easy to compare A vs B vs C in terms of:

   * detection metrics,
   * loss curves,
   * convergence speed.

Underlying principle:

> All components (tokenization, sequence format, losses, ordering rules) should be implemented in a modular way, so that future phases (Hungarian matching, multi-order training, polygon support) can be added as **orthogonal switches** for further ablations.
