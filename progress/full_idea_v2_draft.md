# CoordExp v2: Clean-Prefix Training + Duplicate Unlikelihood for V-LLM Detection

> Goal: train a pretrained V-LLM (e.g. Qwen3-VL) to emit **open-vocabulary object descriptions + norm1000 boxes** in structured text, while keeping **continuous geometric gradients** through CoordExp and adding a **minimal rollout-aligned fix** for autoregressive near-duplication (clean-prefix training + duplicate unlikelihood) — **without adding a DETR-style detection head**.

> This document is the current **center guidance / research spec**.
> It intentionally prioritizes a **simple, scalable core** over a large collection of clever tricks.
> Two problems are separated:
>
> 1. **Near duplication during generation**: solved in the canonical Stage-2 design below.
> 2. **Unlabeled annotation / missing positives**: treated as a future extension, not fused into the core objective yet.

---

## 0. Core Design Contract

### 0.1 Hard constraints

- Base model: pretrained V-LLM (Qwen3-VL family).
- Coordinates are expressed with existing `<|coord_k|>` tokens, `k in {0..999}`.
- No extra detection head, no DETR queries, no separate objectness classifier.
- Training must remain compatible with standard SFT infrastructure:
  - teacher forcing
  - LM logits over the normal vocabulary
  - extra losses computed from those logits

### 0.2 Simplified philosophy

We keep the system centered around three ideas:

1. **Stage-1 learns the language of boxes.**  
   Teacher forcing remains the scalable way to learn the discrete coord-token protocol, structured JSON emission, and basic geometry.

2. **Stage-2 should not try to solve every pathology with one objective.**  
   We explicitly separate:
   - **geometry / self-context stabilization** (Channel-A), and
   - **rollout-time set/cardinality correction** (Channel-B).

3. **Generic unmatched predictions are not the same as self-collision duplicates.**  
   In detection data, some unmatched predictions may be unlabeled true objects. Therefore:
   - **generic unmatched predictions stay neutral by default**;
   - **duplicate-certified predictions are not neutral** and receive a targeted negative signal.

This is the central change relative to the previous design.

### 0.3 Canonical partition in Stage-2

Given a raw rollout, parsed valid objects are partitioned into:

- `A`: **accepted objects** after sequential de-duplication (kept in the clean prefix)
- `D`: **duplicate objects** removed from the clean prefix and converted into negative supervision

Then matching is run on `A` against GT, giving:

- `M`: matched accepted objects
- `U`: accepted but unmatched objects (generic extras; neutral by default)
- `FN`: GT objects not explained by `A`

So the canonical Stage-2 flow is:

```text
raw parsed rollout
    -> sequential dedup
         -> A (accepted clean sequence)
         -> D (duplicate bursts attached to clean boundaries)
    -> Hungarian on A vs GT
         -> M, U, FN
```

### 0.4 What is and is not supervised

- **Matched accepted objects (`M`)**:
  - geometry supervised
  - structure supervised
  - matched `desc` CE remains off by default in Channel-B

- **Accepted unmatched objects (`U`)**:
  - kept as context in the clean prefix
  - no geometry loss
  - no token CE inside their spans
  - interpreted as *possible unlabeled positives or benign extras*

- **Duplicate objects (`D`)**:
  - removed from the positive prefix
  - turned into **unlikelihood** at the boundary where they appeared

- **FN objects**:
  - injected into the same `objects[]` container
  - receive positive structure + desc supervision and geometry supervision

This gives a much cleaner semantic split:
- `U` = “not judged yet”
- `D` = “judged redundant”

---

## 1. Output Format / Token Protocol

### 1.1 Canonical object format

Dense mode uses a single top-level JSON object:

```json
{"objects": [{...}, {...}]}
```

Each record must contain:

- `desc` (required)
- exactly one geometry field (required):
  - `bbox_2d`: `[x1, y1, x2, y2]`
  - or `poly`

Canonical order is **`desc` first, then geometry**.

### 1.2 Coordinate representation

- Assistant output uses bare coord tokens: `<|coord_k|>`
- `k in [0, 999]`
- `0 -> 0.0`, `999 -> 1.0`

### 1.3 Parsing invariants

- top-level object must contain exactly one key: `objects`
- `objects` must be an array
- each record must contain exactly `desc + one geometry field`
- invalid records are dropped by strict parsing; no repair is performed inside Channel-B

---

## 2. CoordExp: Differentiable Geometry from LM Logits

At a coordinate position `t`, gather logits only over coord tokens:

- `s_{t,k} = z_t[coord_k]`, `k = 0..999`
- `p_{t,k} = softmax(s_t / tau)`

Map bins to `[0,1]`:

- `phi(k) = k / 999`

Expectation decode:

- `x_hat_t = sum_k p_{t,k} phi(k)`

For a bbox object, decode four slots into `b_hat in [0,1]^4`.

Key identity:

- `d x_hat_t / d s_{t,k} = (1/tau) p_{t,k} (phi(k) - x_hat_t)`

So SmoothL1 / CIoU can backprop smoothly into LM logits.

### 2.1 ST bridge (kept)

We keep the existing **hard-forward / soft-backward** bridge at two places:

1. coord-slot context embeddings in Channel-A
2. coord decode used by geometry loss

This remains useful because inference is discrete while training often uses soft distributions.

---

## 3. Stage-1: Learn the Language of Boxes

### 3.1 Role of Stage-1

Stage-1 is the scalable foundation.
It should teach the model:

- structured JSON emission
- stable `desc_first` object records
- coord-token vocabulary confinement
- basic geometric calibration

### 3.2 Canonical Stage-1 objective

The default Stage-1 recipe keeps:

- normal CE on non-coordinate tokens
- coord-distribution supervision on coord positions:
  - hard coord CE anchor
  - soft CE
  - W1 / ordinal regularization
  - gate / leakage control where applicable

A compact view is:

```text
L_stage1 = L_text_ce + lambda_coord * (L_coord_ce + L_soft_ce + L_w1 + L_gate)
```

### 3.3 Why Stage-1 keeps soft/distributional supervision

We do **not** drop soft/distributional coord supervision in v2.
The working belief is:

- hard CE is useful for discrete coord-token selection
- soft CE / W1 keep the distribution better shaped
- the combination is stronger than either single-objective extreme

Optional small bbox-level geometry loss in Stage-1 is allowed, but it is not the defining novelty of this design.

---

## 4. Stage-2: Two Problems, Two Layers

### 4.1 Problem A: near duplication during rollout

This is the current canonical Stage-2 target.
The failure mode is:

- same or near-same object is emitted again and again
- sometimes contiguously, sometimes intermittently
- later correct objects may still appear

So the fix must work **within one autoregressive sequence**, not only at the end of the sample.
Concretely, we treat this as a **local decoding/stability problem**: under a given prefix, the model keeps choosing a redundant continuation.
The canonical solution is **clean-prefix training** plus **duplicate unlikelihood** applied at the **LCP-divergence token** of the duplicate continuation.

### 4.2 Problem B: unlabeled annotation / missing positives

This is real, but we keep it as a separate research thread.
For now the core design only assumes:

- **do not punish generic unmatched accepted objects**
- let them remain neutral unless we later build a separate missing-annotation extension (Section 11)

### 4.3 Recommended channel roles

- **Channel-A (hot path)**: stabilize geometry and self-context cheaply
- **Channel-B (cold path)**: correct rollout-time set/cardinality failures under true generation context

Recommended practice:
- start A-hot / B-cold
- only raise `b_ratio` after duplication / truncation are under control

---

## 5. Channel-A (Hot Path): Iterative Soft/ST Self-Context

### 5.1 Purpose

Channel-A is not the main tool for fixing near-duplication.
Its role is to:

- stabilize coord beliefs
- reduce self-context drift
- keep geometry from becoming noisy while Channel-B stays sparse

### 5.2 Mechanism

Let `e^(m)` be the coord-slot context embeddings at iteration `m`.
At each iteration:

1. forward once with current coord-slot context
2. read coord distributions from logits
3. rebuild coord-slot context embeddings using soft or ST embeddings
4. run the next iteration

Default:
- `n_softctx_iter = 2`
- `coord_ctx_embed_mode = st`
- `coord_decode_mode = exp` or `st`
- `softctx_grad_mode = unroll`

Fallback:
- `softctx_grad_mode: em_detach` when self-conditioning gradients become too noisy (detach the coord-distribution used to build the self-context embeddings)

### 5.3 Small A1 anchor (recommended)

A weak A1 anchor is recommended when crowded scenes show noisy first-iterate coord beliefs.
The anchor may include small weights on:

- bbox SmoothL1 / CIoU
- coord soft CE / W1

The point is not to over-constrain A1, but to reduce seed noise before the final self-context iteration.

### 5.4 Channel-A objective

A compact canonical form is:

```text
L_A = L_anchor_ce(gt) + lambda_geo_A * L_geo_A + lambda_coordreg_A * L_coordreg_A
```

Where:
- `L_anchor_ce(gt)` is the normal teacher-forced text/structure anchor
- `L_geo_A` uses CoordExp / ST decode on Channel-A forwards
- `L_coordreg_A` is a small distributional regularizer on coord bins

---

## 6. Channel-B (Cold Path): Clean-Prefix Rollout Matching

This is the canonical Stage-2 novelty in v2.

### 6.1 Step B0 — Raw rollout (no grad)

Generate a raw autoregressive output:

```text
y_raw = Rollout(f_theta, image, prompt)
```

### 6.2 Step B1 — Strict parse (no grad)

Parse valid objects from `y_raw`:

```text
O_raw = [o_1, o_2, ..., o_m]
```

Each object stores:

- normalized `desc`
- discrete bbox / poly
- token span
- tokenized object continuation

Invalid objects are dropped; no repair is attempted.

### 6.3 Step B2 — Sequential de-duplication (no grad)

We scan `O_raw` in order and build the **accepted clean sequence** `A`.

For each raw object `o_j`:

1. compare it to all previously accepted objects with the same normalized `desc`
2. let `r_j` be the accepted object with highest IoU
3. if `IoU(o_j, r_j) >= tau_dup`, mark `o_j` as **duplicate**
4. otherwise add `o_j` to `A`

Important:
- the duplicate test is against **previously accepted objects**, not GT
- the duplicate is attached to the **current clean boundary**
- this works even when duplication is intermittent

Operational default:

```text
duplicate(o_j) = 1  iff  exists accepted a < j with same_desc(o_j, a) and IoU(o_j, a) >= tau_dup
```

Typical starting point:
- `tau_dup = 0.9` in norm1000 space

The output of this step is:

- `A = [a_1, ..., a_K]`  (accepted objects, kept in positive prefix)
- `D_k` = duplicate objects that appeared while the clean prefix was `[a_1, ..., a_k]`

So duplicates are not just “objects of the same class”; they are objects judged to be **redundant continuations under the current prefix**.

### 6.4 Step B3 — Hungarian matching on the accepted clean sequence (no grad)

Run Hungarian / gated matching on `A` against GT objects `G`.
This produces:

- `M`: matched accepted objects
- `U`: accepted unmatched objects
- `FN`: GT objects not matched by `A`

Interpretation:

- `M` = supervised positives already present in the clean rollout
- `U` = accepted extras that are **not** punished by default
- `FN` = positives still missing and must be injected

### 6.5 Step B4 — Build the clean teacher-forced sequence

Construct `y_in` from the accepted clean sequence `A`:

1. keep accepted objects in their original rollout order
2. keep them inside the same top-level `{"objects": [...]}` container
3. append FN records inside that same array in GT canonical order
4. close the container normally

Crucially:

- **duplicates are removed from `y_in`**
- **generic accepted unmatched objects remain in `y_in` as context**

So `y_in` is **deduplicated**, but not “GT-purified”.

This is the canonical answer to a key training question:

> Later correct objects use the **clean deduplicated prefix**, not the raw duplicate-contaminated prefix.

But we do **not** delete generic accepted extras, because they may be unlabeled positives.

In other words, Channel-B performs **clean-prefix training**:
- the teacher-forced prefix is the deduplicated accepted sequence (plus neutral accepted extras),
- and the supervised continuation is the canonical clean continuation (including injected FN positives).

### 6.6 Step B5 — Token-level CE masks on `y_in`

Canonical mask policy:

#### Accepted matched objects (`M`)

- `CE_struct = 1`
- `CE_desc = 0`
- `CE_coord = 0`

#### Accepted unmatched objects (`U`)

- `CE_struct = 0`
- `CE_desc = 0`
- `CE_coord = 0`

They remain as context only.

#### FN injected objects

- `CE_struct = 1`
- `CE_desc = 1`
- `CE_coord = 0`

#### Global closure / EOS

- top-level closing brace / array closure is supervised
- `<|im_end|>` remains supervised

This preserves format discipline while avoiding hard negative supervision on ambiguous extras.

### 6.7 Step B6 — Duplicate-start unlikelihood on clean boundaries

This is the new core trick.

For each clean boundary `k`, define:

- `p_k`: the clean prefix ending after `a_k`
- `c_k^+`: the **canonical positive continuation** in `y_in`
  - either the next accepted object `a_{k+1}`
  - or the first FN injection if accepted objects are exhausted
  - or closure if nothing remains

Now consider a duplicate object `d in D_k` that appeared in the raw rollout after this clean prefix.
Let:

- `T_pos(d)` = token sequence of the canonical positive continuation from that boundary
- `T_dup(d)` = token sequence of the duplicate continuation from that boundary

We do **not** assume the bad token is always the first `desc` token.
That toy version is too naive when the next valid object may share the same class label.

Instead, define:

- `l = LCP(T_pos(d), T_dup(d))` = longest common prefix length
- `u(d)` = first token in `T_dup(d)` after that common prefix

Then apply unlikelihood at the first divergence point:

```text
L_dup = sum_{k} sum_{d in D_k} alpha(d) * [ -log( 1 - p_theta(u(d) | x, p_k + T_pos(d)[:l]) ) ]
```

Interpretation:

- we lower the probability of the **first token where the duplicate path diverges from the canonical clean continuation**
- this is the right definition of “the first redundant token”
- when the next valid object is a different class, `u(d)` is often an early desc token
- when the next valid object is the **same class but a different instance**, `u(d)` may be a later token (often the first differing coord token)

This avoids the obvious failure mode of blindly suppressing a class token like `book` when the next valid object is also a `book`.

#### Why this is the right granularity

- It is **not** a blanket span-level punishment.
- It directly targets the **local continuation decision** that created the redundancy.
- It is compatible with a single teacher-forced forward on `y_in`.

#### Default weighting

Start simple:

```text
alpha(d) = 1
```

Optional later weighting:
- by IoU to the reference accepted object
- by duplicate burst length
- by parse confidence

But weighting is not part of the core idea.

### 6.8 Step B7 — Geometry losses from the same forward

Using the same logits from the single teacher-forced forward on `y_in`:

- apply CoordExp / ST decode on matched accepted objects `M`
- apply CoordExp / ST decode on injected `FN`
- no geometry loss on `U`
- duplicates are already handled by `L_dup`

Compactly:

```text
L_geo_B = mean(L_geo_M) + mean(L_geo_FN)
```

### 6.9 Channel-B objective

The canonical v2 Channel-B loss is:

```text
L_B = L_struct_clean + L_desc_FN + lambda_dup * L_dup + lambda_geo_B * L_geo_B
```

This is intentionally small.

What is **not** in the core objective:
- no generic FP penalty
- no extra objectness head
- no repulsive set prior by default
- no soft-OT entropy term by default
- no RL reward shaping by default

Those may be explored later, but they are not needed for the first serious version.

---

## 7. Why the Clean-Prefix Formulation Matters

The key logic is:

- raw rollout contains the actual failure mode
- duplicate objects must be **observed** so they can be identified
- but later correct objects should **not** be teacher-forced under duplicate-contaminated prefixes

So Channel-B uses two views of the same rollout:

1. **raw view**: to discover duplicates
2. **clean view**: to define the positive teacher-forced prefix

This gives the clean separation:

- positive CE uses the clean deduplicated sequence
- negative UL uses raw duplicate continuations folded back onto clean boundaries

This is the minimal mechanism that changes model distribution rather than just post-processing outputs.

---

## 8. Overall Stage-2 Objective

Let `rho = b_ratio` be the fraction of Channel-B steps.

The overall objective is:

```text
L_stage2 = (1 - rho) * L_A + rho * L_B
```

Expanded:

```text
L_A = L_anchor_ce(gt) + lambda_geo_A * L_geo_A + lambda_coordreg_A * L_coordreg_A
L_B = L_struct_clean + L_desc_FN + lambda_dup * L_dup + lambda_geo_B * (mean(L_geo_M) + mean(L_geo_FN))
```

Recommended schedule policy:

- start with A-hot / B-cold
- raise `rho` only after duplication / truncation are under control
- do not let Channel-B dominate before the rollout distribution is stable

---

## 9. Optional Decode-Time Stabilizers (Not the Core Training Idea)

These are allowed as practical inference-side safety rails, but they are not the main scientific contribution.

### 9.1 Object-level duplicate veto

After a new object is fully parsed during decoding:

- if it has same normalized `desc` and very high IoU with an already accepted object, reject it
- if many such rejections happen consecutively, early-stop the list

This is a valid engineering stabilizer, but it does not replace train-time correction.

### 9.2 Mild repetition penalty / shorter token budget

These may reduce catastrophic loops in practice:

- moderate `repetition_penalty`
- controlled `max_new_tokens`

Treat them as guardrails, not as the center of the method.

---

## 10. Diagnostics and Acceptance Criteria

The core near-dup metrics are:

- `dup/max_desc_count`
- `dup/near_iou90_pairs_same_desc`
- `dup/near_iou90_pairs_any_desc`
- `dup/saturation_rate`
- `rollout/pred_per_sample`
- `rollout/rollout_len_mean`
- `rollout/parse_truncated_rate`
- `rollout/matched_maskiou_mean`

A good fix should satisfy:

1. `dup/near_iou90_pairs_same_desc` drops sharply
2. `pred_per_sample` and `rollout_len_mean` stop drifting upward
3. `parse_truncated_rate` does not grow over training
4. `matched_maskiou_mean` stays stable
5. recall does not collapse while duplicate enumeration is suppressed

Useful additional bookkeeping:

- `extra_dup_certified`: accepted duplicates identified by sequential dedup
- `extra_generic_unmatched`: accepted unmatched objects left neutral

The target is **not** to minimize all extras.
The target is to suppress **redundant self-collision** while preserving room for unlabeled true positives.

---

## 11. Planned Extension for Problem 2: Unlabeled Annotation as Latent Positives

This is **not** part of the current canonical objective, but it is the intended next research direction.

### 11.1 Guiding principle

Do not add an explicit binary objectness head.
Instead, use **likelihood-based latent-variable modeling** on top of the accepted unmatched set.

### 11.2 Candidate future view

For an accepted unmatched candidate `u in U`, define a latent responsibility:

```text
r(u) ~= probability that u is a true but unlabeled positive
```

The future extension should estimate `r(u)` from sequence likelihood / continuation likelihood, not from a separate objectness classifier.

### 11.3 Why it stays out of the core for now

Because the near-dup issue must be removed first.
Otherwise the latent-positive posterior is contaminated by self-collision duplicates and becomes much harder to interpret.

So the research order is:

1. fix duplicate instability cleanly
2. then build EM-like latent-positive updates on top of the cleaned accepted set

---

## 12. Minimal Working Examples

### 12.1 Example A: duplicate books before a different-class object

Raw rollout:

```text
[ book(A), book_dup(A1), book_dup(A2), person(P) ]
```

Sequential dedup gives:

```text
A = [ book(A), person(P) ]
D_1 = [ book_dup(A1), book_dup(A2) ]
```

Clean teacher-forced sequence uses:

```text
[ book(A), person(P) ]
```

At the boundary after `book(A)`:

- positive continuation = `person(P)`
- duplicate continuation = `book_dup(A1)` or `book_dup(A2)`
- the first divergent token is usually an early desc token
- CE pushes `person`
- UL lowers the duplicate continuation token(s)

### 12.2 Example B: duplicate book before another valid book

Raw rollout:

```text
[ book(A), book_dup(A1), book(B) ]
```

Sequential dedup gives:

```text
A = [ book(A), book(B) ]
D_1 = [ book_dup(A1) ]
```

Now the next valid continuation is also `book`.
So it would be wrong to blindly put unlikelihood on the first desc token `book`.

Instead:

- compare the duplicate continuation and the positive continuation token-by-token
- they share a long prefix (`{"desc":"book", ...}`)
- UL is applied only at the **first token where the duplicate path diverges from the valid next book**
- in practice this is often the first differing coord token

This is why v2 defines the negative target via **LCP divergence**, not via a hard-coded “first desc token” rule.

---

## 13. What v2 Explicitly Does Not Claim

- It does **not** solve unlabeled annotation in the core objective.
- It does **not** claim any global monotonicity guarantee for the multi-loss training recipe.
- It does **not** require RL.
- It does **not** require extra detection heads.
- It does **not** require span-level punishment of all unmatched objects.

Its claim is narrower and cleaner:

> A V-LLM detection model can keep generic unmatched predictions neutral, while still suppressing autoregressive near-duplication by removing duplicates from the positive prefix and applying unlikelihood at the first divergence point of the redundant continuation.

---

## 14. Summary

The entire v2 design can be remembered in one sentence:

> **Keep Stage-1 as the language-of-boxes foundation, keep Channel-A as the cheap geometry stabilizer, and change Channel-B from “all unmatched are neutral” to “generic unmatched stay neutral, but duplicate continuations are folded into clean-boundary unlikelihood.”**

That is the minimal upgrade that directly targets rollout instability without collapsing recall or introducing a second detection/objectness tower.
