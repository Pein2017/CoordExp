diff --git a/full_idea.md b/full_idea.md
index 1f6e9d1..b3a4c92 100644
--- a/full_idea.md
+++ b/full_idea.md
@@ -12,19 +12,36 @@
 ## 0. Core Design Principles
 
 ### 0.1 Constraints
 - Base model: pretrained V-LLM (Qwen3-VL).
 - Coordinate representation: existing `<|coord_k|>` tokens with `k ∈ {0..999}` (norm1000; 1000 bins, 0 = near edge, 999 = just inside far edge).
 - Do NOT introduce a separate detection head or DETR-like query decoder.
 - Training should remain compatible with standard SFT infrastructure (teacher forcing, token CE), with additional losses computed from LM logits.
 
 ### 0.2 What “EM-ish” means here
-- **E-step (latent alignment):** model generates a hypothesis output (rollout). Parse it into a set of predicted objects. Use Hungarian matching to align predictions to GT objects (latent correspondence).
-- **M-step (parameter update):** update model parameters using (i) standard SFT on reordered GT for generation/format, and (ii) differentiable geometric losses computed from the model’s logits under the model’s own rollout context (“self-context forward”).
+- We separate **continuous / smooth geometry calibration** from **discrete / set-level correction**:
+  - **Channel-A (hot path):** no autoregressive rollout; do **two parallel forwards** to approximate “self-context” *only on coord slots* using soft coord embeddings (differentiable, fast).
+  - **Channel-B (cold path):** occasional autoregressive rollout + Parse + Hungarian matching to handle **permutation / missing / extra / malformed-json** and to ground “true self-context” learning.
+
+Under this view:
+- **E-step (latent alignment):** performed only on Channel-B steps via rollout→set→Hungarian (stop-grad correspondence).
+- **M-step (parameter update):** always uses SFT-compatible losses; geometry losses can be applied under:
+  - Channel-A: “soft self-context proxy” (no sampling)
+  - Channel-B: “true self-context” (teacher-forced on rollout tokens)
 
 ### 0.3 Why “self-context forward”
 Hungarian matching is computed from the model’s rollout output. To make the geometric gradient consistent with the same context that produced that rollout, we recompute logits using teacher forcing on the rollout tokens (`ŷ`) and apply GT-based geometric supervision there.
 
 This avoids supervising coordinates under a mismatched context (pure GT teacher forcing) and behaves like a policy-improvement loop, but stays fully differentiable (no REINFORCE/PPO).
+
+**Update (practical):** in large V-LLMs, “true self-context forward” is expensive because it requires autoregressive rollout.
+We therefore introduce a **soft self-context proxy** that replaces only `<|coord_*|>` token embeddings by their expected embedding
+under the model’s own coord distribution, enabling a *non-sampling* approximation that is compatible with Qwen3’s tie-weights.
 
 ---
 
 ## 1. Output Format / Token Protocol
@@ -175,13 +192,18 @@
 ## 4. Full Training Pipeline (Multi-Stage)
 
 ### Overview
 We separate training into two conceptual stages:
 - **Stage-1:** Pure SFT / format stabilization / coord token adaptation.
-- **Stage-2:** EM-ish loop with rollout + matching + GT-supervised geometric calibration under self-context.
+- **Stage-2:** two-channel mixture:
+  - **Channel-A (default):** 2× forward (no rollout), geometry calibration under “soft self-context proxy” on coord slots.
+  - **Channel-B (sparse):** rollout + matching + GT-supervised geometric calibration under true self-context + reordered-GT SFT for set-level correction.
 
 ---
 
 ## 5. Stage-1: Standard SFT to “learn the language of boxes”
@@ -215,90 +237,205 @@
 
 ## 6. Stage-2: EM-ish Training Loop (Rollout → Match → Update)
 
-### 6.1 Per-sample loop (single image)
-
-#### Step A — Rollout (E-step hypothesis, no grad)
-1) Run autoregressive generation using current model θ:
-   - `ŷ = Rollout(f_θ, image, prompt)` (greedy or beam)
-2) Parse rollout into predicted objects:
-   - `Ô = Parse(ŷ)` returns a list:
-     - `Ô = { (d̂_i, b̂_i_disc, idx_i) }_{i=1..N}`
-   Where:
-  - `d̂_i`: decoded description string/tokens from `<desc> ... </desc>`
-  - `b̂_i_disc`: discrete box from the 4 coord tokens (ints 0..999, or normalized to [0,1])
-   - `idx_i`: token indices for the 4 coord positions in the rollout sequence (used later)
-
-> Notes:
-> - If parsing fails, optionally fall back to Stage-1 SFT only for that sample.
-
-#### Step B — Hungarian matching (E-step correspondence, no grad)
-3) Build cost matrix `C ∈ R^{N×M}` between predicted objects `i` and GT objects `j`:
-   - Geometry cost: `L_geo_disc(b̂_i_disc, b_j_gt)` (can be L1 on norm1000 + IoU-based term)
-   - Optional semantic cost: `D_sem(d̂_i, d_j_gt)` (use frozen text encoder; keep small early)
-   - `C_{i,j} = λ_geo * L_geo_disc + λ_sem * D_sem`
-4) Run Hungarian to get assignment:
-   - `σ = Hungarian(C)`
-   - Produces a set of matched pairs `(i -> j)`.
-
-5) Optional gating for stability:
-   - Only accept a match if (IoU_disc(b̂_i_disc, b_j_gt) > iou_thr) or (L1 < thr).
-   - Let `ValidMatched = { i | match accepted }`.
-
-> Purpose:
-> - Avoid wrong matches producing wrong geometric gradients.
+### 6.1 Two-channel schedule (recommended)
+We run Stage-2 as a **mixture** over steps / samples:
+- **Channel-A (hot path, e.g., 95% steps):** no rollout; always available and fast.
+- **Channel-B (cold path, e.g., 5% steps or 1/P batches):** true rollout + Hungarian to fix discrete failure modes.
+
+Rationale:
+- Your recent ablations suggest “traditional hard CE on coord tokens is already strong” in Stage-1/SFT.
+  So Stage-2 should **not** assume coord-distribution losses will magically dominate.
+- The *main* remaining instability is typically **self-context generation** (long JSON, permutation, missing/extras).
+  Channel-B targets that directly; Channel-A provides a cheap, smooth geometry signal to stabilize coord decoding and reduce drift.
+
+---
+
+### 6.2 Channel-A (hot): Soft self-context proxy via 2× forward (no rollout)
+
+Key idea: approximate “self-context” only where it matters most (coord slots), without sampling.
+We keep all **text/struct** tokens teacher-forced as GT, but feed **coord tokens** as *soft embeddings*
+derived from the model’s own coord distribution.
+
+#### Step A1 — Teacher-forced GT forward (with grad)
+1) Teacher force on GT (or reordered-GT if you already have one) to get logits:
+   - `z_t^gt = f_θ(image, prompt, y_GT_<t)`
+
+2) Identify coord positions `t ∈ coord_positions` from labels/template (packing-safe).
+   For each coord position `t`, gather coord-subspace logits:
+   - `s_t ∈ R^K`, `s_{t,k} = z_t^gt[coord_k]`
+   - `p_t = softmax(s_t / τ_A)` over `k=0..999`
+
+> Note: For stability, **detach** the soft distribution early:
+> - `p_t^stop = stopgrad(p_t)`
+> This keeps Channel-A “EM-ish”: the soft context is treated like an E-step artifact,
+> while gradients come from the second forward + the usual SFT CE.
+
+#### Step A2 — Build soft coord embeddings (no sampling)
+3) Let `E_coord ∈ R^{K×d}` be the embedding table restricted to `<|coord_k|>`.
+   Construct the expected coord embedding at each coord slot:
+   - `ē_t = Σ_k p_{t,k}^stop * E_coord[k]`
+
+4) Prepare `inputs_embeds` for a second forward:
+   - Start from standard embeddings of `y_GT` (or from the model’s embedding lookup).
+   - For coord positions, **replace** token embedding by `ē_t`.
+   - Keep attention mask unchanged.
+
+This is cheap because `K=1000` and coord slots are sparse compared to total tokens.
+
+#### Step A3 — Soft self-context forward (with grad)
+5) Second forward using the mixed embeddings:
+   - `z_t^soft = f_θ(image, prompt, inputs_embeds=ȳ_<t)`
+
+Interpretation: the model is now conditioned on “its own” coord belief (softly),
+while still anchored to GT text/structure → a controlled self-conditioning.
+
+#### Step A4 — Geometry + (optional) coord sharpening losses
+6) Decode continuous coords from `z_t^soft` at coord slots via CoordExp, assemble boxes per object.
+7) Apply geometry loss against GT boxes **by identity alignment** (no Hungarian in Channel-A):
+   - `L_geo_soft = Σ_obj [ α||b̂_obj^cont - b_obj^gt||_1 + β(1-GIoU(...)) ]`
+
+8) (Optional) add a *light* coord-distribution sharpening term computed from `z_t^soft`:
+   - `L_coordCE_soft = -Σ_{t∈coord_positions} log p_θ(y_t^gt | ... , softctx)`
+   - or your existing (softCE/W1/gate) but keep it small; treat as regularizer, not the main driver.
+
+Channel-A per-sample loss:
+- `L_A = L_CE(y_GT) + λ_geo^A L_geo_soft + λ_coord^A L_coord_sharp(optional)`
+
+> Why this is “unified” with rollout-matching:
+> - Channel-A trains geometry under a form of self-conditioning (but without sampling).
+> - Channel-B trains under true self-conditioning (rollout tokens), but sparsely.
+> Together they approximate “train on what you infer” while keeping throughput high.
 
 ---
 
 ## 7. Stage-2 Updates (M-step): Two complementary supervision paths
 
 Stage-2 consists of **two backprop-able losses** that target different failure modes:
 
-### 7.1 Path A: Geometric calibration under self-context (CoordExp + L1/GIoU)
+### 7.1 Channel-A (hot path): Geometric calibration under soft self-context (CoordExp + L1/GIoU)
 
-#### Step C — Self-context forward (with grad)
-6) Run a teacher-forced forward pass using **the rollout tokens** as context:
-   - Input tokens: `ŷ_{<t}` (rollout prefix)
-   - Compute logits for all positions:
-     - `z_t^self = f_θ(image, prompt, ŷ_<t)`
-   - This is standard teacher forcing, but the sequence is `ŷ` (model’s own outputs), not GT.
+Channel-A uses the **2× forward** construction from Section 6.2:
+- First forward: teacher-forced GT to obtain `p_t` over coord bins.
+- Second forward: replace coord token embeddings by expected embedding `ē_t` and forward again.
+
+We denote the second-forward logits as:
+- `z_t^self = z_t^soft`
 
 #### Step D — CoordExp decode on coord positions
 7) For each predicted object `i` and each of its 4 coord token positions `t ∈ idx_i`:
    - Gather coord logits: `s_{t,k} = z_t^self[coord_k]`
    - Softmax → `p_{t,k}`
    - Expectation → `ĉ_t`
 8) Assemble continuous predicted box:
    - `b̂_i_cont = (ĉ_{t_x1}, ĉ_{t_y1}, ĉ_{t_x2}, ĉ_{t_y2})`
 
-#### Step E — Geometric loss using matched GT (GT supervision)
-9) For each accepted match `(i -> j)` with `i ∈ ValidMatched`:
-   - `ℓ_geo(i,j) = α * || b̂_i_cont - b_j_gt ||_1 + β * (1 - GIoU(b̂_i_cont, b_j_gt))`
-10) Sum:
-   - `L_geo_self = Σ_{(i->j), i∈ValidMatched} ℓ_geo(i,j)`
+#### Step E — Geometric loss using GT (identity alignment)
+9) For Channel-A, object indices come from the GT template, so we use direct alignment:
+   - `ℓ_geo(i) = α * || b̂_i_cont - b_i_gt ||_1 + β * (1 - GIoU(b̂_i_cont, b_i_gt))`
+10) Sum:
+   - `L_geo_self = Σ_i ℓ_geo(i)`
 
 > Key property:
-> - Context tokens are `ŷ` (self context).
-> - Supervision target is GT box `b_j_gt`.
-> - No coordinate pseudo-labeling from `ŷ`.
+> - Context is **partially self-conditioned** (coord embeddings come from model belief).
+> - Supervision target is GT box.
+> - No sampling; stable gradients; throughput-friendly.
 
 ---
 
+### 7.2 Channel-B (cold path): Rollout-matching for set-level / discrete correction
+
+Channel-B is the original EM-ish loop, run sparsely to correct:
+- permutation/order mismatch,
+- missing GT objects / extra predicted objects,
+- malformed JSON / broken segments,
+- true “self-context degeneration” that Channel-A cannot expose.
+
+#### Step B0 — Rollout (E-step hypothesis, no grad)
+1) Run autoregressive generation using current model θ:
+   - `ŷ = Rollout(f_θ, image, prompt)` (greedy / low-T sampling / beam)
+2) Parse rollout into predicted objects:
+   - `Ô = Parse(ŷ)` returns:
+     - `Ô = { (d̂_i, b̂_i_disc, idx_i) }_{i=1..N}`
+
+> Notes:
+> - If parsing fails, fall back to Channel-A only for that sample, and log it as “invalid-rollout”.
+
+#### Step B1 — Hungarian matching (E-step correspondence, no grad)
+3) Build cost matrix `C ∈ R^{N×M}` between predicted objects `i` and GT objects `j`:
+   - Geometry cost: `L_geo_disc(b̂_i_disc, b_j_gt)` (L1 on norm1000 + IoU term)
+   - Optional semantic cost: `D_sem(d̂_i, d_j_gt)` (frozen encoder; keep small early)
+   - `C_{i,j} = λ_geo * L_geo_disc + λ_sem * D_sem`
+4) Hungarian:
+   - `σ = Hungarian(C)` → matched pairs `(i -> j)`
+5) Gating:
+   - accept only if IoU/L1 passes thresholds
+   - `ValidMatched = { i | match accepted }`
+
+#### Step B2 — True self-context forward (with grad)
+6) Teacher force on rollout tokens to recompute logits:
+   - `z_t^roll = f_θ(image, prompt, ŷ_<t)`
+
+7) CoordExp decode on coord positions `t ∈ idx_i`, assemble `b̂_i_cont`.
+
+8) Apply geo loss on matched GT targets:
+   - `L_geo_roll = Σ_{(i->j), i∈ValidMatched} [ α||b̂_i_cont - b_j_gt||_1 + β(1-GIoU) ]`
+
+#### Step B3 — Reordered-GT SFT (with grad)
+9) Build `y_GT_reordered` following predicted order + append missing GT objects.
+10) Standard SFT on `y_GT_reordered`:
+   - `L_CE_text`, `L_CE_coord(optional)`, `L_CE_eos(optional)`
+
+Channel-B per-sample loss:
+- `L_B = L_CE(y_GT_reordered) + λ_geo^B L_geo_roll + λ_coord^B L_coord_sharp(optional)`
+
 ### 7.2 Path B: Generation / format / recall control via reordered GT SFT
 
 This path handles:
 - Missing objects (not generated in rollout, thus no coord positions exist)
 - Extra objects (hallucinations)
 - Caption correctness and formatting robustness
@@ -309,20 +446,38 @@
 
 ## 8. Stage-2 Total Objective
 
 Combine both paths:
 
-- `L_stage2 = L_CE(y_GT_reordered) + λ_geo * L_geo_self + λ_coordCE * L_CE_coord + λ_ent * L_entropy(optional)`
+- Stage-2 is a mixture objective:
+  - `L_stage2 = (1-ρ) * L_A + ρ * L_B`
+  - where `ρ` is the Channel-B frequency/probability (small, e.g. 0.05).
+
+Expanded:
+- `L_A = L_CE(y_GT) + λ_geo^A * L_geo_soft + λ_coord^A * L_coord_sharp(optional)`
+- `L_B = L_CE(y_GT_reordered) + λ_geo^B * L_geo_roll + λ_coord^B * L_coord_sharp(optional)`
 
 Where:
-- `L_CE(y_GT_reordered)` includes text + structure + optional EOS weighting.
-- `L_geo_self` provides continuous geometric gradients under self context.
-- `L_CE_coord` sharpens coord distributions and reduces multi-peak ambiguity.
-- `L_entropy` (optional) regularizes coord distributions.
+- Channel-A geo term improves coord calibration cheaply under soft self-conditioning.
+- Channel-B terms fix true self-context + set-level discrete problems (order/missing/extras/format).
+
+Practical defaults (based on your current evidence that hard-CE is not weak):
+- Keep `L_CE` as the anchor in both channels.
+- Treat coord-distribution losses (softCE/W1/gate/entropy/top-k) as optional regularizers, not mandatory.
 
 ---
 
 ## 9. Inference Pipeline (Deployment)
@@ -346,6 +501,18 @@
 ## 10. Implementation Checklist (What must exist)
 
 ### 10.1 Token / formatting utilities
 - Template builder: GT objects → token sequence (`y_GT`)
 - Parser: predicted sequence → list of objects + coord token indices
 - Coord vocab index list: indices of `<|coord_0|>.. <|coord_999|>` in tokenizer
+
+### 10.1.1 (New) Soft self-context utilities (Channel-A)
+- A packing-safe way to get `coord_positions` (from labels/template).
+- `E_coord` gather: embedding rows for coord token IDs (K=1000).
+- Build `inputs_embeds` where coord positions are replaced by `ē_t = Σ_k p_{t,k} * E_coord[k]`.
+- Control whether `p_t` is stop-grad (recommended early) via config flag.
+- Model forward must support `inputs_embeds` (Qwen3-VL does; ensure your trainer path passes it correctly).
 
 ### 10.2 Matching utilities
 - Cost matrix builder:
   - geometry cost from discrete coords (fast IoU/L1)
   - optional semantic cost from frozen text embedding model
@@ -380,10 +547,14 @@
 
 ### Stage-2 (EM-ish improvements)
 - Add rollout + matching + self-context geometric calibration.
 - Use gating and curriculum:
   - early: geometry-only matching, strict gating
   - later: relax gating; optionally add semantic term
 - Evaluate:
   - localization improvement (IoU distribution, GIoU loss reduction)
   - robustness to permutation/order
   - reduced sensitivity to quantization boundary cases
+
+**Update:** run Stage-2 as Channel-A/Channel-B mixture:
+- Channel-A provides high-throughput, stable geometry calibration without rollout.
+- Channel-B provides sparse but essential “true self-context + set correction”.
 
 ### Optional Stage-3 (light rollout consistency)
 - Low-frequency additional rollouts with the same machinery.
 - Focus on structure validity and geometric calibration; avoid heavy RL.
