# Stage-2 (Channel-A only) Diagnosis: Coord Loss Dynamics from Monitor Dumps (2026-02-25)

This note focuses on *what the model is actually doing* in crowded scenes, using monitor-dump visualizations as the primary evidence source (loss curves are treated as supporting signals).

Run under analysis:
- Artifact root: `output/stage2_ab/coco_bbox_max60/a_only/epoch_2-eff_size_64-n_softctx_iter_2-`
- Example monitor dumps (step 900):
  - `.../monitor_dumps/vis_step_000900/step_000900_s08_base000002.png` (bookshelf)
  - `.../monitor_dumps/vis_step_000900/step_000900_s06_base000354.png` (sport court / audience)

Config reference:
- Baseline: `configs/stage2_two_channel/prod/desc_first_a_only.yaml`
  - `b_ratio=0.0` (Channel-A only)
  - `n_softctx_iter=2`
  - `softctx_grad_mode=unroll` (default)
  - Coord ctx embedding: `coord_ctx_embed_mode=st` (default)
  - Geometry decode: `coord_decode_mode=exp` (default)

---

## 1) What the monitor dumps show (image-first)

### 1.1 Crowded “bookshelf” scene (high FP + high FN)
Observed (monitor overlay):
- `gt=17`, `pred=43`, `matched=5` → `p=0.116`, `r=0.294`, `f1=0.167`
- **FP explosion**: `FP(pred)=38` dominated by `book`.
- **FN remains large**: `FN(gt)=12` dominated by `book`.

Qualitative failure mode:
- Predicted “book” boxes are frequently **long horizontal strips** covering *many books at once* (group-box behavior).
- Many predictions are “near duplicated” around the shelf region.

Interpretation:
- This looks less like “model can’t see books” and more like **instance separation / coordinate precision failure** under heavy density:
  - the model “knows” there are many books,
  - but it does not commit to *atomic per-instance boxes* (and/or its decoded coordinates are too diffuse / averaged).

Important caveat (label noise):
- This is a classic COCO-style **unlabeled-instance** regime: many “FP” are plausibly real books that are simply not annotated.
  - Therefore, `rollout/f1` can be pessimistic here even when visual behavior is reasonable.
  - However, the **group-box** pattern is still undesirable even under unlabeled noise (it is not atomic and tends to hurt matching).

### 1.2 Sport court scene (long flat boxes in the audience)
Observed:
- `gt=29`, `pred=33`, `matched=11` → `p=0.333`, `r=0.379`, `f1=0.355`
- `FN(gt)=18` (includes persons + benches)
- `FP(pred)=22` (dominated by persons)

Qualitative failure mode:
- Predicted boxes over the audience become **long, flat, row-like** shapes covering multiple spectators.
- Many additional “person” predictions in stands likely correspond to unlabeled people, but their **geometry is not instance-atomic**.

Interpretation:
- The model has “semantic intent” (it wants to output people), but coordinates are not stable/atomic in dense regions.
- The “long strip” boxes are exactly the pattern that the prompt already discourages; this strongly suggests it is **not a prompt-only problem**.

---

## 2) Why coord/geo loss can be hard to lower (neural dynamics view)

Channel-A uses iterative soft self-context:

Let:
- A1 logits: `z₀ = f_θ(x, e_gt)` (GT token ids, but coord-token *embeddings* later get replaced)
- Self-context embedding (ST by default): `e₁ = g(z₀)` where `g` uses argmax+ST on coord bins
- A2 logits: `z₁ = f_θ(x, e₁)`
- Main coord/geo objectives are applied on the A2 forward.

Key dynamic:
- The **context for A2 is a function of A1 predictions**.
- If A1 coord beliefs are diffuse/multimodal (common in crowded scenes), then:
  - ST embedding selects a *single hard-ish* coordinate embedding per slot,
  - but that selection can be unstable across steps/samples (high variance),
  - which makes A2 coordinate gradients noisy.

This creates an optimization loop where:
- token/format CE continues to improve steadily (easier, dense supervision),
- but bbox/coord objectives see high variance gradients and appear “shaky” (especially `bbox_geo`, since it depends on a decoded coordinate expectation).

Why “long flat group boxes” are a natural attractor under this dynamic:
- With expectation-based geometry (`coord_decode_mode=exp`), a **multimodal distribution** can decode to an “average” coordinate.
- Averaging endpoints across multiple plausible instances tends to create **large, flat boxes** (covering several instances).
- Those boxes can be “visually plausible” yet systematically fail IoU matching, so FN/FP remain high in crowded regions.

---

## 3) Design decision: introduce coord/geo supervision in A1?

### Recommendation (evidence-based)
Yes — introduce a **weak A1 anchor** for:
- `bbox_geo` (SmoothL1 + CIoU) and
- `coord_reg` (SoftCE + W1 only; keep hard coord CE/gates disabled for A1).

Rationale:
- A1 is the seed for self-context. If the seed is noisy, A2 learns under a distribution shift it created itself.
- A weak A1 anchor reduces seed noise → reduces self-context drift → improves coord stability in crowded scenes.

Why *not* hard coord token CE in A1 (initially):
- Hard CE can encourage overly peaky / brittle distributions early, which can amplify ST-embedding discontinuities.
- SoftCE/W1 is a softer “shape prior” that tends to stabilize without forcing a single-bin collapse.

---

## 4) Actionable next experiment (config-first)

New config (inherits the baseline and only turns on small A1 anchors):
- `configs/stage2_two_channel/prod/desc_first_a_only_a1_anchor_small.yaml`

Key knobs:
- `bbox_geo.config.a1_smoothl1_weight: 0.2`
- `bbox_geo.config.a1_ciou_weight: 0.02`
- `coord_reg.config.a1_soft_ce_weight: 0.02`
- `coord_reg.config.a1_w1_weight: 0.02`

Expected outcome (what to look for):
- Fewer “long strip” / group-box predictions in dense regions.
- More stable per-step bbox_geo telemetry (lower variance), *even if* absolute loss values remain noisy.
- Qualitatively: more atomic boxes converting some FN→matched (not necessarily changing raw “FP” count in unlabeled regions).

Verification:
- Compare monitor dumps every N steps on the *same fixed sample IDs* (bookshelf, audience) across:
  - baseline: `desc_first_a_only.yaml`
  - anchor: `desc_first_a_only_a1_anchor_small.yaml`
- Focus on: instance-atomicity + duplicate rate + group-box rate (visual).

---

## 5) If FN remains high / group boxes persist (next knobs to ablate)

These are *second-stage* knobs after the A1 anchor sanity check:

1) **Increase CIoU influence** (discourage extreme aspect ratios)
   - Raise `bbox_geo.config.ciou_weight` (or reduce SmoothL1) to penalize long strips more strongly.

2) **Sharpen coord distributions**
   - Slightly raise `coord_reg.config.coord_ce_weight` (hard CE) *or*
   - lower `coord_reg.config.target_sigma` / temperature (careful: can destabilize).

3) **Align training decode with discrete inference**
   - Try `stage2_ab.coord_decode_mode: st` (geometry loss uses straight-through argmax-ish decode).
   - Hypothesis: reduces “averaging across instances” behavior.

4) **unroll vs em_detach**
   - `unroll` gives better credit assignment through the self-context loop, but can be more unstable.
   - `em_detach` trades bias for variance reduction; useful if bbox loss is too noisy.

---

## 6) Metric divergence note: why f1 can move opposite to mAP

In crowded/unlabeled regimes:
- `rollout/f1` is highly sensitive to “extra predictions” that are unlabeled → counted as FP.
- COCO `mAP` can increase if localization/ranking improves on *labeled* instances, even if the model also becomes more willing to output additional plausible instances.

Therefore:
- Use monitor dumps + mAP together to decide whether behavior improved.
- Treat f1 drops in dense “unlabeled” scenes as ambiguous without qualitative inspection.

