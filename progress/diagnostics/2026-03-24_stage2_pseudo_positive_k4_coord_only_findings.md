---
title: Stage-2 K=4 Pseudo-Positive Findings (Coord-Only, Prefix Structure Disabled)
status: active-diagnostic
scope: stage2-channel-b
topics: [stage2, channel-b, pseudo-positive, k4, train-dynamics, recall, duplication]
references:
  - docs/PROJECT_CONTEXT.md
  - docs/training/STAGE2_RUNBOOK.md
  - progress/diagnostics/2026-03-12_stage2_triage_posterior_coco1024_train_dynamics.md
  - progress/diagnostics/2026-03-05_stage2_near_duplication.md
---

# Stage-2 K=4 Pseudo-Positive Findings (Coord-Only, Prefix Structure Disabled) (2026-03-24)

Date: 2026-03-24  
Status note: this note records the first qualitative + train-side diagnostic read of the new `3+1`
Channel-B pseudo-positive setup (`1` anchor + `3` explorers) under the current prod profile.

The short version is:

- the new `K=4` pseudo-positive path is alive and stable enough to train,
- the actual active supervision is **coord-only** for pseudo-positive prefix objects,
- the expected pseudo-positive prefix `structure_ce` is **not** active in the current implementation (will be fixed in later runs),
- the configured duplicate-burst unlikelihood penalty is present in the objective, but it is effectively **inactive** on the
  inspected windows,
- and the reviewed "suspicious duplication" cases do **not** support a story that the new run is mostly a
  dead-duplicate collapse.

The deeper finding is that the current algorithm is still much better at **refining / retaining existing anchor
hypotheses** than at improving **object birth / recall**, especially for crowded repeated classes like
`person`.

---

## 0) Artifacts

Run:

- Run dir:
  `output/stage2_ab/prod/pseudo_positive/k_4-eff_size_96-b_ratio_0.75-epoch_1/v2-20260324-062041/`
- Train log:
  `.../logging.jsonl`
- Monitor dumps:
  `.../monitor_dumps/step_000002.json`
  `.../monitor_dumps/step_000040.json`
  `.../monitor_dumps/step_000080.json`
- Config:
  `configs/stage2_two_channel/prod/ab_mixed_coco1024_bmajority_channel_b_pseudo_positive.yaml`

Rendered suspicious-case review pack:

- Review dir:
  `.../analysis/suspicious_duplication_review/`
- Manifest:
  `.../analysis/suspicious_duplication_review/manifest.json`
- PNGs:
  `.../analysis/suspicious_duplication_review/review_pngs/vis_0000.png`
  `.../analysis/suspicious_duplication_review/review_pngs/vis_0001.png`
  `.../analysis/suspicious_duplication_review/review_pngs/vis_0002.png`
  `.../analysis/suspicious_duplication_review/review_pngs/vis_0003.png`

Resolved config surfaces relevant to interpretation:

- `stage2_ab.schedule.b_ratio = 0.75`
- `stage2_ab.channel_b.triage_posterior.num_rollouts = 4`
- `stage2_ab.channel_b.pseudo_positive.enabled = true`
- `stage2_ab.channel_b.pseudo_positive.coord_weight = 0.5`
- `stage2_ab.channel_b.duplicate_iou_threshold = 0.95`
- `stage2_ab.channel_b.triage_posterior.recovered_ground_truth_weight_multiplier = 2.0`
- `stage2_ab.pipeline.objective.loss_duplicate_burst_unlikelihood.weight = 2.0`
- `bbox_geo.ciou_weight = 0.5`
- `bbox_size_aux.log_wh_weight = 0.05`
- `bbox_size_aux.oversize_penalty_weight = 0.0`

So the run keeps the standard `ciou` + `bbox_size_aux` geometry regularization surface while adding the
new `K=4` pseudo-positive evidence path.

---

## 1) What The Current Objective Actually Trains

The current training contract is narrower than the intended mental model in one important way.

### 1.1 Pseudo-positive prefix objects are coord-only

In the current implementation:

- pseudo-positive prefix objects add weighted bbox groups in
  `src/trainers/stage2_two_channel/target_builder.py`,
- those weights flow into `bbox_geo`, `bbox_size_aux`, and `coord_reg`,
- but they do **not** add positive prefix text supervision.

This means the new pseudo-positive path currently teaches:

- box geometry,
- coord token distributions,
- and bbox-size regularization

for selected prefix objects, but **not**:

- object description CE,
- nor prefix structure CE for those pseudo-positive objects.

### 1.2 Prefix structure CE remains matched-only

The active `token_ce` path in `src/trainers/teacher_forcing/modules/token_ce.py` still behaves as:

- Channel-B prefix structure CE:
  matched-prefix only
- Channel-B tail desc CE:
  FN-append objects only
- pseudo-positive prefix objects:
  no positive text CE

This was unexpected relative to the informal assumption that pseudo-positive objects were also getting
structure supervision. They are not.

### 1.3 Dead-anchor suppression is configured, but not contributing

The objective includes:

- `loss_duplicate_burst_unlikelihood.weight = 2.0`

However, the observed run-side telemetry still shows:

- `train/optimization/loss_duplicate_burst_unlikelihood = 0.0`
- `diag/duplicate_burst/num_terms = 0.0`
- `diag/duplicate_burst/num_ul_boundaries = 0.0`

and the sampled monitor dumps continue to show:

- `duplicate_burst_unlikelihood_boundary_count = 0`
- `duplicate_burst_unlikelihood_skipped_no_divergence = 0`

So the negative duplicate-dead branch penalty is currently not a real contributor to the training dynamics.

---

## 2) Why The "Suspicious Duplication" Review Was Misleading If Read Too Literally

The current train monitor dump is not a neutral sample of errors.

It is explicitly selected by the trainer as:

- `selection = suspicious_duplication`

and the sorter ranks samples by duplication-style signals derived from raw anchor predictions, including:

- `duplicates`
- `near_iou90_pairs_same_desc_count`
- `duplicate_bursts`
- `near_iou90_pairs_any_desc_count`
- saturation-like repetition metrics

So the monitor pack is useful for surfacing one class of pathology, but it is **not** the right sample
distribution to answer:

- "what is the main failure mode of the run?"
- or "should we tune toward stronger dead-negative penalties?"

That matters here, because the rendered cases suggest:

- `vis_0000` is a true catastrophic collapse (`apple` enumeration),
- but the remaining rendered cases are mostly:
  - real crowded scenes,
  - low recall on clear repeated classes,
  - or mild overprediction that is not severe enough to justify a strong dead-negative redesign.

In other words:

- the current monitor pack over-samples repeated-class scenes,
- but that does **not** mean the model is mostly making dead-anchor duplicate errors.

---

## 3) Main Diagnostic Findings

### 3.1 The dominant issue is still recall, not dead-duplicate collapse

After visual review of the suspicious pack:

- one sample is a real collapse case,
- the others are mostly real dense scenes with imperfect cardinality,
- especially repeated clear classes like `person`.

So the stronger statement is:

- the current `K=4` pseudo-positive run is **not mainly a duplicate-burst failure**,
- it is mainly a **recall-limited / object-birth-limited** regime with some acceptable extra predictions.

### 3.2 The current pseudo-positive design mostly helps existing anchor hypotheses

The current pseudo-positive candidate pool is still anchor-centric:

- candidates begin from unmatched anchor clean objects,
- explorer views provide support evidence,
- but explorer-only non-GT-backed objects are not promoted into training targets.

This means the current unlabeled path is good at:

- keeping or geometrically refining objects the anchor already proposed,

but weak at:

- creating training signal for real objects that only the explorers see.

### 3.3 Recovered GT tail append is useful, but structurally limited

The current recovered-GT path still matters:

- explorer-found / anchor-missed GT objects become FN-appended tail objects,
- they receive coord supervision and desc CE,
- and their weight is controlled by
  `recovered_ground_truth_weight_multiplier`.

But this path is still limited because it teaches those misses as **tail additions**, not as part of the
main ordered anchor prefix. So it is a weaker mechanism for improving:

- early object birth,
- stable per-object ordering,
- and crowded-scene recall in the natural prefix regime.

### 3.4 The current run is more "coord-correction" than "count/structure expansion"

Putting the pieces together:

- pseudo-positive prefix objects only get coord-side learning,
- matched prefix keeps the structure CE,
- recovered GT tail carries desc CE,
- duplicate-burst unlikelihood is inactive,

so the overall learning pressure is biased toward:

- "fix / preserve geometry for objects already present"

more than toward:

- "emit more correct objects in the right ordered sequence."

That aligns with the observed qualitative failure profile.

---

## 4) Implications For Next Training Iterations

### 4.1 Dead-anchor suppression should not be the main tuning lever right now

Given the rendered pack and inactive suppression telemetry:

- it is reasonable to keep duplicate-burst unlikelihood only as a guardrail for rare collapse cases,
- but it should not drive the next major setup decision.

### 4.2 The more important missing piece is recall-oriented text/structure supervision

Two concrete gaps stand out:

1. pseudo-positive prefix objects are not structure-supervised,
2. explorer-only non-GT-backed objects cannot currently create pseudo labels.

So the most promising next directions are:

- add structure-only prefix supervision for selected pseudo-positive objects,
- or allow high-support explorer-consensus objects to become pseudo-FN-style tail additions,
- or both.

### 4.3 We need a recall-oriented monitor pack, not only a duplication-oriented one

The current diagnostics surface should be extended with a second monitor selection that prioritizes:

- low recall,
- high FN count,
- high recovered-GT count,
- or large anchor/explorer recall gaps.

Without that, we will keep over-reading repeated-class scenes through a duplication lens.

---

## 5) Working Conclusion

For the current `3+1` pseudo-positive experiment:

- the run is operationally healthy,
- the new pseudo-positive path is active,
- the active pseudo-positive learning is coord-only,
- prefix structure CE for pseudo-positive objects is currently absent,
- duplicate-burst unlikelihood is not materially affecting the run,
- and the main remaining bottleneck appears to be **recall / sequence object birth**, not broad dead-duplicate collapse.

So the next research step should be framed as:

- "how do we convert explorer evidence into stronger recall supervision?"

not primarily as:

- "how do we punish duplicate bursts harder?"

