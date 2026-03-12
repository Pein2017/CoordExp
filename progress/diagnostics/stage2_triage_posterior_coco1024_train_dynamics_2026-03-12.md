---
title: Stage-2 AB COCO1024 Triage-Posterior Train Dynamics Through Step 300
status: active-diagnostic
scope: stage2-channel-b
topics: [stage2, channel-b, triage-posterior, train-dynamics, rollout-instability, resolution-1024]
references:
  - docs/PROJECT_CONTEXT.md
  - docs/training/STAGE2_DESIGN.md
  - docs/training/STAGE2_RUNBOOK.md
  - progress/directions/full_idea_v3.md
  - progress/diagnostics/stage2_ul_capture_highres1024_2026-03-09.md
---

# Stage-2 AB COCO1024 Triage-Posterior Train Dynamics Through Step 300 (2026-03-12)

Date: 2026-03-12  
Status note: this note records the run through the first eval at `global_step = 300` and includes an artifact note for the malformed final `logging.jsonl` row.

The short version is:

- the run shows real learning on the train-side rollout canaries,
- the model has become much shorter, cleaner, and more precise than the early windows,
- but the tail is still **oscillatory** rather than fully settled,
- the remaining instability appears as **intermittent concentrated same-desc duplicate bursts**, not as the earlier global rollout-length explosion,
- and the first eval confirms a **precision-heavy / recall-limited** endpoint rather than a broad recall expansion.

This note should be read together with:

- the stable v3 design summary in `docs/training/STAGE2_DESIGN.md`,
- the long-form direction in `progress/directions/full_idea_v3.md`,
- and the earlier high-res Stage-2 diagnostic in `progress/diagnostics/stage2_ul_capture_highres1024_2026-03-09.md`.

---

## 0) Artifacts

Run:

- Run dir:
  `output/stage2_ab/prod/ab_mixed_coco1024_bmajority_channel_b_triage_posterior/eff_size_96-b_ratio_0.75-triage_posterior-epoch_1/v0-20260311-165111/`
- Train log:
  `.../logging.jsonl`
- Config:
  `configs/stage2_two_channel/prod/ab_mixed_coco1024_bmajority_channel_b_triage_posterior.yaml`
- Init model:
  `output/stage2_ab/prod/ul-res_1024-v2-ckpt_300_merged`

Resolved config surfaces relevant to interpretation:

- `training.logging_steps = 10`
- `training.eval_steps = 300`
- `stage2_ab.schedule.b_ratio = 0.75`
- `stage2_ab.channel_b.triage_posterior.num_rollouts = 2`
- `stage2_ab.channel_b.triage_posterior.explorer_temperature = 0.7`
- `stage2_ab.channel_b.triage_posterior.explorer_top_p = 0.95`
- `stage2_ab.channel_b.triage_posterior.explorer_top_k = 64`
- `stage2_ab.channel_b.triage_posterior.unlabeled_consistent_iou_threshold = 0.9`
- `stage2_ab.channel_b.triage_posterior.recovered_ground_truth_weight_multiplier = 2.0`
- `rollout_matching.train_monitor_dump.every_steps = 40`

Important artifact caveat:

- this run still uses the older telemetry surface,
- there is now a normal train row at step `300` and a normal eval row at step `300`,
- there is still **no `monitor_dumps/` directory** under the run dir despite the config enabling train dumps,
- the log still lacks the later explicit `train/triage/*` and anchor-vs-explorer split metrics,
- and the final line in `logging.jsonl` is **not** a flat metric row:
  it is a final trainer-state append carrying `last_model_checkpoint`, `best_model_checkpoint`,
  `best_metric`, `global_step`, `log_history`, and `memory`.

So the analysis below is necessarily **train-window behavioral diagnosis**, not a fully instrumented v3 telemetry read.

Malformed-row provenance:

- the bad final append is not produced by CoordExp code,
- it is appended by the external ms-swift trainer stack in `swift/llm/train/sft.py::_save_trainer_state`,
- where the trainer state payload is added to the same `logging.jsonl` stream via `append_to_jsonl(...)`.

---

## 1) Config Intent vs What The Log Can Actually Show

The intended v3 contract for this run is:

- Channel-B active most of the time via `b_ratio = 0.75`,
- `K = 2` rollout with a deterministic anchor plus one stochastic explorer,
- explorer decode at `temperature = 0.7`, `top_p = 0.95`, `top_k = 64`,
- triage into GT-backed / unlabeled-consistent / dead-style categories before building the clean training target.

However, this specific artifact still predates the later telemetry repair. In practice, the log only exposes:

- aggregate `rollout/*` train canaries,
- duplicate-collapse counters under `dup/*` and `stage2_ab/channel_b/dup/*`,
- and the single aggregate dead-anchor suppression loss under
  `train/optimization/loss_dead_anchor_suppression`.

That means we can judge:

- whether the run is becoming cleaner or more unstable,
- whether duplication is broad or concentrated,
- whether supervision efficiency is improving,
- and whether the model is drifting toward over-conservative behavior.

But we cannot directly answer:

- whether anchor or explorer is responsible for a bad window,
- whether the burst mass is mostly dead-anchor vs explorer-only dead,
- or whether recovered-GT behavior is improving.

---

## 2) High-Level Trajectory

The run has gone through five distinct train-side phases so far:

| Phase | Steps | F1 | Precision | Recall | Pred / sample | Gen tokens mean | Trunc rate | Near-IoU90 any-desc | `N_duplicates` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Early stabilization | `10-40` | `0.355` | `0.232` | `0.785` | `26.85` | `831.46` | `0.161` | `1108.8` | `9.3` |
| First clean regime | `120-140` | `0.528` | `0.388` | `0.830` | `15.84` | `395.45` | `0.024` | `188.0` | `18.0` |
| Burst regime | `150-180` | `0.528` | `0.389` | `0.826` | `15.26` | `382.48` | `0.034` | `5607.8` | `163.0` |
| Recovery regime | `190-240` | `0.552` | `0.426` | `0.798` | `14.37` | `358.56` | `0.019` | `87.8` | `29.7` |
| New tail | `250-280` | `0.594` | `0.484` | `0.772` | `11.82` | `303.35` | `0.013` | `1858.5` | `57.0` |

Two points stand out:

1. The model is clearly learning to shorten and regularize its rollouts.
2. The remaining instability is now episodic and concentrated, not diffuse.

Compared with the early `10-40` phase, the new tail `250-280` is:

- much shorter: `1148.30 -> 303.35` tokens on the mean rollout length proxy,
- far less truncated: `0.229 -> 0.013` at the single-step extrema, and `0.161 -> 0.013` by phase mean,
- much more precise: `0.181 -> 0.484` at the phase edges,
- and much less over-predictive: `32.33 -> 11.82` predictions per sample by phase-edge comparison.

So the basic answer to “is the model learning?” is **yes**.

---

## 3) Key Step Snapshots

### 3.1 Step 10: the initial failure mode is global over-generation

At step `10`:

- `rollout/f1 = 0.2882`
- `rollout/precision = 0.1814`
- `rollout/recall = 0.7011`
- `rollout/pred_per_sample = 32.33`
- `rollout/gen_new_tokens_mean = 1148.30`
- `rollout/parse_truncated_rate = 0.2292`
- `stage2/invalid_rollout = 10`

Interpretation:

- the model is still in the classic Stage-2 bad regime:
  long, noisy, over-complete rollouts with substantial truncation pressure.

### 3.2 Step 130: first clearly healthy train-side window

At step `130`:

- `rollout/f1 = 0.5830`
- `rollout/precision = 0.4481`
- `rollout/recall = 0.8342`
- `rollout/pred_per_sample = 11.34`
- `rollout/parse_truncated_rate = 0.0104`
- `dup/near_iou90_pairs_any_desc_count = 94`
- `stage2_ab/channel_b/dup/N_duplicates = 4`

Interpretation:

- this is the first clean operating point in the run,
- and it shows that the run can produce compact, low-overlap, high-utility rollouts under the current objective.

### 3.3 Steps 150-180: late duplicate-burst instability

The burst regime is not a return to early global chaos.
Instead it is a new, narrower pathology:

- step `150`: `N_duplicates = 174`, `same_desc_pairs = 6098`
- step `170`: `N_duplicates = 154`, `same_desc_pairs = 4681`
- step `180`: `N_duplicates = 254`, `same_desc_pairs = 10384`

Important detail:

- prediction volume is no longer huge in these windows,
- but duplicate burden inside a small number of loci becomes extreme.

This is why the phase looks unstable despite acceptable F1:

- the run is no longer failing by everywhere-overgeneration,
- it is failing by **concentrated repeated same-desc continuation bursts**.

### 3.4 Step 270: strongest window so far

At step `270`:

- `rollout/f1 = 0.6758`
- `rollout/precision = 0.5902`
- `rollout/recall = 0.7904`
- `rollout/pred_per_sample = 9.58`
- `rollout/gen_new_tokens_mean = 242.39`
- `rollout/parse_truncated_rate = 0.0104`
- `dup/near_iou90_pairs_any_desc_count = 26`
- `stage2_ab/channel_b/dup/N_duplicates = 4`

This is the best row in the log so far for:

- `rollout/f1`,
- `rollout/precision`,
- shortest prediction count,
- shortest rollout length,
- and lowest overlap burden.

Interpretation:

- the model can now enter a very strong clean regime,
- so the objective is not merely preventing catastrophe,
- it is sometimes pushing the model into genuinely good rollout behavior.

### 3.5 Step 280: the oscillation has not disappeared

At step `280`:

- `rollout/f1 = 0.5228`
- `rollout/precision = 0.4166`
- `rollout/recall = 0.7017`
- `dup/near_iou90_pairs_any_desc_count = 2609`
- `dup/near_iou90_pairs_same_desc_count = 1954`
- `stage2_ab/channel_b/dup/N_duplicates = 79`

Interpretation:

- a clean window like step `270` can still be followed immediately by a noticeably worse bursty window,
- so the tail is still **oscillatory** rather than stably converged.

### 3.6 Step 300: clean train window, conservative eval

At the final train row for step `300`:

- `rollout/f1 = 0.6249`
- `rollout/precision = 0.5175`
- `rollout/recall = 0.7885`
- `rollout/pred_per_sample = 11.33`
- `rollout/gen_new_tokens_mean = 286.83`
- `rollout/parse_truncated_rate = 0.0`
- `dup/near_iou90_pairs_any_desc_count = 8`
- `dup/near_iou90_pairs_same_desc_count = 1`
- `stage2_ab/channel_b/dup/N_duplicates = 1`

Interpretation:

- by the final train row, the duplicate-burst pathology is almost absent,
- and the train-side rollout canaries end in a clearly clean regime.

At the first eval row for step `300`:

- `eval/detection/mAP = 0.4424`
- `eval/detection/precision = 0.7804`
- `eval/detection/recall = 0.6533`
- `eval/detection/f1 = 0.7112`
- `eval/detection/pred_objects = 2850`
- `eval/detection/gt_objects_total = 3404`
- `eval/parsing/parse_truncated_rate = 0.0`
- `eval/parsing/parse_dropped_invalid = 0.0`

Interpretation:

- eval-time generation is clean and parse-stable,
- the model is not in an over-generation regime at eval,
- but the endpoint is clearly **precision-heavy and recall-limited**,
- with `pred / gt ≈ 0.837`, meaning the model is under-predicting rather than over-predicting.

---

## 4) Learning Performance

### 4.1 Optimization dynamics are improving in a meaningful way

Across the observed train log:

- total `loss` improves from `0.667` at step `10` to `0.556` at step `280`,
- `loss/B_rollout_text/struct_ce` falls strongly over the run,
- `coord_diag/B/expected_bin_mae` improves from `27.16` at step `10` to a best of `16.49` at step `160`,
- and rollout length plus truncation both come down sharply.

This is not a fake “lower loss but same behavior” story.
The behavioral canaries do improve alongside the optimization scalars.

### 4.2 The biggest quality gain is in compactness and precision

The model’s main progress is:

- much lower prediction count,
- much lower rollout length,
- much lower truncation pressure,
- much higher precision,
- and a much higher matched-for-supervision share of valid predictions.

In practical terms, the model is learning:

- to stop talking so long,
- to emit fewer obviously bad objects,
- and to produce a larger fraction of predictions that survive matching into useful supervision.

### 4.3 The run is getting more conservative

There is, however, a real tradeoff in the tail:

- precision improves in the new `250-280` window,
- but mean recall is lower than in the earlier `190-240` recovery phase.

So the current trajectory is:

- better cleaned-up rollouts,
- but with a risk of over-pruning if the trend continues without recovery by step `300+`.

That does not make the trend bad.
It just means the current optimization path is emphasizing precision and compactness more strongly than recall.

The step-300 eval confirms that this is not just a train-window artifact:

- the final eval precision is strong (`0.7804`),
- recall is modest (`0.6533`),
- and the model emits fewer boxes than GT on average (`2850` vs `3404`).

So the conservative drift is now visible at evaluation time as well.

---

## 5) Rollout Instability

### 5.1 The residual instability is not the old failure mode

Earlier Stage-2 bad windows were dominated by:

- very long rollouts,
- truncation,
- invalid parse pressure,
- and broad over-generation.

This run still shows that pattern early, but the late instability is different.

The current late-phase failure mode is:

- short-to-moderate rollout length,
- low truncation,
- but intermittent spikes in **same-desc near-IoU90 duplicate mass**.

That matters because it changes what kind of fix is likely to help.

### 5.2 The bursts are concentrated rather than widespread

At step `250`:

- `N_duplicate_bursts = 10`
- `N_duplicates = 110`

At step `280`:

- `N_duplicate_bursts = 10`
- `N_duplicates = 79`

This means the number of problematic continuation sites is not enormous.
Instead, a relatively small number of loci are generating many repeated duplicate objects.

Operationally, that is a better failure mode than early diffuse chaos:

- it is narrower,
- easier to target,
- and more consistent with a late-training self-collision bug than with a global model-collapse story.

### 5.3 Same-desc overlap dominates the bad windows

The clearest sign that the late bursts are “repeat the same object again” rather than generic scene confusion is the same-desc share:

- step `250`: `4677 / 4694 = 0.996`
- step `280`: `1954 / 2609 = 0.749`

So the problematic mass in the worst new rows is mostly:

- exact or near-exact same-desc regional repetition,
- not broad any-desc overlap across many semantic alternatives.

### 5.4 Dead-anchor suppression loss does not track burst severity cleanly

The scalar
`train/optimization/loss_dead_anchor_suppression`
keeps shrinking into the tail:

- step `180`: `0.2623`
- step `250`: `0.1732`
- step `270`: `0.1004`
- step `280`: `0.0832`

Yet steps `250` and `280` still show major duplicate bursts.

Interpretation:

- the aggregate suppression loss is not a reliable proxy for burst severity in this artifact,
- likely because the old log surface is collapsing many terms into one mean-like scalar without the needed denominators.

This is exactly the observability gap the later telemetry patch was meant to close.

---

## 6) Current Judgment

If the question is:

> “Does this run look like it is learning in a good direction?”

the answer is:

**yes, but the tail is still oscillatory.**

More precisely:

- the learning trend is good,
- the best windows are clearly getting better,
- the remaining failure mode is narrower and more concentrated than the old one,
- the final train row is much cleaner than the bursty middle,
- but the run is not yet stably converged because the clean regime can still be followed by duplicate-burst regressions,
- and the first eval shows the endpoint has become more conservative than recall-maximizing.

If the question is:

> “Would I describe the run as healthy enough to keep?”

the answer is also **yes**.

If the question is:

> “Would I describe it as settled enough to stop diagnosing?”

the answer is **no**.

The main pending uncertainty is no longer “what happens at the first eval?”
We now know that the first eval is:

- clean,
- high precision,
- but recall-limited.

The new pending question is:

- whether later training keeps the run in this clean-but-conservative regime,
- or whether a better recall / precision balance can still emerge.

---

## 7) Malformed Final Logging Row

The last line in `logging.jsonl` is semantically malformed for a flat metric stream.

It is valid JSON, but it is **not** a metric event row.
Instead it is a final trainer-state payload with:

- `last_model_checkpoint`
- `best_model_checkpoint`
- `best_metric`
- `global_step`
- `log_history`
- `memory`

Why it is problematic:

- it changes the schema of the stream after the eval row,
- it nests a second copy of recent events inside `log_history`,
- and a downstream parser expecting “one flat event dict per line” will treat it as a malformed final record.

Origin:

- the append happens in the external ms-swift stack,
- specifically `swift/llm/train/sft.py::_save_trainer_state`,
- which calls `append_to_jsonl(jsonl_path, self.train_msg, strict=False)`.

Practical interpretation rule for this artifact:

- treat the train row at step `300` and the eval row at step `300` as the final real metric rows,
- and ignore the trailing trainer-state row for scalar analysis.

---

## 8) Practical Handles

Useful checkpoints / windows for future comparison:

- `step 130`
  - first clean regime
- `steps 150-180`
  - concentrated same-desc burst regime
- `steps 190-240`
  - recovery regime
- `step 270`
  - strongest train-side row so far
- `step 280`
  - reminder that the oscillation is not gone yet

Most useful next evidence:

- whether post-`300` training preserves the clean step-`300` train regime without further recall erosion at eval,
- and, once available in a later run, the richer triage telemetry:
  `train/triage/*`,
  `rollout/anchor/*`,
  `rollout/explorer/*`,
  and dead-anchor denominators.
