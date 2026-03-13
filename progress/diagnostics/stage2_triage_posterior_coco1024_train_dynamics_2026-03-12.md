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
Status note: this note records the run through the first eval at `global_step = 300`.
Update `2026-03-13`: the note now also includes a checkpoint comparison against the matched
Stage-1-init run, a qualitative comparison of the shared step-`300` eval monitor dumps, and a
correction note for the custom visualization coordinate mapping.

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

Matched comparison run:

- Run dir:
  `output/stage2_ab/prod/ab_mixed_coco1024_bmajority_channel_b_triage_posterior/stage1_ckpt-eff_size_96-b_ratio_0.75-triage_posterior-epoch_1/v0-20260312-162706/`
- Train log:
  `.../logging.jsonl`
- Init model:
  `output/stage1/coco_bbox_max60-coco80-desc_first/epoch_4-softce_w1-coco80-ckpt_1832-merged`

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
- the run dir now contains an eval monitor dump at
  `monitor_dumps/step_000300.json`,
- but the expected train monitor dumps at `every_steps = 40` were still not emitted for this
  older artifact,
- the log still lacks the later explicit `train/triage/*` and anchor-vs-explorer split metrics,
- and this historical run originally had a trailing trainer-state append after the eval row,
  although the local copy of `logging.jsonl` has since been sanitized in place to keep the stream flat.

So the analysis below is necessarily **train-window behavioral diagnosis**, not a fully instrumented v3 telemetry read.

Historical malformed-row provenance:

- the original bad final append was not produced by CoordExp code,
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

## 7) Historical Malformed Final Logging Row

The original unsanitized artifact had a semantically malformed final line in `logging.jsonl`.

It was valid JSON, but it was **not** a metric event row.
Instead it was a final trainer-state payload with:

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

Current status:

- the local copy of
  `output/stage2_ab/prod/ab_mixed_coco1024_bmajority_channel_b_triage_posterior/eff_size_96-b_ratio_0.75-triage_posterior-epoch_1/v0-20260311-165111/logging.jsonl`
  has now been sanitized in place,
- so the current file ends cleanly on the step-`300` eval row,
- but the provenance above is still relevant for understanding the original older artifact.

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

---

## 9) Addendum (2026-03-13): Matched Checkpoint Comparison

This section compares the main run above against the matched Stage-1-init run:

- Stage-2 UL-v2 init:
  `output/stage2_ab/prod/ab_mixed_coco1024_bmajority_channel_b_triage_posterior/eff_size_96-b_ratio_0.75-triage_posterior-epoch_1/v0-20260311-165111/`
- Stage-1 init:
  `output/stage2_ab/prod/ab_mixed_coco1024_bmajority_channel_b_triage_posterior/stage1_ckpt-eff_size_96-b_ratio_0.75-triage_posterior-epoch_1/v0-20260312-162706/`

The two runs use the same Stage-2 config and differ primarily by initialization checkpoint.

### 9.1 High-level comparison

The Stage-1-init run is much more conservative on train rollouts *at the start*:

- first logged Channel-B row:
  - Stage-2 UL-v2 init: `f1 = 0.288`, `precision = 0.181`, `recall = 0.701`,
    `pred_per_sample = 32.33`, `parse_truncated_rate = 0.229`
  - Stage-1 init: `f1 = 0.759`, `precision = 0.811`, `recall = 0.712`,
    `pred_per_sample = 7.34`, `parse_truncated_rate = 0.0`

So the Stage-1 checkpoint clearly injects a colder and more precision-biased initial policy.

However, by step `300`, the stronger endpoint is still the Stage-2 UL-v2 init run:

| Eval step `300` | Stage-2 UL-v2 init | Stage-1 init |
|---|---:|---:|
| `eval/detection/mAP` | `0.4424` | `0.4244` |
| `eval/detection/precision` | `0.7804` | `0.6638` |
| `eval/detection/recall` | `0.6533` | `0.6160` |
| `eval/detection/f1` | `0.7112` | `0.6390` |
| `eval/detection/pred_objects` | `2850` | `3159` |
| `eval/parsing/parse_truncated_rate` | `0.0000` | `0.00625` |

Interpretation:

- the Stage-1-init run is not simply “the same model, but more cautious,”
- it starts cleaner on-policy,
- but it does **not** land in a better eval regime after `300` steps.

### 9.2 Different training trajectories

The Stage-2 UL-v2 init run learns upward:

- train rollout `f1` rises from `0.288` to a best of `0.676`, ending at `0.625`,
- precision rises sharply,
- rollout length and truncation fall sharply,
- and the final train row is nearly free of duplicate mass.

The Stage-1-init run starts high and then drifts downward:

- it begins at `f1 = 0.759`,
- but ends at `f1 = 0.577`,
- while its late rows still contain heavy same-desc duplicate bursts.

The clearest late-stage contrast is the final train row:

- Stage-2 UL-v2 init step `300`:
  `N_duplicates = 1`, `near_iou90_pairs_same_desc_count = 1`
- Stage-1 init step `300`:
  `N_duplicates = 162`, `near_iou90_pairs_same_desc_count = 2612`

So the Stage-1-init policy is cleaner by default early on, but it is **not** more stable by the
end of the run.

### 9.3 What the newer telemetry reveals about Stage-1-init failure mode

The Stage-1-init run has the newer telemetry surface, so we can localize the late duplicate bursts.

The important pattern is that the bad mass is overwhelmingly **anchor-driven** rather than
explorer-driven.

Examples:

- row `12`:
  - `rollout/anchor/near_iou90_same = 5998.0`
  - `rollout/explorer/near_iou90_same = 0.0`
- row `24`:
  - `rollout/anchor/near_iou90_same = 4832.0`
  - `rollout/explorer/near_iou90_same = 0.0`
- row `30`:
  - `rollout/anchor/near_iou90_same = 1306.0`
  - `rollout/explorer/near_iou90_same = 0.5`

This matters because it means:

- the explorer path is relatively well-behaved,
- the anchor policy is the one getting trapped in local same-class repetition,
- and the late-stage instability in the Stage-1-init run is not just “sampling noise.”

---

## 10) Addendum (2026-03-13): Shared Step-300 Eval Dump Comparison

Both runs now have a step-`300` eval dump:

- Stage-2 UL-v2 init:
  `.../eff_size_96-b_ratio_0.75-triage_posterior-epoch_1/v0-20260311-165111/monitor_dumps/step_000300.json`
- Stage-1 init:
  `.../stage1_ckpt-eff_size_96-b_ratio_0.75-triage_posterior-epoch_1/v0-20260312-162706/monitor_dumps/step_000300.json`

The two dumps share `9` probe images in common:

- `base_idx = 2, 17, 56, 238, 325, 341, 361, 391, 499`

Across those `9` shared images:

- mean sample F1:
  - Stage-2 UL-v2 init: `0.658`
  - Stage-1 init: `0.520`
- mean sample recall:
  - Stage-2 UL-v2 init: `0.683`
  - Stage-1 init: `0.500`
- mean sample precision:
  - Stage-2 UL-v2 init: `0.683`
  - Stage-1 init: `0.701`
- mean matched pairs:
  - Stage-2 UL-v2 init: `8.0`
  - Stage-1 init: `5.44`
- mean FN count:
  - Stage-2 UL-v2 init: `4.0`
  - Stage-1 init: `6.56`

The Stage-2 UL-v2 init model wins `6/9` shared samples on F1, ties `2`, and loses `1`.

### 10.1 Preference difference

The Stage-1-init model has a strong **salience bias**:

- it prefers a small set of highly salient objects,
- undercounts dense or repeated small objects,
- and only rarely commits to broad scene completion.

This shows up in the per-sample prediction counts:

- Stage-1 init shared-sample median prediction count: `5`
- Stage-2 UL-v2 init shared-sample median prediction count: `11`

So the Stage-1-init model is indeed more conservative *in its default behavior*.

But that is not the whole story.
Its shared-sample mean prediction count is actually **higher** (`21.1` vs `15.8`) because one hard
bookshelf scene explodes into a local attractor failure with `128` predictions.

That is why the right description is:

- conservative by default,
- but still vulnerable to rare catastrophic repeated-class collapse.

### 10.2 Capability difference

The Stage-2 UL-v2 init model is better at:

- enumerating repeated instances when the scene genuinely contains many of them,
- recovering medium-salience support objects in clutter,
- and preserving recall without collapsing into extreme undercounting.

The Stage-1-init model is better at:

- keeping a sparse set on simple or canonical scenes,
- and occasionally being slightly tighter in clean crowd scenes.

But it is worse at:

- dense bookshelf / repeated small-object regions,
- scenes where the correct output requires broad scene completion,
- and robustness against local same-class repetition.

### 10.3 Representative shared samples

`base_idx = 238` (bookshelf room):

- Stage-2 UL-v2 init:
  `pred = 25`, `f1 = 0.439`, roughly
  `laptop -> bottle -> chair -> couch -> 16x book -> ...`
- Stage-1 init:
  `pred = 128`, `f1 = 0.125`, truncated, ending in
  `... -> 116x book`

Interpretation:

- both models understand “room with bookshelf,”
- but the Stage-1-init model loses instance atomicity far more severely and turns the shelf region
  into a `book` attractor.

`base_idx = 499` (baseball-bat scene):

- Stage-2 UL-v2 init:
  `pred = 18`, `f1 = 0.667`
- Stage-1 init:
  `pred = 4`, `f1 = 0.316`

Interpretation:

- the Stage-2 UL-v2 init model recognizes the repeated bat-like structure of the scene,
- while the Stage-1-init model mostly refuses to enumerate that structure and falls back to a few
  salient non-bat anchors.

`base_idx = 391` (kitchen line):

- Stage-2 UL-v2 init:
  `2x person -> bowl -> oven -> knife -> spoon`
- Stage-1 init:
  `2x person -> bowl`

Interpretation:

- both are conservative here,
- but the Stage-2 UL-v2 init still performs better scene completion on medium-salience kitchen objects.

`base_idx = 341` (snowboard crowd):

- Stage-2 UL-v2 init:
  `person -> 2x snowboard -> 14x person`, `f1 = 0.625`
- Stage-1 init:
  `snowboard -> person -> snowboard -> 15x person`, `f1 = 0.667`

Interpretation:

- the Stage-1-init model can still look competitive or slightly better on clean repeated crowd
  scenes with strong object cues,
- so the comparison is not “one model dominates every scene.”

### 10.4 Joint read

The best summary of the two inits is:

- **Stage-1 init**:
  conservative, anchor-heavy, salience-biased, recall-limited on clutter, and vulnerable to
  rare local repeated-class attractors.
- **Stage-2 UL-v2 init**:
  broader scene completion, better repeated-instance recall, still vulnerable to local repetition,
  but much stronger as an endpoint by step `300`.

---

## 11) Visualization Correction Note

The custom side-by-side overlays produced during the `2026-03-13` comparison originally used the
wrong axis ordering and should not be trusted.

The mistaken assumption was:

- interpreting `points_norm1000` as `[y1, x1, y2, x2]`

The correct contract is:

- `bbox_2d = [x1, y1, x2, y2]`
- `norm1000 -> pixel` conversion uses `(W - 1)` for `x` slots and `(H - 1)` for `y` slots

This is consistent with:

- `src/config/prompts.py`
- `src/common/geometry/coord_utils.py::ints_to_pixels_norm1000`

The corrected overlays are under:

- `temp/vis_compare_stage2_inits_step300_corrected/`

Representative corrected figures:

- `temp/vis_compare_stage2_inits_step300_corrected/compare_base000238.png`
- `temp/vis_compare_stage2_inits_step300_corrected/compare_base000499.png`
- `temp/vis_compare_stage2_inits_step300_corrected/compare_base000002.png`

The earlier incorrect overlays under:

- `temp/vis_compare_stage2_inits_step300/`

should be ignored.
