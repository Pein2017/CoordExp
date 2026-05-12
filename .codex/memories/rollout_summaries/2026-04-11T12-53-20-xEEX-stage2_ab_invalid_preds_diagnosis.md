thread_id: 019d7c9a-9d0c-7bf1-b7fe-f5490ccef563
updated_at: 2026-04-11T13:07:57+00:00
rollout_path: /data/home/xiaoyan/AIteam/data/CoordExp/.codex/sessions/2026/04/11/rollout-2026-04-11T12-53-20-019d7c9a-9d0c-7bf1-b7fe-f5490ccef563.jsonl
cwd: /data/home/xiaoyan/AIteam/data/CoordExp
git_branch: main

# Stage-2 AB experiment diagnosis: sampled Channel-B invalid predictions were mostly structural decode failures, while greedy eval remained reasonably healthy.

Rollout context: The user pointed to a specific Stage-2 two-channel run under `/data/home/xiaoyan/AIteam/data/CoordExp` and asked to explore/diagnose it, then followed up asking what the main symptom of the `invalid preds` was. The analysis used the canonical docs, the resolved config, `logging.jsonl`, `monitor_dumps/prepare_failures`, and step-300 `eval_detection` artifacts.

## Task 1: Diagnose the Stage-2 two-channel pseudo-positive run

Outcome: success

Preference signals:

- The user asked to diagnose a specific experiment from the run artifacts and config, indicating that future diagnostics should start from the pointed-to run directory and resolve the config/artifacts first rather than guessing from memory.
- The user named the running artifacts explicitly and asked to “explore and diagnose this experiment,” which suggests future agents should ground conclusions in the exact artifact tree and not rely only on high-level training summaries.

Key steps:

- Loaded the repo routing docs and stage-2 runbook/specs before inspecting artifacts, so the diagnosis stayed aligned with the active Stage-2 two-channel contract.
- Confirmed the run config was the production-style Stage-2 AB pseudo-positive profile:
  - `custom.trainer_variant: stage2_two_channel`
  - `stage2_ab.schedule.b_ratio: 0.85`
  - `stage2_ab.channel_b.pseudo_positive.enabled: true`
  - `stage2_ab.channel_b.triage_posterior.num_rollouts: 4`
  - `stage2_ab.channel_b.insertion_order: sorted`
  - `stage2_ab.channel_b.invalid_rollout_policy: dump_and_continue`
  - `training.packing: true`
  - `training.effective_batch_size: 96`
  - `rollout_matching.rollout_backend: vllm`
  - `rollout_matching.eval_rollout_backend: vllm`
  - `rollout_matching.max_new_tokens: 3084`
- Parsed `logging.jsonl` across 48 logged steps (step 1 through step 460) and found the run was still mid-epoch (`460/1221` in the latest log entry).
- Inspected step-300 eval artifacts in `eval_detection/step_0000300`, including `metrics.json`, `infer_summary.json`, `raw_rollouts.jsonl`, `gt_vs_pred_scored.jsonl`, `per_class.csv`, and `semantic_desc_report.json`.
- Inspected `monitor_dumps/prepare_failures` and representative JSON examples to separate training-side rollout formatting failures from eval-side decoding quality.

Reusable knowledge:

- In this repo, the Stage-2 two-channel diagnosis should treat `logging.jsonl`, `monitor_dumps/prepare_failures`, and `eval_detection/step_<N>` as the primary evidence chain.
- The resolved config and the runbook are needed to interpret whether a metric value is expected behavior or a problem in this specific Stage-2 profile.
- `monitor_dumps/prepare_failures` can be very large; aggregate it before spot-reading so the dominant failure signature is visible.
- Training-side Channel-B and greedy eval can diverge sharply: sampled explorer views can be very noisy even when eval-step greedy decoding is healthy.

Failures and how to do differently:

- The first pass showed that some logged “rates” are not straightforward probabilities and can exceed 1.0; those should be treated as diagnostic ratios whose denominator semantics should be checked before over-interpreting them.
- The training-side rollout path was the expensive and unstable part of the run; future similar investigations should compare train sampled rollouts against eval greedy rollouts rather than assuming one quality signal covers both.
- Because `prepare_failures` had 14,787 files, the right approach was to aggregate counts and failure classes first, then inspect a few concrete examples.

References:

- [1] Run config: `configs/stage2_two_channel/prod/4b-pure_ce_ckpt-ab_mixed_coco1024_lvis_proxy_channel_b_pseudo_positive_dup_targeting.yaml`
- [2] Run directory: `output/stage2_ab/prod/4b_lvis_proxy_pseudo_positive_dup_targeting_ce_ciou_only/epoch_1-k4-eff_size_96-b_ratio_0.85-lvis_proxy-4b-ce_ciou_only-dup_targeting-sorted/v0-20260408-152158/`
- [3] Training log summary from `logging.jsonl`: loss fell roughly `0.315 -> 0.171` from step 1 to step 460, but rollout quality stayed weak (`rollout/precision` about `0.40 -> 0.50`, `rollout/recall` about `0.062 -> 0.098`, `rollout/parse_truncated_rate` about `0.62-0.65`, `rollout/gating_rejection_rate` about `0.73-0.82`)
- [4] Step-300 eval: `eval_detection/step_0000300/metrics.json` reported `bbox_AP 0.40246237409671215`, `bbox_AP50 0.5490706801763594`, `bbox_AP75 0.43722851962660014`
- [5] Step-300 eval monitor dump: `eval_detection/step_0000300/metrics.json` reported detection `precision 0.6353818462329648`, `recall 0.6460130718954248`, `f1 0.6406533575317604`, with clean parse counters (`invalid_json=0`, `invalid_geometry=0`, `invalid_coord=0`, `unknown_desc=0`)
- [6] `monitor_dumps/prepare_failures/` contained 14,787 files; aggregated invalid rollout views totaled 19,052, and all of them had `pred_objects=0` and `valid_pred_objects=0`
- [7] Representative prepare-failure example: `monitor_dumps/prepare_failures/step_000002_rank_00_sample_000_views_explorer_1-explorer_2.json` showed a runaway malformed output with a long `]]]]...` tail after an incomplete bbox, plus another view that stopped mid-JSON
- [8] Step-300 eval images with large count gaps included `image_372.jpg` (`32` GT vs `128` pred, truncated rollout), `image_147.jpg` (`26` GT vs `90` pred, heavy knife/fork overprediction), and `image_329.jpg` (`51` GT vs `23` pred, undercounted books)

## Task 2: Explain the main symptom of `invalid preds`

Outcome: success

Preference signals:

- The user’s follow-up wording, “What are the main symptom for thos `invalid preds`?”, indicates they wanted a symptom-level explanation rather than a deep architectural postmortem.
- They repeatedly came back to the same topic after an interruption, which suggests future responses should answer the symptom question directly and compactly before adding detail.

Key steps:

- Reframed the training-side invalid predictions into symptom classes instead of treating them as one generic failure.
- Distinguished between hard-invalid empty rollouts and partially valid rollouts where some objects are dropped by parsing/validation.
- Used both the prepare-failure dumps and the step-300 eval artifacts to identify representative failure signatures.

Reusable knowledge:

- The dominant symptom pattern for these invalid preds is **format collapse**, not subtle box-quality degradation.
- Most invalid training-side Channel-B views never yield any parseable object records at all; many run to `max_new_tokens=3084` and then degenerate into malformed structure.
- The common symptom buckets observed were:
  - runaway bracket / punctuation tails,
  - incomplete JSON or unfinished object lists,
  - wrong bbox arity,
  - missing `desc`,
  - unexpected keys / schema drift,
  - invalid coord-slot contents such as literal integers or stray non-coord tokens.
- There is also a milder class of “partially valid but dropped object” outputs where most of the response is usable but one or more objects are removed by the parser.
- Step-300 eval itself was comparatively clean, so the invalid-pred problem is primarily on the sampled training/explorer path rather than the greedy eval path.

Failures and how to do differently:

- The right answer was not to say “the model is bad at boxes” in the abstract; the concrete issue is that the output often stops being a valid CoordJSON object stream.
- When asked about invalid predictions again, lead with the symptom taxonomy and a few representative failure signatures before discussing metrics or code.

References:

- [1] Aggregated prepare-failure evidence: across 19,052 invalid rollout views, every one had `pred_objects=0` and `valid_pred_objects=0`
- [2] Representative malformed output: `monitor_dumps/prepare_failures/step_000002_rank_00_sample_000_views_explorer_1-explorer_2.json`
- [3] Step-460 strict-drop reasons in `logging.jsonl`: `unexpected_keys` was the largest family late in training, with `wrong_arity` and `missing_desc` also recurring
- [4] Step-300 eval examples: `image_372.jpg` (`32` GT / `128` pred, truncated), `image_147.jpg` (`knife`/`fork` overprediction), `image_329.jpg` (`51` GT / `23` pred, undercounted books)
- [5] Code handles for the relevant counters and dumps: `src/trainers/stage2_two_channel.py` (`_write_channel_b_prepare_failure_dump`, strict-drop counters, `train/triage/*` counters)
