thread_id: 019ddf39-2843-7430-b949-910dd6db152f
updated_at: 2026-04-30T17:19:17+00:00
rollout_path: /data/CoordExp/.codex/archived_sessions/rollout-2026-04-30T16-29-18-019ddf39-2843-7430-b949-910dd6db152f.jsonl
cwd: /data/CoordExp
git_branch: main

# Compared step-wise eval artifacts for the Stage-1 set-continuation support-reweighted ET-RMP-CE run and diagnosed how outputs evolve

Rollout context: The user asked to inspect `output_remote/stage1_2b/set_continuation/coco1024_sota1332_setcont_et_rmp_ce_support2_effbsz128_v1/setcont-coco1024-sota1332-et-rmp-ce-support2_eff_bs_128-v1/v0-20260429-162104/eval_detection`, compare results across training steps, and analyze how `support reweight` and `RMP` affect model capacity, object emission behavior, and decoding dynamics. The rollout was interrupted previously, so the agent first re-established the artifact surface, then sampled raw predictions, metrics, and token traces.

## Task 1: Inspect latest eval artifacts and compare across training steps

Outcome: success

Preference signals:
- The user explicitly asked to compare outputs “across different training steps” and analyze how the outputs evolve -> future similar requests should be handled as a step-wise artifact comparison, not a single-metric report.
- The user framed the goal as understanding how “support reweight” and “RMP” affect “capacity, object emission behavior, and decoding dynamics” -> future similar analyses should prioritize behavioral diagnosis (emission counts, duplicates, false negatives, termination) over AP alone.

Key steps:
- Verified the requested artifact directory did not literally exist and located the actual run directory under `support2-eff_bs_128-v1` rather than the user-typed `support2_eff_bs_128-v1`.
- Confirmed the eval bundle contained `metrics.json`, `per_image.json`, `gt_vs_pred.jsonl`, `pred_token_trace.jsonl`, `confidence_postop_summary.json`, and related per-step outputs for steps `100` through `916`.
- Read the run provenance from `resolved_config.json`, `effective_runtime.json`, `experiment_manifest.json`, `eval_data_provenance.json`, and `logging.jsonl`.
- Aggregated metrics across steps and bucketed performance by GT count, especially to isolate high-count scenes.
- Sampled concrete examples from `000000009590.jpg`, `000000018380.jpg`, `000000000139.jpg`, and `000000000285.jpg` to show how output shape changed.

Failures and how to do differently:
- The user-supplied path had a naming mismatch (`support2_eff_bs_128-v1` vs actual `support2-eff_bs_128-v1`); future agents should verify the exact run directory before assuming artifact absence.
- `config_source.yaml` / manifest prose still contained stale `bsz32` / `256` language, while the resolved runtime was actually `16/128`; future agents should privilege resolved config and runtime files over authored prose when provenance conflicts.

Reusable knowledge:
- The eval surface is `val200` on `public_data/coco/rescale_32_1024_bbox_max60/val.coord.jsonl`, with `sample_limit=200`, `coord_token_xyxy`, `temperature=0`, `top_p=1`, `repetition_penalty=1.1`, and `confidence_postop` scoring via `bbox_logprob_confidence_exp`.
- Every step in this run had valid parsing and no empty predictions; the main behavioral changes were in emission count, duplicate suppression, and precision/recall balance rather than parse failures.
- The run is the support-reweighted ET-RMP-CE variant with `branch_support_weight=2.0` and `branch_balance_weight=1.0`, resolved on the `16/128` contract (`per_device_train_batch_size=16`, `gradient_accumulation_steps=1`, `effective_batch_size=128`).

References:
- [1] Actual run directory used for analysis: `output_remote/stage1_2b/set_continuation/coco1024_sota1332_setcont_et_rmp_ce_support2_effbsz128_v1/setcont-coco1024-sota1332-et-rmp-ce-support2-eff_bs_128-v1/v0-20260429-162104/eval_detection`
- [2] Step-100 metrics: `bbox_AP=0.399900383`, `f1ish@0.30_f1_full_micro=0.645468998`, `pred_total=1107`, `fp_full=260`, `fn_full=632`
- [3] Step-900 metrics: `bbox_AP=0.422105991`, `f1ish@0.30_f1_full_micro=0.705931847`, `pred_total=960`, `fp_full=94`, `fn_full=605`
- [4] Example emission collapse: `images/val2017/000000009590.jpg` changed from `123` predictions at step 100 (`118` bowls) to `23` predictions at step 900 (`14` bowls)
- [5] Decoding tail issue: traces usually contain `<|im_end|>` but only about `28-29%` end exactly there; many continue with repeated `<|endoftext|>` tokens afterward

## Task 2: Diagnose support reweight and RMP effects

Outcome: success

Preference signals:
- The user explicitly asked to understand the effect of “support reweight” and “RMP” on “capacity, object emission behavior, and decoding dynamics” -> future similar tasks should separate training-side branch-mass metrics from free-rollout behavior.

Key steps:
- Compared metric curves across steps 100–916.
- Broke down performance by GT-count buckets to see whether higher-capacity scenes improved differently from easy scenes.
- Compared the support2 run against the nearby non-support ET-RMP-CE run as a directional, not fully isomorphic, reference.
- Inspected training logs to see whether RMP branch metrics improved during optimization.

Failures and how to do differently:
- The adjacent `et_rmp_ce_v1` run is useful only as a directional comparator, not a pure ablation; future agents should label it as non-isomorphic if exact controls differ.
- The run does not show a strong recall/capacity gain, so future diagnosis should not over-attribute the improvement to “more objects emitted”; the evidence supports “cleaner valid continuations” more than “greater object emission capacity.”

Reusable knowledge:
- Step-wise behavior: after step 100 the model rapidly suppresses duplicate/loop-like emissions and becomes much more selective; exact duplicates disappear after step 200, and near-duplicate pairs collapse from thousands to single digits.
- Object emission count stabilizes around `~960-971` predictions on the 200-image val slice, or roughly `0.66-0.67` predictions per GT, which is below full coverage.
- The model is already strong on low-count scenes: 1-object images are at `0.947` recall and 2–3-object images around `0.822-0.828` recall, so the main weakness is crowded/high-count continuation rather than basic detection.
- High-count scenes remain the bottleneck: for GT count `>=11`, recall stays around `0.47-0.49` while precision rises from `0.64` at step 100 to about `0.85` at the best later steps.
- RMP training-side metrics improve: `rmp/valid_child_mass_mean` rises from about `0.279` at step 1 to `0.333` at step 900, and `rmp/valid_child_top1_acc` rises from about `0.324` to `0.372`, indicating better valid-branch pressure under teacher forcing.
- The observed effect of support reweight + RMP is best summarized as regularizing valid branch choice and reducing bad continuations, not dramatically increasing rollout capacity.

References:
- [1] Support2 best step: step 900 with `bbox_AP=0.422105991`, `AP50=0.555319930`, `F1@0.30=0.705931847`, `P@0.30=0.899249732`, `R@0.30=0.581024931`, `pred_total=960`
- [2] Non-support comparator (`et_rmp_ce_v1`) was only partially queried, but step 300 showed `bbox_AP=0.420454286`, `F1@0.30=0.694964628`, `P@0.30=0.870698644`, `R@0.30=0.578254848`, `pred_total=992`
- [3] High-count bucket at step 900: `gt=853`, `pred=501`, `tp=418`, `fp=73`, `fn=435`, `P=0.851323829`, `R=0.490035170`
- [4] Training log at step 900: `rmp/valid_child_mass_mean=0.33303473`, `rmp/valid_child_top1_acc=0.37238674`, `loss/rmp_branch_support=0.66095452`, `loss/rmp_branch_total=2.11728554`

## Task 3: Resolve provenance and runtime contract

Outcome: success

Preference signals:
- The user’s request was about the latest artifacts, so future agents should treat exact run provenance as part of the analysis rather than assuming the visible folder name is always correct.

Key steps:
- Read the resolved config and runtime files to distinguish authored metadata from actual execution settings.
- Verified the run used the `16/128` runtime contract even though some manifest prose still mentioned `32/256`.

Failures and how to do differently:
- Stale prose in `config_source.yaml` and `experiment_manifest.json` can mislead if used as the source of truth; future agents should always check resolved config and runtime artifacts first.

Reusable knowledge:
- Actual resolved training settings: `per_device_train_batch_size=16`, `gradient_accumulation_steps=1`, `effective_batch_size=128`.
- Actual experiment objective: `entry_trie_rmp_ce` with `branch_support_weight=2` and `branch_balance_weight=1`.
- The benchmark report surface is `val200`, `eval_view=coco_map_with_logprob_confidence_plus_f1ish_annotated`, with `greedy_temp0_top_p1_rep1p10` decoding.

References:
- [1] `resolved_config.json` showed `training.per_device_train_batch_size: 16`, `training.gradient_accumulation_steps: 1`, `training.effective_batch_size: 128`
- [2] `effective_runtime.json` / manifest still contained stale `bsz32` / `effective_batch_size=256` prose, so resolved config should be preferred
- [3] `eval_data_provenance.json` confirmed `dataset_seed=17`, `sample_limit=200`, and the COCO val JSONL source path `public_data/coco/rescale_32_1024_bbox_max60/val.coord.jsonl`
