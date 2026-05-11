thread_id: 019ddf3b-8db4-7433-9243-314cb0a9ca31
updated_at: 2026-04-30T16:41:43+00:00
rollout_path: /data/CoordExp/.codex/archived_sessions/rollout-2026-04-30T16-31-55-019ddf3b-8db4-7433-9243-314cb0a9ca31.jsonl
cwd: /data/CoordExp
git_branch: main

# Stage-1 ET-RMP-CE support2 evaluation across training steps showed a precision/stability gain with only modest recall/FN improvement.

Rollout context: In /data/CoordExp, the user asked to analyze rollout evolution across training steps for the artifact family under `output_remote/stage1_2b/set_continuation/coco1024_sota1332_setcont_et_rmp_ce_support2_effbsz128_v1/.../eval_detection`, focusing on whether predictions become more valid over time, whether FN cases decrease, how decoding stability changes, and what support reweight and RMP do to class diversity/object recall/capacity. The agent inspected local workflow guidance, then analyzed the merged per-step eval artifacts, TensorBoard scalars, and sibling comparison runs.

## Task 1: Analyze step-by-step rollout evolution across training steps
Outcome: success

Preference signals:
- The user asked to “Compare rollout outputs across different steps” and “Identify how predictions evolve during training” -> future similar tasks should compare per-step artifacts directly, not only final metrics.
- The user asked to analyze the effects of “support reweight” and “RMP” -> future similar tasks should separate the objective effect from the runtime/decoding effect.
- The user asked specifically: “Does the model emit more valid objects over time?”, “Do FN cases decrease?”, and “Does decoding become more stable or less stable?” -> future analyses should report valid-output rate, FN/TP/FP trends, and stability/entropy/stop behavior, not just AP.

Key steps:
- Located the actual on-disk run under `output_remote/stage1_2b/set_continuation/coco1024_sota1332_setcont_et_rmp_ce_support2_effbsz128_v1/setcont-coco1024-sota1332-et-rmp-ce-support2-eff_bs_128-v1/v0-20260429-162104/eval_detection`.
- Verified the eval surface exists for steps `100, 200, 300, 400, 500, 600, 700, 800, 900, 916` and that each step has merged `gt_vs_pred.jsonl`, `gt_vs_pred_scored.jsonl`, `metrics.json`, `per_image.json`, `matches.jsonl`, `matches@0.30.jsonl`, `pred_token_trace.jsonl`, and confidence post-op outputs.
- Read `metrics.json`, `infer_summary.json`, `confidence_postop_summary.json`, `per_image.json`, `matches@0.30.jsonl`, and `pred_token_trace.jsonl` for representative steps, then compared hashes across steps to confirm the outputs really differ across steps.
- Pulled TensorBoard scalars from the support2 run and compared them to sibling runs (`rmp_ce_no_support` and an older candidate-balanced/bidirgate run).

Failures and how to do differently:
- One provenance check showed that `infer_summary.json` reused the same base `model_checkpoint` string across steps; that field should not be treated as proof of which training step was evaluated when step artifacts themselves differ.
- The user’s provided run path had an underscore in `support2_eff_bs_128-v1`, but the actual on-disk run-name segment used a hyphen: `support2-eff_bs_128-v1`. Future retrieval should trust the filesystem path, not the paraphrased artifact label.
- The analysis could not rely on `infer_summary.json` alone; step-level hashes and per-step metric files were needed to establish actual evolution.

Reusable knowledge:
- This run is a `limit=200` / `val200` style CoordJSON detection evaluation in `coord` mode with `temperature=0.0`, `repetition_penalty=1.1`, `prompt_variant=coco_80`, sorted object ordering, 8-way distributed merge, and confidence post-op applied to all predictions.
- The eval artifact family is clean: `invalid_json=0`, `invalid_geometry=0`, `empty_pred=0` for every step; confidence post-op kept all predictions (`kept_fraction=1.0`).
- The best way to assess “evolution” here is to combine per-step `metrics.json`, `per_image.json`, `matches@0.30.jsonl`, and `pred_token_trace.jsonl`, then compare against sibling runs.

References:
- [1] On-disk run root used for analysis: `output_remote/stage1_2b/set_continuation/coco1024_sota1332_setcont_et_rmp_ce_support2_effbsz128_v1/setcont-coco1024-sota1332-et-rmp-ce-support2-eff_bs_128-v1/v0-20260429-162104/eval_detection`
- [2] Per-step artifact inventory: every step from `step_0000100` through `step_0000916` contains `gt_vs_pred.jsonl`, `gt_vs_pred_scored.jsonl`, `metrics.json`, `per_image.json`, `matches.jsonl`, `matches@0.30.jsonl`, `pred_token_trace.jsonl`, `confidence_postop_summary.json`, `coco_gt.json`, and `coco_preds.json`.
- [3] `metrics.json` trend: `bbox_AP` rose from `0.3999` at step 100 to `0.423` at step 700 and ended at `0.4181` at step 916; `f1ish@0.30_fn_loc` fell from `618` to `603`.
- [4] TensorBoard scalars: `train/rmp/valid_child_mass_mean` improved from about `0.2785` to `0.2998`, while `train/loss/rmp_branch_support` fell from `0.9856` to `0.6555`.

## Task 2: Interpret the causal effects of support reweight and RMP
Outcome: success

Preference signals:
- The user asked directly how `support reweight` changes class diversity or object recall, so future similar tasks should compare class coverage/entropy and recall separately rather than assuming they move together.
- The user asked how RMP affects “model capacity and decoding behavior” -> future similar tasks should distinguish structured-output validity/capacity from recall quantity.

Key steps:
- Compared support2 against the sibling no-support ET-RMP run and against an older candidate-balanced/bidirgate run.
- Computed class coverage/entropy and per-image object-count behavior from `gt_vs_pred_scored.jsonl`.
- Examined example images to see where the model improved and where it regressed.

Failures and how to do differently:
- Support reweight did not behave like a broad recall booster in this artifact family; it behaved more like a precision/stability pressure. Future similar analyses should not assume “more support mass” means “more objects predicted.”
- The candidate-balanced/bidirgate comparison is not a perfectly controlled ablation; it is still useful as a capacity/reference check, but the support2 vs no-support comparison is the better signal for the support-reweight effect.

Reusable knowledge:
- Support2’s predicted class coverage stayed essentially flat at `73-74` classes out of `75` GT classes present, with class entropy around `4.46 -> 4.40`; it did not materially increase class diversity.
- Support2 reduced false positives much more than it increased true positives: at `@0.30`, `FP` fell `246 -> 92`, `precision` rose `0.771 -> 0.901`, while `recall` only rose `0.572 -> 0.582`.
- Compared with the sibling no-support ET-RMP run, support2 is more conservative at the end (`support2@916: TP 841 / FP 92 / FN 603 / AP 0.418` vs no-support `step300: TP 848 / FP 111 / FN 596 / AP 0.420`), so support reweight appears to trade some recall for cleaner predictions.
- RMP is the decisive ingredient for keeping the model in a usable structured-output regime: the older candidate-balanced/bidirgate run had parse validity around `0.50`, many empty predictions, and much worse AP/recall, whereas the ET-RMP runs had `parse_valid_rate=1.0`, zero empty predictions, and much higher AP/recall.
- The remaining bottleneck is not schema validity; it is crowded/small-object recall and branch sharpness, especially the coordinate branch, where `train/rmp/valid_child_mass_coord` stayed very low (`~0.0185 -> ~0.0234`) relative to the desc/text branch (`~0.388 -> ~0.405`).

References:
- [1] Support2 final metrics (`step_0000916`): `bbox_AP=0.4181`, `bbox_AP50=0.5525`, `f1ish@0.30_tp_loc=841`, `f1ish@0.30_fp_loc=92`, `f1ish@0.30_fn_loc=603`, `f1ish@0.30_precision_loc_micro=0.9014`, `f1ish@0.30_recall_loc_micro=0.5824`.
- [2] Support2 TensorBoard endpoints: `train/rmp/valid_child_mass_mean 0.2785 -> 0.2998`, `train/rmp/teacher_branch_top1_acc 0.1357 -> 0.1579`, `train/rmp/valid_child_mass_coord 0.0185 -> 0.0234`, `train/rmp/valid_child_mass_desc_text 0.3875 -> 0.4053`.
- [3] Sibling no-support ET-RMP run at `output_remote/stage1_2b/set_continuation/coco1024_sota1332_setcont_et_rmp_ce_v1/.../step_0000300`: `bbox_AP=0.4205`, `f1ish@0.30_tp_loc=848`, `f1ish@0.30_fp_loc=111`, `f1ish@0.30_fn_loc=596`.
- [4] Older candidate-balanced/bidirgate run: `eval/det_parse_valid_rate` around `0.515 -> 0.505`, `eval/det_empty_pred` around `97 -> 99`, `eval/det_bbox_AP` around `0.193 -> 0.213`, showing a much weaker capacity regime.

## Task 3: Summarize step-level decoding behavior and instability
Outcome: success

Preference signals:
- The user asked whether decoding becomes “more stable or less stable” -> future similar tasks should report both schema validity and generation tail behavior (stop/end-of-text cleanup, over/under-generation, and per-image variability).

Key steps:
- Inspected `pred_token_trace.jsonl` and `gt_vs_pred.jsonl` for multiple steps and multiple example images.
- Compared adjacent-step prediction retention by class+IoU and checked how much of the prediction set changes across training.

Reusable knowledge:
- The model is schema-stable throughout the run: all steps are valid JSON, with no geometry errors and no empty predictions.
- The main decoding change is not parse validity; it is a shift from early over-generation to a tighter, more stable prediction count basin.
- Adjacent-step prediction retention improves over time: class+IoU retained-pred Jaccard rises from `0.634` for `100->200` to `0.909` for `900->916`, indicating increasingly stable decoding at the object level.
- Token traces show the model usually emits `<|im_end|>`, but many rows continue with `<|endoftext|>` tail tokens afterward; this tail cleanup is still present late in training, though the tail length is shorter than at step 100.

Failures and how to do differently:
- Step 100 had a very long post-`<|im_end|>` tail on at least some examples, so if the future question is about stop behavior rather than object quality, the token-trace tail should be inspected directly.
- The model’s “stability” is image-dependent: some crowded examples improve a lot, while others regress or stay flat; avoid collapsing the whole run into one monotonic narrative.

References:
- [1] Adjacent-step retention/Jaccard from the analysis: `100->200 retained@cls+IoU.5=806, jacc=0.634`; `900->916 retained@cls+IoU.5=915, jacc=0.909`.
- [2] Step 100 token-tail snapshot: only `56/200` rows ended exactly with `<|im_end|>` and the average tail after `<|im_end|>` was about `158.6` tokens; by step 916, `58/200` rows ended exactly at `<|im_end|>` and the average tail after `<|im_end|>` dropped to about `100.1` tokens.
- [3] Example stable image: `images/val2017/000000000139.jpg` stayed around `pred=11-13`, `FN=9-11`, with mostly the same object set across steps.
- [4] Example improving image: `images/val2017/000000019109.jpg` improved from `pred=59, TP=12, FP=47, FN=15` at step 100 to `pred=22, TP=18, FP=4, FN=9` at the end.
- [5] Example regressing image: `images/val2017/000000017959.jpg` shifted from `pred=10, TP=8, FN=16` at step 100 to `pred=16, TP=3, FP=13, FN=21` at the end, showing that not all crowded/tiny-class cases improve.

## Task 4: Identify best checkpoint/useful stopping point
Outcome: success

Preference signals:
- The user asked for evolution across steps, which implicitly makes step selection part of the analysis; future similar tasks should state which step is best for which criterion instead of only quoting the final step.

Key steps:
- Compared AP/FN/F1 trends across all available steps and identified the most favorable step for each metric family.

Reusable knowledge:
- Best AP is around step `700` (`bbox_AP=0.423`, `bbox_AP50=0.562`, `APl=0.563`).
- Best FN/F1 behavior is around step `900` (`FN@0.30=594`, `TP@0.30=850`, `F1@0.30=0.715`).
- Final step `916` is slightly worse than step `900` for FN/F1 and slightly worse than step `700` for AP, so the final checkpoint is not the single best choice on either axis.

References:
- [1] Step 700 metrics: `bbox_AP=0.423`, `bbox_AP50=0.562`, `f1ish@0.30_f1_loc_micro=0.707`, `f1ish@0.30_fn_loc=601`.
- [2] Step 900 metrics: `bbox_AP=0.422`, `f1ish@0.30_tp_loc=850`, `f1ish@0.30_fn_loc=594`, `f1ish@0.30_f1_loc_micro=0.715`.
- [3] Final step 916 metrics: `bbox_AP=0.418`, `f1ish@0.30_tp_loc=841`, `f1ish@0.30_fn_loc=603`, `f1ish@0.30_f1_loc_micro=0.708`.
