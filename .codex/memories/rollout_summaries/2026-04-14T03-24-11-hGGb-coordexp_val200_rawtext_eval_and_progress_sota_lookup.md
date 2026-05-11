thread_id: 019d8a04-9f0c-7311-b4d8-6b48074c13d7
updated_at: 2026-04-18T07:11:17+00:00
rollout_path: /data/CoordExp/.codex/sessions/2026/04/14/rollout-2026-04-14T03-24-11-019d8a04-9f0c-7311-b4d8-6b48074c13d7.jsonl
cwd: /data/CoordExp
git_branch: codex/raw-text-continuity-probe

# Raw-text val200 eval completed; later progress history lookup clarified the prior ~0.38 SOTA was a 200-sample benchmark, not full val.

Rollout context: The main work was in `/data/CoordExp` and centered on evaluating a trained Stage-1 2B `raw_text_xyxy` / `norm1000` checkpoint, then answering follow-up questions about the historical ~0.38 mAP SOTA in `progress/` and whether it was from 200-sample or full-val evaluation. A major side effect of the eval work was debugging unstable multi-GPU inference/merge behavior and recovering a clean artifact by fanout-sharding.

## Task 1: Run raw-text norm1000 eval for val200 and recover a clean merged artifact

Outcome: success

Preference signals:
- when the user said: "跑完了，请开始 eval。请注意目前是的是raw text, norm1000的坐标体" -> the user wanted the evaluation to respect the exact raw-text / norm1000 coordinate contract, not a coord-token or alternative-box-format path.
- when the user later repeated: "跑完了，请开始 eval。请注意目前是的是raw text, norm1000的坐标体系" -> the user was reinforcing that the benchmark should be interpreted in raw-text norm1000 terms throughout the eval chain.
- when the user asked: "只跑r2和r7，并用batch size =8" -> the user preferred a minimal restart of only the missing shards rather than re-running already finished work.
- when the user asked to continue monitoring and mentioned "cuda 0还有1134MB 的占用" -> the user cared about leftover GPU occupancy and wanted it investigated, not ignored.

Reusable knowledge:
- For this checkpoint family, the native distributed `--gpus 8` inference path was unreliable; the successful workaround was to fan out the dataset into 8 independent single-card `run_infer.py` shard runs, then merge by `_fanout_source_index` into one run directory before post-op and bundled eval.
- The initial split attempt failed because the shard JSONL files were written under `temp/`, but their image paths were still being resolved relative to the temp shard directory; the fix was to rewrite the `images` paths to the original preset root (`public_data/coco/rescale_32_1024_bbox_max60_lvis_proxy`).
- `scripts/postop_confidence.py` on this raw-text path produced a real scored artifact with `confidence_method = bbox_logprob_confidence_exp`; this was not the constant-score compatibility path used for `center_log_size`.
- The final merged run directory was `output/infer/coco1024_val200_lvis_proxy_rawtext_xyxy_v1_ckpt552_fanout8`, and the merged artifacts were complete at 200/200 after manual merge.
- The resulting proxy-eval headline on `coco_real` was `bbox_AP = 0.3440`, `bbox_AP50 = 0.4500`, `bbox_AP75 = 0.3712`, with `f1ish@0.50_full_micro = 0.5900`.

Failures and how to do differently:
- The first distributed `--gpus 8` attempt looked like a success in logs because worker-side `Pipeline complete` messages are not global completion signals; future runs should not treat that message as rank-0 merge completion.
- The raw distributed path left the top-level merged artifact missing when some ranks had not reached manifest writeout; for future similar runs, verify all shard summaries/manifests before assuming the merge is done.
- The temporary-fanout split initially failed because shard file paths were not normalized; future sharded runs should either preserve canonical image roots in shard JSONL or explicitly rewrite them before inference.
- GPU 0 showed 1134 MiB reserved with no live process in `nvidia-smi`/`ps`, which looked like stale CUDA context rather than an active worker; future cleanup should distinguish stale driver residency from an actual running process.

References:
- [1] `configs/stage1/profiles/2b/center_log_size_pure_ce_coco80_desc_first_1024_lvis_proxy.yaml` established the Stage-1 `center_log_size` experiment, but the successful eval here used the separate raw-text `xyxy` / `norm1000` path.
- [2] `temp/rawtext_xyxy_v1_ckpt552_val200_fanout8/infer_rank{0..7}.yaml` were the 8 shard configs used for the working fanout run; `r2` and `r7` were the only ones manually re-run with `batch_size: 8` after the initial incomplete pass.
- [3] `output/infer/coco1024_val200_lvis_proxy_rawtext_xyxy_v1_ckpt552_fanout8/gt_vs_pred.jsonl` and `pred_token_trace.jsonl` contain the merged 200-sample artifacts.
- [4] `output/infer/coco1024_val200_lvis_proxy_rawtext_xyxy_v1_ckpt552_fanout8/proxy_eval_bundle_summary.json` contains the bundled COCO proxy metrics for `coco_real`, `coco_real_strict`, and `coco_real_strict_plausible`.
- [5] The raw-text post-op summary reported `total_samples: 200`, `total_pred_objects: 1115`, `pred_score_source: confidence_postop`, and `confidence_method: bbox_logprob_confidence_exp`.

## Task 2: Locate the historical ~0.38 SOTA checkpoint in `progress/` and determine whether it was 200-sample or full-val

Outcome: success

Preference signals:
- when the user asked: "请参考我之前的`0.38`左右mAP的 sota 的 checkpoint，是哪个实验设计下的。查看`progress`下" -> the user wanted the historical experiment design, not just the number.
- when the user asked: "sota的有没有说是200还是full val的结果？" -> the user specifically cared about the evaluation scope and did not want `val200` and `full val` conflated.

Reusable knowledge:
- The most likely “~0.38” historical SOTA the user was referring to is the Stage-1 **mixed objective** benchmark, not pure CE: `hard CE + softCE + W1 + gate`.
- That benchmark is documented in `progress/benchmarks/stage1_coco_2b_ce_softce_res_768_vs_1024_2026-02-27.md`, which explicitly says it is a **200-sample** COCO bench.
- The key checkpoint is `output/stage1_2b/coco_bbox_max60-hard_ce_soft_ce_w1_gate_merged-1332` (adapter checkpoint under `output/stage1_2b/coco_bbox_max60-hard_ce_soft_ce_w1_gate/epoch_4-from-base-2B/v0-20260227-050057/checkpoint-1332`).
- The reported numbers in that note are `mAP = 0.3896` at 768 and `mAP = 0.3879` at 1024; both are `limit=200` results, not full-val.
- The later `~0.3731727356` reference belongs to a different family and is the **full-val** reference discussed in `progress/diagnostics/cxcy_logw_logh_retrained_performance_analysis_2026-04-15.md`, not the `~0.389` benchmark.
- The user’s “0.38” memory should therefore be treated as a **200-sample mixed Stage-1 benchmark**, while full-val references are lower and should not be mixed into the same comparison.

Failures and how to do differently:
- It was easy to conflate multiple “strong” numbers around 0.37–0.39; future lookups should always ask whether the source note says `limit=200`, `val200`, prefix bundle, or `full val` before comparing.
- The `0.389x` benchmark is not apples-to-apples with the raw-text `norm1000` run because it used a different training recipe and output contract; future comparisons should keep the objective/geometry/output split explicit.

References:
- [1] `progress/benchmarks/stage1_coco_2b_ce_softce_res_768_vs_1024_2026-02-27.md` — title says `COCO Bench (200 samples): 2B Mixed CE+softCE Checkpoint`, with `Sample limit: 200` and `mAP = 0.3896` / `0.3879`.
- [2] `progress/benchmarks/stage1_coco80_4b_res_768_vs_1024_2026-02-26.md` — another 200-sample benchmark around `0.3856` / `0.3891`, also not full-val.
- [3] `progress/diagnostics/cxcy_logw_logh_retrained_performance_analysis_2026-04-15.md` — distinguishes the stronger `same-prefix` reference (`bbox_AP = 0.4026465592`) from the `baseline prior full-val reference` (`bbox_AP = 0.3731727356`).
- [4] `progress/diagnostics/duplication_collapse_final_analysis_2026-04-13.md` — notes that `stage1_2b_center_param_ckpt1564` was an important center-family comparison anchor and that the CE-like continuation branches were not clean pure-CE baselines.

