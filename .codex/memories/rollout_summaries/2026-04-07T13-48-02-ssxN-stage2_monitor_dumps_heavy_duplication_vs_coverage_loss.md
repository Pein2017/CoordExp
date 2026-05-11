thread_id: 019d6833-3f65-74d1-b74b-d340d74deeff
updated_at: 2026-04-08T01:44:39+00:00
rollout_path: /data/CoordExp/.codex/sessions/2026/04/07/rollout-2026-04-07T13-48-02-019d6833-3f65-74d1-b74b-d340d74deeff.jsonl
cwd: /data/CoordExp
git_branch: main

# Stage-2 monitor_dumps deep analysis of failure pattern, duplicate bursts, and later performance

Rollout context: The user asked twice about the same Stage-2 training artifact under `output/stage2_ab/prod/2b_lvis_proxy_pseudo_positive_dup_targeting_ckpt1564_merged/.../monitor_dumps`, first to analyze the current failure pattern and whether heavy duplication exists, then to continue with deeper analysis, and finally to check later performance after new dumps were added. The analysis used the project’s Stage-2 runbook/specs and then inspected the actual `monitor_dumps` files, including early, mid, and later snapshots. Later dumps extended the directory from steps `2, 36, 71, 106, 142` to `177, 212, 248, 283, 300`.

## Task 1: Analyze current failure pattern and heavy duplication in Stage-2 monitor dumps

Outcome: success

Preference signals:
- The user asked for a direct diagnosis of the failure pattern and specifically whether “heavy duplication exist” in the Stage-2 artifact. This suggests that in similar triage tasks, the user wants a concrete artifact-level verdict, not just generic commentary.
- When the user later asked to “conduct deeper analysis,” that indicates they prefer follow-up drilling into the same artifact rather than a one-shot shallow summary when the initial answer is still ambiguous.

Key steps:
- Loaded repo routing docs and Stage-2 runbook/specs first, then inspected `monitor_dumps` contents and schema.
- Enumerated the artifact directory and found five initial `step_*.json` snapshots plus `prepare_failures/` and `ddp_phase_trace/`.
- Read actual snapshot structure: top-level keys were `epoch`, `global_step`, `kind`, `meta`, `metrics`, `samples`, `time`.
- Inspected one snapshot deeply (`step_000142.json`) to map fields such as `stats`, `match`, `duplication`, `triage`, `pred`, `gt`, `pred_objects`, `gt_objects`, `anchor_rollout_text`, `explorer_rollout_text`, etc.
- Recomputed aggregate metrics across all sampled scenes and later corrected an initial aggregation mistake that had looked for a non-existent nested `summary` key inside `duplication`.
- Confirmed the actual monitor schema is emitted by `_build_stage2_train_monitor_record` and prepare-failure dumps by `_write_channel_b_prepare_failure_dump` in `src/trainers/stage2_two_channel.py`.

Failures and how to do differently:
- An early aggregation pass incorrectly assumed duplication and triage summaries were nested under subkeys that did not exist. The fix was to inspect the raw sample object directly and recompute using the actual schema.
- Another early attempt treated `rg`-style file discovery as if it would recurse cleanly in the artifact directory; `find`/`rtk ls` worked better for enumerating the dump tree.

Reusable knowledge:
- `step_*.json` snapshots in this artifact are train-monitor dumps; `prepare_failures/*.json` are explicit Channel-B malformed-rollout dumps; `ddp_phase_trace/` contains DDP phase-trace files.
- The per-sample stats block includes useful fields like `raw_valid_pred_objects`, `clean_accepted_pred_objects`, `matched`, `fp_count`, `fn_count`, `precision`, `recall`, `f1`, `duplicate_burst_unlikelihood_boundary_count`, and `parse_truncated`.
- `duplication` fields include `clusters_total`, `clusters_exempt`, `clusters_suppressed`, `objects_suppressed`, `near_iou90_pairs_same_desc_count`, `near_iou90_pairs_any_desc_count`, `duplicate_like_max_cluster_size`, `saturation_rate`, and the anchor index lists.
- `triage` fields include `anchor_support_counts`, `anchor_support_rates`, `shielded_anchor_indices`, `dead_anchor_indices`, `pseudo_positive_anchor_indices`, and `recovered_gt_indices`.
- The dominant global pattern in the early snapshots was not pure duplicate flooding; it was under-coverage after matching/triage: high gate rejections, many dead anchors, and recall lagging precision.

References:
- [1] `output/stage2_ab/prod/2b_lvis_proxy_pseudo_positive_dup_targeting_ckpt1564_merged/.../monitor_dumps/step_000142.json` sample schema and fields.
- [2] Aggregate across 148 early samples: `matched=706`, `fp=543`, `fn=946`, mean precision `0.668`, mean recall `0.565`, mean F1 `0.577`, `gating_rejections=5188`, `dead_anchor_count=471`, `pseudo_positive_selected=32`, `clusters_suppressed=68`, `objects_suppressed=336`.
- [3] Strong burst case in `step_000002.json`: sample `87140591504203` with `128` raw valid predictions, `120` bananas, `objects_suppressed=119`, `duplicate_like_max_cluster_size=120`, `near_iou90_pairs_same_desc_count=829`.
- [4] Strong burst case in `step_000071.json`: sample `87140591505772` with `64` boats, `max_cluster=63`, but `clusters_exempt=1`, `clusters_suppressed=0`, `objects_suppressed=0`.
- [5] `prepare_failures/` contained only 3 files, all tagged `kind: channel_b_prepare_failure` with `policy: dump_and_continue`.
- [6] `src/common/duplicate_control.py` and `src/trainers/rollout_matching/matching.py` were the main code paths used to interpret the counters.

## Task 2: Deeper analysis of failure taxonomy and code-path grounding

Outcome: success

Preference signals:
- The user explicitly requested “Please continue to conduct deeper analysis,” indicating they want the agent to keep digging until the artifact’s main modes are separated rather than stopping at one aggregate verdict.
- The later evidence shows the user values separating “heavy duplication” from “coverage loss” rather than collapsing them into one vague failure label.

Key steps:
- Pulled code for duplicate control and matching to interpret the counters:
  - `pair_is_duplicate_like`
  - `build_duplicate_clusters`
  - `compute_duplicate_metrics`
  - `apply_duplicate_policy`
  - `hungarian_match_maskiou`
- Quantified the dataset into “heavy_dup” vs “no_dup” vs “truncated” groups.
- Counted class-level patterns in predictions and false positives/false negatives.
- Identified that duplicate suppression is often concentrated in a minority of samples, while many other samples fail mainly through missed objects and gating.
- Verified why some very large same-class clusters can be exempt: cluster exemption occurs when a cluster is explorer-supported or spatially spread, so very large clusters may be intentionally kept.

Failures and how to do differently:
- A first attempt to inspect duplicate clusters via a nested `summary` structure was wrong; the correct fields were top-level in `duplication`.
- A later attempt to use a broad cluster-count extractor returned zero because the data structure did not expose the summary in the expected form; direct inspection of per-sample `duplication` objects was necessary.

Reusable knowledge:
- In `src/common/duplicate_control.py`, two objects are duplicate-like only if they share normalized `desc` and either have IoU above threshold or center distance within `center_radius_scale * sqrt(min(area))`.
- `apply_duplicate_policy` can mark a cluster as exempt if at least two members have explorer support or the cluster is spatially spread; in that case, `objects_suppressed` can remain zero even with huge clusters.
- In `hungarian_match_maskiou`, `gating_rejections` increments when a pruned candidate pair fails the mask-IoU gate; this is a genuine geometry-mismatch signal, not just an arbitrary debug count.
- The triage builder `_build_channel_b_triage` only promotes pseudo-positives when support is strong enough; in the analyzed runs, anchor support was overwhelmingly zero, so pseudo-positive rescue remained sparse.
- Heavy duplicate bursts correlated strongly with truncation/saturation-like scenes, but the broader ongoing failure was still coverage loss and low recall, not duplication alone.

References:
- [1] `src/common/duplicate_control.py:233-249` `pair_is_duplicate_like` uses same-desc plus IoU or center-distance logic.
- [2] `src/common/duplicate_control.py:252-283` `build_duplicate_clusters` forms connected components.
- [3] `src/common/duplicate_control.py:286-333` `compute_duplicate_metrics` emits `dup/raw/*` metrics including `duplicate_like_max_cluster_size`, `saturation_rate`, and same-desc near-IoU counts.
- [4] `src/common/duplicate_control.py:496-666` `apply_duplicate_policy` implements explorer-supported / spatially-spread exemptions.
- [5] `src/trainers/rollout_matching/matching.py:104-224` `hungarian_match_maskiou` defines `gating_rejections`.
- [6] Late/hidden burst examples: `87140591504203`, `87140591572482`, `87140591505772`, `87140591498979`, `87140591511156`.
- [7] Descriptor-level heavy-dup pattern: suppressed duplicate descriptions were dominated by `banana`, `cup`, `person`, `chair`, `car`; overall false positives were often `boat`, `person`, `cup`, `pizza`, `chair`.

## Task 3: Check later performance after new dumps arrived

Outcome: success

Preference signals:
- The user explicitly pointed to the same dump directory and asked to “check the later performance,” which suggests future agents should assume the user wants incremental trend analysis when new step files appear, not just a snapshot of the latest file.
- The user’s phrasing “later performance” indicates that trend direction matters more than a single-step headline.

Key steps:
- Re-inventoried the directory and found new snapshots through `step_000300.json`, extending the earlier sequence with `177, 212, 248, 283, 300`.
- Recomputed step-wise metrics separately for early and late train-monitor windows.
- Discovered `step_000300.json` is an `eval` dump with a different schema, so it should not be mixed with the train-monitor snapshots.
- Compared early vs late train windows and isolated best/worst late cases.

Failures and how to do differently:
- One late file (`step_000300.json`) initially skewed aggregation because it is `meta.phase: eval` and has a different sample schema (`gt_objects`/`pred_objects` without the same train-monitor payload fields). Future similar comparisons should exclude eval-format dumps from train-monitor trendlines unless explicitly converting schemas.
- The aggregate late window needed a schema check before interpretation, because the file-level keys were not identical in meaning even though the file shape looked similar.

Reusable knowledge:
- The later train-monitor windows showed a mild decrease in burstiness/truncation but not a clean quality improvement.
- Early vs late train-window comparison:
  - Early: mean precision `0.668`, recall `0.565`, F1 `0.577`, `gating_rejections=35.054`, `objects_suppressed=2.27`, `max_cluster=3.824`, `dead_anchor_count=3.182`, `pseudo_positive_count=0.216`.
  - Late (steps `177, 212, 248, 283`): mean precision `0.635`, recall `0.538`, F1 `0.540`, `gating_rejections=30.158`, `objects_suppressed=1.783`, `max_cluster=2.683`, `dead_anchor_count=2.183`, `pseudo_positive_count=0.25`.
- Step `177` was the strongest late snapshot; steps `212`, `248`, and `283` regressed or plateaued. The late run does not show a clean upward trajectory.
- Heavy duplication still exists late, but it is episodic, not dominant:
  - `step_000177.json` sample `87140591496285`: `pred=127`, `objects_suppressed=109`, `max_cluster=107`, truncated.
  - `step_000283.json` sample `87140591578910`: `pred=51`, `objects_suppressed=39`, `max_cluster=23`.
- The dominant late failure remains missed coverage / recall loss rather than duplicate flooding.
- `step_000300.json` is an eval-format dump with `meta.phase: eval`, `metric_key_prefix: eval`, `rollout_backend: vllm`, `vllm_mode: server`, and per-sample `gt_objects` / `pred_objects` rather than the train-monitor `gt` / `pred` payload shape; it should be treated separately.

References:
- [1] Directory inventory after new dumps: `step_000002.json`, `step_000036.json`, `step_000071.json`, `step_000106.json`, `step_000142.json`, `step_000177.json`, `step_000212.json`, `step_000248.json`, `step_000283.json`, `step_000300.json`.
- [2] Late train-window aggregate over steps `177, 212, 248, 283`: precision `0.635`, recall `0.538`, F1 `0.540`, `gating_rejections=30.158`, `objects_suppressed=1.783`, `max_cluster=2.683`.
- [3] `step_000177.json` meta: `selection: suspicious_duplication`, `candidate_count: 37`, `rollout_backend: vllm`, `decode_mode: greedy`; example heavy burst sample `87140591496285`.
- [4] `step_000300.json` meta: `phase: eval`, `metric_key_prefix: eval`, `rollout_backend: vllm`, `vllm_mode: server`, `decode_mode: greedy`, `max_new_tokens: 3084`, `candidate_top_k: 5`, `maskiou_gate: 0.5`, `maskiou_resolution: 256`.
- [5] Best-late and poor-late examples: `87140591562229`, `87140591554820`, `87140591578910`, `87140591492563`, `87140591573073`.
- [6] Pseudo-positive counts improved only modestly late (`~0.25/sample` in the late train window), which was not enough to recover recall.

