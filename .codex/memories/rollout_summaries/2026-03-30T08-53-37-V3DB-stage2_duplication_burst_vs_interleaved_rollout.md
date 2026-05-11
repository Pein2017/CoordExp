thread_id: 019d3df2-d3c5-7e91-aef5-8c0449e96d0b
updated_at: 2026-03-30T12:46:55+00:00
rollout_path: /data/CoordExp/.codex/sessions/2026/03/30/rollout-2026-03-30T08-53-37-019d3df2-d3c5-7e91-aef5-8c0449e96d0b.jsonl
cwd: /data/CoordExp
git_branch: main

# The user’s duplication question was answered from actual rollout artifacts, and the empirical conclusion was that heavy duplication is usually a contiguous burst rather than an alternating/interleaved pattern.

Rollout context: The user first corrected an earlier GT-only proxy and insisted the real task was to separate model rollout duplication from GT-acceptable overlap using real `output/stage2_ab/prod/` artifacts. After that, the user asked a narrower follow-up: for heavy duplication, are the repeated predictions generated as a contiguous burst (e.g. `cat1, cat2, cat3, ..., dog`) or as an interleaved sequence (e.g. `cat1, dog, cat2, cat3, bird`)? The assistant inspected the serialized `gt_vs_pred.jsonl` review bundles and the Stage-2 rollout/dedup code to answer from evidence rather than guessing.

## Task 1: Separate model-rollout duplication from GT-acceptable overlap

Outcome: success

Preference signals:

- The user explicitly corrected the earlier approach: “No! The task was to find a real proxy that allows/admits the real annotations in the `GT` and doesn't treat those GT as `same_instance_proxy` and manage to find a real way to separate the `model's rollout duplication` VS `GT acceptable overlapping.`” -> future work should not use GT-only geometric proxies when the user is asking about rollout failure modes; use actual rollout artifacts and GT only as context/oracle.
- The user said “You may refer to my real rollout artifacts under `output/stage2_ab/prod/`” -> future analysis should prefer real artifact paths under `output/stage2_ab/prod/` when diagnosing model behavior.
- The user later asked “This is a task without `GT response` and it's more like a research attempt.” -> the default should be exploratory, evidence-first analysis rather than a prescriptive answer.

Key steps:

- Built and iterated on a standalone analysis over Stage-2 monitor dumps and review artifacts, then explicitly pivoted to rollout-based duplication evidence after the user correction.
- Located review bundles under `output/stage2_ab/prod/.../vis_resources/gt_vs_pred.jsonl` and used those rather than GT-only JSONL pairs.
- Inspected the actual serialized record structure: top-level keys `schema_version`, `source_kind`, `record_idx`, `image`, `width`, `height`, `coord_mode`, `gt`, `pred`, `matching`, `provenance`.
- Verified that the review bundles contain ordered prediction sequences nested under `pred`, with GT and matching metadata available for context.

Failures and how to do differently:

- The first analysis path was GT-only and the user rejected it. Future similar tasks should immediately ask whether the target is GT structure, rollout structure, or both, and should not infer a rollout-failure proxy from GT alone.
- A temporary-analysis cleanup step removed more ignored `temp/` artifacts than intended. The safe practice is to avoid broad `git clean -fdX` on shared temp roots unless the exact ignored paths are fully enumerated and the user is okay with collateral deletion.

Reusable knowledge:

- The rollout review artifacts already contain the ordering needed to study duplication bursts; the useful schema is `pred` as an ordered object list plus `matching` for context.
- The Stage-2 review bundles are good evidence sources for this question because they were already curated around suspicious duplication/high-FN duplicate cases.

References:

- [1] `output/stage2_ab/prod/pseudo_positive_hardened_spiky_coord-from_stage1/epoch_1-ciou1p0-coordce0p04-soft0p05-w10p01/v0-20260327-073530/monitor_dumps/review_high_fn_duplicate_20260327_fixed/vis_resources/gt_vs_pred.jsonl`
- [2] `output/stage2_ab/prod/pseudo_positive/k_4-eff_size_96-b_ratio_0.75-epoch_1/v2-20260324-062041/analysis/suspicious_duplication_review/vis_resources/gt_vs_pred.jsonl`
- [3] `output/stage2_ab/prod/pseudo_positive/k_4-eff_size_96-b_ratio_0.75-epoch_1/v2-20260324-062041/analysis/suspicious_duplication_review_step160/vis_resources/gt_vs_pred.jsonl`
- [4] The review record schema observed in the first sample: `gt`, `pred`, `matching`, `provenance` nested under the top-level JSON object.

## Task 2: Contiguous burst vs interleaved heavy duplication

Outcome: success

Preference signals:

- The user asked directly: “For those `heavy duplication`, were they generated in sequence/oder like: `cat1,cat2,cat3,...,dog` or `cat1,dog,cat2,cat3,bird` ? Are they Contiguous Burst or `Interleaved`?” -> future answers should classify the sequence form explicitly rather than only describing duplication severity.
- The user framed the question as ordering/sequence behavior, not just count or overlap, so future analyses should inspect prediction order and run-length structure.

Key steps:

- Read the actual ordered prediction lists from the review JSONL bundles rather than inferring from summaries.
- Computed the dominant same-desc run length per sample, its contiguity, and interleaving slots in each review bundle.
- Cross-checked against Stage-2 rollout code that parses predictions in order and treats duplicates as boundary-indexed bursts: `build_channel_b_rollout_view` and `_sequential_dedup_bbox_objects`.
- Compared several representative cases, including highly repetitive ones (`apple` x128, `carrot` x46, `sports ball` x18, `chair` x13, etc.) and a mixed case (`chair` separated by person/cell phone/baseball bat/bench) to distinguish pure bursts from partial interleaving.

Failures and how to do differently:

- The first attempt at parsing the JSONL assumed a different nesting (`pred_objects`/`gt_objects`) and failed; the correct schema had `pred` as either a list or a nested object with `objects`. Future similar inspections should open one raw record first, then branch parsing logic from the observed schema.

Reusable knowledge:

- The heavy duplication cases are **mostly contiguous bursts**, not alternating interleavings.
- Representative ordered sequences showed long repeated blocks after a short prefix or occasional separator, e.g.:
  - `000000015759.jpg`: `apple` repeated 128 times contiguously.
  - `000000072729.jpg`: `broccoli, bowl, broccoli, carrot, carrot, ...` with a long contiguous `carrot` burst.
  - `000000240403.jpg`: `person` prefix, then a long `sports ball` run, then a `tennis racket` tail.
  - `000000150646.jpg`: mixed prefix/noise, then a long `cup` run with only small interruptions.
  - `000000457503.jpg`: a more mixed example with two chair bursts separated by a person-heavy block, but still bursty rather than object-by-object alternation.
- Aggregate run-length measurements across the checked bundles supported the same conclusion:
  - fixed review: mean contiguous fraction of the dominant label run ≈ `0.959`
  - suspicious review: mean contiguous fraction ≈ `0.873`
  - step160 review: mean contiguous fraction ≈ `0.896`
- The Stage-2 implementation aligns with this interpretation: `build_channel_b_rollout_view` preserves prediction order, and `_sequential_dedup_bbox_objects` records duplicates as boundary-indexed bursts rather than as globally interleaved events.

References:

- [1] `output/stage2_ab/prod/pseudo_positive_hardened_spiky_coord-from_stage1/epoch_1-ciou1p0-coordce0p04-soft0p05-w10p01/v0-20260327-073530/monitor_dumps/review_high_fn_duplicate_20260327_fixed/vis_resources/gt_vs_pred.jsonl`
- [2] `output/stage2_ab/prod/pseudo_positive/k_4-eff_size_96-b_ratio_0.75-epoch_1/v2-20260324-062041/analysis/suspicious_duplication_review/vis_resources/gt_vs_pred.jsonl`
- [3] `output/stage2_ab/prod/pseudo_positive/k_4-eff_size_96-b_ratio_0.75-epoch_1/v2-20260324-062041/analysis/suspicious_duplication_review_step160/vis_resources/gt_vs_pred.jsonl`
- [4] `src/trainers/stage2_two_channel/rollout_views.py` — ordered rollout parsing.
- [5] `src/trainers/stage2_two_channel/target_builder.py` (`_sequential_dedup_bbox_objects`) — duplicates are attached as boundary-indexed bursts.
- [6] Representative sequences observed directly in the artifacts:
  - `000000015759.jpg` → `apple` x128 contiguous
  - `000000072729.jpg` → `carrot` long contiguous burst after a short prefix
  - `000000240403.jpg` → `sports ball` long contiguous burst
  - `000000457503.jpg` → mixed with two chair runs separated by other labels, but still not fully interleaved

