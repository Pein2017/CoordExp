thread_id: 019d2d68-d052-7a11-b721-47bdf20cc045
updated_at: 2026-03-27T05:20:07+00:00
rollout_path: /data/CoordExp/.codex/sessions/2026/03/27/rollout-2026-03-27T03-48-57-019d2d68-d052-7a11-b721-47bdf20cc045.jsonl
cwd: /data/CoordExp
git_branch: main

# Heavy-duplication inspection and top-10 visualization for Stage-2 monitor dumps

Rollout context: The user asked whether the `heavy duplication` issue was improving during training in a Stage-2 AB run, then asked for the most duplicated images/predictions in selected monitor dumps (`step_000177.json`, `step_000212.json`, `step_000248.json`) and requested visualizations for manual inspection. The work was done from `/data/CoordExp` and focused on `output/stage2_ab/prod/pseudo_positive_hardened_spiky_coord/epoch_1-ciou1p0-coordce0p04-soft0p05-w10p01-from_300_v1/v1-20260326-075838/`.

## Task 1: Determine whether heavy duplication was improving over training

Outcome: success

Preference signals:

- The user asked, “Please inspect this run, whether the `heavy duplication` issue is being improved as the model training,” indicating they wanted a trajectory-based assessment rather than a single-endpoint read.
- Later they asked for “Among these dump rollouts, any `heavy` duplication trajectories?” which reinforces that the user wants duplicated-case identification from monitor dumps, not just aggregate metrics.

Key steps:

- The assistant inspected `logging.jsonl` and confirmed it contained training-side duplication telemetry such as `stage2_ab/channel_b/dup/N_duplicates`, `dup/near_iou90_pairs_same_desc_count`, `rollout/anchor/near_iou90_same`, `rollout/parse_truncated_rate`, `rollout/precision`, and `rollout/f1`.
- The run had **no `eval/*` rows** in `logging.jsonl`, so the conclusion was explicitly limited to training rollouts / train monitor dumps, not eval predictions.
- A compact trajectory over Channel-B rows showed a clear pattern: very bad early collapse, some much cleaner middle windows, then relapse later.
- Representative progression from the log:
  - step 10: `N_duplicates = 240`, `near_iou90_pairs_same_desc_count = 7436`, `precision = 0.3596`, `f1 = 0.4815`
  - step 70: `N_duplicates = 0`, `near_iou90_pairs_same_desc_count = 1`, `precision = 0.5066`, `f1 = 0.5719`
  - step 100: `N_duplicates = 3`, `precision = 0.6052`, `f1 = 0.6589`
  - step 190: relapse with `N_duplicates = 125`, `near_iou90_pairs_same_desc_count = 3231`, `precision = 0.3120`, `f1 = 0.4266`
  - step 260: still not clean, with `N_duplicates = 37`, `precision = 0.4053`, `f1 = 0.5137`
- The conclusion recorded in the rollout was that the run is **improved from the initial collapse, but metastable/bursty rather than solved**.

Reusable knowledge:

- In these Stage-2 AB logs, the most useful duplicate-stability canaries were:
  - `stage2_ab/channel_b/dup/N_duplicates`
  - `dup/near_iou90_pairs_same_desc_count`
  - `rollout/anchor/near_iou90_same`
  - `rollout/parse_truncated_rate`
  - `rollout/precision`
  - `rollout/f1`
- A single best row is misleading; windowed reads are better. In this run, early-to-mid improvement was real, but late-window relapse still happened.

Failures and how to do differently:

- The assistant initially tried to judge “heavy duplication” from the log alone before checking whether eval artifacts existed. It later confirmed `logging.jsonl` had no `eval/*` rows and correctly narrowed the claim to training-side evidence only.
- The first attempt to render selected samples failed because the generated filtered JSONs did not match the renderer’s expected directory/file pattern. The fix was to preserve the original multimodal `messages` and rename filtered files to `step_*.json` so the canonical renderer could discover them.

References:

- [1] `output/stage2_ab/prod/pseudo_positive_hardened_spiky_coord/epoch_1-ciou1p0-coordce0p04-soft0p05-w10p01-from_300_v1/v1-20260326-075838/logging.jsonl`
- [2] `output/stage2_ab/prod/pseudo_positive_hardened_spiky_coord/epoch_1-ciou1p0-coordce0p04-soft0p05-w10p01-from_300_v1/v1-20260326-075838/monitor_dumps/step_000177.json`
- [3] `output/stage2_ab/prod/pseudo_positive_hardened_spiky_coord/epoch_1-ciou1p0-coordce0p04-soft0p05-w10p01-from_300_v1/v1-20260326-075838/monitor_dumps/step_000212.json`
- [4] `output/stage2_ab/prod/pseudo_positive_hardened_spiky_coord/epoch_1-ciou1p0-coordce0p04-soft0p05-w10p01-from_300_v1/v1-20260326-075838/monitor_dumps/step_000248.json`
- [5] `output/stage2_ab/prod/pseudo_positive_hardened_spiky_coord/epoch_1-ciou1p0-coordce0p04-soft0p05-w10p01-from_300_v1/v1-20260326-075838/temp_heavy_dup_review_top10/top10_manifest.json`

## Task 2: Rank the top duplicated samples and render a review pack

Outcome: success

Preference signals:

- The user explicitly asked, “If exist, please help me visualize the top 10 duplicated image and prediction. I’ll inspect them manually,” which is a strong signal that future similar requests should prioritize a ranked visual pack over a prose-only summary.
- The user provided exact dump paths and asked for manual inspection, implying they value artifact generation that is easy to open directly.

Key steps:

- The assistant used the repo’s canonical visualization stack rather than building a custom renderer:
  - `vis_tools/vis_monitor_dump_gt_vs_pred.py`
- It ranked cases across the three requested dumps by duplication severity using the structured per-sample fields already present in the monitor dumps, especially:
  - `duplication.near_iou90_pairs_same_desc_count`
  - `duplication.duplicates`
  - `duplication.max_desc_count`
  - `stats.precision`, `stats.recall`, `stats.f1`, and `stats.parse_truncated`
- It created a top-10 manifest and a review folder with rendered PNGs.
- The final rendered pack was placed under:
  - `output/stage2_ab/prod/pseudo_positive_hardened_spiky_coord/epoch_1-ciou1p0-coordce0p04-soft0p05-w10p01-from_300_v1/v1-20260326-075838/temp_heavy_dup_review_top10/`
- The canonical renderer succeeded after the filtered JSONs were renamed to the `step_*.json` pattern expected by the tool.

Reusable knowledge:

- The monitor dump samples already contain the best high-signal fields for ranking duplication without reverse-engineering the raw text:
  - `sample['duplication']`
  - `sample['stats']`
  - `sample['pred']` / `sample['pred_objects']`
  - `sample['messages']` for renderer fallbacks
- The canonical renderer `vis_tools/vis_monitor_dump_gt_vs_pred.py` can render a directory of `step_*.json` files, and it relies on `_extract_image_path`, which falls back from top-level `image` / `image_path` into the multimodal `messages`.
- If you create filtered one-sample review JSONs, keep the original `messages` structure intact and name the files `step_*.json`; otherwise the renderer may render `0` images.
- For ranked manual review, a manifest file plus one PNG per sample is a good artifact pair:
  - `top10_manifest.json`
  - a `rendered/` folder with the PNGs in rank order.

Failures and how to do differently:

- The first filtered pack failed to render because the synthetic JSON filenames did not match the renderer’s `step_*.json` directory convention.
- The assistant diagnosed that the renderer’s directory traversal only globbed `step_*.json`, renamed the files accordingly, and reran the render successfully.
- The user did not ask for a custom plotting implementation; future similar tasks should default to the shared renderer and only fall back to custom visualization if the canonical path truly cannot express the request.

References:

- [1] Canonical renderer: `vis_tools/vis_monitor_dump_gt_vs_pred.py`
- [2] Output manifest: `output/stage2_ab/prod/pseudo_positive_hardened_spiky_coord/epoch_1-ciou1p0-coordce0p04-soft0p05-w10p01-from_300_v1/v1-20260326-075838/temp_heavy_dup_review_top10/top10_manifest.json`
- [3] Rendered review folder: `output/stage2_ab/prod/pseudo_positive_hardened_spiky_coord/epoch_1-ciou1p0-coordce0p04-soft0p05-w10p01-from_300_v1/v1-20260326-075838/temp_heavy_dup_review_top10/rendered/`
- [4] Selected one-sample monitor JSONs: `output/stage2_ab/prod/pseudo_positive_hardened_spiky_coord/epoch_1-ciou1p0-coordce0p04-soft0p05-w10p01-from_300_v1/v1-20260326-075838/temp_heavy_dup_review_top10/selected_monitor_jsons/`
- [5] Exact top-ranked render files:
  - `step_000177_s00_base093685.png` — `bottle x58`, `pairs=1655`, `duplicates=57`, `parse_truncated=True`
  - `step_000212_s00_base007677.png` — `wine glass x47`, `pairs=1128`, `duplicates=60`, `parse_truncated=True`
  - `step_000248_s00_base108409.png` — `suitcase x51`, `pairs=120`, `duplicates=20`, `parse_truncated=True`
  - `step_000212_s00_base045805.png` — `cup x26`, `pairs=13`, `duplicates=4`
  - `step_000248_s00_base092952.png` — `person x54`, `pairs=4`, `duplicates=0`
  - `step_000248_s00_base101872.png` — `person x10`, `pairs=3`, `duplicates=1`
  - `step_000212_s00_base058296.png` — `backpack x84`, `pairs=3`, `duplicates=0`
  - `step_000177_s00_base105858.png` — `pizza x1`, `pairs=1`, `duplicates=1`
  - `step_000248_s00_base073838.png` — `book x71`, `pairs=1`, `duplicates=0`
  - `step_000177_s00_base027741.png` — `cup x102`, `pairs=0`, `duplicates=0`

## Task 3: Interpret which selected cases are “true heavy duplication” versus just class spam

Outcome: success

Preference signals:

- The user asked for “heavy” duplication trajectories, not just any repeated-class scene. That implies the next agent should distinguish geometric near-duplication from mere high cardinality.

Key steps:

- The ranked pack showed that the top 3 are the strongest geometric collapse cases.
- The later ranks are a mix of:
  - genuine but weaker duplication,
  - over-cardinality / class spam,
  - and truncation-heavy outputs.
- This distinction was made explicit in the final response: the truly heavy geometric duplication is mostly the top 3, with rank 4 moderate, and the rest useful but not equally “heavy.”

Reusable knowledge:

- `near_iou90_pairs_same_desc_count` is the best first-pass geometric-duplication ranker, but it should be interpreted alongside `duplicates`, `max_desc_count`, and `parse_truncated`.
- A sample can have very high same-class repetition while still having zero or low near-IoU duplication; those cases are better described as class spam / repeated enumeration rather than geometric near-dup collapse.

Failures and how to do differently:

- The initial attempt to rank by a generated key encountered a tie-sorting issue. The fix was to add a deterministic tie-break key before sorting.
- The final result preserved the user’s manual-inspection goal by giving explicit PNG filenames and a manifest rather than only aggregate counts.

References:

- [1] `output/stage2_ab/prod/pseudo_positive_hardened_spiky_coord/epoch_1-ciou1p0-coordce0p04-soft0p05-w10p01-from_300_v1/v1-20260326-075838/temp_heavy_dup_review_top10/top10_manifest.json`
- [2] `output/stage2_ab/prod/pseudo_positive_hardened_spiky_coord/epoch_1-ciou1p0-coordce0p04-soft0p05-w10p01-from_300_v1/v1-20260326-075838/temp_heavy_dup_review_top10/rendered/class_summary.json`
- [3] `vis_tools/vis_monitor_dump_gt_vs_pred.py` expects a directory of `step_*.json` files and writes `class_summary.json` plus rendered PNGs.
