---
title: Stage-1 2B Val200 Detection Leaderboard
date: 2026-05-12
status: active-dashboard
topics: [stage1, 2b, val200, leaderboard, detection, coord-token, raw-text]
tags: [benchmarks, dashboard, leaderboard, val200, 2b]
summary: Live progress-layer dashboard for Stage-1 2B-ish val200 detection benchmarks already recorded under progress, with a CSV backing table for scriptable comparisons.
---

# Stage-1 2B Val200 Detection Leaderboard

This dashboard consolidates the currently recorded `val200` + `2B` detection
benchmarks from `progress/`.

The machine-readable table is:

- [artifacts/stage1_2b_val200_leaderboard.csv](artifacts/stage1_2b_val200_leaderboard.csv)

## Scope And Rules

Included rows must satisfy all of the following:

- recorded under `progress/`
- detection-style benchmark or diagnostic benchmark with explicit metrics
- 2B or 2B-derived checkpoint family
- `val200`, `limit=200`, or first-200 validation slice
- AP-style detection metric available

Primary sort:

- descending `coco_real` or equivalent AP

Important caveat:

- This is a useful operational leaderboard, but not every row is strictly
  apples-to-apples.
- Use the CSV `comparable_group` column before making a paper-table claim.
- Differences include sequence format, checkpoint export style, `max_new_tokens`,
  dataset surface, proxy-eval policy, and launch shape.

## Current Topline Ranking

| Rank | Entry | AP | AP50 | AP75 | F1@0.50 | Group | Notes |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | `coord_token_ckpt1332_rp105_sweep` | `0.4584` | `0.6307` | `0.4770` | `0.7056` | `coord_token_rp_sweep` | Best old mixed-objective coord-token RP cell. |
| 2 | `coord_token_ckpt1332_rp100_sweep` | `0.4532` | `0.6177` | `0.4755` | `0.6626` | `coord_token_rp_sweep` | Same checkpoint, lower RP. |
| 3 | `coord_token_ckpt1332_rp110_sweep` | `0.4419` | `0.6064` | `0.4579` | `0.7002` | `coord_token_rp_sweep` | Same checkpoint, higher RP. |
| 4 | `compact_full_et_rmp_support2_ckpt3664` | `0.4247` | `0.5752` | `0.4477` | `0.6138` | `compact_full` | Different compact-full sequence format. |
| 5 | `center_parameterization` | `0.4221` | `0.6007` | n/a | `0.6108` | `coord_family_diagnostic` | Strong center-parameterization family row. |
| 6 | `mixed_objective_sota_ckpt1332` | `0.4137` | `0.5910` | `0.4203` | `0.6926` | `mixed_objective_probe` | Adapter-runtime probe with `max_new_tokens=1024`. |
| 7 | `compact_full_prefix_rollin_a4_eos_ckpt3664` | `0.4001` | `0.5502` | n/a | `0.5612` | `compact_full_prefix_rollin_followup` | A4 follow-up: EOS-trust prior improves A3 AP/F1 but has more invalid/collapse risk. |
| 8 | `compact_full_random_sft_bsz1_accum16_ckpt3664` | `0.3992` | `0.5577` | `0.4140` | `0.4433` | `compact_full` | Different compact-full sequence format. |
| 9 | `compact_full_prefix_rollin_a3_ckpt3664` | `0.3980` | `0.5444` | n/a | `0.5597` | `compact_full_prefix_rollin_followup` | A3 follow-up: cleaner than A4 but below A2 and random-SFT AP. |
| 10 | `coord_component_hard_ce_rp110` | `0.3901` | `0.5042` | `0.3954` | `0.5582` | `coord_component_ablation` | Current clean coord-component ablation winner. |
| 11 | `historical_ce_softce_mixed_2b_768` | `0.3896` | `0.5628` | `0.3905` | `0.6569` | `historical_res_sweep` | Historical 768-res merged-model result. |
| 12 | `historical_ce_softce_mixed_2b_1024` | `0.3879` | `0.5599` | `0.3963` | `0.6444` | `historical_res_sweep` | Historical 1024-res merged-model result. |

The full table, including weaker diagnostic references and raw-text cells, is
in the CSV.

## Best Rows By Comparable Group

| Comparable group | Best entry | AP | Why this group exists |
|---|---|---:|---|
| `coord_token_rp_sweep` | `coord_token_ckpt1332_rp105_sweep` | `0.4584` | Same old mixed-objective coord-token checkpoint across RP settings. |
| `compact_full` | `compact_full_et_rmp_support2_ckpt3664` | `0.4247` | Compact-full recursive-detection sequence format. |
| `compact_full_prefix_rollin_followup` | `compact_full_prefix_rollin_a4_eos_ckpt3664` | `0.4001` | A3/A4 prefix-rollin follow-up; useful as a behavior study, not yet a more reliable replacement for A2. |
| `coord_family_diagnostic` | `center_parameterization` | `0.4221` | Cross-family diagnostic comparison across coordinate parameterizations. |
| `mixed_objective_probe` | `mixed_objective_sota_ckpt1332` | `0.4137` | Focused adapter-runtime probe of the strong mixed-objective checkpoint. |
| `coord_component_ablation` | `coord_component_hard_ce_rp110` | `0.3901` | Clean 2B CoordExp coordinate-loss component ablation. |
| `historical_res_sweep` | `historical_ce_softce_mixed_2b_768` | `0.3896` | Historical 768-vs-1024 merged-model comparison. |
| `raw_text_rp_sweep` | `raw_text_ckpt552_rp110` | `0.3782` | Raw-text xyxy checkpoint across RP settings after scorer repair. |

## Current Read

The progress layer currently says three things at once:

1. The strongest recorded `val200` 2B-ish AP row is still the older
   mixed-objective coord-token checkpoint at `rp=1.05` from the RP sweep.
2. Within the newly cleaned coord-component ablation surface, the current best
   row is `hard_ce_only + rp=1.10`.
3. Within compact-full recursive detection, the older A2/support+balance row is
   still the best AP row. A3/A4 prefix-rollin follow-ups did not beat A2 on this
   `val200` surface; A4 improved over A3, but its rollout behavior is less
   predictable and carries more duplicate/collapse risk.

These are not contradictory. They answer different questions:

- the old mixed-objective checkpoint bundles hard CE, soft CE, W1, and gates,
  and remains the strongest historical 2B reference in `progress`;
- the new coord-component matrix intentionally isolates simpler loss choices,
  where `hard_ce_only + rp=1.10` is the clean current winner.
- the compact-full prefix-rollin A3/A4 follow-up is evidence about continuation
  behavior, not a new compact-full group winner.

## Rows Not Treated As Primary Leaderboard Entries

Some `progress/` files mention first-200 or val200-like results but are not in
the primary CSV leaderboard because their scope is not a standard Stage-1 2B
single-decode detector comparison:

- Stage-2 Channel-A historical note:
  LVIS bbox-only `768`, `limit=100` or `1000`, historical `stage2_ab`.
- Stage-2 Oracle-K and rollout-temperature notes:
  useful for latent recoverability and rollout policy, but not a direct
  single-decode Stage-1 2B AP leaderboard.
- ET-RMP continuation `core6` diagnostics:
  core-six research subset, not `val200`.

## Maintenance Contract

When adding a new row:

- add it to
  [artifacts/stage1_2b_val200_leaderboard.csv](artifacts/stage1_2b_val200_leaderboard.csv)
- preserve `entry_id` stability
- fill `comparable_group`
- put missing values as blank cells
- keep `rank_ap` sorted by AP descending
- update the topline table only if the row enters the top band or changes a
  group winner

This dashboard is intentionally lightweight: it is a progress-layer scoreboard,
not a replacement for raw run artifacts.
