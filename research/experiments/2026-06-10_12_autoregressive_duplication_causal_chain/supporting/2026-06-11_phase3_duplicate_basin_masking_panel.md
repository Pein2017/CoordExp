# Phase 3 Duplicate-Basin Masking Panel

Date: 2026-06-11

## Scope

This note records the first full Phase 3 causal perturbation panel over the
validated Phase 1 manifest. The intervention is duplicate-basin image masking
paired against a no-op replay control for the same replay prefix and coordinate
anchor.

This is causal input-perturbation evidence for visual-basin dependence. It is
not, by itself, a full hidden-state root-cause proof.

## Artifact Root

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133
```

Primary outputs:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/shards/shard-*/phase3_masking/masking_readout_rows.jsonl
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/shards/shard-*/phase3_masking/masking_delta_rows.jsonl
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/phase3_masking_full_panel_summary.json
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/phase3_masking_full_panel_report.md
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/phase3_masking_stratified_summary.json
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/phase3_masking_stratified_report.md
```

## Smoke

Before the full panel, a one-case smoke was run on record `54`,
checkpoint `aligner_parent_ckpt1824`.

Smoke output:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/smoke/phase3_masking_one_case
```

Smoke validation:

- replay cases: `1`;
- intervention plan rows: `2` (`no_op_control`, `duplicate_basin_mask`);
- readout rows: `88`;
- paired delta rows: `44`;
- strongest rank-improvement phase: `post_y1/pre_x2`
  (`mean_target_rank_delta=-10.2727`, `rank_improved_count=11/11`).

## Full Panel Run

All four shards completed on GPUs `0,1,2,3`.

| Shard | GPU | replay cases | intervention plan rows | readout rows | paired anchors |
|---|---:|---:|---:|---:|---:|
| `shard-00-of-04` | 0 | 8 | 16 | 720 | 360 |
| `shard-01-of-04` | 1 | 8 | 16 | 632 | 316 |
| `shard-02-of-04` | 2 | 7 | 14 | 560 | 280 |
| `shard-03-of-04` | 3 | 7 | 14 | 600 | 300 |
| **Total** |  | **30** | **60** | **2512** | **1256** |

The paired-anchor count equals the Phase 1 coordinate-logit row count, so every
coordinate-bearing Phase 1 anchor has a paired no-op/masked readout.

## Full-Panel Phase Summary

`target_prob_delta` is `duplicate_basin_mask - no_op_control` for the target
coordinate token at the replayed anchor.

| Phase | n | mean target-prob delta | mean target-rank delta | rank improved | rank worse | top-1 changed |
|---|---:|---:|---:|---:|---:|---:|
| `box_start/pre_x1` | 314 | -0.001771 | +45.3981 | 81 | 129 | 123 |
| `post_x1/pre_y1` | 314 | -0.002956 | +20.2643 | 81 | 135 | 148 |
| `post_y1/pre_x2` | 314 | -0.002468 | +6.9045 | 108 | 136 | 169 |
| `post_x2/pre_y2` | 314 | +0.000292 | +5.6178 | 113 | 141 | 161 |

The broad pattern is that masking the duplicate basin usually reduces target
coordinate probability or worsens rank for the x1/y1/x2 decision surface. The
y2 decision surface is mixed, with a small positive mean probability delta but
still a positive mean rank delta.

## Stratified Checkpoint Summary

| Checkpoint | n | records | mean target-prob delta | mean target-rank delta | rank improved | rank worse | top-1 changed |
|---|---:|---:|---:|---:|---:|---:|---:|
| `aligner_parent_ckpt1824` | 232 | 5 | -0.000620 | +19.9784 | 79 | 87 | 113 |
| `aux_latest_ckpt32` | 324 | 8 | -0.001422 | +24.3488 | 101 | 150 | 154 |
| `no_aligner_parent_ckpt3668` | 452 | 11 | -0.001853 | +10.0509 | 136 | 177 | 221 |
| `none_latest_ckpt32` | 248 | 6 | -0.002923 | +30.1734 | 67 | 127 | 113 |

The strongest checkpoint-level negative probability effect is
`none_latest_ckpt32`, which is also the continuation checkpoint where the
earlier Phase 2 layer-17/head-1 table showed record `33` as the highest
duplicate-density routing case.

## Strongest Checkpoint-Phase Effects

Largest negative mean target-probability deltas:

| Checkpoint | Phase | n | mean target-prob delta | mean target-rank delta |
|---|---|---:|---:|---:|
| `none_latest_ckpt32` | `post_x1/pre_y1` | 62 | -0.004443 | +15.3871 |
| `none_latest_ckpt32` | `post_y1/pre_x2` | 62 | -0.004356 | +9.3065 |
| `no_aligner_parent_ckpt3668` | `post_x1/pre_y1` | 113 | -0.003925 | +14.2743 |
| `no_aligner_parent_ckpt3668` | `box_start/pre_x1` | 113 | -0.002499 | +14.1239 |
| `aux_latest_ckpt32` | `post_x1/pre_y1` | 81 | -0.002378 | +10.0370 |

The intervention result strengthens the current mechanism picture: selected
duplicate-basin visual regions are not merely attended to; masking them changes
the coordinate-token readout under fixed replay prefixes. The effect is most
consistent on early/mid coordinate decisions and strongest in the `none`
continuation checkpoint.

## Interpretation Boundary

This does not yet prove that image-region masking would fix full free-rollout
duplication or that the visual basin is the only causal source. It shows that
the duplicate-basin image region participates causally in the coordinate-token
distribution at selected replay anchors. Further work should connect this to
no-op continuation replay parity and to targeted route/content or hidden-state
patches for the high-density records, especially record `33`.
