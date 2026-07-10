# Phase 1 Full Hidden/Logit Panel

Date: 2026-06-11

## Scope

This note records the full four-shard Phase 1 hidden-state and coordinate-logit
readout over the validated Phase 0 selected-window manifest. This is a
forward-readout panel, not a causal intervention result.

## Artifact Root

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133
```

Inputs:

- Phase 0 root:
  `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase0_joint_onset_ledger_20260610-073246`
- selected windows: `30`
- token-window rows: `2512`
- anchor status: `ok=2512`
- shards: `4`, assigned to GPUs `0,1,2,3`

## Forward Run

All four shard forward readouts completed successfully.

| Shard | GPU | replay cases | checkpoint loads | hidden rows | coord-logit rows |
|---|---:|---:|---:|---:|---:|
| `shard-00-of-04` | 0 | 8 | 3 | 6480 | 360 |
| `shard-01-of-04` | 1 | 8 | 2 | 5688 | 316 |
| `shard-02-of-04` | 2 | 7 | 3 | 5040 | 280 |
| `shard-03-of-04` | 3 | 7 | 3 | 5400 | 300 |
| **Total** |  | **30** |  | **22608** | **1256** |

Per-shard outputs live under:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/shards/shard-*/forward_readouts
```

## Reduced Outputs

Reducer outputs:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/phase1_coord_logit_onset_summary.jsonl
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/phase1_hidden_state_onset_summary.jsonl
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/phase1_transition_summary.json
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/phase1_transition_report.md
```

Reduction summary:

- coordinate summary rows: `192`;
- hidden-state summary rows: `3456`;
- coordinate phases:
  - `box_start/pre_x1`;
  - `post_x1/pre_y1`;
  - `post_y1/pre_x2`;
  - `post_x2/pre_y2`;
- hidden layers: `0,4,8,12,16,20,24,28,-1`;
- checkpoints present:
  - `no_aligner_parent_ckpt3668`;
  - `aligner_parent_ckpt1824`;
  - `aux_latest_ckpt32`;
  - `none_latest_ckpt32`.

## First Coordinate-Alignment Readout

Best coordinate alignment by checkpoint in the reducer summary:

| Checkpoint | Best phase | Relative row offset | rank-1 fraction | mean target rank | mean target prob | mean top-1 distance |
|---|---|---:|---:|---:|---:|---:|
| `aligner_parent_ckpt1824` | `post_y1/pre_x2` | -1 | 0.6000 | 5.4000 | 0.182718 | 3.8000 |
| `aux_latest_ckpt32` | `box_start/pre_x1` | 5 | 0.5000 | 9.3333 | 0.008714 | 13.0000 |
| `no_aligner_parent_ckpt3668` | `post_x1/pre_y1` | 0 | 0.6364 | 9.0000 | 0.043366 | 8.4545 |
| `none_latest_ckpt32` | `post_y1/pre_x2` | 0 | 0.5000 | 2.5000 | 0.033340 | 4.3333 |

Interpretation guardrail: this table identifies dynamic coordinate-slot
alignment inside replayed selected windows. It is evidence for when coordinate
basins become sharp in the chosen windows, but it is not by itself causal proof
of the duplication trigger. Attention/routing readouts and interventions remain
required before assigning direct-trigger status.

## Next Step

Proceed to Phase 2 region materialization and attention/routing readouts using
the same manifest root. Keep the Phase 1 selected windows fixed; do not reselect
cases based on this first coordinate alignment table.
