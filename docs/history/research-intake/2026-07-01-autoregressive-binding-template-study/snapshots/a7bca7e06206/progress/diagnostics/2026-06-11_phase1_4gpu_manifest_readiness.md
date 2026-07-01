# Phase 1 Four-GPU Manifest Readiness

Date: 2026-06-11

## Scope

This note records the Phase 1 readiness update after validating the Phase 0
joint onset ledger. The available execution budget for this continuation is
GPUs `0,1,2,3`, so Phase 1 manifest sharding and downstream shard aggregators
were aligned to a four-shard run instead of the earlier eight-shard assumption.

## Phase 0 Gate

Validated Phase 0 root:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase0_joint_onset_ledger_20260610-073246
```

Validation summary:

- ledger rows: `512`;
- selected windows: `30`;
- checkpoint labels: `aligner_parent_ckpt1824`, `aux_latest_ckpt32`,
  `no_aligner_parent_ckpt3668`, `none_latest_ckpt32`;
- required records present in selected windows: `33`, `47`, `54`, `82`, `96`;
- selected-window trace paths: no missing paths.

This satisfies the roadmap gate that GPU probes should wait until Phase 0
selected windows validate.

## Phase 1 Manifest

Fresh four-shard manifest root:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133
```

Manifest summary:

- selected windows: `30`;
- shard count: `4`;
- GPU ids: `0,1,2,3`;
- shard selected-window counts: `8,8,7,7`;
- token-window rows: `2512`;
- phase anchor status: `ok=2512`;
- generated command file now contains concrete token-window, forward-shard, and
  reduce commands rather than a placeholder runtime string.

## Code Readiness

Updated surfaces:

- `configs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_val128.yaml`
  now uses four shards on GPUs `0,1,2,3`.
- `src/analysis/autoregressive_duplication_mechanism/phase1_manifest.py`
  writes concrete Phase 1 command stubs.
- Phase 1, Phase 2, and Phase 3 root reducers now glob `shard-*-of-*` instead
  of hard-coding `shard-*-of-08`, so four-shard artifacts will be aggregated.

## Verification

Focused direct test execution passed for the affected modules:

```text
tests/analysis/autoregressive_duplication_mechanism/test_phase1_manifest.py
tests/analysis/autoregressive_duplication_mechanism/test_phase1_reduce.py
tests/analysis/autoregressive_duplication_mechanism/test_phase2_attention.py
tests/analysis/autoregressive_duplication_mechanism/test_phase2_regions.py
tests/analysis/autoregressive_duplication_mechanism/test_phase3_masking.py
```

Result: `31` direct test functions passed.

Artifact validator passed for:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133
```

The next bounded step is a Phase 1 hidden/logit GPU smoke over the fresh
four-shard manifest, preferably starting with one shard or a tiny token-window
slice before launching all four shard forward passes.

## GPU Smoke

Completed one Phase 1 hidden/logit forward smoke on GPU `0`.

Smoke input:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/smoke/token_windows_one_case.jsonl
```

Smoke output:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/smoke/forward_readouts_one_case
```

Smoke case:

- checkpoint label: `aligner_parent_ckpt1824`;
- record index: `54`;
- token-window rows: `88`;
- replay cases: `1`;
- checkpoint loads: `1`;
- hidden-state rows: `792`;
- coordinate-logit rows: `44`;
- layers: `0,4,8,12,16,20,24,28,-1`;
- hidden phases: `row_start`, `desc_end`, `box_start/pre_x1`,
  `post_x1/pre_y1`, `post_y1/pre_x2`, `post_x2/pre_y2`,
  `post_y2/row_boundary`, `next_row_or_stop_decision`;
- coordinate phases: `box_start/pre_x1`, `post_x1/pre_y1`,
  `post_y1/pre_x2`, `post_x2/pre_y2`.

The smoke verifies that the current forward adapter can consume Phase 0-derived
token windows and write Phase 1 hidden/logit readouts under the four-GPU run
contract. It is not yet a full-shard or full-panel mechanism result.
