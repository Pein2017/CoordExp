# X2 Overcorrection Hidden-Shift Probe

Date: 2026-06-11

## Scope

Launched the first GPU hidden-state probe from the validated x2 overcorrection
probe panel. This is a targeted four-forward run over two replay cases, not a
broad sweep.

GPU use was restricted to `CUDA_VISIBLE_DEVICES=0`, within the available device
set `0,1,2,3`.

## Inputs

- Candidate token windows:
  `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_probe_panel_whole_head/candidate_token_windows.jsonl`
- Candidate region rows:
  `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_probe_panel_whole_head/candidate_region_rows.jsonl`
- Candidate target manifest:
  `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_probe_panel_whole_head/phase4_candidate_probe_target_manifest.json`

The candidate target manifest pins route head `layer=17`, `head=1` for all
seven x2 overcorrection candidates and preserves exact `row_idx` so hidden
readouts can be linked back to candidate rows.

## Outputs

Hidden-shift output root:

`/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_hidden_shift_whole_head_layer17_head1`

Files:

- `hidden_shift_rows.jsonl`
- `phase4_hidden_shift_summary.json`

Counts:

- Source replay cases: `2`
- Targeted replay cases: `2`
- Intervention plan rows: `4`
- Hidden-shift rows: `63`
- Checkpoints loaded: `2`

Candidate row-link sanity check:

- `aux_latest_ckpt32` record 33 row 18 -> `aux_latest_ckpt32_record33_row18_whole_head_x2_overcorrection`
- `aux_latest_ckpt32` record 33 row 20 -> `aux_latest_ckpt32_record33_row20_whole_head_x2_overcorrection`
- `aux_latest_ckpt32` record 33 row 21 -> `aux_latest_ckpt32_record33_row21_whole_head_x2_overcorrection`
- `aux_latest_ckpt32` record 33 row 23 -> `aux_latest_ckpt32_record33_row23_whole_head_x2_overcorrection`
- `aux_latest_ckpt32` record 33 row 26 -> `aux_latest_ckpt32_record33_row26_whole_head_x2_overcorrection`
- `aux_latest_ckpt32` record 33 row 28 -> `aux_latest_ckpt32_record33_row28_whole_head_x2_overcorrection`
- `no_aligner_parent_ckpt3668` record 36 row 2 -> `no_aligner_parent_ckpt3668_record36_row2_whole_head_x2_overcorrection`

## First Readout

Mean hidden delta from duplicate-basin image masking grows strongly in later
layers for the auxiliary checkpoint:

| checkpoint | layer | n | mean hidden delta L2 | mean abs delta |
|---|---:|---:|---:|---:|
| `aux_latest_ckpt32` | 0 | 6 | 0.000000 | 0.000000 |
| `aux_latest_ckpt32` | 16 | 6 | 35.547380 | 0.589477 |
| `aux_latest_ckpt32` | 20 | 6 | 97.246207 | 1.667284 |
| `aux_latest_ckpt32` | 24 | 6 | 232.254289 | 4.048598 |
| `aux_latest_ckpt32` | 28 | 6 | 1092.048594 | 18.889872 |
| `no_aligner_parent_ckpt3668` | 0 | 1 | 0.000000 | 0.000000 |
| `no_aligner_parent_ckpt3668` | 16 | 1 | 5.729350 | 0.092887 |
| `no_aligner_parent_ckpt3668` | 20 | 1 | 17.380585 | 0.296702 |
| `no_aligner_parent_ckpt3668` | 24 | 1 | 48.442524 | 0.742012 |
| `no_aligner_parent_ckpt3668` | 28 | 1 | 307.081024 | 5.263050 |

This is not yet causal sufficiency. It says the duplicate-basin visual
perturbation produces much larger late residual-state movement in the auxiliary
x2 overcorrection case than in the one no-aligner parent control candidate. The
next useful step is to connect these hidden shifts to attention route movement
and coordinate-basin logits, especially around layers 16-20 where the route head
`17/1` sits.

## Reproduction

Target manifest:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/analysis/run_autoregressive_duplication_phase4_candidate_target_manifest.py \
  --candidate-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_probe_panel_whole_head/candidate_probe_rows.jsonl \
  --output-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_probe_panel_whole_head \
  --target-layer 17 \
  --target-head 1 \
  --target-kind x2_overcorrection_route_head
```

Hidden shift:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 python scripts/analysis/run_autoregressive_duplication_phase4_hidden_shift_shard.py \
  --token-windows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_probe_panel_whole_head/candidate_token_windows.jsonl \
  --region-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_probe_panel_whole_head/candidate_region_rows.jsonl \
  --target-manifest-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_probe_panel_whole_head/phase4_candidate_probe_target_manifest.json \
  --output-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_hidden_shift_whole_head_layer17_head1 \
  --device auto \
  --torch-dtype bfloat16 \
  --attn-implementation auto \
  --prompt-ordering random \
  --top-k 7
```

Verification:

```bash
python -m py_compile \
  src/analysis/autoregressive_duplication_mechanism/phase4_targets.py \
  src/analysis/autoregressive_duplication_mechanism/phase4_hidden_shift.py \
  scripts/analysis/run_autoregressive_duplication_phase4_candidate_target_manifest.py
```

The direct harness for `test_phase4_targets.py` and `test_phase4_hidden_shift.py`
also passed, including the row-aware target enrichment regression.
