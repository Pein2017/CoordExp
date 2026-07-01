# Pre-Onset Residual Patch Comparison

## Scope

This note extends the first pre-onset residual patch probe beyond the pilot `no_aligner_parent_ckpt3668` record `33` case. The goal is not to prove that the duplicate-basin visual region is always harmful or useful. The goal is to separate subfamilies:

- harmful attractor cases, where masking the duplicate-basin region improves the next coordinate token and control-state patching reinstalls the harm;
- useful support cases, where masking removes visual evidence needed for the correct coordinate token and control-state patching repairs the damage;
- weak or neutral cases, where the same intervention only shifts probability inside an already unstable coordinate basin.

Manifest root:

`/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133`

Shared probe settings:

- phase: `post_y1/pre_x2`
- region rows: `phase2_region_rows.jsonl`
- patch layers: `20,24,26,27`
- patch sites: `mlp,decoder_layer`
- patch directions: `masked_to_control,control_to_masked`
- target top-k: `1`
- reported top-k: `8`
- dtype: `bfloat16`

## Artifact Roots

Pilot:

`/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_causal_patch_record33_no_aligner/residual_patch_layers20_24_26_27_mlp_decoder`

Additional comparison cases:

- `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_causal_patch_comparison/no_aligner_record36_skis_offsetm1/residual_patch_layers20_24_26_27_mlp_decoder`
- `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_causal_patch_comparison/no_aligner_record48_bus_offsetm1/residual_patch_layers20_24_26_27_mlp_decoder`
- `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_causal_patch_comparison/none_latest_record33_bottle_offsetm2/residual_patch_layers20_24_26_27_mlp_decoder`

Each comparison run produced `16` residual patch rows: `4` layers x `2` sites x `2` directions.

## Case Table

`mask sign` compares the duplicate-basin masked run against the normal control image at the target next coordinate token. Lower rank and higher probability are better for the target. `masked_to_control` patches normal-image residual state into the masked run. `control_to_masked` patches masked-image residual state into the normal run.

| case | ckpt | rec/row/off | desc | target | control rank/prob | masked rank/prob | mask sign | best masked_to_control decoder | best control_to_masked mlp |
| --- | --- | --- | --- | ---: | --- | --- | --- | --- | --- |
| pilot_no_aligner_r33_wine_offsetm2 | no_aligner_parent_ckpt3668 | 33/2/-2 | wine glass | 145 | 5/0.032961 | 2/0.036959 | mask helps | L24 rank 5 prob 0.033273 d_rank +0 rec -3 | L27 rank 2 prob 0.037936 d_prob +0.004975 rec +0.000977 |
| no_aligner_r36_skis_offsetm1 | no_aligner_parent_ckpt3668 | 36/2/-1 | skis | 448 | 2/0.038703 | 2/0.038927 | mask helps | L20 rank 3 prob 0.038155 d_rank +1 rec -1 | L24 rank 2 prob 0.040047 d_prob +0.001344 rec +0.001120 |
| no_aligner_r48_bus_offsetm1 | no_aligner_parent_ckpt3668 | 48/2/-1 | bus | 899 | 1/0.061880 | 2/0.061519 | mask harms | L20 rank 1 prob 0.063190 d_rank +0 rec +1 | L26 rank 1 prob 0.062938 d_prob +0.001058 rec +0.001419 |
| none_latest_r33_bottle_offsetm2 | none_latest_ckpt32 | 33/21/-2 | bottle | 209 | 27/0.014387 | 31/0.013195 | mask harms | L24 rank 27 prob 0.014200 d_rank +0 rec +4 | L27 rank 31 prob 0.013370 d_prob -0.001017 rec +0.000174 |

## Mechanism Read

The causal sign is mixed, which is the useful result.

The pilot `wine glass` case remains the clearest harmful-attractor example. Masking the duplicate-basin region improves the target coordinate from rank `5` to rank `2`. Patching control decoder-layer residual state into the masked run returns the target to rank `5` across late layers. This supports a local coordinate-basin story: visual state from the duplicate basin can install a harmful residual direction before visible duplicate onset.

The `skis` case is a weaker version of the same sign. Masking keeps rank `2` but improves probability slightly. Patching control decoder-layer state at layer `20` worsens the masked run to rank `3`. This looks like a low-margin attractor effect rather than a dramatic onset precursor.

The `bus` and `bottle` cases have the opposite sign. Masking the duplicate-basin region harms the target, and `masked_to_control` decoder-layer patching repairs the masked run toward the control condition. For these rows, the duplicate-basin region is not merely a spurious repeat attractor; it is also carrying useful visual support for the coordinate slot.

So the population-level mechanism should not be framed as "duplicate basin is bad." A better current hypothesis is:

1. Pre-onset coordinate slots can be close to competing coordinate basins.
2. Duplicate-basin visual state can push the next coordinate token in either direction.
3. The sign depends on whether the region state aligns with the current row's intended object/box or with a repeated spatial anchor from the emerging burst.
4. Whole decoder-layer residual state at layers `20-27` is sufficient to carry the sign in the tested cases; MLP-only patching is weaker and more layer-specific.

## Consequence For The Roadmap

The next high-leverage step is stratified selection, not broader averaging. We should build or use a small selector that labels pre-onset windows by:

- mask helps vs mask harms;
- target rank margin and probability margin;
- repeated spatial anchor overlap with prior rows;
- semantic row relation to nearby duplicate-burst rows;
- coordinate-bin basin shift under region masking.

Then run deeper probes on one harmful-attractor case and one useful-support case with matched phase and slot. The useful pair for the next slice is:

- harmful: `no_aligner_parent_ckpt3668`, record `33`, row `2`, `wine glass`, offset `-2`;
- useful: `no_aligner_parent_ckpt3668`, record `48`, row `2`, `bus`, offset `-1`, or `none_latest_ckpt32`, record `33`, row `21`, `bottle`, offset `-2`.

This comparison should keep attention/logit/residual reads separated. If we average across signs too early, we will erase the actual mechanism split.

## Verification

Checked artifact presence for the three added comparison cases:

- `pre_onset_patch_target_manifest.json`
- `residual_patch_layers20_24_26_27_mlp_decoder/residual_patch_rows.jsonl`
- `residual_patch_layers20_24_26_27_mlp_decoder/phase4_residual_patch_summary.json`

All three comparison roots contain non-empty residual rows and summary files. The pilot root was previously recorded in `progress/diagnostics/2026-06-12_pre_onset_residual_patch_record33.md`.
