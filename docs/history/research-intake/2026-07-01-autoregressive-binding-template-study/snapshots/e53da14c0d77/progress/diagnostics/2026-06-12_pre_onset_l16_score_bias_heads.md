# Pre-Onset L16 Score-Bias Head Probe

Date: 2026-06-12

## Scope

This slice follows `2026-06-12_pre_onset_l16_attention_source_routing.md`.
The attention-source readout identified:

- visual-source candidates: `L16H8`, `L16H13`;
- history-loop comparator: `L16H1`;
- bridge head: `L16H12`.

This run tests whether duplicate-basin score-bias patching on those heads moves
coordinate-token outcomes in the same direction as the broader L16 residual
boundary CTM/MTC effect.

This is still a causal proxy over attention scores, not a full value/content
patch.

## Runs

All four heads used the same panel:

```text
token windows: /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_balanced_rank_onset_v1/pre_onset_patch_token_windows.jsonl
region rows: /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_balanced_rank_onset_v1/pre_onset_patch_region_rows.jsonl
layer: 16
region-kind: duplicate_basin
patch-directions: masked_to_control,control_to_masked
patch-components: duplicate_basin
score-bias-modes: scalar_logsumexp,per_source_delta
```

Parallel launches:

```bash
CUDA_VISIBLE_DEVICES=0 ... --attention-head 8  --output-dir .../attention_score_bias_l16_head8
CUDA_VISIBLE_DEVICES=1 ... --attention-head 13 --output-dir .../attention_score_bias_l16_head13
CUDA_VISIBLE_DEVICES=2 ... --attention-head 1  --output-dir .../attention_score_bias_l16_head1
CUDA_VISIBLE_DEVICES=3 ... --attention-head 12 --output-dir .../attention_score_bias_l16_head12
```

Each run completed with:

```text
attention_score_bias_patch_row_count=52
checkpoint_count=3
replay_case_count=8
region_row_count=56
```

Artifact roots:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_balanced_rank_onset_v1/attention_score_bias_l16_head8
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_balanced_rank_onset_v1/attention_score_bias_l16_head13
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_balanced_rank_onset_v1/attention_score_bias_l16_head1
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_balanced_rank_onset_v1/attention_score_bias_l16_head12
```

## CTM/MTC Positive Summary

Rows below use `per_source_delta` and join back to the site-localization
taxonomy. The CTM/MTC join covered 8 targets with materialized score-bias rows;
one CTM/MTC target from the source-readout panel did not materialize a
duplicate-basin score-bias row.

`masked_to_control` direction:

| head | attention-mass recovery | prob recovery | rank recovery | mass16 recovery | abs-error recovery |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 0.2028 | 0.000033 | 34.62 | -0.0044 | -16.09 |
| 13 | 0.2321 | 0.000058 | -5.88 | -0.0020 | 3.04 |
| 1 | -0.0173 | 0.000172 | 3.25 | 0.0014 | -2.44 |
| 12 | 0.1623 | 0.000110 | 6.00 | 0.0028 | 5.11 |

`control_to_masked` direction:

| head | attention-mass damage | prob damage | rank damage | mass16 damage | abs-error damage |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | -0.0911 | -0.001957 | 2.38 | -0.0355 | 13.02 |
| 13 | -0.1527 | -0.000745 | 3.88 | -0.0101 | 0.40 |
| 1 | 0.0144 | 0.000004 | 0.88 | -0.0020 | 0.52 |
| 12 | -0.0856 | 0.001113 | 2.75 | 0.0195 | -2.21 |

## Case-Level Peaks

Top CTM/MTC `masked_to_control` probability recoveries:

| head | case | slot | prob recovery | rank recovery |
| ---: | --- | --- | ---: | ---: |
| 8 | `none_latest_ckpt32 record114 row5 person` | `y1` | 0.001788 | 102 |
| 8 | `none_latest_ckpt32 record114 row3 backpack` | `x1` | 0.001243 | 5 |
| 13 | `none_latest_ckpt32 record114 row3 backpack` | `x1` | 0.001302 | 6 |
| 13 | `none_latest_ckpt32 record114 row5 person` | `y1` | 0.000567 | 52 |
| 1 | `none_latest_ckpt32 record114 row3 backpack` | `x1` | 0.000990 | 4 |
| 12 | `aux_latest_ckpt32 record79 row9 bottle` | `x2` | 0.002577 | 31 |

## Read

This is the first place where the visual-source hypothesis weakens a little:

1. `L16H8` and `L16H13` do move duplicate-basin attention mass, matching the
   attention-source readout, but their mean coordinate-logit repair is small and
   inconsistent across CTM/MTC positives.
2. `L16H12` is not the top visual-mass head, but it has the cleanest positive
   `masked_to_control` mass16/probability recovery among the averaged CTM/MTC
   rows and the strongest bridge-case probability recovery on `x2 bottle`.
3. `L16H1` remains a plausible generated-history comparator. It has weak
   attention-mass movement but modest positive probability/mass16 repair, so it
   should not be discarded.
4. `control_to_masked` is not a simple mirror of `masked_to_control`. The
   negative mass/prob damage values for `H8/H13` suggest that score-biasing the
   duplicate-basin route alone is not equivalent to transplanting the full L16
   residual-boundary state.

Mechanism update:

```text
The L16 route is probably not a single visual head whose duplicate-basin score
movement is sufficient for the pre-onset coord repair. H8/H13 look like strong
visual routing sensors; H12 and H1 are better candidates for coord-logit
consequence. The next step should separate score routing from value/content:
run head value/content patching or Q/K origin checks for H8/H13/H12/H1, with
case-level emphasis on x2 bottle, y1 person, and the backpack exception.
```

## Caveats

- Only 8 CTM/MTC targets materialized duplicate-basin score-bias rows.
- The mean effects are small; case-level peaks are more informative than the
  aggregate at this stage.
- The probe manipulates score bias for one source component, not the whole head
  output or downstream MLP state.
- `scalar_logsumexp` and `per_source_delta` were both run, but this note reports
  `per_source_delta` to keep the comparison single-valued.
