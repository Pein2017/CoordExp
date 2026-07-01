# Duplication Onset Precursor Panel

## Scope

This slice pivots the FN coordinate-basin vocabulary back to autoregressive duplication. It reuses the existing selected-window manifest:

`/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133`

No model forward pass was run. The new reducer joins existing:

- `token_windows.jsonl`
- `forward_readouts/coord_logit_rows.jsonl`
- `attention_readouts/attention_region_rows.jsonl`
- `phase3_masking/masking_delta_rows.jsonl`

and writes onset-aligned precursor summaries.

## New Artifact Contract

Command:

```bash
python scripts/analysis/run_autoregressive_duplication_phase1_onset_precursor_panel.py \
  --manifest-root /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133 \
  --output-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/phase1_onset_precursor_panel
```

Output root:

`/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/phase1_onset_precursor_panel`

Files:

- `onset_precursor_coord_rows.jsonl`
- `onset_precursor_coord_summary_rows.jsonl`
- `onset_precursor_attention_summary_rows.jsonl`
- `onset_precursor_masking_summary_rows.jsonl`
- `phase1_onset_precursor_panel_summary.json`
- `phase1_onset_precursor_panel_report.md`

Counts:

- token rows after row-level collapse: `314`
- onset cases: `30`
- coord rows: `1256`
- coord summary rows: `192`
- attention summary rows: `384`
- masking summary rows: `192`

## Operational Definitions

- Onset alignment uses `relative_row_offset` against `primary_burst_onset_row`.
- A repeated anchor is a row that shares the onset row's `same_desc_component_id` or `spatial_basin_component_id`.
- In this selected-window artifact, all coord rows share the onset anchor component. This means the panel is best interpreted as an onset-centered temporal profile, not an anchor-vs-nonanchor contrast.

## Main Positive Signal: Coord-Basin Precursor Drift

The strongest pre-onset coordinate alignment appears in coordinate phases immediately before the duplicate/onset row:

| checkpoint | phase | offset | rows | target prob | target rank | mass r4 | rank1 frac |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `aligner_parent_ckpt1824` | `post_y1/pre_x2` | -1 | 5 | 0.182718 | 5.4000 | 0.328296 | 0.6000 |
| `aligner_parent_ckpt1824` | `box_start/pre_x1` | -1 | 5 | 0.103269 | 11.2000 | 0.225946 | 0.2000 |
| `aligner_parent_ckpt1824` | `post_y1/pre_x2` | -3 | 4 | 0.062238 | 15.7500 | 0.191607 | 0.5000 |
| `no_aligner_parent_ckpt3668` | `post_y1/pre_x2` | -2 | 10 | 0.061681 | 17.2000 | 0.208719 | 0.1000 |
| `no_aligner_parent_ckpt3668` | `post_x2/pre_y2` | -3 | 10 | 0.056678 | 17.1000 | 0.164144 | 0.2000 |
| `no_aligner_parent_ckpt3668` | `post_x1/pre_y1` | -1 | 11 | 0.048243 | 19.8182 | 0.255538 | 0.4545 |

The temporal view is especially useful:

- `aligner_parent_ckpt1824`, `post_y1/pre_x2`: offset `-3` prob/rank/r4 `0.0622/15.75/0.1916`, offset `-2` `0.0194/16.60/0.1441`, offset `-1` `0.1827/5.40/0.3283`, onset `0.0225/21.60/0.1663`.
- `no_aligner_parent_ckpt3668`, `post_y1/pre_x2`: offset `-3` `0.0368/8.80/0.2451`, offset `-2` `0.0617/17.20/0.2087`, offset `-1` `0.0331/20.27/0.2425`, onset `0.0363/10.82/0.2466`.
- `none_latest_ckpt32`, `post_y1/pre_x2`: offset `-3` `0.0066/54.00/0.0510`, offset `-2` `0.0251/15.83/0.1770`, offset `-1` `0.0213/7.17/0.1517`, onset `0.0333/2.50/0.2184`.

This supports a precursor interpretation: selected duplicate windows often have measurable coordinate-basin alignment before visible onset, especially in `post_y1/pre_x2` and nearby coordinate phases. The signal is not monotonic in every checkpoint, so it should be treated as onset instability or transient basin attraction rather than a smooth ramp.

## Masking Sensitivity

The strongest pre-onset mask-sensitive rows also concentrate around coordinate phases:

| checkpoint | phase | offset | rows | prob delta | rank delta | top1 changed |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `none_latest_ckpt32` | `post_x2/pre_y2` | -2 | 6 | -0.005671 | 70.8333 | 0.5000 |
| `none_latest_ckpt32` | `post_x1/pre_y1` | -2 | 6 | -0.005397 | 64.6667 | 0.3333 |
| `none_latest_ckpt32` | `box_start/pre_x1` | -2 | 6 | -0.005276 | 127.0000 | 0.3333 |
| `aligner_parent_ckpt1824` | `post_y1/pre_x2` | -2 | 5 | -0.005092 | -6.6000 | 0.4000 |
| `aux_latest_ckpt32` | `post_y1/pre_x2` | -3 | 8 | -0.005001 | 27.5000 | 0.6250 |
| `no_aligner_parent_ckpt3668` | `post_y1/pre_x2` | -3 | 10 | -0.004385 | 3.1000 | 0.5000 |

This is consistent with visual evidence participating before onset, but the rank-delta sign is mixed. It should guide targeted causal windows rather than serve as a final causal proof.

## Attention Negative/Caveat

After correcting the artifact region names, the pre-onset duplicate-basin attention summary does not show positive duplicate-minus-rest concentration. The largest pre-onset rows are still negative, for example:

- `no_aligner_parent_ckpt3668`, `post_x2/pre_y2`, offset `-2`: duplicate `0.000026`, rest `0.008967`, duplicate-minus-rest `-0.008941`.
- `no_aligner_parent_ckpt3668`, `post_x2/pre_y2`, offset `-3`: duplicate `0.001119`, rest `0.014285`, duplicate-minus-rest `-0.013166`.
- `aligner_parent_ckpt1824`, `post_x2/pre_y2`, offset `-2`: duplicate `0.004397`, rest `0.038510`, duplicate-minus-rest `-0.034113`.

So the strongest precursor signal in this reduction is not raw duplicate-region attention concentration. The more plausible read is:

1. coordinate-slot basin attraction is visible before onset,
2. visual masking can matter pre-onset in selected phases,
3. attention routing needs a more specific normalization or head/source decomposition before claiming duplicate-region concentration.

This is an important guardrail against overfitting the story to the earlier attention/routing hypothesis.

## Current Mechanism Update

The FN work showed prefix-state coordinate basin lock and late MLP basin repair in one recoverable FN. This duplication panel shows a related but not identical phenomenon: before visible duplicate emission, selected duplicate windows already show transient coordinate-basin alignment in autoregressive coordinate slots. The common substrate may be coordinate-token basin attraction under local prefix history, but the duplication case does not yet reduce to a simple attention concentration story.

## Next Step

The most attractive next probe is to pick a small set of high-precursor rows from this panel and run a causal patch/masking comparison at the precise pre-onset phase, especially:

- `aligner_parent_ckpt1824`, `post_y1/pre_x2`, offset `-1`
- `no_aligner_parent_ckpt3668`, `post_y1/pre_x2`, offset `-2`
- `none_latest_ckpt32`, `post_y1/pre_x2`, offset `-2` or onset `0`

The question should be: can we patch or suppress the coordinate-basin precursor before the duplicate row is emitted?

## Verification

```bash
python - <<'PY'
import importlib.util, tempfile
from pathlib import Path
path='tests/analysis/autoregressive_duplication_mechanism/test_phase1_onset_precursor_panel.py'
spec=importlib.util.spec_from_file_location('onset_tests', path)
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
for name in sorted(n for n in dir(mod) if n.startswith('test_')):
    with tempfile.TemporaryDirectory() as tmp:
        getattr(mod, name)(Path(tmp))
print('direct onset precursor harness passed')
PY

python -m py_compile \
  src/analysis/autoregressive_duplication_mechanism/phase1_onset_precursor_panel.py \
  scripts/analysis/run_autoregressive_duplication_phase1_onset_precursor_panel.py \
  tests/analysis/autoregressive_duplication_mechanism/test_phase1_onset_precursor_panel.py
```

Both checks passed before regenerating the real artifact.
