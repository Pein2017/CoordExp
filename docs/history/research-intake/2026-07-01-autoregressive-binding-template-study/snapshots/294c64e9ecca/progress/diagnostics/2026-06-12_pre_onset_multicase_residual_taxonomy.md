# Pre-Onset Multicase Residual Taxonomy

Date: 2026-06-12

## Scope

This slice broadens the matched-pair residual-site result into a small
pre-onset taxonomy. It starts from the existing onset precursor panel and
selects exact coordinate rows where a focused residual patch should be
informative:

- phase: `post_y1/pre_x2`;
- rank-band candidates: target rank `2-10`;
- probability-drift candidates: target already rank `1`, but duplicate-basin
  masking shifts target probability by at least `0.01`;
- patch sites: `mlp,post_attention_residual`;
- patch layers: `16,20,24,27`;
- patch directions: `masked_to_control,control_to_masked`.

This is not a full validation set. It is a bounded causal probe over selected
pre-onset rows.

## Implementation

New selector:

```text
src/analysis/autoregressive_duplication_mechanism/phase4_pre_onset_patch_selector.py
```

New CLI:

```text
scripts/analysis/run_autoregressive_duplication_phase4_pre_onset_patch_selector.py
```

Test:

```text
tests/analysis/autoregressive_duplication_mechanism/test_phase4_pre_onset_patch_selector.py
```

The selector emits:

- `pre_onset_patch_target_manifest.json`
- `pre_onset_patch_selected_rows.jsonl`
- `pre_onset_patch_token_windows.jsonl`
- `pre_onset_patch_region_rows.jsonl`
- `pre_onset_patch_residual_command.sh`
- `pre_onset_patch_selector_summary.json`

The filtered token-window and region files are important: they make the target
manifest directly runnable by the existing residual patch CLI, and prevent the
matched-pair `--target-top-k` failure mode from recurring silently.

## Selector Run

```bash
PYTHONDONTWRITEBYTECODE=1 \
python scripts/analysis/run_autoregressive_duplication_phase4_pre_onset_patch_selector.py \
  --coord-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/phase1_onset_precursor_panel/onset_precursor_coord_rows.jsonl \
  --masking-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/shards/shard-00-of-04/phase3_masking/masking_delta_rows.jsonl \
  --masking-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/shards/shard-01-of-04/phase3_masking/masking_delta_rows.jsonl \
  --masking-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/shards/shard-02-of-04/phase3_masking/masking_delta_rows.jsonl \
  --masking-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/shards/shard-03-of-04/phase3_masking/masking_delta_rows.jsonl \
  --token-window-source-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/shards/shard-00-of-04/token_windows.jsonl \
  --token-window-source-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/shards/shard-01-of-04/token_windows.jsonl \
  --token-window-source-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/shards/shard-02-of-04/token_windows.jsonl \
  --token-window-source-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/shards/shard-03-of-04/token_windows.jsonl \
  --region-rows-source-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/phase2_region_rows.jsonl \
  --output-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_multicase_v1 \
  --max-targets 8 \
  --rank-band 2,10 \
  --probability-drift-threshold 0.01 \
  --phases post_y1/pre_x2 \
  --patch-layers 16,20,24,27 \
  --patch-sites mlp,post_attention_residual \
  --patch-directions masked_to_control,control_to_masked \
  --target-layer 20
```

Selector output root:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_multicase_v1
```

Counts:

- selected targets: `8`;
- selection kinds: `top1_probability_drift=5`, `rank_band_precursor=3`;
- filtered token-window rows: `8`;
- filtered region rows: `49`;
- unique replay cases: `7`, because record `37` contributes two selected rows.

## Residual Patch Run

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 \
python scripts/analysis/run_autoregressive_duplication_phase4_residual_patch_shard.py \
  --token-windows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_multicase_v1/pre_onset_patch_token_windows.jsonl \
  --region-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_multicase_v1/pre_onset_patch_region_rows.jsonl \
  --target-manifest-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_multicase_v1/pre_onset_patch_target_manifest.json \
  --output-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_multicase_v1/residual_patch_selected_cases \
  --device auto \
  --torch-dtype bfloat16 \
  --attn-implementation auto \
  --patch-layers 16,20,24,27 \
  --patch-sites mlp,post_attention_residual \
  --patch-directions masked_to_control,control_to_masked \
  --top-k 8 \
  --target-top-k 8
```

Residual patch output:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_multicase_v1/residual_patch_selected_cases
```

Counts:

- checkpoint count: `3`;
- source replay cases: `7`;
- targeted replay cases: `7`;
- residual patch rows: `128`;
- selected target rows covered: `8`, with `16` patch rows each.

## Taxonomy

Classification rule used for this note:

- **rank-moving repair:** best patch improves target rank from masked replay by
  at least `2`;
- **top-1 stable calibration:** control and masked replay are both rank `1`, and
  masking shifts target probability by at least `0.01`;
- **patch-damage fragile basin:** no rank-moving repair, but some patch worsens
  the control target rank;
- **no-effect hard/stable:** none of the above.

| class | checkpoint | record | desc | target | selection | control -> masked | best repair | worst damage/read |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| `top1_stable_calibration` | `no_aligner_parent_ckpt3668` | `88` | `cow` | `999` | prob drift | `1/0.396549 -> 1/0.483701` | MTC `mlp L16`, rank `1`, prob `0.501549` | CTM `mlp L16` lowers prob to `0.345653`, rank stable |
| `top1_stable_calibration` | `aligner_parent_ckpt1824` | `37` | `bowl` | `999` | prob drift | `1/0.178303 -> 1/0.194282` | MTC `mlp L27`, rank `1`, prob `0.202988` | CTM `mlp L20` lowers prob to `0.176771`, rank stable |
| `rank_moving_repair` | `aux_latest_ckpt32` | `79` | `bottle` | `579` | prob drift | `1/0.028354 -> 70/0.005647` | CTM `mlp L16`, rank `1`, prob `0.024092` | MTC `mlp L16` remains damaged, rank `62` |
| `patch_damage_fragile_basin` | `no_aligner_parent_ckpt3668` | `48` | `person` | `33` | prob drift | `1/0.071768 -> 1/0.063316` | CTM `mlp L24`, rank `1`, prob `0.073206` | MTC `mlp L16` damages rank to `2`, top1 `40` |
| `top1_stable_calibration` | `aux_latest_ckpt32` | `50` | `person` | `395` | prob drift | `1/0.028724 -> 1/0.044507` | MTC `mlp L16`, rank `1`, prob `0.044231` | CTM `mlp L16` lowers prob to `0.029246`, rank stable |
| `rank_moving_repair` | `no_aligner_parent_ckpt3668` | `36` | `person` | `407` | rank band | `2/0.047594 -> 31/0.010959` | CTM `mlp L16`, rank `2`, prob `0.047545` | MTC `mlp L16` remains damaged, rank `29` |
| `patch_damage_fragile_basin` | `aligner_parent_ckpt1824` | `37` | `carrot` | `513` | rank band | `1/0.032102 -> 2/0.007163` | CTM `mlp L20`, rank `1`, prob `0.028231` | MTC `mlp L16` damages rank to `5` |
| `patch_damage_fragile_basin` | `no_aligner_parent_ckpt3668` | `50` | `cell phone` | `178` | rank band | `2/0.066171 -> 3/0.060744` | CTM `mlp L20`, rank `2`, prob `0.069339` | MTC `mlp L20` keeps rank `3`, top1 stable |

Direction abbreviations:

- `MTC`: `masked_to_control`;
- `CTM`: `control_to_masked`.

## Read

The broadened panel supports a sharper split than the matched pair alone:

1. **Rank-moving repair exists and is not rare in selected windows.** The
   `aux_latest_ckpt32` bottle case and no-aligner person case have large masked
   rank damage that can be reversed by early MLP/post-attention insertion.
2. **Top-1 coordinate basins can still be meaningfully unstable.** Three cases
   stay rank `1` but show sizable probability movement under masking and under
   residual patches. These are calibration or basin-depth changes, not hidden
   target recovery.
3. **Some high-confidence or near-correct basins are fragile to the wrong
   residual direction.** The no-aligner person target `33`, aligner carrot
   target `513`, and no-aligner cell-phone target `178` expose patch-damage
   behavior even when best repair is small.
4. **The strongest effects again live in MLP/post-attention sites.** In this
   hook surface, `mlp` and `post_attention_residual` rows are identical, so the
   evidence should be read as MLP-side residual state, not as mathematical proof
   that pre/post-MLP tensors are identical.

Mechanistically, this supports the current working hypothesis: duplication
onset is preceded by local coordinate-basin instability, but the instability has
subfamilies. Some windows have a recoverable residual direction; some are
already top-1 but basin-depth-sensitive; some are fragile to injecting the wrong
state. This is deeper than "attention to duplicate region is high" and more
consistent with an autoregressive residual-basin composition story.

## Next Step

The attractive next path is to compare these selected rows against generated
history features:

- repeated desc/component before onset;
- repeated nearby coordinate anchors;
- target slot type and edge/saturation targets such as `<|coord_999|>`;
- whether the damaging direction is consistently `masked_to_control` or
  `control_to_masked` for each subfamily.

That can likely be done post-hoc from token windows and residual rows before
launching more GPU work.

## Verification

Code checks:

```bash
python - <<'PY'
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import tempfile
path=Path('tests/analysis/autoregressive_duplication_mechanism/test_phase4_pre_onset_patch_selector.py')
spec=spec_from_file_location('pre_onset_patch_selector_test', path)
assert spec and spec.loader
mod=module_from_spec(spec)
spec.loader.exec_module(mod)
mod.test_build_pre_onset_patch_targets_selects_rank_band_and_probability_drift()
with tempfile.TemporaryDirectory() as tmp:
    mod.test_materialize_pre_onset_patch_selector_writes_manifest_rows_and_command(Path(tmp))
print('direct_pre_onset_patch_selector_tests_passed=2')
PY

python -m pytest tests/analysis/autoregressive_duplication_mechanism/test_phase4_pre_onset_patch_selector.py -q

python -m py_compile \
  src/analysis/autoregressive_duplication_mechanism/phase4_pre_onset_patch_selector.py \
  scripts/analysis/run_autoregressive_duplication_phase4_pre_onset_patch_selector.py \
  tests/analysis/autoregressive_duplication_mechanism/test_phase4_pre_onset_patch_selector.py
```

The repo-local pytest wrapper printed its usual compact `No tests collected`
label but exited `0`; direct invocation ran both test functions and passed.

Artifact checks:

```bash
python - <<'PY'
from pathlib import Path
roots=[
Path('/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_multicase_v1'),
Path('/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_patch_selector_multicase_v1/residual_patch_selected_cases'),
]
required=[
(roots[0], 'pre_onset_patch_target_manifest.json'),
(roots[0], 'pre_onset_patch_selected_rows.jsonl'),
(roots[0], 'pre_onset_patch_token_windows.jsonl'),
(roots[0], 'pre_onset_patch_region_rows.jsonl'),
(roots[0], 'pre_onset_patch_residual_command.sh'),
(roots[1], 'residual_patch_rows.jsonl'),
(roots[1], 'phase4_residual_patch_summary.json'),
]
for root, name in required:
    path=root/name
    assert path.is_file() and path.stat().st_size > 0, path
    print(path.name, path.stat().st_size)
print('pre-onset multicase selector and residual patch artifacts exist')
PY
```
