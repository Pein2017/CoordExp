# X2 Overcorrection Probe Panel

Date: 2026-06-11

## Scope

Created a probe-ready mini panel from the x2 overcorrection candidates selected
from the phase-4 coordinate basin band-flow rows. This is a deterministic
post-hoc packaging step only; no GPU model probes were launched.

The panel is intended to be the launch surface for hidden-state probing and
attention mechanism analysis of the auxiliary-loss coordinate-slot attraction
failure mode.

## Inputs

- Candidate rows:
  `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_candidates_whole_head/x2_overcorrection_candidate_rows.jsonl`
- Source token windows:
  `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/target_token_windows.jsonl`
- Source region rows:
  `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/target_region_rows.jsonl`
- Source cross-phase rows:
  `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/target_cross_phase_rows.jsonl`

## Output

Output root:

`/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_probe_panel_whole_head`

Files:

- `candidate_probe_rows.jsonl`
- `candidate_token_windows.jsonl`
- `candidate_region_rows.jsonl`
- `candidate_cross_phase_rows.jsonl`
- `phase4_candidate_probe_panel_summary.json`
- `phase4_candidate_probe_panel_report.md`

## Counts

- Candidates: `7`
- Cases: `2`
- Token windows: `7`
- Region rows: `14`
- Cross-phase rows: `2`
- Missing token candidates: `0`
- Candidate kinds:
  - `x2_overcorrection_masked_exact`: `2`
  - `x2_overcorrection_masked_near`: `5`
- Checkpoint split:
  - `aux_latest_ckpt32`: `6`
  - `no_aligner_parent_ckpt3668`: `1`

## Top Rows

The two strongest deterministic rows are both in `aux_latest_ckpt32`, record 33,
phase `post_y1/pre_x2`, and have masked top1 exactly on the target before the
whole-head patch pushes x2 below target while reducing wide target-near mass.

1. `aux_latest_ckpt32_record33_row21_whole_head_x2_overcorrection`
   - generated-target: `+173`
   - masked-target: `0`
   - patched-target: `-12`
   - wide target-near mass delta: `-0.183812`
   - score: `309.542`
2. `aux_latest_ckpt32_record33_row20_whole_head_x2_overcorrection`
   - generated-target: `+174`
   - masked-target: `0`
   - patched-target: `-7`
   - wide target-near mass delta: `-0.0791955`
   - score: `194.935`

## Interpretation

This panel isolates a sharper auxiliary-checkpoint failure surface than the
broader coordinate basin reports: the language-side masked state can already be
at the correct x2 coordinate, while the whole-head patch overcorrects into a
lower-coordinate attractor and loses target-near probability mass.

This does not by itself prove the origin of the attractor. It gives a compact
set of deterministic windows for the next hidden-state and attention work:

- compare masked vs patched hidden-state trajectories at the x2 slot;
- decompose whether query, key, or residual state movement predicts the lower
  control basin;
- test whether attention sinks into the lower-control/duplicate basin before
  the patched x2 token;
- probe whether the coordinate-token embedding basin amplifies this movement.

Available GPU budget for future probes in this thread is limited to devices
`0,1,2,3`.

## Reproduction

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/analysis/run_autoregressive_duplication_phase4_candidate_probe_panel.py \
  --candidate-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_candidates_whole_head/x2_overcorrection_candidate_rows.jsonl \
  --token-windows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/target_token_windows.jsonl \
  --region-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/target_region_rows.jsonl \
  --cross-phase-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/target_cross_phase_rows.jsonl \
  --output-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_probe_panel_whole_head
```

Verification:

```bash
python - <<'PY'
import importlib.util
path='tests/analysis/autoregressive_duplication_mechanism/test_phase4_candidate_probe_panel.py'
spec=importlib.util.spec_from_file_location('t', path)
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.test_build_candidate_probe_panel_filters_token_rows_and_keeps_case_regions()
print('direct harness passed')
PY

python -m py_compile \
  src/analysis/autoregressive_duplication_mechanism/phase4_candidate_probe_panel.py \
  scripts/analysis/run_autoregressive_duplication_phase4_candidate_probe_panel.py \
  tests/analysis/autoregressive_duplication_mechanism/test_phase4_candidate_probe_panel.py
```
