# FN Coordinate Direction Case Panel

## Scope

This panel extends the vase direction decomposition beyond one case. It compares the source-to-target direction signature for several known FN guidance cases from the existing parent-checkpoint panel:

- source condition: `prefix=0, tier=desc_x1`
- target condition: `prefix=all, tier=desc_x1`
- checkpoint: `/data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/compact_full_fullobj_random_sft_bsz16_4epoch_tokenrows_v2/compact-full-fullobj-random-sft-bsz16-4epoch-tokenrows-v2/v1-20260601-062428/checkpoint-3668`
- layers: decoder layers `26,27`
- sites: `mlp,self_attn`
- prompt/readout tiers: `desc_x1,desc_x1_y1,desc_x1_y1_x2`
- prefix limits: `0,all`

The goal is to check whether the vase lift-then-suppress signature is a general recoverable-FN mechanism or a special prefix-locked case.

## Artifact Roots

- `vase`: `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_coordslot_logit_probe_parent_val128/no_aligner_parent_ckpt3668_vase_prefix_coordslot_direction_patch_sites_l24_27`
- `microwave`: `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_coordslot_logit_probe_parent_val128/no_aligner_parent_ckpt3668_direction_case_panel_l26_27_mlp_attn/image139_gt8_microwave`
- `chair_gt11`: `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_coordslot_logit_probe_parent_val128/no_aligner_parent_ckpt3668_direction_case_panel_l26_27_mlp_attn/image139_gt11_chair`
- `chair_gt15`: `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_coordslot_logit_probe_parent_val128/no_aligner_parent_ckpt3668_direction_case_panel_l26_27_mlp_attn/image139_gt15_chair`
- `book`: `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_coordslot_logit_probe_parent_val128/no_aligner_parent_ckpt3668_direction_case_panel_l26_27_mlp_attn/image139_gt17_book`

All roots contain:

- `fn_coordslot_logit_condition_rows.jsonl`
- `fn_coordslot_logit_layer_rows.jsonl`
- `fn_coordslot_hidden_delta_rows.jsonl`
- `fn_coordslot_direction_rows.jsonl`
- `fn_coordslot_residual_patch_rows.jsonl`
- `phase4_fn_coordslot_logit_probe_summary.json`
- `phase4_fn_coordslot_logit_probe_report.md`

## Case Summary

Baseline refers to the target condition `prefix=all, tier=desc_x1`. Direction values are coordinate-logit deltas for `source - target` at the selected site.

| case | guidance outcome | target next bin | baseline rank/top1 | best MLP patch | MLP26 target/wrong/margin | MLP27 target/wrong/margin | read |
| --- | --- | ---: | --- | --- | --- | --- | --- |
| `vase` | prefix-locked, y1 unlocks | `468` | `52/501` | `L27 rank 12 rec 40 top1 479` | `+1.250/+0.016/+1.234` | `+0.250/-1.656/+1.906` | L26 target lift, L27 wrong-basin suppression |
| `microwave` | clean desc+x1 rescue | `483` | `18/467` | `L26 rank 18 rec 0 top1 472` | `+0.031/+0.094/-0.062` | `-0.156/-0.125/-0.031` | weak/mixed |
| `chair_gt11` | x1 and wrong-control rescue | `512` | `3/519` | `L27 rank 1 rec 2 top1 513` | `-0.125/+0.031/-0.156` | `+1.594/+1.406/+0.188` | shallow target lift from already-good rank |
| `chair_gt15` | x1 and wrong-control rescue | `524` | `6/519` | `L27 rank 6 rec 0 top1 520` | `-0.250/-0.250/+0.000` | `-0.438/-0.625/+0.188` | weak/mixed |
| `book` | hard no-rescue | `716` | `59/682` | `L26 rank 52 rec 7 top1 687` | `+0.047/-0.062/+0.109` | `+0.281/+0.219/+0.062` | weak, no basin repair |

## Site Detail

| case | site | layer | patch recovery | patched rank | target delta | wrong/top1 delta | margin change |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `vase` | `mlp` | 26 | 33 | 19 | +1.250 | +0.016 | +1.234 |
| `vase` | `mlp` | 27 | 40 | 12 | +0.250 | -1.656 | +1.906 |
| `vase` | `self_attn` | 26 | 0 | 52 | +0.219 | +0.281 | -0.062 |
| `vase` | `self_attn` | 27 | 14 | 38 | +1.195 | +0.031 | +1.164 |
| `microwave` | `mlp` | 26 | 0 | 18 | +0.031 | +0.094 | -0.062 |
| `microwave` | `mlp` | 27 | -1 | 19 | -0.156 | -0.125 | -0.031 |
| `microwave` | `self_attn` | 26 | 2 | 16 | +0.625 | +0.625 | +0.000 |
| `microwave` | `self_attn` | 27 | 2 | 16 | -0.188 | -0.250 | +0.062 |
| `chair_gt11` | `mlp` | 26 | 0 | 3 | -0.125 | +0.031 | -0.156 |
| `chair_gt11` | `mlp` | 27 | 2 | 1 | +1.594 | +1.406 | +0.188 |
| `chair_gt11` | `self_attn` | 26 | 1 | 2 | -0.062 | -0.094 | +0.031 |
| `chair_gt11` | `self_attn` | 27 | 2 | 1 | +0.547 | +0.453 | +0.094 |
| `chair_gt15` | `mlp` | 26 | -1 | 7 | -0.250 | -0.250 | +0.000 |
| `chair_gt15` | `mlp` | 27 | 0 | 6 | -0.438 | -0.625 | +0.188 |
| `chair_gt15` | `self_attn` | 26 | 0 | 6 | -0.031 | +0.016 | -0.047 |
| `chair_gt15` | `self_attn` | 27 | 2 | 4 | -0.266 | -0.453 | +0.188 |
| `book` | `mlp` | 26 | 7 | 52 | +0.047 | -0.062 | +0.109 |
| `book` | `mlp` | 27 | 7 | 52 | +0.281 | +0.219 | +0.062 |
| `book` | `self_attn` | 26 | 7 | 52 | -0.141 | -0.203 | +0.062 |
| `book` | `self_attn` | 27 | 2 | 57 | -0.469 | -0.375 | -0.094 |

## Interpretation

The vase case remains qualitatively distinct. It is the only case in this small panel with:

1. a bad full-prefix baseline rank (`52`) plus a wrong vertical top1 basin (`501`),
2. large MLP patch recovery (`52 -> 12`), and
3. a two-step MLP direction signature: layer 26 target-local lift followed by layer 27 wrong-basin suppression.

The hard `book` case is an important contrast. It starts with a similarly poor target rank (`59`) but the same source-to-target patch only recovers to rank `52`, with weak target/wrong margin changes at both MLP layers. This argues against a generic "empty-prefix residual vector fixes all FNs" account. The book failure is more consistent with visual/extent/semantic evidence not being recoverable from the same language-side prefix guidance.

The chair cases are not strong prefix-lock cases. Their full-prefix baselines are already near the correct bin (`rank 3` and `rank 6`), so small patch recoveries should be read as basin refinement, not hidden-object recovery. The microwave case has moderate baseline uncertainty (`rank 18`) but weak source-to-target MLP direction, suggesting its successful guidance decode may happen through later generation dynamics or non-MLP/other-slot effects rather than this exact y1-basin mechanism.

## Current Mechanism Split

This small panel supports splitting FNs into at least three mechanistic subfamilies:

- **Prefix-basin lock:** visible object and correct x1 exist, but full-prefix state installs a wrong coordinate basin. Vase is the clearest example. Late MLP state can causally reshape the basin.
- **Shallow basin refinement:** full-prefix target is already high-rank; guidance or patch mostly nudges an already-plausible local basin. Chair cases fit this better than deep blindness.
- **Hard residual FN:** target remains poor and source-to-target state transfer gives little margin repair. Book fits this group and should not be explained away as ordinary prefix fragility.

## Next Step

The attractive next path is to connect this FN split back to duplication-onset windows. Specifically, reuse the same direction/logit-margin vocabulary around duplication precursor windows: do repeated local spatial anchors correspond to a growing wrong-basin margin before visible duplicate emission?

## Verification

The four new case probes ran successfully on GPUs `0,1,2,3`. Each emitted `condition_row_count=6`, `layer_row_count=174`, `hidden_delta_row_count=58`, `direction_row_count=4`, and `patch_row_count=4`.

Artifact existence check:

```bash
python - <<'PY'
from pathlib import Path
roots=[
Path('/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_coordslot_logit_probe_parent_val128/no_aligner_parent_ckpt3668_direction_case_panel_l26_27_mlp_attn/image139_gt8_microwave'),
Path('/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_coordslot_logit_probe_parent_val128/no_aligner_parent_ckpt3668_direction_case_panel_l26_27_mlp_attn/image139_gt11_chair'),
Path('/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_coordslot_logit_probe_parent_val128/no_aligner_parent_ckpt3668_direction_case_panel_l26_27_mlp_attn/image139_gt15_chair'),
Path('/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_coordslot_logit_probe_parent_val128/no_aligner_parent_ckpt3668_direction_case_panel_l26_27_mlp_attn/image139_gt17_book'),
]
required=[
'fn_coordslot_logit_condition_rows.jsonl',
'fn_coordslot_logit_layer_rows.jsonl',
'fn_coordslot_hidden_delta_rows.jsonl',
'fn_coordslot_direction_rows.jsonl',
'fn_coordslot_residual_patch_rows.jsonl',
'phase4_fn_coordslot_logit_probe_summary.json',
'phase4_fn_coordslot_logit_probe_report.md',
]
for root in roots:
    for name in required:
        path=root/name
        assert path.is_file() and path.stat().st_size > 0, path
print('case panel artifacts exist')
PY
```
