# Phase 4 Basin Fine-Spatial QK Sign-Flip Probe

Date: 2026-06-11

Scope: four fine-spatial sign-changing candidates selected from the fine-spatial route/content probe. This is a targeted mechanistic follow-up, not a validation-scale estimate.

## Inputs

- Fine-spatial promotion root: `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_basin_fine_spatial_promotions_layer17_head1_reversals`
- Route/content summary joined into this probe: `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_basin_fine_spatial_promotions_layer17_head1_reversals/fine_spatial_route_content_summary/fine_spatial_route_content_rows.jsonl`
- QK sign-flip summary: `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_basin_fine_spatial_promotions_layer17_head1_reversals/fine_spatial_qk_signflip_summary`
- GPU scope: devices `0,1,2,3`; all four one-row QK probes completed with failure count 0.

Candidates:

- `record114_target7_component_row_7_handbag__fine_right_band`
- `record114_target7_component_row_7_handbag__fine_expanded_1p50`
- `record50_target9_component_row_9_person__fine_expanded_1p25`
- `record50_target9_component_row_9_person__fine_expanded_1p50`

Each candidate was probed at layer 17, head 1 with QK source components `duplicate_basin`, `visual_near_ring`, and `visual_far_background`.

## Key Table

| candidate | route total prob | duplicate attn delta | duplicate score delta | duplicate key/control-query | duplicate query/masked-keys | duplicate lse delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `record114_target7_component_row_7_handbag__fine_right_band` | -0.002841210 | -0.084757 | -0.518173 | -1.216080 | 0.697906 | 0.044075 |
| `record114_target7_component_row_7_handbag__fine_expanded_1p50` | 0.002482514 | 0.235476 | 4.996083 | 1.443911 | 3.552173 | 10.615189 |
| `record50_target9_component_row_9_person__fine_expanded_1p25` | -0.002293150 | 0.817075 | 7.635741 | 3.643070 | 3.992671 | 13.300469 |
| `record50_target9_component_row_9_person__fine_expanded_1p50` | 0.007959660 | 0.273189 | 5.664747 | 2.759049 | 2.905698 | 12.094870 |

## Mechanistic Read

The QK follow-up splits the fine-spatial sign flips into at least two regimes.

For `record114_target7_component_row_7_handbag`, the negative right-band route/content effect aligns with a negative duplicate-basin QK route signal:

- right band route/content total recovery: `-0.002841210`
- duplicate-basin attention delta: `-0.084757`
- duplicate-basin mean score delta: `-0.518173`
- key/control-query term: `-1.216080`

The positive wider expansion in the same case flips all the duplicate-basin QK terms positive:

- expanded-1.50 route/content total recovery: `0.002482514`
- duplicate-basin attention delta: `0.235476`
- duplicate-basin mean score delta: `4.996083`
- key/control-query term: `1.443911`
- query/masked-keys term: `3.552173`

Interpretation: for the handbag case, the fine-spatial sign flip is consistent with a genuine route-origin flip. The right-side pocket is not merely selected and then harmful; it is also less selected by the control-side QK route.

For `record50_target9_component_row_9_person`, the smaller expanded-1.25 box is downstream-negative despite very strong positive duplicate-basin QK routing:

- expanded-1.25 route/content total recovery: `-0.002293150`
- duplicate-basin attention delta: `0.817075`
- duplicate-basin mean score delta: `7.635741`
- key/control-query term: `3.643070`
- query/masked-keys term: `3.992671`
- duplicate-basin logsumexp delta: `13.300469`

The wider expanded-1.50 box is downstream-positive with also-positive but weaker duplicate-basin QK routing:

- expanded-1.50 route/content total recovery: `0.007959660`
- duplicate-basin attention delta: `0.273189`
- duplicate-basin mean score delta: `5.664747`
- key/control-query term: `2.759049`
- query/masked-keys term: `2.905698`

Interpretation: for the person case, QK route strength is not sufficient for useful coordinate-slot repair. The model can strongly route to a fine visual basin while the downstream value/content contribution still pushes the wrong coordinate slot or damages the target coordinate distribution.

## Consequence For The Core Mechanism

The fine-spatial results now support a sharper decomposition:

1. Some destructive pockets are route-origin problems: QK routing itself flips sign or suppresses the useful basin.
2. Some destructive pockets are value/content problems: QK routing is strong, but the selected content is misaligned with the current coordinate slot.
3. Spatial scale matters because changing the region boundary can alter both which keys are selected and which values enter the output projection.

This directly strengthens the coordinate-slot basin/attraction picture. The duplicated coordinate token does not appear to be caused by a single missing-object perception failure or by uniform object-box duplication. It is better modeled as a structured interaction among region boundary, QK route, selected visual values, and the coordinate-token output basin.

## Recommended Next Step

Run a fine-spatial value-basin projection probe for the same four candidates, with special attention to:

- `record50_target9_component_row_9_person__fine_expanded_1p25`, where QK is strongly positive but route/content is negative;
- `record50_target9_component_row_9_person__fine_expanded_1p50`, the positive scale contrast;
- `record114_target7_component_row_7_handbag__fine_right_band`, the route-origin negative pocket;
- `record114_target7_component_row_7_handbag__fine_expanded_1p50`, the positive route-origin contrast.

If the value-basin projection separates the record-50 pair, that would make the final picture much cleaner: QK chooses a basin, but coordinate-slot usefulness is decided by whether the basin's value projection points toward the active coordinate-token attractor.
