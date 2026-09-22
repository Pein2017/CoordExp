# Sorted root-position diagnostic

This is a descriptive, CPU-only root-context readout. Post-root rows and sorted due index are excluded from every association because position and due order are confounded.

## legacy_12

- Images: `1584, 2685, 4134, 5001, 6040, 7511, 10707, 13348, 13923, 14038, 14439, 16228`
- Owners: `346`
- Frozen root support (lower bound): `49/346` (`0.142`)
- Frozen root support (upper bound): `55/346` (`0.159`)
- Native outcomes: `{"native_fn": 205, "native_tp": 141}`
- Support dispositions: `{"native_true_positive_calibration_control": 141, "persistent_no_tested_localization_support": 72, "resolved_tested_localization_support": 114, "unresolved_ambiguity_bound_disposition_flip": 16, "unresolved_owner_outside_the_native_matching_universe": 3}`

### Quadrants

| Quadrant | Owners | Lower-bound root support |
| --- | ---: | ---: |
| bottom_left | 102 | 14/102 (0.137) |
| bottom_right | 96 | 13/96 (0.135) |
| top_left | 73 | 10/73 (0.137) |
| top_right | 75 | 12/75 (0.160) |

### Image-stratified associations

Spearman coefficients are computed separately within each image against frozen lower-bound root support; the table robustly summarizes those image-level coefficients.

| Feature | Defined images | Q1 rho | Median rho | Q3 rho |
| --- | ---: | ---: | ---: | ---: |
| center_x_normalized | 12 | -0.182 | 0.012 | 0.213 |
| center_y_normalized | 12 | -0.317 | -0.103 | 0.048 |
| area_normalized | 12 | 0.158 | 0.189 | 0.329 |
| same_category_competitor_count | 12 | -0.323 | -0.245 | -0.126 |
| manhattan_distance_to_visual_block_end_normalized | 12 | -0.167 | 0.010 | 0.247 |
| diagonal_distance_to_visual_block_end_normalized | 12 | -0.196 | 0.003 | 0.191 |
| raster_token_distance_to_visual_block_end_normalized | 12 | -0.068 | 0.091 | 0.173 |

## Interpretation boundary

descriptive root-context association only; no causal, attention, RoPE, routing-mechanism, architecture, loss, or population claim. Raw cross-image log probabilities are not emitted or compared; associations are computed within image from frozen root support.
