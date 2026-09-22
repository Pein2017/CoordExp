# Sorted root-position diagnostic

This is a descriptive, CPU-only root-context readout. Post-root rows and sorted due index are excluded from every association because position and due order are confounded.

## prospective_2299

- Images: `2299`
- Owners: `46`
- Frozen root support (lower bound): `4/46` (`0.087`)
- Frozen root support (upper bound): `5/46` (`0.109`)
- Native outcomes: `{"native_fn": 27, "native_tp": 19}`
- Native matching universe: `27/46` native FNs (`0.587`); outside universe: `0`
- Support dispositions: `{"native_true_positive_transfer_control": 19, "withheld_calibration_nontransfer": 27}`

### Quadrants

| Quadrant | Owners | Lower-bound root support | Native FN / matching universe | Outside universe |
| --- | ---: | ---: | ---: | ---: |
| bottom_left | 12 | 0/12 (0.000) | 7/12 (0.583) | 0 |
| bottom_right | 11 | 0/11 (0.000) | 8/11 (0.727) | 0 |
| top_left | 11 | 3/11 (0.273) | 5/11 (0.455) | 0 |
| top_right | 12 | 1/12 (0.083) | 7/12 (0.583) | 0 |

### Image-stratified associations

Spearman coefficients are computed separately within each image against frozen lower-bound root support; the table robustly summarizes those image-level coefficients.

| Feature | Defined images | Q1 rho | Median rho | Q3 rho |
| --- | ---: | ---: | ---: | ---: |
| center_x_normalized | 1 | -0.128 | -0.128 | -0.128 |
| center_y_normalized | 1 | -0.323 | -0.323 | -0.323 |
| area_normalized | 1 | 0.134 | 0.134 | 0.134 |
| same_category_competitor_count | 1 | 0.142 | 0.142 | 0.142 |
| manhattan_distance_to_visual_block_end_normalized | 1 | 0.285 | 0.285 | 0.285 |
| diagonal_distance_to_visual_block_end_normalized | 1 | 0.209 | 0.209 | 0.209 |
| raster_token_distance_to_visual_block_end_normalized | 1 | 0.378 | 0.378 | 0.378 |

### Image-stratified native-FN associations

This second screen is independent of frozen root support. Native FN is defined only inside the native matching universe; outside-universe owners are excluded and reported in the quadrant table.

| Feature | Defined images | Q1 rho | Median rho | Q3 rho |
| --- | ---: | ---: | ---: | ---: |
| center_x_normalized | 1 | 0.156 | 0.156 | 0.156 |
| center_y_normalized | 1 | 0.095 | 0.095 | 0.095 |
| area_normalized | 1 | -0.221 | -0.221 | -0.221 |
| same_category_competitor_count | 1 | -0.035 | -0.035 | -0.035 |
| manhattan_distance_to_visual_block_end_normalized | 1 | -0.175 | -0.175 | -0.175 |
| diagonal_distance_to_visual_block_end_normalized | 1 | -0.171 | -0.171 | -0.171 |
| raster_token_distance_to_visual_block_end_normalized | 1 | -0.088 | -0.088 | -0.088 |

## legacy_12

- Images: `1584, 2685, 4134, 5001, 6040, 7511, 10707, 13348, 13923, 14038, 14439, 16228`
- Owners: `346`
- Frozen root support (lower bound): `49/346` (`0.142`)
- Frozen root support (upper bound): `55/346` (`0.159`)
- Native outcomes: `{"native_fn": 205, "native_tp": 141}`
- Native matching universe: `202/343` native FNs (`0.589`); outside universe: `3`
- Support dispositions: `{"native_true_positive_calibration_control": 141, "persistent_no_tested_localization_support": 72, "resolved_tested_localization_support": 114, "unresolved_ambiguity_bound_disposition_flip": 16, "unresolved_owner_outside_the_native_matching_universe": 3}`

### Quadrants

| Quadrant | Owners | Lower-bound root support | Native FN / matching universe | Outside universe |
| --- | ---: | ---: | ---: | ---: |
| bottom_left | 102 | 14/102 (0.137) | 61/102 (0.598) | 0 |
| bottom_right | 96 | 13/96 (0.135) | 54/93 (0.581) | 3 |
| top_left | 73 | 10/73 (0.137) | 41/73 (0.562) | 0 |
| top_right | 75 | 12/75 (0.160) | 46/75 (0.613) | 0 |

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

### Image-stratified native-FN associations

This second screen is independent of frozen root support. Native FN is defined only inside the native matching universe; outside-universe owners are excluded and reported in the quadrant table.

| Feature | Defined images | Q1 rho | Median rho | Q3 rho |
| --- | ---: | ---: | ---: | ---: |
| center_x_normalized | 12 | -0.101 | -0.017 | 0.194 |
| center_y_normalized | 12 | -0.258 | -0.162 | -0.020 |
| area_normalized | 12 | -0.568 | -0.529 | -0.396 |
| same_category_competitor_count | 12 | 0.002 | 0.105 | 0.237 |
| manhattan_distance_to_visual_block_end_normalized | 12 | -0.035 | -0.009 | 0.108 |
| diagonal_distance_to_visual_block_end_normalized | 12 | -0.016 | 0.049 | 0.100 |
| raster_token_distance_to_visual_block_end_normalized | 12 | 0.044 | 0.156 | 0.238 |

## Interpretation boundary

descriptive root-context association only; no causal, attention, RoPE, routing-mechanism, architecture, loss, or population claim. Raw cross-image log probabilities are not emitted or compared. Frozen-root-support and native-FN associations are separate, within-image screens; the native-FN screen excludes owners outside the native matching universe and uses no post-root t or sorted due index.
