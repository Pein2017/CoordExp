# Phase 4 Coord-Basin Boundary Amplification Findings

Date: 2026-06-11

## Scope

This note extends the Phase 4 residual patch probes with coord-slice distribution
telemetry, so residual-boundary causal patches can be compared directly against
the layer-17 head-1 duplicate-basin value-source patch.

The question is not whether duplication exists. The question is where the
masked-image coordinate basin is restored, amplified, or attenuated after the
duplicate-basin visual value contribution enters the autoregressive stream.

## Code change

Residual patch rows now include the same coordinate distribution delta fields as
the value-source patch rows:

- `control_coord_mass_radius_{r}`, `masked_coord_mass_radius_{r}`,
  `patched_coord_mass_radius_{r}`
- `coord_mass_radius_{r}_recovery_from_masked`
- `coord_expected_abs_error_recovery_from_masked`
- `coord_top1_distance_recovery_from_masked`
- `coord_entropy_recovery_from_masked`
- control/masked/patched coord top bins and probabilities

The shared row helper lives at:

```text
src/analysis/autoregressive_duplication_mechanism/phase4_coord_distribution.py
```

## New artifacts

Manifest root:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920
```

Layer-17 module-boundary rerun:

```text
phase4_layer17_module_boundary_coord_distribution_top4_allshards_report.md
phase4_layer17_module_boundary_coord_distribution_top4_allshards_summary.json
phase4_layer17_module_boundary_coord_distribution_top4_allshards_shard-*/residual_patch_rows.jsonl
```

Counts:

- `residual_patch_row_count`: 768
- nonempty shards: `shard-03-of-08`, `shard-04-of-08`
- nonempty shard rows: 384 each

Decoder input/output layer sweep rerun:

```text
phase4_decoder_input_output_layers_14_15_16_17_18_coord_distribution_top4_allshards_report.md
phase4_decoder_input_output_layers_14_15_16_17_18_coord_distribution_top4_allshards_summary.json
phase4_decoder_input_output_layers_14_15_16_17_18_coord_distribution_top4_allshards_shard-*/residual_patch_rows.jsonl
```

Counts:

- `residual_patch_row_count`: 960
- nonempty shards: `shard-03-of-08`, `shard-04-of-08`
- nonempty shard rows: 480 each

Both reruns preserve the same protocol as the previous residual probes:

- `top_k=8`
- `target_top_k=4`
- `patch_directions=masked_to_control,control_to_masked`
- same `phase4_probe_target_manifest.json`
- same `phase2_region_rows.jsonl`

## Primary boundary read

Primary slice:

- checkpoint: `none_latest_ckpt32`
- record: 33
- phase: `post_y1/pre_x2`
- direction: `masked_to_control`
- rows per site/layer: 12

### Layer-17 module sites

| Site | Target prob recovery | Rank recovery | Radius-4 mass recovery | Radius-8 mass recovery | Expected abs-error recovery | Top-1 distance recovery | Entropy recovery |
|---|---:|---:|---:|---:|---:|---:|---:|
| `self_attn_input` | 0.007815 | 19.833333 | 0.051862 | 0.093308 | -3.718605 | -8.833333 | 0.093795 |
| `decoder_layer_input` | 0.010274 | 21.916667 | 0.067451 | 0.121950 | -5.032284 | -11.583333 | 0.061889 |
| `self_attn` | 0.015438 | 22.833333 | 0.107745 | 0.190230 | -8.581084 | -12.833333 | -0.152146 |
| `decoder_layer` | 0.014757 | 21.166667 | 0.103071 | 0.183724 | -8.890916 | -12.916667 | -0.214654 |
| `mlp_input` | 0.010906 | 18.666667 | 0.074699 | 0.134115 | -6.524414 | -9.333333 | -0.083248 |
| `mlp` | 0.010906 | 18.666667 | 0.074699 | 0.134115 | -6.524414 | -9.333333 | -0.083248 |

Interpretation:

- The layer-17 self-attention output is where the coord-basin recovery becomes
  sharply stronger than the layer input.
- The full layer-17 decoder output keeps nearly all of the self-attention
  coord-basin recovery.
- The MLP-side sites do not amplify beyond the self-attention output in this
  slice; they look more like partial carry-through or attenuation.
- Entropy becomes lower after `self_attn` and `decoder_layer`, which means the
  repair is not only moving expected coordinate mass closer. It is also making
  the coord-slice distribution more concentrated.

### Decoder input/output layer sweep

| Layer/site | Target prob recovery | Rank recovery | Radius-4 mass recovery | Radius-8 mass recovery | Expected abs-error recovery | Top-1 distance recovery | Entropy recovery |
|---|---:|---:|---:|---:|---:|---:|---:|
| `14/decoder_layer_input` | 0.009225 | 18.500000 | 0.062387 | 0.112532 | -5.260328 | -9.166667 | -0.016745 |
| `14/decoder_layer` | 0.008552 | 17.833333 | 0.056808 | 0.104077 | -4.961705 | -6.916667 | -0.006301 |
| `15/decoder_layer_input` | 0.008552 | 17.833333 | 0.056808 | 0.104077 | -4.961705 | -6.916667 | -0.006301 |
| `15/decoder_layer` | 0.011091 | 20.166667 | 0.077354 | 0.140336 | -6.652592 | -9.083333 | -0.061556 |
| `16/decoder_layer_input` | 0.011091 | 20.166667 | 0.077354 | 0.140336 | -6.652592 | -9.083333 | -0.061556 |
| `16/decoder_layer` | 0.010274 | 21.916667 | 0.067451 | 0.121950 | -5.032284 | -11.583333 | 0.061889 |
| `17/decoder_layer_input` | 0.010274 | 21.916667 | 0.067451 | 0.121950 | -5.032284 | -11.583333 | 0.061889 |
| `17/decoder_layer` | 0.014757 | 21.166667 | 0.103071 | 0.183724 | -8.890916 | -12.916667 | -0.214654 |
| `18/decoder_layer_input` | 0.014757 | 21.166667 | 0.103071 | 0.183724 | -8.890916 | -12.916667 | -0.214654 |
| `18/decoder_layer` | 0.013712 | 19.916667 | 0.094625 | 0.169801 | -8.567530 | -10.500000 | -0.212572 |

Interpretation:

- The strongest boundary jump in this sweep is at layer 17 output.
- Layer 18 input exactly inherits the layer 17 output state, as expected from
  residual-stream continuity.
- Layer 18 output slightly attenuates the basin repair by target probability,
  radius mass, rank, and top-1 distance, but keeps most of the concentrated
  near-target basin.
- Earlier layers already carry a recoverable basin shift, but it is weaker and
  less concentrated than the layer-17 output.

## Comparison to the layer-17 head-1 value-source patch

For the same primary slice, the duplicate-basin value-source patch from the
previous artifact is:

```text
phase4_value_source_patch_coord_distribution_layer17_head1_top4_allshards_report.md
```

Primary `none_latest_ckpt32`, record 33, `post_y1/pre_x2`, `masked_to_control`,
layer 17, head 1, duplicate-basin source:

| Patch | Target prob recovery | Rank recovery | Radius-4 mass recovery | Radius-8 mass recovery | Expected abs-error recovery | Top-1 distance recovery | Entropy recovery |
|---|---:|---:|---:|---:|---:|---:|---:|
| head-1 duplicate-basin value source | 0.019106 | 20.916667 | 0.130624 | 0.227583 | -10.066869 | -10.916667 | -0.322221 |
| layer-17 `self_attn` output residual | 0.015438 | 22.833333 | 0.107745 | 0.190230 | -8.581084 | -12.833333 | -0.152146 |
| layer-17 full decoder output residual | 0.014757 | 21.166667 | 0.103071 | 0.183724 | -8.890916 | -12.916667 | -0.214654 |

Read:

- The value-source patch is larger than the residual boundary patch in target
  probability and local radius mass.
- The layer-17 self-attention output preserves most, but not all, of that local
  coord-basin repair.
- This supports a route of:

```text
duplicate-basin visual value contribution
  -> layer-17 self-attention output
  -> layer-17 / layer-18 residual stream coord-basin state
```

The residual stream does not look like the origin of the strongest repair. It
looks like the carrier after attention injects a near-target coordinate basin.

## Auxiliary checkpoint contrast

For `aux_latest_ckpt32`, record 33, same `post_y1/pre_x2` slice and direction:

| Boundary | Target prob recovery | Radius-4 mass recovery | Expected abs-error recovery |
|---|---:|---:|---:|
| layer-17 `decoder_layer_input` | -0.001311 | -0.011875 | 0.860994 |
| layer-17 `self_attn` | 0.003393 | 0.026312 | -2.109238 |
| layer-17 `decoder_layer` | 0.003842 | 0.025064 | -2.513067 |
| layer-18 `decoder_layer` | 0.002215 | 0.014109 | -1.945980 |

This contrast is useful. The same boundary locations remain readable, but the
basin shift is far weaker under the auxiliary-loss checkpoint. That separates
the location question from the magnitude question: layer-17 attention remains
the plausible handoff location, while checkpoint mechanics determine how much
duplicate-basin coordinate attraction is available to carry.

## Current conclusion

The deepest current mechanism branch is no longer a coarse residual-layer
question. The best-supported path is:

1. Visual duplicate-basin source tokens carry a local coordinate-basin value
   contribution through layer-17 head 1.
2. The contribution is visible at the layer-17 self-attention output as a
   concentrated near-target coord-slice shift.
3. The residual stream carries that shift into layer 18 with mild attenuation.
4. The MLP side does not appear to be the main amplifier for this primary
   slice.

The next deterministic step should therefore inspect attention composition
inside layer 17 head 1 more deeply, especially how duplicate-basin source
attention and value projection interact with coord-token output directions.

## Verification

Commands run:

```bash
python -m py_compile \
  src/analysis/autoregressive_duplication_mechanism/phase4_coord_distribution.py \
  src/analysis/autoregressive_duplication_mechanism/phase4_residual_patch.py \
  src/analysis/autoregressive_duplication_mechanism/phase4_value_source.py \
  scripts/analysis/run_autoregressive_duplication_phase4_residual_patch_shard.py
```

```bash
python -m pytest \
  tests/analysis/autoregressive_duplication_mechanism/test_phase4_residual_patch.py \
  tests/analysis/autoregressive_duplication_mechanism/test_phase4_value_source.py \
  -q
```

The local pytest wrapper returned `Pytest: No tests collected`, matching the
known collection issue in this worktree. Focused direct importlib test harnesses
were run instead:

- `tests/analysis/autoregressive_duplication_mechanism/test_phase4_residual_patch.py`
  ran 17 tests.
- `tests/analysis/autoregressive_duplication_mechanism/test_phase4_value_source.py`
  ran 9 tests with a `tmp_path` and `monkeypatch` fixture shim.

Artifact checks:

- layer-17 boundary rerun: 768 rows, expected nonempty shards only.
- layer sweep rerun: 960 rows, expected nonempty shards only.
- row schema includes `coord_mass_radius_*`, `coord_expected_abs_error_*`,
  `coord_entropy_*`, and coord top-bin fields.
