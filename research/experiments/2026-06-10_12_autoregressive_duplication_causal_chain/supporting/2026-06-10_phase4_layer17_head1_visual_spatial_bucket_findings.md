# Phase 4 Layer 17 Head 1 Visual-Spatial Bucket Findings

## Question

The source-bucket decomposition localized the opposing coordinate direction to
non-basin visual tokens. This probe asks whether that opposition comes from
nearby visual context around the duplicate basin or from broad far/background
visual mass.

## Scope

- Artifact root: `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920`
- Full-run prefix: `phase4_projected_direction_visual_spatial_buckets_layer17_head1_top4_allshards`
- Summary JSON: `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_projected_direction_visual_spatial_buckets_layer17_head1_top4_allshards_summary.json`
- Markdown report: `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_projected_direction_visual_spatial_buckets_layer17_head1_top4_allshards_report.md`
- Smoke prefix: `phase4_projected_direction_visual_spatial_buckets_layer17_head1_primary_smoke`
- Shards: `8`
- Rows: `10048`
- Replay cases: `30`
- Target source: decoder layer `17`, attention head `1`

Spatial split:

- `visual_near_ring`: non-duplicate visual tokens within Chebyshev radius `1`
  of duplicate-basin visual tokens on the merged visual-token grid.
- `visual_far_background`: remaining non-duplicate visual tokens.
- `visual_non_basin`: `visual_near_ring + visual_far_background`.

Implementation support was added in this slice:

- `src/analysis/autoregressive_duplication_mechanism/phase4_projected_direction.py`
- `scripts/analysis/run_autoregressive_duplication_phase4_projected_direction_shard.py`
- `scripts/analysis/run_autoregressive_duplication_phase4_projected_contribution_patch_shard.py`
- `tests/analysis/autoregressive_duplication_mechanism/test_phase4_projected_direction.py`

## Primary Anchor

Primary diagnostic anchor:

- checkpoint label: `none_latest_ckpt32`
- record: `33`
- role: `post_y1/pre_x2`

Mean direction metrics across the 12 primary-window rows:

| Component | Count | Pre delta L2 | Post delta L2 | Pre target projection | Post target projection | Post target cos | Post residual cos |
|---|---:|---:|---:|---:|---:|---:|---:|
| `duplicate_basin` | `1.0` | `43.063` | `73.508` | `212.408108` | `16.136784` | `0.034142` | `0.106510` |
| `visual_near_ring` | `8.0` | `38.159` | `65.306` | `-120.648186` | `-9.164838` | `-0.010813` | `-0.016893` |
| `visual_far_background` | `331.0` | `2.301` | `3.636` | `1.958537` | `0.145139` | `0.007726` | `0.277686` |
| `visual_non_basin` | `339.0` | `39.688` | `67.764` | `-118.689661` | `-9.004802` | `-0.008469` | `0.157225` |
| `text_prefix` | `484.0` | `0.752` | `1.007` | `-0.579180` | `-0.044032` | `-0.002968` | `0.103453` |
| `special_control` | `151.0` | `0.014` | `0.016` | `-0.016335` | `-0.001241` | `-0.006089` | `-0.010122` |
| `non_region_complement` | `974.0` | `39.682` | `67.743` | `-119.285157` | `-9.082249` | `-0.008849` | `0.159157` |
| `whole_head` | `975.0` | `20.835` | `34.045` | `93.122946` | `7.054535` | `0.027696` | `0.540098` |

## Mechanism Update

The opposing visual direction is local. At the primary anchor, only about
eight near-ring visual tokens carry almost the entire negative target
projection. The far/background visual bucket is much smaller and slightly
positive.

This sharpens the layer-17 head-1 picture again:

- duplicate-basin tokens supply the coordinate-supporting vector;
- immediately neighboring visual tokens supply the counter-vector;
- broad far/background visual mass, text prefix, and special/control tokens are
  not the main source of coordinate opposition;
- the whole-head output is a local visual competition balance around the
  duplicate basin.

## Verification

- Near/far partition unit tests:
  `python - <<'PY' ... pytest.main(['-q','tests/analysis/autoregressive_duplication_mechanism/test_phase4_projected_direction.py']) ... PY`
  passed with `10 passed`.
- Syntax and CLI checks:
  `python -m py_compile src/analysis/autoregressive_duplication_mechanism/phase4_projected_direction.py scripts/analysis/run_autoregressive_duplication_phase4_projected_direction_shard.py scripts/analysis/run_autoregressive_duplication_phase4_projected_contribution_patch_shard.py`
  passed, and both CLIs expose `visual_near_ring` and `visual_far_background`.
- Real-model smoke:
  `phase4_projected_direction_visual_spatial_buckets_layer17_head1_primary_smoke`
  completed with `96` rows.
- Full 8-GPU sweep:
  `phase4_projected_direction_visual_spatial_buckets_layer17_head1_top4_allshards_shard-00-of-08`
  through `shard-07-of-08` completed with `10048` rows.

## Next Deterministic Step

Test spatial specificity causally:

1. patch or ablate `visual_near_ring` at `self_attn_output`;
2. compare against `visual_far_background`;
3. verify whether the near-ring component is necessary/sufficient for the
   opposing local visual competition found here.

