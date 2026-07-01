# Phase 4 Layer 17 Head 1 Source-Bucket Direction Findings

## Question

The component decomposition showed that the non-region complement carries an
opposing coordinate direction before `o_proj`. This source-bucket probe asks
which non-region source subset supplies that opposition: other visual tokens,
post-visual text/generated prefix tokens, or pre-visual special/control tokens.

## Scope

- Artifact root: `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920`
- Full-run prefix: `phase4_projected_direction_source_buckets_layer17_head1_top4_allshards`
- Summary JSON: `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_projected_direction_source_buckets_layer17_head1_top4_allshards_summary.json`
- Markdown report: `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_projected_direction_source_buckets_layer17_head1_top4_allshards_report.md`
- Smoke prefix: `phase4_projected_direction_source_buckets_layer17_head1_primary_smoke`
- Shards: `8`
- Rows: `7536`
- Replay cases: `30`
- Target source: decoder layer `17`, attention head `1`

Bucket semantics:

- `duplicate_basin`: visual tokens inside the duplicate-basin region.
- `visual_non_basin`: visual-span keys excluding duplicate-basin keys.
- `text_prefix`: post-visual text/generated keys through the current query.
- `special_control`: remaining pre-visual/control keys.
- `non_region_complement`: all keys except duplicate-basin keys.
- `whole_head`: all keys.

The three non-duplicate source buckets add up to `non_region_complement`.

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

Mean source-bucket metrics across the 12 primary-window rows:

| Component | Count | Pre delta L2 | Post delta L2 | Pre target cos | Post target cos | Pre target projection | Post target projection | Post residual cos |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `duplicate_basin` | `1.0` | `43.063` | `73.508` | `0.211286` | `0.034142` | `212.408108` | `16.136784` | `0.106510` |
| `visual_non_basin` | `339.0` | `39.688` | `67.764` | `-0.059451` | `-0.008469` | `-118.689661` | `-9.004802` | `0.157225` |
| `text_prefix` | `484.0` | `0.752` | `1.007` | `-0.014865` | `-0.002968` | `-0.579180` | `-0.044032` | `0.103453` |
| `special_control` | `151.0` | `0.014` | `0.016` | `-0.038857` | `-0.006089` | `-0.016335` | `-0.001241` | `-0.010122` |
| `non_region_complement` | `974.0` | `39.682` | `67.743` | `-0.059968` | `-0.008849` | `-119.285157` | `-9.082249` | `0.159157` |
| `whole_head` | `975.0` | `20.835` | `34.045` | `0.161091` | `0.027696` | `93.122946` | `7.054535` | `0.540098` |

## Mechanism Update

Almost all of the negative non-region complement projection comes from
`visual_non_basin`. The text/generated prefix and special/control buckets have
negative signs, but their target-projection magnitudes are tiny compared with
the visual non-basin bucket.

This localizes the opposing direction to other visual tokens in the same
attention head. The current layer-17 head-1 picture is therefore a visual
competition inside the head:

- duplicate-basin visual tokens support the duplicated coordinate;
- non-basin visual tokens oppose that coordinate;
- the whole-head delta is the partially cancelled sum of those two visual
  directions;
- language/template and special/control tokens are not the main origin of the
  opposing coordinate direction in this anchor.

## Verification

- Source-bucket unit tests:
  `python - <<'PY' ... pytest.main(['-q','tests/analysis/autoregressive_duplication_mechanism/test_phase4_projected_direction.py']) ... PY`
  passed with `9 passed`.
- Real-model smoke:
  `phase4_projected_direction_source_buckets_layer17_head1_primary_smoke`
  completed with `72` rows.
- Full 8-GPU sweep:
  `phase4_projected_direction_source_buckets_layer17_head1_top4_allshards_shard-00-of-08`
  through `shard-07-of-08` completed with `7536` rows.

## Next Deterministic Step

Split `visual_non_basin` spatially:

1. near-ring visual tokens around the duplicate basin;
2. far/background visual tokens;
3. optionally same-object-row versus other-object visual neighborhoods when
   row/object geometry supports it.

This should separate local context competition from broad image/background
mass.

