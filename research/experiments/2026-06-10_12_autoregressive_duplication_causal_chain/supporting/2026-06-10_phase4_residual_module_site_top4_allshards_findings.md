---
title: Phase 4 Residual Module-Site Top-4 All-Shard Findings
date: 2026-06-10
status: active-evidence-note
owner: codex
evidence_scope: phase4-module-site-top4-allshards
---

# Phase 4 Residual Module-Site Top-4 All-Shard Findings

## Scope

This note records the first all-shard module-site residual patch comparison
for the autoregressive duplication mechanism study. It expands the earlier
single-shard module-site smoke to the top-4 target windows across all eight
token-window shards.

Input manifest root:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920
```

Main artifacts:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_module_patch_sites_layers_16_20_24_top4_allshards_summary.json
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_module_patch_sites_layers_16_20_24_top4_allshards_report.md
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_module_patch_sites_layers_16_20_24_top4_allshards_directionality.json
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_module_patch_sites_layers_16_20_24_top4_allshards_directionality.md
```

Probe settings:

- eight shard jobs launched across GPUs `0..7`;
- `target_top_k=4`;
- patch layers `16,20,24`;
- patch sites `decoder_layer,self_attn,mlp`;
- patch directions `masked_to_control,control_to_masked`;
- `top_k=8`.

Aggregate counts:

- `summary_count=8`;
- `residual_patch_row_count=864`;
- `targeted_replay_case_count=2`;
- `checkpoint_count_sum=2`;
- paired directionality rows `36`.

The attempted broader `target_top_k=24` run under prefix
`phase4_module_patch_sites_layers_16_20_24_top24_allshards` was stopped after
shards `03` and `04` remained CPU-active for more than ten minutes with no
rows or summary files. Completed partial shard outputs were left in place for
post-hoc inspection, but this note interprets only the completed top-4 panel.

## Findings

The top-4 panel preserves the main split from the single-shard smoke: full
decoder-layer residual patching remains the strongest causal readout, while
MLP output is the strongest submodule candidate for asymmetric repair at the
high-signal `none_latest_ckpt32|post_y1/pre_x2` window. Isolated `self_attn`
patching still does not look like the complete restoration path.

Strong full-layer symmetric candidates:

- `aux_latest_ckpt32|post_x1/pre_y1|site=decoder_layer|layer=24`:
  `prob_repair_from_masked=0.015628`,
  `prob_damage_into_control=0.015854`,
  `rank_repair_from_masked=3.083`,
  `rank_damage_into_control=4.000`.
- `aux_latest_ckpt32|post_x1/pre_y1|site=decoder_layer|layer=20`:
  `prob_repair_from_masked=0.014904`,
  `prob_damage_into_control=0.015518`,
  `rank_repair_from_masked=3.250`,
  `rank_damage_into_control=3.500`.
- `none_latest_ckpt32|post_y1/pre_x2|site=decoder_layer|layer=20`:
  `prob_repair_from_masked=0.013875`,
  `prob_damage_into_control=0.013179`,
  `rank_repair_from_masked=17.917`,
  `rank_damage_into_control=16.833`.
- `none_latest_ckpt32|post_y1/pre_x2|site=decoder_layer|layer=24`:
  `prob_repair_from_masked=0.013667`,
  `prob_damage_into_control=0.013004`,
  `rank_repair_from_masked=17.917`,
  `rank_damage_into_control=16.083`.

High-value asymmetric repair candidates:

- `none_latest_ckpt32|post_y1/pre_x2|site=decoder_layer|layer=16`:
  `prob_downstream_repair_score=0.009302`,
  `prob_repair_from_masked=0.009302`,
  `prob_damage_into_control=0.000000`,
  `rank_repair_from_masked=18.833`.
- `none_latest_ckpt32|post_y1/pre_x2|site=mlp|layer=24`:
  `prob_downstream_repair_score=0.008964`,
  `prob_repair_from_masked=0.008964`,
  `prob_damage_into_control=0.000000`,
  `rank_repair_from_masked=14.583`.
- `none_latest_ckpt32|post_y1/pre_x2|site=mlp|layer=20`:
  `prob_downstream_repair_score=0.006170`,
  `prob_repair_from_masked=0.006170`,
  `prob_damage_into_control=0.000000`,
  `rank_repair_from_masked=10.917`.

`self_attn` remains weak or counterproductive for the same repair target:

- `none_latest_ckpt32|post_y1/pre_x2|site=self_attn|layer=16`:
  `prob_repair_from_masked=-0.000913`,
  `rank_repair_from_masked=-3.583`.
- `none_latest_ckpt32|post_y1/pre_x2|site=self_attn|layer=20`:
  `prob_repair_from_masked=-0.000861`,
  `rank_repair_from_masked=-3.167`.
- `none_latest_ckpt32|post_y1/pre_x2|site=self_attn|layer=24`:
  `prob_repair_from_masked=-0.000225`,
  `rank_repair_from_masked=-0.583`.

## Mechanism Update

This result makes the current best picture more specific:

1. The full residual state at layers `20/24` is still the strongest repair and
   injection surface.
2. For `none_latest_ckpt32|post_y1/pre_x2`, the MLP output at layers `20/24`
   carries much more of the asymmetric repair signal than the attention output.
3. Attention can still be upstream or routing-relevant, but the coordinate-slot
   attraction basin does not look like a raw attention-output vector copied
   into the slot.
4. The next branch should test an attention-to-MLP transformation hypothesis:
   layer-16 attention heads may route the visual/coordinate perturbation, while
   MLP and downstream residual updates transform it into a coordinate-token
   basin.

This remains compatible with dynamic roadmap adjustment. The module-site
branch is now more promising than another coarse layer sweep, because it has a
clear path to separating entry, transformation, and basin/readout.

## Next Probes

Recommended deterministic branch:

- run a targeted multi-head layer-16 attention-output patch for the same top-4
  replay cases, not only head `13`;
- compare the multi-head patch against `mlp` layer `20/24` repair on the same
  rows;
- add a representation-transfer probe from layer-16 attention output into
  later MLP/residual sites if the multi-head patch is still weaker than the MLP
  site;
- keep `none_latest_ckpt32|post_y1/pre_x2` as the anchor window because it has
  the cleanest rank and probability repair split.

Guardrail:

- This is an all-shard top-4 panel, not the full top-24 module-site atlas. It
  is strong enough to prioritize the attention-to-MLP transformation branch,
  but final claims should wait for targeted multi-head and transfer probes.
