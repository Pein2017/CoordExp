---
title: Phase 4 Grouped Attention Patch Findings
date: 2026-06-10
status: active-evidence-note
owner: codex
evidence_scope: phase4-layer16-grouped-attention-top4-allshards
---

# Phase 4 Grouped Attention Patch Findings

## Scope

This note records the first grouped layer-16 attention-output patch for the
autoregressive duplication mechanism study. It tests whether the weak
single-head repair result was simply because head `13` needed nearby duplicate
region heads patched together.

Input manifest root:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920
```

Main artifacts:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_attention_patch_layer16_multigroup_top4_allshards_v2_summary.json
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_attention_patch_layer16_multigroup_top4_allshards_v2_report.md
```

Probe settings:

- eight shard jobs launched across GPUs `0..7`;
- `target_top_k=4`;
- single-head baseline `patch_heads=13`;
- grouped heads `12,13`; `8,12,13`; `8,9,12,13`;
  `6,7,8,9,12,13`;
- patch directions `masked_to_control,control_to_masked`;
- `top_k=8`.

Aggregate counts:

- `summary_count=8`;
- `attention_patch_row_count=480`;
- `targeted_replay_case_count=2`;
- `checkpoint_count_sum=2`.

Invalid artifact warning:

- Prefix `phase4_attention_patch_layer16_multigroup_top4_allshards` should not
  be interpreted. The launch command accidentally used Bash's special
  `GROUPS` variable name, so the CLI received `0` and ran a head-0 control
  condition instead of the intended grouped heads.
- Prefix `phase4_attention_patch_layer16_multigroup_top4_allshards_v2` is the
  corrected run. Its shard summaries record
  `patch_head_groups=[[12,13],[8,12,13],[8,9,12,13],[6,7,8,9,12,13]]`.

## Findings

The grouped-head result does not rescue the attention-output repair path. At
the anchor window `none_latest_ckpt32|post_y1/pre_x2`, all grouped layer-16
attention patches remain far below the full decoder-layer and MLP repair
signals.

Anchor attention repair results:

- `head=13`: `prob_repair=0.000132`, `rank_repair=-0.167`.
- `heads=12,13`: `prob_repair=-0.000278`, `rank_repair=-0.167`.
- `heads=8,12,13`: `prob_repair=0.000197`, `rank_repair=0.500`.
- `heads=8,9,12,13`: `prob_repair=-0.000864`, `rank_repair=-2.250`.
- `heads=6,7,8,9,12,13`: `prob_repair=-0.000472`,
  `rank_repair=-0.667`.

Comparable module-site repair results from the top-4 module-site panel:

- `decoder_layer|layer=16`: `prob_repair=0.009302`,
  `rank_repair=18.833`.
- `site=mlp|layer=20`: `prob_repair=0.006170`,
  `rank_repair=10.917`.
- `site=mlp|layer=24`: `prob_repair=0.008964`,
  `rank_repair=14.583`.
- `site=self_attn|layer=16`: `prob_repair=-0.000913`,
  `rank_repair=-3.583`.

Grouped attention can still inject or perturb in some directions. For example,
`none_latest_ckpt32|post_y1/pre_x2|control_to_masked|heads=12,13` has
`prob_recovery=0.014166`, but this statistic is not a repair of the masked
path and remains weaker for causal basin restoration than the full residual or
MLP repair surfaces.

## Mechanism Update

This result strengthens the attention-to-MLP transformation picture:

1. The weak single-head repair was not merely a missing-neighbor-head problem.
2. Layer-16 attention-output groups can perturb the coordinate-slot state, but
   they do not reconstruct the healthier masked-to-control residual state at
   the anchor window.
3. The coordinate-slot basin repair is much more visible at the full residual
   and MLP surfaces, especially layers `20/24`.
4. The next best probe is not a wider attention-head atlas by itself; it should
   test whether layer-16 attention information is transformed by MLP/downstream
   residual updates.

## Next Probes

Recommended deterministic branch:

- add representation-transfer or cross-site patching from layer-16 attention
  output groups into later MLP/residual contexts;
- test MLP input vs MLP output at layers `20/24` if the model hooks expose
  those sites cleanly;
- keep `none_latest_ckpt32|post_y1/pre_x2` as the main anchor because it shows
  the cleanest split between grouped attention failure and MLP/full-layer
  repair.

Guardrail:

- This is an all-shard top-4 grouped-head panel, not a full attention atlas.
  It is sufficient to deprioritize "just patch more layer-16 heads together"
  as the main explanation, but not sufficient to rule out attention as an
  upstream routing component.
