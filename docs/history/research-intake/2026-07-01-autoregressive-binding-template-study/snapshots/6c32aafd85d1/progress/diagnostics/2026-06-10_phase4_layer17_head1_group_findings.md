---
title: Phase 4 Layer 17 Head 1 Group Findings
date: 2026-06-10
status: active-evidence-note
owner: codex
evidence_scope: phase4-layer17-head1-groups-top4-allshards
---

# Phase 4 Layer 17 Head 1 Group Findings

## Scope

This note records the layer-17 head-1 grouped attention patch probe. It follows
the all-head singleton scan, which localized the primary
`none_latest_ckpt32|post_y1/pre_x2` effect mostly to layer-17 attention head
`1`.

Input manifest root:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920
```

Main artifacts:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_attention_patch_layer17_head1_groups_top4_allshards_v2_summary.json
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_attention_patch_layer17_head1_groups_top4_allshards_v2_report.md
```

Probe settings:

- eight shard jobs launched across GPUs `0..7`;
- `target_top_k=4`;
- patch layer override `17`;
- singleton patch head `1`;
- grouped heads `1,13`, `1,15`, `1,3`, `1,8`, `1,12`, `1,7`,
  and `1,3,8,13,15`;
- patch directions `masked_to_control,control_to_masked`;
- `top_k=8`.

Aggregate counts:

- `summary_count=8`;
- `attention_patch_row_count=768`;
- `targeted_replay_case_count=2`;
- `checkpoint_count_sum=2`.

Implementation note:

- A first attempted prefix without `_v2` was interrupted after empty-shard
  validation showed `patch_head_groups=[[0]]`. The cause was using Bash's
  special `GROUPS` variable name. The valid artifact prefix is the `_v2`
  prefix above.

## Primary Anchor

For `none_latest_ckpt32|post_y1/pre_x2`, grouped masked-to-control repair is:

| heads | repair | rank repair | patched target-prob delta |
| --- | ---: | ---: | ---: |
| `1,3` | `0.014843` | `22.583` | `0.000366` |
| `1,7` | `0.014688` | `22.083` | `0.000210` |
| `1,12` | `0.014413` | `21.583` | `-0.000065` |
| `1` | `0.013860` | `21.833` | `-0.000618` |
| `1,8` | `0.013836` | `21.333` | `-0.000642` |
| `1,15` | `0.013748` | `20.917` | `-0.000729` |
| `1,3,8,13,15` | `0.013698` | `20.917` | `-0.000779` |
| `1,13` | `0.012466` | `19.750` | `-0.002012` |

Paired control-to-masked target-prob drop magnitudes are:

| heads | target-prob drop | rank damage |
| --- | ---: | ---: |
| `1,3,8,13,15` | `0.015329` | `23.083` |
| `1,13` | `0.014735` | `21.333` |
| `1,15` | `0.013799` | `19.583` |
| `1` | `0.012836` | `18.167` |
| `1,3` | `0.012612` | `16.833` |
| `1,8` | `0.012404` | `17.000` |
| `1,12` | `0.011139` | `13.417` |
| `1,7` | `0.011060` | `13.417` |

## Mechanism Update

Head `1` remains the core layer-17 attention head for the primary anchor.
Adding head `3` or head `7` slightly improves repair, but the gains are small
relative to the singleton head-1 effect. Larger groups do not improve repair
and can dilute it. For reverse-direction damage, groups containing `1,13` and
the compact multihead group damage more than singleton head `1`, suggesting
that head `13` may help express the basin when degrading the control state even
though it does not independently repair the masked state well.

The best current picture is:

1. Layer-17 head `1` is sufficient for most masked-to-control coordinate-slot
   repair at the primary duplication onset.
2. Heads `3` and `7` are small repair amplifiers.
3. Head `13` is more important for reverse damage/grouped basin expression
   than for singleton repair.
4. The full self-attention output effect is therefore mostly head-1 centered,
   with minor asymmetric contributions from nearby heads.

## Next Probes

Recommended deterministic branch:

- inspect layer-17 head-1 attention patterns for the primary anchor;
- compare head `1` against heads `3`, `7`, and `13` on the same cases;
- classify whether head `1` attends to visual-region tokens, previous
  coordinate tokens, object-description tokens, or structural delimiter/order
  tokens;
- then decide whether the next intervention should be Q/K/V-level, attention
  score masking, or value-vector source patching.

Guardrail:

- The grouped-head result localizes the current selected-case causal path, but
  it does not prove head `1` is globally responsible for all duplication bursts.
  A broader case panel is still needed after the source-attention story is
  understood.
