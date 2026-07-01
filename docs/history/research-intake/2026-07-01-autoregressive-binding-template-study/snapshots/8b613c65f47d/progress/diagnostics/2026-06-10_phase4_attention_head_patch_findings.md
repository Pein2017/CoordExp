---
title: Phase 4 Attention Head Patch Findings
date: 2026-06-10
status: active-evidence-note
owner: codex
evidence_scope: phase4-top4-head13-allshards-smoke
---

# Phase 4 Attention Head Patch Findings

## Scope

This note records the first real-runtime attention-head output patch probe for
the autoregressive duplication mechanism study. The probe patches the selected
head slice in the input to a decoder attention `o_proj`, at the same generated
coordinate-slot token used by the Phase 4 residual patch probes.

Input manifest root:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920
```

Main artifacts:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_attention_patch_top4_head13_allshards_summary.json
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_attention_patch_top4_head13_allshards_report.md
```

Probe settings:

- `target_top_k=4`;
- `top_k=8`;
- patch head `13`;
- patch directions `masked_to_control,control_to_masked`;
- eight shard jobs launched across GPUs `0..7`.

Aggregate counts:

- `summary_count=8`;
- `attention_patch_row_count=96`;
- `targeted_replay_case_count=2`;
- `checkpoint_count_sum=2`.

## Findings

Head `13` at layer `16` is causally relevant but not sufficient by itself to
explain the full late residual-state damage seen in the residual patch panel.
The cleanest reverse-direction statistic is `prob_damage_from_control`, because
`prob_recovery_from_masked` can be positive for `control_to_masked` when the
patched control path remains healthier than the fully masked path.

Observed continuation-checkpoint effects:

- `none_latest_ckpt32|post_x1/pre_y1|control_to_masked|layer=16|head=13`:
  `prob_damage_from_control=-0.00279448`, `rank_damage_from_control=1.000`.
- `none_latest_ckpt32|post_y1/pre_x2|control_to_masked|layer=16|head=13`:
  `prob_damage_from_control=-0.00065010`, `rank_damage_from_control=-0.750`.
- `aux_latest_ckpt32|post_x1/pre_y1|control_to_masked|layer=16|head=13`:
  `prob_damage_from_control=-0.00210295`, `rank_damage_from_control=-0.417`.
- `aux_latest_ckpt32|post_y1/pre_x2|control_to_masked|layer=16|head=13`:
  `prob_damage_from_control=-0.00431523`, `rank_damage_from_control=3.083`.

The `masked_to_control` repair direction is much weaker for this isolated
head-output patch than for full decoder-layer residual patching:

- `none_latest_ckpt32|post_x1/pre_y1|masked_to_control|layer=16|head=13`:
  `prob_recovery_from_masked=0.00033610`,
  `rank_recovery_from_masked=-4.250`.
- `none_latest_ckpt32|post_y1/pre_x2|masked_to_control|layer=16|head=13`:
  `prob_recovery_from_masked=0.00021932`,
  `rank_recovery_from_masked=0.250`.
- `aux_latest_ckpt32|post_x1/pre_y1|masked_to_control|layer=16|head=13`:
  `prob_recovery_from_masked=0.00015734`,
  `rank_recovery_from_masked=-5.667`.
- `aux_latest_ckpt32|post_y1/pre_x2|masked_to_control|layer=16|head=13`:
  `prob_recovery_from_masked=0.00043385`,
  `rank_recovery_from_masked=1.167`.

## Mechanism Update

This supports a partial-upstream role for layer-16 head `13`, not a complete
single-head explanation. Head `13` can inject some masked-image coordinate-slot
damage into the clean/control path, especially in `aux_latest_ckpt32` at
`post_y1/pre_x2` and `none_latest_ckpt32` at `post_x1/pre_y1`. However, the
weak `masked_to_control` repair means the late residual state is not simply the
head-13 output slice copied forward unchanged.

Current best update:

1. Layer-16 head `13` is a plausible routing contributor.
2. The larger residual-patch effects likely require either multiple heads,
   attention output plus MLP transformation, or downstream accumulation across
   layers `20/24`.
3. The next causal branch should test multi-head layer-16 attention-output
   patching and compare it against `self_attn`/`mlp` submodule residual patches
   at the same layers.

Guardrail:

- This is a top-4/head-13 smoke, not a full head atlas. It tests one prominent
  attention head from the Phase 4 manifest and should be expanded before final
  mechanism claims.
