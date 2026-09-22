---
title: Phase 4 Residual Module-Site Findings
date: 2026-06-10
status: active-evidence-note
owner: codex
evidence_scope: phase4-module-site-top2-shard04-smoke
---

# Phase 4 Residual Module-Site Findings

## Scope

This note records the first module-site residual-patch smoke for the
autoregressive duplication mechanism study. It follows the bidirectional
decoder-layer residual patch and layer-16 head-13 attention patch probes, but
patches the output of specific decoder submodules instead of the full decoder
layer residual output.

Input manifest root:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920
```

Main artifacts:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_module_patch_sites_smoke_top2_shard04/phase4_residual_patch_summary.json
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_module_patch_sites_smoke_top2_shard04/residual_patch_rows.jsonl
```

Probe settings:

- shard `04`;
- `target_top_k=2`;
- patch layers `16,20,24`;
- patch sites `self_attn,mlp`;
- patch directions `masked_to_control,control_to_masked`;
- `top_k=8`.

Aggregate counts:

- `residual_patch_row_count=288`;
- `targeted_replay_case_count=1`;
- `checkpoint_count=1`;
- summary keys `24`.

## Findings

The cleanest signal in this smoke is a split between attention-side injection
and MLP/later transformation. `self_attn` patching at layers `16` and `20`
strongly damages the clean/control state when the masked state is patched into
control for `post_x1/pre_y1`, but the reverse repair direction is weak or
negative. This is consistent with attention being able to route or inject
coordinate-slot damage without being sufficient to reconstruct the healthier
late residual state on its own.

Representative `self_attn` damage cases:

- `none_latest_ckpt32|post_x1/pre_y1|control_to_masked|site=self_attn|layer=16`:
  `mean_prob_recovery_from_masked=0.01031466`,
  `mean_prob_damage_from_control=-0.00093648`,
  `mean_rank_recovery_from_masked=4.583`,
  `mean_rank_damage_from_control=0.167`.
- `none_latest_ckpt32|post_x1/pre_y1|control_to_masked|site=self_attn|layer=20`:
  `mean_prob_recovery_from_masked=0.01036657`,
  `mean_prob_damage_from_control=-0.00088457`,
  `mean_rank_recovery_from_masked=4.167`,
  `mean_rank_damage_from_control=0.583`.

For `post_y1/pre_x2`, `mlp` patching carries stronger repair than `self_attn`,
especially at layer `24`:

- `none_latest_ckpt32|post_y1/pre_x2|masked_to_control|site=mlp|layer=24`:
  `mean_prob_recovery_from_masked=0.00913343`,
  `mean_rank_recovery_from_masked=16.75`.

The corresponding `self_attn` repair for `post_y1/pre_x2` is near zero or
negative:

- `layer=16`: `mean_prob_recovery_from_masked=-0.00016681`,
  `mean_rank_recovery_from_masked=0.083`;
- `layer=20`: `mean_prob_recovery_from_masked=-0.00100640`,
  `mean_rank_recovery_from_masked=-3.917`;
- `layer=24`: `mean_prob_recovery_from_masked=-0.00021578`,
  `mean_rank_recovery_from_masked=-1.000`.

## Mechanism Update

This smoke strengthens the current transformation picture:

1. Layer-16 attention can be causally relevant, but isolated attention output
   is not the whole coordinate-slot basin mechanism.
2. Attention-side patching appears more like a routing or injection path for
   masked-image damage than a complete restoration path.
3. MLP and later-layer outputs are stronger candidates for transforming
   routed visual/coordinate information into the residual state that repairs
   coordinate-token probability and rank at `post_y1/pre_x2`.
4. The next high-value branch should separate "where information enters" from
   "where it becomes a coordinate-slot attractor."

This is a promising and attractive path with likely influence over the final
mechanistic picture, so dynamic adjustment is acceptable. If follow-up probes
continue to concentrate on the attention-to-MLP transition, the roadmap should
shift more budget toward submodule and representation-transfer tests rather
than exhaustively expanding lower-value coarse sweeps.

## Next Probes

Recommended deterministic branch:

- expand module-site patching from this single shard/top-2 smoke to all shards
  for the selected high-signal windows;
- include `decoder_layer`, `self_attn`, and `mlp` in the same summary so
  submodule effects can be compared against the full residual patch baseline;
- keep both directions, because injection and repair are currently asymmetric;
- pair the module-site result with multi-head layer-16 attention patching to
  test whether attention provides the upstream route that MLP/later layers
  transform.

Guardrail:

- This note is a top-2 single-shard smoke. It is strong enough to redirect the
  next probe branch, but not sufficient for final causal claims about module
  responsibility.
