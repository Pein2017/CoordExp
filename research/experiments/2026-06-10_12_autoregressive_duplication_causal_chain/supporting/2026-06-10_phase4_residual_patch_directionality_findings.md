---
title: Phase 4 Residual Patch Directionality Findings
date: 2026-06-10
status: active-evidence-note
owner: codex
evidence_scope: phase4-top24-allshards-smoke
---

# Phase 4 Residual Patch Directionality Findings

## Scope

This note records the first all-shard bidirectional residual-patch sweep for
the autoregressive duplication mechanism study. It is a launch-ranking and
mechanism-refinement artifact, not a final causal diagnosis.

Input manifest root:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920
```

Main artifacts:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_residual_patch_bidirectional_layersweep_12_16_20_24_top24_allshards_summary.json
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_residual_patch_bidirectional_layersweep_12_16_20_24_top24_allshards_report.md
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_residual_patch_bidirectional_layersweep_12_16_20_24_top24_allshards_directionality.json
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_residual_patch_bidirectional_layersweep_12_16_20_24_top24_allshards_directionality.md
```

Sweep settings:

- `target_top_k=24`;
- `top_k=8`;
- patch layers `12,16,20,24`;
- patch directions `masked_to_control,control_to_masked`;
- eight shard jobs launched across GPUs `0..7`.

Aggregate counts:

- `summary_count=8`;
- `residual_patch_row_count=2184`;
- `targeted_replay_case_count=7`;
- `checkpoint_count_sum=6`;
- directionality paired rows `36`.

## Findings

The dominant update is that the residual-patch signal is often bidirectional in
probability space. The earlier top-case impression that downstream layers were
mostly clean-state restoration points was too narrow. In the broader panel,
many `masked_to_control` repairs have comparable `control_to_masked` damage,
which means the prob-space coordinate-slot state is reversible at the patched
residual location for several phases and checkpoints.

The strongest continuation cases are still concentrated in the continuation
checkpoints:

- `none_latest_ckpt32|post_y1/pre_x2|layer=20`:
  `prob_repair=0.01477210`, `prob_damage=0.01389848`,
  `rank_repair=20.750`, `rank_damage=18.833`.
- `none_latest_ckpt32|post_y1/pre_x2|layer=24`:
  `prob_repair=0.01461257`, `prob_damage=0.01331987`,
  `rank_repair=20.583`, `rank_damage=17.583`.
- `aux_latest_ckpt32|post_x1/pre_y1|layer=24`:
  `prob_repair=0.01377835`, `prob_damage=0.01359171`,
  `rank_repair=1.583`, `rank_damage=2.333`.
- `aux_latest_ckpt32|post_x1/pre_y1|layer=20`:
  `prob_repair=0.01354669`, `prob_damage=0.01336445`,
  `rank_repair=2.583`, `rank_damage=2.167`.

Layer `16` remains important, but the evidence now says it is not the only
place where the bad coordinate-slot state can be causally injected. Examples:

- `none_latest_ckpt32|post_x1/pre_y1|layer=16`:
  `prob_repair=0.00452422`, `prob_damage=0.01287674`,
  `rank_repair=4.833`, `rank_damage=5.333`.
- `aligner_parent_ckpt1824|post_y1/pre_x2|layer=16`:
  `prob_repair=-0.00543566`, `prob_damage=0.00511456`,
  `rank_repair=-0.364`, `rank_damage=15.636`.
- `no_aligner_parent_ckpt3668|post_x1/pre_y1|layer=16`:
  `prob_repair=-0.00168782`, `prob_damage=0.00257709`,
  `rank_repair=3.250`, `rank_damage=7.417`.

Parent checkpoint effects are smaller in prob space than the continuation
effects, but they are not empty. The parent evidence is more rank-heavy and
phase-specific, especially at `post_y1/pre_x2` for the aligner parent and
`post_x1/pre_y1` for the no-aligner parent.

## Mechanism Update

The current best picture is not simply "layer 16 seeds the bad state, layers
20/24 amplify it." A better working model is:

1. The visual/coordinate perturbation creates a coordinate-slot residual state
   that becomes patch-causal around the mid-to-late residual stream.
2. Several layer/phase points are reversible under single-token residual
   patching, especially continuation-checkpoint coordinate slots.
3. Layer `16` remains a likely routing/readout transition point because it
   appears in attention target ranking and can inject rank damage, including in
   parent checkpoints.
4. Layers `20/24` are stronger probability-level state carriers for some
   continuation phases, especially `none_latest_ckpt32|post_y1/pre_x2`.

This shifts the next question from "which single layer causes duplication" to
"which module family transforms the mid-layer visual/routing change into the
late coordinate-slot residual state."

## Next Probes

Recommended next deterministic branch:

- run head-level attention patch or ablation around layer `16`, especially
  head `13` and nearby heads selected by the Phase 4 target manifest;
- pair that with residual patching at layers `20/24` for the same replay cases
  to test whether layer-16 attention is upstream of the late coordinate-slot
  residual state;
- keep parent and continuation checkpoints in the same panel, because parent
  rank damage suggests the mechanism may predate the auxiliary/none
  continuation split.

Guardrail:

- The directionality report ranks paired residual-patch effects. It does not
  by itself identify the responsible submodule inside a layer or prove that
  attention, MLP, or coordinate-token embeddings are the origin.
