---
title: Phase 4 Layer 17 Attention Head Findings
date: 2026-06-10
status: active-evidence-note
owner: codex
evidence_scope: phase4-layer17-attention-head-top4-allshards
---

# Phase 4 Layer 17 Attention Head Findings

## Scope

This note records the first layer-17 attention-head output patch probe. It was
launched after the layer-17 module-boundary probe showed that full
self-attention output has a strong primary-anchor effect, while the previously
selected target manifest only contained layer-16 attention targets.

This run used a new `--patch-layers` override in the attention patch runner, so
the layer-16 target heads from the manifest were patched at layer `17`.

Input manifest root:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920
```

Main artifacts:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_attention_patch_layer17_top4_allshards_summary.json
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_attention_patch_layer17_top4_allshards_report.md
```

Probe settings:

- eight shard jobs launched across GPUs `0..7`;
- `target_top_k=4`;
- patch layer override `17`;
- default selected target heads from the top-4 target manifest;
- patch directions `masked_to_control,control_to_masked`;
- `top_k=8`.

Aggregate counts:

- `summary_count=8`;
- `attention_patch_row_count=96`;
- `targeted_replay_case_count=2`;
- `checkpoint_count_sum=2`.

## Primary Anchor

The primary anchor is again `none_latest_ckpt32|post_y1/pre_x2`. For the
manifest-selected layer-16 target head patched at layer `17`:

- `head=13`, masked-to-control repair: `0.001690`;
- paired control-to-masked damage: `0.001973`;
- masked-to-control rank repair: `4.417`;
- paired control-to-masked rank damage: `-1.417`.

This is much weaker than the full layer-17 self-attention output patch from the
module-boundary panel:

- full `self_attn` output repair: `0.015438`;
- full `self_attn` output damage: `0.009726`;
- full `self_attn` rank repair: `22.833`;
- full `self_attn` rank damage: `11.250`.

## Mechanism Update

The first head-level result argues against a simple "the layer-16 head-13
target becomes the layer-17 causal basin head" story. The full layer-17
self-attention output is strongly causal for the primary anchor, but the
layer-16-selected head `13` alone captures only a small fraction of that effect
at layer `17`.

The leading explanations are now:

1. The layer-17 self-attention effect is distributed across multiple heads.
2. The relevant layer-17 head is not among the top layer-16 attention targets.
3. The important operation is not a single head output, but a grouped attention
   subspace plus downstream layer-wrapper interaction.

This keeps layer-17 attention as a promising origin branch, but shifts the next
probe from single reused head `13` toward broader layer-17 head/group scans.

## Next Probes

Recommended deterministic branch:

- run layer-17 all-head or grouped-head attention patching for the primary
  anchor;
- include singleton heads first if affordable, then grouped top subsets if the
  effect is diffuse;
- keep the module-boundary full `self_attn` and `decoder_layer` effects as the
  reference ceiling;
- add an attention directionality report helper if repeated attention-head
  scans continue, so paired repair/damage tables are generated automatically.

Guardrail:

- This run reused layer-16-selected target heads and only tested the default
  selected head set under `target_top_k=4`. It should be treated as a negative
  localization result for head `13`, not as a negative result for layer-17
  attention overall.
