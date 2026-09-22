---
title: Phase 4 Layer 17 All-Heads Findings
date: 2026-06-10
status: active-evidence-note
owner: codex
evidence_scope: phase4-layer17-allheads-top4-allshards
---

# Phase 4 Layer 17 All-Heads Findings

## Scope

This note records the layer-17 all-singleton-head attention patch probe. It was
launched after the first layer-17 head probe showed that reused layer-16 target
head `13` did not explain the full layer-17 self-attention output effect.

Input manifest root:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920
```

Main artifacts:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_attention_patch_layer17_allheads_top4_allshards_summary.json
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_attention_patch_layer17_allheads_top4_allshards_report.md
```

Probe settings:

- eight shard jobs launched across GPUs `0..7`;
- `target_top_k=4`;
- patch layer override `17`;
- singleton patch heads `0..15`;
- patch directions `masked_to_control,control_to_masked`;
- `top_k=8`.

Aggregate counts:

- `summary_count=8`;
- `attention_patch_row_count=1536`;
- `targeted_replay_case_count=2`;
- `checkpoint_count_sum=2`.

## Primary Anchor

For `none_latest_ckpt32|post_y1/pre_x2`, singleton head effects are sharply
concentrated:

| head | masked-to-control repair | rank repair | paired control-to-masked target-prob drop | rank damage |
| ---: | ---: | ---: | ---: | ---: |
| `1` | `0.013860` | `21.833` | `0.012836` | `18.167` |
| `13` | `0.001690` | `4.417` | weak/opposite signed in aggregate | weak |
| `15` | `0.000444` | `1.583` | weak | weak |
| `3` | `0.000358` | `1.583` | weak | weak |
| `8` | `0.000086` | `0.083` | weak | weak |

The reference ceilings from the layer-17 module-boundary run were:

- full `self_attn` output repair `0.015438`, damage `0.009726`;
- full `decoder_layer` output repair `0.014757`, damage `0.013846`.

Thus, singleton head `1` almost saturates the full self-attention output repair
and has a reverse-direction damage magnitude comparable to the full
decoder-layer output effect. The previously tested head `13` is not the main
layer-17 causal head for this anchor.

## Cross-Anchor Context

The top singleton repair outside the primary anchor is smaller:

- `aux_latest_ckpt32|post_x1/pre_y1|head=12`: repair `0.006413`;
- `none_latest_ckpt32|post_x1/pre_y1|head=12`: repair `0.004811`;
- `aux_latest_ckpt32|post_x1/pre_y1|head=8`: repair `0.003849`;
- `none_latest_ckpt32|post_x1/pre_y1|head=8`: repair `0.002365`.

This reinforces the earlier decision to keep
`none_latest_ckpt32|post_y1/pre_x2` as the main origin anchor. It is the only
window in this panel with a large, clean singleton-head localization.

## Mechanism Update

The current best picture is now:

1. Layers `14-16` carry a repair-capable but mostly asymmetric coordinate-slot
   direction.
2. At layer `17`, self-attention output exposes the transition into a
   bidirectional basin.
3. That transition is largely localized to attention head `1` for the primary
   anchor, not to the layer-16-local-attention target head `13`.
4. Full decoder-layer output keeps the basin, so downstream wrapper/residual
   behavior preserves or slightly completes the head-1 effect.

This is the strongest mechanistic localization so far in the run.

## Next Probes

Recommended deterministic branch:

- run grouped attention patching around layer-17 head `1`, especially
  `1`, `1,13`, `1,12`, and small groups containing the weaker positive heads;
- inspect layer-17 head-1 attention patterns for the primary anchor, including
  whether it attends to the duplicated object region, previous coordinate
  tokens, object-description tokens, or delimiter/order tokens;
- if attention pattern analysis confirms a meaningful source, follow with
  Q/K/V or attention-score interventions for head `1`;
- preserve the primary anchor and use the `post_x1/pre_y1` windows only as
  robustness controls.

Guardrail:

- This is still a top-4 selected-case panel. It strongly localizes the current
  causal effect, but broader cases are needed before claiming head `1` as the
  general duplication-basin origin.
