---
title: Phase 4 Layer 17 Head 1 Source Region Findings
date: 2026-06-10
status: active-evidence-note
owner: codex
evidence_scope: phase4-layer17-head1-source-region-top4-allshards
---

# Phase 4 Layer 17 Head 1 Source Region Findings

## Scope

This note records the first source-attention readout for the localized
layer-17 head-1 causal path. It uses filtered Phase 2 attention-region readouts
with `attention_layers=17` and `attention_heads=1`.

Input manifest root:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920
```

Main artifacts:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_attention_source_layer17_head1_region_top4_allshards_summary.json
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_attention_source_layer17_head1_region_top4_allshards_report.md
```

Probe settings:

- eight shard jobs launched across GPUs `0..7`;
- `attention_layers=17`;
- `attention_heads=1`;
- `attn_implementation=eager`;
- visual region memberships from `phase2_region_rows.jsonl`.

Aggregate counts:

- `summary_count=8`;
- `attention_row_count=17584`;
- `replay_case_count_sum=30`;
- `checkpoint_count_sum=17`.

## Primary Anchor

Primary filter:

```text
checkpoint_label=none_latest_ckpt32
record_idx=33
phase=post_y1/pre_x2
layer=17
head=1
```

For this primary case, averaged across the twelve `post_y1/pre_x2` anchors:

| region | mean mass | mean density | mean tokens |
| --- | ---: | ---: | ---: |
| `duplicate_basin` | `0.427484` | `0.42748358` | `1.00` |
| `same_desc_component_envelope` | `0.811320` | `0.05408799` | `15.00` |
| `spatial_basin_component_envelope` | `0.811320` | `0.05408799` | `15.00` |
| `matched_gt_regions` | `0.933628` | `0.00305107` | `306.00` |
| `rest_of_image` | `0.953168` | `0.00280344` | `340.00` |

The region boxes overlap, so masses across rows are not mutually exclusive.
The important signal is density and the single-token duplicate-basin mass.
Head `1` is not merely spreading attention across the whole image: during the
primary onset phase, it frequently places large mass on the one-token
duplicate-basin visual cell.

Anchor-level pattern:

- Early anchors in the same phase have low duplicate-basin mass, e.g. rows
  `20-22` are near zero to `0.001983`.
- Later anchors become sharply duplicate-basin focused: row `23` is `0.593535`,
  row `25` is `0.594768`, row `26` is `0.766472`, row `28` is `0.780058`,
  and row `31` is `0.851418`.

## Mechanism Update

This source readout connects the causal head-localization result to a concrete
visual routing story:

1. Layer-17 head `1` is causally sufficient for most of the primary
   masked-to-control coordinate-slot repair.
2. The same head puts high attention density on the duplicate-basin visual
   cell during the `post_y1/pre_x2` phase.
3. The attention becomes strongest after the early anchors in the phase,
   consistent with a burst-internal attraction/lock-in rather than a uniform
   row-level visual read.

The current best explanation is no longer just "there is a residual basin."
It is more specific: layer-17 head `1` appears to route information from the
duplicate-basin visual cell into the coordinate-slot readout during the
problematic `y1 -> x2` transition, and this route is strong enough to recreate
most of the causal basin under activation patching.

## Next Probes

Recommended deterministic branch:

- add a token-category source readout for layer-17 head `1`, separating visual
  cells from previous coordinate tokens, object-description tokens, row
  delimiters, and prompt/prefix tokens;
- compare the head-1 duplicate-basin visual mass under control vs masked images
  to see whether the mask removes source attention, value content, or both;
- if the source category remains visual-dominant, follow with value-vector or
  attention-score interventions for head `1`;
- if previous coordinate tokens also carry high mass, test whether head `1`
  is coupling visual duplicate evidence to autoregressive coordinate history.

Guardrail:

- This readout is routing evidence, not a causal intervention by itself. Its
  force comes from being aligned with the already-causal layer-17 head-1 patch
  result.
