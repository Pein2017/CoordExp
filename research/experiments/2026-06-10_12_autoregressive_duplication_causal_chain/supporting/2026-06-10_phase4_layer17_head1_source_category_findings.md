---
title: Phase 4 Layer 17 Head 1 Source Category Findings
date: 2026-06-10
status: active-evidence-note
owner: codex
evidence_scope: phase4-layer17-head1-source-category-top4-allshards
---

# Phase 4 Layer 17 Head 1 Source Category Findings

## Scope

This note records the token-category source readout for the localized
layer-17 head-1 causal path. It follows the visual source-region readout, which
showed high attention density on the one-token duplicate-basin visual cell.

Input manifest root:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920
```

Main artifacts:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_attention_source_category_layer17_head1_top4_allshards_summary.json
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_attention_source_category_layer17_head1_top4_allshards_report.md
```

Probe settings:

- eight shard jobs launched across GPUs `0..7`;
- `attention_layers=17`;
- `attention_heads=1`;
- `attn_implementation=eager`;
- source categories: `visual_tokens`, `prompt_prefix_nonvisual`,
  `generated_prior_coord`, `generated_prior_text`,
  `generated_prior_structure`, `generated_prior_special`,
  `current_query_token`, and `generated_future`.

Aggregate counts:

- `summary_count=8`;
- `attention_source_row_count=20096`;
- `source_replay_case_count_sum=30`;
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

| source category | mean mass | mean density | mean tokens |
| --- | ---: | ---: | ---: |
| `visual_tokens` | `0.953168` | `0.00280344` | `340.00` |
| `generated_prior_coord` | `0.031821` | `0.00032552` | `103.00` |
| `current_query_token` | `0.006196` | `0.00619633` | `1.00` |
| `generated_prior_special` | `0.004704` | `0.00009023` | `53.00` |
| `prompt_prefix_nonvisual` | `0.003437` | `0.00000895` | `384.00` |
| `generated_prior_text` | `0.000673` | `0.00001513` | `48.00` |
| `generated_prior_structure` | `0.000000` | `0.00000000` | `0.00` |
| `generated_future` | `0.000000` | `0.00000000` | `46.00` |

The causal head's source mass is therefore overwhelmingly visual. Prior
generated coordinate tokens are not absent, but they are small compared with
visual-token mass: roughly `0.032` vs `0.953`.

## Mechanism Update

The source-category readout rules out a tempting but weaker explanation:
layer-17 head `1` is not mainly copying or attending to previous coordinate
tokens at the primary `y1 -> x2` onset. It is primarily a visual-source head.

Combining the current evidence:

1. Layer-17 head `1` is causally sufficient for most masked-to-control repair
   at `none_latest_ckpt32|post_y1/pre_x2`.
2. The same head attends densely to the duplicate-basin visual cell.
3. Coarse token-category mass shows the same head is overwhelmingly visual,
   not dominated by prior coordinate history.
4. Prior coordinate-token attention remains a small secondary channel, worth
   tracking but not the main current mechanism.

The current best picture is now: layer-17 head `1` routes visual duplicate
evidence into the coordinate-slot basin during the `y1 -> x2` transition. This
visual route appears to be the main source of the causal patch effect, while
autoregressive coordinate history may modulate but does not dominate it.

## Next Probes

Recommended deterministic branch:

- compare control vs masked source-category readouts for layer-17 head `1` to
  determine whether masking changes attention allocation, value content, or
  both;
- if attention allocation is stable but patching is causal, move to head-1
  value-vector source patching;
- if attention allocation changes sharply under masking, test attention-score
  or source-position interventions on the duplicate-basin visual cell;
- keep `none_latest_ckpt32|post_y1/pre_x2` as the primary anchor and use the
  broader post-y1/pre-x2 category aggregates only as context.

Guardrail:

- This readout uses coarse categories and a teacher-forced prefix. Future
  generated tokens appear in the input sequence but receive zero attention
  under the causal mask. The category result should be interpreted with the
  causal patch evidence, not as standalone proof.
