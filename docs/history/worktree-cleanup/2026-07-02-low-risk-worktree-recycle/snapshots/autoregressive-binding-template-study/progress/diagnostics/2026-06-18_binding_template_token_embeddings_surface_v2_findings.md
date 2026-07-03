---
doc_id: progress.diagnostics.binding_template_token_embeddings_surface_v2_findings
layer: progress
doc_type: diagnostic-findings
status: branch-provenance
domain: research-history
summary: Token-embeddings-adapter surface migration and second-pass post-hoc mechanism split for the checkpoint-928 binding-template ablation.
tags: [progress, diagnostics, autoregressive-binding, token-embeddings-adapter, coord-logits, hidden-states, attention]
updated: 2026-06-18
branch: codex/autoregressive-binding-template-study
---

# Binding Template Token-Embeddings Surface V2 Findings

## Scope

This note records the `v2` continuation of the checkpoint-928
`desc_first` versus `geometry_first` compact `compact_object_box_closed`
mechanism study.

Executable config:

```text
configs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200.yaml
```

Probe artifact root:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200_deep_probe_v2_token_embeddings_surface
```

Post-hoc mechanism artifacts:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200_deep_probe_v2_token_embeddings_surface/posthoc_mechanism_summary.json
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200_deep_probe_v2_token_embeddings_surface/posthoc_mechanism_report.md
```

Baseline comparison root:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200_deep_probe_v1
```

## Adapter-Surface Migration

The `geometry_first` checkpoint already uses the current
`token_embeddings_adapter` surface. The `desc_first` checkpoint was a
historical `coord_offset_adapter` checkpoint, so this branch materialized a
derived adapter-surface conversion rather than continuing to rely on an
in-probe compatibility shim.

Converted `desc_first` checkpoint:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/converted_adapters/desc_first_ckpt928_token_embeddings_adapter/checkpoint-928
```

Conversion manifest:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/converted_adapters/desc_first_ckpt928_token_embeddings_adapter/checkpoint-928/conversion_manifest.json
```

Conversion details:

- source checkpoint:
  `/data/CoordExp/outputs/stage1_2b/coco_bbox_len12000_1024-coco80-desc_first-compact_object_box_closed-sorted-packed12k-natural_adjacent-pure_ce/epoch_4-pure_ce-coco80-desc_first-1024-coco_bbox_len12000-compact_object_box_closed-sorted-packed12k-natural_adjacent-llm_only/v0-20260617-133335/checkpoint-928`
- base model relocated to:
  `/data/CoordExp/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent`
- rewritten tensor keys:
  - `base_model.model.coord_offset_adapter.coord_ids` ->
    `base_model.model.token_embeddings_adapter.token_ids`
  - `base_model.model.coord_offset_adapter.embed_offset` ->
    `base_model.model.token_embeddings_adapter.embed_offset`
- passthrough tensor key count: `588`
- token id count: `1004`
- source adapter weight SHA256:
  `421f084f398408130afc5df808db556946c8e49fc679971ace609ed9b7606473`
- converted adapter weight SHA256:
  `089d6debe13241d257d5b025c7fc27266a39f85f61067949347905054cb11c47`

The converted adapter passed the strict inference checkpoint contract for
`compact_object_box_closed` with `modules_to_save=('token_embeddings_adapter',)`
and `1004` token ids.

## V2 Replay Verification

The `v2` replay used 8 GPU shards and merged cleanly:

```json
{
  "attention_rows": 4932,
  "coord_logit_rows": 548,
  "role_rows": 14796,
  "shard_errors": [],
  "shards_seen": 8
}
```

The `v2` JSONL rows are exactly equal to `v1` for:

- `case_surface_rows.jsonl`
- `object_step_rows.jsonl`
- `token_role_rows.jsonl`
- `replay_role_rows.jsonl`
- `coord_logit_rows.jsonl`
- `attention_rows.jsonl`

Interpretation: the adapter-surface migration is behavior-preserving for this
probe. The `v2` evidence should supersede the `v1` loader caveat for ongoing
analysis, while still preserving the legacy source checkpoint path and
conversion manifest as provenance.

## Main New Mechanism Finding

The second-pass post-hoc report separates two failure modes that were partly
collapsed in the initial read:

1. Previous-IoU duplicates are a **previous-anchor reuse branch**.
2. Unmatched rows are a **high-entropy anchor-search branch**.

Both are autoregressive failures, but they are not the same local basin event.

All parsed object rows still had positive `remaining_gt_before`, including
duplicate and unmatched rows:

| kind | n | mean object idx | mean remaining GT | positive remaining GT | object idx >= 21 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `duplicate_iou70` | 70 | 50.89 | 6.51 | 1.00 | 0.81 |
| `new_gt` | 138 | 10.62 | 12.48 | 1.00 | 0.13 |
| `repeated_gt` | 16 | 17.44 | 8.38 | 1.00 | 0.50 |
| `unmatched` | 237 | 41.11 | 7.49 | 1.00 | 0.66 |

So the failure is not simple set exhaustion. It is a late-prefix regime shift
while viable GT objects remain.

The `x1` previous-anchor readout is much sharper than generic duplication
counts:

| kind | n | mean emitted x1 delta from previous | median emitted x1 delta | previous emitted x1 in current top-k |
| --- | ---: | ---: | ---: | ---: |
| `duplicate_iou70` | 7 | 13.57 | 8.00 | 0.57 |
| `new_gt` | 65 | 193.45 | 138.00 | 0.08 |
| `repeated_gt` | 9 | 91.11 | 28.00 | 0.22 |
| `unmatched` | 42 | 155.02 | 80.00 | 0.31 |

This means a visible duplicate often begins as a near-reuse of the immediately
previous `x1` coordinate basin. In contrast, many unmatched rows are not
near-copies of the previous anchor; they are broader uncertain searches.

The phase-controlled residuals reinforce the split. Residuals control only
family, object index, and `remaining_gt_before`, so they are observational,
but they are useful confound checks:

| metric | duplicate residual mean | unmatched residual mean | new-GT residual mean |
| --- | ---: | ---: | ---: |
| `last_pre_x1_hidden_norm` | +114.89 | +118.54 | -78.42 |
| `x1_entropy` | +0.293 | +0.582 | -0.364 |
| `x1_expected_abs_error` | -18.57 | +46.40 | -24.18 |
| `pre_x1_assistant_previous_objects` | +0.0168 | +0.0032 | -0.0048 |
| `pre_x1_wrapper_tokens` | +0.0174 | +0.0029 | -0.0045 |

Mechanism interpretation:

- Duplicates: high state load, elevated previous/wrapper routing, elevated
  `x1` entropy, but the emitted `x1` remains close to the previous anchor.
  This looks like local re-entry into the immediately previous coordinate
  basin.
- Unmatched rows: high state load and even higher `x1` entropy, plus positive
  `x1_expected_abs_error` residual. This looks like an unresolved anchor search
  rather than a clean copy.
- New-GT rows: lower residual hidden-state load and lower residual `x1`
  entropy after the same coarse phase controls.

This is the first concrete evidence in this clean template ablation that
`duplication` and `unmatched drift` share the same late-prefix pressure but
split at the `x1` anchor selection mechanism.

## Practical Consequence

The next causal probe should not treat "duplicate or unmatched" as one class.
It should intervene separately on:

- duplicate branch: previous-object / wrapper-state removal or patching before
  `pre_x1`, then measure whether the next `x1` stays near the previous anchor;
- unmatched branch: guidance or candidate-field sharpening at `pre_x1`, then
  measure whether the high-entropy x1 basin collapses toward a remaining GT;
- false negatives: continue to test whether missing objects can be language or
  coordinate guided into the candidate field, but stratify by whether the
  current failure row is an anchor-reuse duplicate or a high-entropy unmatched
  branch.

This supports dynamic follow-up: if a causal patch strongly moves either
branch, it is worth diving deeper before returning to the broader roadmap,
because the branch split changes the final mechanistic picture more than
another aggregate duplicate-count comparison would.

## Boundaries

- Scope is `val200` panel evidence over existing free-rollout artifacts with
  `temperature=0.0`, `rp=1.10`, and `max_new_tokens=3084`.
- The replay rows are teacher-forced on rendered parsed predictions.
- Parse-failure cases still contribute surface-token evidence only.
- Attention summaries and residuals are observational.
- Phase residuals control only family, object index, and remaining GT; they do
  not prove a causal role for attention or hidden-state magnitude.
