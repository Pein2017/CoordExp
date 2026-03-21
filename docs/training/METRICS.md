# Training Metrics and Losses (Current Baseline)

This reference reflects the active single-pass Stage-2 contract after removal
of Channel-A self-context iteration.

## Stage-2 Two-Channel Metrics

Channel-A emits only the normal single-pass groups:

- `loss/text/{struct_ce,desc_ce}`
- `loss/coord/{bbox_smoothl1,bbox_ciou,bbox_log_wh,bbox_oversize,coord_token_ce,coord_soft_ce,coord_w1,coord_el1,coord_ehuber,coord_entropy,coord_gate,text_gate}`
- `coord_diag/*`
- `gradmon/*/coord/*`

Channel-B keeps rollout-specific provenance:

- `loss/B_rollout_text/{struct_ce,desc_ce}`
- `train/optimization/loss_dead_anchor_suppression`
- `loss/B_coord/{bbox_smoothl1,bbox_ciou,bbox_log_wh,bbox_oversize,coord_token_ce,coord_soft_ce,coord_w1,coord_el1,coord_ehuber,coord_entropy,coord_gate,text_gate}`
- `coord_diag/B/*`
- `gradmon/*/B_coord/*`

## Removed Legacy Groups

Legacy iterative Channel-A provenance groups are no longer emitted by active
training. If they appear in old logs, treat them as historical artifacts rather
than current contract surfaces.

## Diagnostic Reference

The deprecation rationale lives in:

- `progress/diagnostics/stage2_channel_a_self_context_iter_ablation_2026-03-20.md`
