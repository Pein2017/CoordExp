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

## Channel-B Pseudo-Positive And Arbitrary-K Notes

When `stage2_ab.channel_b.pseudo_positive.enabled=true`, Channel-B still emits a
single clean teacher-forced forward, but the rollout evidence path widens from
`1` anchor + `1` explorer to `1 + (K-1)` views.

Operational semantics:

- `stage2/raw_rollouts`
  - total number of rollout generations used for the batch
  - under the default pseudo-positive profile this is `4` per eligible sample
- `rollout/explorer/*`
  - preserved as compatibility metrics
  - now interpreted as means over valid explorer views under arbitrary `K`
  - with legacy `K=2`, these still reduce to the single explorer values
- `train/triage/unlabeled_consistent_count`
  - total shielded-anchor count
  - includes support-positive-but-subthreshold anchors and cluster-demoted pseudo-positive candidates
- `train/triage/pseudo_positive_candidate_count`
  - unmatched anchors that meet the promotion floor before overlap clustering
- `train/triage/pseudo_positive_subthreshold_count`
  - unmatched anchors with non-zero support that stay below the promotion threshold
- `train/triage/pseudo_positive_selected_count`
  - final pseudo-positive winners after clustering
- `train/triage/pseudo_positive_cluster_demoted_count`
  - pseudo-positive candidates demoted back to shielded due to overlap clustering
- `train/triage/pseudo_positive_support_rate_num`
  - summed explorer-support numerators over pseudo-positive candidates
- `train/triage/pseudo_positive_support_rate_den`
  - summed explorer-support denominators over pseudo-positive candidates
- `train/triage/pseudo_positive_selected_support_rate_num`
  - summed explorer-support numerators over selected pseudo-positive winners
- `train/triage/pseudo_positive_selected_support_rate_den`
  - summed explorer-support denominators over selected pseudo-positive winners
- `train/triage/recovered_ground_truth_rate_num`
  - summed explorer-hit numerators for recovered GT objects missed by the anchor
- `train/triage/recovered_ground_truth_rate_den`
  - summed valid-explorer denominators for those recovered GT objects
- `train/triage/recovered_ground_truth_rate`
  - `rate_num / rate_den` when the denominator is non-zero
- `train/triage/anchor_preparation_dropped_count`
  - enabled pseudo-positive samples dropped because anchor accepted-clean preparation was malformed

Failure telemetry:

- malformed explorer preparation under enabled pseudo-positive mode aborts the
  step instead of emitting an ordinary finalized `train/triage/*` counter
- treat those aborts as failure telemetry / run outcome, not as a step-level
  rolling metric

## Removed Legacy Groups

Legacy iterative Channel-A provenance groups are no longer emitted by active
training. If they appear in old logs, treat them as historical artifacts rather
than current contract surfaces.

## Diagnostic Reference

The deprecation rationale lives in:

- `progress/diagnostics/stage2_channel_a_self_context_iter_ablation_2026-03-20.md`
