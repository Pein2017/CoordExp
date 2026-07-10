# Cross-Phase Visual-Causal Synthesis

Date: 2026-06-11

## Scope

This note joins Phase 1 coordinate-logit evidence, Phase 2 layer-17/head-1
duplicate-basin routing evidence, and Phase 3 duplicate-basin masking evidence
for the current selected-window panel.

The goal is prioritization for deeper mechanism probes, not a new standalone
metric. The joined score identifies case/phase rows where attention routing to
the duplicate basin and masking sensitivity agree.

## Artifact Root

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133
```

Synthesis artifacts:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/cross_phase_case_synthesis/cross_phase_case_evidence_rows.jsonl
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/cross_phase_case_synthesis/cross_phase_case_evidence_summary.json
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/cross_phase_case_synthesis/cross_phase_case_evidence_report.md
```

Inputs:

- Phase 1: `shards/shard-*-of-04/forward_readouts/coord_logit_rows.jsonl`
- Phase 2: `shards/shard-*-of-04/attention_readouts/attention_region_rows.jsonl`
- Phase 3: `shards/shard-*-of-04/phase3_masking/masking_delta_rows.jsonl`

## Counts

- case/phase rows: `240`;
- complete Phase 1 + Phase 2 + Phase 3 triplets: `120`;
- checkpoints: `4`;
- records: `15`;
- phases: `8`.

Only coordinate-bearing phases have complete triplets because Phase 1/3
coordinate-logit rows exist only where the next token is a coordinate token.

## Score Definition

The visual-causal prioritization score is:

```text
duplicate_minus_rest_attention_density * max(-mask_target_prob_delta, 0)
```

where:

- `duplicate_minus_rest_attention_density` comes from the Phase 2 layer-17/head-1
  attention panel;
- `mask_target_prob_delta` is
  `duplicate_basin_mask_target_prob - no_op_control_target_prob` from Phase 3.

This score is deliberately simple. It is useful for ranking rows where routed
attention and causal masking harm agree, but it is not an independent causal
estimator and should not be promoted as a stable benchmark metric.

## Top Aligned Rows

| Checkpoint | Record | Phase | score | duplicate-rest density | mask target-prob delta | mask rank delta | Phase 1 rank1 fraction |
|---|---:|---|---:|---:|---:|---:|---:|
| `none_latest_ckpt32` | 33 | `post_y1/pre_x2` | 0.006148 | 0.424680 | -0.014477 | +20.5000 | 0.0833 |
| `no_aligner_parent_ckpt3668` | 48 | `post_y1/pre_x2` | 0.004114 | 0.391187 | -0.010516 | +4.0000 | 0.1667 |
| `no_aligner_parent_ckpt3668` | 36 | `post_y1/pre_x2` | 0.001861 | 0.294322 | -0.006324 | +5.3636 | 0.0909 |
| `aux_latest_ckpt32` | 33 | `box_start/pre_x1` | 0.001382 | 0.220806 | -0.006259 | +256.9167 | 0.0833 |
| `aux_latest_ckpt32` | 33 | `post_y1/pre_x2` | 0.001095 | 0.352291 | -0.003108 | -1.1667 | 0.0000 |
| `none_latest_ckpt32` | 33 | `box_start/pre_x1` | 0.000969 | 0.221795 | -0.004367 | +230.9167 | 0.0833 |
| `no_aligner_parent_ckpt3668` | 34 | `box_start/pre_x1` | 0.000758 | 0.057409 | -0.013198 | -0.2500 | 0.8333 |
| `aligner_parent_ckpt1824` | 54 | `post_y1/pre_x2` | 0.000466 | 0.238146 | -0.001956 | -10.2727 | 0.0000 |

## Interpretation

The strongest joined evidence is `none_latest_ckpt32`, record `33`,
`post_y1/pre_x2`.

Why it is high priority:

- Phase 2 routing: duplicate-basin density strongly exceeds rest-of-image
  density by `0.424680`.
- Phase 3 masking: duplicate-basin masking lowers target-coordinate probability
  by `-0.014477` and worsens mean rank by `+20.5`.
- This same record/checkpoint pair was already prominent in the Phase 2
  duplicate-density table and in earlier route/content plus mask-geometry
  findings.

The parent-checkpoint comparators `no_aligner_parent_ckpt3668` records `48` and
`36` are also useful because their joined score is high without being the latest
continuation checkpoint. They can help separate mechanism inherited from the
random-SFT parent from mechanisms amplified or rerouted by the continuation.

Record `54` remains useful as a validated smoke and broad top-left duplicate
basin case, but the joined score says it is not the strongest aligned
routing-plus-masking case under this layer/head panel. It is more likely an
amplitude or regime-comparison case than the best single direct-trigger target.

## Next Probe Targets

Recommended next focused probes:

1. Use `none_latest_ckpt32`, record `33`, phase `post_y1/pre_x2` as the main
   route/content or hidden-state causal patch target.
2. Use `aux_latest_ckpt32`, record `33`, phases `box_start/pre_x1` and
   `post_y1/pre_x2` as the same-image continuation contrast.
3. Use `no_aligner_parent_ckpt3668`, records `48` and `36`, phase
   `post_y1/pre_x2`, as parent-checkpoint comparators.
4. Keep `aligner_parent_ckpt1824`, record `54`, phase `post_y1/pre_x2` as a
   smoke-calibrated top-left basin comparator, not the primary direct-trigger
   case.

The next mechanism step should avoid averaging these rows together. They
represent different regimes: latest-checkpoint record-33 aligned
routing-plus-masking, parent-checkpoint inherited visual-basin dependence, and
record-54 broad-basin amplitude behavior.
