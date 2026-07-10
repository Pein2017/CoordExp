---
doc_id: progress.diagnostics.binding_mechanism_phase3_audit_adjusted_findings
layer: progress
doc_type: diagnostic-findings
status: branch-provenance
domain: research-history
summary: Audit-adjusted scaled Phase 3 evidence for the checkpoint-928 binding-template ablation, including P0 gates, held-out nulls, candidate-only identity posterior, and behavioral-readout boundary.
tags: [progress, diagnostics, autoregressive-binding, audit-adjusted, repetition-penalty, nulls, behavioral-patch]
updated: 2026-06-18
branch: codex/autoregressive-binding-template-study
---

# Binding Mechanism Phase 3 Audit-Adjusted Findings

## Scope

This note records the first audit-adjusted scaled evidence bundle for the
checkpoint-928 `desc_first` versus `geometry_first`
`compact_object_box_closed` binding-template study.

Executable config:

```text
configs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200.yaml
```

Evidence scope:

- COCO `val200` panel, both checkpoint families, existing free-rollout
  artifacts with `temperature=0.0`, `repetition_penalty=1.10`, and
  `max_new_tokens=3084`.
- Surface rows over 200 images per family.
- Teacher-forced GPU replay over parsed predictions, sharded across 8 GPUs.
- Deterministic P0 gates, candidate ledger, coverage polarity, held-out null
  leaderboard, same-class identity posterior, and gated behavioral-patch plan.

This is not a causal patch success note. The run did not produce realized
before/after behavioral rows, and no short continuation changed next-object
behavior.

## Audit Adjustments

The Phase 3 run incorporated the audit objections directly:

- Repetition-penalty correction was applied before interpreting duplicate
  coordinate-token support.
- Previous-anchor evidence was not treated as a duplicate label; the run
  separated immediate previous, nearest previous same-desc, and any-prior
  distances, then used upstream next-step labels for predictive claims.
- `desc_first` and `geometry_first` were kept family-specific at their
  selection events:
  - `desc_first`: `pre_desc`, `desc_end`
  - `geometry_first`: `pre_box_start`, `pre_x1`
- Sequential-greedy candidate coverage was not described as eval recall.
- Behavioral patching was gated behind P0 and held-out/null evidence.

Two execution adjustments were needed and are part of the evidence boundary:

1. The exact null-leaderboard command failed on strict join keys because
   terminal coverage rows have no possible `next_step_*` upstream label, and
   all-object upstream labels include non-focus rows that were never replayed
   as coord coverage. I materialized explicit joinable inputs:

   ```text
   /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase3_audit_adjusted/coverage_polarity_rows_upstream_joinable.jsonl
   /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase3_audit_adjusted/upstream_onset_label_rows_for_coverage.jsonl
   /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase3_audit_adjusted/repetition_penalty_gate_rows_for_coverage.jsonl
   /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase3_audit_adjusted/null_leaderboard_joinable_inputs_summary.json
   ```

   This kept `1732` unique object-step keys and `6928` coverage rows, excluding
   `366` terminal/unlabeled coverage keys and `1033` label-only noncoverage
   keys.

2. The exact identity-posterior command failed on unmatched/no-target candidate
   groups, because same-class identity posterior requires exactly one target
   candidate. I materialized matched-target-only inputs:

   ```text
   /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase3_audit_adjusted/candidate_step_rows_matched_target_only.jsonl
   /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase3_audit_adjusted/coverage_polarity_rows_matched_target_only.jsonl
   /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase3_audit_adjusted/identity_posterior_matched_target_inputs_summary.json
   ```

   This kept `4256` candidate groups with exactly one target and excluded
   `2066` no-target or non-single-target groups. Excluded coverage rows were
   `1820` `unmatched` rows and `92` `duplicate_iou70` rows.

These adjustments are not model-result failures; they define the valid scoring
domain for next-step prediction and candidate-only identity posterior.

## Artifact Roots

Primary adjusted root:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase3_audit_adjusted
```

Scaled probe root:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase3_audit_adjusted/scaled_probe_val200
```

Key files:

```text
split_manifest.json
gt_candidates.json
phase3_probe_config.yaml
scaled_probe_val200/probe_manifest.json
scaled_probe_val200/replay_merge_summary.json
scaled_probe_val200/probe_report_summary.json
repetition_penalty_gate_summary.json
anchor_circularity_summary.json
candidate_ledger_summary.json
coverage_polarity_summary.json
null_leaderboard.json
identity_posterior_summary.json
behavioral_patch/behavioral_patch_plan.json
```

Replay completed with `8` shards seen and no shard errors:

```json
{
  "attention_rows": 75528,
  "coord_logit_rows": 8392,
  "role_rows": 226584,
  "shard_errors": [],
  "shards_seen": 8
}
```

Scaled probe report:

| family | cases | parseable cases | predicted objects | duplicate objects | coord rows | attention rows | mean pre-x1 emitted rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `desc_first` | 200 | 197 | 1458 | 53 | 4052 | 36468 | 2.406 |
| `geometry_first` | 200 | 199 | 1703 | 69 | 4340 | 39060 | 2.460 |

## Split Manifest

The adjusted manifest preserved the reserved split:

| split | images |
| --- | ---: |
| `discovery` | 38 |
| `reserve` | 24 |
| `val200_remainder` | 138 |

The split manifest is:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase3_audit_adjusted/split_manifest.json
```

## Repetition-Penalty Gate

Artifact:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase3_audit_adjusted/repetition_penalty_gate_summary.json
```

Rows: `2098`.

The P0 repetition-penalty gate is mixed and does not support the old raw
`p_emitted` story:

| contrast | raw p gap | penalty p gap | p gap survives? | raw rank gap | penalty rank gap | rank gap survives? |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| duplicate minus new | +0.0162 | -0.0266 | false | +0.8160 | +0.3246 | true |
| duplicate minus non-duplicate | +0.0206 | -0.0232 | false | +0.6799 | +0.2531 | true |

Interpretation: repetition-penalty correction collapses the raw duplicate
`p_emitted` advantage, but a weaker duplicate rank gap remains. This is a
reason to demote the strongest version of the duplicate-probability hypothesis
and hold only the narrower rank/selection version for further observation.

## Anchor Circularity Gate

Artifacts:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase3_audit_adjusted/anchor_circularity_summary.json
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase3_audit_adjusted/upstream_onset_label_rows.jsonl
```

Anchor rows: `3161`. Upstream next-step label rows: `2765`.

Upstream label positives:

| label | positives |
| --- | ---: |
| `next_step_duplicate_onset` | 221 |
| `next_step_unmatched_onset` | 995 |

Anchor-distance summaries:

| focus scope | rows | immediate previous x1 mean | nearest same-desc x1 mean | any-prior min x1 mean |
| --- | ---: | ---: | ---: | ---: |
| all rows | 3161 | 216.790 | 111.276 | 96.592 |
| focus rows | 2098 | 252.955 | 149.168 | 134.786 |

Interpretation: the anchor rows avoid the circular `duplicate_iou70` label, but
the predictive claim must come from the upstream/null leaderboard rather than
from the distance summary itself.

## Candidate Ledger Counts

Artifacts:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase3_audit_adjusted/candidate_step_rows.jsonl
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase3_audit_adjusted/candidate_ledger_summary.json
```

Candidate rows: `44411`.

| split | candidate rows |
| --- | ---: |
| `discovery` | 33263 |
| `reserve` | 5452 |
| `val200_remainder` | 5696 |

| family | candidate rows |
| --- | ---: |
| `desc_first` | 20368 |
| `geometry_first` | 24043 |

| prediction kind | candidate rows |
| --- | ---: |
| `new_gt` | 25782 |
| `unmatched` | 15257 |
| `repeated_gt` | 1701 |
| `duplicate_iou70` | 1671 |

Candidate roles:

| role | rows |
| --- | ---: |
| `remaining_gt` | 28356 |
| `previously_covered_gt` | 14049 |
| `target_gt` | 2006 |

## Penalty-Aware Coverage Polarity

Artifacts:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase3_audit_adjusted/coverage_polarity_rows.jsonl
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase3_audit_adjusted/coverage_polarity_summary.json
```

Coverage rows: `8392`, split evenly between `raw` and `penalty_adjusted`
probability modes.

| probability mode | target GT mass | previous-anchor mass | remaining-GT mass | unassigned top-k mass | fragmentation |
| --- | ---: | ---: | ---: | ---: | ---: |
| `raw` | 0.1893 | 0.0179 | 0.0305 | 0.1477 | 0.4044 |
| `penalty_adjusted` | 0.3533 | 0.0461 | 0.0992 | 0.5015 | 0.4003 |

Best-role counts:

| best role | rows |
| --- | ---: |
| `target_gt` | 3216 |
| `unassigned_topk` | 4020 |
| `remaining_gt` | 762 |
| `previous_anchor` | 394 |

Interpretation: penalty adjustment increases mass assigned to candidate buckets,
including target, remaining-GT, and previous-anchor buckets, but many top-k bins
remain unassigned. This is a candidate-field readout, not a causal repair.

## Held-Out Null Leaderboard

Artifacts:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase3_audit_adjusted/null_leaderboard.json
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase3_audit_adjusted/null_leaderboard_summary.md
```

The leaderboard was run on the joinable next-step prediction domain described
above.

Joined rows: `6928`.

| split | joined rows |
| --- | ---: |
| `discovery` | 2756 |
| `reserve` | 1344 |
| `val200_remainder` | 2828 |

| family | joined rows |
| --- | ---: |
| `desc_first` | 3324 |
| `geometry_first` | 3604 |

For `next_step_duplicate_onset`, the top reserve and `val200_remainder` locked
threshold scores were dominated by simple baselines or weak evidence:

| split | top signal | family | held-out accuracy | held-out AUC | positives / n |
| --- | --- | --- | ---: | ---: | ---: |
| `reserve` | `object_idx` | `desc_first` | 0.9477 | 0.8496 | 8 / 153 |
| `reserve` | `same_class_gt_count` | `desc_first` | 0.9477 | 0.8379 | 16 / 306 |
| `val200_remainder` | `object_idx` | `desc_first` | 0.9427 | 0.6494 | 20 / 349 |
| `val200_remainder` | `remaining_gt_mass` raw | `desc_first` | 0.9427 | 0.5840 | 40 / 698 |

For `next_step_unmatched_onset`, held-out accuracy was also led by simple
signals and class imbalance:

| split | top signal | family | held-out accuracy | held-out AUC | positives / n |
| --- | --- | --- | ---: | ---: | ---: |
| `reserve` | `rank_delta` | `desc_first` | 0.7852 | 0.3942 | 33 / 149 |
| `reserve` | `remaining_gt_mass` raw | `desc_first` | 0.7712 | 0.5477 | 70 / 306 |
| `val200_remainder` | `object_idx` | `desc_first` | 0.8481 | 0.6178 | 53 / 349 |
| `val200_remainder` | `remaining_gt_mass` raw | `desc_first` | 0.8481 | 0.4724 | 106 / 698 |

Interpretation: held-out null evidence does not promote the mechanism. It
shows that duplicate/unmatched onset labels are strongly predictable from
simple phase/index/count structure in at least the `desc_first` family, and the
candidate-mass signals do not cleanly beat those nulls in a way that satisfies
the promotion gate.

## Same-Class Identity Posterior

Artifacts:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase3_audit_adjusted/identity_posterior_rows.jsonl
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase3_audit_adjusted/identity_posterior_summary.json
```

This posterior is candidate-only and matched-target-only. It is not a realized
behavioral before/after artifact.

Rows: `6480`.

| split | rows |
| --- | ---: |
| `discovery` | 1948 |
| `reserve` | 1092 |
| `val200_remainder` | 3440 |

| family | rows |
| --- | ---: |
| `desc_first` | 3196 |
| `geometry_first` | 3284 |

Mean bucket target margins:

| family | split | raw margin | penalty-adjusted margin |
| --- | --- | ---: | ---: |
| `desc_first` | `discovery` | 0.1723 | 0.3258 |
| `desc_first` | `reserve` | 0.2597 | 0.4660 |
| `desc_first` | `val200_remainder` | 0.2927 | 0.4687 |
| `geometry_first` | `discovery` | 0.1109 | 0.2493 |
| `geometry_first` | `reserve` | 0.1953 | 0.3697 |
| `geometry_first` | `val200_remainder` | 0.2563 | 0.4355 |

Interpretation: within matched-target candidate groups, penalty adjustment
increases target identity bucket margins in both families and all splits. This
is useful candidate-selection evidence, but it cannot establish behavioral
causality or cover unmatched/no-target rows.

## Behavioral Patch Smoke

Artifacts:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase3_audit_adjusted/behavioral_patch/behavioral_patch_plan.json
```

Plan result:

```json
{
  "gate_passes": false,
  "required_gates": [
    "rep_penalty_survives",
    "anchor_circularity_cleared",
    "held_out_beats_nulls"
  ]
}
```

The plan selected `4` candidate-only cases and annotated each with:

```text
short_rollout_ready: false
short_rollout_block_reason: candidate-only patch case has no realized before/after behavior rows; provide realized before/after behavior rows before short-rollout
```

The short-rollout command refused before model load:

```text
behavioral patch short-rollout requires explicit behavioral patch gate metadata with boolean keys ['rep_penalty_survives', 'anchor_circularity_cleared', 'held_out_beats_nulls']; accepted fields are 'gates', ['promotion_gate', 'behavioral_patch_gate', 'gate_summary']
```

This is the correct Task 9 boundary. `identity_posterior_rows.jsonl` is a
candidate-only selection artifact. There is no separate realized before/after
case artifact, and the gate metadata did not pass.

## Promotion Decision

| decision | condition | Task 9 read |
| --- | --- | --- |
| `promote_to_scaled_behavioral_causal` | P0 gates survive, upstream signal beats nulls on reserve or val200, within-family signs match, short continuation changes next-object behavior, controls are null | Not met. P0 repetition evidence is mixed, held-out nulls are not beaten cleanly, and no short continuation ran. |
| `hold_for_more_observation` | P0 gates survive but held-out effect is one-family-only, weak, or missing behavioral movement | Best current state. The rank component and matched-target identity margins survive as observational signals, but the evidence is weak/mixed and has no behavioral movement. |
| `demote_current_hypothesis` | repetition-penalty correction collapses the duplicate entropy/rank gap, anchor signal is circular, or held-out nulls win | Demote the strong raw duplicate-probability and previous-anchor causal versions. Repetition penalty flips the raw `p_emitted` gap, and held-out leaderboard results are dominated by simple null/phase signals. |

Promotion decision: **hold for more observation while demoting the strong
raw-probability/previous-anchor causal claim**. Do not launch scaled behavioral
causal patching from this bundle alone.

## Boundaries

- Replay is teacher-forced on rendered parsed predictions, not free
  continuation.
- Parse-failure cases contribute surface evidence but no rendered GPU replay.
- Repetition-penalty probabilities are top-bin approximations, not full-vocab
  exact probabilities.
- The null leaderboard required explicit joinable next-step prediction inputs;
  terminal objects have no next-step label.
- The same-class identity posterior excludes unmatched/no-target groups and is
  candidate-only.
- Behavioral patch planning selected candidate-only cases and did not produce
  realized before/after rows.
- No model perturbation or short continuation result exists in this bundle.

## Next Branch

Before any scaled behavioral-causal claim, create a small branch that produces
the missing bridge artifact:

```text
realized_before_after_behavior_rows.jsonl
```

The next branch should:

- derive explicit boolean behavioral gate metadata from the Phase 3 summaries;
- select only cases that pass the gate and have matched target candidates;
- materialize realized before/after patch cases before invoking
  `--stage short-rollout`;
- include controls for noop, wrong role, wrong image, post-commit, shuffled
  candidate labels, norm-matched random, and repetition-penalty on/off;
- report any continuation change as next-object behavior, not candidate mass.

Until that bridge exists, the current Phase 3 bundle should be used as
audit-adjusted observational evidence and a negative/pause signal for causal
promotion.
