---
doc_id: docs.training.stage1_et_rmp_ce
layer: docs
doc_type: implementation-export
status: experimental
domain: training
summary: Factual export of the implemented Stage-1 Entry-Trie Recursive Multi-Positive CE objective, config surface, verification files, and current run artifacts.
tags: [training, stage1, set-continuation, et-rmp-ce, full-suffix]
updated: 2026-04-29
---

# Stage-1 ET-RMP-CE Implementation Export

This page records the implemented Stage-1 **Entry-Trie Recursive
Multi-Positive CE** objective, abbreviated **ET-RMP-CE**. It is a factual
implementation and artifact export only.

## Current Status

- Branch/checkpoint state inspected from `/data/CoordExp` on `2026-04-29`.
- The implementation is present in the repo root checkout on `main`.
- The current production profile is
  `configs/stage1/set_continuation/production.yaml`.
- That checked-in profile currently targets the support-mass experiment
  (`branch_support_weight: 2.0`, `branch_balance_weight: 1.0`,
  `artifact_subdir: ..._support2_bsz32_v1`).
- The recorded step-100/step-200 artifacts below are from the earlier
  pre-support-mass ET-RMP `v1` run, before the checked-in profile was retargeted
  to `support2_bsz32_v1`.
- The objective mode is
  `custom.stage1_set_continuation.objective.mode: entry_trie_rmp_ce`.
- The companion full-suffix hard-CE mode is
  `custom.stage1_set_continuation.objective.mode: full_suffix_ce`.
- The Stage-1 set-continuation trainer remains the runtime owner:
  `custom.trainer_variant: stage1_set_continuation`.

## Objective Contract

ET-RMP-CE trains one teacher-forced full remaining suffix for each sampled
set-continuation state.

For a full object multiset `O`:

1. Sample a prefix subset `S0`.
2. Let `R0 = O - S0`.
3. Resolve one suffix order over `R0`.
4. Render one continuation row containing all remaining object entries,
   inter-object separators, the final global close, and the assistant stop/EOS
   target.
5. For each recursive state, build an entry trie over the serialized object
   entries for the currently remaining objects.
6. Inside object entries, apply object-uniform soft CE at trie nodes with
   multiple child tokens and hard CE at trie nodes with one child token.
7. Outside object entries, apply ordinary hard CE for schema opener tokens,
   comma/separator tokens, final global close, and assistant stop/EOS tokens.

The entry trie covers serialized object entries only. It excludes:

- inter-object comma tokens,
- final global close tokens,
- EOS or chat-template stop tokens,
- schema opener tokens.

The main ET-RMP objective uses the full vocabulary probability space for both
hard CE and entry-trie branch supervision. Coordinate tokens are handled by the
same trie rule as text and structural entry tokens.

At a trie branch node, the target for each valid next token is object-uniform:

```text
q(v) = object_count(child_v) / object_count(current_node)
```

The branch-node loss is decomposed into valid support and valid balance terms:

```text
P_valid = sum_{v in V} p_theta(v | context)
L_valid_support = -log(P_valid)
L_valid_balance = - sum_{v in V} q(v) * log(p_theta(v | context) / P_valid)
L_branch = branch_support_weight * L_valid_support
         + branch_balance_weight * L_valid_balance
```

`branch_support_weight=1.0` and `branch_balance_weight=1.0` recover the old
object-uniform full-vocabulary soft CE. The checked-in experimental profile
uses `2.0/1.0` to test whether raising valid-child support mass reduces
under-prediction without changing decoding.

Exact duplicate serialized object entries are represented as multiplicity under
the same trie path. The implementation does not add artificial divergence for
exact duplicate entries.

## Code Map

- `src/config/schema.py`
  - Defines `Stage1SetContinuationObjectiveConfig`.
  - Accepted objective modes are `candidate_balanced`, `full_suffix_ce`, and
    `entry_trie_rmp_ce`.
  - Accepted suffix orders are `random` and `dataset`.
  - `branch_support_weight` and `branch_balance_weight` configure the
    entry-trie branch decomposition.
  - Defines `Stage1SetContinuationSubsetSamplingConfig` with prefix mixture
    fields `empty_prefix_ratio`, `random_subset_ratio`,
    `leave_one_out_ratio`, `full_prefix_ratio`, and `prefix_order`.

- `src/sft.py`
  - Selects `Stage1SetContinuationTrainer` for
    `custom.trainer_variant: stage1_set_continuation`.
  - Emits effective-runtime provenance for the set-continuation objective,
    train-forward runtime, packing rejection, metric schema, prefix sampling,
    and effective coordinate-slot scoring.
  - Records `full_vocab_recursive_suffix` for
    `full_suffix_ce` and `entry_trie_rmp_ce`.

- `src/trainers/stage1_set_continuation/entry_trie.py`
  - Defines `EntryTrieCandidate`, `EntryTrieTarget`,
    `EntryTrieTargetStep`, and trie construction helpers.
  - `build_entry_trie_target_steps(...)` builds the entry-trie target sequence
    for a teacher-forced object-entry token path.
  - Each target step stores the teacher token id, active object count, valid
    child token targets, object-count probabilities, and token type.

- `src/trainers/stage1_set_continuation/full_suffix.py`
  - Defines full-suffix row and tensor dataclasses.
  - `resolve_full_suffix_order(...)` resolves random or dataset suffix order.
  - `encode_full_suffix_branch(...)` renders and encodes one full-suffix row.
  - `compute_full_suffix_loss(...)` computes hard or entry-trie soft CE loss
    from full-suffix target steps.
  - `score_full_suffix_batch_retained(...)` scores full-suffix rows through the
    retained smart-batch path.

- `src/trainers/stage1_set_continuation/trainer.py`
  - Routes `full_suffix_ce` and `entry_trie_rmp_ce` through
    `_process_full_suffix_batch(...)`.
  - Uses `_score_full_suffix_rows_smart_batched(...)` when
    `train_forward.branch_runtime.mode: smart_batched_exact`.
  - Keeps the existing candidate-balanced path as the default for
    `candidate_balanced`.

- `src/trainers/stage1_set_continuation/metrics.py`
  - Adds compact ET-RMP metric keys while preserving the set-continuation metric
    schema surface.

## Config Profiles

Production ET-RMP profile:

```text
configs/stage1/set_continuation/production.yaml
```

Smoke profiles present in the repo:

```text
configs/stage1/set_continuation/smoke/rmp_ce_tiny.yaml
configs/stage1/set_continuation/smoke/rmp_ce_memstress.yaml
configs/stage1/set_continuation/smoke/worktree_rmp_ce_tiny.yaml
```

`rmp_ce_tiny.yaml` is the fast procedural smoke. `rmp_ce_memstress.yaml`
keeps the same objective, decode controls, geometry, and `packing=false`
contract, but raises sample limits and per-device batch size to exercise all 8
GPUs before a production run; it is a throughput/memory smoke, not a comparable
benchmark result.

The production ET-RMP profile sets:

```yaml
custom:
  stage1_set_continuation:
    objective:
      mode: entry_trie_rmp_ce
      suffix_order: random
      branch_support_weight: 2.0
      branch_balance_weight: 1.0
    structural_close:
      close_start_suppression_weight: 0.0
      final_schema_close_weight: 0.0
      json_structural_weight: 0.0
      annotation_completeness_weight:
        enabled: false
    bidirectional_token_gate:
      enabled: false
    positive_evidence_margin:
      objective: disabled
    train_forward:
      branch_runtime:
        mode: smart_batched_exact
      branch_batching:
        max_branch_rows: 32
        max_branch_tokens: 65536
      logits:
        mode: supervised_suffix
      ddp_sync:
        candidate_padding: none
      budget_policy:
        enabled: false
```

The production profile also scales local GPU memory use relative to the older
candidate-balanced production profile by using `per_device_train_batch_size: 32`
and `gradient_accumulation_steps: 1` (`effective_batch_size: 256`). Packing
remains disabled because set-continuation rows own their metadata and suffix
alignment.

The effective runtime for the recorded production run inherited this prefix
mixture:

```yaml
subset_sampling:
  empty_prefix_ratio: 0.3
  random_subset_ratio: 0.45
  leave_one_out_ratio: 0.2
  full_prefix_ratio: 0.05
  prefix_order: random
```

## Smart-Batch Runtime

ET-RMP uses the existing `smart_batched_exact` row scheduler. The row unit is
one full-suffix row per sample:

```text
prefix + full remaining suffix + final close/EOS
```

The recorded production effective runtime used:

```yaml
branch_runtime:
  mode: smart_batched_exact
  checkpoint_use_reentrant: false
  preserve_rng_state: true
logits:
  mode: supervised_suffix
branch_batching:
  enabled: true
  strategy: ms_swift_constant_volume_buckets
  max_branch_rows: 8
  min_fill_ratio: 0.7
  padding_waste_warn_fraction: 0.4
ddp_sync:
  candidate_padding: none
budget_policy:
  enabled: false
```

Rows remain independent padded rows. This implementation does not introduce
true packed-varlen multimodal attention, branch-to-branch attention, or KV
prefix sharing.

## Metrics

Trainer-native ET-RMP metrics include:

- `rmp/branch_nodes`
- `rmp/branch_nodes_desc_text`
- `rmp/branch_nodes_coord`
- `rmp/branch_nodes_structural`
- `rmp/branch_nodes_other`
- `rmp/valid_children_mean`
- `rmp/target_entropy_mean`
- `rmp/valid_child_mass_mean`
- `rmp/valid_child_mass_min`
- `rmp/valid_child_mass_p10`
- `rmp/valid_child_mass_p50`
- `rmp/valid_child_mass_p90`
- `rmp/valid_child_mass_desc_text`
- `rmp/valid_child_mass_coord`
- `rmp/valid_child_mass_structural`
- `rmp/valid_child_mass_other`
- `rmp/teacher_branch_top1_acc`
- `rmp/valid_child_top1_acc`
- `loss/rmp`
- `loss/rmp_branch_support`
- `loss/rmp_branch_balance`
- `loss/rmp_branch_total`
- `loss/rmp_branch_ce`
- `loss/rmp_unique_ce`
- `loss/rmp_coord_branch_ce`
- `loss/rmp_desc_text_branch_ce`
- `loss/rmp_boundary_ce`
- `loss/rmp_close_ce`
- `loss/rmp_eos_ce`
- `rmp/gt_count_ge7_samples`

Compatibility metrics from the set-continuation surface remain present. For
ET-RMP, selected candidate scoring metrics such as `mp/num_candidates_scored`
are emitted as `0.0` in the recorded run because this objective path does not
score independent candidate chunks.

## Verification Files

Focused ET-RMP test files:

- `tests/test_stage1_set_continuation_entry_trie.py`
- `tests/test_stage1_set_continuation_full_suffix.py`

Adjacent config, metric, and trainer smoke tests:

- `tests/test_stage1_set_continuation_benchmark_profiles.py`
- `tests/test_stage1_set_continuation_config.py`
- `tests/test_stage1_set_continuation_metric_keys.py`
- `tests/test_stage1_set_continuation_trainer_smoke.py`

Broader Stage-1 set-continuation tests remain under:

```text
tests/test_stage1_set_continuation_*.py
```

## Recorded Baseline Production Run

This section records the earlier pre-support-mass ET-RMP `v1` run. Treat it as
the baseline artifact export for comparison, not as evidence from the current
`support2_bsz32_v1` production config.

Run directory:

```text
output_remote/stage1_2b/set_continuation/coco1024_sota1332_setcont_et_rmp_ce_v1/setcont-coco1024-sota1332-et-rmp-ce-v1/v0-20260429-022918
```

Wrapper config recorded for launch:

```text
temp/et_rmp_smoke_configs/worktree_rmp_ce_production.yaml
```

The recorded run used:

- `trainer_variant: stage1_set_continuation`
- `objective.mode: entry_trie_rmp_ce`
- `objective.suffix_order: random`
- default-equivalent branch objective weights
  (`branch_support_weight=1.0`, `branch_balance_weight=1.0`)
- `train_forward.branch_runtime.mode: smart_batched_exact`
- `train_forward.logits.mode: supervised_suffix`
- `train_forward.ddp_sync.candidate_padding: none`
- `train_forward.budget_policy.enabled: false`
- eval scope: `val200`
- eval decoding: greedy `temperature=0.0`, `top_p=1.0`,
  `repetition_penalty=1.1`
- eval `max_new_tokens: 3084`

Eval artifact directories present:

```text
eval_detection/step_0000100
eval_detection/step_0000200
```

### Training Log Snapshots

| Metric | Step 100 | Step 200 |
|---|---:|---:|
| `loss/rmp` | `0.52594686` | `0.51878185` |
| `loss/rmp_unique_ce` | `0.51611834` | `0.51180358` |
| `loss/rmp_branch_ce` | `1.45096664` | `1.41881638` |
| `mp/num_prefix_objects` | `3.378125` | `3.14765625` |
| `mp/num_remaining_objects` | `4.03125` | `4.09921875` |
| `mp/selected_mode_empty_prefix` | `0.290625` | `0.31328125` |
| `mp/selected_mode_random_subset` | `0.42890625` | `0.39296875` |
| `mp/selected_mode_leave_one_out` | `0.22109375` | `0.23515625` |
| `mp/selected_mode_full_prefix` | `0.059375` | `0.05859375` |
| `rmp/gt_count_ge7_samples` | `0.39453125` | `0.38046875` |
| `rmp/branch_nodes` | `4.80546875` | `4.9171875` |
| `rmp/valid_child_mass_mean` | `0.30491047` | `0.29301984` |
| `loss/rmp_close_ce` | `0.00014564` | `0.00017377` |
| `loss/rmp_eos_ce` | `0.00001226` | `0.0000085` |
| `loss/rmp_boundary_ce` | `0.0000057` | `0.00000127` |

### Eval Metrics

Metrics are from `eval_detection/step_*/metrics.json` and callback logging.

| Metric, val200 | Step 100 | Step 200 |
|---|---:|---:|
| `bbox_AP` | `0.4102029919` | `0.4107351214` |
| `bbox_AP50` | `0.5479411901` | `0.5443579103` |
| `bbox_AP75` | `0.4280731744` | `0.4284220718` |
| `bbox_APs` | `0.1059542506` | `0.0804849697` |
| `bbox_APm` | `0.1864009830` | `0.2044108638` |
| `bbox_APl` | `0.5329763990` | `0.5298760446` |
| `bbox_AR100` | `0.4589831200` | `0.4598827838` |
| `f1ish@0.50_pred_total` | `999` | `908` |
| `f1ish@0.50_pred_eval` | `955` | `876` |
| `f1ish@0.50_pred_ignored` | `44` | `32` |
| `f1ish@0.50_tp_full` | `779` | `751` |
| `f1ish@0.50_fp_full` | `176` | `125` |
| `f1ish@0.50_fn_full` | `665` | `693` |
| `f1ish@0.50_precision_full_micro` | `0.8157068063` | `0.8573059361` |
| `f1ish@0.50_recall_full_micro` | `0.5394736842` | `0.5200831025` |
| `f1ish@0.50_f1_full_micro` | `0.6494372655` | `0.6474137931` |
| `f1ish@0.30_f1_full_micro` | `0.6894539391` | `0.6818965517` |
| `empty_pred` | `0` | `0` |
| `invalid_json` | `0` | `0` |
| callback `eval_det_parse_valid_rate` | `1.0` | `1.0` |

### Count Buckets

Count-bucket values were computed from `eval_detection/step_*/per_image.json`.

| Bucket | Step | Images | GT | Pred | Pred/GT | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `0-3` | 100 | `84` | `171` | `160` | `0.936` | `138` | `12` | `33` | `0.920` | `0.807` | `0.860` |
| `0-3` | 200 | `84` | `171` | `158` | `0.924` | `138` | `12` | `33` | `0.920` | `0.807` | `0.860` |
| `4-6` | 100 | `36` | `172` | `138` | `0.802` | `113` | `13` | `59` | `0.897` | `0.657` | `0.758` |
| `4-6` | 200 | `36` | `172` | `132` | `0.767` | `114` | `11` | `58` | `0.912` | `0.663` | `0.768` |
| `7-10` | 100 | `31` | `248` | `167` | `0.673` | `137` | `24` | `111` | `0.851` | `0.552` | `0.670` |
| `7-10` | 200 | `31` | `248` | `156` | `0.629` | `140` | `12` | `108` | `0.921` | `0.565` | `0.700` |
| `11+` | 100 | `49` | `853` | `534` | `0.626` | `391` | `127` | `462` | `0.755` | `0.458` | `0.570` |
| `11+` | 200 | `49` | `853` | `462` | `0.542` | `359` | `90` | `494` | `0.800` | `0.421` | `0.551` |
| `7+` | 100 | `80` | `1101` | `701` | `0.637` | `528` | `151` | `573` | `0.778` | `0.480` | `0.593` |
| `7+` | 200 | `80` | `1101` | `618` | `0.561` | `499` | `102` | `602` | `0.830` | `0.453` | `0.586` |

## Reference Pointers

- Design note:
  `docs/superpowers/specs/2026-04-28-stage1-et-rmp-ce-design.md`
- Implementation plan:
  `docs/superpowers/plans/2026-04-28-stage1-et-rmp-ce.md`
- Stage-1 objective overview:
  `docs/training/STAGE1_OBJECTIVE.md`
- Metric reference:
  `docs/training/METRICS.md`
