# Stage-1 Set-Continuation Training Report

Date: 2026-04-28

Scope: factual summary of the current Stage-1 set-continuation training pipeline, loss mechanisms, evaluation procedure, and observed symptoms across recent runs under `/data/CoordExp`.

This report intentionally avoids unverified causal hypotheses. It records what the code, configs, logs, metrics, and rollout artifacts currently show.

## 1. Reference Checkpoint And Evaluation Scope

The production set-continuation experiments continue from:

```text
output_remote/stage1_2b/coco_bbox_max60-hard_ce_soft_ce_w1_gate/epoch_4-from-base-2B/v0-20260227-050057/checkpoint-1332-merged-full
```

The current production config is:

```text
/data/CoordExp/configs/stage1/set_continuation/production.yaml
```

The training-time detection callback evaluates on `custom.val_sample_limit: 200` / `custom.eval_detection.limit: 200`. The output surface is coord-token `xyxy`, `score_mode: confidence_postop`, `confidence_method: bbox_logprob_confidence_exp`, strict parsing, greedy decoding with `temperature: 0.0`, `top_p: 1.0`, and `max_new_tokens: 3084`.

Current standalone reference artifacts for the starting checkpoint on `val200` / `coco_real` show:

| Reference artifact | Decode RP | bbox AP | AP50 | Loc recall@0.50 | Loc precision@0.50 | Pred total |
|---|---:|---:|---:|---:|---:|---:|
| `output/infer/coco1024_val200_compare_coordtoken_ckpt1332_20260423T080326Z/eval_coco_real/metrics.json` | artifact default | 0.4584 | 0.6307 | 0.6614 | 0.7561 | 1315 |
| `output/infer/coco1024_val200_compare_coordtoken_ckpt1332_rp1p00_20260423T095845Z/eval_coco_real/metrics.json` | 1.00 | 0.4532 | 0.6177 | 0.6724 | 0.6530 | 1518 |
| `output/infer/coco1024_val200_compare_coordtoken_ckpt1332_rp1p10_20260423T091701Z/eval_coco_real/metrics.json` | 1.10 | 0.4419 | 0.6064 | 0.6420 | 0.7699 | 1285 |
| `output/infer/coco1024_val200_lvis_proxy_mixed_objective_sota/eval_coco_real/metrics.json` | artifact default | 0.4137 | 0.5910 | 0.6530 | 0.7526 | 1300 |

Historical note: earlier discussion referred to an approximately `0.38` result. Memory records that this value came from a 200-sample mixed-objective Stage-1 benchmark and should not be conflated with full-val results or with every coord-token reference surface.

## 2. Current Training Pipeline

The active production path is `custom.trainer_variant: stage1_set_continuation`.

The trainer does not perform ordinary fixed-order full-output SFT. For each sample it:

1. Reads the serialized target object set from `assistant_payload.objects`.
2. Samples a prefix subset and remaining object set.
3. Scores candidate continuations from the remaining GT objects.
4. Optimizes a continuation objective over the selected candidate branches.
5. Runs train-time detection eval every `eval_steps: 100`.

The active production subset mixture is:

```yaml
empty_prefix_ratio: 0.30
random_subset_ratio: 0.45
leave_one_out_ratio: 0.20
full_prefix_ratio: 0.05
prefix_order: random
```

Candidate selection in the production config is:

```yaml
candidates:
  mode: exact
  max_candidates: null
  tail_positive_count: 1
```

Runtime execution in the current production config uses:

```yaml
train_forward.branch_runtime.mode: smart_batched_exact
train_forward.logits.mode: supervised_suffix
train_forward.ddp_sync.candidate_padding: none
train_forward.budget_policy.enabled: true
exact_until.max_candidates: 8
fallback.max_candidates: 8
fallback.mode: approximate_uniform_subsample
```

The config keeps `training.packing: false` and `training.eval_packing: false`.

## 3. Branch Serialization And Masking

`EncodedSetContinuationBranch` carries separate masks for:

- `objective_label_mask`
- `candidate_entry_label_mask`
- `candidate_object_label_mask`
- `schema_open_label_mask`
- `json_structural_label_mask`
- `coord_label_mask`
- `non_coord_label_mask`
- `structural_close_start_mask`
- `structural_close_sequence_mask`

In `encode_set_continuation_branch`, empty-prefix candidate branches set `objective_start = 0`, so the generated schema opener is included in the optimized objective span. Non-empty-prefix candidate branches start the objective span at the end of the already-rendered prefix.

Candidate branch text is continuation-aware:

- If remaining GT objects still exist after the candidate, the candidate branch appends `, `.
- If the candidate exhausts the observed remaining GT set, the candidate branch appends the global CoordJSON close `]}`.

The branch encoder computes span masks from tokenizer offset overlap inside the real chat-template rendering. This was added to handle tokenizer-merged boundary tokens.

## 4. Loss Mechanisms

### 4.1 Candidate Full-Entry Scoring

For each candidate branch, `compute_candidate_full_entry_logprob` applies the same next-token shift as the model language modeling objective.

Within the candidate objective:

- Non-coordinate labels use full-vocabulary log probability.
- Coordinate labels use coord-vocabulary-normalized log probability over `coord_token_ids`.
- The score is the sum of coord and non-coord log probabilities over valid objective labels.
- `schema_score` and `json_structural_score` are recorded separately for metrics and auxiliary losses.

### 4.2 Candidate-Balanced Objective

With `positive_evidence_margin.objective: disabled`, the optimized candidate objective is:

```text
loss/candidate_balanced = mean(-score(candidate) / candidate_tokens(candidate))
```

The MP/logZ quantities remain diagnostic in code, but the optimized path uses candidate-balanced CE.

### 4.3 PEM / Threshold-Loss Path

The earlier `pem_close_suppress` runs used `positive_evidence_margin.objective: threshold_loss`. In that mode, `compute_mp_pem_losses` uses a thresholded loss over estimated `logZ`. The later candidate-balanced production profiles disable this path.

### 4.4 Structural Close Controls

The current trainer records and may add:

- `loss/anti_close_start`: when GT remains, penalizes the probability of the first global close token.
- `loss/weak_schema_close`: when no observed GT remains, applies weak close-sequence loss scaled by `final_schema_close_weight` and, when enabled, `annotation_completeness_weight`.
- `stop/p_close_start_when_remaining_exists`
- `stop/p_continue_start_when_remaining_exists`
- `stop/p_close_start_when_remaining_empty`

The current production config uses:

```yaml
close_start_suppression_weight: 0.05
final_schema_close_weight: 0.05
json_structural_weight: 0.05
annotation_completeness_weight.enabled: true
```

The current annotation-completeness weights are configured from the original-checkpoint `val200` rollout by treating localization FPs as likely unlabeled objects:

```yaml
1: 0.9500
3: 0.8306
6: 0.8306
10: 0.8306
20: 0.8068
1000000000: 0.7595
```

### 4.5 JSON Structural Auxiliary Loss

`loss/json_structural` is a weighted CE term over schema/key/punctuation/boundary tokens inside the scored continuation span. It excludes description payload text and coordinate values.

### 4.6 Bidirectional Token Gate

The current `bidirgate_warmup10` config enables:

```yaml
bidirectional_token_gate:
  enabled: true
  coord_gate_weight: 0.5
  text_gate_weight: 0.1
  temperature: 1.0
  scope: objective_tokens
```

The gate uses the same next-token shift and objective-label mask:

```text
p_coord(t) = sum softmax(logits(t) / T)[coord_token_ids]
loss/coord_gate = mean(-log(p_coord(t)) over objective coord slots)
loss/text_gate  = mean(-log(1 - p_coord(t)) over objective non-coord slots)
```

Special stop tokens are excluded from the gate; coord tokens are not excluded even when they are tokenizer special tokens.

## 5. Training-Time Eval Procedure

`Stage1DetectionEvalCallback.on_evaluate` receives the live trainer model, unwraps it, sets `engine.model = runtime_model`, switches it to eval mode, calls `engine.infer()`, then restores training mode. Rank 0 runs confidence post-op and detection evaluation.

The eval callback logs AP/F1 metrics plus generation hygiene metrics:

- `eval_det_empty_pred`
- `eval_det_parse_valid_rate`
- `eval_det_start_objects_wrapper_rate`
- `eval_det_start_bare_desc_rate`
- `eval_det_start_coord_first_rate`

The callback uses strict parsing. It does not loosen parsing to accept bare object streams.

## 6. Unit And Smoke Validation Surfaces

The current test surface includes explicit checks for:

- empty-prefix schema-opener scoring
- append-boundary and terminal-close-boundary masks
- tokenizer offset overlap for merged boundary tokens
- non-empty-prefix exclusion of already-generated prefix objects
- candidate-balanced objective instead of logZ/MP as the optimized loss
- JSON structural mask contents
- bidirectional gate token-type assignment and shift alignment
- tail-positive candidate preservation
- metric-key presence and absence

Relevant tests are under:

```text
/data/CoordExp/tests/test_stage1_set_continuation_preflight.py
/data/CoordExp/tests/test_stage1_set_continuation_loss.py
/data/CoordExp/tests/test_stage1_set_continuation_trainer_smoke.py
/data/CoordExp/tests/test_stage1_set_continuation_metric_keys.py
/data/CoordExp/tests/test_stage1_set_continuation_sampler.py
/data/CoordExp/tests/test_stage1_set_continuation_runtime_policy.py
```

Tiny smoke artifacts show parse-valid first evals:

| Smoke artifact | Eval step | bbox AP | Empty pred | Pred total |
|---|---:|---:|---:|---:|
| `output_remote/stage1_2b/set_continuation_smoke/schemafix_tiny/smoke-schemafix-tiny/v0-20260427-070353` | 2 | 0.4442 | 0 | 45 |
| `output_remote/stage1_2b/set_continuation_smoke/schemafix_tiny/smoke-schemafix-tiny/v1-20260427-071135` | 2 | 0.4503 | 0 | 39 |
| `output_remote/stage1_2b/set_continuation_smoke/bidirgate_tiny/smoke-bidirgate-tiny/v1-20260428-024028` | 2 | 0.4380 | 0 | 26 |

## 7. Run Configuration Summary

| Run family | Key objective/config facts | Eval RP | Epochs | Eval steps |
|---|---|---:|---:|---:|
| `pem_close_suppress/v2-20260426-114031` | PEM threshold-loss, no full-prefix ratio, no weak final close | 1.05 | 4 | 100 |
| `candbal_closefix/v0-20260426-175258` | Candidate-balanced, closefix boundary handling, weak close enabled | 1.05 | 4 | 100 |
| `candbal_boundaryfix/v0-20260427-021823` | Candidate-balanced, boundary fix, no schema-aware opener | 1.05 | 4 | 100 |
| `candbal_schemafix/v0-20260427-071728` | Schema-aware empty-prefix objective, candidate-balanced | 1.05 | 4 | 100 |
| `candbal_schemafix/v1-20260427-172240` | Schema-aware plus JSON structural loss and annotation completeness | 1.10 | 1 | 100 |
| `candbal_bidirgate_warmup10/v0-20260428-101109` | Schema-aware plus JSON structural loss, annotation completeness, bidirectional gate, warmup 0.1 | 1.10 | 1 | 100 |

## 8. Main Eval Metrics Across Runs

All rows below are `val200`, strict parse, confidence post-op, coord-token `xyxy`.

| Run | Step | RP | bbox AP | AP50 | Loc recall@0.50 | Loc precision@0.50 | Pred total | Empty pred |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `pem_close_suppress/v2` | 100 | 1.05 | 0.1677 | 0.1950 | 0.1288 | 0.9394 | 198 | 2 |
| `candbal_closefix/v0` | 100 | 1.05 | 0.1784 | 0.2000 | 0.1295 | 0.9167 | 204 | 0 |
| `candbal_closefix/v0` | 200 | 1.05 | 0.1235 | 0.1466 | 0.1253 | 0.9235 | 197 | 3 |
| `candbal_closefix/v0` | 300 | 1.05 | 0.1255 | 0.1504 | 0.1177 | 0.9091 | 188 | 12 |
| `candbal_boundaryfix/v0` | 100 | 1.05 | 0.2052 | 0.2894 | 0.3657 | 0.6219 | 861 | 104 |
| `candbal_schemafix/v0` | 100 | 1.05 | 0.4025 | 0.5524 | 0.5665 | 0.7119 | 1191 | 8 |
| `candbal_schemafix/v0` | 200 | 1.05 | 0.3140 | 0.4058 | 0.3109 | 0.5987 | 764 | 54 |
| `candbal_schemafix/v1` | 100 | 1.10 | 0.1990 | 0.2581 | 0.1683 | 0.5855 | 431 | 100 |
| `candbal_schemafix/v1` | 200 | 1.10 | 0.2115 | 0.2697 | 0.1807 | 0.9000 | 301 | 92 |
| `candbal_schemafix/v1` | 300 | 1.10 | 0.2389 | 0.2986 | 0.2022 | 0.6418 | 470 | 85 |
| `candbal_bidirgate_warmup10/v0` | 100 | 1.10 | 0.1927 | 0.2512 | 0.1787 | 0.3981 | 663 | 97 |

## 9. Training-Loss Dynamics

The long runs show decreasing training losses while eval metrics are low or later decline.

Representative examples:

| Run | Early loss | Later loss | Early candidate CE | Later candidate CE |
|---|---:|---:|---:|---:|
| `candbal_schemafix/v0` | step 1: 1.4277 | step 250: 0.5668 | step 1: 1.1685 | step 250: 0.5592 |
| `candbal_schemafix/v1` | step 1: 1.4565 | step 310: 0.5645 | step 1: 1.1668 | step 310: 0.5547 |
| `candbal_bidirgate_warmup10/v0` | step 1: 1.4778 | step 150: 0.5856 | step 1: 1.1668 | step 150: 0.5707 |
| `pem_close_suppress/v2` | step 1: 19.0751 | step 100: 12.7578 | N/A | N/A |

For `bidirgate_warmup10/v0`, gate metrics also move in the expected direction under teacher forcing:

- `loss/coord_gate`: step 1 `0.0460`, step 150 `0.0079`
- `gate/coord_slot_coord_mass_mean`: step 1 `0.9562`, step 150 `0.9921`
- `gate/text_slot_coord_mass_mean`: step 1 `9e-08`, step 150 `3.41e-06`

## 10. Rollout Hygiene And Decoding Symptoms

The table below is computed from `gt_vs_pred.jsonl` plus `pred_token_trace.jsonl`.

| Run | Step | Empty pred | Parse-valid proxy | Starts with objects wrapper | Starts bare object | Hit max tokens | Array-close then extra object | Extra key after array | Missing separator/post-close object | `<|endoftext|>` tail | Mean generated tokens | Empty rate for GT>=7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `pem_close_suppress/v2` | 100 | 2 | 0.990 | 200 | 0 | 0 | 0 | 0 | 0 | 195 | 30.0 | 0.025 |
| `candbal_closefix/v0` | 100 | 0 | 1.000 | 200 | 0 | 0 | 0 | 0 | 0 | 200 | 31.5 | 0.000 |
| `candbal_closefix/v0` | 300 | 11 | 0.945 | 200 | 0 | 0 | 0 | 0 | 0 | 192 | 30.0 | 0.100 |
| `candbal_boundaryfix/v0` | 100 | 104 | 0.480 | 97 | 103 | 48 | 0 | 0 | 12 | 143 | 976.4 | 0.350 |
| `candbal_schemafix/v0` | 100 | 8 | 0.960 | 200 | 0 | 17 | 0 | 0 | 5 | 143 | 548.3 | 0.087 |
| `candbal_schemafix/v0` | 200 | 54 | 0.730 | 200 | 0 | 76 | 32 | 0 | 22 | 137 | 1295.4 | 0.588 |
| `candbal_schemafix/v1` | 100 | 100 | 0.500 | 200 | 0 | 44 | 84 | 12 | 8 | 143 | 877.0 | 0.887 |
| `candbal_schemafix/v1` | 300 | 85 | 0.575 | 200 | 0 | 37 | 77 | 1 | 11 | 141 | 772.2 | 0.738 |
| `candbal_bidirgate_warmup10/v0` | 100 | 97 | 0.515 | 200 | 0 | 56 | 77 | 10 | 13 | 141 | 1027.2 | 0.863 |

Recorded malformed-output patterns include:

```text
{"objects": [... {"desc": "dining table", ... }],  {"desc": "vase", ... }]}
```

This starts with a valid objects wrapper but places another object after the `objects` array has already been closed.

Another recorded pattern is:

```text
{"objects": [... {"desc": "stop sign", ... }, {"desc": "truck", ... }], "car": {"desc": "truck", ... }]}
```

This adds an extra top-level key after the `objects` array.

The `candbal_boundaryfix/v0` step-100 artifact differs from later schema-aware runs: 103 of 200 traces start as bare object entries rather than with the `{"objects": ...}` wrapper.

## 11. Count-Bucket Symptoms

Prediction volume and empty parsed outputs vary strongly with the number of GT objects.

| Run | GT bucket | Rows | GT sum | Pred sum | Empty rate | Pred / GT |
|---|---|---:|---:|---:|---:|---:|
| `pem_close_suppress/v2 step100` | 1 | 19 | 19 | 19 | 0.000 | 1.000 |
| `pem_close_suppress/v2 step100` | 2-3 | 65 | 152 | 65 | 0.000 | 0.428 |
| `pem_close_suppress/v2 step100` | 4-6 | 36 | 172 | 36 | 0.000 | 0.209 |
| `pem_close_suppress/v2 step100` | 7-10 | 31 | 248 | 31 | 0.000 | 0.125 |
| `pem_close_suppress/v2 step100` | >=11 | 49 | 853 | 47 | 0.041 | 0.055 |
| `candbal_schemafix/v0 step100` | 1 | 19 | 19 | 23 | 0.000 | 1.211 |
| `candbal_schemafix/v0 step100` | 2-3 | 65 | 152 | 143 | 0.000 | 0.941 |
| `candbal_schemafix/v0 step100` | 4-6 | 36 | 172 | 130 | 0.028 | 0.756 |
| `candbal_schemafix/v0 step100` | 7-10 | 31 | 248 | 295 | 0.032 | 1.190 |
| `candbal_schemafix/v0 step100` | >=11 | 49 | 853 | 600 | 0.122 | 0.703 |
| `candbal_schemafix/v0 step200` | 7-10 | 31 | 248 | 80 | 0.516 | 0.323 |
| `candbal_schemafix/v0 step200` | >=11 | 49 | 853 | 419 | 0.633 | 0.491 |
| `candbal_bidirgate_warmup10 step100` | 7-10 | 31 | 248 | 33 | 0.839 | 0.133 |
| `candbal_bidirgate_warmup10 step100` | >=11 | 49 | 853 | 402 | 0.878 | 0.471 |

The early PEM and closefix runs are parse-valid but emit approximately one prediction per image in crowded buckets. The schemafix v0 step-100 artifact emits substantially more predictions in crowded buckets, while the same run has lower prediction volume and more empty parses by step 200. The later RP 1.10 schema/structure/gate runs have high empty rates in GT>=7 buckets.

## 12. Factual Symptom Summary

Observed across the artifact families:

1. Training losses decrease over 100-300 steps.
2. Evaluation AP is generally below the starting-checkpoint reference after 100-300 steps.
3. Early PEM/closefix variants have very low prediction counts and high precision/low recall.
4. Boundaryfix step 100 has a schema-start issue: 103 of 200 traces begin as bare object entries.
5. Schemafix v0 step 100 is the best set-continuation checkpoint in this group, with AP 0.4025, 1191 predictions, and 8 empty predictions, but it drops to AP 0.3140 and 54 empty predictions by step 200.
6. Schemafix v1 and bidirgate warmup10 start with the correct objects wrapper in all 200 traces, but many traces parse to empty predictions because the generated sequence is malformed or overlong.
7. Malformed patterns include object emission after `]}`, extra top-level keys after the `objects` array, missing separators/post-close object emission, repeated `<|endoftext|>` tails, and max-token truncation.
8. Degradation is strongest in higher-GT-count buckets. For example, `bidirgate_warmup10` step 100 has an empty parsed prediction rate of 0.863 for rows with at least 7 GT objects.
9. The train-time stop diagnostics do not show a uniform increase in close-start probability when remaining GT exists. For example, in `bidirgate_warmup10`, `stop/p_close_start_when_remaining_exists` decreases from 0.6103 at step 1 to around 0.08-0.10 by steps 90-150.
10. The bidirectional gate appears active in teacher-forced metrics: coord-slot coord mass increases and coord-gate loss decreases during training.

## 13. Procedural Caveats

1. Train-time eval artifacts are produced by the callback using the live trainer model replica. The callback constructs an `InferenceEngine`, injects `engine.model = runtime_model`, and then runs inference. Any summary field that only records the configured base checkpoint is therefore not sufficient by itself to identify the live training weights used for that callback eval.
2. The reported AP values are `val200` / `limit=200`, not full COCO val.
3. The reported f1ish values use `f1ish_pred_scope: annotated`.
4. The strict parser remains the benchmark contract in these runs. Empty-pred counters include cases where the model generated text but parsing yielded no valid prediction objects.
5. The current report does not assert a root cause for degradation. It only records infrastructure, metrics, and rollout symptoms visible in the current code and artifacts.
