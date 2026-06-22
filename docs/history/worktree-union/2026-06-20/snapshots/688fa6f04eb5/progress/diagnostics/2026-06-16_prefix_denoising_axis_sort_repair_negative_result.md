# Prefix Denoising SFT: Axis-Sort Repair Negative Result

Date: 2026-06-16
Scope: worktree `/data/CoordExp/.worktrees/geometry-aware-denoising-sft`
Branch: `codex/prefix-denoising-sft`

## Context

We evaluated whether malformed or invalid bbox outputs from the prefix-denoising
SFT checkpoint are mostly caused by inverted `x1/y1/x2/y2` endpoint order, and
whether a simple axis-sort repair can recover the expected val200 mAP level.

Starting inference artifact:

```text
/data/CoordExp/outputs/infer/prefix_denoising_sft_v1/prefix_denoising_ckpt450_val200_free_decode_diagnostic_eval_8gpu
```

Checkpoint:

```text
/data/CoordExp/outputs/stage1_2b/detection_teacher_forcing/compact_full_prefix_denoising_kl_w0p05_2b_base_sorted_marker_bsz1x128_2epoch/compact-full-prefix-denoising-kl-w0p05-2b-base-sorted-marker-bsz1x128-2epoch/v25-20260615-124531/checkpoint-450
```

Original inference setup:

- `limit: 200`
- sorted compact-full prompt
- `row_separator: none`
- free greedy decode
- `max_new_tokens: 3084`
- `repetition_penalty: 1.10`
- no compact grammar / no logits constraints

## Probe 1: Post-Repair From Token Traces

Artifact:

```text
/data/CoordExp/outputs/infer/prefix_denoising_sft_v1/prefix_denoising_ckpt450_val200_free_decode_diagnostic_eval_8gpu/postops_invalid/axis_sort_repair
```

Method:

- Reparse generated token traces.
- For each object-like span with exactly four coord tokens, canonicalize:

```text
x1, x2 = sorted(x1, x2)
y1, y2 = sorted(y1, y2)
```

- Drop equal-axis degenerate boxes because pure sorting cannot make them valid.
- Score repaired boxes with `exp(mean(original generated coord-token logprobs))`.

Repair summary:

```text
objects_emitted                         1321
salvaged_pred_objects_on_invalid_rows    569
objects_repaired                          75
invalid_by_order_or_equal                 75
degenerate_after_axis_sort               114
wrong_coord_arity                        464
edge_sat_source                          755
```

Result:

```text
original raw AP       0.1176
post-repair raw AP    0.1457

original raw AP50     0.2072
post-repair raw AP50  0.2556

original guarded AP   0.1132
post-repair guarded AP 0.1417

original empty_pred   64
post-repair empty_pred 1

original pred_total   769
post-repair pred_total 1321
```

Interpretation:

Post-repair greatly improves materialization coverage, but does not restore
localization quality. The best diagnostic AP remains far below the expected
`0.35+` range.

## Probe 2: Immediate Repair During Inference Materialization

Artifact:

```text
/data/CoordExp/outputs/infer/prefix_denoising_sft_v1/prefix_denoising_ckpt450_val200_decode_axis_sort_repair_eval
```

Method:

Added a diagnostic compact-full parse mode in the worktree:

```yaml
infer:
  parsing:
    compact_full:
      mode: marker_delimited_axis_sort_repair
```

This mode repairs only the four-token bbox endpoint order after the parser reads
exactly four coord tokens and before positive-area validation. It still rejects:

- wrong arity
- extra coord tokens
- missing markers
- newline separator violations
- empty descriptions
- equal-axis degenerate boxes

Standard confidence result:

```text
raw AP       0.1227
raw AP50     0.2212
raw AP75     0.1147
guarded AP   0.1182

empty_pred   49
pred_total scored 933
```

The standard confidence post-op kept `933 / 984` materialized predictions and
dropped `51` as `missing_span`. This is expected for boxes whose geometry was
sorted: the repaired coord-token sequence no longer byte-matches the generated
coord-token sequence.

## Probe 3: Immediate Repair With Repair-Aware Confidence

Artifact:

```text
/data/CoordExp/outputs/infer/prefix_denoising_sft_v1/prefix_denoising_ckpt450_val200_decode_axis_sort_repair_eval/postops_invalid/repair_aware_confidence
```

Method:

Use the immediate repaired `pred` artifact, but score each prediction from the
original generated four coord-token logprobs rather than requiring the repaired
sorted coord-token sequence to appear in the token trace.

Result:

```text
raw AP      0.1214
raw AP50    0.2188
raw AP75    0.1141
empty_pred  49
pred_total  984
```

Interpretation:

Repair-aware confidence does not improve the immediate-repair conclusion. The
51 dropped boxes in the standard confidence path were not the limiting factor.

## Code Surface Added For The Diagnostic

Worktree-local changes:

- `src/detection/teacher_forcing/compact_full_policy.py`
  - added `marker_delimited_axis_sort_repair`
  - axis-sorts bbox endpoints before positive-area validation
- `src/infer/pipeline.py`
  - allows the diagnostic parse mode in `infer.parsing.compact_full.mode`
- `src/detection/evaluation.py`
  - registers the parse mode in parser-mode validation
- `tests/test_compact_full_marker_policy.py`
  - covers inverted endpoint repair and degenerate rejection
- `tests/test_infer_compact_full_policy_contract.py`
  - covers config propagation and repaired artifact payload

Verification:

```bash
python -m pytest tests/test_compact_full_marker_policy.py tests/test_infer_compact_full_policy_contract.py tests/test_detection_template_parsing_eval.py -q
```

Result:

```text
52 passed
```

## Conclusion

This is a negative result for the endpoint-order hypothesis.

Axis-sort repair is useful as a diagnostic because it separates pure endpoint
inversion from broader coordinate failure. However, it does not recover mAP to
the expected level. The degraded checkpoint is not merely producing swapped
`x1/x2` or `y1/y2`; it also produces many edge-saturated, degenerate, wrong-arity,
or poorly localized boxes.

Most likely interpretation:

- compact-full schema is mostly learned;
- coordinate-token rollout quality is the bottleneck;
- sorting valid-looking four-token spans improves artifact coverage but not
  enough localization quality;
- the prefix-denoising checkpoint needs coordinate-learning diagnosis rather
  than a decode/materialization post-op fix.

Recommended supervisor discussion point:

The failure mode looks like a coordinate distribution / training-objective
problem, not a simple parser or geometry-normalization problem. Next probes
should focus on slot-wise coordinate accuracy, edge-token overproduction,
teacher-forced vs free-rollout coordinate logits, and whether the 2-epoch
prefix-denoising run undertrained coordinate behavior relative to the pure-CE
baseline.
