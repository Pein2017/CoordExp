---
doc_id: progress.diagnostics.compact_full_coord_confidence_stop_gate
layer: progress
doc_type: diagnostic-study
status: active-reference
domain: stage1-compact-full-prefix-rollin
summary: Root-cause diagnostic for compact-full prefix-rollin / multiple-positive training, focused on low training loss but conservative free decode, coord_mean_logprob confidence separation, repetition-penalty stability, the 2026-05-08 HF batch prompt-offset bug that caused compact-grammar state drift under left padding, and the 2026-05-12 A3/A4 follow-up showing less predictable prefix-rollin behavior.
tags: [stage1, compact-full, prefix-rollin, et-rmp-ce, coord-confidence, stop-gate, eos, repetition-penalty, val200, diagnostics, prompt-offset, left-padding]
updated: 2026-05-12
---

# Compact-Full Coord-Confidence And Stop-Gate Diagnostics

This note records the 2026-05-08 diagnostic state for the compact-full
prefix-rollin / multi-positive training line.

Use this note when the question is:

- why training loss can be low while free decode remains conservative or misses objects;
- whether apparent FPs are hallucinations or plausible unlabeled objects;
- whether `coord_mean_logprob` can separate reliable object emission from
  invalid or duplicate continuation tail;
- how `repetition_penalty` changes compact-full rollout stability;
- why batched HF compact-grammar decode can differ from single-sample replay;
- how the later A3/A4 prefix-rollin checkpoints changed continuation behavior
  without yet giving a reliable production diagnostic;
- what counterfactual boundary probe should be run next.

Do not use this as a full validation result. The evidence here is diagnostic:
`val32` smoke/probe runs, `val200` selected-token trace analysis, and one
first-200 fixed-inference comparison. Current implementation contracts still
live in `docs/` and configs.

Related prior diagnostic source:

- [2026-04-29_et_rmp_rp_continuation_bias_hypothesis.md](2026-04-29_et_rmp_rp_continuation_bias_hypothesis.md)

## Short Diagnosis

The current best root-cause model has changed after the counterfactual replay
probe and the follow-up batch-path audit.

The strongest implementation root cause found on 2026-05-08 is:

```text
The pre-fix HF batched inference path passed attention_mask.sum() as
prompt_lengths to compact-grammar logits processors, but those processors treat
the value as an absolute input_ids column offset. Under decoder-only left
padding, short rows in a batch therefore exposed prompt suffix tokens as
"generated history". This could put the compact grammar in the wrong finite
state and allow or favor <|im_end|> too early.
```

This explains the previously confusing observation:

```text
Original batch artifact selected <|im_end|>, but single-sample replay and
processed boundary argmax both chose continuation.
```

The remaining model-behavior hypothesis is still useful, but now it must be
read after fixing the batch prompt-offset implementation:

```text
The model can learn local object-row generation under teacher-forced or
prefix-conditioned supervision, so training loss can be low. Free decode remains
recall-limited when recursive object discovery is bottlenecked at the
free-boundary stop decision: continue with "\n" or stop with "<|im_end|>".
Many unmatched objects have TP-like coordinate confidence, so they are often
plausible or unlabeled rather than random hallucinations. True runaway failure is
a different mode: low-confidence invalid or duplicate tail, especially under
weaker repetition penalty or later object index.
```

A useful shorthand:

```text
The model often knows how to write an object row, but it is too conservative
about opening the next row. If pushed too hard, it can enter a low-confidence
duplicate or invalid tail.
```

The A3/A4 follow-up keeps this shorthand but adds an important caution:

```text
Weakening EOS can open more recall, but it also makes the model's rollout basin
less predictable. A4 improves over A3 on the aggregate val200 surface, yet it
adds more invalid/border/collapse events and manual review does not support a
clean "A4 is simply better" conclusion.
```

## Scope Guard

The main surfaces in this note are:

1. `val32` smoke/probe runs created under `temp/` in the worktree:

```text
/data/CoordExp/.worktrees/compact-prefix-rollin-et-rmp-ce/temp/
```

2. Existing `val200` selected-token trace artifacts under `output_remote`:

```text
/data/CoordExp/output_remote/infer/recursive_detection_ce_latest/compact_full_support2_tokenrows_v2_ckpt3664_val200_bsz8_temp0_rep1p10_compact_grammar
/data/CoordExp/output_remote/infer/recursive_detection_ce_latest/compact_full_support2_tokenrows_v2_ckpt3664_val200_bsz8_temp0_rep1p00_compact_grammar
```

3. Fixed `val200` prompt-offset artifact under `output_remote`:

```text
/data/CoordExp/output_remote/infer/recursive_detection_ce_latest/compact_full_support2_tokenrows_v2_ckpt3664_val200_bsz8_temp0_rep1p10_prompt_offset_fix_8gpu
```

Important limitations:

- `val32` is a tiny diagnostic, not a benchmark.
- The fixed `val200` result is still a first-200 diagnostic comparison, not
  full validation.
- The pre-fix `val200` coord-confidence analysis is a diagnostic over
  selected-token traces. Its stop behavior is contaminated by the left-padding
  prompt-offset bug.
- Existing `pred_token_trace.jsonl` records selected token logprobs, but not
  unselected EOS/newline alternatives at boundary states.
- Therefore, `coord_mean_logprob` can analyze generated objects, but it cannot
  by itself prove what would have happened at a boundary where the model stopped.
- The original `val200` `rp=1.10` artifact was generated before the HF batch
  prompt-offset fix. Treat its conservative stopping behavior as potentially
  contaminated by the left-padding compact-grammar bug until compared against a
  fixed artifact.

## Artifact Map

Primary temporary diagnostic outputs:

```text
temp/logprob_confidence_probe/small_fire_rollout_health_summary.json
temp/logprob_confidence_probe/teacher_forced_eos_margin_summary.json
temp/logprob_confidence_probe/sequence_anomaly_summary.json
temp/logprob_confidence_probe/existing_rollout_object_confidence_summary.json
temp/logprob_confidence_probe/val200_coord_confidence/val200_coord_confidence_summary.json
temp/logprob_confidence_probe/val200_coord_confidence/val200_root_cause_tables.json
temp/logprob_confidence_probe/val200_coord_confidence/ckpt3664_rp1p10_val200_compact_full_object_rows.jsonl
temp/logprob_confidence_probe/val200_coord_confidence/ckpt3664_rp1p00_val200_compact_full_stress_object_rows.jsonl
temp/logprob_confidence_probe/val200_coord_confidence/ckpt3664_rp1p10_val200_prompt_offset_fix_object_rows.jsonl
```

Val200 source roots:

```text
output_remote/infer/recursive_detection_ce_latest/compact_full_support2_tokenrows_v2_ckpt3664_val200_bsz8_temp0_rep1p10_compact_grammar
output_remote/infer/recursive_detection_ce_latest/compact_full_support2_tokenrows_v2_ckpt3664_val200_bsz8_temp0_rep1p00_compact_grammar
```

Prompt-offset fix smoke outputs:

```text
temp/infer/compact_full_support2_ckpt3664_val16_bsz8_rp1p10_prompt_offset_fix_smoke
temp/infer_configs/compact_full_support2_ckpt3664_val16_bsz8_rp1p10_prompt_offset_fix_smoke.yaml
```

Durable fixed val200 config launched after the smoke:

```text
configs/infer/recursive_detection_ce_latest/compact_full_support2_tokenrows_v2_ckpt3664_val200_bsz8_rep1p10_prompt_offset_fix.yaml
output_remote/infer/recursive_detection_ce_latest/compact_full_support2_tokenrows_v2_ckpt3664_val200_bsz8_temp0_rep1p10_prompt_offset_fix_8gpu
```

Small-fire checkpoint roots used in the `val32` smoke comparison:

```text
temp/recursive_detection_ce_latest/output/compact_full_prefix_rollin_separator1p0_adapter_probe20_effbs8/smoke-compact-full-prefix-rollin-separator1p0-adapter-probe20-effbs8/v0-20260508-063344/checkpoint-20
temp/recursive_detection_ce_latest/output/compact_full_prefix_rollin_separator1p25_adapter_probe20_effbs8/smoke-compact-full-prefix-rollin-separator1p25-adapter-probe20-effbs8/v0-20260508-063344/checkpoint-20
temp/recursive_detection_ce_latest/output/compact_full_prefix_rollin_separator1p5_adapter_probe20_effbs8/smoke-compact-full-prefix-rollin-separator1p5-adapter-probe20-effbs8/v0-20260508-063344/checkpoint-20
```

A3/A4 follow-up artifacts:

```text
temp/a4_rp110_tf_probe_and_manual_review_20260512/tf_probe_high_ge10_32_summary/summary.md
temp/a4_rp110_tf_probe_and_manual_review_20260512/manual_audit_a4_vs_a3_image20_v2_pixelgt/manual_audit_a4_vs_a3_image20_v2_pixelgt.csv
temp/a4_rp110_tf_probe_and_manual_review_20260512/manual_audit_a4_vs_a3_image20_v2_pixelgt/manifest_no_gt.json
output_remote/infer/recursive_detection_ce_latest/compact_full_prefix_rollin_balance2_a3_ckpt3664_val200_bsz8_temp0_rp1p10_max3084_chatfix_4gpu
output_remote/infer/recursive_detection_ce_latest/compact_full_prefix_rollin_balance2_a4_eos_ckpt3664_val200_bsz8_temp0_rp1p10_max3084_chatfix_4gpu
```

Baseline adapter checkpoint used for comparison:

```text
output_remote/stage1_2b/recursive_detection_ce_latest/compact_full_et_rmp_ce_support2_bsz16_4epoch_tokenrows_v2/compact-full-et-rmp-ce-support2-bsz16-4epoch-tokenrows-v2/v0-20260504-071356/checkpoint-3664
```

## Decode Contract Observed

The compact-full inference logs confirmed the Qwen chat generation contract:

```text
eos_token = <|im_end|>
eos_token_id = 151645
pad_token = <|endoftext|>
pad_token_id = 151643
processor_do_resize = false
compact_grammar.enabled = true
compact_grammar.force_row_start = true
```

Under `temperature=0.0`, Transformers reported that `temperature`, `top_p`, and
`top_k` flags may be ignored. These runs should therefore be read as greedy
compact-grammar decode with repetition penalty.

## HF Batch Prompt-Offset Bug

### Bug

The production HF batch path used left padding for decoder-only generation. In
the pre-fix implementation, it computed per-sample `prompt_lengths` using:

```text
attention_mask.sum(dim=1)
```

However, compact grammar and stop-pressure processors use that value as an
absolute `input_ids` slice point:

```text
generated_ids = input_ids[row_idx, prompt_len:]
```

For a left-padded batch:

```text
[pad ... pad][real prompt tokens][generated tokens]
```

the generated-history absolute start is the padded prompt width:

```text
input_ids.shape[1]
```

not the unpadded prompt length. On shorter rows in a batch, the old code could
therefore feed prompt suffix tokens into the compact grammar as if they were
already generated output.

### Why It Matters

The compact grammar is stateful over generated history:

- empty generated history allows row start or `<|im_end|>`;
- after four coord tokens, it allows newline or `<|im_end|>`;
- after newline, it allows object start or `<|im_end|>`;
- inside bbox it forces coord tokens.

If prompt suffix tokens are misread as generated history, the first generation
step of a short row can begin in the wrong grammar state. This can make a batched
decode choose `<|im_end|>` even when a single-sample replay, with no left-padding
offset mismatch, chooses continuation.

### Patch

The fix is to pass padded prompt width to HF logits processors:

```text
prompt_padded_len = input_ids.shape[1]
prompt_lengths = [prompt_padded_len for each batch row]
```

This is now applied to:

- `src/infer/backends.py::generate_hf_batch`
- `src/infer/engine.py::_generate_hf` private helper, to avoid diagnostics using
  a different prompt-offset convention

Unit/contract checks:

```text
conda run -n ms python -m py_compile src/infer/engine.py src/infer/backends.py temp/counterfactual_boundary_probe.py
rtk conda run -n ms python -m pytest tests/test_infer_compact_grammar.py -q
rtk conda run -n ms python -m pytest tests/test_infer_batch_decoding.py -q
rtk conda run -n ms python -m pytest tests/test_infer_compact_grammar.py tests/test_infer_batch_decoding.py -q
```

Observed results:

```text
py_compile: passed
progress/config YAML parse: passed
tests/test_infer_compact_grammar.py + tests/test_infer_batch_decoding.py: 24 passed
```

### Val16 Smoke Evidence

Smoke config:

```text
temp/infer_configs/compact_full_support2_ckpt3664_val16_bsz8_rp1p10_prompt_offset_fix_smoke.yaml
```

Smoke artifact:

```text
temp/infer/compact_full_support2_ckpt3664_val16_bsz8_rp1p10_prompt_offset_fix_smoke
```

Scope:

```text
checkpoint = checkpoint-3664
slice = first 16 COCO val records
decode = HF compact_full, temperature 0.0, rp=1.10, batch_size=8
change = current worktree code with padded prompt offsets for logits processors
```

Aggregate comparison against the original pre-fix val200 artifact, restricted
to the same first 16 records:

| field | original pre-fix | prompt-offset fix smoke | delta |
|---|---:|---:|---:|
| total predictions | 50 | 89 | +39 |
| images with more predictions | 0 | 14 / 16 | +14 |
| images unchanged | 16 | 1 / 16 | -15 |
| images with fewer predictions | 0 | 1 / 16 | +1 |
| parse/error entries | 1 | 0 | -1 |

Two records that motivated the counterfactual replay:

| record_idx | image_id | GT | original preds | fixed-smoke preds | delta | original trace len | fixed trace len |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 632 | 17 | 8 | 14 | +6 | 64 | 114 |
| 9 | 1000 | 17 | 9 | 14 | +5 | 72 | 116 |

Both fixed-smoke outputs remained parseable with no error entries.

Interpretation:

```text
The conservative-stop symptom is not only a loss/objective problem. At least a
large part of the observed pre-fix val200 conservatism came from an inference
implementation bug: compact grammar was reading the wrong generated-history
suffix under batched left padding.
```

This does not disprove remaining stop calibration issues. It means any future
stop-gate or coord-confidence conclusion must be rerun on fixed prompt-offset
artifacts.

### Fixed Val200 Evidence

After the `val16` smoke, the same checkpoint and primary decode surface were
rerun on first-200 val records with eight GPUs:

```text
config = configs/infer/recursive_detection_ce_latest/compact_full_support2_tokenrows_v2_ckpt3664_val200_bsz8_rep1p10_prompt_offset_fix.yaml
artifact = output_remote/infer/recursive_detection_ce_latest/compact_full_support2_tokenrows_v2_ckpt3664_val200_bsz8_temp0_rep1p10_prompt_offset_fix_8gpu
checkpoint = checkpoint-3664
decode = HF compact_full, temperature 0.0, rp=1.10, batch_size=8
scope = first 200 COCO val records
```

Top-level generation health:

| field | pre-fix val200 | fixed val200 | delta |
|---|---:|---:|---:|
| errors_total | 4 | 0 | -4 |
| prediction rows in JSONL | 590 | 1141 | +551 |
| images with more predictions | 0 | 156 / 200 | +156 |
| images unchanged | 200 | 35 / 200 | -165 |
| images with fewer predictions | 0 | 9 / 200 | +9 |
| duplicate suppressed | 152 | 233 | +81 |
| duplicate-affected records | n/a | 61 | n/a |

Raw eval deltas:

| metric | pre-fix | fixed | delta |
|---|---:|---:|---:|
| bbox_AP | 0.164 | 0.425 | +0.261 |
| bbox_AP50 | 0.199 | 0.575 | +0.376 |
| bbox_AP75 | 0.171 | 0.448 | +0.277 |
| bbox_AR100 | 0.171 | 0.472 | +0.302 |
| P50 micro | 0.775 | 0.711 | -0.064 |
| R50 micro | 0.296 | 0.543 | +0.247 |
| F50 micro | 0.429 | 0.616 | +0.187 |
| TP loc | 428 | 784 | +356 |
| FP loc | 124 | 318 | +194 |
| FN loc | 1016 | 660 | -356 |

Guarded eval deltas:

| metric | pre-fix guarded | fixed guarded | delta |
|---|---:|---:|---:|
| bbox_AP | 0.152 | 0.412 | +0.260 |
| bbox_AP50 | 0.184 | 0.557 | +0.373 |
| bbox_AP75 | 0.159 | 0.435 | +0.275 |
| bbox_AR100 | 0.157 | 0.457 | +0.300 |
| P50 micro | 0.860 | 0.801 | -0.059 |
| R50 micro | 0.247 | 0.483 | +0.237 |
| F50 micro | 0.383 | 0.603 | +0.220 |
| TP loc | 356 | 698 | +342 |
| FP loc | 58 | 173 | +115 |
| FN loc | 1088 | 746 | -342 |

Interpretation:

```text
The prompt-offset bug was not a cosmetic artifact issue. Fixing it converts a
large amount of hidden continuation mass into generated objects and materially
improves first-200 AP, recall, and F1 on the same checkpoint/decode surface.
```

The tradeoff is also visible:

```text
The fixed run emits many more objects. A large fraction are useful, but the
duplicate/tail burden also increases. The next problem is no longer simply
"make the model continue"; it is "continue when the next object has reliable
object/coordinate evidence, then suppress duplicate or low-confidence tail."
```

This is exactly the regime where `coord_mean_logprob`, duplicate guards, and
counterfactual next-row probes become useful as a ruler.

### Fixed Val200 Coord-Confidence Recompute

The selected-token object-row analysis was rerun after adding the fixed artifact
to the same temp diagnostic script:

```text
temp/logprob_confidence_probe/analyze_val200_coord_confidence.py
temp/logprob_confidence_probe/val200_coord_confidence/root_cause_tables.py
```

Derived outputs:

```text
temp/logprob_confidence_probe/val200_coord_confidence/val200_coord_confidence_summary.json
temp/logprob_confidence_probe/val200_coord_confidence/val200_root_cause_tables.json
temp/logprob_confidence_probe/val200_coord_confidence/ckpt3664_rp1p10_val200_prompt_offset_fix_object_rows.jsonl
```

Bucket shift from pre-fix `rp=1.10` to fixed `rp=1.10`:

| bucket | pre-fix trace | fixed trace | delta |
|---|---:|---:|---:|
| matched TP | 422 | 776 | +354 |
| unmatched possible unlabeled/FP | 154 | 330 | +176 |
| duplicate/near-duplicate unmatched | 6 | 32 | +26 |
| invalid geometry | 12 | 3 | -9 |
| suspicious duplicate-or-invalid | 18 | 35 | +17 |
| total trace objects | 594 | 1141 | +547 |

`coord_mean_logprob` by fixed-artifact bucket:

| bucket | n | mean | p10 | median | p90 |
|---|---:|---:|---:|---:|---:|
| matched TP | 776 | -2.662 | -3.209 | -2.641 | -2.169 |
| unmatched possible unlabeled/FP | 330 | -3.336 | -4.237 | -3.320 | -2.578 |
| duplicate/near-duplicate unmatched | 32 | -3.896 | -4.745 | -3.789 | -3.200 |
| invalid geometry | 3 | -3.162 | -3.332 | -3.148 | -2.998 |

AUC using `coord_mean_logprob` after the prompt-offset fix:

| comparison | pre-fix AUC | fixed AUC | read |
|---|---:|---:|---|
| matched TP vs suspicious duplicate-or-invalid | 0.864 | 0.956 | stronger tail-separation after restored continuation |
| matched TP vs duplicate/near-duplicate | 0.887 | 0.965 | duplicate tail is very separable |
| matched TP vs invalid geometry | 0.852 | 0.870 | only 3 invalid rows remain, so read cautiously |
| matched TP vs unmatched possible unlabeled/FP | 0.804 | 0.813 | still overlapping; high-confidence unmatched remains real |

Threshold reads on the fixed artifact:

| cutoff source | cutoff | matched retained | high-conf unmatched | low-conf suspicious |
|---|---:|---:|---:|---:|
| matched p05 / best suspicious | -3.377 | 737 / 776 | 178 / 330 | 29 / 35 |
| matched p10 | -3.209 | 698 / 776 | 140 / 330 | 29 / 35 |
| kmeans k2 split | -3.083 | 657 / 776 | 115 / 330 | 31 / 35 |
| matched p25 | -2.907 | 582 / 776 | 87 / 330 | not primary |

Interpretation:

```text
The fixed artifact confirms the earlier hypothesis but sharpens it: restoring
continuation unlocks many true positives and many plausible unmatched objects,
while duplicate/invalid tail remains lower-confidence and largely separable by
coord_mean_logprob.
```

Important caution:

```text
High coord confidence is not the same as "definitely useful detection". Some
high-confidence unmatched examples include very large scene-support boxes, such
as frame-filling dining-table/couch rows. The score is best treated as a tail
risk ruler, not a production TP classifier.
```

Still, the practical signal is strong:

```text
coord_mean_logprob around -3.2 to -3.4 remains the right diagnostic band. Around
the matched-p05 / best-suspicious cutoff, the fixed artifact retains about 95%
of matched TPs, keeps about 54% of unmatched plausible rows, and marks about 83%
of suspicious duplicate/invalid rows as low-confidence.
```

### Counterfactual Boundary Probe Instrumentation

The temp probe was extended to avoid mixing score surfaces:

```text
temp/counterfactual_boundary_probe.py
```

It now records:

- raw forward `continue` vs `<|im_end|>` logits and logprobs;
- processed logits after repetition penalty plus compact grammar;
- processed allowed/masked booleans for `continue` and `<|im_end|>`;
- processed argmax token;
- original selected EOS logprob from trace;
- optional one-step `generate(max_new_tokens=1)` boundary replay;
- state-sensitive continue token role:
  - `row_separator` means continue token is `\n`;
  - `row_start` means continue token is `<|object_ref_start|>`.

The initial maxcases2 replay showed:

| row_count | rescued_gt_iou50 | raw margin mean | processed margin mean | original EOS lp mean | processed argmax continue |
|---:|---:|---:|---:|---:|---:|
| 2 | 2 | 7.25 | 6.59 | -0.462 | 2 / 2 |

Read:

```text
On the replayed single-sample boundary, both raw and processed scoring preferred
continuation. The original batch artifact selected EOS. This mismatch was the
clue that pushed the investigation toward the batched prompt-offset bug.
```

### Fixed-Artifact Targeted One-Row Probe

After the prompt-offset fix, an artifact-only candidate screen found:

| candidate type | count |
|---|---:|
| fixed val200 records | 200 |
| `pred_count < GT_count` | 103 |
| stopped and under-generated | 20 |
| clean first-pass forced-next-row candidates | 14 |

The cleanest replay candidates were selected from stopped-underfull fixed
artifact rows with reasonable existing precision and no obvious low-confidence
tail:

```text
record_idxes = 96,137,0,140,54
```

The temp probe was adjusted for targeted replay:

```text
--record-idxes
--progress-every
incremental per-case JSONL writes
first-compact-row stopping criterion for forced continuation
```

Targeted replay artifact:

```text
temp/counterfactual_boundary_probe/ckpt3664_val200_rp1p10_prompt_offset_fix_top5_onerow
```

Scope:

```text
checkpoint = checkpoint-3664
decode source = fixed prompt-offset val200 artifact
boundary = final stopped compact-full row boundary
forced action = generate exactly one next compact object row
temperature = 0.0
rp = 1.10
max_new_tokens = 24 with first-row stopping
```

Top-level result:

| metric | value |
|---|---:|
| replayed stopped-underfull cases | 5 |
| raw boundary margin <= 0 | 5 / 5 |
| processed boundary margin <= 0 | 5 / 5 |
| processed argmax EOS | 5 / 5 |
| forced next row rescued GT @ IoU50 | 3 / 5 |
| forced next row low-confidence unmatched | 2 / 5 |
| high-confidence candidates at cutoff `-3.4` | 3 / 5 |
| high-confidence rescued GT | 3 / 3 |

Per-case result:

| record_idx | image_id | GT | pred | boundary margin | processed margin | forced row | coord_mean | best GT IoU | bucket |
|---:|---:|---:|---:|---:|---:|---|---:|---:|---|
| 0 | 139 | 20 | 13 | -1.000 | -0.909 | `vase [378,466,395,497]` | -2.645 | 0.559 | rescued GT |
| 54 | 5586 | 14 | 7 | -0.500 | -0.455 | `person [0,0,73,170]` | -3.792 | 0.000 | low-conf unmatched |
| 96 | 9590 | 29 | 18 | -2.625 | -2.386 | `cup [512,670,629,730]` | -3.734 | 0.384 | low-conf near-miss |
| 137 | 14038 | 24 | 7 | -0.375 | -0.341 | `book [789,638,840,697]` | -3.383 | 0.628 | rescued GT |
| 140 | 14439 | 20 | 15 | -1.125 | -1.023 | `person [282,109,311,154]` | -2.811 | 0.558 | rescued GT |

Interpretation:

```text
The fixed artifact still has stopped-underfull boundaries where EOS is the local
argmax, but a one-row forced continuation can reveal real GT objects. The
candidate coord-confidence band is useful: all candidates above the initial
-3.4 cutoff were rescued GT, while the two below it were low-confidence
unmatched or near-miss rows.
```

This supports the user's hypothesis in a constrained form:

```text
The useful future decode experiment is not "always continue". It is closer to:
speculatively force exactly one next row at selected stopped/near-stop
boundaries, then accept continuation only when the next row has valid geometry,
is not duplicate/scene-support-risk, and has sufficient coord/object evidence.
```

Runtime note:

```text
Each Qwen3-VL boundary forward and forced one-row generate remained expensive
at this resolution, about 64 seconds each in this temp probe. Future probes
should use explicit record allowlists and write per-case outputs incrementally.
```

The targeted probe was then extended to all 14 artifact-screened
stopped-underfull candidates, using three GPUs with disjoint record allowlists.

Merged artifact:

```text
temp/counterfactual_boundary_probe/ckpt3664_val200_rp1p10_prompt_offset_fix_candidates14_merged
```

Merged top-level result:

| metric | value |
|---|---:|
| replayed stopped-underfull candidates | 14 |
| raw boundary margin <= 0 | 14 / 14 |
| processed boundary margin <= 0 | 14 / 14 |
| processed argmax EOS | 14 / 14 |
| rescued GT @ IoU50 | 8 / 14 |
| high-confidence unmatched plausible | 3 / 14 |
| low-confidence unmatched or near-miss | 3 / 14 |
| high-conf candidates at `coord_mean >= -3.4` | 9 / 14 |
| high-conf rescued GT | 6 / 9 |
| high-conf plausible unmatched | 3 / 9 |
| low-conf candidates at `coord_mean < -3.4` | 5 / 14 |
| low-conf rescued GT | 2 / 5 |
| low-conf unmatched/near-miss | 3 / 5 |

Merged margin/score summary:

| field | mean | p10 | median | p90 |
|---|---:|---:|---:|---:|
| raw continue-minus-EOS margin | -0.813 | -1.825 | -0.563 | -0.250 |
| processed continue-minus-EOS margin | -0.739 | -1.659 | -0.511 | -0.227 |
| candidate `coord_mean_logprob` | -3.242 | -3.720 | -3.304 | -2.695 |
| original selected EOS logprob | -0.428 | -0.586 | -0.470 | -0.178 |

Per-case buckets:

| record_idx | image_id | GT | pred | forced row | coord_mean | best GT IoU | bucket |
|---:|---:|---:|---:|---|---:|---:|---|
| 0 | 139 | 20 | 13 | `vase` | -2.645 | 0.559 | rescued GT |
| 36 | 3255 | 10 | 8 | `person` | -2.826 | 0.624 | rescued GT |
| 54 | 5586 | 14 | 7 | `person` | -3.792 | 0.000 | low-conf unmatched |
| 74 | 7511 | 16 | 9 | `person` | -3.131 | 0.000 | high-conf plausible |
| 92 | 9378 | 10 | 8 | `person` | -3.689 | 0.599 | rescued GT |
| 93 | 9400 | 24 | 22 | `keyboard` | -2.463 | 0.878 | rescued GT |
| 96 | 9590 | 29 | 18 | `cup` | -3.734 | 0.384 | low-conf near-miss |
| 111 | 11197 | 16 | 13 | `traffic light` | -3.112 | 0.000 | high-conf plausible |
| 137 | 14038 | 24 | 7 | `book` | -3.383 | 0.628 | rescued GT |
| 140 | 14439 | 20 | 15 | `person` | -2.811 | 0.558 | rescued GT |
| 152 | 15517 | 13 | 10 | `bus` | -3.554 | 0.524 | rescued GT |
| 160 | 16249 | 11 | 10 | `bench` | -3.358 | 0.165 | high-conf plausible |
| 181 | 18380 | 53 | 33 | `cup` | -3.251 | 0.620 | rescued GT |
| 190 | 19109 | 27 | 21 | `motorcycle` | -3.634 | 0.378 | low-conf near-miss |

Read:

```text
Even after the HF prompt-offset fix, the model can choose EOS at stopped
underfull boundaries where a one-row forced continuation recovers real GT
objects. This is direct evidence for remaining conservative stop behavior on a
small but targeted fixed-artifact sample.
```

The same result also warns against a blunt decode intervention:

```text
Forced continuation does not always help. Some rows are low-confidence unmatched
or near-miss. A coord threshold around -3.4 is a useful first gate, but it is not
a hard production cutoff: it keeps all high-confidence plausible rows and most
rescued GT rows, while two rescued GT rows are below the cutoff.
```

## Val32 Small-Fire Training And Rollout Read

Three small-fire continuation weights were trained for 20 steps on top of the
existing compact-full adapter:

```text
separator_continue_weight = 1.0
separator_continue_weight = 1.25
separator_continue_weight = 1.5
```

All used:

```text
eos_stop_weight = 0.5
boundary component_weight = 0.3
effective_batch_size = 8
max_steps = 20
train_sample_limit = 32
val_sample_limit = 4
```

Training was stable at this scale:

| separator_continue_weight | loss_mean | grad_norm_mean | grad_norm_max | continue_mass_mean | trie_valid_mass_mean | stop_top1_mean |
|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | 1.358 | 72.36 | 317.18 | 0.881 | 0.568 | 0.906 |
| 1.25 | 1.355 | 63.93 | 136.73 | 0.884 | 0.569 | 0.906 |
| 1.5 | 1.357 | 60.70 | 108.51 | 0.884 | 0.569 | 0.900 |

Teacher-forced generated-prefix boundary diagnostics showed that small-fire
training increased continuation pressure at the free boundary:

| run | free-boundary margin mean | margin <= 0 | valid mass |
|---|---:|---:|---:|
| baseline | -1.661 | 14/14 | 0.206 |
| sep1.0@20 | 0.179 | 7/14 | 0.749 |
| sep1.25@20 | 0.438 | 7/14 | 0.763 |
| sep1.5@20 | 0.455 | 6/14 | 0.766 |
| sep2@20 | 0.241 | 9/14 | 0.740 |
| sep2@60 | 0.795 | 5/14 | 0.819 |

Interpretation:

- Free-boundary EOS pressure is real in the baseline.
- Training can improve the local continuation-vs-EOS margin.
- The margin alone is not sufficient: `sep2@60` looks good in teacher-forced
  boundary metrics but is unhealthy in free rollout.

### Val32 Free-Rollout Health

The table below uses a diagnostic greedy same-class IoU matcher, not official
COCO AP. `R50_high` is recall on `GT_count >= 10` samples.

| run | pred_mean | err | raw_invalid | repeat_rows | P50 | R50 | F50 | R50_high | P30 | R30 | F30 | coordAUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline-rp1.0 | 6.375 | 1 | 1 | 3 | 0.480 | 0.443 | 0.461 | 0.299 | 0.520 | 0.480 | 0.499 | 0.914 |
| baseline-rp1.10 | 5.500 | 0 | 0 | 0 | 0.659 | 0.525 | 0.584 | 0.410 | 0.693 | 0.552 | 0.615 | 0.879 |
| sep1.0-rp1.10 | 6.062 | 0 | 1 | 0 | 0.582 | 0.511 | 0.545 | 0.427 | 0.670 | 0.588 | 0.627 | 0.869 |
| sep1.25-rp1.10 | 6.125 | 0 | 1 | 0 | 0.582 | 0.516 | 0.547 | 0.427 | 0.653 | 0.579 | 0.614 | 0.856 |
| sep1.5-rp1.10 | 6.562 | 1 | 2 | 0 | 0.533 | 0.507 | 0.520 | 0.427 | 0.610 | 0.579 | 0.594 | 0.871 |
| sep2.0-rp1.10 | 6.062 | 0 | 3 | 0 | 0.624 | 0.548 | 0.583 | 0.487 | 0.680 | 0.597 | 0.636 | 0.837 |
| sep2.0x60-rp1.10 | 6.719 | 21 | 24 | 2 | 0.563 | 0.548 | 0.555 | 0.453 | 0.623 | 0.606 | 0.615 | 0.896 |

Key reads:

- `rp=1.10` is the correct primary decode surface for this family. It removed
  exact-repeat rows in the small-fire val32 runs and greatly improved baseline
  stability relative to `rp=1.0`.
- `sep2@20 + rp=1.10` gave the best high-count recall in this tiny diagnostic
  (`R50_high=0.487`) but produced `raw_invalid=3`.
- `sep1.25@20 + rp=1.10` was safer but did not beat the baseline overall at
  20 steps.
- `sep2@60` was overcooked: even with `rp=1.10`, it retained `err=21` and
  `raw_invalid=24`. Repetition penalty cannot fully rescue an overcooked
  continuation checkpoint.

## Val200 Coord-Mean Confidence Study

The primary `val200` selected-token trace study used:

```text
checkpoint = checkpoint-3664
format = compact_full
temperature = 0.0
repetition_penalty = 1.10
batch_size = 8
max_new_tokens = 1024
```

Primary derived buckets from the pre-fix selected-token trace:

| bucket | count |
|---|---:|
| matched TP | 422 |
| unmatched possible unlabeled/FP | 154 |
| invalid geometry | 12 |
| duplicate/near-duplicate unmatched | 6 |
| total trace objects | 594 |

`coord_mean_logprob` by bucket:

| bucket | n | mean | p10 | median | p90 | median obj_idx | median bbox area frac |
|---|---:|---:|---:|---:|---:|---:|---:|
| matched TP | 422 | -2.744 | -3.364 | -2.713 | -2.147 | 1.0 | 0.0599 |
| unmatched possible unlabeled/FP | 154 | -3.371 | -4.140 | -3.389 | -2.654 | 4.0 | 0.00393 |
| invalid geometry | 12 | -3.573 | -4.239 | -3.727 | -2.764 | 5.0 | 0.0 |
| duplicate/near-duplicate unmatched | 6 | -3.667 | -4.356 | -3.693 | -2.951 | 2.0 | 0.0201 |

AUC using `coord_mean_logprob`:

| comparison | AUC |
|---|---:|
| matched TP vs duplicate-or-invalid suspicious | 0.864 |
| matched TP vs duplicate/near-duplicate | 0.887 |
| matched TP vs invalid geometry | 0.852 |
| matched TP vs unmatched possible unlabeled/FP | 0.804 |

Interpretation:

- `coord_mean_logprob` separates reliable objects from invalid/duplicate tail
  much better than it separates matched TPs from plausible unmatched objects.
- This supports reading many unmatched objects as plausible or unlabeled rather
  than random hallucinations.
- It also supports using `coord_mean_logprob` as a low-confidence tail filter,
  not as a universal TP-vs-FP classifier.
- Because this trace was produced before the prompt-offset fix, its object-row
  confidence separation remains informative, but its stop-rate and count
  distribution must be remeasured on the fixed artifact.

### Stress Control: `rp=1.00`

The stress surface used the same checkpoint and format with
`repetition_penalty=1.00`.

| bucket | count at rp=1.10 | count at rp=1.00 |
|---|---:|---:|
| matched TP | 422 | 376 |
| unmatched possible unlabeled/FP | 154 | 100 |
| duplicate/near-duplicate unmatched | 6 | 86 |
| invalid geometry | 12 | 20 |

Key read:

```text
Lowering rp from 1.10 to 1.00 massively increases duplicate/near-duplicate tail
without making high-confidence unmatched disappear as a category.
```

This confirms that `rp=1.00` is useful as a runaway stress control, but should
not be treated as the primary decode surface.

### Threshold Candidates

For the primary `rp=1.10` val200 surface:

| threshold source | cutoff | TP retention | possible-unlabeled retention | suspicious rejection |
|---|---:|---:|---:|---:|
| matched p05 | -3.506 | 94.8% | 59.1% | 66.7% |
| best suspicious split | -3.417 | 92.7% | 53.2% | 72.2% |
| matched p10 | -3.364 | 89.8% | 48.7% | 72.2% |
| kmeans k2 split | -3.049 | 76.5% | 29.2% | 72.2% |
| duplicate-optimal | -2.932 | 67.3% | 22.7% | 88.9% |

Diagnostic read:

- `coord_mean_logprob < -3.4` is a plausible low-confidence marker.
- `-3.4` retains roughly `90-93%` of matched TPs and rejects roughly `72%` of
  duplicate/invalid suspicious objects.
- Stricter thresholds around `-3.05` drop too many TPs and plausible unmatched
  objects for a continuation gate.

### High-Confidence Unmatched Objects

Using the matched-p10 cutoff `coord_mean_logprob >= -3.364` on `rp=1.10`:

| field | value |
|---|---:|
| high-confidence unmatched | 75 / 154 |
| rate | 48.7% |
| image count | 45 |
| median object index | 2 |
| p75 object index | 5 |
| median bbox area fraction | 0.00373 |

Examples include early, high-confidence unmatched objects:

| image_id | gt_count | obj_idx | desc | coord_mean_logprob |
|---:|---:|---:|---|---:|
| 5503 | 1 | 0 | person | -1.811 |
| 17031 | 1 | 0 | person | -1.889 |
| 11813 | 2 | 0 | chair | -1.939 |
| 2431 | 9 | 1 | bowl | -2.080 |

This is strong evidence against a simple hallucination explanation. These cases
need visual review or broader proxy labels before being counted as true FP.

By GT-count bucket at the same cutoff:

| GT bucket | high-conf unmatched | total unmatched | rate |
|---|---:|---:|---:|
| gt_00_02 | 14 | 14 | 100.0% |
| gt_03_05 | 12 | 19 | 63.2% |
| gt_06_10 | 18 | 25 | 72.0% |
| gt_11_20 | 24 | 63 | 38.1% |
| gt_21_plus | 7 | 33 | 21.2% |

Dense scenes contain many unmatched objects in absolute count, but high-confidence
unmatched objects are not confined to dense scenes. Annotation incompleteness is
therefore plausible, but it is not only a high-count-scene phenomenon.

### Low-Confidence Duplicate / Invalid Tail

Primary `rp=1.10`, matched-p10 cutoff `-3.364`:

| field | value |
|---|---:|
| low-confidence suspicious | 13 / 18 |
| rate | 72.2% |
| suspicious median obj_idx | 6 |
| suspicious p75 obj_idx | 10 |
| suspicious GT-count median | 17 |
| idx_10_19 low-confidence suspicious rate | 100% |

Representative suspicious examples:

| image_id | gt_count | obj_idx | desc | type | coord_mean_logprob |
|---:|---:|---:|---|---|---:|
| 5001 | 17 | 10 | person | duplicate/near-duplicate | -4.401 |
| 14439 | 20 | 10 | person | invalid geometry | -4.343 |
| 5001 | 17 | 11 | person | duplicate/near-duplicate | -4.312 |
| 14439 | 20 | 11 | person | invalid geometry | -4.261 |

This tail is the main danger when continuation pressure is too strong. It is
later, lower-confidence, and more duplicate/invalid-prone than the high-confidence
unmatched group.

### Correlations

For valid objects on primary `rp=1.10`:

| relation | Pearson | Spearman | read |
|---|---:|---:|---|
| coord vs obj_idx | -0.551 | -0.586 | later objects are lower confidence |
| coord vs prefix_token_count | -0.549 | -0.578 | longer generated prefix lowers confidence |
| coord vs GT count | -0.458 | -0.548 | dense scenes lower confidence |
| coord vs bbox_area_frac | 0.439 | 0.441 | larger boxes are easier |
| coord vs log bbox area | 0.422 | 0.441 | area is a real factor |

Do not overread this as "late objects are hallucinations." Later object index
mixes several effects: smaller objects, dense scenes, annotation incompleteness,
self-generated prefix drift, and tail risk.

## A3/A4 Prefix-Rollin Follow-Up (2026-05-12)

The later production-scale A3/A4 checkpoints were evaluated on the same
first-200 COCO `compact_full`, `rp=1.10`, `temperature=0.0`,
`max_new_tokens=3084` surface used for the compact-full follow-up dashboard.

Run definitions:

| ID | Objective | Intended mechanism |
|---|---|---|
| A3 | prefix-rollin support+balance | add prefix-closed coverage while keeping EOS supervision fully trusted |
| A4 | A3 + EOS-trust prior | reduce EOS force according to the empirical missing-label prior |

Aggregate result:

| Run | AP | AP50 | F1@0.50 | Recall | Precision | Pred | Invalid or bad-geom | Suppressed | Guard AP | Guard F1@0.50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A3 prefix-rollin | `0.3980` | `0.5444` | `0.5597` | `0.4917` | `0.6496` | `1162` | `11` | `301` | `0.3854` | `0.5485` |
| A4 EOS-trust | `0.4001` | `0.5502` | `0.5612` | `0.5173` | `0.6133` | `1300` | `72` | `401` | `0.3915` | `0.5638` |

Training-side snapshot:

| Run | Final ckpt | Trainer best ckpt | Final eval CE | Coord top1 | Type mass | EOS trust |
|---|---:|---:|---:|---:|---:|---:|
| A3 | `3664` | `3664` | `1.5964` | `0.1315` | `0.9705` | `1.0000` |
| A4 | `3664` | `3600` | `1.5710` | `0.1318` | `0.9727` | `0.3113` |

Read:

- A4 improves A3 on aggregate AP, AP50, recall, and guarded F1.
- A4 also increases invalid/border/collapse events (`11 -> 72`) and duplicate
  suppression burden (`301 -> 401`).
- A3 is cleaner but more conservative; A4 opens recall but is more chaotic.
- Neither A3 nor A4 beats the older A2/support+balance row on the same
  compact-full `rp=1.10` leaderboard.

### Boundary Probe

The A3/A4 teacher-forced probe used 32 high-density first-val200 images with
`GT_count >= 10`, GT prefixes `K=0,1,3,5,10,N`, and generated-prefix boundary
states from each run's own `rp=1.10` rollout.

| Run | Generated-boundary n | Generated sep-minus-EOS mean | Generated sep <= 0 | Generated entry mean | GT sep mean | GT sep <= 0 | True-end EOS prob |
|---|---:|---:|---:|---:|---:|---:|---:|
| A2 support+balance | `27` | `-1.306` | `1.000` | `15.162` | `8.525` | `0.056` | `0.880` |
| A3 prefix-rollin | `22` | `-1.000` | `1.000` | `15.099` | `7.087` | `0.048` | `0.858` |
| A4 EOS-trust | `18` | `-0.667` | `1.000` | `16.066` | `7.387` | `0.032` | `0.692` |

Read:

- A4 weakens EOS relative to A3, but generated free-boundary states still prefer
  EOS in every sampled generated-prefix case.
- Object-entry confidence after forced continuation remains high. The model is
  not primarily failing to emit `<|object_ref_start|>` once the row is opened.
- The gap is still autoregressive state quality: the model behaves much better
  under GT prefixes than under its own generated prefixes.

### Manual Review Read

The 20-image A4-vs-A3 review should be read qualitatively only. The earlier
GT-green overlay had a pixel-vs-norm1000 coordinate rendering bug; use the v2
pixel-GT/no-GT overlays.

Reviewer synthesis:

- A4 adds many plausible unlabeled positives, especially in dense scenes.
- A4 also produces frequent top-left/border coordinate collapse and duplicate
  bursts. These are often purple/red overlay events.
- Magenta duplicate-guard candidates are mixed: many are true duplicates, but
  some are visually plausible separate dense objects.
- A3 is often tighter and cleaner; A4 is often higher recall. The two effects
  are entangled at the image level.

Current conclusion:

```text
A3/A4 are useful research probes, but not a reliable solved version of the
multiple-positive/prefix-rollin line. The next reliable gate must distinguish
valid continuation from duplicate/collapse tail before A4-like EOS weakening can
be promoted.
```

## Root-Cause Ranking

| rank | hypothesis | evidence strength | current read |
|---:|---|---|---|
| 1 | HF batched compact-grammar prompt-offset bug under left padding | very strong | confirmed implementation root cause for much of the pre-fix conservative stopping |
| 2 | Duplicate/tail control after restored continuation | strong | fixed val200 and A4 emit more useful objects, but also more duplicate/low-confidence or border-collapse tail |
| 3 | Free-boundary EOS/continue calibration is still too conservative after the offset fix | strong but incomplete | A4 weakens EOS, yet generated-prefix boundary probes still prefer EOS; blunt weakening is not enough |
| 4 | Annotation incompleteness / eval FP contamination | strong for "FP != hallucination", moderate for confirmed unlabeled | critical to interpretation |
| 5 | Local teacher-forced objective vs global free-decode stop mismatch | strong as a general mechanism | still relevant after implementation fixes and A3/A4 |
| 6 | Rollout exposure mismatch | strong | A3/A4 boundary probe shows GT-prefix continuation is healthy while generated-prefix continuation remains weak |
| 7 | Coord confidence can gate failure modes | very strong for duplicate/invalid tail filtering, weak as universal TP-vs-FP classifier | useful diagnostic ruler |
| 8 | True perception failure | present but not primary in these artifacts | likely concentrated in small, dense, late objects |

## Why Low Training Loss Can Coexist With Conservative Decode

Training loss can be low when the model learns the local conditional:

```text
prefix -> next object row tokens
```

Free decode has an additional global control problem:

```text
at every object boundary: continue with "\n" or stop with "<|im_end|>"
```

A single premature stop prevents all later object rows from being generated.
Therefore the model can score object rows well under teacher forcing but still
miss objects in autoregressive rollout.

The 2026-05-08 prompt-offset finding adds a second, implementation-level reason:

```text
The training objective may be healthy locally, but the batched inference
processor can put the decoder into the wrong grammar state before the first
generated token of a row. Then decode behavior no longer reflects the trained
conditional distribution.
```

The current evidence supports this distinction:

- Many generated matched TPs have high coordinate confidence.
- Many unmatched objects also have high coordinate confidence and plausible
  object classes/geometry.
- Invalid/duplicate tail is lower-confidence and separable.
- Teacher-forced free-boundary continuation metrics improve under continuation
  training, but free rollout can still fail or overcook.

## Recommended Next Probe

Existing selected-token traces are insufficient for unselected EOS alternatives,
and the pre-fix val200 trace is now known to be contaminated by the batched
prompt-offset bug. The fixed val200 object-confidence recompute is now done.
The next diagnostic should focus on remaining stopped or near-stop boundaries
and on visual review of high-confidence unmatched rows from the fixed artifact.

Minimal probe contract:

```text
for each val200 image:
  run or replay the normal compact-full decode prefix

  at free boundary states, especially when decode stops or EOS is competitive:
    record logprob(<|im_end|>)
    record logprob("\n")
    record continue_minus_eos_margin

    if EOS wins or the margin is near zero:
      force "\n<|object_ref_start|>"
      generate or score exactly one next object row
      record coord_mean_logprob
      record coord_min_logprob
      record valid geometry
      record duplicate / near-duplicate against already emitted objects
      record diagnostic GT match bucket when available
```

Decision table:

| counterfactual next object | root-cause implication |
|---|---|
| high coord confidence, valid, matched TP | premature EOS truncated a real GT object |
| high coord confidence, valid, unmatched plausible | supports unlabeled/plausible object and conservative eval |
| low coord confidence or invalid/duplicate | stop was likely reasonable; forcing continuation would create tail errors |
| low coord confidence, valid but unmatched | perception or objectness mass is weak; data/objective may need work |

Start threshold for diagnostics:

```text
coord_mean_logprob >= -3.4 => plausible object band
coord_mean_logprob <  -3.4 => low-confidence / suspicious candidate
```

This threshold is not a production contract. It is a starting point for
`val200` counterfactual curves.

## Near-Term Guidance

Do not blindly increase continuation pressure.

Recommended near-term priorities:

1. Build artifact-only joint curves for `coord_mean_logprob`, actual
   duplicate-guard action, object index, GT-count bucket, bbox area, and
   scene-support-risk flags. Treat `coord_mean` as a tail-risk ruler, not a
   universal TP classifier.
2. Visually review high-confidence unmatched rows from the fixed `val200`
   artifact, separating real unlabeled objects from large scene-support boxes
   and localization failures.
3. Compare at least one continuation-trained checkpoint such as `sep2@20` under
   the same probe if artifact budget allows.
4. Report rescued TP, high-confidence unmatched, low-confidence invalid, and
   duplicate tail counts by `coord_mean_logprob` threshold.
5. Only after the counterfactual probe confirms useful hidden objects, consider
   a decode-side experiment: `force continue unless next object confidence is
   low`, guarded by geometry and duplicate checks.
6. For A3/A4 specifically, add a collapse-aware acceptance gate before drawing
   conclusions from more EOS weakening: border-touch flags, top-left basin
   flags, self-duplicate IoU, large scene-support box flags, and coord/object
   confidence should be logged together.

## Open Questions

- Does the 14-case forced-row rescue rate hold under a larger stratified
  fixed-artifact boundary sample, or is it concentrated in stopped-underfull
  candidates selected by artifact-only screening?
- Are high-confidence unmatched objects visibly real under manual review?
- Does `sep2@20` increase rescued high-confidence objects without increasing
  low-confidence invalid/duplicate tail under the same probe?
- How stable is the `-3.4` coord cutoff across checkpoints, count buckets, and
  object sizes?
- Should future confidence thresholds be size/count-aware rather than global?
- Can A4's useful continuation rows be separated from its purple/top-left
  collapse rows using only logits, geometry, duplicate overlap, and prefix-state
  features available at decode time?
- Is A3/A4 instability caused mostly by EOS trust, by prefix-rollin generated
  state exposure, or by coordinate basin attraction in dense scenes?
