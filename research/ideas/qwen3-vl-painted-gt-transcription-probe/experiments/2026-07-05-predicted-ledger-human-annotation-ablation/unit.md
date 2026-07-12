---
title: Predicted-Ledger Human-Annotation Ablation
description: Tests whether feeding back the model's own emitted row and painting that predicted box improves coverage on the 512-image training slice.
type: idea
role: research-unit
authority: non_normative_research
promotion_status: not_promoted
unit_id: 2026-07-05-predicted-ledger-human-annotation-ablation
topic: qwen3-vl-painted-gt-transcription-probe
status: complete
tags: [coordexp-swift, painted-gt, predicted-ledger, self-prefix, coverage, corrected-rerun]
updated: 2026-07-06
---

# Predicted-Ledger Human-Annotation Ablation

## 2026-07-06 Corrected Rerun Verdict

The corrected rerun is now metric-bearing for the predicted-ledger question.
It retrained after fixing the `ledger_teacher_prefix` rendering/prompt gates,
then evaluated both the mismatched one-shot readout and the intended
predicted-ledger self-prefix controller.

Corrected training checkpoint:

```text
/data/CoordExp/outputs/painted_gt/train_ledger_gate512/painted_gt_ledger_teacher_prefix_geo_gate512_2epoch_warm_start_dora_all_towers_accelerate8_ebs8-prefixfix-8gpu-20260706T014333Z/checkpoints/step-120
```

Corrected one-shot artifact:

```text
/data/CoordExp/outputs/painted_gt/trained_inference/ledger_source_train512/ledger_prefixfix_step120_standard_rp110
```

Corrected predicted-ledger artifact:

```text
/data/CoordExp/outputs/painted_gt/trained_inference/predicted_ledger_source_train512_parallel/prefixfix_step120_rp110_step160_dp8/merged
```

The predicted-ledger artifact was run as 8 independent source-image shards and
merged into a validated scored artifact set. The controller leakage policy is:

```json
{
  "gt_used_for_decoding": false,
  "gt_used_for_painting": false,
  "gt_used_for_stopping": false,
  "paint_source": "model_predicted_bbox",
  "prefix_source": "model_parser_text"
}
```

Corrected headline metrics on the 512-image / 3803-object training slice:

| Condition | Decode | F1 | Precision | Recall | mAP | mRecall | Pred | Matched |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| step917 baseline | one-shot | 0.5923 | 0.6384 | 0.5525 | 0.4072 | 0.4744 | 3291 | 2101 |
| standard 512 2epoch | one-shot | 0.5898 | 0.6055 | 0.5748 | 0.4381 | 0.4938 | 3610 | 2186 |
| corrected ledger 512 2epoch | one-shot | 0.2265 | 0.8611 | 0.1304 | 0.1348 | 0.1389 | 576 | 496 |
| corrected ledger 512 2epoch | predicted-ledger self-prefix | 0.2724 | 0.8408 | 0.1625 | 0.1678 | 0.1744 | 735 | 618 |

The predicted-ledger controller improves over the corrected ledger adapter's
one-shot readout, but only modestly. It remains far below the unpainted
same-source standard baseline and the step917 baseline. The dominant behavior
is high precision with very low recall: the model emits too few objects, not
too many malformed or noisy objects.

Controller summary for the corrected predicted-ledger run:

- accepted predictions: `731`;
- total decode steps: `1243`;
- stop reasons: `731` continue steps followed by `512` no-valid-prediction
  image stops;
- duplicate prediction count: `0`;
- parser failures: `2`;
- scoreable predictions: `731`.

The generic merged `summary.json` shows `decode_stop_reason_counts.length=512`
because each image-level controller row is stored as a long controller trace.
For mechanism interpretation, use `predicted_ledger_report.json` instead; it
records the controller-level `continue` and `no_valid_prediction` decisions.

### Corrected Interpretation

The corrected run does not support the strong "hidden enumeration unlocked by
painting committed predictions" hypothesis.

It does support a narrower statement: feeding back the model's own emitted row
and painting that committed predicted box can improve the prefix-specialized
ledger adapter over a mismatched one-shot readout. However, that improvement is
not enough to recover normal coverage. Painting only objects that the model has
already emitted acts more like an occupied-region ledger than a positive cue
for unseen objects. If the core failure is under-enumeration, this intervention
does not itself create new candidate evidence.

This result is compatible with the earlier positive GT-paint evidence. GT
painting and stepwise current-object marks inject object identity or object
candidate information directly. Predicted-ledger painting intentionally avoids
GT leakage and therefore only marks already-committed predictions.

## 2026-07-06 Invalidated First Attempt

This result is invalid as evidence about the predicted-ledger hypothesis.

Root-cause diagnosis found that `ledger_teacher_prefix` rows were not accepted
by the shared teacher-prefix rendering gates. Training therefore rendered
non-empty ledger steps as only the current target row, without the intended
ignored `teacher_prefix_text`. The checkpoint below was trained on that wrong
contract, so the low one-shot and predicted-ledger metrics cannot be interpreted
as a model or mechanism limitation.

The inference trace did include self-prefix during predicted-ledger rollout, but
that was not enough to validate the experiment because the adapter itself had
already been trained on prefix-dropped ledger examples. A corrected ledger run
must retrain after the renderer/prompt fix and then rerun one-shot plus
predicted-ledger evaluation before this ablation receives a new verdict.

## Question

This ablation tests the human-annotation-style hypothesis:

1. start from the raw unpainted image;
2. decode using the model's own predicted prefix, not GT prefix;
3. after each emitted object, paint/highlight the model-predicted bbox;
4. feed the painted image plus self-prefix back to predict the next object.

If this improves coverage, it would suggest the model can recognize or enumerate
more objects than the normal AR output reveals, and that a missing visual ledger
or coverage memory is the limiting factor.

## Dataset And Checkpoints

Evaluation slice:

```text
/data/CoordExp/outputs/painted_gt/materialized/ledger_source_unpainted_gate512_v1/ledger_source_unpainted_gate512.examples.jsonl
```

This source-level eval view contains `512` raw unpainted training images and
`3803` GT objects reconstructed from the ledger teacher-prefix materialization.

Training runs:

```text
/data/CoordExp/outputs/painted_gt/train_ledger_gate512/painted_gt_ledger_teacher_prefix_geo_gate512_2epoch_warm_start_dora_all_towers_accelerate8_ebs8-ledger512-2epoch8-20260705T190159Z
/data/CoordExp/outputs/painted_gt/train_standard_gate512/painted_gt_standard_unpainted_gate512_2epoch_warm_start_dora_all_towers_accelerate8_ebs8-std512-2epoch8-20260705T191921Z
```

The ledger run trained on per-object teacher-prefix examples and completed
`114` optimizer steps. The standard run trained on source-level full-object
rows for the same 512 images and completed `17` optimizer steps because rows
pack much more densely. Therefore the standard run is a same-source/same-epoch
control, not a same-compute control.

Summary artifact:

```text
/data/CoordExp/outputs/painted_gt/trained_inference/ledger_source_train512/predicted_ledger_ablation_summary.json
```

## Results

| Condition | Decode | F1 | Precision | Recall | mAP | mRecall | Pred | Matched |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| step917 baseline | one-shot | 0.5923 | 0.6384 | 0.5525 | 0.4072 | 0.4744 | 3291 | 2101 |
| standard 512 2epoch | one-shot | 0.5898 | 0.6055 | 0.5748 | 0.4381 | 0.4938 | 3610 | 2186 |
| ledger 512 2epoch | one-shot | 0.1928 | 0.8125 | 0.1094 | 0.1042 | 0.1055 | 512 | 416 |
| ledger 512 2epoch | predicted-ledger self-prefix | 0.1937 | 0.8164 | 0.1099 | 0.1020 | 0.1038 | 512 | 418 |

The predicted-ledger controller made `1019` decode steps:

- `507` steps emitted one accepted prediction and continued;
- `512` steps emitted no valid prediction and stopped the image;
- no image emitted more than one accepted prediction.

Trace inspection confirmed that predicted-ledger rollout prompts included the
step 0 compact row as assistant self-prefix and used the painted step-001 image.
That observation only validates the rollout controller. It does not validate the
training contract, which was later found to have dropped `teacher_prefix_text`
for `ledger_teacher_prefix` rows.

## Object-Count Pattern

The collapse is coverage-specific and becomes more severe as object count
increases.

For the ledger-trained predicted-ledger run:

- 1-object images: F1 `0.8929`, recall `0.8929`;
- 2-3 object images: F1 `0.5388`, recall `0.3808`;
- 4-6 object images: F1 `0.2802`, recall `0.1683`;
- 7-10 object images: F1 `0.1847`, recall `0.1031`;
- 11+ object images: F1 `0.0621`, recall `0.0327`.

The same 11+ bucket under the step917 one-shot baseline had F1 `0.4740` and
recall `0.4337`. This means the predicted-ledger mechanism did not recover
hidden coverage; it converted the model into a high-precision first-object
transcriber with severe early termination.

## Verdict

This is not a valid negative result for the predicted-ledger setup.

The hypothesis "paint the model's own emitted boxes to reveal hidden enumeration
capacity" remains untested by this run. The observed one-object collapse is
consistent with the prefix-dropped training bug and must not be used to argue
that predicted painting adds nothing over one-shot decoding.

The stronger positive painted-GT result remains the earlier GT-marked stepwise
teacher-prefix/current-object setup, where the mark tells the model which
object to transcribe. This predicted-ledger ablation is different: it asks the
model to use its own prior emission as a coverage ledger. That failed here.

## Next Implication

Do not treat this artifact set as an inference-time coverage verdict. The next
required step is a corrected ledger retrain followed by the same one-shot and
predicted-ledger evaluation on the fixed contract.

Possible next probes:

- multi-row self-prefix training where the current target is not always the
  final/only supervised row;
- explicit continue-vs-stop supervision after painted ledgers;
- same-compute standard baseline if compute fairness becomes the central
  comparison;
- a targeted small run that supervises the model to continue after a painted
  predicted-like prefix, not only after GT teacher-prefix steps.

## Research Unit Closeout

Observed:

- The first predicted-ledger run was invalidated by a training-contract bug:
  non-empty `ledger_teacher_prefix` rows dropped the intended ignored
  `teacher_prefix_text`.
- The corrected prefixfix run is metric-bearing on the 512-image / 3803-object
  training slice.
- Corrected predicted-ledger self-prefix improved over the corrected ledger
  one-shot readout, but remained far below the unpainted same-source standard
  baseline and the step917 baseline.

Supported:

- Painting the model's own committed prediction can act as an occupied-region
  ledger for the prefix-specialized adapter.
- The corrected result does not recover normal coverage; the dominant failure
  remains high precision with very low recall.

Not supported yet:

- The strong hypothesis that predicted painting reveals hidden enumeration
  capacity.
- A deployable coverage-memory mechanism.
- Any claim based on the invalidated first attempt.

Next decider:

- A future coverage probe should change the candidate-generation route, not
  merely paint already-emitted predictions.

Promotion decision:

- Not promoted. This remains non-normative research evidence and does not
  define current inference or evaluation behavior.
