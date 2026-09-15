---
type: idea
title: Qwen3-VL Painted-GT Transcription Probe
description: Tests whether Qwen3-VL can translate explicit painted GT object annotations into compact detection rows strongly enough to overcome language-prior and prefix-prior failures.
tags: [coordexp-swift, qwen3-vl, painted-gt, visual-annotation, transcription-probe, language-prior]
state: migrated
updated: 2026-07-12
---

# Qwen3-VL Painted-GT Transcription Probe

## Current State

This is a diagnostic research lineage, not a final detector architecture. Its
temporary painted-GT and proposal-bridge branches are historical evidence
sources, not current implementation bases. New experiments should be rebuilt
against canonical CoordExp-Swift infrastructure and cite the relevant unit
rather than importing the temporary code wholesale.

The source-truth experiment contract is [Experiment Plan](experiment-plan.md).
OpenSpec and implementation plans must preserve that research meaning unless
the user explicitly approves a changed research question. The current docs
authorize OpenSpec drafting and Superpowers planning, not implementation or GPU
launch by themselves.

The core question is whether Qwen3-VL can use obvious image-side GT-derived
object annotations to emit compact detection rows, rather than drifting under
autoregressive language priors, prefix priors, or learned object-count priors.

This is a ceiling test for visual evidence usage. It does not claim painted GT
annotations are an inference-time method.

## Taste And Research Posture

The branch should be led by evidence rather than implementation convenience.
The user's bigger-picture priority order is:

1. accuracy, precision, and research validity;
2. training and overall execution efficiency;
3. simplicity that avoids over-design and redundant machinery;
4. scalability and extensibility after the mechanism is clearer.

The agent owns implementation details, code organization, smoke execution, and
dynamic tactical adjustment when new evidence appears. These adjustments do not
require user approval when they preserve the core research question, improve
validity, reduce risk, or make the execution cleaner. They must be recorded in
the relevant plan, manifest, or review note.

Changing the branch away from the painted-GT transcription hypothesis,
deleting major artifacts, external publication, or presenting a broader claim
than the evidence supports still requires explicit user approval.

The intended final outcome of the first engineering cycle is one of:

- blocked, with concrete evidence and a narrow reason;
- or a passed 256-image smoke gate followed by launch of a larger two-epoch
  painted-input training run.

This branch should not optimize for a pretty method story. It should make the
model "tell us" whether painted visual annotation is a useful interface for
overcoming language-prior and prefix-prior failures.

After this research blueprint is approved, execution must go through OpenSpec
and Superpowers governance before code changes begin:

1. create an OpenSpec change for branch-local painted-data, training, decoding,
   artifact, and evaluation interfaces;
2. validate the OpenSpec proposal/design/spec/task artifacts through review
   convergence;
3. draft a Superpowers implementation plan under `docs/superpowers/plans/`;
4. review the implementation plan until no unresolved P0/P1 findings remain;
5. implement and verify from the approved plan.

Do not bypass OpenSpec and Superpowers just because the experiment is
branch-local or urgent.

## Hypothesis

If GT object information is painted directly into the image, then Qwen3-VL
should be able to translate or rephrase that image information into the compact
CoordExp row format after limited painted-input adaptation.

If the model still cannot overfit a tiny painted training set, then the problem
is likely deeper than ordinary visual salience. In that case, subtle learned
slots, current-object vectors, or coverage mechanisms are premature until the
representation or decoding interface is reconsidered.

The strongest positive result is not merely a loss drop. The meaningful signal
is decoded behavior: valid compact rows, improved training-set mAP/F1-style
metrics on the exact painted surface, and interpretable differences among
paint-all, stepwise teacher-prefix, and stepwise self-prefix modes.

## 2026-07-05 Counterfactual Evidence

After 16-epoch tiny overfit adaptation, the required train256 counterfactual
panels strongly support causal use of the painted visual marks:

- paint-all `painted_correct`: `debug_detection_f1=0.678098`;
- paint-all `unpainted_same_prompt`: `debug_detection_f1=0.517294`;
- paint-all `offset_all_marks`: `debug_detection_f1=0.256769`;
- paint-all `style_matched_false_marks`: `debug_detection_f1=0.258271`;
- stepwise `painted_correct`: `per_target_step_debug_f1=0.936071`;
- stepwise `unpainted_same_prompt`: `per_target_step_debug_f1=0.133606`;
- stepwise `wrong_object_mark`: `per_target_step_debug_f1=0.019097`;
- stepwise `shuffled_or_offset_mark`: `per_target_step_debug_f1=0.023994`.

The wrong-object control is especially diagnostic. When scored against the
scheduled target row it nearly collapses, but a follow-mark analysis shows that
among rows with a prediction, `96.8%` of first predictions overlap the wrong
painted mark at IoU >= `0.50`, while only `1.4%` overlap the scheduled target
at IoU >= `0.50`. This means the model is not merely benefiting from global
image decoration; it is being steered by the current painted object.

Primary summary artifact:

```text
/data/CoordExp/outputs/painted_gt/counterfactual_inference/counterfactual_summary_gate256_overfit16.json
```

This evidence supports the branch hypothesis that explicit visual annotation
can overcome the language/prefix prior on the tiny overfit surface. It does not
yet authorize the larger two-epoch run by itself; the bounded self-prefix
diagnostic and tiny-gate classification remain separate gates.

## 2026-07-08 PVCI Step 0 Preparation Evidence

The PVCI/VCI preparation pass added held-out and coarseness controls before any
hidden-cursor or feature-cursor implementation. The detailed result note is
[PVCI Step 0 Preparation Results](experiments/2026-07-08-pvci-step0-preparation-results/unit.md).

Key outcomes:

- held-out val100 tight correct mark: `per_target_step_debug_f1=0.748340`;
- held-out val100 unpainted same prompt:
  `per_target_step_debug_f1=0.143558`;
- held-out val100 wrong mark scored against intended target:
  `per_target_step_debug_f1=0.017857`;
- held-out wrong-mark first-prediction follow analysis:
  `97.4%` of rows with a prediction overlap the marked object at IoU >= `0.50`,
  while only `0.46%` overlap the intended target at IoU >= `0.50`;
- coarseness val32: tight outline+center `0.748062`, outline-only `0.757282`,
  semi-transparent fill `0.433022`, 1.5x box `0.208897`, grid-snapped box
  `0.196911`, center point `0.156863`, center blob `0.126160`, 2.0x box
  `0.050193`.

Interpretation: the mark is a true steering actuator and generalizes beyond the
tiny train slice, but the current teacher is precise-boundary dependent and
does not tolerate coarse boxes well. Future PVCI should treat selector errors
as likely confident false-positive steering and should probably strengthen or
broaden the painted teacher before serious hidden/feature cursor distillation.

## 2026-07-05 Predicted-Ledger Human-Annotation Ablation

The follow-up predicted-ledger ablation tested a more deployable-looking
human-annotation loop: start from the raw image, decode with the model's own
prefix, paint the model-predicted bbox after each emitted object, and ask for
the next object.

The first run was invalidated. On 2026-07-06, diagnosis found that
`ledger_teacher_prefix` rows were excluded from the shared teacher-prefix
rendering gates. Non-empty ledger training steps therefore rendered as only the
current target row, without the intended ignored `teacher_prefix_text`.

A corrected prefixfix retrain and 8-way predicted-ledger rerun are now
metric-bearing for this ablation. The corrected result shows a real but small
benefit over the mismatched one-shot readout, while remaining far below the
standard same-source detector baseline. This suggests that painting the model's
own committed predictions can function as an occupied-region ledger, but does
not by itself recover missing object enumeration.

Primary result note:

```text
research/ideas/qwen3-vl-painted-gt-transcription-probe/experiments/2026-07-05-predicted-ledger-human-annotation-ablation/unit.md
```

Corrected artifacts:

```text
/data/CoordExp/outputs/painted_gt/train_ledger_gate512/painted_gt_ledger_teacher_prefix_geo_gate512_2epoch_warm_start_dora_all_towers_accelerate8_ebs8-prefixfix-8gpu-20260706T014333Z/checkpoints/step-120
/data/CoordExp/outputs/painted_gt/trained_inference/ledger_source_train512/ledger_prefixfix_step120_standard_rp110
/data/CoordExp/outputs/painted_gt/trained_inference/predicted_ledger_source_train512_parallel/prefixfix_step120_rp110_step160_dp8/merged
```

Corrected headline metrics on the 512-image / 3803-object training slice:

- step917 one-shot baseline: F1 `0.5923`, mAP `0.4072`;
- standard same-source two-epoch one-shot: F1 `0.5898`, mAP `0.4381`;
- corrected ledger one-shot: F1 `0.2265`, mAP `0.1348`;
- corrected ledger predicted-ledger self-prefix: F1 `0.2724`, mAP `0.1678`.

The corrected predicted-ledger run uses no GT for decoding, painting, or
stopping: the prefix comes from model-parser text and the paint source is the
model-predicted bbox. It improves recall from `0.1304` to `0.1625` compared
with the corrected ledger one-shot run, but remains a high-precision /
low-recall condition (`precision=0.8408`, `recall=0.1625`).

## 2026-07-06 Next-Object Steering Revision

The corrected predicted-ledger result exposed a stronger infrastructure
question: the previous stepwise target contract still encouraged local
assistant termination after object rows. The active follow-up paradigm is now
`next_object_steering`.

The central unit is one annotation action:

```text
current painted image
+ optional committed-row assistant prefix
-> one next unmarked object row or terminal stop
```

The effective supervised target has exactly two legal forms:

```text
object_continuation:
  row(X) + <|object_ref_start|>

terminal_stop:
  <|im_end|>
```

There is no third legal target form. In object-continuation steps, any
structural chat-template `<|im_end|>\n` that exists only to close the rendered
assistant message must be masked from loss. The trailing
`<|object_ref_start|>` is a real atomic Qwen token and acts as a lookahead
continuation sentinel; it is not part of the committed ledger row.

Two matched policies are first-class:

- `text_image`: current painted image plus canonical committed rows as
  assistant prefix;
- `image_only`: current painted image with empty assistant prefix at every
  step.

Both policies use the same `next_unmarked_object` prompt, `geo_sorted` V1
training schedule, target grammar, painter, parser, decode surface, and
evaluator. Cross-policy runs are diagnostic only.

The V1 painter is `outline_center_flat_v1`: magenta rectangle outline, yellow
center point, no fill, no opacity accumulation, no overlap-depth color coding,
and no visible class text, object id, or order cue. Overlap is handled through
the prompt and recorded diagnostics, not through darker or brighter pixels.

The rollout controller commits at most one valid row per step. It stops only
when `<|im_end|>` is the first generated token. Malformed candidates are
dropped; if a step has no valid row and does not start with `<|im_end|>`, the
controller continues with unchanged state up to `no_progress_limit=2`.
Duplicates are committed, painted, plotted, and penalized by metrics; V1 does
not suppress or guard them.

Rollout also has a GT-independent `max_rollout_steps=64` default. Backend stop
facts and controller decisions are separate: `generation_stop_reason` records
whether backend generation ended by EOS or length, while `controller_outcome`
records semantic events such as `terminal_stop`, `committed_row`,
`post_row_im_end_violation`, `step_cap_stop`, or `no_progress_limit_stop`.

This revision makes the two ablations sharper:

```text
text_image = visual ledger + language ledger
image_only = visual ledger only
```

Standard one-shot autoregressive detection remains a reference baseline, not a
central ablation that needs to be retrained for this branch.

## Canonical Terms

- `painted GT`: an image-level augmentation derived from GT object boxes and
  center points.
- `paint-all`: all GT objects in one image are painted at once, and the target
  response contains all compact rows.
- `one-by-one`: conversational name for stepwise painted rollout.
- `stepwise painted rollout`: one target object is emphasized per forward or
  per materialized example, the assistant-side prefix carries previous compact
  rows, and the target response or generated suffix contains exactly the
  current object's row.
- `painted-input adaptation`: continued training that lets the adapter and
  selected trainable modules learn how to use painted images.

Avoid calling this a deployable detector. Avoid claiming it proves raw visual
features are already sufficient. It proves whether an explicit rendered
annotation channel can drive the decoder.

## Non-Goals

- Do not build a final detector architecture.
- Do not claim painted GT is available at inference time.
- Do not treat the resulting adapter as the next normal unpainted detector
  baseline.
- Do not add learned slots, coverage-state machinery, or rollout training until
  the painted-input diagnostic result justifies that direction.
- Do not hide failures by normalizing or salvaging generated prefixes in the
  self-prefix test.
- Do not collapse paint-all and stepwise results into a single score.
- Do not launch larger training just because the pipeline runs.

## Approved Experiment Families

Paint-all and stepwise painted rollout are equal-status probes. Neither is
assumed to be easier, harder, more faithful, or more important before seeing
the metrics. The comparison is part of the mechanism study.

### Paint-All Full Output

Paint bbox rectangles and center points for all GT objects in each image, then
ask the model to output all compact detection rows.

This family is not assumed to be simpler than one-by-one decoding. It is chosen
as a multi-object transcription ceiling test, not as an implementation shortcut.
The user explicitly expects it may be harder than one-by-one sequential decoding.

Paint-all should require no model-architecture change. It is primarily an
image-level augmentation plus ordinary compact-row supervision.

### Stepwise Painted Rollout

Emphasize one object per prediction target and ask for that object's compact
row. The first ablations should include:

- `geo_sorted` target schedule;
- random target schedule.

The implementation may choose either:

- materialized per-object painted images with corresponding one-row assistant
  targets; or
- training-runtime support for multiple forwards with dynamic image
  augmentation.

The implementation choice is delegated to the agent, but the research meaning
must stay: one emphasized target object, one supervised output row.

The training and decoding modes must stay consistent. If decoding evaluates a
current painted target under an autoregressive prefix, training must expose the
same concept: current painted target image plus assistant-side prefix plus one
current-row target. Prefix-free single-object training is only an ablation, not
the primary one-by-one route.

For stepwise training, previous GT rows are conditioning prefix only. Their
labels must be ignored in the loss; only the current row is supervised. This is
a correctness requirement, not an implementation detail.

## Painting Content

The first painted inputs should use:

- bbox rectangle;
- center point;
- no category text label.

This makes object location explicit while avoiding direct leakage of the class
name through text labels or OCR.

## Prompt And Ordering

The main route may explicitly tell the model that objects are visually marked
and should be returned as compact rows. A prompt-only zero-shot check against
the frozen four-epoch baseline adapter is optional, not central.

For paint-all, the default target order is canonical `geo_sorted` unless a later
experiment deliberately adds a visible order cue. Hidden random order is not a
fair paint-all target.

For one-by-one, random ordering means the target-object schedule is randomized;
each forward or materialized example still has one emphasized object and one
target row.

## Training Posture

The starting point is the CoordExp-Swift pure-CE four-epoch baseline adapter.

Baseline adapter handle:

```text
/data/CoordExp/.worktrees/CoordExp-swift/outputs/prod/coordexp_swift/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_r16a32_llm_12000_accelerate8_ebs64_4epoch_warmup0p1-prod8-r16a32-ebs64-warmup0p1-20260702T170007Z/checkpoints/step-917/adapter
```

This baseline is an external read-only source artifact for this branch. New
painted-probe outputs should be written under this worktree's configured
`outputs` root, with manifests recording both the external source adapter and
the new local artifact root.

The step-917 adapter is LLM-only DoRA. It is a frozen comparison baseline and a
competence seed, not the directly loaded all-tower adapter surface. The adopted
seed strategy is `warm_start_expand_dora`: keep the original pretrained base
model from `model_cache`, construct an all-tower DoRA adapter surface, copy
language-side adapter tensors from the step-917 adapter by exact canonical key
mapping, initialize missing vision/aligner adapter tensors, and load the
repaired source coordinate/wrapper special-token embedding payload. Directly
loading the LLM-only adapter as an all-tower trainable adapter is not a valid
fallback.

Before painted-input claims, run or verify a normal unpainted val200 baseline
check for this adapter. The accepted reference is
`eval_coco_fixed_gt_scale/metrics.json` with `mAP = 0.4111788135144427`, and
the strict gate is `mAP >= 0.40` on val200. If the current inference/eval path
produces a lower value on the same unpainted reference surface, treat that as an
infrastructure or artifact identity problem and stop before interpreting
painted-probe results.

Unlike the original LLM-only adapter baseline, this branch should maximize
painted-input adaptation capacity by enabling DoRA over all linear modules
across the vision tower, MLP aligner, and language tower. The intended meaning
is adapter-based trainable adaptation across those towers plus approved
special-token embedding parameters, not full dense base-weight finetuning unless
a later decision explicitly approves that heavier path.

Run manifests must make this seed boundary explicit: base/tokenizer/processor
identity, source adapter identity, repaired source special-token embedding
identity, exact source-to-target warm-start key map, copied tensor hashes,
post-copy equality checks, newly initialized tensor counts, target towers,
target modules, optimizer groups, dense-base freeze status, coordinate/wrapper
embedding trainability, and total trainable counts. Any trainable parameter
outside the expanded adapter and approved special-token embeddings is a launch
blocker unless explicitly approved.

## Tiny Smoke Plan

The first tiny dataset size is `256` training images.

The tiny smoke is an overfitting and decoding benchmark, not a val200
generalization test. It should focus on:

- training-set mAP;
- F1-style decoding benchmark;
- output validity;
- whether the model improves after tiny painted-input overfitting.

For one-by-one ablations, the effective training sample count may scale with
the total object count because each object can become its own prediction target
or forward. This compute cost is accepted for this research worktree.

Train for at least two epochs. To avoid checkpoint fishing, the first gate uses
a predeclared budget: `2` initial epochs, extension up to `8` total epochs only
under recorded improving-but-not-yet-passing behavior, and a review note before
any further extension. Even memorizing this tiny training set is considered
non-trivial evidence.

Use a fixed deterministic 256-image training slice with a recorded manifest
rather than a changing random slice. The preferred builder is coverage-aware:
fixed seed, stable input order, and explicit buckets for simple controls,
multi-object images, crowded images, repeated-class cases, medium object counts,
and high object counts. Manual curation is not the default. The manifest must
record image ids, source JSONL paths, object counts, bucket labels, selection
seed, and the selection code/config identity.

The first implementation slice should build the shared painted-image and target
materialization primitive before either experiment family becomes a serious GPU
run. `paint-all` and `stepwise painted rollout` must share painting style,
artifact provenance, object ordering records, target-row construction, and
visual-audit outputs so that later differences are model behavior rather than
pipeline drift.

For V1, offline materialization is preferred over runtime image augmentation.
The OpenSpec should define the exact materialized plans, manifests, visual
audits, and training examples before code is written.

Tiny overfit success requires all of the following:

- loss decreases on the relevant painted-input training surface;
- generated compact rows meet the row-validity threshold in the experiment plan;
- training-set mAP and F1-style decoding metrics improve over the frozen
  four-epoch baseline on the same painted evaluation surface;
- correct painted marks outperform no-mark, wrong-mark, and shuffled/offset-mark
  counterfactuals on the same slice;
- visual artifact audit confirms the painted boxes and centers match GT.

The exact pass, hard-fail, and gray-zone thresholds live in the experiment plan.
Gray-zone results require user review before larger training.

Do not treat either experiment family as scientifically meaningful until both
paint-all and teacher-prefix step-pair tiny overfit routes have been run on the
fixed 256-image slice.

## Historical Stepwise Training And Decoding Contract

This section records the original stepwise framing that produced the first
painted-GT and predicted-ledger evidence. It is now superseded for the
human-annotation-style follow-up by the `2026-07-06 Next-Object Steering
Revision` above. New implementation must follow `next_object_steering` for the
text-image and image-only ablation unless a later review explicitly reopens
this decision.

Do not implement from this historical section for the active next-object wave.
Mentions of oracle current-target painting, random schedules, row-only targets,
teacher-prefix/self-prefix step-pair decoding, or fixed `96`-token stepwise
budgets are historical evidence from earlier probes, not the active
`next_object_steering` contract.

The stepwise training and decoding modes must share one semantic unit:

```text
current painted target image
+ assistant-side prefix of previous compact rows
-> exactly one current compact row
```

The first training route should use teacher-prefix step pairs:

1. choose an oracle target schedule, initially `geo_sorted` and random;
2. materialize or construct the image with the current target visually
   emphasized;
3. place previous GT compact rows in the assistant-side prefix;
4. supervise exactly the current object's compact row.

The first decoding/evaluation route should mirror this as teacher-prefix
step-pair decoding:

1. use the same current-target painted image contract;
2. use previous GT compact rows as the assistant-side prefix;
3. generate exactly one row;
4. score the generated row against the current GT target.

This is the first acceptance gate because it tests whether the model can
translate the current painted visual hint when the autoregressive history is
clean.

The harder second decoding route is self-prefix closed-loop rollout:

1. generate an object span or row;
2. append the model's raw generated row text to the assistant-side prefix;
3. parse the emitted row for scoring and failure taxonomy;
4. advance the oracle target schedule and update the current painted target
   image;
5. continue until the oracle schedule length is exhausted.

V1 does not ask the model to choose the next target or decide when to stop. The
controller supplies the target schedule. Stop/EOS and autonomous coverage are
future problems.

Self-prefix rollout must preserve the model's raw generated text in the prefix
instead of appending normalized or salvaged rows. Invalid rows should be scored
and recorded, not silently repaired, because prefix contamination is one of the
failure modes under study.

Both teacher-prefix and self-prefix evaluations should report per-step metrics
and reconstructed image-level metrics. Per-step metrics expose wrong
description, malformed row, bad `x1/y1/x2/y2`, duplicate/local-competitor
errors, and late-step degradation. Reconstructed image-level metrics connect the
stepwise procedure back to detection usefulness.

Self-prefix closed-loop rollout should be implemented after teacher-prefix
step-pair overfit shows signal. Teacher-prefix step pairs are the first gate for
whether the model can use the current painted visual hint under clean history;
self-prefix then tests whether the model survives its own accumulated text.

## Optional Zero-Shot Check

Prompt-only zero-shot exploration beyond the required frozen painted baselines
is allowed but not central.

The main question is not zero-shot painted-input use. The main question is
whether behavior improves after tiny overfitting on painted inputs.

## Frozen And Counterfactual Baselines

Frozen-baseline painted inference is mandatory before tiny painted training for
at least:

- `paint_all`;
- `stepwise_teacher_prefix`.

The same fixed 256-image slice must also include a mark-dependence control panel
for frozen and trained models:

- `painted_correct`: the intended GT-derived mark;
- `unpainted_same_prompt`: same text prompt without the painted mark;
- mode-specific negative painted controls defined in the experiment plan.

For stepwise, negative controls include wrong-object and shuffled/offset marks.
For paint-all, negative controls use all-object offset or style-matched false
marks rather than a single wrong-object target.

For stepwise outputs, the report must compare the generated row against the
current painted target, the next object by schedule, the previous-prefix object,
same-description competitors, local non-marked competitors, and the best image
level match. Improvement without mark dependence is not sufficient evidence for
the central hypothesis.

## Decode Policy

Launch-gate decoding should be deterministic:

- primary decode surface `free_raw_generation`;
- no compact grammar, trie, or row-syntax-constrained decoding in the primary
  launch gate;
- `temperature = 0`;
- no sampling;
- primary repetition penalty `1.10`;
- optional repetition-penalty sensitivity panel `1.05`;
- fixed max-new-token budget per mode;
- strict parser plus raw diagnostics.

Primary pass/fail comparisons must keep decode settings identical across
frozen, trained, and counterfactual rows. Do not tune decoding until the first
gate report proves that baseline reproduction, painted-input metrics, and
mark-dependence controls are coherent.

## Larger Diagnostic Training Gate

Do not launch full-dataset painted training merely because the code runs.

Launch a larger painted-input training run only after the 256-image tiny smoke
shows that the model can learn from the painted inputs through the selected
adapter capacity and produce improved training-set decoding metrics for:

- paint-all;
- teacher-prefix step-pair;
- the required mark-dependence control panel.

Before the larger run, also execute a bounded self-prefix diagnostic. If
self-prefix catastrophically fails through invalid prefixes, mark insensitivity,
or prefix-copy behavior, block the larger run. If self-prefix is weaker than
teacher-prefix but still produces interpretable partial signal, the larger run
may proceed only with the narrower claim: painted visual hints are learnable
under clean or teacher-provided prefix, while closed-loop prefix robustness
remains unresolved.

The larger run is a diagnostic scale-up, not a production detector claim.

## Interpretation Logic

Treat paint-all and stepwise painted rollout as peer probes. Do not assume
either one is easier, harder, more faithful, or more important before seeing the
metrics. The research posture is to let the model behavior reveal which visual
annotation interface is compatible with its internal mechanism.

The primary conclusion table must keep separate rows for:

- `paint_all`;
- `stepwise_teacher_prefix`;
- `stepwise_self_prefix`.

Do not collapse these modes into a single painted-GT score, pick only the best
condition, or average them. They answer different questions.

If painted-input overfitting works, the decoder can use explicit visual
annotation signals, and future work can gradually weaken the oracle signal or
decompose paint-all versus one-by-one behavior.

If paint-all works but one-by-one fails, the model may rely on global annotation
context or output-order priors more than current-object emphasis.

If one-by-one works but paint-all fails, sequential current-object conditioning
may be the easier or more natural route, and all-at-once transcription may be
too hard under prefix or set-size pressure.

If paint-all and stepwise disagree, treat the disagreement itself as a primary
result. Analyze whether the burden is multi-object transcription, clean-history
current-row translation, self-prefix contamination, output ordering, duplicate
pressure, or coordinate-slot degradation.

Remember that V1 marks expose geometry only: bbox rectangle plus center point,
with no category text label. The model still has to recognize the class from
image content. Interpret failures through the separate axes of wrong
description, bad box, coordinate-slot error, and prefix/order pressure.

Paint-all also remains tied to the canonical `geo_sorted` output order unless a
future experiment adds visible order cues. Do not interpret a paint-all result
as an order-free set-prediction result.

If neither works on the 256-image overfit setting, pause the direction before
adding subtler architecture. The failure would suggest that painted visual
evidence is still not enough to overcome the decoding/interface bottleneck.

## Sources

- User discussion in this branch on 2026-07-04.
- CoordExp-Swift authority:
  `docs/COORDEXP_SWIFT.md`
- Prior autoregressive rollout anatomy note:
  `docs/history/superpowers/specs/2026-06-01-autoregressive-object-rollout-anatomy-design.md`
- Prior instance-binding mechanism note:
  `docs/history/superpowers/specs/2026-04-24-qwen3-vl-instance-binding-mechanism-design.md`
