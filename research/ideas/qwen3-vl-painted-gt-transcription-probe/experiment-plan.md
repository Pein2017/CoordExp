---
type: idea
title: Qwen3-VL Painted-GT Transcription Probe Experiment Plan
description: Defines the review-gated evidence contract for painted-GT training, decoding, metrics, and launch gates.
tags: [coordexp-swift, qwen3-vl, painted-gt, experiment-plan, launch-gate]
state: draft
updated: 2026-07-04
---

# Qwen3-VL Painted-GT Transcription Probe Experiment Plan

## Status

This plan is the source-truth experiment contract for the branch
`codex/qwen3-vl-painted-gt-transcription-probe`.

It is a docs/spec/plan artifact. It now authorizes drafting the branch-local
OpenSpec change and Superpowers implementation plan. It does not approve
implementation, smoke runs, production training, or stable OpenSpec promotion
by itself.

## Decision Posture

This plan inherits the branch overview's priorities:

1. accuracy, precision, and research validity;
2. training and execution efficiency;
3. simplicity and avoidance of redundant machinery;
4. scalability after the mechanism is clearer.

Agents may make tactical adjustments without user approval when new evidence
appears, as long as the adjustment preserves the painted-GT transcription
hypothesis, improves validity, and is recorded in a plan, manifest, or review
note. Changing the research question, deleting major artifacts, or broadening
the claim beyond the evidence still requires explicit user approval.

## Objective

Test whether Qwen3-VL can learn to translate obvious GT-derived visual
annotations into compact detection rows strongly enough to overcome
language-prior, prefix-prior, and learned object-count failures.

The branch is diagnostic. Painted GT annotations are oracle visual hints, not an
inference-time detector design.

## Source Baseline

Start from the CoordExp-Swift pure-CE four-epoch baseline adapter:

```text
/data/CoordExp/.worktrees/CoordExp-swift/outputs/prod/coordexp_swift/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_r16a32_llm_12000_accelerate8_ebs64_4epoch_warmup0p1-prod8-r16a32-ebs64-warmup0p1-20260702T170007Z/checkpoints/step-917/adapter
```

The normal unpainted detector reference remains the accepted CoordExp-Swift
val200 run:

```text
/data/CoordExp/.worktrees/CoordExp-swift/outputs/coordexp_swift/infer/val200/qwen3-vl-2b-desc-first-geo-sorted-pure-ce-dora-r16a32-step917-val200-20260703T035007Z
```

This baseline adapter is a read-only external source artifact for this branch.
New painted-probe runs write under the current worktree output root, but every
run manifest must record the absolute source adapter, source inference/eval
reference, resolved config, local artifact root, evaluated adapter checkpoint,
base model path, tokenizer identity, processor identity, generation-config
identity, and special-token id map identity.

Canonical base model path:

```text
/data/CoordExp/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent
```

Canonical source special-token embedding payload:

```text
/data/CoordExp/.worktrees/CoordExp-swift/outputs/coordexp_swift/infer/val200_support/repaired_special_token_embeddings_step917
```

Use the repaired payload rather than the raw checkpoint payload because the
repaired copy preserves the same tensor values while adding the base-config and
tokenizer identity hashes used by the accepted val200 inference reference.

The step-917 adapter is LLM-only DoRA. It is both the frozen detector baseline
and the competence seed for this branch, but it must not be directly resumed as
the trainable all-tower adapter surface through the current strict
`load_existing` path. The painted-training seed strategy is
`warm_start_expand_dora`:

1. load the original pretrained base model from `model_cache`;
2. construct the requested all-tower DoRA adapter surface over `vision`,
   `aligner`, and `language`;
3. copy every expected language-side adapter tensor from the source step-917
   LLM-only adapter into the matching target adapter tensor by exact canonical
   state-dict key mapping, with shape checks as a secondary guard;
4. initialize missing target adapter tensors for `vision` and `aligner` through
   the normal DoRA initialization policy;
5. load the canonical repaired source special-token embedding payload,
   including coordinate tokens and wrapper tokens, into the trainable
   special-token embedding mechanism;
6. keep dense base-model weights frozen unless a later user decision explicitly
   approves dense base-weight training;
7. train only the expanded adapter modules and approved special-token embedding
   parameters.

This keeps the runtime philosophy intact: original pretrained base model plus
adapter and embedding payloads. Do not materialize or save a merged full model as
the primary path. Do not silently fall back to direct `load_existing` all-tower
training from the LLM-only adapter. A direct load of the source adapter with
all-tower target discovery is an adapter-surface mismatch, not a valid
implementation choice.

The run manifest must record source adapter identity, source special-token
embedding identity, base model identity, tokenizer identity, processor identity,
generation-config identity, special-token id-map hash, warm-start report id,
evaluated adapter checkpoint, warm-start tensor copy counts, newly initialized
tensor counts, missing-source tensors allowed by tower, unexpected/missing
source tensors, target towers, target modules, optimizer groups, dense-base
freeze status, and trainable counts.

The warm-start copy report must be key-map based, not shape-match based. It
must enumerate `source_key -> target_key` rows in the canonical PEFT/DoRA key
space, including adapter-name normalization and DoRA magnitude-vector key
normalization. It must reject ambiguous shape-only matches, record tensor
shape and hash or checksum for copied tensors, and verify post-copy equality
for every expected copied language tensor. Missing `lora_A`, `lora_B`, or DoRA
magnitude tensors for an expected language target are launch blockers.
Vision/aligner tensors may be absent from the source only when they are
explicitly listed as newly initialized target tensors.

The expanded all-tower pre-training seed is intended to preserve the source
LLM-only adapter's unpainted competence before painted adaptation. Do not assume
that automatically. Before painted training, run a deterministic unpainted
expanded-seed sanity probe against the source LLM-only seed on the same
base/tokenizer/processor/decode surface. The OpenSpec may choose a tiny sample
or val200 depending on cost, but it must define the sample, tolerated drift,
artifact fields, and block condition. Catastrophic drift blocks painted
training until the warm-start path is repaired.

The authoritative unpainted val200 baseline metric is:

```text
/data/CoordExp/.worktrees/CoordExp-swift/outputs/coordexp_swift/infer/val200/qwen3-vl-2b-desc-first-geo-sorted-pure-ce-dora-r16a32-step917-val200-20260703T035007Z/eval_coco_fixed_gt_scale/metrics.json
```

The key is `mAP`, with accepted value `0.4111788135144427`. The launch gate is
strict: the reproduced or verified unpainted val200 metric must have
`row_count == 200`, `metric_family == coordexp_swift_detection_coco_bbox_v1`,
and `mAP >= 0.40` under the same adapter/decode/eval identity. If the branch
gets `mAP < 0.40`, stop as an infrastructure or artifact-identity failure
unless the user explicitly accepts a new baseline.

Do not use `eval_coco/metrics.json` as the headline reference for this gate;
the accepted reference is the fixed-GT-scale metric above.

## Dataset Slice

Use a fixed deterministic 256-image training slice from the same CoordExp-Swift
training data family as the four-epoch pure-CE baseline.

The slice must be built by a coverage-aware deterministic builder unless a
later recorded review decision replaces it. The manifest must include enough
multi-object, crowded, repeated-class, medium-object-count, and high-object-count
examples to exercise both paint-all and stepwise behavior, while retaining some
simple controls.

Do not use a changing random slice. Do not treat the first 256 rows as adequate
unless they satisfy and record the same coverage intent. The slice manifest must
record:

- source dataset path and checksum or equivalent identity;
- selection seed;
- image ids and source row ids;
- object counts;
- bucket labels;
- builder code/config identity;
- the exact reusable slice id used by all frozen, trained, paint-all, stepwise,
  and counterfactual conditions.

## Shared Painted-Data Primitive

Before serious GPU runs, build one shared painted-image and target
materialization primitive used by both experiment families.

The shared primitive must keep these surfaces aligned:

- painting style;
- original image identity and dimensions;
- GT object identity and object order records;
- box and center provenance;
- compact-row target construction;
- painted JSONL or equivalent materialized example contract;
- visual-audit gallery;
- artifact manifest.

This exists to prevent pipeline drift. Differences between experiment families
must reflect model behavior, not incompatible painting or target construction.

The primitive must also emit enough provenance to audit image and geometry
alignment:

- source image dimensions;
- model-input image dimensions;
- painter coordinate space;
- GT box-to-pixel conversion receipt;
- `do_resize=false` or any other image preprocessing setting that affects
  coordinate interpretation;
- hash or stable id for each painted image plan.

Geometry provenance is an acceptance gate, not only a manifest field. The
painted bbox and center pixels must agree with the supervised coordinate tokens
converted through `src/data/geometry.py::coord_bins_to_pixel_xyxy` in the
no-resize model-input pixel space planned by `src/qwen/images.py`. The OpenSpec
must define the pixel tolerance, automated audit, and visual-gallery spot check.
Do not route this Swift worktree through a nonexistent `src/datasets/geometry.py`
path.

For V1, prefer offline materialization over runtime image augmentation. The
OpenSpec may refine file names, but it should define at least:

- `slice_manifest.json`;
- `painted_plan.jsonl`;
- `schedule_manifest.<schedule_id>.json`;
- `condition_manifest.json`;
- `visual_audit/index.json`;
- materialized training JSONL or an equivalent branch-local painted example
  contract that converts into the existing training path.

Runtime dynamic augmentation remains a later option if offline materialization
cannot represent the experiment cleanly.

The OpenSpec must also pin the artifact root policy for this branch. The
current worktree may use an `outputs` symlink, so manifests must store resolved
absolute artifact roots rather than relying on ambiguous "local outputs"
wording.

## Painting Policy

Use maximal, obvious visual salience for V1:

- high-contrast bbox rectangle;
- high-contrast center point;
- no category text label;
- no visible numeric object IDs.

This is a ceiling test. If the model fails, the failure should not be because
the mark was visually too subtle.

Category text labels remain out of V1 because they turn the task into OCR or
label copying as much as visual grounding. They may become a later upper-bound
control only after this branch establishes the no-text painted setting.

## Experiment Families

### Paint-All

Paint all GT objects in the image and run a standard compact-row full-response
generation path.

Target response:

- all GT compact rows;
- canonical `geo_sorted` order unless a later experiment adds visible order
  cues.

Paint-all is not assumed to be simpler than stepwise painted rollout. It is a
peer probe measuring multi-object painted transcription.

### Stepwise Teacher-Prefix

Prepare one current target at a time:

```text
current painted target image
+ assistant-side prefix of previous GT compact rows
-> exactly one current compact row
```

Target schedules:

- `geo_sorted`;
- random.

V1 uses one frozen random schedule per image as a single-seed diagnostic
contrast. It must not be interpreted as order-robustness evidence. Any
order-robustness claim requires additional recorded random schedules.

Each schedule must be recorded in a deterministic schedule manifest with:

- `schedule_id`;
- seed for random schedules;
- original GT object ids and order;
- canonical `geo_sorted` order;
- frozen random order;
- step index to target object id mapping;
- reuse rules across frozen, trained, teacher-prefix, self-prefix, and
  counterfactual runs.

This is the first stepwise gate. It tests current-row visual translation with a
clean autoregressive history.

The training representation must prevent silent prefix supervision. Previous GT
compact rows are conditioning text only. Their token labels must be ignored in
the loss. Only the current object's compact row is label-bearing. The OpenSpec
must pin the chat/message representation, assistant-span boundaries, label mask,
EOS/stop handling, and generation prompt state. A materialized sample dump must
show token text, input ids, labels, ignored prefix labels, current-row labels,
and generation prompt bytes for at least one stepwise example.

### Stepwise Self-Prefix

After teacher-prefix shows signal, run closed-loop self-prefix evaluation:

1. generate exactly one object row for the current target step;
2. append the model's raw generated row text to the assistant-side prefix;
3. parse the emitted row for scoring and failure taxonomy;
4. advance the oracle target schedule;
5. update the current target painted image;
6. continue until the oracle schedule length is exhausted.

The controller supplies the target schedule. V1 does not ask the model to choose
the next target or decide when to stop.

Do not append normalized or salvaged rows to the prefix. Prefix contamination is
one of the mechanisms under test.

Before larger diagnostic training, run a bounded self-prefix diagnostic. If
self-prefix catastrophically fails through invalid prefixes, mark-insensitive
outputs, or prefix-copy behavior, the larger run is blocked. If self-prefix is
weaker than teacher-prefix but still yields interpretable partial signal, the
larger run may proceed only with a narrower claim: painted visual hints are
learnable under clean or teacher-provided prefix, while closed-loop prefix
robustness remains unresolved.

## Training Modes

Train painted-input adaptation from the `warm_start_expand_dora` seed. The
source LLM-only baseline adapter supplies the language-side competence seed and
the frozen comparison baseline. The trainable painted adapter is an expanded
all-tower adapter whose matching language tensors are warm-started from the
source adapter and whose missing vision/aligner tensors are newly initialized.

For paint-all, training uses painted all-object images with full compact-row
targets.

For stepwise, the primary training route uses teacher-prefix step pairs. This
keeps training and the first stepwise decoding benchmark semantically aligned.

Prefix-free single-object training is only an ablation. It is not the primary
one-by-one route because it does not test current painted target behavior under
an autoregressive prefix.

The branch should maximize painted-input adaptation capacity by enabling DoRA
over all linear modules across the vision tower, MLP aligner, and language
tower. This means adapter-based trainable adaptation across those towers, not
full dense base-weight finetuning unless the user later approves that heavier
path explicitly.

Before any real research run, upgrade the training infrastructure to support
`warm_start_expand_dora`. This infrastructure work is part of the branch plan
and must pass its own tests, receipts, and review gate before tiny painted
training.

Every training run must save an adapter-load, warm-start, and trainable-surface
receipt. The receipt must include source adapter path, source special-token
embedding path, base model path, tokenizer/processor/generation identities,
evaluated adapter checkpoint when present, copied tensor names/counts,
source-to-target key map, copied tensor hashes, post-copy equality status,
initialized tensor names/counts, missing/unexpected adapter keys, dense-base
trainability status, DoRA target module patterns, optimizer groups, and
trainable parameter counts by `vision`, `aligner`, `language`,
coordinate-token embeddings, and wrapper-token embeddings. Fail before GPU
launch if:

- an expected warm-start language adapter tensor is missing or shape-mismatched;
- a source tensor is silently ignored without being listed as expected or
  unexpected;
- any intended tower has zero matched trainable adapter parameters;
- coordinate-token or wrapper-token embedding trainability is missing from the
  receipt;
- the canonical repaired special-token embedding payload is not loaded before
  optimizer grouping;
- dense base weights are trainable without explicit approval;
- optimizer parameter groups do not cover every trainable parameter exactly
  once.

## Frozen Baselines And Mark-Dependence Controls

Frozen-baseline painted inference is mandatory before tiny training for:

- `paint_all`;
- `stepwise_teacher_prefix`.

Prompt-only or broader zero-shot exploration is optional. The mandatory frozen
baselines exist to produce a before/after contrast on the exact same painted
evaluation surfaces.

For both frozen and trained models, the 256-image gate must include this
mark-dependence control panel, with separate semantics by mode:

- `painted_correct`: intended GT-derived mark.
- `unpainted_same_prompt`: same text prompt without any painted mark.
- Stepwise `wrong_object_mark`: mark points to a different object while the
  target row remains the current scheduled object.
- Stepwise `shuffled_or_offset_mark`: spatially shuffled, displaced, or
  otherwise wrong mark with similar superficial painting style.
- Paint-all `offset_all_marks`: all marks are style-matched but displaced away
  from their GT boxes.
- Paint-all `style_matched_false_marks`: style-matched false boxes/points are
  placed on non-object regions when possible.
- Paint-all optional `partial_or_omitted_marks`: a recorded upper/lower-bound
  diagnostic, not required for the first gate.

The negative-control manifest must define how each wrong mark is selected. For
stepwise, prefer the hardest available negative in this order:

1. same-description competitor;
2. local or overlapping/nearby competitor;
3. next object by schedule;
4. previous-prefix object;
5. deterministic easy fallback.

If no meaningful hard negative exists, record `no_hard_negative_available` and
scope the claim for that example. For offset controls, keep the mark in image
when possible, preserve approximate box size/style, avoid GT overlap when
possible, and record IoU to the nearest GT object.

For stepwise metrics, compare each generated row against:

- the current painted target;
- the next object by schedule;
- the previous-prefix object;
- same-description competitors;
- local non-marked competitors;
- best image-level match.

Correct painted marks must meet the mark-dependence margins in the launch gate
before the branch can claim the visual mark has causal control over decoding.

## Decode Policy

Launch-gate decoding must be deterministic:

- primary decode surface `free_raw_generation`;
- no compact grammar, trie, or row-syntax-constrained decoding in the primary
  launch gate;
- `temperature = 0`;
- no sampling;
- primary repetition penalty `1.10`;
- optional sensitivity repetition penalty `1.05`;
- fixed max-new-token budget per mode;
- explicit stop/EOS policy per mode;
- strict parser plus raw diagnostics.

Pass/fail comparisons must use one identical decode setting across frozen,
trained, and counterfactual rows. The primary launch-gate setting is
`decode_surface=free_raw_generation`, `temperature=0`, no sampling, and
`repetition_penalty=1.10`. If `1.05` is run, it is a sensitivity panel, not
mixed into the primary gate. Compact grammar, trie constraints, stop-policy
experiments, or other generation constraints may be run only as separately
labeled sensitivity panels; their row-validity numbers are not evidence of
native output-format competence for the primary launch gate. Reports must fail
if paired comparison rows differ in decode settings except for the intended
model state or condition axis.

Do not tune decoding until baseline reproduction, painted-input metrics, and
mark-dependence controls are coherent.

The OpenSpec must define a `decode_surface` identity table used by every
condition manifest and comparison report. At minimum it must include free vs
constrained generation, grammar/trie flags, stop-string and EOS behavior,
allowed token constraints, max-new-token semantics, parser version, raw-output
retention, repetition penalty, temperature, sampling flags, and decode hash.

## Tiny Overfit Run

The first GPU gate uses the fixed 256-image training slice.

Train for at least two epochs. The first gate uses a predeclared budget rather
than unbounded checkpoint fishing: plan `2` epochs first, allow extension up to
`8` total epochs only when the intermediate report shows improving but
not-yet-passing behavior, and require a recorded review note before any further
extension. The gate report must list planned epochs, actual epochs, eval
cadence, every evaluated checkpoint, full checkpoint metrics, loss trajectory,
selected-checkpoint rule, selected-checkpoint reason, and selected-checkpoint
provenance. Passing from a single best checkpoint is invalid unless the full
planned checkpoint table and provenance fields are reported.

For stepwise, the effective training sample count may scale with the total
object count because each object can become a prediction target or forward.
This compute cost is accepted for this research worktree.

## Metrics

Report paint-all, stepwise teacher-prefix, and stepwise self-prefix separately.
Do not average them, choose only the best condition, or collapse them into one
painted-GT score.

Predeclared primary gate metrics:

| mode | primary gate metric | primary scope | secondary metrics |
| --- | --- | --- | --- |
| `paint_all` | `debug_detection_f1` | fixed 256-image training slice, invalid/unparseable rows retained in denominators | mAP, mRecall, row validity, duplicate rows, missing GT rows, coordinate-slot breakdown |
| `stepwise_teacher_prefix` | `per_target_step_debug_f1` | fixed 256-image object-step schedule, invalid/unparseable steps retained in denominators | reconstructed image-level mAP/F1, per-step IoU, late-step degradation, competitor breakdown |
| `stepwise_self_prefix` | `per_target_step_debug_f1` for interpretable comparison; hard-safety block metrics for larger-run launch | bounded self-prefix subset, invalid/unparseable steps retained in denominators | reconstructed image-level mAP/F1, prefix-contamination rate, schedule-completion rate |

The OpenSpec may refine exact field names, but it must preserve the meaning:
one primary score per mode before results are seen, denominator-bearing invalid
outputs, and mark-control margins applied to that same primary score. Official
detection mAP remains reported but is not allowed to replace the predeclared
debug primary score for tiny-launch decisions.

Paint-all metrics:

- training-set mAP;
- mRecall;
- F1-style detection score;
- row validity;
- missing GT rows;
- duplicate rows;
- wrong descriptions;
- valid descriptions with bad boxes;
- coordinate-slot breakdown for `x1`, `y1`, `x2`, and `y2`.

Stepwise metrics:

- per-step row validity;
- per-step description correctness;
- per-step box IoU;
- per-step coordinate-slot errors;
- step index and late-step degradation;
- duplicate or local-competitor error;
- reconstructed image-level mAP;
- reconstructed image-level F1-style detection score.

Use per-step metrics as the diagnostic source of truth for stepwise behavior.
Use reconstructed image-level metrics to connect the controller procedure back
to detection usefulness.

The current CoordExp-Swift V1 evaluator should not be assumed to already expose
all debug surfaces listed here. The branch-local OpenSpec must define any new
metric, artifact, parser, and report contracts needed for painted-probe
evaluation. Keep official detection metrics separate from debug F1, per-step,
counterfactual, and reconstructed-image reports.

The OpenSpec must define debug F1 before it is used for a gate. At minimum,
define the precision denominator, recall denominator, IoU or matching rule,
description-match rule, invalid-row handling, duplicate handling, primary score
field name, metric scope, threshold source, and whether a counterfactual score
is official, debug-only, per-step, or reconstructed-image.
Malformed, invalid, and unparseable outputs must remain in diagnostic
denominators as failures or misses. They may be excluded from a parseable-row
subtable, but they must not shrink the recall, target-step, or debug-F1
denominator used for launch decisions.

## Failure Taxonomy

At minimum, classify failures as:

- malformed row;
- wrong description;
- valid description with bad box;
- coordinate-slot failure, preserving `x1`, `y1`, `x2`, and `y2` separately;
- duplicate or local-competitor prediction;
- missing target;
- prefix contamination;
- late-step degradation;
- mark ignored;
- wrong mark followed;
- next-by-order prior;
- previous-prefix object prediction;
- same-description competitor;
- local non-marked competitor;
- mark-target geometry shift.

## Diagnostic Artifact Contract

Every inference/eval condition must preserve enough raw evidence to diagnose
failure instead of only saving scored predictions.

At minimum, artifacts must record:

- condition name;
- base model path and identity;
- source adapter path;
- evaluated adapter checkpoint or model artifact root;
- warm-start report id when applicable;
- source special-token embedding payload path and identity;
- resolved config;
- decode settings, including repetition penalty;
- decode surface, stop/EOS policy, parser version, and decode hash;
- slice id and schedule id;
- image id and target object id when applicable;
- painted image plan id and visual-audit path;
- raw prompt or assistant-side prefix before generation;
- raw generated suffix;
- raw prefix after append for self-prefix;
- parse status and parse error when present;
- normalized prediction only if parseable;
- scoreability flag;
- dropped, invalid, malformed, and unparsed counters.

Self-prefix artifacts must preserve raw generated text in the continued prefix.
Salvaged or normalized rows may be scored separately, but they must not replace
the raw prefix used for the closed-loop test.

Frozen self-prefix is optional only for the narrow hard-safety question, "does
the trained model catastrophically fail closed-loop prefixing under this
controller?" Any claim about self-prefix improvement, degradation, or
training-caused prefix robustness requires paired frozen and trained
self-prefix rows with identical slice, schedule, source, and decode identities.
If the frozen self-prefix baseline is skipped, the report must set
`no_self_prefix_baseline: true` and disable comparative language.

Comparison reports must include at least:

- `model_state`;
- `condition`;
- `mode`;
- `slice_id`;
- `schedule_id`;
- `decode_hash`;
- `repetition_penalty`;
- `max_new_tokens`;
- `parser_status`;
- `metric_scope`;
- source adapter path;
- source dataset identity.

The gate report must refuse to compare rows whose slice, schedule, source
adapter, or primary decode identity differs when those fields are expected to be
paired.

## Tiny Success Gate

Tiny overfit success requires all of:

- loss decreases on the relevant painted-input training surface;
- generated compact rows meet the row-validity threshold;
- training-set mAP and F1-style decoding metrics improve over the frozen
  baseline on the same painted evaluation surface;
- visual artifact audit confirms painted boxes and centers match GT;
- correct painted marks meet the required margin over no-mark, wrong-mark, and
  shuffled/offset controls on the same slice;
- both paint-all and stepwise teacher-prefix routes show learnability on the
  fixed 256-image slice.

Do not require val200 improvement for this gate. Do not require near-perfect
memorization as the first pass. Do require visible behavioral improvement
beyond loss decrease.

Use a three-zone gate:

- `hard_fail`: unpainted val200 baseline identity fails; visual audit finds
  geometry/image misalignment; trained row validity is below `0.50`; malformed
  or unparsed rows exceed `0.50`; painted-correct does not improve over the
  frozen painted baseline on the predeclared primary gate metric; or correct
  marks beat the best negative control by less than `0.02` on the predeclared
  primary gate metric for that mode.
- `pass`: visual audit passes; row validity is at least `0.80`; malformed and
  unparsed rows are at most `0.20`; the predeclared primary gate metric improves
  by at least `0.05` absolute over the frozen painted baseline for both
  paint-all and teacher-prefix; and painted-correct beats each required
  negative control by at least `0.10` absolute on the predeclared primary gate
  metric for that mode. If paired bootstrap or sign tests are implemented, the
  positive direction must also hold in the paired test.
- `gray_zone`: any result between hard-fail and pass, including row validity
  between `0.50` and `0.80`, metric gain below `0.05`, mark-control margin
  between `0.02` and `0.10`, missing scoreability policy metadata, or sparse
  hard negatives. Scoreless debug-F1 outputs remain eligible when the report
  explicitly records emission-order fallback. Gray-zone results require user
  review before larger training.

## Hard Failure Gate

Treat the tiny run as a hard failure if loss decreases but decoded outputs
remain invalid, ignore markings, or fail to improve over the frozen baseline on
the same painted surface.

If this hard failure occurs for both paint-all and stepwise teacher-prefix,
pause before adding subtler mechanisms or launching larger training.

## Larger Diagnostic Training Gate

Do not launch full-dataset painted training merely because the code runs.

Launch larger painted-input training only after the 256-image tiny smoke shows
improvement for both:

- paint-all;
- stepwise teacher-prefix.

The gate must also include the mark-dependence control panel. Improvement
without correct-mark advantage over no/wrong/shuffled marks is not enough.

Stepwise self-prefix closed-loop behavior must be checked before this larger
run. Catastrophic self-prefix invalidity, prefix-copy, or mark insensitivity
blocks larger training. Partial self-prefix weakness may still allow the larger
run only under a scoped claim: clean-prefix painted transcription is promising,
but closed-loop prefix robustness remains unresolved.

The bounded self-prefix diagnostic uses a deterministic subset of the fixed
256-image slice: `64` images or `512` object steps, whichever is smaller, unless
the OpenSpec chooses an equivalent stricter bound. Use the primary decode
setting. Include a frozen self-prefix reference when feasible; if it is skipped,
the gate report must say so and scope the comparison.

Self-prefix blocks larger training when row validity is below `0.50`, malformed
or unparsed rows exceed `0.50`, correct-mark advantage is at most `0.02`,
previous-prefix or next-by-order errors dominate more than `0.30` of steps, or
raw-prefix contamination prevents completion of more than `25%` of schedules.
Self-prefix may produce `narrow_proceed` only when teacher-prefix passed, the
self-prefix report is interpretable, and the claim is explicitly limited to
clean or teacher-provided prefix robustness.

The intended larger run is two additional epochs from the source baseline
adapter, using the approved painted-input adaptation setup.

This larger run is a diagnostic scale-up, not a production detector launch.

## Interpretation Rules

Paint-all and stepwise painted rollout are peer probes. Do not assume either is
easier, harder, more faithful, or more important before seeing the metrics. Let
the model behavior reveal which annotation interface matches its internal
mechanism.

If paint-all and stepwise disagree, the disagreement is a primary result.
Analyze whether the burden is multi-object transcription, current-row visual
translation, clean-history dependence, self-prefix contamination, duplicate
pressure, output ordering, or coordinate-slot degradation.

The V1 mark supplies location evidence, not category text. Positive and negative
interpretations must distinguish coordinate transcription from class
recognition. A wrong-description failure is not the same as a bad-box failure,
and a classification-bound null result must not be summarized as pure coordinate
transcription failure.

Paint-all remains entangled with the canonical `geo_sorted` output-order prior
because all objects are painted simultaneously and no visible per-object order
cue is introduced. V1 paint-all results are therefore multi-object painted
transcription under a known ordering prior, not an order-free set prediction
claim.

Stepwise self-prefix rollout is sequential within each image because each step
depends on the previous raw generated row. The OpenSpec must pin the controller
serialization boundary, object-step cap, and max-new-token budget per step
before launch.

If neither route learns on the 256-image overfit setting, pause the direction
before learned slots, coverage state, rollout training, or subtler visual
interventions.

## Review And Approval Gates

This plan must pass a review-convergence loop before implementation begins.

Post-approval sequence is mandatory:

1. create an OpenSpec change for the branch-local painted-GT data, training,
   decoding, artifact, and evaluation interfaces;
2. draft a Superpowers implementation plan from the accepted OpenSpec;
3. review the OpenSpec and implementation plan with independent subagents until
   no unresolved P0/P1 findings remain;
4. implement;
5. run targeted tests and tiny smokes;
6. either stop blocked with concrete reasons or launch larger two-epoch
   painted-input training after passing gates.

Run a review-convergence loop after each important phase: research docs,
OpenSpec, Superpowers plan, implementation slices, tiny-smoke launch gate, and
larger-run launch gate.

The Superpowers implementation plan must be saved under
`docs/superpowers/plans/` and must be detailed enough for subagent-driven
development. It must not replace OpenSpec where stable config, artifact,
training, decoding, or metric interfaces are being introduced.

Reviewer timeout, disconnection, missing evidence, or vague reviewer output is
unresolved, not approval. Each phase review report must record reviewer status,
scope, verdict, accepted P0/P1/P2 findings, rejected findings with reason, and
the exact next gate.

The next OpenSpec must introduce branch-local owner surfaces without mutating
the normal CoordExp-Swift full-response inference path by accident. It should
define owners for:

- painted slice, painting, counterfactual, and schedule materialization;
- paint-all full-response inference over painted examples;
- stepwise teacher-prefix and self-prefix controller behavior;
- parser/eval artifacts for raw, normalized, per-step, reconstructed, and
  counterfactual reports;
- all-tower DoRA config, optimizer groups, adapter-load report, and trainable
  surface receipt;
- canonical artifact names and validators for slice/schedule/decode identity.

## Current Stop State

This document authorizes OpenSpec drafting and Superpowers implementation-plan
drafting. It is not implementation approval, GPU launch approval, or larger-run
approval.
