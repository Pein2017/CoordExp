---
title: Masked Spatial Policy and Accepted-Row Prefix Policy Disentanglement
description: Compares masked-input spatial policy, accepted-row prefix policy, native-scale tiling, and matched-call resampling in dense Qwen3 Vision-Language detection rollouts.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: complete_for_unit
unit_id: 2026-07-13-spatial-scope-history-disentanglement
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
tags:
  - coordexp-infras
  - research-unit
  - qwen3-vl
  - dense-enumeration
  - spatial-policy
  - accepted-row-prefix-policy
  - no-resize
updated: 2026-07-13
---

# Masked Spatial Policy and Accepted-Row Prefix Policy Disentanglement

The exact frozen scientific and execution semantics are owned by the
[readiness amendment](readiness-amendment.md). That immutable pre-output
contract passed focused review. Both required seed-root executions are now
complete and locally verified; [the results record](results.md) owns their
artifact handles, observed facts, bounded verdict, and next discriminator.
Pre-execution language below is retained as the frozen protocol rather than
rewritten after observing results.

The active long-running research goal is owned by Codex task
`019f4a19-d81c-75a2-84b0-2c20379e686e`. Its valid terminal outcomes include a
supported result, a falsified hypothesis, unsafe output expansion, an
underpowered denominator, or unsupported runtime semantics. None of those
outcomes promotes an architecture.

## Terminology and Name Registry

- **Qwen3-VL — Qwen3 Vision-Language**: the pretrained multimodal model family
  under investigation.
- **COCO-80 — Common Objects in Context 80-category ontology**: the closed set
  of reportable categories for this unit.
- **DoRA — Weight-Decomposed Low-Rank Adaptation**: the adapter method used by
  the frozen checkpoint.
- **CE — cross-entropy**: the ordinary next-token training objective referred
  to by the checkpoint comparison.
- **Gaussian Soft-Target Coordinate Cross-Entropy with Ordered
  Cumulative-Distribution Penalty**: the coordinate auxiliary used by the
  primary checkpoint. It combines Gaussian soft-target cross-entropy with a
  squared cumulative-distribution discrepancy over the ordered coordinate
  bins. This is the canonical research name; it deliberately has no
  abbreviation.
- **Bounding box (`bbox`)**: the four-coordinate rectangular localization
  emitted for one object.
- **Repetition penalty**: the decode-time token-logit adjustment applied to
  token identifiers already present in the processor's seen-token set. A value
  of `1.0` is neutral. The historical shorthand `rp` may appear in existing
  paths or receipts but is not a canonical research term.
- **Validation-200 (`val200`)**: the fixed 200-image validation cohort used by
  the existing inference artifacts.
- **K spatial calls (`K`)**: the number of cells in the selected spatial grid;
  a `4 x 4` grid has `K=16` calls.
- **Experimental arm (`A`)**: one fully specified input, history, call-budget,
  decode, and aggregation condition in the matrix below.
- **Reference object (`o`)**: one reportable object instance from the explicitly
  named official-annotation or audit-augmented reference ledger.
- **Probability (`P`)**: an empirical conditional fraction over paired audited
  objects, not an assumed parametric probability model.
- **Core region**: the disjoint spatial cell that owns predictions whose
  bounding-box centers fall inside it.
- **Halo region**: clipped context surrounding a core region; it supplies
  pixels for recognition but never owns a prediction.
- **Per-region reset**: decode each spatial input from the same fresh prompt,
  without rows accepted from earlier spatial inputs.
- **Cumulative accepted-row prefix policy**: after each spatial call, admit only
  rows that pass the frozen parsing, ontology, ownership, and validity rules;
  serialize those rows through the frozen canonical global-coordinate template
  into the next call's prompt. This is a total prompt-state intervention, not a
  pure history-length manipulation.
- **Independent bagging**: repeat full-image decoding from the same base prompt
  with the frozen K-seed vector, then apply the same merge policy used by the
  spatial arms. It requires a predeclared nonzero sampling temperature; repeated
  deterministic decoding is not bagging.
- **Baseline seed**: the immutable pseudo-random seed used only by
  `FULL_SINGLE`. It is not a member of the `K`-seed vector, preventing the
  rescue denominator from conditioning one owning-seed comparator to failure.
- **K-seed vector**: the immutable list of pseudo-random seeds indexed by
  canonical spatial-cell identity. Each spatial input keeps its cell's seed
  even when traversal order is permuted; full-image bagging uses the same seed
  multiset in canonical order.
- **Per-call decode cap**: the maximum number of generated tokens allowed in one
  model invocation.
- **Merge policy**: the frozen deterministic procedure that normalizes classes,
  maps boxes to global coordinates, and combines overlapping predictions after
  all calls finish.
- **Matched-call comparison**: two arms use the same number of invocations,
  per-call decode cap, and declared K-seed vector. This does not claim identical
  total computation or the same number of sampled opportunities per object when
  image-token, prompt, or spatial-ownership conditions differ; those realized
  budgets are reported separately.
- **Owning-seed per-object comparison**: for reference object `o` owned by
  canonical cell `j`, compare the raw spatial-call outcome from cell `j` with
  the raw full-image bagging outcome generated with the same seed `j`. This
  equalizes one sampled opportunity for that object without claiming equal
  visual input.
- **Raw union**: the set of distinct reference objects detected in any raw call
  before object-level merge. It exposes repeated opportunities separately from
  aggregation behavior.
- **Local Rescue Rate (`LRR`)**: the probability that an arm detects an object
  missed by the paired Full-Image Single Rollout.
- **Average Precision (`AP`)**: the standard precision-recall detection metric
  under a fully named evaluator, category policy, crowd policy, and object-score
  source; it is secondary to manual object-level rescue in this unit.
- **Bounding-box Intersection over Union (`IoU`)**: intersection area divided by
  union area for a predicted and reference bounding box.
- **Official-Annotation Local Rescue Rate**: Local Rescue Rate computed only on
  original Common Objects in Context annotations.
- **Audit-Augmented Local Rescue Rate**: Local Rescue Rate computed on the
  original annotations plus arm-blind image-review additions sealed before any
  experimental-arm output is inspected.
- **Policy-Utility Rescue Difference**: a paired difference between post-merge
  Local Rescue Rates. It evaluates the final multi-call policies and does not
  localize an internal mechanism.
- **Owning-Seed Raw Rescue Difference**: a paired difference between raw
  one-opportunity detection rates for an object's owning spatial call and its
  seed-matched full-image call, conditioned on the paired Full-Image Single
  Rollout missing that object.
- **Image-Clustered Bootstrap Confidence Interval**: an uncertainty interval
  obtained by resampling images, while keeping every object and arm result from
  a selected image together.
- **Retention**: the fraction of reference objects detected by `FULL_SINGLE`
  that remain detected by another arm.
- **Natural closure**: a rollout ends through the model's normal terminal
  no-more-objects decision rather than token cap, controller cap, error, or
  forced truncation.
- **Mask-harm retention floor**: the predeclared minimum retention for
  `FULL_SINGLE`-detected objects wholly inside a visible mask core. Failure means
  a null masked result cannot be interpreted against spatial restriction.
- **Minimum meaningful effect**: the smallest paired rescue difference declared
  scientifically relevant before outputs are inspected.
- **Superiority**: the complete lower bound of the paired 95-percent
  Image-Clustered Bootstrap Confidence Interval exceeds the frozen minimum
  meaningful effect and every required safety guardrail passes.
- **Equivalence**: the complete paired 95-percent Image-Clustered Bootstrap
  Confidence Interval lies inside the frozen symmetric equivalence margin.
  Absence of statistical significance is not equivalence.
- **Noninferiority guardrail**: a predeclared confidence-bound test showing that
  retention, precision, validity, natural closure, duplicates, or prediction
  count has not worsened beyond its allowed margin.
- **Safely exceeds**: superiority on the named rescue estimand together with all
  required noninferiority guardrails. Any other pattern is `unresolved` or an
  explicitly unsafe rescue gain.
- **Prediction-set diversity**: one minus the mean pairwise Jaccard similarity
  of pre-merge reference-object identifier sets detected by raw bagging calls,
  averaged over images with a nonempty union. The readiness amendment freezes
  the empty-union rule and minimum floor.
- **Weak-diversity repeated-sampling control**: a sampled full-image multi-call
  arm whose Prediction-Set Diversity falls below the frozen floor. It remains a
  descriptive control but cannot falsify a spatial policy.
- **Manual unique recall**: accepted audit-ledger objects matched at least once
  by final valid predictions divided by accepted audit-ledger objects.
- **Manual precision**: final valid predictions matched one-to-one to accepted
  audit-ledger objects divided by matched plus unmatched final valid
  predictions after the frozen crowd and uncertainty-ignore operations.
  Malformed rows are excluded from this denominator and enter the invalid-row
  safety metric; ignored predictions are reported separately.
- **Strict duplicate component**: a connected component of valid predictions
  under identical normalized class and pairwise bounding-box Intersection over
  Union at least `0.85`; it is computed separately before and after object
  merge.
- **Strict duplicate rate**: the number of valid predictions beyond the
  highest-ranked representative in every strict duplicate component, divided
  by all valid predictions in the named pre-merge or post-merge view.
- **Invalid-row rate**: malformed or semantically invalid attempted object spans
  divided by all attempted object spans; a separately named invalid-call rate
  uses attempted calls as its denominator.
- **Natural-closure rate**: calls ending through the model's normal terminal
  no-more-objects decision divided by all attempted calls, with token-cap,
  controller-cap, error, and forced-truncation endings kept separate.
- **Prediction-count inflation**: total final valid predictions in the named
  arm divided by total final valid predictions in `FULL_SINGLE` on the same
  scope; its allowed margin is frozen before execution.
- **Merge-created or merge-destroyed reference match**: a reference-object
  match absent before but present after merge, or present before but absent
  after merge, using the same frozen one-to-one matcher.
- **Sampling-enabled generation (`do_sample`)**: the backend generation switch
  that activates stochastic token sampling; it must be `true` for independent
  bagging.
- **No-resize processor switch (`do_resize`)**: the image-processor option that
  must be `false`; any geometric resize invalidates the unit.
- **Red-Green-Blue color space (`RGB`)**: the three-channel input pixel space in
  which masking and padding values are computed before processor normalization.
- **Unsigned 8-bit integer pixels (`uint8`)**: original pixel storage with
  integer values from 0 through 255.
- **32-bit floating-point arithmetic (`float32`)**: the precision used to
  calculate per-image channel means before deterministic rounding back to
  unsigned 8-bit integer pixels.
- **Visible merged-token support count**: the number of executed merged visual
  tokens whose receptive-field centers lie inside the visible core-plus-halo
  region. The exact minimum required for the eight-by-eight panel is frozen
  before execution.
- **Common Objects in Context crowd flag (`iscrowd`)**: source annotation
  provenance that marks crowd regions with official ignore semantics rather
  than ordinary individual-instance semantics.
- **STOP — terminal no-more-objects decision**: the model action that ends
  object enumeration.

Arm names:

- **Full-Image Single Rollout (`FULL_SINGLE`)**: one sampled rollout over the
  complete image using the independent baseline seed and otherwise the same
  decode policy as the multi-call primary arms. Existing greedy artifacts
  remain motivation only.
- **Full-Image K-Rollout Independent Bagging (`FULL_BAG_K`)**: `K` independent
  complete-image rollouts followed by the frozen merge policy.
- **Native-Scale Tile with Per-Tile Reset (`TILE_RESET`)**: unresized
  core-plus-halo tiles, each decoded from a fresh prompt.
- **Native-Scale Tile with Cumulative Accepted-Row Prefix
  (`TILE_CUMULATIVE_EXPLORATORY`)**: the same tiles decoded while retaining
  accepted rows. It is exploratory and disabled until a coherent tile/global
  coordinate contract is explicitly validated.
- **Native-Scale Tile Core-Only Reset (`TILE_RESET_CORE_ONLY`)**: unresized core
  tiles without halo, decoded from fresh prompts on the audited subset.
- **Full-Canvas Masked Region with Per-Region Reset (`MASK_RESET`)**: full-size
  masked canvases, each decoded from a fresh prompt.
- **Full-Canvas Masked Region with Cumulative Accepted-Row Prefix
  (`MASK_CUMULATIVE`)**: the same masked canvases decoded through the cumulative
  accepted-row prefix policy in one shared global image-coordinate frame.
- **Full-Canvas Masked Core-Only Reset (`MASK_RESET_CORE_ONLY`)**: the reset
  masked-canvas arm without halo context.
- **Full-Canvas Masked Cumulative Counterbalanced Order Panel
  (`MASK_CUMULATIVE_ORDER_PANEL`)**: the cumulative masked-canvas arm evaluated
  under multiple predeclared, counterbalanced cell orders on the audited subset.
- **Full-Canvas Masked Reset Counterbalanced Order Control
  (`MASK_RESET_ORDER_PANEL`)**: the reset masked-canvas arm evaluated under the
  same order panel to detect order-dependent mechanics unrelated to prefix
  accumulation.

Hypothesis names:

- **Masked-Input Spatial Restriction Utility
  (`HYPOTHESIS_MASKED_SPATIAL_POLICY`)**: exposing only one core-plus-halo region
  on a full-size canvas increases object rescue or final policy utility relative
  to the declared full-image controls. This is the directly tested hypothesis.
- **Post-Vision Spatial Candidate Competition
  (`HYPOTHESIS_POST_VISION_COMPETITION`)**: simultaneous object candidates in a
  fixed full-image visual representation suppress valid continuations. This is
  a deeper competing mechanism that the current pixel-input intervention does
  not directly test.
- **Accepted-Row Prefix Policy Interference
  (`HYPOTHESIS_ACCEPTED_ROW_PREFIX_POLICY`)**: the declared cumulative
  accepted-row prompt policy harms later object rescue relative to per-region
  reset. It bundles length, content, correctness, order, and visibility
  consistency; it is not a pure history-horizon hypothesis.
- **History-Horizon Instability (`HYPOTHESIS_HISTORY_HORIZON`)**: prefix length
  alone degrades retrieval or continuation when content, correctness, order,
  visual input, and decode processors are controlled. This remains a later
  discriminator rather than a direct claim of this unit.
- **Tile-Local Computation Explanation
  (`HYPOTHESIS_TILE_LOCAL_COMPUTATION`)**: fewer image tokens or a locally
  encoded visual frame, rather than spatial scheduling alone, explains tile
  gains. This unit intentionally treats those two causes as a residual family;
  a positive result requires a later orthogonal discriminator.
- **Partition and Boundary Context (`HYPOTHESIS_CONTEXT_BOUNDARY`)**: object
  cuts or lost surrounding context explain differences between core-only and
  core-plus-halo conditions.
- **Geometry-Sorted Traversal Prior (`HYPOTHESIS_GEOMETRY_ORDER`)**: raster
  cell order helps because it agrees with geometry-sorted adapter training.
- **Dense-Scene Annotation Gaps (`HYPOTHESIS_ANNOTATION_GAPS`)**: new unmatched
  predictions are real but unlabeled reportable objects.
- **Residual Recognition or Control-State Limit
  (`HYPOTHESIS_RESIDUAL_LIMIT`)**: objects missed by all conditions remain
  outside reliable recognition or object-specific decoder control.

Invariant names:

- **No-Resize Scale Invariant (`INVARIANT_NO_RESIZE_SCALE`)**: every input
  preserves the object pixel scale of the frozen processed source canvas; any
  additional geometric resize invalidates the unit rather than supporting or
  falsifying a hypothesis. This does not claim raw Common Objects in Context
  pixel scale.

## Question

When a sampled dense full-image rollout misses valid COCO-80 objects, do
structured input-level spatial policies or a reset accepted-row prefix policy
recover those objects more safely than matched-call full-image sampling, and
which unresolved mechanism should be tested next?

The unit measures policy-level effects of masked spatial restriction, native
tiling, cumulative accepted-row prompting, ordinary repeated sampling, and
incomplete annotations. It does not directly assign those effects to a
post-vision decoder mechanism.

## Decision Relevance

This unit decides the next diagnostic seam, not a final architecture.

- If masked spatial restriction is superior on the owning-seed raw contrast and
  safely exceeds the matched-call bagging policy, test the same
  restriction after one fixed full-image visual encoding. Do not yet add slots
  or a ledger.
- If reset safely beats the cumulative accepted-row prefix policy, run a small
  fixed-visual-input prefix panel that separates length, correct content, wrong
  content, and self-generated errors before proposing compact state.
- If only native tiles help, investigate image-token competition, local vision
  encoding, and crop-frame effects before blaming language history.
- If no arm rescues the same objects, return to visual observability and
  object-specific control-state synthesis.

## Claim Boundary

A positive tile result means that the complete native-tile policy is useful. A
positive full-size masked-canvas result means that the declared masked-input
policy is useful without reducing the nominal image-token grid. Neither result
proves that the unmodified full-image representation already recognized the
rescued objects, that competition occurs after the vision tower, or that the
defect lies in the language tower. Both interventions change pixels before
visual encoding.

Matched call count is not matched per-object opportunity. Consequently,
masked-input rescue exceeding bagging is a strong conservative policy result,
but equality or inferiority does not falsify spatial restriction unless the
owning-seed raw contrast and mask-harm guardrail are also interpretable.

Reset versus cumulative accepted-row prompting estimates the total effect of
that prompt-state policy. It does not isolate pure context length, memory
capacity, error propagation, or a missing ledger.

This unit does not test training, learned commit state, same-feature post-vision
gating, slots, a detector proposal head, or an architecture intervention.

## Competing Hypotheses

| Hypothesis identifier | Mechanism | Testable prediction | Falsifier or strongest alternative |
|---|---|---|---|
| `HYPOTHESIS_MASKED_SPATIAL_POLICY` | A full-canvas mask supplies useful spatial restriction under unchanged object scale and nominal full-grid token count. | The owning-seed raw contrast passes its frozen superiority rule, and the final masked policy safely exceeds or usefully complements full-image bagging. | No raw superiority and no safe final-policy advantage after the mask-harm gate passes. A null with poor mask retention remains unresolved. |
| `HYPOTHESIS_POST_VISION_COMPETITION` | Candidate competition after visual encoding suppresses valid continuation paths. | Not directly tested; a positive masked policy raises its priority. | A later same-feature post-vision restriction fails while an input-level mask helps. |
| `HYPOTHESIS_ACCEPTED_ROW_PREFIX_POLICY` | The declared cumulative accepted-row prompt policy harms later rescue. | `MASK_RESET` safely exceeds `MASK_CUMULATIVE` under neutral repetition penalty and a frozen prompt state transition. | The two policies are equivalent inside a predeclared margin, or cumulative safely exceeds reset. |
| `HYPOTHESIS_HISTORY_HORIZON` | Prefix length itself degrades retrieval or continuation. | Not directly tested; a reset advantage raises its priority. | A later content- and correctness-matched prefix panel shows equivalence across length. |
| `HYPOTHESIS_TILE_LOCAL_COMPUTATION` | Tile gains come from fewer image tokens, local frame, or local vision encoding. | `TILE_RESET` exceeds `MASK_RESET`, while `MASK_RESET` is close to `FULL_BAG_K`. | A safe masked-policy gain without a tile increment weakens a tile-only explanation but does not isolate or falsify the residual tile-local family. |
| `HYPOTHESIS_CONTEXT_BOUNDARY` | Boundary cuts or missing context dominate spatial behavior. | Core-plus-halo exceeds core-only near boundaries in both masked-canvas and native-tile families without broad duplicate growth. | Halo has no boundary-local benefit or only increases duplicate union. |
| `HYPOTHESIS_GEOMETRY_ORDER` | Geometry-sorted traversal compatibility contributes to cumulative-policy behavior. | Multiple counterbalanced non-raster orders affect cumulative but not reset controls. | Both reset and cumulative order panels are equivalent inside the predeclared margin. Attribution to training order still requires a random-order-trained checkpoint. |
| `HYPOTHESIS_ANNOTATION_GAPS` | New unmatched dense-scene predictions are real unlabeled COCO-80 objects. | Manual precision and unique recall improve while official precision falls. | Blinded manual audit also labels the new predictions unsupported. |
| `HYPOTHESIS_RESIDUAL_LIMIT` | Persistent misses reflect remaining recognition or object-control limits. | The same manually visible objects remain missed across `FULL_BAG_K`, `TILE_RESET`, and `MASK_RESET`. | Reliable rescue removes the named object from this unit's persistent-miss set; it does not falsify the broader recognition/control family for other objects. |

## Assumptions and Confounds

- COCO-80 is the closed reporting ontology for this unit. A visible object
  outside COCO-80 is an **out-of-scope visible object**: it cannot match a
  reference and counts as a closed-task false prediction if emitted, but it is
  not described as a visual hallucination merely because it is out of scope.
- The primary checkpoint uses geometry-sorted training; results cannot be
  generalized to random-order training until that adapter is available.
- Tiling changes image dimensions and image-token count even without resizing.
- Masked full canvas preserves dimensions and token-grid budget but still
  changes vision-tower activations and can introduce mask boundaries.
- Independent arms remove useful prior-object history as well as harmful
  history. Their gain cannot be called a ledger failure without a later state
  intervention.
- Aggregation can create or destroy apparent rescue. Matched-call full-image
  bagging is mandatory for final-policy utility, while owning-seed raw outcomes
  are mandatory for the one-opportunity object comparison.
- Official COCO false positives are not trusted as hallucinations on dense
  images until manual audit.
- Primary mechanism attribution uses neutral repetition penalty `1.0` so that
  reset and cumulative prompts do not receive different prompt-token penalties.
  The historical `1.10` policy is an explicitly named secondary sensitivity on
  a predeclared subset, not part of the primary contrast.

## Frozen Model and Existing Evidence

Primary checkpoint:

The immutable artifact path below contains legacy compressed tokens. In
particular, `coordexp_infras` is the current native CoordExp training and
inference stack namespace; `prod` is the production-training artifact family;
`infer` is the inference-artifact family; `qwen3_vl_2b` means the approximately
two-billion-parameter Qwen3
Vision-Language model; `desc_first` means description-first row serialization;
`geo_sorted` means geometry-sorted object order; `gaussian_rps` is the
historical token for Gaussian Soft-Target Coordinate Cross-Entropy with
Ordered Cumulative-Distribution Penalty; `dora_r16a32`
means Weight-Decomposed Low-Rank Adaptation with rank 16 and Low-Rank
Adaptation alpha hyperparameter 32; `llm_12000` means a large-language-model
global maximum packed sequence length of 12,000 tokens;
`accelerate8` means eight Hugging Face Accelerate distributed training
processes; `ebs24` means an effective batch size of 24; `8epoch` means eight
training epochs; and
`warmup0p1` means a scheduler warmup ratio of 0.1; `step-4887` means the
checkpoint saved at optimization step 4,887; and `checkpoint.json` is the
JavaScript Object Notation checkpoint manifest. Hyphenated tokens in the
inference path have the same meanings. These strings remain only because
renaming an existing artifact path would break provenance.

```text
/data/CoordExp/.worktrees/coordexp-infras/outputs/prod/coordexp_swift/qwen3_vl_2b_desc_first_geo_sorted_gaussian_rps_dora_r16a32_llm_12000_accelerate8_ebs24_8epoch_warmup0p1/checkpoints/step-4887/checkpoint.json
```

Existing val200 rollout used for cohort planning and baseline sanity only:

```text
/data/CoordExp/.worktrees/coordexp-infras/outputs/coordexp_swift/infer/val200/qwen3-vl-2b-desc-first-geo-sorted-gaussian-rps-dora-r16a32-step4887-val200/
```

The new runner must execute a fresh `FULL_SINGLE` under the frozen unit config.
The independent baseline seed is not reused by any `FULL_BAG_K` call. Existing
outputs cannot silently stand in for a paired run and may not rank, select, or
exclude cohort images. They may only size the study and verify that planned
denominators are viable. Selection of this checkpoint is operational, not
evidence about Gaussian Soft-Target Coordinate
Cross-Entropy with Ordered Cumulative-Distribution Penalty versus pure
cross-entropy. The pure cross-entropy checkpoint with token-type gating at step
4,887 is reserved for a later replication, not mixed into the first
discriminator.

## Cohort Construction

The [readiness amendment](readiness-amendment.md) freezes two nested execution
scopes. Every primary arm executes on all 200 ordered `val200` images. The
scientific headline uses **Dense-Union-51 — Annotation-Derived Dense Union of
51 Images**, the exhaustive subset selected by the frozen annotation-only
Boolean rule. Dense-Union-51 is not sampled and has no selection seed.

Before any experimental-arm output is inspected, two independent image-only
COCO-80 review passes and one adjudication pass must materialize and seal the
audit-augmented reference ledger for all 51 images. Every image remains in the
ledger; review can mark object ambiguity but cannot remove an image. Official
individual annotations, official crowd-ignore regions, reviewer additions, and
adjudicated states remain distinguishable. Any post-output pooled-candidate
adjudication is separately named and cannot rewrite the sealed headline
denominator.

The exact source digests, 200-image order, Dense-Union-51 selection rule and
identifiers, calibration cohort, review protocol, ledger schema, and seal rules
are frozen in the amendment. The unit remains `planned` until the required
manifests and sealed ledger are materialized after separate authorization.

## Spatial Protocol

The only first-run grid is four-by-four (`4 x 4`). A two-by-two (`2 x 2`) or
eight-by-eight (`8 x 8`) grid requires a separate pre-output budget amendment
under the eligibility rules in the readiness amendment. Neither is an
authorized secondary panel and neither may replace the primary result.

- Never resize a tile or source canvas. Processor and runtime receipts must show
  no additional geometric resize and unchanged source-canvas object pixel
  dimensions.
- Partition the frozen source canvas in merged-token coordinates using the
  32-pixel merged-token quantum. Every primary source dimension is divisible by
  32; padding is neither required nor authorized. Any contrary runtime shape
  stops the run.
- Each tile has a disjoint half-open core and a clipped halo. Primary halo width
  is `25%` of the core width/height, rounded outward to the executed
  patch/merged-token grid with a minimum one merged-token cell where the image
  boundary permits it.
- A prediction is owned only by the tile whose core contains its global bbox
  center. Halo content supplies context but never ownership.
- Masked-canvas arms keep the original full canvas dimensions. Compute the
  per-image channel mean over every source-canvas `RGB` `uint8` pixel using
  `float32`, round to nearest with ties to even, clamp to `[0, 255]`, and cast
  back to `uint8`. Pixels outside the current core-plus-halo use that frozen
  value. No outline, alpha cue, padding, or resize is allowed.
- Materialize Visible merged-token support count for every spatial input. The
  readiness amendment freezes its eight-by-eight minimum and the action when a
  cell falls below it; cells are never silently dropped after scoring.
- Aggregation maps all predictions to global coordinates, applies the same
  COCO-80 normalization and frozen merge policy, and exposes pre-merge as well
  as post-merge rows.

The mask boundary is part of the declared input-level intervention, not a
post-hoc mechanics failure. Interpretation is governed by the predeclared
mask-harm retention floor. No alternative fill or padding policy may be selected
after model scores are inspected.

## Arm Matrix and Execution Tiers

Let `K` equal the number of spatial cells for the selected grid. Use an
identical K-seed vector and per-call decode cap wherever an arm has `K` calls.

The primary metric-bearing panel has five arms. Secondary and exploratory arms
cannot delay or replace the primary panel and run only on their frozen audited
scope after the primary mechanics gate passes.

| Arm | Input | History | Calls | Tier and role |
|---|---|---|---:|---|
| `FULL_SINGLE` | Full image | Fresh base prompt | 1 | Primary sampled single-rollout reference using the independent baseline seed and owning the rescue denominator. |
| `FULL_BAG_K` | Full image | Fresh base prompt per call | K | Primary matched-call final-policy sampling control. |
| `TILE_RESET` | Native-scale core-plus-halo tiles | Reset per tile | K | Primary no-resize tile baseline. |
| `MASK_RESET` | Full-size masked core-plus-halo canvas | Reset per mask | K | Primary masked-input spatial-policy arm with full-grid token count. |
| `MASK_CUMULATIVE` | Full-size masked core-plus-halo canvas | Cumulative accepted-row prefix policy | K | Primary total accepted-row prefix-policy arm. |
| `MASK_RESET_CORE_ONLY` | Full-size masked core only | Reset per mask | K | Secondary masked boundary/context ablation. |
| `TILE_RESET_CORE_ONLY` | Native-scale core tiles | Reset per tile | K | Secondary native-tile boundary/context ablation. |
| `MASK_CUMULATIVE_ORDER_PANEL` | Same masked inputs under multiple counterbalanced orders | Cumulative accepted-row prefix policy | K per order | Secondary order-sensitivity panel. |
| `MASK_RESET_ORDER_PANEL` | Same masked inputs and orders | Reset per mask | K per order | Secondary order-mechanics negative control. |
| `TILE_CUMULATIVE_EXPLORATORY` | Native-scale core-plus-halo tiles | Cumulative accepted-row prefix policy | K | Exploratory only after a validated tile/global coordinate contract. |

`FULL_BAG_K`, `MASK_RESET`, `MASK_CUMULATIVE`, `MASK_RESET_CORE_ONLY`,
`MASK_CUMULATIVE_ORDER_PANEL`, and `MASK_RESET_ORDER_PANEL` must have the same
full-canvas image-token count per invocation. Native-scale tile arms have fewer
image tokens and must be reported separately rather than described as
equal-token comparisons. Same nominal token count does not mean the encoded
features or total computation are equal.

For a four-by-four grid, the five-arm primary panel requires `1 + 4K = 65`
model calls per image. The readiness amendment freezes the all-200 primary,
calibration, and conditional second-root replication budgets. Every secondary
panel is excluded and requires a new pre-output amendment before it may run.

## Frozen Decode and Aggregation Factors

The readiness amendment resolves the desired generation, seed, calibration,
prompt, score, merge, matching, and budget semantics. The current inference
backend cannot yet execute those sampled per-request semantics, so this section
is a scientific contract rather than a claim about current runtime capability.
Before metric-bearing execution, materialized receipts must resolve and hash:

- authored and resolved inference config;
- prompt/template and tokenizer identity;
- sampling-enabled flag, positive sampling temperature,
  nucleus-probability cutoff, independent baseline seed, K-seed-vector
  derivation, neutral repetition
  penalty `1.0`, maximum decode tokens, parser, malformed-row policy, and
  closure policy;
- canonical grid order and every counterbalanced order in
  `MASK_CUMULATIVE_ORDER_PANEL` and `MASK_RESET_ORDER_PANEL`;
- cumulative accepted-row admission stage, canonical row serializer, global
  coordinate frame, message role, row order, invalid/duplicate/closure policy,
  exact prompt bytes and token identifiers, cache/reset semantics, and
  fail-fast context-window behavior;
- COCO-80 aliases and class normalization;
- bounding-box ownership, coordinate mapping, object merge, object-score source,
  and duplicate policy;
- per-arm invocation and token budgets.

The primary policy uses `do_sample=true` and a one-time, held-out,
annotation-informed decode calibration that selects a positive temperature from
the frozen `0.2`, `0.4`, and `0.6` candidates, with nucleus cutoff `0.95` and
neutral repetition penalty `1.0`. Calibration selects the lowest candidate
passing batch-size-four replay and scheduling-order invariance, parsing,
closure, seed-response, and Prediction-Set Diversity gates; it does not optimize
Average Precision, Local Rescue Rate, or any primary estimand. The baseline seed
is derived from the
frozen root seed, immutable image identifier, and a reserved baseline label.
Each K-seed-vector entry is derived from the same root and image identifier plus
its canonical cell index; arm identity and traversal position do not change the
seed for a matched full-image/spatial call. The reserved baseline label must be
disjoint from every canonical cell index. Each spatial call uses the seed bound
to its canonical cell, including counterbalanced-order arms.

Exactly one Compute Unified Device Architecture (`CUDA`) sampled-runtime
attestation invocation loads the frozen runtime once and covers all three exact
temperature policies (`0.2`, `0.4`, and `0.6`). For every policy it persists
batch-size-four, batch-size-three, request-order, same-seed replay, and
admitted-production-path evidence in one aggregate artifact. Once the
predeclared calibration gates select the first
passing policy, that aggregate evidence authorizes an exact-policy process-local
capability rebind in each of the eight production workers. The rebind must match
the live frozen runtime and selected policy exactly; no unattested temperature
may execute. This mechanics arrangement removes attestation/admission
circularity without changing candidate order, calibration estimands, or gates.

The implementation must provide a per-request pseudo-random generator or run
requests serially so that batching, worker count, and traversal order do not
silently reassign random streams. A mechanics receipt must demonstrate stable
same-request replay, baseline/K-seed disjointness, and Prediction-Set Diversity
across bagging calls. The readiness amendment must predeclare its empty-union
rule and diversity floor. Below that floor, `FULL_BAG_K` is labeled a
weak-diversity repeated-sampling control and cannot falsify a spatial policy. If
the policy remains deterministic, it is
only a repeated-invocation determinism control and cannot be called independent
bagging.

Physical batches are sealed inside dependency-wave partitions rather than over
the flattened request stream. One independent partition contains every
non-cumulative arm, and each of the 16 canonical `MASK_CUMULATIVE` cells owns a
separate cumulative-cell partition. A batch cannot cross partitions. Batch size
four (`B4`) is canonical; batch size three (`B3`) is allowed only as one natural
final tail per partition and is covered by the same request-scoped runtime
attestation. The 200-image primary schedule is exactly 3,250 B4 batches. The
optional Dense-Union-51 replication is exactly 816 B4 batches plus 17 B3 tails,
for 833 physical batches and the unchanged 3,315 calls. The 17 B3 tails consist
of one independent-partition tail and one tail for each cumulative-cell
partition. Resume consumes only dependency-safe whole batches from the sealed
plan; it never dynamically repacks pending requests.

The sampled-runtime admission gate requires exact forward/reverse request replay
within B4 and independently within B3. Cross-cardinality B4-versus-B3 replay is
recorded as a diagnostic rather than required to be identical, because the
executed CUDA probe demonstrated that long sampled trajectories can diverge
when only batch cardinality changes even though request-order replay remains
exact. Every scheduled call therefore retains its sealed physical cardinality
as execution evidence.

The process barrier means that every request in a physical batch has exactly one
legal terminal attempt, not that every attempt is `completed`. The legal statuses
are `completed`, `failed`, `skipped`, `capped`, and `invalid`. Only infrastructure
or artifact-protocol failure aborts the execution wave; dependency-aware resume
handles legal non-completed scientific outcomes and remains responsible for any
explicit continuation-plan requirement.

The historical repetition-penalty value `1.10` may be evaluated only as a
separately named sensitivity for `FULL_BAG_K`, `MASK_RESET`, and
`MASK_CUMULATIVE` on a frozen subset. The receipt must state whether prompt
token identifiers enter the repetition-penalty seen-token set. It must not
describe the same token as receiving a multiplicatively compounded penalty for
every prior occurrence unless the executed processor actually has that
semantics.

No arm may receive a larger per-call decode cap because its outputs are shorter.
Report configured and realized prompt, decode, image-token, and call budgets.

## Cumulative Accepted-Row State Transition

The readiness amendment freezes this transition. Every call is reconstructed
from scratch with exactly one current image and the unchanged system and user
instructions:

1. parse a call's raw output without repairing unsupported content;
2. normalize only through the frozen COCO-80 category and coordinate policy;
3. reject malformed, out-of-ontology, non-finite, or non-owning rows from the
   cumulative state while retaining them in raw failure artifacts;
4. preserve every accepted owning row in deterministic emission order; do not
   apply cross-call object merge before prompt admission;
5. serialize accepted rows in global full-canvas coordinates and concatenate
   them in call order and generated order inside the still-open assistant turn.
   Preserve row-level delimiters, but exclude the terminal no-more-objects
   decision, end-of-sequence tokens, end-of-message tokens, conversation
   terminators, and prior call-level closure markers;
6. continue immediately after the final accepted `<|box_end|>` token. Add no
   separator, continuation instruction, prior image, prior user turn, or prior
   visual token. Record exact prompt bytes, token identifiers, admitted rows,
   exclusions, and the single current-image placeholder;
7. fail before generation if the full prompt would exceed the frozen context
   budget; silent left/right truncation is forbidden.

A two-cell synthetic receipt must show the exact second-call prompt under one
valid row, one invalid row, one halo-only non-owning row, one duplicate-like row,
an empty first call, and a first call that naturally emits `STOP`. The final
case must prove that terminal tokens are recorded in the raw artifact but do not
terminate or poison the next region's active prefix.
`TILE_CUMULATIVE_EXPLORATORY` remains disabled because
tile-local normalized coordinates cannot be made globally coherent merely by
string remapping; a separate accepted coordinate semantics is required.

## Object Matching Contract

Post-merge predictions own final policy utility. Raw owning-call predictions
own the one-opportunity spatial diagnostic. Raw any-call union and the complete
pre-to-post merge transition are mandatory mechanism accounting rather than
optional parser diagnostics.

The frozen object-level merger is class-wise greedy Non-Maximum Suppression at
bounding-box Intersection over Union `0.70`, without score threshold, fusion,
or transitive suppression. It ranks by the frozen Compact Object Selected-Token
Score, version 1, followed by immutable provenance tie-breaks; ownership and
global coordinate mapping occur before merge. Every suppression edge preserves
parent and suppressor identifiers. A distributed artifact-shard merger is not
an object merger. The exact ordering and lineage contract is in the readiness
amendment.

For either named reference ledger:

1. normalize each valid prediction and reference object through the frozen
   Common Objects in Context 80-category policy; official metrics use official
   category identity, while any research alias table is versioned and produces
   separately named metrics;
2. create a candidate match only when normalized classes agree and
   bounding-box Intersection over Union is at least `0.50`;
3. choose a one-to-one bipartite matching that first maximizes match count and
   then maximizes total Intersection over Union; break an exact remaining tie
   lexicographically by immutable prediction and reference identifiers;
4. do not apply an extra post hoc confidence filter beyond the frozen parser
   and merge contract;
5. never match malformed, invalid, or out-of-ontology rows; extra same-ontology
   duplicate rows cannot increase recall and remain false predictions for
   valid-row precision, while malformed/invalid rows enter a separate mandatory
   safety denominator;
6. return `not applicable` rather than zero when a conditional denominator is
   empty, and always report the numerator and denominator;
7. report the primary `0.50` threshold and a secondary `0.75` threshold
   sensitivity view without selecting between them after seeing results.

Only `accepted` individual objects enter audit-augmented rescue and retention
denominators. `ambiguous` and `partial` objects are ignored in the primary view
and retained for named sensitivity summaries. Source crowd regions preserve
official Common Objects in Context ignore semantics; they do not enter
individual-object rescue denominators unless the sealed audit ledger contains
an independently individuated accepted object. A prediction of an out-of-scope
visible object cannot match and is a false prediction under the closed COCO-80
task, but is not labeled a visual hallucination solely for being out of scope.

The official-annotation, audit-augmented, and optional post-output adjudication
ledgers produce separately named metrics. They must never be pooled into one
unlabeled Local Rescue Rate. Audit-Augmented Local Rescue Rate is primary on the
sealed audited subset; Official-Annotation Local Rescue Rate is secondary.
Uncertainty uses 10,000 image-clustered bootstrap replicates and a 95-percent
Image-Clustered Bootstrap Confidence Interval; its pseudo-random seed is frozen
in the readiness amendment. Whole images, all their objects, every paired arm,
and every call travel together in a bootstrap replicate. The amendment freezes
the percentile interval algorithm, empty-denominator behavior, unweighted
headline, minimum missed-object denominator, effect and equivalence thresholds,
safety guardrails, and conditional second-root replication trigger. The
interval is conditional on the frozen seed vector.

## Primary Estimands

For any experimental arm `A`, relative to objects in the paired
`FULL_SINGLE` arm:

```text
Local_Rescue_Rate_A = P(A detects object o | FULL_SINGLE misses object o)
Retention_A         = P(A detects object o | FULL_SINGLE detects object o)
```

Primary behavior-level contrasts:

```text
Masked_vs_Full_Owning_Seed_Raw_Rescue_Difference
  = P(MASK_RESET owning call detects o | FULL_SINGLE misses o)
  - P(seed-matched FULL_BAG_K raw call detects o | FULL_SINGLE misses o)

Masked_Canvas_vs_Full_Image_Bagging_Policy_Utility_Rescue_Difference
  = Local_Rescue_Rate_MASK_RESET
  - Local_Rescue_Rate_FULL_BAG_K

Masked_Reset_vs_Cumulative_Accepted_Row_Prefix_Rescue_Difference
  = Local_Rescue_Rate_MASK_RESET
  - Local_Rescue_Rate_MASK_CUMULATIVE

Native_Tile_vs_Masked_Canvas_Rescue_Difference
  = Local_Rescue_Rate_TILE_RESET
  - Local_Rescue_Rate_MASK_RESET
```

The serialized names above mean, respectively: **Masked versus Full Owning-Seed
Raw Rescue Difference**, **Masked-Canvas versus Full-Image Bagging Policy-Utility
Rescue Difference**, **Masked Reset versus Cumulative Accepted-Row Prefix Rescue
Difference**, and **Native-Tile versus Masked-Canvas Rescue Difference**.
Positive values favor the first named condition. These are behavior-level
contrasts; none is named as an internal mechanism.

Each estimand is emitted once for the official-annotation ledger and once for
the audit-augmented ledger, using the corresponding full metric name from the
registry. The scientific headline on the sealed audited subset contains two
distinct views:

1. the owning-seed raw contrast, which matches one sampled opportunity per
   object; and
2. the post-merge `MASK_RESET` versus `FULL_BAG_K` versus `MASK_CUMULATIVE`
   policy panel, which compares final utility at matched call count.

Sign agreement is necessary but not sufficient. A masked spatial interpretation
is strengthened only when the owning-seed raw contrast passes its own frozen
superiority rule and the post-merge policy contrast safely exceeds its
comparator. If either view fails its minimum meaningful effect or uncertainty
rule, the joint mechanism reading is `unresolved`. Disagreement is a result
about opportunity allocation or merge behavior, not a reason to select the
more favorable view. Broad-cohort and
official COCO metrics are secondary generalization/sanity views. All other
strata, thresholds, and secondary arms are descriptive unless the readiness
amendment explicitly promotes one.

Retention is an arm-specific safety measure, not a comparative mechanism
signal. The independent baseline seed is deliberately absent from every
multi-call arm.

Also report:

- manual unique recall and manual precision;
- official COCO precision/recall/AP only after raw crowd provenance and finite
  object scores are restored, with an explicit label-hole caveat;
- raw owning-call detection, raw any-call union, post-merge detection, and
  merge-created/merge-destroyed reference matches;
- duplicate components before and after merge, including components whose
  merge would collapse predictions matched to different reference objects;
- invalid/malformed rate per attempted call and attempted object span, natural
  closure versus token/controller cap, and valid prediction count;
- object rescue by original pixel area, class stratum, overlap, and distance to
  tile boundary;
- row/tile-index slopes versus realized prompt length;
- incremental unique recall per invocation, image token, and decoded token;
- persistent-miss identities shared by all arms.

Before any metric-bearing execution, the readiness amendment's frozen values
must be materialized unchanged in the resolved run contract:

- a minimum meaningful effect and a paired superiority rule;
- an equivalence margin whose full uncertainty interval must fit inside the
  margin before declaring equivalence;
- noninferiority or bounded-inflation gates for retention, audit manual
  precision, duplicates, invalid rows, natural closure, and prediction count;
- a mask-harm retention floor on `FULL_SINGLE`-detected, core-interior objects;
- the minimum paired missed-object count and target interval width.

An arm that raises rescue while failing any required safety gate is an unsafe
output-expansion result, not improved enumeration.

## Verification and Falsification Gates

Mechanics must pass before model results are interpreted:

1. source image, checkpoint, processor, tokenizer, prompt, and ontology hashes
   match the frozen plan;
2. every arm preserves the frozen processed source-canvas object pixel scale;
3. every full-canvas arm in the matrix has the declared identical image-token
   count per invocation;
4. tile and masked-canvas coordinate round trips are exact within the declared
   quantization tolerance;
5. core ownership is unique, including boundary centers;
6. `K`, request-seed derivation, decode cap, parser, and merge policy match the
   frozen contract, and owning spatial calls pair with the correct full-image
   seed;
7. all attempted images and failures remain in denominators;
8. the primary image-only audit ledger was sealed before outputs, and any
   post-output pooled-candidate adjudication is source-blind and separately
   named;
9. budgets record realized forward calls and token counts, not only config
   maxima;
10. sampled generation passes same-request replay, seed-change, batching/order
    invariance, and the predeclared prediction-set diversity floor;
11. cumulative prompting passes exact prompt-byte/token receipts and fails fast
    before context overflow;
12. a deterministic fixture passes duplicate-chain separation, two nearby
    same-class instances, a wrong-class overlap, an exact matching tie, an
    audit-added object, a crowd ignore region, an out-of-scope prediction, an
    invalid row, an injected partial failure, and an empty conditional
    denominator;
13. pre-merge, raw union, owning-call, post-merge, and merge-transition metric
    primitives recompute every reported headline;
14. superiority, equivalence, mask-harm, and all safety thresholds are sealed
    before any metric-bearing output is inspected.

Key falsification signatures:

- `MASK_RESET` shows neither an owning-seed raw advantage nor a safe post-merge
  advantage after the mask-harm gate passes: no evidence that this masked-input
  policy adds rescue under the declared scope. Post-vision competition remains
  untested.
- `MASK_RESET` is equivalent to `MASK_CUMULATIVE`: no practically meaningful
  total effect of the declared cumulative accepted-row prefix policy was
  detected. Pure history length remains untested.
- `TILE_RESET` improves but `MASK_RESET` does not: fewer visual tokens or local
  re-encoding remains a stronger explanation than this masked-input policy.
- the mask-harm retention floor fails: a null or negative masked contrast is
  inconclusive about spatial restriction.
- manual precision falls with official precision: new predictions are not
  explained primarily by the sealed annotation additions on this subset.
- all arms miss the same manually visible objects: scope/history are not the
  dominant limitation for that residual set.

## Outcome Map

| Result | Next smallest discriminator |
|---|---|
| Owning-seed raw superiority and safe post-merge masked superiority both pass | Test the same spatial restriction after one fixed full-image visual encoding; do not yet add slots or a ledger. |
| Owning-seed raw masked advantage is positive, but final masked policy is equivalent to bagging | Local restriction helps one sampled opportunity, while K full-image opportunities catch up in final utility; inspect opportunity allocation and merge before choosing a route. |
| Final masked policy exceeds bagging, but owning-seed raw advantage is absent or opposite | Treat aggregation, ownership, repeated opportunities, and merge behavior as unresolved; do not localize a scope mechanism. |
| Masked and bagging policies are equivalent, owning-seed raw difference is equivalent, and mask retention passes | No incremental masked-input benefit was detected under this protocol; return to observability or object-specific control-state synthesis without claiming generic sampling is the mechanism. |
| Mask retention fails | Amend the input intervention; the spatial-scope question remains unresolved. |
| `TILE_RESET` safely exceeds `MASK_RESET` and bagging | Investigate token count, local visual encoding, coordinate frame, padding, and context with an orthogonal visual-side panel. |
| `MASK_RESET` safely exceeds `MASK_CUMULATIVE` under neutral repetition penalty | Run a fixed-visual-input prefix panel separating length, clean content, model errors, shuffled content, and length-matched neutral structure; do not yet build compact state. |
| `MASK_CUMULATIVE` safely exceeds reset | Prior rows provide useful information under the declared policy; next isolate useful coverage information from raw prefix length. |
| Official precision falls while manual precision holds | Build the next evaluation unit around incomplete-label accounting before training against the apparent false positives. |
| Any rescue gain fails a safety gate | Diagnose nonspecific output expansion; do not promote the policy. |

Every outcome leaves `architecture_promotion_status: not_promoted`.

## Research and Implementation Roadmap

The ordered plan is intentionally staged so that no model output can repair an
underdefined contract after the fact.

1. **Readiness amendment**: the amendment now freezes cohort, ledger-generation
   rules, decode policy, seed derivation, cumulative prompt state, spatial
   construction, merger, matcher, effect thresholds, safety gates, and symbolic
   budgets. Its materialized ledgers and runtime receipts remain absent.
2. **Focused fixed-point review**: independently verify that the amendment is
   scientifically discriminative and contains executable generation and seal
   rules for every later materialized value. Passing this review approves only
   the contract; the unit remains `planned` and execution-blocked.
3. **Separate implementation authorization**: completed on 2026-07-13. The
   authorization covers bounded research implementation, synthetic and
   mechanics gates, and—only after those gates pass—the frozen metric-bearing
   execution ceilings.
4. **Synthetic semantics fixtures**: verify masks, tiles, global coordinate
   round trips, half-open ownership, prompt-state transitions, seeded sampling,
   duplicate chains, one-to-one matching, crowd handling, merge accounting,
   failure rows, and empty denominators without a model run.
5. **Tiny mechanics gate**: use a few predeclared images only to attest pixel
   scale, image-token count, seed replay and diversity, prompt integrity,
   context capacity, raw/merged artifact completeness, and budget accounting.
   It is not metric-bearing evidence.
6. **Five-arm primary metric panel**: execute `FULL_SINGLE`, `FULL_BAG_K`,
   `TILE_RESET`, `MASK_RESET`, and `MASK_CUMULATIVE` on all 200 frozen images
   under the four-by-four grid. The sealed Dense-Union-51 ledger owns the
   scientific headline; all-200 official metrics are secondary views from the
   same calls.
7. **Separately amended secondary panels**: core-only boundary,
   counterbalanced order, historical repetition-penalty, two-by-two, and
   eight-by-eight panels are excluded from this run. A triggered panel still
   requires a new pre-output scope and budget amendment; it cannot replace the
   primary result.
8. **Evidence closure**: recompute raw and post-merge estimands from stored
   metric primitives, apply the frozen safety and uncertainty rules, write
   `results.md`, update beliefs in the investigation overview, and leave every
   architecture status unpromoted.

The active narrow OpenSpec change
`add-request-scoped-inference-controls` governs the authorized stable sampled-
request and open-assistant-continuation seams. Scientific hypotheses, cohorts,
spatial policies, thresholds, and research artifacts remain in this unit.

## Stop Rules

- Stop immediately on resize, coordinate, ownership, token-budget, or seed
  mismatch.
- Keep `execution_readiness: blocked` until the authorized implementation,
  readiness artifacts, synthetic fixtures, and mechanics gates all pass. User
  authorization permits work to begin but does not make the unit
  execution-ready.
- After separate implementation authorization, run deterministic synthetic
  fixtures before any model call and a tiny non-metric-bearing mechanics gate
  before any cohort.
- Run the primary four-by-four `FULL_SINGLE`, `FULL_BAG_K`, `TILE_RESET`,
  `MASK_RESET`, and `MASK_CUMULATIVE` panel before secondary arms.
- Do not add grids, mask styles, checkpoints, or decode policies after seeing
  model scores. Amend the unit and start a new run identifier instead.
- A result is `complete` only after evidence handles are fixed, the manual audit
  scope is declared, failures are counted, and the bounded verdict is written.

## Artifact and Implementation Handles

Logical run root:

```text
outputs/research/qwen3-vl-dense-enumeration/2026-07-13-spatial-scope-history-disentanglement/<run-id>/
```

Here `<run-id>` means one immutable execution identifier.

Resolved durable root:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-13-spatial-scope-history-disentanglement/<run-id>/
```

Each run must retain authored/resolved config, exact command, Git/worktree and
dirty-state identity, dependency/runtime/device identities, cohort and ledger
manifests, variant plan, per-attempt records, invariant receipt,
execution/budget receipt, exact cumulative prompts, raw per-arm rows,
owning-call/raw-union/pre/post-merge predictions, failures, metric primitives,
analyzer version, and terminal status.
The later `results.md` records exact handles, hashes, evidence scope, observed
facts, and the bounded verdict; it does not copy bulk artifacts into Git.

The authorized reusable implementation uses thin command-line interfaces under
`scripts/research/` and tested semantic owners under `src/analysis/`. This unit
authorizes only the bounded implementation required by the frozen protocol; it
does not define or promote a new stable production artifact schema.

## Result

Execution and bounded interpretation are complete. Across two independently
derived seed roots, full-canvas masking reproducibly improved the one-call
Owning-Seed Raw Rescue Difference but did not improve post-merge policy utility
over equal-call full-image bagging. The cumulative accepted-row prefix policy
was reproducibly worse than reset as a total policy effect, native-scale tiling
was worse than full-canvas masking, and required retention, mask-harm, manual-
precision, and prediction-count safety gates failed. See
[Masked Spatial Policy and Accepted-Row Prefix Policy Results](results.md) for
the exact metrics, receipts, hashes, limitations, and next discriminator.

No mechanism or architecture is promoted.
