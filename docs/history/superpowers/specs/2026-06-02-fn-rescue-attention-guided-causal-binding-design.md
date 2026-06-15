# FN-Rescue Attention-Guided Causal Binding Design

Status: draft superpowers design for the next independent research exploration.

Date: 2026-06-02

Owner: CoordExp mechanism research

## Goal

Design the next independent FN-rescue mechanism study for checkpoint-3664:
use attention mining to choose target, same-description competitor,
wrong-control source, and sink/background intervention candidates, then use
paired no-op and masked rollout continuations to determine which
attention-selected image regions are causally relevant for
`desc -> x1/y1/x2/y2` instance binding.

The goal is not to train a new model yet.  The goal is to convert attention
evidence from "where the model may be reading" into case-linked causal evidence
about which visual regions change the autoregressive continuation.  Image-region
occlusion can validate attention-guided region dependence; it cannot by itself
prove that a specific attention head, attention score, or value stream is
causal.

## Worktree Scope

All code, config, docs, tests, and launch commands for this exploration must be
scoped to this checkout:

```text
/data/CoordExp/.worktrees/fn-rescue-attention-probes
```

Do not edit the parent checkout at `/data/CoordExp` for this exploration unless
the user explicitly asks to sync or transplant changes.  Artifact roots may
remain under the shared output tree:

```text
/data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_val200
```

but repo files must be read and written from the worktree above.

## Starting Evidence

Primary checkpoint:

```text
/data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/compact_full_et_rmp_ce_support2_bsz16_4epoch_tokenrows_v2/compact-full-et-rmp-ce-support2-bsz16-4epoch-tokenrows-v2/v0-20260504-071356/checkpoint-3664
```

Primary FN-rescue artifact root:

```text
/data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_val200/fn_rescue_continuation
```

Primary Phase-2 artifact root:

```text
/data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_val200/fn_rescue_desc_x1_phase2
```

Durable findings note:

```text
progress/diagnostics/2026-06-02_fn_rescue_attention_binding_findings.md
```

Current evidence summary:

- `desc_only` can recover many rollout false negatives, so at least part of the
  recall failure is not pure visual invisibility.
- Correct `desc_x1` improves primary rescue success from about `0.495` to
  `0.807`; wrong-control `x1` drops to about `0.257`.
- Same-description competitor cases expose the clearest binding weakness:
  `desc_only` success is low, correct `x1` helps, wrong `x1` strongly
  misdirects.
- Full attention mining streamed `4,106,816` rows and joined `1,268,288`
  attention rows to generation outcomes.
- Strong target-vs-competitor margins concentrate in middle-layer
  `desc_x1/pre_y1` heads, especially layers 13, 16, and 17.
- Wrong-control rows reverse target-versus-competitor attention in several of
  those heads.
- Far-background attention mass is high in the same heads, but it is not yet
  known whether this is harmful sink behavior, layout bookkeeping, or benign
  context.
- The Lane-D x1 logit-lens smoke does not show exact target `x1` as cleanly
  readable at `desc_end`; `box_start/pre_x1` improves the target rank but top-1
  remains rare.
- The first GPU causal smoke had 4/4 no-op exact parity.  Target-GT masking
  changed three of four generated tails, flipped two successful continuations
  to failure, and had mean IoU delta about `-0.3163`.

## Central Research Question

When the compact-full V-LLM is forced into a desc-first continuation for a
missed object, does the model bind the next instance by routing evidence through
target regions, same-description competitors, wrong-control source regions, or
background/context sink regions?

This question is narrower than "where is the attention."  Attention rows define
candidate mechanisms.  Paired rollout interventions decide whether those
attention-selected image regions actually change the generated continuation.
Claims about attention-head or attention-pattern causality require a separate
activation, value-patching, or attention-intervention lane and are out of scope
for the first Phase-3 implementation.

## Hypotheses

### H1: Target-Dependent Binding

For successful `desc_x1` continuations, target-region visual evidence is
causally required.  No-op replay should reproduce the baseline, while
`target_gt_mask` should reduce target IoU, change coordinate tails, or flip
primary success for a meaningful subset of cases.

Expected evidence:

- no-op exact-tail parity remains high;
- target mask produces negative IoU deltas or success flips;
- attention heads with high target mass predict stronger target-mask damage.

### H2: Same-Description Competitor Steals Binding

Some failed or wrong-control continuations are not visually blind; they bind to
a same-description competitor.  In those cases, masking the competitor or
wrong-control source should reduce competitor binding, improve target binding,
or change the coordinate basin.

Expected evidence:

- wrong-control source masks reduce wrong-control failure strength;
- competitor masks improve `target_iou` or target-vs-competitor coordinate
  attribution in same-desc cases;
- attention competitor mass predicts where competitor masks matter.

### H3: Background Sink Is Mixed, Not Globally Harmful

High far-background attention is not automatically bad.  It may provide layout,
image-boundary, or coordinate-grid information.  Broad background masking should
not be promoted as a training signal unless a finer sink subset is shown to
hurt target binding.

Expected evidence:

- broad far-background mask may be noisy, destructive, or inconclusive;
- fine-grained high-attention non-object patch masks may separate harmful sinks
  from benign context;
- target/competitor interventions should be interpreted before sink
  interventions.

### H4: X1 Is a Binding Seed, Not Just a Coordinate Token

Correct `x1` changes the target-versus-competitor routing; wrong `x1` can route
the model toward a different same-description instance.  The causal test should
therefore analyze `desc_x1`, `desc_x1_wrong_control`, and `desc_only` separately
instead of collapsing them.

Expected evidence:

- correct `x1` rows show target-favoring intervention effects;
- wrong-control rows show competitor/source-favoring intervention effects;
- role-specific attention and hidden-state evidence shifts around `pre_y1` and
  coordinate boundary states.

## Non-Goals

- No production training.
- No broad model surgery or upstream HF model edits.
- No claim that attention mass alone proves causality.
- No claim that image-region masking proves attention-head causality.
- No global foreground/background classifier copied from vision-token pruning
  papers.
- No global COCO background penalty; unlabeled objects and partial annotation
  remain a core caveat.
- No full-validation metric claim from `val200` mechanism artifacts.
- No background-suppression objective until target, competitor, and source
  interventions have produced stable paired evidence.

## Guardrails

- Every intervention run must include paired no-op rows for the same prompt,
  image, tier, and decode settings.
- No-op exact-tail parity is the first gate.  If no-op parity is below `0.95`
  over interpretable paired rows, do not interpret masked rows beyond a
  blocker report.
- Masked rows are interpretable only when the paired no-op row exactly replays
  the baseline, parses successfully, and has zero IoU delta.
- A lane-level headline requires at least `20` interpretable masked rows unless
  it is explicitly labeled `tiny` or `smoke`.
- Invalid parse rate above `0.05` in a masked lane blocks mechanism headlines
  for that lane.
- Effect summaries must report denominators before effects:
  `selected_cases -> attempted_generation_rows -> reconstructable_rows ->
  selected_generation_rows -> phase3_cases -> valid_paired_rows`.
- Report scope labels on every output: checkpoint, artifact root, data split,
  rescue tier, prefix quality, binding bucket, intervention kind, and decode
  settings.
- Treat `same_desc_iou > 0.95` as duplication only; other extra predictions
  are not central negatives in this partial-label analysis.
- Keep background interventions behind target/competitor sanity gates.
- Separate image-region occlusion from future vision-token suppression.  Do not
  mix their effects in one metric table.
- Use all available GPUs as analysis parallelism when helpful, but do not
  interpret "8 cards" as production training.

## Experimental Design

### Lane A: Target-Mask Scale-Up

Purpose:

Scale the current 4-case target-mask smoke to the planned `target_gt_mask`
surface.

Inputs:

```text
fn_rescue_desc_x1_phase2/intervention_plan/selected_interventions.jsonl
fn_rescue_continuation/rescue_generation_rows.jsonl
```

Interventions:

```text
no_op_control
target_gt_mask
```

Required outputs:

```text
fn_rescue_desc_x1_phase3_causal_binding/target_mask/intervention_rows.jsonl
fn_rescue_desc_x1_phase3_causal_binding/target_mask/summary.json
fn_rescue_desc_x1_phase3_causal_binding/target_mask/report.md
```

Primary metrics:

- no-op exact-tail parity;
- no-op IoU delta;
- target-mask mean and median IoU delta;
- target-mask primary-success flip count;
- target-mask exact-tail match count;
- invalid-parse count;
- breakdown by rescue tier, prefix quality, depth bucket, object count bucket,
  and binding bucket.

Decision gate:

Proceed to competitor/source interventions only if no-op exact-tail parity is
at least `0.95`, invalid-parse rate is at most `0.05`, and target-mask rows
show either mean target-IoU delta below `-0.10` or at least one primary-success
flip in an interpretable stratum.  Otherwise record a blocker and do not promote
Lane A beyond smoke evidence.

### Lane B: Same-Description Competitor and Wrong-Source Intervention

Purpose:

Test whether same-description competitors and wrong-control source regions
causally steal instance binding.

Interventions:

```text
no_op_control
same_desc_competitor_mask
same_desc_rollout_prediction_mask
wrong_control_source_region_mask
```

Required outputs:

```text
fn_rescue_desc_x1_phase3_causal_binding/competitor_source/intervention_rows.jsonl
fn_rescue_desc_x1_phase3_causal_binding/competitor_source/summary.json
fn_rescue_desc_x1_phase3_causal_binding/competitor_source/report.md
```

Primary metrics:

- target IoU delta;
- success flip rate;
- competitor-to-target binding flip rate when competitor/source boxes are
  available;
- best same-description competitor/source IoU50 and IoU75 after intervention;
- target-vs-competitor/source IoU margin;
- mask area and target-overlap area for each region intervention;
- coordinate tail edit distance versus baseline;
- `x1/y1/x2/y2` slot-wise delta;
- attention-predicted competitor mass versus intervention effect.

Decision gate:

If competitor/source masks improve target binding in a subset of same-desc or
wrong-control cases, promote these cases to a case-linked causal atlas.  If
they mostly hurt or do nothing, treat same-desc attention as diagnostic but not
necessarily harmful.

Lane B claims require controls in the row schema even if control execution is
deferred: mask area, target-overlap area, source/competitor IoU before and
after, and low-attention/background or unrelated-object controls when available.
Do not use competitor/source masking alone as proof that the attention pattern
itself stole binding.

### Lane C: Attention-Linked Case Table

Purpose:

Join intervention outcomes to attention-mining and x1-probe evidence so that
each case can be analyzed as a mechanism record rather than a loose aggregate.

Required output:

```text
fn_rescue_desc_x1_phase3_causal_binding/case_linked/case_mechanism_rows.jsonl
fn_rescue_desc_x1_phase3_causal_binding/case_linked/summary.json
fn_rescue_desc_x1_phase3_causal_binding/case_linked/report.md
```

Each row must include:

```text
case_id
rescue_tier
target_desc
target_gt_idx
prefix_quality
binding_bucket
depth_bucket
object_count_bucket
intervention_kind
baseline_generated_box_xyxy
intervention_generated_box_xyxy
target_iou_delta
primary_success_changed
top_attention_heads_for_case
target_attention_mass
competitor_attention_mass
wrong_source_attention_mass
far_background_attention_mass
target_minus_competitor_attention
x1_logit_lens_rank_when_available
mechanism_bucket
```

Mechanism buckets:

```text
target_dependent
competitor_dependent
wrong_source_dependent
robust_to_region_masks
diffuse_or_context_dependent
invalid_or_uninterpretable
```

Decision gate:

Only mechanism buckets backed by no-op parity and successful parsing can be
used in the report headline.

### Lane D: Fine-Grained Background/Sink Triage

Purpose:

Prepare a background/sink intervention that is not a broad foreground versus
background clone.  This lane should not execute broad background suppression
until the region definition is narrowed.

Candidate subsets:

```text
high_attention_non_object_patch
context_ring_patch
image_boundary_patch
low_attention_background_control
unlabeled_object_like_patch
```

Required first output:

```text
fn_rescue_desc_x1_phase3_causal_binding/sink_triage/sink_candidate_rows.jsonl
fn_rescue_desc_x1_phase3_causal_binding/sink_triage/summary.json
```

Decision gate:

Only run sink masking after the candidate rows distinguish object-like unlabeled
patches from true texture/background or image-boundary patches.

## Artifact Root

Use a new root to avoid mixing Phase-2 completed artifacts with Phase-3
causal-binding evidence:

```text
/data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_val200/fn_rescue_desc_x1_phase3_causal_binding
```

Expected top-level files:

```text
summary.json
report.md
manifest.json
```

Expected subdirectories:

```text
target_mask/
competitor_source/
case_linked/
sink_triage/
logs/
```

## GPU Allocation Policy

The 8 GPUs should be used as sharded analysis workers, not as data-parallel
training.

Recommended launch shape:

- target-mask scale-up: 1 to 4 GPUs depending on case count;
- competitor/source intervention: up to 8 shards, one GPU per shard;
- case-linked table and reports: CPU only;
- sink triage: CPU first, GPU only after candidate subsets are selected.

Every sharded GPU run must materialize shard-local summaries before merge.

## ScriptMaster Orchestration

The first ScriptMaster for this exploration is a worktree-local tmux launcher:

```text
scripts/analysis/launch_autoreg_fn_rescue_attention_guided_causal_binding_tmux.sh
```

Its job is to execute the validated Phase-3 lanes in dependency order:

1. `target_mask` on one assigned GPU.
2. `competitor_source` on a second assigned GPU after the target-mask gate
   passes.
3. `sink_triage`, `case_linked`, and `report` on CPU.

This launcher intentionally does not claim 8-way shard utilization until the
Phase-3 intervention lanes have shard-local outputs and a merge contract.  The
available 8 GPUs are treated as a resource pool; a run may use fewer cards when
the current dependency graph has fewer safe parallel GPU lanes.

Required ScriptMaster guarantees:

- derive `REPO_ROOT` from the script path unless explicitly overridden;
- generate commands that `cd` into
  `/data/CoordExp/.worktrees/fn-rescue-attention-probes`;
- keep all repo-file references worktree-local while shared artifact roots stay
  under `/data/CoordExp/outputs`;
- write a reproducible command file under the artifact `logs/`;
- guard existing Phase-3 outputs unless `ALLOW_OVERWRITE=1`;
- label the command file as analysis orchestration, not production training;
- run CPU-only stages with `CUDA_VISIBLE_DEVICES=`.

Future shard-safe ScriptMaster work must add explicit shard subdirectories,
per-shard summaries, and merge validation before using all 8 cards on one lane.

## Reporting Requirements

The final report must separate:

- attention correlation evidence;
- hidden/logit-lens evidence;
- no-op replay parity evidence;
- target-mask causal evidence;
- competitor/source causal evidence;
- background/sink candidate evidence;
- residual risks and failed gates.

The report headline must not claim that background attention is harmful unless
the sink intervention passes its decision gate.

## Acceptance Criteria

- A worktree-scoped implementation plan exists under `docs/superpowers/plans/`
  before code changes begin.
- Target-mask scale-up produces paired no-op and target-mask rows with a valid
  summary.
- No-op parity is reported before any masked-intervention interpretation.
- Competitor/source interventions either produce a validated summary or a
  documented blocker with exact failing cases.
- Case-linked rows join intervention outcomes with attention evidence by
  stable case identifiers.
- Sink/background work stops at candidate triage unless target/competitor
  evidence justifies deeper intervention.
- All reports label evidence as `tiny`, `smoke`, `val200`, or another exact
  scope.

## Failure Modes To Record

- no-op replay no longer matches baseline;
- image occlusion changes processor geometry unexpectedly;
- target-mask breaks format rather than binding;
- competitor mask removes too much target context;
- broad background masks destroy scene layout rather than isolate sink tokens;
- high attention predicts no intervention effect;
- low attention regions still have strong causal effects;
- partial COCO annotations make a presumed background patch object-like.

## Next Implementation Plan

The next plan should implement the lanes in this order:

1. Target-mask scale-up with no-op parity.
2. Competitor/source intervention.
3. Case-linked attention/intervention table.
4. Sink candidate triage.
5. Optional sink intervention only if the triage gate passes.

The implementation plan must use this worktree as its root:

```text
/data/CoordExp/.worktrees/fn-rescue-attention-probes
```
