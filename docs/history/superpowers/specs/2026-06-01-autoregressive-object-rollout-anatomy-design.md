# Autoregressive Object Rollout Anatomy Design

Status: draft superpowers design; revised with 2026-06-02 attention-routing
addendum before the next implementation plan.

Date: 2026-06-01

Owner: CoordExp mechanism research

## Goal

Design a long-running, artifact-first mechanism study for one fixed CoordExp
Stage-1 compact-full recursive-detection checkpoint:

```text
/data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/compact_full_et_rmp_ce_support2_bsz16_4epoch_tokenrows_v2/compact-full-et-rmp-ce-support2-bsz16-4epoch-tokenrows-v2/v0-20260504-071356/checkpoint-3664
```

The study asks how an autoregressive V-LLM rolls out detection objects one by
one, and why recall remains low despite strong precision. The final output must
distinguish object discovery, coverage, attention allocation, stop/continue
decisions, prefix contamination, coordinate-basin commitment, and
loss-supervision credit assignment.

This design is not a training recipe yet. It is the superpowers-level contract
that constrains how later implementation plans, analysis scripts, reports, and
optional follow-up training proposals must be built and interpreted.

## Scope

Primary checkpoint:

```text
/data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/compact_full_et_rmp_ce_support2_bsz16_4epoch_tokenrows_v2/compact-full-et-rmp-ce-support2-bsz16-4epoch-tokenrows-v2/v0-20260504-071356/checkpoint-3664
```

Primary dataset:

```text
/data/CoordExp/public_data/coco/rescale_32_1024_bbox_max60/val.coord.jsonl
```

Primary existing inference artifact:

```text
/data/CoordExp/outputs/infer/recursive_detection_ce_latest/compact_full_support2_tokenrows_v2_ckpt3664_val200_bsz8_temp0_rep1p10_prompt_offset_fix_8gpu
```

Primary existing files in that artifact root:

```text
gt_vs_pred.jsonl
pred_token_trace.jsonl
pred_confidence.jsonl
gt_vs_pred_scored.jsonl
gt_vs_pred_scored_guarded.jsonl
summary.json
eval/metrics.json
eval/metrics_guarded.json
eval/matches.jsonl
eval/matches_guarded.jsonl
eval/duplicate_guard_report.json
resolved_config.json
```

Required scope labels in every report:

```text
checkpoint-3664
compact_full
coord-token
recursive_detection_ce_latest
val200 or full-val or tiny
teacher_forced or self_prefix or free_decode
greedy or sampled
decode config: temperature, repetition_penalty, max_new_tokens, grammar
dataset slice policy: first_200 or named split
metric family: raw or guarded or debug-f1ish-only
artifact root
```

Do not compare `val200`, `limit=200`, proxy, first-200, and full validation as
if they are the same evidence surface.

## Non-Goals

- No production training change in the first implementation plan.
- No new stable CLI flag unless an implementation plan proves a config-first
  route cannot express the required behavior.
- No OpenSpec change unless later work promotes a behavior, artifact, or metric
  contract into a stable current interface.
- No full validation sweep before `val200` mechanism gates converge.
- No interpretation from attention maps alone.
- No claim that forced continuation solves recall unless the added continuation
  moves probability mass toward correct or acceptable object candidates.
- No parser relaxation to hide malformed outputs.
- No broad refactor of inference, evaluation, or training runtime.
- No claim that a supervision change improves recall unless it is backed by a
  matched baseline artifact with the same scope.

## 2026-06-02 Attention-Routing Addendum

The next mechanism phase is recall-first under partial COCO labels. Its primary
object is **missed-GT evidence routing**: why a labeled, visible, not-yet-emitted
object fails to become the next generated object under autoregressive rollout.

This addendum supersedes any interpretation that treats `free_boundary` versus
`entry_after_separator` logits as a sufficient mechanism explanation. Forced
continuation can remain a diagnostic state constructor, but it is not decisive
unless the continuation state allocates probability mass to the right object
candidate, an acceptable unlabeled-positive candidate, or a clearly recoverable
candidate set. A high stop hazard should instead be decomposed into competing
causes:

- weak or absent visual evidence, especially for small, occluded, or crowded
  objects;
- foreground evidence diverted into context, background, already-emitted
  objects, same-description competitors, or generated text history;
- language, length, schema, or set-size priors that make stopping attractive
  after some number of generated rows;
- coordinate-basin conversion failure where attention reaches the target region
  but `pre_x1` / `x1` binds to another local candidate.

Because COCO has substantial missing annotations, non-duplication false
positives must not be treated as hallucination negatives by default. To keep the
next phase focused on attention mechanisms, generated extras should use a simple
first-pass split:

```text
same_desc_iou_gt_0p95_duplicate
format_or_geometry_invalid_extra
other_extra_prediction
```

The duplication rule for this phase is intentionally narrow: if an extra
prediction has the same description as an already emitted prediction and their
box IoU is greater than `0.95`, classify it as
`same_desc_iou_gt_0p95_duplicate`. Other extras are not the main research target
and should be treated as `other_extra_prediction` unless they are clearly
format/geometry invalid. This weakens FP taxonomy on purpose so the attention
study can focus on missed-object evidence routing.

### Recall-First Precision Guardrail

The default evaluation stance for this attention phase is recall-first:

```text
primary objective: maximize labeled-GT recall and plausible visible-object coverage
hard negatives: same_desc_iou_gt_0p95_duplicate,
                format_or_geometry_invalid_extra
soft/neutral extras: other_extra_prediction
```

This stance does not redefine official COCO metrics. It constrains mechanism
interpretation and follow-up training proposals under partial labels: a method
that increases plausible object coverage may be desirable even if it creates
extra predictions that COCO annotations do not label, while a method that mostly
adds same-description near-identical duplicates or invalid boxes should be
treated as a failure. Reports must therefore separate labeled-GT recall,
same-description IoU>`0.95` duplicate rate, invalid-extra rate, and other-extra
rate before making any precision or recall claim.

The attention phase should therefore be organized around one shared schema:

```text
target_object: missed labeled GT, prioritized in under-generated images
candidate_regions: target GT, same-desc GT, emitted GT,
                   same-desc IoU>0.95 duplicate candidate,
                   context ring, far background
decision_readouts: natural stop hazard, next-desc/category mass,
                  x1 target rank, x1 top attribution, duplicate/local mass
attention_readouts: layer/head/token region mass, entropy,
                   foreground-vs-context ratio, history-vs-vision ratio
causal_readouts: decision changes after foreground/context/background/history
                masking, attenuation, or patching
```

The four attention lanes are:

- stop-hazard cause decomposition, not forced-continue success;
- foreground, context, and background competition;
- candidate competition and object-binding consistency between attention
  regions and coordinate peaks;
- self-prefix degradation, using teacher-prefix states only as paired controls
  rather than as a result to rediscover.

Evidence standards remain causal-first: attention visualizations are supporting
evidence only. Mechanism claims require either predictive value after controlling
for size, crowding, prefix depth, and remaining count, or a region intervention
that changes stop hazard, object-candidate mass, `x1` target rank, or duplicate
basin mass.

## Current Evidence Starting Point

This checkpoint family already has a useful mechanism baseline:

- hard CE and Gaussian SoftCE coordinate-logit locality studies show that
  static coordinate-token row geometry is strong and does not explain the
  dynamic behavior difference;
- `x1` is the fragile coordinate slot under both teacher forcing and
  self-prefixes;
- the informative comparison surface is teacher-forced logits plus
  self-prefix logits plus guarded rollout metrics;
- existing `val200` guarded inference has high precision and low recall:
  `f1ish@0.50_precision_loc_micro` is about `0.80`, while
  `f1ish@0.50_recall_loc_micro` is about `0.48`;
- raw object counts in the main existing `val200` artifact show
  under-generation: `1444` GT objects versus `1141` raw predicted objects
  across `200` images.

The study must begin from this concrete evidence rather than generic VLM
intuition.

## Hypotheses

Each experiment lane must report which hypotheses it supports, weakens, or
leaves inconclusive. A final report may choose a mixed conclusion only after
the decision gates below are satisfied.

### H1: Global Set Then Sequential Translation

The model forms a broad multi-object representation early, then serializes the
object set through language rows. Under this hypothesis, later autoregressive
tokens are mostly a translation bottleneck.

Expected evidence:

- prompt-end or early hidden states predict object count, remaining count, and
  coarse object set better than chance;
- forced prefixes preserve support for remaining objects;
- missed objects often have measurable teacher-forced or probe support before
  they are suppressed by decoding;
- occlusion or patching evidence indicates early support for multiple objects,
  not only the current emitted row; attention may visualize this pattern but is
  not decisive by itself.

Evidence against:

- next-object posterior changes almost entirely with prefix content;
- hidden states cannot recover remaining-object identity until row-local
  `pre_x1` or post-coordinate positions;
- object support appears only after generated context points toward a specific
  local region.

### H2: Sequential Dynamic Discovery

The model discovers objects dynamically as autoregressive rollout proceeds.
Each emitted object changes where the model looks and what object it can find
next.

Expected evidence:

- occlusion or patching sensitivity migrates across visual regions by row
  ordinal, with attention used only as supporting visualization unless paired
  with a causal intervention;
- prompt-end hidden states weakly predict later objects;
- generated prefix quality strongly changes next-object posterior and recall;
- bad or false-positive prefixes redirect discovery toward wrong regions.

Evidence against:

- early hidden states already encode most of the remaining object set;
- forced prefix depth has little effect on remaining-object support;
- attention stays globally distributed while object choices are controlled by
  language-side stop or serialization boundaries.

### H3: Soft Pre-x1 Binding, Hard x1 Commitment

The model has partial object support before the first coordinate token, but the
specific instance is not hard-bound until `pre_x1` / `x1`. This is the current
leading mixed mechanism based on prior locality and instance-binding evidence.

Expected evidence:

- `pre_x1` distributions have target advantage but remain multi-modal;
- `post_x1` or `post_y1` probes are much stronger than `pre_x1` probes;
- wrong `x1` peaks often map to another same-image GT object, previous
  generated object, or prefix-contaminated object;
- hidden-state patching at late schema or immediate `pre_x1` positions moves
  `x1` mass between candidate instances.

Evidence against:

- target identity is already cleanly decodable before `pre_x1`;
- `x1` errors are diffuse and do not correspond to candidate objects;
- coordinate slots fail uniformly rather than showing a row-onset-specific
  weakness.

### H4: Decode Stop and Free-Boundary Bottleneck

The model may show rising stop hazard before all labeled GT objects are emitted.
This is no longer framed as a sufficient forced-continuation bottleneck. The
useful question is why stopping becomes attractive: weak visual evidence,
foreground evidence diversion, language/set-size priors, prefix-state damage, or
coordinate-basin conversion failure.

Expected evidence:

- stop hazard remains high for remaining labeled GT even after controlling for
  object size, crowding, prefix depth, generated row count, and remaining count;
- attention or intervention evidence shows whether the stop state is driven by
  weak target-region evidence, background/context diversion, generated-history
  overwrite, or language/set-size priors;
- EOS hazard rises while `remaining_gt_count > 0`;
- under-generated images show higher stop probability than matched-count
  images at comparable prefix quality;
- forced continuation is useful only if it also recovers correct or acceptable
  next-object candidate mass, not merely because more tokens are produced.

Evidence against:

- stop hazard is explained almost entirely by target object invisibility,
  extreme smallness, occlusion, or crowding;
- region interventions change attention maps but not stop hazard or next-object
  candidate mass;
- low recall persists even when stop pressure is reduced and candidate mass is
  directed toward plausible remaining objects.

### H5: Supervision and Credit-Assignment Mismatch

The training objective may optimize local next-token CE and multi-positive
entry support without assigning enough credit to full object coverage under
self-prefix rollout. Teacher-forced losses can look healthy while rollout
recall stays low because the objective does not punish coverage collapse at
the right prefix states.

Expected evidence:

- teacher-forced coordinate or trie metrics are healthier than self-prefix
  rollout behavior;
- clean prefixes behave well but false-positive or duplicate prefixes cause
  remaining coverage to collapse;
- wrong `x1` peaks are often plausible objects, showing support exists but is
  credited to the wrong continuation;
- loss/target-mix trends do not predict recall, row count, or coverage survival
  without prefix-condition labels;
- recovery states after false-positive, duplicate, or invalid prefixes are
  absent from the supervised target distribution, unweighted, or only weakly
  represented compared with clean teacher-forced states;
- the H5 claim remains necessary after H3-style `x1` attribution and H4-style
  boundary calibration have been measured.

Evidence against:

- low recall is explained almost entirely by visual absence or impossible
  objects;
- teacher-forced and self-prefix behavior fail in the same way;
- coverage improves directly from decode calibration without changing
  supervision assumptions.

Required discriminators:

- H5 versus H3: H3 explains a wrong local coordinate basin at object onset; H5
  requires evidence that the failing prefix state was not adequately supervised
  or credited even when the model has plausible candidate support.
- H5 versus H4: H4 explains a stop/free-boundary failure; H5 requires residual
  coverage or recovery failure after `free_boundary` versus
  `entry_after_separator` is separated.
- H5 versus generic prefix contamination: H5 requires a target/loss accounting
  readout, such as target mix, missing recovery states, per-prefix target
  weights, or matched checkpoint trends.

## Hypothesis Discriminator Matrix

Synthesis reports must include a compact discriminator table. The table can be
generated from JSON summaries, but it must cover at least these rows:

| Evidence row | Favors H1 | Favors H2 | Favors H3 | Favors H4 | Favors H5 |
|---|---|---|---|---|---|
| Prompt-end object-set probe | strong object/count support before rollout | weak later-object support | partial support only near row onset | non-diagnostic | non-diagnostic |
| Forced `gt_prefix` depth curve | remaining support stable across `K` | support changes with each prefix | support hardens around `pre_x1/x1` | `entry_after_separator` healthy but `free_boundary` weak | target states healthy but rollout recovery missing |
| Generated-prefix corruption | support survives corruption | next posterior redirects by prefix | wrong `x1` basin maps to prefix object | stop margin worsens after prefix | corrupt-prefix recovery state lacks target credit |
| `pre_x1` versus `post_x1` identity | already strong before `pre_x1` | grows as row unfolds | large hardening jump at `x1/y1` | non-diagnostic unless tied to EOS | non-diagnostic without target/loss readout |
| Occlusion/patching migration | early multi-object causal support | row-ordinal causal migration | local onset candidate routing | non-diagnostic | non-diagnostic |
| Training target/loss accounting | non-diagnostic | non-diagnostic | non-diagnostic unless target is onset-specific | non-diagnostic unless stop target is implicated | missing or weak coverage/recovery credit |

Rows that are compatible with more than one hypothesis must be marked as
compatible-only in the report, not as decisive support.

## Study Architecture

The study is a staged evidence pipeline. Each lane writes machine-readable
artifacts and a small summary. Later lanes may consume earlier outputs, but
earlier artifacts remain valid standalone evidence.

Primary artifact root:

```text
/data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_val200
```

Mandatory top-level outputs:

```text
resolved_inputs.json
gate_report.json
```

`resolved_inputs.json` is the source-of-truth manifest for every downstream
lane. It must record analyzer version or git commit when available, requested
and resolved artifact roots, source artifact absolute paths, file sizes, SHA256
hashes, dataset JSONL path and hash, slice policy, config path, checkpoint path,
prompt/decode/model fingerprints when available, score and duplicate-guard
policy ids when available, and the exact `raw` versus `guarded` metric family
used by the report.

### Lane A: Rollout Anatomy

Purpose:

Describe what happens in free autoregressive decoding at row and token
granularity before any new model-forward probes are added.

Inputs:

```text
gt_vs_pred.jsonl
pred_token_trace.jsonl
pred_confidence.jsonl
gt_vs_pred_scored.jsonl
gt_vs_pred_scored_guarded.jsonl
eval/metrics.json
eval/metrics_guarded.json
eval/matches.jsonl
eval/matches_guarded.jsonl
eval/duplicate_guard_report.json
summary.json
resolved_config.json
```

Outputs:

```text
rollout_anatomy/per_row.jsonl
rollout_anatomy/per_image.jsonl
rollout_anatomy/coverage_survival.csv
rollout_anatomy/summary.json
rollout_anatomy/report.md
```

Required metrics:

- generated row count versus GT object count;
- under/equal/over generation counts;
- row ordinal histogram;
- raw and guarded TP/FP/FN summary where available, with the two index domains
  kept separate;
- duplicate-guard kept/suppressed raw indices when guarded metrics are used;
- per-row keys: `source_line_idx`, `coco_image_id`, `image`, source row hash,
  `raw_pred_idx`, nullable `guarded_pred_idx`, `suppressed_by_guard`, raw match
  label, guarded match label, and guard mapping source;
- row labels: `tp_like`, `duplicate_like`, `near_duplicate_like`,
  `wrong_desc_fp`, `unmatched_fp`, `invalid_row`, `unparsed_row`;
- coverage survival curve by generated row index;
- token-kind logprob summaries for desc, `x1`, `y1`, `x2`, `y2`, separator,
  and EOS;
- pad-after-EOS and max-new-token truncation indicators;
- `first_im_end_index`, `tokens_after_first_im_end`,
  `endoftext_after_im_end_count`, `raw_ends_with_im_end`, and
  `hit_max_new_tokens` as distinct fields.
- parse policy and counters: strict parse setting, raw errors, error entries,
  invalid rows, unparsed rows, degenerate boxes, and dropped predictions.

Interpretation discipline:

Grammar-forced structure tokens can have near-zero negative logprob because
the compact grammar masks alternatives. Treat description tokens, coordinate
tokens, and free-boundary EOS/separator decisions as more informative than
forced row-start tokens.

The existing confidence sidecar may not expose compact-row descriptor spans.
Lane A must implement a compact-row trace parser over `generated_token_text`
instead of assuming JSON-shaped `desc` span metadata exists.

Token-logprob summaries must end at the first `<|im_end|>` unless a lower-level
response-token span proves a different valid span. Post-EOS `<|endoftext|>`
padding is an artifact-health indicator, not evidence for H4 stop dynamics.

### Lane B: Prefix Boundary Causality

Purpose:

Separate "the model cannot select the next object" from "the model can select
it only after a separator/new row is forced."

Primary existing tool:

```text
src/analysis/prefix_rollin_teacher_forced_diagnostic.py
```

Outputs:

```text
prefix_boundary/per_case.jsonl
prefix_boundary/summary.json
prefix_boundary/by_prefix_quality.csv
prefix_boundary/report.md
```

Required modes:

- `gt_prefix`, with `K=0..N` for each image in the chosen slice;
- `generated_prefix`, replaying the existing free-decode artifact;
- separate readouts for `gt_prefix_free_boundary` and
  `gt_prefix_entry_after_separator`;
- prefix labels: `empty_prefix`, `clean_prefix`, `fp_prefix`,
  `duplicate_prefix`, `invalid_prefix`, `ambiguous_prefix`.

Required preflight fixes:

- the exact historical checkpoint config may not parse under the current
  training config schema; the implementation must use either an analysis-only
  compatibility loader or a derived analysis config with hard assertions for
  dataset path, image root, prompt hash, `compact_full`, coord-token `xyxy`,
  and random-permutation ordering;
- boundary mode must be explicit: `marker_object_ref_boundary` versus
  `legacy_newline_boundary`;
- generated-prefix targets must be selected from Lane A match state, not by raw
  `pred_count` or ordinal depth.
- the implementation must support shard/range parameters before Lane B is run
  on multiple GPUs. The current single-output diagnostic must not be launched
  eight times into one output directory.

Required metrics:

- `continue_minus_eos_margin`;
- valid next-object mass;
- EOS logprob and rank;
- separator/newline logprob and rank;
- remaining GT count;
- prefix depth and generated row count;
- breakdown by object count, repeated-description count, small/medium/large
  area bucket, and crowding bucket.

Gate contribution:

This lane is the primary test for H4.

The first Lane B implementation gate is a targeted fix for the existing
prefix-rollin diagnostic boundary test. If
`tests/test_prefix_rollin_teacher_forced_diagnostic.py` does not pass, H4
evidence is not trusted.

### Lane C: Next-Object Posterior and x1 Basin Attribution

Purpose:

Explain what the model's first coordinate distribution is pointing at when it
does not point at the target object.

Primary existing tool to extend:

```text
src/analysis/hard_ce_coord_logit_locality.py
scripts/analysis/run_hard_ce_coord_logit_locality.py
```

Outputs:

```text
x1_basin_attribution/per_slot.jsonl
x1_basin_attribution/per_case.jsonl
x1_basin_attribution/summary.json
x1_basin_attribution/plots/
x1_basin_attribution/report.md
```

Required readouts:

- keep `x1`, `y1`, `x2`, and `y2` separate;
- compare teacher-forced prefixes and self-prefixes;
- label self-prefixes as clean, FP, duplicate, invalid, or ambiguous;
- do not select self-prefix targets by ordinal depth alone for FP, duplicate,
  invalid, or ambiguous prefixes;
- consume Lane A match state to record `matched_prefix_gt_indices`,
  `remaining_gt_indices`, `fp_prefix_object_indices`,
  `intended_target_gt_idx`, and `target_selection_rule`;
- attribute `x1` top peaks to:
  - target GT object;
  - other same-image GT object;
  - same-desc competitor GT object;
  - previous generated object;
  - false-positive prefix object;
  - no local object / diffuse;
- report target rank, best-other rank, target margin, entropy, mass@4,
  mass@8, GT top1, top1 distance, and wrong-object attribution.

Implementation constraint:

The current locality harness stages are not enough by themselves for this
anatomy lane. Add an explicit wrapper or stage that consumes Lane A match state
and writes anatomy-specific `per_slot` / `per_case` rows. Do not infer the
self-prefix target from `normalized.objects[depth]` when the prefix is FP,
duplicate, invalid, or ambiguous.

Gate contribution:

This lane is the primary test for H3 and an important test for H5.

### Lane D: Hidden-State Probe and Causal Patching

Purpose:

Test whether object set, remaining objects, and next-object identity are
encoded before rollout hardens at coordinates.

Primary implementation reference:

```text
src/analysis/qwen3_vl_instance_binding.py
```

This reference is not directly reusable without adaptation because its current
rendering path is JSON-oriented. The compact-full study must use compact-full
row rendering and token inventory. The implementation plan must create a
compact-full rendering and position-inventory adapter before any GPU hidden
state extraction, rather than only changing checkpoint paths in the existing
Qwen3-VL instance-binding config.

Outputs:

```text
hidden_state_probe/selected_cases.jsonl
hidden_state_probe/position_inventory.jsonl
hidden_state_probe/probe_rows.jsonl
hidden_state_probe/patch_rows.jsonl
hidden_state_probe/summary.json
hidden_state_probe/report.md
```

Required positions:

- prompt end;
- row start;
- desc end;
- box start;
- `pre_x1`;
- `post_x1`;
- `post_y1`;
- row end or separator;
- final generated prefix state when using generated-prefix replay.

Required inventory contract:

- roles must be config-driven and must survive config parsing into
  `resolved_inputs.json`;
- JSON-oriented roles such as `desc_closing_quote` and `bbox_open_bracket` may
  remain in the older instance-binding study but cannot be treated as the
  compact-full role set;
- each row must record assistant-relative and absolute token positions;
- each row must record `render_source` as one of `strict_compact_full`,
  `lane_c_forced_continuation`, or `generated_prefix_replay`;
- each row must record source-aware separator truth: `strict_compact_full` with
  `prefix_mode=teacher_forced` requires `none_marker_delimited`, while
  `lane_c_forced_continuation` and `generated_prefix_replay` may record either
  `none_marker_delimited` or `newline` to match the actual rendered text;
- each row must record `prefix_state_kind`; ordinary roles use
  `not_applicable`, while `final_generated_prefix_state` uses
  `teacher_forced_prefix_boundary` for non-empty teacher-forced prefixes,
  `empty_prefix_prompt_end` for empty prefixes, and later generated-prefix work
  may use `generated_prefix_boundary` or `partial_row`;
- `assistant_relative_token_index` may be null for `prompt_end` and for
  `final_generated_prefix_state` only when its state is
  `empty_prefix_prompt_end`;
- layer groups must be parsed from config and written to stage metadata before
  hidden vectors are extracted;
- duplicate, missing, negative, or unknown compact roles must fail fast in CPU
  tests.

Required tasks:

- object count prediction;
- remaining object count prediction;
- next-object identity among same-image candidates;
- same-desc ordinal prediction;
- target bbox center or `x1/y1` candidate ranking;
- coverage bitset or recoverable-FN label when feasible.

Probe task rows must include:

```text
task
role
layer
prefix_condition
target_label_source
case_id
object_count
remaining_count
intended_target_gt_idx
prefix_quality
x1_top_peak_attribution
x1_target_rank
```

Required causal interventions:

- attenuation and copy/patch controls at late schema / immediate `pre_x1`;
- current description span;
- previous row geometry span;
- wrong-image same-desc donor;
- same-image same-desc donor;
- self-noop control.

Gate contribution:

This lane may close H1, H2, or H3 only when probe and causal intervention
evidence agree. Probe-only evidence is not sufficient for convergence.

### Lane E: Attention and Occlusion Migration

Purpose:

Visualize and test how attention or visual sensitivity moves across objects
during rollout.

Outputs:

```text
attention_migration/per_token.jsonl
attention_migration/per_row.jsonl
attention_migration/summary.json
attention_migration/plots/
attention_migration/report.md
```

Required controls:

- run only on a small curated slice after Lanes A-C identify the relevant
  failure cohorts;
- use an attention implementation that can expose attention weights;
- record the attention implementation and memory settings;
- aggregate attention by semantic regions, not raw token heatmaps only;
- pair attention with occlusion or patching before making causal claims.

Region buckets:

- current target GT bbox region;
- other remaining GT regions;
- previously generated matched object regions;
- duplicate or false-positive prefix regions;
- background image tokens;
- text prefix / schema tokens.

Gate contribution:

This lane is explanatory and visual. It cannot be the only evidence for H1 or
H2.

### Lane F: Supervision and Credit-Assignment Readout

Purpose:

Connect rollout failures to the training objective without changing training
yet.

Inputs:

```text
training logging.jsonl
checkpoint trainer_state.json
resolved_config.json
hard_ce_coord_logit_locality outputs
prefix_boundary outputs
x1_basin_attribution outputs
```

Outputs:

```text
supervision_credit/summary.json
supervision_credit/report.md
```

Required readouts:

- target-mix availability for continuation and multi-positive states where
  logged;
- trie support and balance trends where available;
- coordinate slot locality, especially `x1`;
- relationship between teacher-forced health and self-prefix degradation;
- whether false-positive prefixes create uncredited recovery states;
- which possible loss changes would be justified by evidence, and which would
  be speculation.

Gate contribution:

This lane is the primary test for H5.

## Decision Gates

### Gate 0: Artifact Validity

Before any mechanism claim:

- `gt_vs_pred.jsonl` must be indexed by physical JSONL source line
  `source_line_idx`; `pred_token_trace.jsonl.line_idx` must equal that
  `source_line_idx`;
- evaluator ordinal ids must be renamed to `eval_record_idx`; they must not be
  confused with COCO `image_id`;
- raw generated text, parsed predictions, and trace token counts must be
  internally consistent;
- invalid rows must be counted, not silently dropped;
- decode config, checkpoint path, prompt hash, model fingerprint when available,
  dataset JSONL hash, and dataset slice policy must be recorded in
  `resolved_inputs.json`;
- source JSONL row identity, `coco_image_id`, image path, row hash, and GT
  object counts must reconcile with the chosen slice;
- raw prediction totals and invalid-row counters must reconcile with
  `summary.json`;
- guarded metric totals used for interpretation must reconcile with
  `eval/metrics_guarded.json` or be explicitly marked unavailable;
- requested and resolved artifact roots must both be recorded;
- `resolved_inputs.json` hashes must match current source files before any lane
  reads them.

If this gate fails, stop mechanism interpretation and repair or regenerate the
artifact.

### Gate 0b: Eval and Join Validity

Before any AP/COCO-style, guarded, or raw-vs-guarded comparison claim:

- `load_comparable_artifact(..., require_score=True)` must pass for
  `gt_vs_pred_scored.jsonl` and `gt_vs_pred_scored_guarded.jsonl`; otherwise the
  report must label score-bearing claims as `debug-f1ish-only`;
- scored artifacts must expose or be paired with score provenance such as
  `score_policy_fingerprint`, source raw artifact identity, prompt/decode/model
  fingerprints, and `metric_bearing=true` when current loaders require them;
- raw and guarded row-index domains must be joined only through
  `eval/duplicate_guard_report.json` kept/suppressed raw-index mappings;
- every guarded match pred index must map back to exactly one raw pred index;
- token-kind summaries must exclude tokens after the first `<|im_end|>` unless a
  verified lower-level valid response span says otherwise;
- report scope fields must be copied from `resolved_config.json` and
  `summary.json`, not inferred from artifact path names.

### Gate 1: Free-Decode Anatomy Before Forward Probes

Lane A must run before any hidden-state or attention work. The first report
must state whether the main symptom is under-generation, duplication, invalid
format, wrong-desc FP, geometry error, or mixed.

### Gate 2: Boundary Causality Before Decode Recommendations

No stop/continue recommendation is allowed until Lane B separates
`free_boundary` from `entry_after_separator`. If the latter is healthy and the
former is weak, H4 becomes the leading decode hypothesis.

### Gate 3: x1 Attribution Before Objective Recommendations

No onset-aware coordinate loss recommendation is allowed until Lane C shows
where wrong `x1` mass goes. A diffuse failure, a previous-object local failure,
and an other-GT local failure imply different objective changes.

### Gate 4: Causal Hidden-State Evidence Before Representation Claims

No claim that the model globally perceives the object set, dynamically
discovers objects, or stores identity at a schema position may be considered
converged without Lane D causal controls.

### Gate 5: Attention Is Not Causality

Attention maps can appear in the final report only when labeled as
visualization or when paired with occlusion/patching evidence that supports the
same mechanism.

### Gate 6: Training Proposal Requires Mechanism Convergence

Any proposed training change must name the hypothesis it targets:

- H1: serialization/reranking/set-readout support;
- H2: discovery-oriented rollout-prefix training;
- H3: onset-aware `x1` or candidate-contrastive supervision;
- H4: stop/continue boundary calibration;
- H5: rollout-prefix coverage credit assignment or residual correction.

No new production training run should launch from an inconclusive mechanism
report.

Early reports may include only gate-limited candidate implications. A report
with a `not_converged_*` status must state:

```text
production_training_recommendation: none
```

Training recommendations become eligible only after the relevant hypothesis
gate has passed and the report records the matched-baseline scope required for
that recommendation.

## Convergence Labels

Each report must choose exactly one status label:

```text
not_converged_artifact_validity
not_converged_rollout_anatomy_only
not_converged_missing_prefix_causality
not_converged_missing_x1_attribution
not_converged_missing_causal_hidden_state
not_converged_missing_supervision_credit
not_converged_attention_only
converged_global_set_then_serialize_H1
converged_dynamic_discovery_H2
converged_x1_commitment_H3
converged_decode_stop_bottleneck_H4
converged_credit_assignment_mismatch_H5
converged_mixed_H3_H4
converged_mixed_H3_H5
converged_mixed_H4_H5
converged_mixed_H3_H4_H5
converged_mixed_other
```

The final status must explain why stronger alternatives were rejected or left
open. In addition to the single headline status, every report must include a
per-hypothesis state table:

```text
H1: supported | weakened | rejected | inconclusive_missing_lane | not_applicable
H2: supported | weakened | rejected | inconclusive_missing_lane | not_applicable
H3: supported | weakened | rejected | inconclusive_missing_lane | not_applicable
H4: supported | weakened | rejected | inconclusive_missing_lane | not_applicable
H5: supported | weakened | rejected | inconclusive_missing_lane | not_applicable
```

If the headline is `converged_mixed_other`, the exact hypothesis combination
must be named in the report summary.

## Implementation Sequence

The implementation plan should be split into small tasks and should not begin
GPU-heavy work until CPU artifact-contract tests pass.

1. Create the analysis config and resolved-input contract.
2. Implement Gate 0 and Gate 0b artifact, score-provenance, and join checks.
3. Implement Lane A rollout anatomy reader and row matcher.
4. Add CPU tests for trace alignment, row parsing, row labels, and coverage
   survival.
5. Run Lane A on the existing `val200` artifact and write the first report.
6. Fix or wrap Lane B config compatibility, boundary-mode detection, and
   shard/range outputs before any multi-GPU prefix-boundary run.
7. Add Lane C `x1` basin attribution to the existing locality harness using
   Lane A target-selection state.
8. Write the Gate 1-3 synthesis report.
9. Create sharded execution manifests for any Lane B/C/D GPU work before
   launching more than one process.
10. Decide whether Lane D is required; if yes, adapt compact-full hidden-state
   positions and causal patch controls.
11. Decide whether Lane E is required; if yes, run only on the curated cases
   selected by Lanes A-C.
12. Write Lane F supervision/credit assignment synthesis.
13. Write the final mechanism report and training-implication memo.

## Eight-GPU Execution Strategy

The implementation plan must explicitly classify every stage by execution
shape before launch:

```text
cpu_only
single_gpu
eight_gpu_sharded
eight_gpu_independent_configs
sequential_after_merge
```

Default launch policy:

- Lane A is CPU-only and should run before any GPU allocation.
- Lane B is single-GPU with the current script. It may use 8 GPUs only after
  shard/range support exists; shard by stable `source_line_idx`, not by `K`, so
  every image keeps its full `K=0..N` curve.
- Lane C is single-output with the current locality harness. It may use 8 GPUs
  only after shard suffixes and a merge stage exist for per-slot rows and dense
  arrays.
- Lane D hidden-state extraction, patching, and donor controls are the main
  8-GPU workloads. Shard selected cases across 8 devices, write per-shard
  summaries, then run merge/report stages sequentially after all shard
  manifests are complete.
- Existing Qwen instance-binding shard stages select local `cuda:0`; parallel
  launches must set `CUDA_VISIBLE_DEVICES=$i` so each worker sees one physical
  GPU.
- Lane E is optional and memory-heavy. Run it on a small curated slice after
  Lanes A-C, and do not run it concurrently with Lane D.
- Lane F is CPU-only once upstream artifacts exist.

Every sharded stage must write:

```text
<stage>/shards_manifest.json
<stage>/shards/shard_{idx:03d}-of-{n:03d}/summary.json
<stage>/shards/shard_{idx:03d}-of-{n:03d}/<stage_rows>.jsonl
merge_summary.json
```

Merge gates:

- fail if any expected shard is missing;
- fail if stale extra shard files are present unless an explicit
  `allow_extra_shards` debug flag is set;
- fail if row counts do not match the manifest;
- fail if any `(source_line_idx, case_id, prefix_mode, prefix_k, role, slot)` key
  is duplicated within the stage's declared uniqueness domain;
- preserve source shard id and case id in every merged row;
- never overwrite a completed shard with a different config fingerprint;
- record CUDA device, `CUDA_VISIBLE_DEVICES`, batch size, base seed, case seed
  policy, stage name, checkpoint, config hash, selected-cases hash, and artifact
  root in every shard summary.

Do not parallelize stages that consume merged outputs until the merge gate has
passed.

Before a real 8-GPU run, perform a dry-run materialization of the shard manifest
and assert all planned output paths are unique.

## Expected File Surfaces

The implementation plan may create or modify these files:

```text
configs/analysis/autoreg_object_rollout/ckpt3664_val200.yaml
scripts/analysis/run_autoreg_object_rollout_anatomy.py
src/analysis/autoreg_object_rollout.py
tests/test_autoreg_object_rollout.py
```

It may modify these files only if the implementation plan proves extension is
cleaner than duplication:

```text
src/analysis/hard_ce_coord_logit_locality.py
scripts/analysis/run_hard_ce_coord_logit_locality.py
src/analysis/prefix_rollin_teacher_forced_diagnostic.py
src/analysis/qwen3_vl_instance_binding.py
```

Implementation should prefer small new analysis helpers over changing
production inference or evaluation code.

## Verification

Minimum verification for the design-to-plan transition:

```bash
python - <<'PY'
from pathlib import Path
p = Path("docs/superpowers/specs/2026-06-01-autoregressive-object-rollout-anatomy-design.md")
text = p.read_text()
required = [
    "H1: Global Set Then Sequential Translation",
    "H2: Sequential Dynamic Discovery",
    "H3: Soft Pre-x1 Binding, Hard x1 Commitment",
    "H4: Decode Stop and Free-Boundary Bottleneck",
    "H5: Supervision and Credit-Assignment Mismatch",
    "Lane A: Rollout Anatomy",
    "Lane B: Prefix Boundary Causality",
    "Lane C: Next-Object Posterior and x1 Basin Attribution",
    "Lane D: Hidden-State Probe and Causal Patching",
    "Lane E: Attention and Occlusion Migration",
    "Lane F: Supervision and Credit-Assignment Readout",
    "Decision Gates",
    "Artifact Validity",
    "Convergence Labels",
    "Verification",
]
missing = [item for item in required if item not in text]
if missing:
    raise SystemExit(f"missing required sections: {missing}")
print("design spec coverage ok")
PY
```

Minimum verification for implementation tasks:

- targeted pytest for each new parser, matcher, and summarizer;
- dry-run or tiny fixture checks before loading the checkpoint;
- `PYTHONPATH=/data/CoordExp` for direct script launches;
- artifact-count checks after every stage;
- `tests/test_prefix_rollin_teacher_forced_diagnostic.py -q` must pass before
  Lane B H4 evidence is trusted;
- `tests/test_hard_ce_coord_logit_locality.py -q` plus anatomy-specific Lane C
  fixture tests must pass before x1-basin claims;
- compact-full Lane D position-inventory tests must cover duplicate, missing,
  negative, unknown roles, render-source vocabulary, source-aware separators,
  and `final_generated_prefix_state` prefix-state vocabulary before hidden-state
  extraction;
- config-load tests must prove checkpoint path, compact-full roles, layer groups,
  prompt hash, coord-token `xyxy`, and dataset slice are resolved into
  `resolved_inputs.json`;
- artifact-contract tests must check raw/guarded index mapping, score provenance,
  EOS/pad truncation, source-line joins, and shard/merge manifests;
- fake-logit tests must exercise exact rank, nullable rank, EOS rank, separator
  rank, and target-margin calculations;
- report generation from machine-readable summaries, not hand-copied numbers.

## Report Shape

Every report should use this structure:

```text
Scope
Inputs
Artifact Validity
Main Symptom
Hypothesis Evidence
Failure Modes
Root Causes
Training Implications
Open Uncertainty
Next Gate
```

Do not present a tiny, `val200`, proxy, or partial-run result as full
validation.

## Training Implication Map

Allowed implication language:

- If H1 is supported: prioritize serialization, reranking, Oracle-K, or set
  readout experiments.
- If H2 is supported: prioritize rollout-prefix discovery training and
  attention/region coverage supervision.
- If H3 is supported: prioritize onset-aware `x1` routing, candidate contrast,
  and same-desc disambiguation objectives.
- If H4 is supported: prioritize stop/continue calibration and boundary
  diagnostics before changing object geometry supervision.
- If H5 is supported: prioritize prefix-conditioned coverage credit,
  residual-correction targets, or self-prefix recovery supervision.

Disallowed implication language:

- "Attention proves discovery" without occlusion or patching.
- "SoftCE improves recall" without matched rollout metrics.
- "The model does not see missed objects" without forced-prefix or probe
  evidence.
- "EOS is the only problem" if `entry_after_separator` is also weak.
- "H3 is solved" if `x1` is analyzed only in aggregate bbox metrics.

## User Review Gate

Before writing the implementation plan, the user must review:

```text
docs/superpowers/specs/2026-06-01-autoregressive-object-rollout-anatomy-design.md
```

Review should explicitly confirm or revise the H4/H5 definitions:

- H4 as decode stop / free-boundary bottleneck;
- H5 as supervision and credit-assignment mismatch.

After approval, write the implementation plan to:

```text
docs/superpowers/plans/2026-06-01-autoregressive-object-rollout-anatomy.md
```
