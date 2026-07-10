---
title: Autoregressive Duplication Mechanism Diagnosis Charter
date: 2026-06-10
status: active-grill-record
owner: codex
evidence_scope: none-yet
---

# Autoregressive Duplication Mechanism Diagnosis Charter

## Purpose

This note records the resolved investigation standard for the compact-full
coordinate-token duplication burst diagnosis. It is a charter for the
mechanism study, not a result report.

The target is not to prove that duplication bursts exist, that a duplicate
guard can suppress them, or that a surface explanation is plausible. Those
phenomena and surface-level explanations are already established. The goal is
to find the deepest direct forward-propagation mechanism and origin of the
burst: what state change happens before the burst, which token position causes
the transition, which visual or coordinate basin becomes dominant, and which
intervention breaks the loop while preserving true positives.

## Evidence Standard

The study must separate:

- symptom: emitted duplicate boxes, invalid rows, empty predictions, or
  over-continuation;
- event: the temporal row where a burst begins or becomes self-sustaining;
- mechanism: hidden-state, logit, attention/routing, boundary, or
  stop/continue change that causes the event;
- intervention: a causal probe or controlled decode change that breaks the
  event;
- interpretation: whether coordinate tokens, visual routing, row-boundary
  dynamics, stop/continue calibration, or multiple interacting mechanisms are
  implicated.

Aggregate AP, duplicate suppression counts, and guarded metrics are context,
not proof of mechanism.

Dynamic adjustment is welcome when an emerging path looks promising,
attractive, and likely to influence the final mechanistic picture. The roadmap
is an execution scaffold, not a ceiling: if a readout exposes a deeper origin
or a higher-leverage causal path, further exploration may be updated in-flight
as long as artifact handles, evidence scope, and interpretation boundaries stay
explicit.

## Anchor Evidence

The clean parent anchor pair uses the same free-rollout protocol: COCO val128,
compact-full, no compact grammar, no row separator, temperature `0.0`,
repetition penalty `1.1`, and `max_new_tokens=3084`.

- No-aligner parent checkpoint:
  `/data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/compact_full_fullobj_random_sft_bsz16_4epoch_tokenrows_v2/compact-full-fullobj-random-sft-bsz16-4epoch-tokenrows-v2/v1-20260601-062428/checkpoint-3668`
- No-aligner parent rollout:
  `/data/CoordExp/outputs/infer/loss_only_instance_enumeration_ablation_active_vs_none/compact_full_prefix_rollin_balance2_no_aligner_parent_val128_freegreedy_ckpt3668_val128_bsz8_temp0_rp1p10_max3084_chatfix_8gpu`
- Aligner-tuned parent checkpoint:
  `/data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/compact_full_fullobj_random_sft_llm_aligner_lora_packed_bsz16_4epoch_tokenrows_v2/compact-full-fullobj-random-sft-llm-aligner-lora-packed-bsz16-4epoch-tokenrows-v2/v5-20260608-081447/checkpoint-1824`
- Aligner-tuned parent rollout:
  `/data/CoordExp/outputs/infer/loss_only_instance_enumeration_ablation_active_vs_none/compact_full_prefix_rollin_balance2_original_latest_aligner_parent_val128_freegreedy_ckpt1824_val128_bsz8_temp0_rp1p10_max3084_chatfix_8gpu`

The parent pair should calibrate onset definitions and baseline mechanism
readouts. Later aux/none continuation checkpoints should be connected through
the same case ledger and analyzed jointly, not as a separate explanation silo.
Continuation checkpoints may amplify, suppress, or reroute a mechanism that
predates them.

## Joint Ledger Scope

Use one joint ledger across parent and continuation checkpoints. Do not create
separate parent and continuation ledgers, and do not add a `checkpoint_family`
field. The schema should avoid baking in a family taxonomy before the mechanism
is known.

Use explicit, lower-level descriptors instead:

- `checkpoint_label`;
- `checkpoint_path`;
- `rollout_root`;
- `aligner_tuned`;
- `aux_loss_kind`;
- `training_ordering`;
- `calibration_role`;
- `decode_protocol_id`.

The clean parent pair can have `calibration_role: parent_anchor`, while aux/none
continuations can have comparison roles such as `continuation_stress` or
`auxiliary_loss_probe`. Those roles are interpretation aids, not mechanism
labels.

## Phase 0 Ledger Contract

Use a dual onset definition:

- `pair_onset`: the first generated row that is duplicate-like with any
  previous same-desc row. This matches existing duplicate-collapse vocabulary
  and is the best target for immediate forward-propagation probes.
- `component_onset`: the first generated row belonging to the eventual largest
  same-desc duplicate component. This is the primary unit for statistical
  position/order analysis.

Use `primary_burst_onset_row` as the canonical Phase-1 probe anchor. It is the
first generated row where the eventual largest same-desc duplicate component
reaches size `2`: in other words, the first repeat row of the final dominant
burst component. This avoids over-centering GPU probes on an early nuisance
`pair_onset` that does not grow into the stable burst, while also avoiding a
seed-only `component_onset` row that may still be a plausible first detection.
The ledger should still preserve `pair_onset`, `component_onset`, and component
growth rows so the anchor can be audited case by case.

If multiple same-desc components tie for largest size, choose the primary
component by earliest size-2 growth row, then earliest seed row, then larger
overlap with the dominant spatial-basin component. This tie-breaker prioritizes
causal onset over final pile geometry.

Track two component surfaces:

- `same_desc_component`: same normalized desc/class and spatially
  duplicate-like. This is the canonical emitted-duplication symptom surface.
- `spatial_basin_component`: class-agnostic boxes whose centers or envelopes
  cluster in the same visual/coordinate basin. This is the mechanism-origin
  surface for testing whether the model first locks onto a region and only
  later recruits or cycles through desc/class tokens.

Use the same duplicate-like geometry predicate for both surfaces: bbox IoU
above the duplicate threshold, or sufficiently similar size with center
distance inside the configured local radius. The only difference is the desc
constraint. `same_desc_component` requires normalized desc/class equality;
`spatial_basin_component` ignores desc/class and reports desc purity or desc
mix separately. This keeps the symptom and basin surfaces comparable while
still allowing the analysis to detect a region attractor before class labels
stabilize.

The ledger should record component growth thresholds, such as the first rows
where the component reaches size `2`, `3`, `5`, and `10`.

Record centers both as continuous normalized coordinates and as a deterministic
`3x3` normalized image-grid bucket. Human labels such as top-left and
middle-left may appear in markdown summaries, but machine-readable ledger rows
should preserve numeric centers and fixed buckets.

For every selected case and checkpoint, record:

- record index, image id, image path, checkpoint label, rollout root;
- parse status and raw emitted prediction count;
- GT count and matched/unmatched/ignored prediction context when available;
- `pair_onset` row, source row, desc/class, bbox, normalized center, and
  center bucket;
- `component_onset` row, largest component size, desc/class, component bbox
  envelope, normalized center, and center bucket;
- normalized order features including `row_idx / pred_count` and
  `row_idx / max(gt_count, 1)`;
- component growth rows and whether the burst is early or late;
- prefix context counters before onset:
  - `prefix_matched_tp_count`: pre-onset predictions matched to GT at
    IoU >= 0.50 when eval matches are available;
  - `prefix_clean_row_count`: pre-onset rows that are valid and not already in
    a duplicate component;
  - `prefix_basin_seed_count`: pre-onset rows in the later burst's
    spatial basin, regardless of desc/class;
- optional `raw_token_window_preview` around onset as a capped human-audit aid.

Use raw `eval/matches.jsonl` at IoU `0.50` as the canonical source for
`prefix_matched_tp_count` and raw TP/unmatched/ignored prediction context.
Guarded match files are context only, because duplicate guarding rewrites the
emitted prefix after the model has already generated it. If raw matches are
absent, use a local same-desc IoU50 fallback matcher and record
`match_source=fallback_local_iou50`.

Keep one ledger row for every `(record, checkpoint)` pair in the joint
comparison, even when prediction parsing is empty, invalid, or truncated. Empty
or invalid outputs should carry `parse_status`, `pred_count=0` or a recoverable
partial count, null onset/component fields, and field-level `exclusion_reason`
values where computation is impossible. Do not drop these rows: "no burst
because no usable emission" and "no burst because healthy stopping" are
different mechanistic states, and aux-vs-none contrast cases rely on that
distinction.

Slot-level token anchors are not mandatory base-ledger fields. Keep the base
ledger focused on stable row, component, and basin identities. Provide analysis
helpers or materialized post-hoc views that can derive row-start, desc span,
`box_start`, `x1`, `y1`, `x2`, `y2`, row-boundary, and stop-token indices from
`pred_token_trace.jsonl` for selected onset windows. The token trace JSONL
remains the canonical token source; ledger token windows are previews only and
must not be used as the authoritative trace.

## Initial Required Cases

The first case panel must include at least:

- record `54`, image id `5586`: severe top-left basin;
- record `33`, image id `2685`: middle-left / later component contrast;
- record `47`, image id `5001`: aux-heavy top-left duplicate pressure;
- record `82`, image id `7991`: over-emission despite tiny GT count;
- record `96`, image id `9590`: contrast case where aux may emit nothing and
  none emits many rows.

Add at least one false-basin case where the duplicate region appears visually
empty or semantically irrelevant.

False-basin selection should do the best possible data analysis first, then
require user manual review when visual emptiness or semantic irrelevance is
part of the claim. The ledger may compute a `false_basin_candidate_score` from
low GT overlap, strong spatial-basin component size, recurrent top-left or
middle-left basin position, and onset order. The final label should include
`false_basin_reviewed` and a short review reason, because COCO partial
annotation can make a region appear false even when unlabeled small objects or
ambiguous evidence exist.

The false-basin score is advisory only. It should select cases for review, not
automatically label a region as visually empty or semantically irrelevant.

## Probe Ordering

The initial execution should be:

1. Build the Phase 0 ledger without GPU model forwards.
2. Use ledger statistics to select representative onset windows centered first
   on `primary_burst_onset_row`.
3. Run targeted logit, hidden-state, attention/routing, and counterfactual
   probes against the selected windows.
4. Only then interpret whether the direct trigger is coordinate-token basin
   attraction, visual routing, hidden-state recurrence, stop/continue collapse,
   boundary/template looping, or a hybrid.

Phase 1 case selection should use a fixed stratified panel, not top severity
alone. The panel must include the required records `54`, `33`, `47`, `82`, and
`96`, then select representative rows from:

- `early_primary_burst`: `primary_burst_onset_row / max(pred_count, 1) <= 0.25`;
- `late_primary_burst`: onset fraction `>= 0.60`;
- `same_desc_dominant`: large same-desc component with high desc purity;
- `spatial_basin_before_desc`: spatial basin forms earlier or larger than the
  same-desc component;
- `false_basin_candidate`: high candidate score, with final visual-emptiness or
  semantic-irrelevance claims gated by manual review;
- `contrast_case`: the same record diverges strongly across checkpoints in
  onset row, component size, or empty-vs-many emission.

Top severity can rank candidates inside a stratum, but must not be the whole
selector. Otherwise the first GPU probes would over-sample obvious pileups and
under-sample the mechanistically decisive contrasts.

For selected cases, materialize a row window from
`primary_burst_onset_row - 3` through `primary_burst_onset_row + 8`, clipped to
available prediction rows. This window is meant to preserve the clean prefix,
burst seed, first repeat, and early self-sustaining phase without turning Phase
0 into a full long-burst classifier.

## Pre-Dessert FN Guidance Probe

Include a false-negative visibility/guidance panel before treating missing
objects as a visual-capacity limit. For unmatched GT objects, first build a
post-hoc manifest from scored rollout artifacts that separates:

- artifact-invalid or parse-blocked cases;
- nearby same-desc proposals, where language-side prefix or coordinate guidance
  may rescue the missing object;
- nearby wrong-desc proposals, where the visual region may be available but
  binding/enumeration failed;
- tiny low-salience objects;
- no-proposal cases, where visual miss is the stronger candidate explanation.

For selected missing objects, emit at least `desc_only`, `desc_x1`, and
`desc_x1_wrong_control` guidance tiers. The later decode interpretation should
be conservative:

- target `desc_x1` succeeds while `desc_only` fails: evidence for prefix/context
  or coordinate-guidance fragility;
- both fail with no nearby visual proxy: stronger evidence for low salience or
  visual miss;
- wrong-control succeeds similarly to target `desc_x1`: coordinate hint leakage
  or weak binding, not a clean target rescue.

The current selector artifact is
`phase4_fn_visibility_guidance_probe_aux_latest_ckpt32_val128`; it is a probe
panel, not proof of perception. It should be promoted to guided continuation
decodes only after the selected cases are reviewed for artifact validity and
research value.

## Phase 1 Probe Contract

After Phase 0 selects exact windows, run hidden-state and logit trajectory
probes before attention/routing probes on the same selected windows.
Hidden-state/logit evidence is closest to the forward-propagation transition;
attention is crucial routing evidence, but should not be treated as a causal
mechanism claim before perturbation.

The hidden-state/logit probes should capture at least these token phases:

- `row_start`;
- `desc_end`;
- `box_start/pre_x1`;
- `post_x1/pre_y1`;
- `post_y1/pre_x2`;
- `post_x2/pre_y2`;
- `post_y2/row_boundary`;
- `next_row_or_stop_decision`.

For similarity analysis, compare the onset row against its duplicate source
row, the immediately previous row, pre-onset clean rows, same-image
nonduplicate rows, and cross-checkpoint counterpart rows for the same record.
This separates literal source-row copying from broader same-image basin
attraction or checkpoint-level calibration effects.

For coordinate logits, report coordinate-only `p_cond` top-k and basin mass for
each coordinate slot, plus full-vocabulary coordinate-token mass as leakage or
mode context. The coordinate-only distribution is the coordinate decision
surface; full-vocabulary mass says whether the model is in coordinate-token
mode at all.

Treat `<|coord_0|>` through `<|coord_999|>` as a special-token mechanism lane,
not merely ordinary vocabulary items. These coordinate tokens are newly added
or task-specific embeddings whose useful supervision comes primarily from the
COCO coordinate-token training surface. Existing locality evidence suggests CE
can preserve numeric neighborhood geometry, but degraded or irregular
smoothness remains mechanistically ambiguous rather than automatically good or
bad. The Phase 1 coordinate-token analysis should therefore track slot-specific
basin attraction at `x1`, `y1`, `x2`, and `y2`, especially whether a repeated
coordinate basin is already attractive before or at `x1`/`y1` for the
`primary_burst_onset_row`. Companion atlas views should include coordinate
token frequency by slot, input/output/effective coordinate-row norms, nearest
neighbors, locality, smoothness or discontinuity scores, and correlation
between duplicate-basin coordinates and token geometry or coverage.

Because strict schema/type loss trains coordinate positions to put probability
mass on coordinate tokens, full-vocabulary coordinate-token mass is a
sanity/regression check rather than a central uncertainty for this study. The
main coordinate-token evidence should be coordinate-internal and slot-specific:
training coverage and rollout duplicate-bin statistics separated for `x1`,
`y1`, `x2`, and `y2`; basin mass at `+/-4`, `+/-8`, and `+/-16` bins around the
component median coordinate and source-row coordinate; and static token geometry
or coverage interpreted only when it aligns with dynamic onset logits. Static
smoothness or discontinuity alone is diagnostic context, not a failure label.
Ordinary-vocab or other-special-token norm/bias comparisons are useful sanity
controls, but should not displace the coordinate-internal basin analysis.

Always include the object-start versus stop/EOS margin at row-boundary probes,
even when the burst looks coordinate-driven. This prevents conflating "bad
coordinate basin" with "the model should already have stopped."

Treat attention maps as routing evidence, not causal proof. Attention can
select candidate sinks, regions, layers, and heads, but causal attention or
visual-routing claims require image-region masking, perturbation, or another
controlled intervention that changes the continuation.

Phase 1 should emit hidden-state/logit and attention/routing artifacts under
separate output roots that consume the same `phase1_selected_windows.jsonl`
manifest from Phase 0. The selected-window manifest is the executable contract
for GPU probes and should include checkpoint label/path, rollout root, record
index, image id, prediction-row window, `primary_burst_onset_row`, component
ids, and token-trace handles when available.

The first hidden-state pass should sample layers `{0, 4, 8, 12, 16, 20, 24, 28,
final}` rather than every layer. Use cosine similarity on normalized hidden
states as the default row-state similarity metric; centered-cosine, PCA, or
other reductions can be post-hoc analyses after the first transition band is
known.

After hidden-state/logit and attention readouts, the first causal intervention
should be no-op replay parity plus image-region masking of the duplicate basin.
Coordinate-basin logit suppression should be second and only targeted after the
coordinate-logit probe shows repeated basin mass before or at `x1`/`y1`. Row
separator or newline interventions should be later template/boundary
diagnostics, because they can hide several mechanisms at once.

## Phase 0 Execution Boundary

The first implementation pass should produce only ledger artifacts and
candidate-review surfaces, with no GPU model-forward probes yet. Expected
outputs:

- `phase0_joint_onset_ledger.jsonl`;
- `phase0_summary.json`;
- `phase0_component_stats.json`;
- `phase0_component_stats.md`;
- `phase0_required_cases.md`;
- optional candidate panel links for manual false-basin review.

Write the first Phase 0 implementation output to an immutable timestamped root
under
`/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase0_joint_onset_ledger_<timestamp>/`.
A `latest` symlink is optional operator convenience and must not be the
canonical evidence handle.

Attention, hidden-state, logit, and intervention probes should wait until the
ledger definitions are validated against the required cases. Phase 1 should
consume exact selected onset windows from Phase 0.

## Open Questions

- Which causal intervention should run first after the ledger: coordinate-basin
  suppression, visual-region masking, row-boundary separator, or stop/continue
  novelty gating?
