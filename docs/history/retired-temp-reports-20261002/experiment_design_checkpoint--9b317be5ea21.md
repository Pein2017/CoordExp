# Boundary/Tail Direction Gate - Design Checkpoint

Status: Design grilling in progress.
Date: 2026-05-13.
Scope: Current compact-full A2/A3/A4 diagnosis design. This is a local checkpoint, not a concluded progress note.

## Purpose

Use the boundary/tail diagnosis as a training-direction gate.

The diagnosis should decide whether the next stage should primarily:

- suppress duplicate/invalid tail for stability;
- encourage valid extra-object emission under incomplete labels;
- build a gated hybrid that separates valid continuation from duplicate/invalid tail;
- prioritize coordinate/object-binding stability before adding more emission pressure;
- or recommend a new objective, data, decode, eval, or hybrid direction if evidence supports it.

The diagnosis is not constrained to existing named directions such as A5, LVIS-proxy, or Stage-2 rollout training. Any recommendation must be evidence-grounded and mapped to concrete repo owner surfaces.

## Primary Comparison

Use A2/A3/A4 together.

- A2: compact-full support+balance stability baseline.
- A3: prefix-rollin support+balance.
- A4: prefix-rollin plus EOS-trust weakening.

Primary decode surface:

- rp1.10 val200 compact-full artifacts.

Sensitivity only if needed:

- A3/A4 rp1.05 and rp1.15.

Do not use stale `.worktrees/...` paths as runnable truth. The live checkout is `/data/CoordExp` on `main`; current artifact truth lives under `output_remote/`, `temp/`, `progress/`, and `research/`.

## Artifact And Worktree Decisions

Preparation/design can remain in chat plus this local checkpoint.

If execution begins, use:

- worktree path: `/data/CoordExp/.worktrees/boundary-tail-direction-gate`
- branch: `codex/boundary-tail-direction-gate`
- base: `main`
- spec mode: none initially

Scratch/intermediate outputs:

- `/data/CoordExp/temp/boundary_tail_direction_gate_20260513/`

If the diagnosis becomes concluded evidence, promote a concise note to:

- `/data/CoordExp/progress/diagnostics/2026-05-13_boundary_tail_direction_gate.md`

Do not create OpenSpec unless the follow-up changes a stable contract.

## First-Pass Tables

Use three linked tables:

1. `image_run_summary`
2. `object_delta_table`
3. `manual_audit_queue`

### image_run_summary

One row per `image_id x run_label`.

Recommended fields:

- `image_id`
- `run_label`
- `gt_count`
- `pred_count_raw`
- `pred_count_scored`
- `pred_count_guarded`
- `invalid_count`
- `duplicate_suppressed_count`
- `matched_count_iou50`
- `unmatched_count_iou50`
- `f1ish_tp50`
- `f1ish_fp50`
- `f1ish_fn50`
- `local_density_bucket`
- `has_high_density_gt`
- `has_duplicate_burst`
- `has_invalid_or_border_tail`
- `has_a4_extra_vs_a2`
- `has_a4_extra_vs_a3`
- `artifact_root`

### object_delta_table

One row per predicted object, with cross-run/match/duplicate/confidence features.

Recommended fields:

- `image_id`
- `run_label`
- `pred_index`
- `generation_order`
- `desc`
- `bbox_xyxy_norm1000`
- `bbox_area`
- `bbox_center`
- `valid_geometry`
- `border_or_top_left_flag`
- `score`
- `coord_confidence_mean`
- `coord_confidence_min`
- `coord_confidence_slot_x1`
- `coord_confidence_slot_y1`
- `coord_confidence_slot_x2`
- `coord_confidence_slot_y2`
- `matched_gt_iou`
- `matched_gt_index`
- `matched_gt_desc`
- `matched_proxy_or_lvis_if_available`
- `duplicate_guard_suppressed`
- `duplicate_cluster_id`
- `nearest_same_desc_pred_iou`
- `nearest_any_desc_pred_iou`
- `nearest_gt_iou`
- `nearest_gt_desc`
- `is_extra_vs_A2`
- `is_extra_vs_A3`
- `cross_run_nearest_A2_iou`
- `cross_run_nearest_A3_iou`
- `cross_run_nearest_A4_iou`
- `cross_run_ambiguous_match`
- `auto_label`
- `layer_tag`
- `manual_audit_needed`
- `manual_audit_reason`

Auto-label values:

- `positive_gt_match`
- `positive_proxy_match`
- `neutral_plausible_proxy_match`
- `neutral_plausible_unmatched_candidate`
- `bad_duplicate_like`
- `bad_invalid_geometry`
- `bad_border_or_top_left`
- `bad_wrong_location_candidate`
- `bad_malformed`
- `ambiguous_dense_instance`
- `unknown`

Layer tags:

- `training_objective`
- `decode_inference`
- `eval_artifact`
- `data_ambiguity_unlabeled`
- `unknown_needs_manual`

### manual_audit_queue

Only include ambiguous, decision-changing cases.

Recommended fields:

- `case_id`
- `image_id`
- `run_label`
- `pred_index`
- `desc`
- `bbox_xyxy_norm1000`
- `auto_label`
- `why_auto_label_is_uncertain`
- `decision_impact`
- `overlay_path_with_gt`
- `overlay_path_no_gt`
- `crop_path`
- `neighbor_context_path_if_available`
- `question_for_user`
- `recommended_manual_options`
- `user_label`
- `user_notes`

Manual options:

- `real_distinct_object`
- `duplicate_same_instance`
- `duplicate_but_distinct_dense_instance_possible`
- `wrong_location`
- `wrong_category_or_desc`
- `invalid_or_artifact`
- `cannot_tell`

## Cross-Run Extra Definition

Define extras by one-to-one cross-run object matching, not raw index or count.

Strict same-desc match:

- same normalized desc/category;
- IoU >= 0.50.

Loose geometry match:

- any desc;
- IoU >= 0.70.

`extra_vs_run` means:

- no strict same-desc match;
- and no loose geometry match.

Ambiguous cross-run match:

- same desc and 0.30 <= IoU < 0.50;
- or any desc and 0.50 <= IoU < 0.70.

## Duplicate-Tail Labeling

Use automatic detection first, with human review only when ambiguity affects the decision.

Automatic strong duplicate examples:

- same-desc nearest prediction IoU >= 0.80 with close centers;
- same-desc nearest prediction IoU >= 0.70 plus low coord confidence or late-tail generation;
- duplicate-guard suppressed plus same-desc IoU >= 0.70 and no separate GT/proxy evidence nearby;
- same-desc local burst cluster size >= 3 with tight boxes/centers and at most one GT/proxy match.

Protect dense-instance ambiguity:

- duplicate guard or same-desc IoU suggests duplication;
- but scene/class is dense;
- multiple visually separate instances may exist;
- geometry is valid;
- coord confidence is not obviously bad.

Use duplicate guard as diagnostic evidence, not label truth.

## LVIS-Proxy / Unlabeled Handling

Use LVIS-proxy as supporting evidence, not unquestioned ground truth.

- COCO unmatched plus strict proxy match => `positive_proxy_match`.
- COCO unmatched plus plausible proxy match => neutral / weak-positive.
- COCO unmatched plus no proxy => not automatically bad.
- Proxy mismatch => not automatically bad.

Relevant sources:

- `/data/CoordExp/public_data/coco/rescale_32_1024_bbox_max60_lvis_proxy/val.coord.jsonl`
- `/data/CoordExp/public_data/coco/rescale_32_1024_bbox_max60_lvis_proxy/val.proxy_summary.json`
- `/data/CoordExp/manifests/public_data_provenance/coco/rescale_32_1024_bbox_max60_lvis_proxy.json`

## Generation Order / Tailness

Treat generation order as a separation feature, not an automatic label.

Boundary definitions:

- EOS boundary: where the model stops;
- COCO coverage boundary: first generated index after the last COCO-GT-matched prediction;
- coverage-aware boundary: first generated index after the last COCO-or-proxy/objectness-supported prediction;
- risk boundary: first generated index after the first invalid geometry, border collapse, severe duplicate, or low-confidence burst marker;
- cross-run boundary: A3/A4 objects without strict/loose counterpart in A2.

Tail features:

- `generation_order`
- `normalized_generation_order`
- `objects_after_last_gt_match`
- `after_coco_boundary`
- `after_coverage_boundary`
- `after_risk_boundary`
- `a3_extra_vs_a2`
- `a4_extra_vs_a3`
- `a4_extra_vs_a2`
- `objects_after_last_good`
- `objects_after_first_duplicate_flag`
- `objects_after_first_invalid_flag`
- `run_pred_count_minus_gt_count`
- `is_after_model_started_repeating_desc`
- `is_after_low_confidence_streak`
- `low_confidence_streak_position`
- `burst_cluster_position`

Late alone is not bad.

Late plus low coord confidence plus duplicate-like is stronger danger-tail evidence.

Late plus valid geometry plus high coord confidence plus proxy/visual plausibility can still be valid extra.

## Coord Confidence

Use coord confidence as a ranking, bucket, and separability feature, not hard label truth.

Initial provisional buckets:

- high confidence: mean logprob >= -3.2
- mid confidence: -3.8 <= mean logprob < -3.2
- low confidence: mean logprob < -3.8
- missing confidence: trace unavailable or alignment failed

These thresholds are only exploratory. The diagnosis should look for statistical thresholds for valid objects and duplication bursts if the data support them.

## Prior Threshold Evidence

Treat existing `progress/` threshold studies as prior baselines, not as final labels:

- `progress/diagnostics/2026-03-11_gt_overlap_threshold_search.md`
  - global hard duplicate safety requires approximately `IoU >= 0.999`;
  - `IoU >= 0.99` is a high-precision severe bucket but not collision-free;
  - `0.95` and `0.97` are unsafe as hard semantics-agnostic deletion thresholds.
- `progress/diagnostics/2026-03-11_rollout_duplication_thresholds_ul_vs_ulv2.md`
  - `IoU >= 0.99` captures only the severe tip;
  - much suspicious rollout duplication mass starts around `0.90-0.95`.
- `progress/diagnostics/2026-05-08_compact_full_coord_confidence_stop_gate_diagnostics.md`
  - `coord_mean_logprob < -3.4` is a plausible low-confidence tail marker;
  - coord confidence separates duplicate/invalid tail better than it separates matched true positives from plausible unlabeled objects.
- `progress/diagnostics/2026-04-11_stage1_coord_basin_duplication_mechanism.md`
  - duplication is broader than literal bbox copying;
  - important coordinate-basin signals include `x1/y1` previous-neighborhood mass, entropy, and predicted-object-vs-exact-duplicate score margin.

Add prior-band columns to the first-pass object table:

- `overlap_prior_safe_hard_iou_0999`
- `overlap_prior_practical_severe_iou_099`
- `overlap_prior_soft_band_iou_095_099`
- `overlap_prior_broad_suspicious_iou_090_095`
- `coord_prior_low_conf_lt_neg3p4`
- `coord_prior_matched_p10_lt_neg3p364`
- `coord_prior_strict_duplicate_opt_lt_neg2p932`

Use these columns to stratify evidence. They must not automatically decide whether an unmatched object is invalid, duplicate, or a plausible unlabeled positive.

Key questions:

- Are duplicate/invalid tails concentrated in low confidence?
- Are positive/neutral extras concentrated in high/mid confidence?
- Does A4 add high-confidence valid extras before low-confidence tail begins?
- Do x1/y1 confidence or entropy better separate coordinate-basin failures than bbox-average confidence?

## Decision Criteria

Use directional thresholds, not rigid percentages.

Duplicate-stability first:

- A4/A3 extra emissions are mostly danger-tail;
- or valid-extra and danger-tail are not separable by simple signals.

Valid-emission first:

- A4/A3 extra emissions are mostly positive or neutral/weak-positive;
- and duplicate/invalid tail is small or filterable.

Gated hybrid:

- both valid-extra and danger-tail are substantial;
- but they separate by coord confidence, duplicate IoU, geometry validity, local density, generation order, desc/category consistency, boundary margin, or proxy/LVIS evidence.

Coordinate/object-binding stability first:

- both are substantial and not separable.

Allowed final diagnosis outcomes:

- duplicate-stability first;
- valid-emission first;
- gated hybrid objective;
- data-layer expansion first;
- prefix-state / rollout-objective first;
- coordinate/object-binding stability first;
- new objective or gating direction if the evidence supports it.

Priority:

- prioritize training / data / objective directions;
- do not make benchmarking the center of the work;
- use decode/eval knobs as containment evidence or diagnostics, not as the primary research destination unless evidence overwhelmingly shows the training distribution is already healthy.

The diagnosis should separate:

- recommended next research direction;
- recommended next engineering implementation;
- recommended next smoke/validation artifact;
- what evidence would falsify the recommendation.

## A2 / A3 / A4 Interpretation

Treat A2 as the stability anchor, not necessarily the final winner.

- A2: stability/control surface.
- A3: prefix-rollin mechanism surface.
- A4: weakened-EOS recall/risk surface.

Cross-run interpretation:

- if A2 emits an object and A3/A4 drop it, check whether prefix-rollin hurts stable valid emission;
- if A2 stops and A3/A4 emit, classify whether the extra is valid, neutral-plausible, or dirty tail;
- if A4 emits many objects beyond A3, classify whether EOS weakening unlocks valid extras or opens collapse;
- if A3 is cleaner but lower recall, treat it as the conservative prefix-rollin reference;
- if A2 is both cleaner and higher metric, do not automatically conclude rollback; ask whether A3/A4 revealed a useful mechanism that needs gating.

## LVIS / Proxy Evidence Policy

Use LVIS/proxy evidence as a label-coverage, objectness, and semantic-neighborhood witness, not as final exact-category truth.

For each COCO-unmatched predicted object:

- COCO matched -> `positive_coco`;
- proxy strict match -> `positive_proxy`;
- proxy loose match -> `neutral_proxy_plausible`;
- no proxy match -> `unresolved_unmatched`, not automatically bad;
- proxy conflict -> `ambiguous_label_space`, not automatically bad.

Separate two questions:

- is there probably a real object there?
- is the generated description/category exactly the desired training label?

Objectness/semantic support levels:

- strict LVIS/COCO semantic mapping plus geometry overlap -> strong objectness plus semantic support;
- plausible LVIS mapping or related semantic category plus geometry overlap -> objectness support and soft semantic support;
- LVIS object exists nearby but category mapping is imperfect -> objectness support and semantic uncertainty;
- no LVIS match -> no proxy support, but not automatically false;
- dense LVIS region with sibling/overlapping categories -> protected ambiguous objectness region.

Training implication:

- LVIS can protect valid object emission from being mislabeled as FP;
- LVIS can justify neutral-positive / weak-positive handling even when exact COCO category matching fails;
- LVIS alone should not authorize strong desc CE on noisy mappings;
- objectness-supported but semantic-uncertain cases may deserve coordinate/objectness encouragement, weak desc weight, neutral FP handling, or no-punish continuation instead of full hard-positive supervision.

## Candidate Objective Space

Graded objectness / semantics:

- strong positive:
  - COCO GT or strict LVIS/COCO proxy;
  - full desc CE plus full coord CE/loss.
- weak / objectness-positive:
  - LVIS/proxy/objectness-supported but semantic uncertain;
  - coordinate/objectness continuation support;
  - reduced desc CE weight;
  - no strong category penalty.
- neutral plausible:
  - unmatched but high-confidence, visually plausible, or dense-region plausible;
  - do not punish as FP;
  - optionally use as continuation-neutral support.
- danger negative:
  - duplicate burst, invalid geometry, border collapse, or local-basin repeat;
  - suppress, downweight, unlikelihood, or reject from positive support.

Protected duplicate-burst suppression:

- treat duplicate bursts as structured local-basin failures, not merely too many predictions;
- hard unlikelihood only for severe safe duplicates, such as exact duplicates, `IoU >= 0.999`, malformed/border-collapse repeats, or repeated same-object local basins with no objectness support;
- soft duplicate-risk penalty for broader bands such as `0.95-0.99` or `0.90-0.95` only when combined with low confidence, repeated desc, local burst, or generated-prefix contamination;
- use burst-aware cluster-level signals rather than isolated pairwise overlap alone;
- protect dense instances when GT/proxy/manual evidence supports separate real objects;
- consider coordinate-basin escape objectives using `x1/y1` entropy or previous-neighborhood mass as diagnostic/target signals.

Avoid the binary assumption `unmatched == false positive`.

The final diagnosis should include lightweight objective sketches, not a full implementation plan. Each sketch should include:

- name;
- mechanism it targets;
- positive signal;
- negative/protection signal;
- expected benefit;
- main risk;
- minimal smoke to test it.

Candidate sketches may include:

- objectness-protected continuation;
- protected duplicate-burst unlikelihood;
- boundary-confidence gated EOS/continue;
- repaired/generated-prefix rollout training.

The report should still end with one recommended next action, not only a menu.

## Evidence Hierarchy

Use this precedence when signals conflict:

1. artifact-level A2/A3/A4 behavior from current predictions, confidence traces, duplicate reports, and per-image summaries;
2. manual review of decision-changing ambiguous cases;
3. LVIS/proxy objectness evidence;
4. prior threshold studies;
5. existing May interpretation docs;
6. leaderboard metrics;
7. older design notes.

Conflict rules:

- prefer actual artifact behavior over old interpretation;
- prefer objectness-protected interpretation over `COCO-unmatched == false`;
- prefer protected duplicate labeling over broad hard suppression in dense scenes;
- prefer a small follow-up probe over a confident training recommendation if the fork is unresolved.

Report both views:

- COCO-strict view for metric comparability;
- coverage-aware view for the research decision about common unlabeled objects.

If these disagree, avoid blunt duplicate/FP suppression. Prefer a gated continuation or protected-plausible-extra handling direction.

## Mechanism Taxonomy

The first-pass diagnosis should distinguish:

- conservative stop;
- dirty continuation;
- mixed separable tail;
- generated-prefix damage.

Per-image mechanism labels:

- `stable_complete`
- `conservative_stop`
- `valid_extra_unlocked`
- `dirty_tail_unlocked`
- `mixed_separable`
- `prefix_state_damaged`
- `unclear_needs_review`

Priority:

- first decide dirty continuation vs valid-extra-unlocked;
- then decide whether generated-prefix damage explains the split.

## Minimum Artifact Bundle

Required before choosing the next experiment:

- concise diagnosis report, promoted to `progress/diagnostics/2026-05-13_boundary_tail_direction_gate.md` only after analysis is actually run;
- machine-readable tables under `temp/boundary_tail_direction_gate_20260513/`:
  - `image_run_summary.jsonl`;
  - `object_delta_table.jsonl`;
  - `manual_audit_queue.jsonl`;
  - optional `threshold_prior_transfer_summary.json`;
- small visual/manual review packet only for ambiguous decision-changing cases;
- one-page gate summary with A2/A3/A4 counts, prior-threshold transfer, and directional recommendation.

No expensive training is required for the minimum diagnosis. GPU work is only an escalation path.

Execution order:

- build offline tables first;
- rank ambiguous decision-changing cases;
- generate visuals/crops only for ranked unresolved cases;
- ask the user for manual audit only if the selected cases can change the gate.

First table pass:

- read A2/A3/A4 artifacts;
- join predictions, scores, confidence, and guarded duplicate info;
- cross-run match objects;
- attach COCO/proxy/LVIS objectness support;
- compute prior threshold bands;
- compute boundary/tail features;
- emit tables and summaries;
- rank manual-audit candidates.

LVIS/proxy source order:

1. primary: `/data/CoordExp/public_data/coco/rescale_32_1024_bbox_max60_lvis_proxy/val.coord.jsonl`;
2. supporting: `/data/CoordExp/public_data/coco/rescale_32_1024_bbox_max60_lvis_proxy/val.proxy_summary.json`;
3. supporting provenance: `/data/CoordExp/manifests/public_data_provenance/coco/rescale_32_1024_bbox_max60_lvis_proxy.json`;
4. fallback: exporter outputs/projection artifacts if per-object strict/plausible labels are missing;
5. final fallback: raw LVIS annotations/projection directories.

Reason: the diagnosis should reflect the actual dataset surface a next training stage would use.

## Post-Diagnosis Escalation

Default to the smallest credible training smoke, not immediate production training, unless evidence is unusually clean.

Execution levels:

- probe first, if objective failure vs prefix-state damage is unresolved;
- tiny/small training smoke, the default when a mechanism is clear enough to test;
- production candidate only if artifact evidence is strong, manual review does not overturn it, LVIS/proxy/data path is ready, risks are protected, and code/config risk is low.

Each recommended direction must include a falsifier:

- objectness-protected continuation: abandon/revise if LVIS/proxy/manual review shows most extras are not real objects, or weak-positive training increases dirty tail without improving valid extras;
- protected duplicate-burst suppression: abandon/revise if many duplicate-risk objects are visually/proxy-supported distinct instances, or suppression damages valid dense-object recall;
- boundary-confidence gated EOS/continue: abandon/revise if coord confidence does not transfer, rejects too many valid/proxy-supported objects, or misses high-confidence dirty duplicates;
- repaired/generated-prefix rollout training: abandon/revise if repaired prefixes do not restore healthy next-object distributions, or generated-prefix failures are not materially different from GT-prefix failures;
- LVIS-enhanced/data expansion: abandon/revise if mapping/objectness noise is too high, schema incompatibility remains unresolved, or proxy additions mostly duplicate existing labels without addressing missing objects;
- coordinate/object-binding objective: abandon/revise if duplicate bursts do not show local-basin/x1-y1 stickiness, or object-binding signals do not separate failures from valid dense emissions.

## Script Policy

Reuse canonical existing code for:

- current eval record parsing;
- bbox IoU / geometry helpers;
- confidence trace loading/alignment;
- duplicate-control logic as a reference signal;
- existing visualization utilities for selected manual cases.

If existing probes do not line up cleanly with the A2/A3/A4 cross-run object-delta structure, create a narrow dedicated analysis harness later, likely:

- `scripts/analysis/diagnose_boundary_tail_direction_gate.py`
- `configs/analysis/boundary_tail_direction_gate/a2_a3_a4_val200.yaml`

The harness should read artifact roots, emit tables into `temp/`, and avoid new training behavior, config-system complexity, evaluator contracts, or broad CLI expansion.

## Manual Audit Triggers

Ask the user to inspect cases only when:

- automatic labels disagree across signals;
- object is unmatched but visually plausible;
- duplicate guard may be over-suppressing distinct dense instances;
- top-left/border collapse heuristic is unsure;
- the decision gate depends on whether neutral/weak-positive count is real.

Do not ask for obvious invalid geometry, exact duplicate clusters, malformed rows, or clean GT matches.

Manual review should be capped initially at 16-32 ambiguous objects/images and expand to 64 only if the diagnosis remains unstable.

Manual labels:

- `valid_extra`
- `duplicate_or_same_object`
- `invalid_or_bad_geometry`
- `unclear_dense`
- `unclear_label_space`

Existing A3/A4 manual review artifacts under `temp/a4_rp110_tf_probe_and_manual_review_20260512` may be used as supporting context, but they are incomplete and less informative than the new object-table-driven audit. Use v2 pixel-GT/no-GT overlays if reusing old material. Avoid the older buggy GT-green overlay.

The new diagnosis owns manual-case selection. Generate a fresh small review packet when existing visuals do not cover decision-changing cases.

## GPU Availability

The user reports that all 8 GPUs are available.

Design implication:

- start with simplest artifact-led analysis;
- keep the possibility of complex counterfactual generation or controlled decode experiments;
- escalate dynamically only when artifact-led tables leave decision-critical gaps.
