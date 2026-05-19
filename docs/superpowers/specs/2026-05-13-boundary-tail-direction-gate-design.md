# Boundary/Tail Direction Gate Design

Date: 2026-05-13

Status: Ready for Execution: pending explicit execution start.

Owner surface: offline analysis and research diagnosis around current compact-full A2/A3/A4 artifacts.

Primary source checkpoint: `/data/CoordExp/temp/boundary_tail_direction_gate_20260513/experiment_design_checkpoint.md`.

## Purpose

Build a table-first diagnosis that decides which training, data, or objective direction should drive the next stage of the autoregressive V-LLM detection work.

This is not a benchmark campaign and not a decode-only tuning exercise. Metrics remain useful as sanity evidence, but the center of the work is mechanism recovery:

- whether duplicate/invalid tail must be suppressed before adding more emission pressure;
- whether valid object emission is being under-trained because COCO labels miss common objects;
- whether A3/A4 reveal a useful continuation mechanism that needs gating;
- whether generated-prefix state damage explains the boundary/tail split;
- whether LVIS/proxy objectness evidence justifies objectness-positive but semantic-soft training;
- whether a new objective direction is required.

The diagnosis must end with one recommended next action and a falsifier for that recommendation.

## Non-Goals

- Do not launch expensive training during the diagnosis.
- Do not make AP/F1 leaderboard movement the primary decision rule.
- Do not create an OpenSpec change unless a later implementation changes stable contracts.
- Do not write or edit `.codex/memories/`.
- Do not treat COCO-unmatched objects as false positives by default.
- Do not treat duplicate guard output as ground-truth duplicate labels.
- Do not treat LVIS/proxy semantic mappings as exact truth when the mapping is only plausible.
- Do not use stale `.worktrees/...` paths as runnable truth.

## Current Runs

Use A2/A3/A4 together on the current `rp=1.10` val200 compact-full decode surface:

| Run | Role | Artifact root |
|---|---|---|
| A2 | stability/control surface | `/data/CoordExp/output_remote/infer/recursive_detection_ce_latest/compact_full_support2_tokenrows_v2_ckpt3664_val200_bsz8_temp0_rep1p10_prompt_offset_fix_8gpu` |
| A3 | prefix-rollin mechanism surface | `/data/CoordExp/output_remote/infer/recursive_detection_ce_latest/compact_full_prefix_rollin_balance2_a3_ckpt3664_val200_bsz8_temp0_rp1p10_max3084_chatfix_4gpu` |
| A4 | weakened-EOS recall/risk surface | `/data/CoordExp/output_remote/infer/recursive_detection_ce_latest/compact_full_prefix_rollin_balance2_a4_eos_ckpt3664_val200_bsz8_temp0_rp1p10_max3084_chatfix_4gpu` |

Required per-run inputs:

- `gt_vs_pred_scored.jsonl`;
- `gt_vs_pred_scored_guarded.jsonl`;
- `pred_confidence.jsonl`;
- `pred_token_trace.jsonl` when token-level confidence or coordinate-basin features are available;
- `eval/metrics.json`;
- `eval/metrics_guarded.json` when available;
- `eval/duplicate_guard_report.json`;
- `eval/per_image.json`.
- `resolved_config.json`.

The diagnosis must record a per-run preflight table from `resolved_config.json` before interpreting cross-run differences. At minimum it records checkpoint/model path, whether that path is stale or exists, `limit`, `batch_size`, `temperature`, `repetition_penalty`, `max_new_tokens`, prompt-offset/fix hints if present, and artifact root. A2 is currently a stability anchor with `max_new_tokens=1024`, while A3/A4 use `max_new_tokens=3084`; reports must not hide this cap difference when comparing boundary/tail behavior.

Sensitivity surfaces such as A3/A4 `rp=1.05` or `rp=1.15` are follow-up only if the primary surface leaves a decision-critical ambiguity.

## Evidence Hierarchy

When evidence conflicts, use this order:

1. Current artifact behavior from A2/A3/A4 predictions, scores, confidence traces, duplicate reports, and per-image summaries.
2. Manual review of decision-changing ambiguous cases.
3. LVIS/proxy objectness evidence.
4. Prior threshold studies.
5. Existing May 2026 interpretation docs.
6. Leaderboard metrics.
7. Older design notes.

Conflict rules:

- prefer actual artifact behavior over older interpretation;
- prefer objectness-protected interpretation over `COCO-unmatched == false`;
- prefer protected duplicate labeling over broad hard suppression in dense scenes;
- prefer a small follow-up probe over a confident training recommendation when the mechanism fork remains unresolved.

## LVIS / Proxy Objectness Policy

Use LVIS/proxy evidence as a label-coverage, objectness, and semantic-neighborhood witness, not as final exact-category truth.

Primary source order:

1. `/data/CoordExp/public_data/coco/rescale_32_1024_bbox_max60_lvis_proxy/val.coord.jsonl`
2. `/data/CoordExp/public_data/coco/rescale_32_1024_bbox_max60_lvis_proxy/val.proxy_summary.json`
3. `/data/CoordExp/manifests/public_data_provenance/coco/rescale_32_1024_bbox_max60_lvis_proxy.json`
4. exporter outputs or projection artifacts if per-object strict/plausible metadata is missing
5. raw LVIS annotations or projection directories only as final fallback

The materialized validation JSONL is primary because it is the dataset surface a next training stage would actually use.

Separate two questions for each COCO-unmatched prediction:

- is there probably a real object there?
- is the generated description/category exactly the desired training label?

Objectness and semantic support levels:

| Evidence | Interpretation |
|---|---|
| COCO GT match | full positive |
| strict LVIS/COCO semantic mapping plus geometry overlap | strong objectness and semantic support |
| plausible LVIS mapping or related semantic category plus geometry overlap | objectness support and soft semantic support |
| LVIS object exists nearby but category mapping is imperfect | objectness support with semantic uncertainty |
| no LVIS/proxy match | no proxy support, not automatic false |
| dense LVIS region with sibling/overlapping categories | protected ambiguous objectness region |

Training implications the diagnosis may recommend:

- full desc and coord supervision for COCO GT or strict proxy positives;
- reduced desc weight with coordinate/objectness support for semantic-uncertain objectness positives;
- neutral or no-punish handling for high-confidence plausible unmatched objects;
- explicit suppression only for danger negatives such as duplicate bursts, invalid geometry, border collapse, or local-basin repeats.

Unsupported unmatched predictions default to `unknown`, not positive and not false-positive. They become `neutral_plausible_unmatched_candidate` only when an explicit plausibility signal exists, such as high coordinate confidence, valid non-border geometry, low duplicate/burst risk, proxy-neighborhood/objectness support, or manual evidence.

## Prior Threshold Baselines

The diagnosis must replay existing threshold studies as named prior bands, not rediscover them from scratch.

Reference docs:

- `progress/diagnostics/2026-03-11_gt_overlap_threshold_search.md`
- `progress/diagnostics/2026-03-11_rollout_duplication_thresholds_ul_vs_ulv2.md`
- `progress/diagnostics/2026-05-08_compact_full_coord_confidence_stop_gate_diagnostics.md`
- `progress/diagnostics/2026-04-11_stage1_coord_basin_duplication_mechanism.md`

Prior bands:

| Band | Meaning |
|---|---|
| `overlap_prior_safe_hard_iou_0999` | semantics-agnostic severe duplicate reference with near-zero GT collision on studied val GT |
| `overlap_prior_practical_severe_iou_099` | high-precision severe duplicate bucket, not collision-free |
| `overlap_prior_soft_band_iou_095_099` | soft duplicate-risk band, unsafe as a hard deletion rule |
| `overlap_prior_broad_suspicious_iou_090_095` | broad duplicate-risk band where much rollout pathology can live |
| `coord_prior_low_conf_lt_neg3p4` | plausible low-confidence tail marker from the earlier diagnostic family |
| `coord_prior_prefixed_matched_p10_lt_neg3p364` | pre-fix / prompt-offset-contaminated matched-p10 reference; contextual only |
| `coord_prior_fixed_matched_p10_lt_neg3p209` | fixed-artifact matched-p10 reference for current A2/A3/A4 transfer when available |
| `coord_prior_strict_duplicate_opt_lt_neg2p932` | strict duplicate-oriented confidence reference, likely too aggressive for valid emission |

These bands stratify evidence. They do not automatically decide labels.

Threshold summaries must record which artifact family each cutoff came from. Pre-fix confidence cutoffs can be carried as contextual priors, but current A2/A3/A4 recommendations should prefer fixed-artifact transfer bands when those are available.

## Boundary And Tail Definitions

Use multiple definitions in parallel:

- EOS boundary: where the model stops.
- COCO coverage boundary: first generated index after the last COCO-GT-matched prediction.
- Coverage-aware boundary: first generated index after the last COCO-or-proxy/objectness-supported prediction.
- Risk boundary: first generated index after the first invalid geometry, border collapse, severe duplicate, or low-confidence burst marker.
- Cross-run boundary: A3/A4 objects without strict or loose counterpart in A2.

Object-level tail features:

- `generation_order`;
- `normalized_generation_order`;
- `after_coco_boundary`;
- `after_coverage_boundary`;
- `after_risk_boundary`;
- `a3_extra_vs_a2`;
- `a4_extra_vs_a3`;
- `a4_extra_vs_a2`;
- `objects_after_last_gt_match`;
- `objects_after_last_good`;
- `objects_after_first_duplicate_flag`;
- `objects_after_first_invalid_flag`;
- `is_after_model_started_repeating_desc`;
- `is_after_low_confidence_streak`;
- `low_confidence_streak_position`;
- `burst_cluster_position`;
- `run_pred_count_minus_gt_count`.

Late alone is not bad. Late plus low confidence and duplicate-like geometry is danger-tail evidence. Late plus valid geometry, high confidence, and proxy/manual plausibility can still be valid extra evidence.

## Coordinate-Space Contract

Inference artifacts can be pixel-space even when proxy JSONL objects are coord-token/norm1000. The diagnosis must not assume one coordinate surface.

For every predicted, GT, and proxy object:

- preserve source coordinates and source coordinate mode;
- emit `bbox_xyxy_pixel` when pixel coordinates are available or can be derived;
- emit `bbox_xyxy_norm1000` when norm1000 coordinates are available or can be derived;
- record image `width` and `height`;
- compute IoU, cross-run matching, duplicate-risk, and proxy/GT matching only after converting compared boxes to the same coordinate space;
- prefer pixel-space matching among inference GT/pred rows that share `width` and `height`;
- prefer norm1000 matching when comparing against LVIS-proxy coord-token JSONL, unless proxy objects are converted to pixel using the same `width` and `height`.

Valid-geometry checks must be coordinate-mode aware. A pixel coordinate above `999` is not invalid when `coord_mode: pixel`; it must be normalized before any norm1000 validity rule is applied.

## Cross-Run Matching

Define extra objects by one-to-one cross-run matching, not raw index or count.

Strict same-desc match:

- same normalized desc/category;
- IoU `>= 0.50`.

Loose geometry match:

- any desc;
- IoU `>= 0.70`.

`extra_vs_run` means no strict same-desc match and no loose geometry match.

Ambiguous cross-run match:

- same desc and `0.30 <= IoU < 0.50`;
- or any desc and `0.50 <= IoU < 0.70`.

## Duplicate-Tail Policy

Treat duplicate bursts as structured local-basin failures, not merely too many predictions.

Strong automatic duplicate evidence:

- same-desc nearest prediction IoU `>= 0.80` with close centers;
- same-desc nearest prediction IoU `>= 0.70` plus low coordinate confidence or late-tail generation;
- duplicate-guard suppressed plus same-desc IoU `>= 0.70` and no separate GT/proxy evidence nearby;
- same-desc local burst cluster size `>= 3` with tight boxes/centers and at most one GT/proxy match.

Protected dense-instance ambiguity:

- duplicate guard or same-desc IoU suggests duplication;
- scene/class is dense;
- multiple visually separate instances may exist;
- geometry is valid;
- coordinate confidence is not obviously bad;
- GT/proxy/manual evidence supports separate real objects or objectness uncertainty.

The diagnosis may recommend:

- hard unlikelihood only for severe safe duplicates, exact duplicates, `IoU >= 0.999`, malformed/border-collapse repeats, or repeated same-object local basins with no objectness support;
- soft duplicate-risk penalties for broader bands only when combined with low confidence, repeated desc, burst cluster, or generated-prefix contamination;
- objectness-protected neutral handling where overlap may reflect real dense instances;
- coordinate-basin escape objectives if `x1/y1` previous-neighborhood mass or entropy separates failure from valid dense emission.

## Output Tables

Write first-pass outputs under `/data/CoordExp/temp/boundary_tail_direction_gate_20260513/`.

### `image_run_summary.jsonl`

One row per `image_id x run_label`.

Required fields:

- `image_id`;
- `image`;
- `run_label`;
- `gt_count`;
- `pred_count_raw`;
- `pred_count_scored`;
- `pred_count_guarded`;
- `invalid_count`;
- `duplicate_suppressed_count`;
- `matched_count_iou50`;
- `unmatched_count_iou50`;
- `f1ish_tp50`;
- `f1ish_fp50`;
- `f1ish_fn50`;
- `local_density_bucket`;
- `has_high_density_gt`;
- `has_duplicate_burst`;
- `has_invalid_or_border_tail`;
- `has_a4_extra_vs_a2`;
- `has_a4_extra_vs_a3`;
- `mechanism_label`;
- `artifact_root`.

### `object_delta_table.jsonl`

One row per predicted object.

Required fields:

- identity: `image_id`, `image`, `run_label`, `pred_index`, `generation_order`;
- object: `desc`, `coord_mode_source`, `bbox_xyxy_pixel`, `bbox_xyxy_norm1000`, `bbox_area`, `bbox_center`, `image_width`, `image_height`;
- geometry: `valid_geometry`, `border_or_top_left_flag`;
- score/confidence: `score`, `coord_confidence_mean`, `coord_confidence_min`, slot confidence fields when available;
- GT/proxy: `matched_gt_iou`, `matched_gt_index`, `matched_gt_desc`, `nearest_gt_iou`, `nearest_gt_desc`, `proxy_support_level`, `proxy_support_tier`, `proxy_support_desc`, `proxy_source`, `mapping_class`, `mapping_kind`, `desc_ce_weight`, `coord_weight`, `objectness_support_level`, `semantic_support_level`;
- duplicate: `duplicate_guard_suppressed`, `duplicate_cluster_id`, `nearest_same_desc_pred_iou`, `nearest_any_desc_pred_iou`, `burst_cluster_size`;
- cross-run: `is_extra_vs_A2`, `is_extra_vs_A3`, `cross_run_nearest_A2_iou`, `cross_run_nearest_A3_iou`, `cross_run_nearest_A4_iou`, `cross_run_ambiguous_match`;
- prior bands: the seven prior threshold columns listed above;
- boundary/tail: the boundary and tail feature columns listed above;
- labels: `auto_label`, `layer_tag`, `mechanism_label`, `manual_audit_needed`, `manual_audit_reason`.

Allowed `auto_label` values:

- `positive_gt_match`;
- `positive_proxy_match`;
- `neutral_plausible_proxy_match`;
- `neutral_plausible_unmatched_candidate`;
- `bad_duplicate_like`;
- `bad_invalid_geometry`;
- `bad_border_or_top_left`;
- `bad_wrong_location_candidate`;
- `bad_malformed`;
- `ambiguous_dense_instance`;
- `unknown`.

Allowed `layer_tag` values:

- `training_objective`;
- `decode_inference`;
- `eval_artifact`;
- `data_ambiguity_unlabeled`;
- `unknown_needs_manual`.

### `manual_audit_queue.jsonl`

Only include ambiguous, decision-changing cases.

Required fields:

- `case_id`;
- `source_gt_vs_pred_jsonl`;
- `line_idx`;
- `case_uid`;
- `object_index`;
- `image_id`;
- `image`;
- `run_label`;
- `pred_index`;
- `desc`;
- `bbox_xyxy_norm1000`;
- `auto_label`;
- `why_auto_label_is_uncertain`;
- `decision_impact`;
- `overlay_path_with_gt`;
- `overlay_path_no_gt`;
- `crop_path`;
- `neighbor_context_path_if_available`;
- `question_for_user`;
- `recommended_manual_options`;
- `user_label`;
- `user_notes`.

Manual labels:

- `real_distinct_object`;
- `duplicate_same_instance`;
- `duplicate_but_distinct_dense_instance_possible`;
- `wrong_location`;
- `wrong_category_or_desc`;
- `invalid_or_artifact`;
- `cannot_tell`.

Final reporting may map these detailed labels into coarser categories such as `valid_extra`, `duplicate_or_same_object`, `invalid_or_bad_geometry`, `unclear_dense`, and `unclear_label_space`, but the audit queue should preserve the detailed options so wrong-location, wrong-category, and dense-instance ambiguity are not collapsed too early.

Initial manual queue cap: 16 to 32 cases/images. Expand to 64 only if the gate remains unstable.

### `threshold_prior_transfer_summary.json`

Summarize how prior bands transfer to A2/A3/A4:

- proxy source summary and provenance counters;
- per-run resolved-config/generation comparability summary;
- counts by run and auto label;
- valid/objectness-supported retention by threshold band;
- danger-tail rejection by threshold band;
- overlap distribution in severe and broad suspicious bands;
- coord-confidence separability for duplicate/invalid vs valid/objectness-supported extras;
- whether a simple threshold, a two-feature gate, or no threshold transfers cleanly.

## Review Packet Policy

Execution is table-first:

```text
tables -> ranked ambiguity queue -> selective review packet -> user audit only if needed
```

Existing material under `/data/CoordExp/temp/a4_rp110_tf_probe_and_manual_review_20260512` can be used as supporting context, but it is incomplete and must not own case selection. If reused, use v2 pixel-GT/no-GT overlays and avoid the older buggy GT-green overlay.

Generate fresh visuals only for selected unresolved cases. Do not generate a broad gallery by default. Queued rows should preserve source JSONL and line/object indices so existing review helpers can recover the original image-level context; otherwise, materialize one-record JSONLs or use the lower-level `src.vis` path for the selected cases only.

If human review is used, labels should be written to a companion artifact such as `/data/CoordExp/temp/boundary_tail_direction_gate_20260513/review_packet/manual_audit_labels.jsonl` with `audit_id`, `audit_label`, and `audit_notes`. The raw `manual_audit_queue.jsonl` stays as the immutable queue; final reporting resolves labels by `audit_id` instead of requiring manual in-place edits.

## Mechanism Taxonomy

Per-image mechanism labels:

- `stable_complete`;
- `conservative_stop`;
- `valid_extra_unlocked`;
- `dirty_tail_unlocked`;
- `mixed_separable`;
- `prefix_state_damaged`;
- `unclear_needs_review`.

Priority:

1. decide dirty continuation vs valid-extra-unlocked;
2. decide whether generated-prefix damage explains the split;
3. decide whether the next direction is duplicate stability, valid emission, gated hybrid, data expansion, prefix-state rollout training, coordinate/object binding, or a new objective/gating direction.

## Final Report Contract

Promote a concise report to `progress/diagnostics/2026-05-13_boundary_tail_direction_gate.md` only after analysis is actually run.

The report must include:

- artifact roots and exact scope;
- current A2/A3/A4 mechanism summary;
- COCO-strict and coverage-aware conclusions;
- prior-threshold transfer result;
- duplicate-collapse diagnosis;
- LVIS/proxy objectness read;
- manual audit summary if used;
- lightweight objective sketches;
- one recommended next action;
- falsifier for the recommendation.

If `manual_audit_queue.jsonl` contains unresolved decision-critical cases, the report must not present a training/data/objective direction as concluded. It should mark the mechanism decision as blocked or provisional and make the immediate next action manual audit or a tiny discriminating probe.

Objective sketches should include:

- name;
- mechanism target;
- positive signal;
- negative/protection signal;
- expected benefit;
- main risk;
- minimal smoke to test it.

Candidate objective families:

- objectness-protected continuation;
- protected duplicate-burst unlikelihood;
- boundary-confidence gated EOS/continue;
- repaired/generated-prefix rollout training;
- coordinate/object-binding stability objective;
- LVIS-enhanced or LVIS-proxy data expansion.

## Escalation Policy

Default to the smallest credible next step:

1. Probe first if objective failure vs prefix-state damage is unresolved.
2. Tiny/small training smoke if the mechanism is clear enough to test a candidate objective or data change.
3. Production candidate only if artifact evidence is strong, manual review does not overturn it, LVIS/proxy/data path is ready, risks are protected, and code/config risk is low.

All recommended directions must include falsifiers:

| Direction | Falsifier |
|---|---|
| objectness-protected continuation | LVIS/proxy/manual review shows most extras are not real objects, or weak-positive training increases dirty tail without improving valid extras |
| protected duplicate-burst suppression | many duplicate-risk objects are visually/proxy-supported distinct instances, or suppression damages valid dense-object recall |
| boundary-confidence gated EOS/continue | coord confidence does not transfer, rejects too many valid/proxy-supported objects, or misses high-confidence dirty duplicates |
| repaired/generated-prefix rollout training | repaired prefixes do not restore healthy next-object distributions, or generated-prefix failures are not materially different from GT-prefix failures |
| LVIS-enhanced/data expansion | mapping/objectness noise is too high, schema incompatibility remains unresolved, or proxy additions mostly duplicate existing labels without addressing missing objects |
| coordinate/object-binding objective | duplicate bursts do not show local-basin or `x1/y1` stickiness, or binding signals do not separate failures from valid dense emissions |

## Worktree And Execution

If implementation begins, use an isolated worktree:

```bash
git worktree add -b codex/boundary-tail-direction-gate \
  /data/CoordExp/.worktrees/boundary-tail-direction-gate \
  main
```

Execution artifacts remain under the main checkout, even when code executes from a worktree:

```text
/data/CoordExp/temp/boundary_tail_direction_gate_20260513/
```

Candidate implementation files:

- `configs/analysis/boundary_tail_direction_gate/a2_a3_a4_val200.yaml`;
- `scripts/analysis/diagnose_boundary_tail_direction_gate.py`;
- `src/analysis/boundary_tail_direction_gate.py`;
- `tests/test_boundary_tail_direction_gate.py`.

The harness should read absolute artifact roots, emit tables under the absolute `/data/CoordExp/temp/...` directory, and avoid new training behavior, config-system complexity, evaluator contracts, or broad CLI expansion. If run from an isolated worktree, it must still resolve artifacts and outputs against `/data/CoordExp`, not against `.worktrees/...`.

## Approval State

The design choices in this spec reflect the checkpointed grilling decisions through Question 48 on 2026-05-13. This written spec remains pending as an execution contract until the user explicitly approves execution after audit revisions.
