# Raw-Text Coordinate Continuity Probe

## Scope

This bundle consolidates the raw-text `xyxy` continuity study across Phase 0 audit, GT-centered good-basin probes, lexical controls, image-swap controls, self-prefix bad-basin probes, and prefix-geometry perturbation.

## Phase 0 Audit

- Serialization surface used for training/inference/probing: `pretty_inline`.
- Both `base` and `pure_ce` tokenize `199 -> ['1', '9', '9']` and `200 -> ['2', '0', '0']` as digit-by-digit sequences rather than single whole-number tokens.
- The active raw-text contract is `{"objects": [{"desc": ..., "bbox_2d": [x1, y1, x2, y2]}, ...]}` with pretty-inline spacing.

## Core Verdicts

### 1. Base Qwen3-VL already has raw-text numeric adjacency / coordinate continuity.

- Verdict: **strongly supported**
- Rationale: Base model shows significant negative numeric-distance coefficient after lexical controls and broad GT-centered local mass remains high for x1/y1.

### 2. Stage-1 pure-CE fine-tuning enhances that continuity.

- Verdict: **partially supported**
- Rationale: Pure-CE improves both mass@4 and local error on 5/6 x1/y1 slice comparisons, but gains are slot-dependent rather than uniform.

### 3. Continuity is stronger under the correct image than under swapped-image controls.

- Verdict: **strongly supported**
- Rationale: Correct-image condition yields uniformly positive GT-score lift and better local mass@4 than swapped-image controls for both models and both early slots.

### 4. Hard repeated-object cases can form wrong local basins around the wrong prefix, especially at x1/y1.

- Verdict: **strongly supported**
- Rationale: Hard repeated-object cases show higher mass around the model-predicted anchor than around GT, and prefix-geometry interventions can move that wrong-anchor advantage.

### 5. If the goal is only local continuity, coord_token remains necessary.

- Verdict: **not supported**
- Rationale: If the objective is only to obtain local continuity, the evidence does not support coord_token as necessary: raw-text pure-CE and even the base model already exhibit numeric local basins. Typing/stability/parameterization benefits remain open.

## High-Signal Evidence

- Lexical control: combined numeric-distance coefficient = `-0.1449` with p-value `4.53e-19`; per-model coefficients remain negative and significant.
- Vision lift: mean GT-score lift spans `3.097` to `4.876` across model/slot cells, and every mass@4 lift is positive.
- Final synthesis: **C + D + E: the models exhibit real numeric continuity that is visually modulated, but hard repeated-object prefixes can also induce wrong local basins; coord_token looks unnecessary if continuity is the only target, while other benefits remain open.**.

## Interpretation For `coord_token`

The probe study supports a narrow claim: raw-text pure-CE does not need coord_token in order to exhibit local coordinate continuity. The open question is not continuity creation, but whether special coordinate parameterization still helps typing discipline, decoding stability, or a cleaner instance-separation geometry under rollout.

## Manual Review

A human-in-the-loop review interface is available for bbox-first auditing on the original image, with 2D heatmaps kept as supporting mechanism evidence rather than the primary annotation surface.

- Review gallery: `/data/CoordExp/output/analysis/raw-text-coord-continuity-probe-2026-04-18/manual_review/review.md`
- Annotation workbook: `/data/CoordExp/output/analysis/raw-text-coord-continuity-probe-2026-04-18/manual_review/annotation_workbook.md`
- BBox audit template: `/data/CoordExp/output/analysis/raw-text-coord-continuity-probe-2026-04-18/manual_review/bbox_annotations_template.jsonl`
- Human findings: `/data/CoordExp/output/analysis/raw-text-coord-continuity-probe-2026-04-18/manual_review/human_findings.md`
- Structured templates: `/data/CoordExp/output/analysis/raw-text-coord-continuity-probe-2026-04-18/manual_review/case_annotations_template.jsonl` and `/data/CoordExp/output/analysis/raw-text-coord-continuity-probe-2026-04-18/manual_review/panel_annotations_template.jsonl`
