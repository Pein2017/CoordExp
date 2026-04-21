---
title: Coordinate Family Basin and Recall Comparison
date: 2026-04-20
status: consolidated-interim
owner: codex
branches:
  - codex/coord-family-comparison
depends_on:
  - progress/diagnostics/raw_text_coord_continuity_probe_2026-04-20.md
---

# Coordinate Family Basin and Recall Comparison

## Scope

This note archives the second super-power study: a 2B-first comparison across
coordinate parameterization / supervision families, with emphasis on:

- detection quality,
- oracle-vs-baseline recall behavior,
- recoverable vs systematic false negatives,
- and family-specific failure modes.

The original design lives in:

- `docs/superpowers/specs/2026-04-20-coord-family-basin-and-recall-comparison-design.md`

The runtime outputs originally lived under `output/analysis/` and are now
copied into `progress/diagnostics/artifacts/coord_family_comparison_2026-04-20/`.

## Final Archival Scope

This permanent note intentionally focuses on the four families that are already
decision-relevant:

- `center_parameterization`
- `raw_text_xyxy_pure_ce`
- `base_xyxy_merged`
- `hard_soft_ce_2b`

The `cxcywh_pure_ce` and `cxcy_logw_logh_pure_ce` families were explored and
their partial artifacts remain useful, but they are not part of the final
decision-oriented synthesis here because:

- their matched-val200 performance trails the `center/raw_text` families,
- and their Oracle-K lane was still in progress at the time this archive was
  written.

## Comparison Method

The family comparison was built in layers rather than through one monolithic
metric.

### 1. Contract audit

The study first audited each family for:

- merged vs adapter loading path
- bbox format
- infer mode
- pred coord mode
- family-native surface compatibility

This was necessary because these families do not share the same output
contract.

### 2. Matched val200 detection evaluation

All families were evaluated on a matched `val200` subset to establish a common
performance baseline.

Headline ranking:

1. `center_parameterization`
2. `raw_text_xyxy_pure_ce`
3. `cxcy_logw_logh_pure_ce`
4. `cxcywh_pure_ce`
5. `hard_soft_ce_2b`
6. `base_xyxy_merged`

### 3. Verifier and Oracle-K recall lanes

For each family, the study separated:

- baseline detection quality
- verifier discrimination quality
- Oracle-K recoverability
- recoverable vs systematic FN structure

This was the core mechanism lane for answering:

- is low recall mostly "the model did not see it"?
- or "the model could say it, but baseline decoding did not realize it"?

### 4. FN slicing and invalid-geometry audits

Two additional analysis assets were added before the comparison was archived:

- recoverable/systematic FN slices by size, crowding, and repeated-desc bucket
- invalid-geometry audits for the weak `base/hard_soft` families

## Main Findings

### 1. `center_parameterization` is the strongest current family

On matched `val200`:

- `bbox_AP = 0.4221`
- `bbox_AP50 = 0.6007`
- `F1@0.50 = 0.6108`

Its recall lane shows a healthy but limited Oracle-K gap:

- baseline `recall_loc = 0.6163`
- Oracle-K `recall_loc = 0.6998`
- baseline FN `216`
- recoverable FN `54`
- systematic FN `162`

Interpretation:

- `center` already realizes much of its usable capacity at baseline decode
- the remaining misses are mostly systematic rather than merely suppressed

### 2. `raw_text_xyxy_pure_ce` is the strongest non-center family and the most interesting mechanism case

On matched `val200`:

- `bbox_AP = 0.3440`
- `bbox_AP50 = 0.4500`
- `F1@0.50 = 0.5900`

Its recall lane is notably different from `center`:

- baseline `recall_loc = 0.4742`
- Oracle-K `recall_loc = 0.7069`
- baseline FN `296`
- recoverable FN `134`
- systematic FN `162`

Interpretation:

- `raw_text` has a much larger recoverable recall gap than `center`
- that means a larger fraction of its missed detections are not permanently
  lost
- baseline decoding is leaving more usable capacity unrealized

### 3. `base_xyxy_merged` and `hard_soft_ce_2b` are not mainly "hesitation" families

Their matched `val200` metrics are very weak:

- `base_xyxy_merged`: `bbox_AP = 0.0549`
- `hard_soft_ce_2b`: `bbox_AP = 0.0568`

Their Oracle-K gains are tiny:

- `base`: `0.1012 -> 0.1119`
- `hard_soft`: `0.1066 -> 0.1208`

Recoverable fractions are near zero:

- `base`: `0.49%`
- `hard_soft`: `0.55%`

Interpretation:

- these families are not primarily underperforming because decoding is timid
- they behave more like systemically weak or surface-unstable families

## Low-Recall Mechanism Read

The most important mechanism findings are:

### `center_parameterization`

- recoverable fraction of baseline FN:
  - `25.0%`
- mechanism mix:
  - `suppressed_fn_rate = 0.0000`
  - `competitive_fn_rate = 0.0417`
  - `weak_visual_fn_rate = 0.9583`

Interpretation:

- low recall is mostly not "the model saw it but would not say it"
- most misses look like support-limited or systematic misses

### `raw_text_xyxy_pure_ce`

- recoverable fraction of baseline FN:
  - `45.27%`
- mechanism mix:
  - `suppressed_fn_rate = 0.0068`
  - `competitive_fn_rate = 0.0135`
  - `weak_visual_fn_rate = 0.9797`

Interpretation:

- `raw_text` has a much larger decode-sensitive recall gap than `center`
- but that gap still does not reduce to a simple "high-support suppressed FN"
  story
- the better interpretation is:
  - partial support exists,
  - baseline decoding is not fully realizing it,
  - and repeated-object / complex-scene competition still matters

## FN Slice Findings

The recoverable/systematic FN slice analysis shows:

### `raw_text`

- recoverable fraction rises on larger targets:
  - `tiny = 0.2973`
  - `small = 0.4500`
  - `medium = 0.6667`
  - `large = 0.5135`

This suggests:

- `raw_text` often has usable support on medium and large objects
- the remaining failure is frequently realization / stability rather than total
  absence of signal

### `center`

- the same trend exists but is weaker:
  - `tiny = 0.1143`
  - `small = 0.3048`
  - `medium = 0.3333`
  - `large = 0.3478`

Interpretation:

- `center` is already more baseline-efficient
- there is less recoverable mass left on the table

### `base` and `hard_soft`

- almost every slice remains overwhelmingly systematic
- neither family shows a meaningful recoverable band even on larger objects

## Invalid-Geometry Failure Audit

The weak families also reveal a more concrete surface-level failure mode.

### `base_xyxy_merged`

- images with invalid geometry:
  - `11`
- invalid-geometry error entries:
  - `152`
- dominant failure families:
  - `overflow_right = 128`
  - `overflow_bottom = 52`
- dominant invalid desc:
  - `book = 117`

### `hard_soft_ce_2b`

- images with invalid geometry:
  - `12`
- invalid-geometry error entries:
  - `46`
- dominant failure families:
  - `overflow_bottom = 49`
  - `overflow_right = 24`
- dominant invalid desc:
  - `cake = 22`
  - `person = 15`
  - `book = 14`

Interpretation:

- these are not merely low-confidence families
- they display directional geometric failure modes
- their failures are structured, not random

## Durable Artifact Map

Copied summaries and reports:

- comparison:
  - [comparison_report.md](artifacts/coord_family_comparison_2026-04-20/comparison_report.md)
  - [comparison_summary.json](artifacts/coord_family_comparison_2026-04-20/comparison_summary.json)
- recall progress:
  - [recall_progress_report.md](artifacts/coord_family_comparison_2026-04-20/recall_progress_report.md)
  - [recall_progress_summary.json](artifacts/coord_family_comparison_2026-04-20/recall_progress_summary.json)
- recall slices:
  - [recall_slices_report.md](artifacts/coord_family_comparison_2026-04-20/recall_slices_report.md)
  - [recall_slices_summary.json](artifacts/coord_family_comparison_2026-04-20/recall_slices_summary.json)
- invalid geometry:
  - [invalid_geometry_audit_report.md](artifacts/coord_family_comparison_2026-04-20/invalid_geometry_audit_report.md)
  - [invalid_geometry_audit_summary.json](artifacts/coord_family_comparison_2026-04-20/invalid_geometry_audit_summary.json)

## Interim Recommendation

At the time of archiving, the recommendation is:

- if the goal is strongest current 2B deployment quality:
  - prefer `center_parameterization`
- if the goal is mechanism research on whether raw text can replace
  `coord_token`-style continuity assumptions:
  - continue with `raw_text_xyxy_pure_ce`
- do not prioritize `base_xyxy_merged` or `hard_soft_ce_2b` for forward
  research until their output-surface instability is better understood

The `cxcywh` and `cxcy_logw_logh` families remain historical side lanes rather
than primary recommendations in this archive.
