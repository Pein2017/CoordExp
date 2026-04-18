# Raw-Text Coordinate Continuity Probe Design

Date: 2026-04-18
Status: proposed
Owner: Codex

## Goal

Design a falsifiable, artifact-driven probe study that determines whether
`raw_text_xyxy` grounding models already exhibit coordinate continuity under
pure cross-entropy training, whether that continuity is visually grounded, and
whether the same continuity can also create wrong-instance local basins in
crowded same-class scenes.

The study is explicitly not a lightweight engineering check. It is a theory
validation plus large-scale probe study whose primary deliverable is a research
report with machine-readable artifacts, not just a new analysis script.

## Primary Questions

The final report must answer these five questions, each with one of:

- `strongly supported`
- `partially supported`
- `not supported`
- `inconclusive`

Questions:

1. Does base `Qwen3-VL-2B-Instruct-coordexp` already exhibit raw-text numeric
   adjacency or coordinate continuity?
2. Does Stage-1 `raw_text_xyxy` pure-CE fine-tuning strengthen that
   continuity?
3. Is that continuity materially stronger under the correct image than under a
   swapped or wrong image condition?
4. In crowded or repeated-object hard cases, does the model form a wrong local
   basin around an incorrect prefix anchor, especially at `x1` and `y1`?
5. If the only goal is to give grounding a local continuity prior, does
   `coord_token` still need to be treated as the core reason for continuity?

## Study Scope

### Checkpoints

- base checkpoint:
  `model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp`
- Stage-1 raw-text pure-CE checkpoint:
  `output/stage1_2b/coco_bbox_max60-coco80-desc_first-1024-lvis_proxy-raw_text_xyxy-pure_ce/epoch_4-raw_text_xyxy-pure_ce-coco80-desc_first-1024-lvis_proxy-from-base-2B/v1-20260417-084341/checkpoint-552`

### Data surfaces

Prefer repo-native COCO JSONL assets, preferring `1024` surfaces when present.
Use:

- a `val`-first headline slice for the main conclusions
- a larger `train` supplemental slice for power and hard-case mining
- hard subsets that emphasize crowded same-class and duplicate-prone scenes

### Resource budget

This design targets the agreed `B`-scale budget:

- methods-first, then medium-to-large execution
- enough compute to run the full four probe phases on meaningful subsets
- no requirement to treat this as a pilot-only exercise

## Study Philosophy

This is an upper-bound-oriented study, but it still needs a clean causal lane.

To balance both goals, the study uses two coordinated lanes:

### Lane A: Canonical Scoring Lane

Purpose:

- establish the cleanest evidence for continuity itself
- minimize prompt confounds where possible
- answer Questions 1 to 3 with the strongest controls

Main properties:

- fixed `raw_text_xyxy` contract
- candidate coordinate chunk scoring under teacher forcing
- lexical and image controls
- headline evidence for `base` versus `raw_text pure-CE`

### Lane B: Upper-Bound Capability Lane

Purpose:

- probe the strongest behavior the base model can express under the same raw
  text contract
- study self-prefix bad basins, hard dense scenes, and counterfactual prefix
  geometry interventions

Main properties:

- prompt rescue is allowed when it does not change the output contract
- free-generation stability is treated as supportive upper-bound evidence, not
  the sole continuity criterion
- main focus is Questions 4 and 5

## Reuse-First Constraint

The study must reuse existing code and artifact contracts where possible.

Primary reuse targets:

- inference entrypoint:
  `scripts/run_infer.py`
- inference orchestration:
  `src/infer/pipeline.py`
- teacher-forced multimodal scoring scaffolding:
  `src/analysis/unmatched_proposal_verifier.py`
- hard-case mining and replay conventions:
  `src/analysis/duplication_collapse_analysis.py`
- prefix-study cell structure and mutation patterns:
  `src/analysis/rollout_fn_factor_study.py`
- shared GT-vs-Pred canonical rendering:
  `src/vis/gt_vs_pred.py`

New code should only cover the gap that is not already present:

- robust multi-token candidate coordinate chunk scoring for raw-text integers
- continuity-specific aggregation and report generation
- 1D basin and 2D heatmap figure builders

## Core Methodological Contract

### The primary measurement is candidate coordinate chunk scoring

The study must not rely on single-token logits as the main continuity signal.

For each probed slot, score a candidate integer `k` by rebuilding the complete
assistant text with that slot value replaced and computing the conditional
score of the slot's serialized coordinate chunk under teacher forcing.

The core observable is therefore:

`score_chunk(k | prefix, image, slot, prompt_surface)`

This chunk-level score must support:

- multi-token numbers
- variable digit lengths
- punctuation and closing boundary effects
- teacher-forced ground-truth prefixes
- self-generated prefixes

### Main score families

Record both:

- `mean_logprob`
- `sum_logprob`

Use `mean_logprob` as the default headline score because it is less directly
confounded by candidate token-count differences. Keep `sum_logprob` as a
sensitivity read.

### Phase 0 audit requirements

Before large-scale probe execution, audit:

- exact output serialization for `raw_text_xyxy`
- tokenization of:
  `0, 1, 9, 10, 99, 100, 199, 200, 210, 999`
- tokenization stability across neighboring punctuation contexts
- teacher-forced assistant-span alignment
- self-prefix assistant-span alignment
- batch padding and left-padding robustness

The report must treat the user-provided tokenizer screenshot as a starting
hypothesis to verify, not as accepted fact without audit.

## Good Basin And Bad Basin Definitions

### Good basin

A good basin exists when, under a correct-image and teacher-forced ground-truth
prefix condition, the slot landscape shows a meaningful local score advantage
near the GT coordinate.

Required measurements:

- `mass@1`, `mass@2`, `mass@4`, `mass@8`, `mass@16`
- distance-to-GT versus score correlation
- local expected absolute error
- basin width and half-height width
- slot-wise stratification for `x1`, `y1`, `x2`, `y2`
- size and crowding stratification

### Bad basin

A bad basin exists when, under self-generated or counterfactual-prefix
conditions, local score mass is concentrated more strongly around an incorrect
anchor than around the GT anchor.

Candidate wrong anchors:

- current predicted coordinate
- previous same-class object coordinate neighborhood
- counterfactually perturbed prefix geometry anchor

Required measurements:

- prediction-centered versus GT-centered local mass
- wrong-anchor advantage
- argmax offset relative to GT
- 2D heatmap peak movement under prefix interventions
- hard-subset versus easy-subset contrast

## Controls

### Lexical confound control

Each candidate record must include at least:

- numeric distance to the active center
- numeric distance to GT
- character edit distance
- token edit distance
- digit-length match
- token count
- same-leading-digit indicator
- shared numeric prefix length

Primary model:

`score ~ numeric_distance + char_edit_distance + token_edit_distance + token_count + digit_length_match + slot + size_bucket + crowding_bucket + image_condition + model_family + interactions`

If sample volume is sufficient, add mixed-effects structure with random
intercepts at the case and object level.

Interpretation rule:

- if numeric distance remains significant after lexical controls, that supports
  real numeric continuity rather than a pure string-tokenization artifact

### Image control

Compare at least:

- correct image
- swapped random image
- blank or dummy image if the interface allows it without violating the model
  contract

Primary derived metric:

- `Vision Lift = score(GT | correct image) - score(GT | swapped image)`

## Data Cohorts

Construct explicit manifests rather than ad hoc in-memory subsets.

Required cohort families:

1. `val_broad_random_headline`
2. `train_broad_random_supplemental`
3. `isolated_easy`
4. `crowded_same_class`
5. `model_mined_duplicate_prone`

Default scale targets:

- `val`: about `500` images for headline large-scale probes
- `train`: about `1500` images for supplemental power
- deep hard subset: about `100` cases
- 2D heatmap subset: about `50` cases

The crowded and duplicate-prone slices should reuse the repo's existing
duplication-collapse mining language where possible, including:

- `max_desc_count`
- same-class overlap counts
- duplicate-like pair counts
- onset-oriented failure cues

## Dense-Scene Failure Alignment

Add an auxiliary analysis block that aligns the new raw-text study with the
existing dense-scene and duplication-collapse research record.

Purpose:

- determine whether base and raw-text pure-CE reproduce failure families
  similar to those previously observed under `<|coord_*|>` routes
- clarify whether continuity-driven benefits and continuity-driven risks are
  specific to coord tokens or are broader autoregressive geometry phenomena

This block is supportive evidence for Questions 4 and 5, not the sole headline
evidence for Questions 1 to 3.

Required views:

- tail metrics:
  `pred_count`, `pred/gt`, `max_same_desc_count`, cap-hit counts, burst-class
  concentration, near-overlap counts
- visual family labels:
  healthy multiplicity, near-duplicate loop, group-box granularity failure,
  unlabeled-real-object ambiguity, enumeration miss
- mechanism alignment probes:
  `predicted_object vs exact_duplicate`, onset-local `x1/y1` escape, and
  prefix-geometry perturbation on a small hard subset

Interpretation rule:

- if raw-text pure-CE still exhibits dense-scene bad basins, continuity itself
  is likely a double-edged phenomenon rather than a coord-token-exclusive one
- if raw-text pure-CE reduces the same failure tails relative to coord-token
  families, that weakens the claim that coord tokens are the only or best path
  to useful continuity

## Execution Plan

### Phase 0: contract and tokenizer audit

Deliverables:

- tokenizer audit section
- serialization audit section
- scorer sanity-check records

Exit condition:

- multi-token candidate chunk scoring is validated for both teacher-forced and
  self-prefix conditions

### Phase 1: instrument validation pilot

Scale:

- roughly `20` to `40` cases across easy, crowded, and hard slices

Purpose:

- verify scorer stability and probe interpretability before scaling

Exit condition:

- pilot landscapes are coherent enough to justify large-scale runs

### Phase 2: canonical scoring lane headline run

Purpose:

- answer Questions 1 to 3

Scale:

- `val` headline cohort
- `train` supplemental cohort
- local sweeps at scale
- `0..999` full sweeps on a curated deep subset

### Phase 3: upper-bound bad-basin lane

Purpose:

- answer Question 4 on hard dense scenes

Scale:

- hard-case subset around `100`
- 2D heatmap subset around `50`

Main conditions:

- self-generated prefixes
- prediction-centered and GT-centered scans
- easy versus hard contrast

### Phase 4: prefix geometry perturbation

Purpose:

- isolate whether the harmful anchor is driven by text presence, bbox geometry,
  or specifically `x1/y1` geometry

Counterfactuals:

1. delete previous same-class object
2. keep text, swap previous bbox to a neighboring instance
3. linearly interpolate previous bbox
4. change only previous `x1/y1`

### Phase 5: synthesis and verdict

Purpose:

- consolidate all evidence into the five mandatory answers
- state the continuity-specific implication for `coord_token`
- explicitly separate direct evidence from open questions

## Artifact Contract

Write outputs under a new analysis root:

`output/analysis/raw_text_coord_continuity_probe_<date>/`

Required contents:

- `report.md`
- `summary.json`
- `per_coord_scores.jsonl` or parquet
- `hard_cases.jsonl`
- cohort manifests
- aggregate metric tables
- 1D basin plots
- 2D heatmaps
- regression coefficient plots
- any required minimal reusable analysis code and targeted tests

The artifact structure must separate:

- cohort selection
- inference reproduction
- raw probe records
- aggregated summaries
- figures
- report synthesis

No notebook-only state should be required to reproduce the final report.

## Evidence Mapping To The Five Questions

### Question 1

Main evidence:

- base-model GT-centered chunk landscapes
- lexical confound controls
- image controls

### Question 2

Main evidence:

- matched probe comparison between base and raw-text pure-CE
- basin-strength deltas
- model-family interactions in the regression layer

### Question 3

Main evidence:

- Vision Lift
- correct-image versus swapped-image basin comparisons

### Question 4

Main evidence:

- self-prefix hard-case landscapes
- prediction-centered versus GT-centered comparisons
- 2D heatmaps
- prefix geometry interventions

### Question 5

Main evidence:

- Questions 1 to 4 jointly
- dense-scene failure alignment

Constraint:

- only make a direct necessity claim for `coord_token` as a continuity device
- do not overclaim about typing, output stability, or parameterization unless
  separate evidence directly supports it

## Verdict Rules

### Strongly supported

Use only if at least two independent evidence chains agree and the key controls
or interventions do not refute the claim.

### Partially supported

Use when the signal is real but limited to certain cohorts, lanes, or controls.

### Not supported

Use when the expected signal fails to appear or key controls point in the
opposite direction.

### Inconclusive

Use when evidence is insufficient, controls remain unresolved, or the evidence
chains materially disagree.

## Risks And Guardrails

- Do not let prompt rescue contaminate the canonical lane.
- Do not treat high same-class multiplicity as automatically pathological in
  dense scenes.
- Do not reduce continuity to token-count artifacts.
- Do not replace chunk scoring with a token-logit shortcut.
- Do not produce a strong `coord_token` claim unless Questions 1 to 5 support
  it directly.

## Assumptions

- `Qwen3-VL-2B-Instruct-coordexp` is the only required base checkpoint for this
  study.
- The Stage-1 raw-text pure-CE checkpoint is treated as the relevant trained
  comparator.
- `val` is the headline conclusion split; `train` is supplemental.
- The study is allowed to target capability upper bounds for base-model probes,
  provided the report keeps rescue-prompt effects explicit.

## Completion Gate

This design is complete when:

- the spec is approved by the user
- the subsequent implementation plan can be written without unresolved
  structural ambiguity
- the report contract, artifact contract, and evidence thresholds are all
  explicit enough to support a reproducible execution phase
