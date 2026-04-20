# 2B Coordinate Family Comparison: Continuity, Bad Basins, and Low-Recall Mechanisms

Date: 2026-04-20
Status: proposed
Owner: Codex

## Goal

Design a new artifact-driven comparative study that extends the completed
raw-text continuity probe into a broader **2B coordinate-family comparison**.

This study is not a lightweight appendix to the prior raw-text work. Its goal
is to compare multiple Stage-1 parameterization and supervision families under
a unified research frame:

- how much **helpful local continuity** they exhibit
- how much **wrong-basin competition** they exhibit in dense same-class scenes
- whether their low-recall behavior is dominated by
  - objects that were likely **seen but not emitted**
  - objects that were likely **not seen or only weakly seen**
  - objects suppressed by **same-class competition**

The primary deliverable is a comparative research bundle with a report,
machine-readable tables, reusable manifests, and reproducible analysis code.

## Relationship To The Prior Raw-Text Study

The existing study under
`output/analysis/raw-text-coord-continuity-probe-2026-04-18/`
is treated as **Study A: baseline anchor study**.

That prior study already answered:

- whether base raw-text models exhibit continuity
- whether raw-text pure-CE strengthens it
- whether continuity is visually modulated
- whether bad basins can form in repeated-object scenes
- whether `coord_token` remains necessary if continuity is the only target

This new study is **Study B: family comparison study**.

It does not replace Study A. Instead, it asks:

- which parameterization or supervision families also exhibit continuity
- which ones exhibit healthier continuity
- which ones exhibit more dangerous bad-basin behavior
- which ones fail recall because they do not emit supported objects versus
  because they never develop strong support for them

## Primary Questions

The final report must answer each question with one of:

- `strongly supported`
- `partially supported`
- `not supported`
- `inconclusive`

Questions:

1. Do the selected 2B model families all exhibit **family-native local
   continuity** in their own parameterization spaces?
2. When mapped back to canonical bbox semantics, do the families differ
   materially in **healthy GT-centered basin strength**?
3. Do the families differ materially in how much their continuity is
   **visually grounded** rather than dominated by language priors?
4. Do the families differ materially in **bad-basin risk** for crowded or
   repeated-object scenes?
5. In low-recall settings, do the families differ in whether false negatives
   are better explained as
   - `suppressed FN`
   - `weak-visual FN`
   - `competitive FN`
6. If the downstream goal is not merely continuity but **healthy grounding
   behavior**, which families look most promising to continue?

## Family Scope

### Headline 2B families

The main comparison lane is fixed to the following 2B families.

1. `base_xyxy_merged`
   - `output/stage1_2b/coco_bbox_max60-coco80-desc_first-1024-lvis_proxy-merged`
2. `raw_text_xyxy_pure_ce`
   - `output/stage1_2b/coco_bbox_max60-coco80-desc_first-1024-lvis_proxy-raw_text_xyxy-pure_ce/epoch_4-raw_text_xyxy-pure_ce-coco80-desc_first-1024-lvis_proxy-from-base-2B`
3. `cxcywh_pure_ce`
   - `output/stage1_2b/coco_bbox_max60-coco80-desc_first-1024-lvis_proxy-cxcywh-pure_ce/epoch_4-cxcywh-pure_ce-coco80-desc_first-1024-lvis_proxy-from-base-2B`
4. `cxcy_logw_logh_pure_ce`
   - `output/stage1_2b/coco_bbox_max60-coco80-desc_first-1024-lvis_proxy-cxcy_logw_logh-pure_ce/epoch_4-cxcy_logw_logh-pure_ce-coco80-desc_first-1024-lvis_proxy-from-base-2B`
5. `center_parameterization`
   - `output/stage1_2b/coco_bbox_max60-1024-lvis_proxy-center_parameterization-ckpt_1564-merged`
6. `hard_soft_ce_2b`
   - `output/stage1_2b/coco_bbox_max60-hard_soft_ce-2b-merged`

### Auxiliary 4B lane

4B checkpoints under `output/stage1/` are allowed only as an auxiliary lane.

Use them for:

- contract sanity checks
- small matched qualitative slices
- historical continuity or duplication-family context

Do not use 4B runs as the primary statistical lane for headline conclusions,
because the main goal is to isolate **family differences at fixed 2B scale**.

## Reuse-First Constraint

The study must reuse the existing inference and evaluation stack rather than
building a bespoke execution path.

Primary reuse targets:

- inference entrypoint:
  `scripts/run_infer.py`
- inference orchestration:
  `src/infer/pipeline.py`
- inference engine and checkpoint resolution:
  `src/infer/engine.py`
- evaluator workflow:
  `scripts/evaluate_detection.py`
  `docs/eval/WORKFLOW.md`
- repeated-sampling / Oracle-K workflow:
  `scripts/evaluate_oracle_k.py`
- duplication-collapse cohort mining and replay conventions:
  `src/analysis/duplication_collapse_analysis.py`
  `configs/analysis/duplication_collapse/*.yaml`
- unmatched proposal and teacher-forced scoring scaffolding:
  `src/analysis/unmatched_proposal_verifier.py`
- shared GT-vs-Pred canonical review rendering:
  `src/vis/gt_vs_pred.py`

New code should fill only the missing pieces:

- family inventory and contract audit
- family-native continuity scoring for non-raw-text parameterizations
- cross-family canonical comparison aggregation
- low-recall mechanism classification and reporting

## Checkpoint Loading Policy

The new study must explicitly support both:

- fully merged checkpoints
- adapter checkpoints loaded dynamically at inference time

Current repository state already supports this through:

- `infer.model_checkpoint`
- optional `infer.adapter_checkpoint`
- `resolve_inference_checkpoint(...)`
- HF runtime adapter loading in `src/infer/engine.py`

Interpretation rules:

- prefer dynamic adapter loading when the checkpoint family is only available as
  an adapter and a merge is not required
- do not create unnecessary merged copies just for this study
- keep `vLLM` excluded for adapter-backed runs when the current engine contract
  does not support adapter inference there

## Core Comparison Axes

### Axis 1: Basin behavior

Measure, compare, and interpret:

- **good basin**
  - GT-centered local support
- **bad basin**
  - wrong-anchor local support in repeated-object scenes
- **perturbation sensitivity**
  - whether prefix geometry changes move the basin

### Axis 2: Recall mechanism

Measure, compare, and interpret whether false negatives are better explained
as:

- `suppressed FN`
  - object appears supported under teacher forcing or proposal insertion, but
    was not emitted under the default generation path
- `weak-visual FN`
  - object remains weakly supported even under teacher forcing and proposal
    insertion
- `competitive FN`
  - object has some support, but another same-class local basin dominates the
    scene

## Family-Native Versus Canonical Comparison

The study must not force every family into the same probe representation too
early.

Instead, comparisons happen in two layers:

### Layer A: family-native probe space

Each family is first evaluated in its own most natural parameterization space.

Examples:

- `raw_text_xyxy` or canonical `xyxy` family:
  - probe `x1`, `y1`, `x2`, `y2`
- `cxcywh` family:
  - probe `cx`, `cy`, `w`, `h`
- `cxcy_logw_logh` family:
  - probe `cx`, `cy`, `logw`, `logh`
- `center_parameterization` family:
  - probe center and size slots consistent with its actual infer/eval surface
- `hard_soft_ce_2b`:
  - first audit the actual infer surface before locking slot semantics

This layer answers:

- whether the family exhibits local continuity in the space it was trained to
  use

### Layer B: shared canonical semantic space

All families are then mapped back into canonical bbox semantics for actual
comparison.

The shared canonical layer includes:

- pixel-space bbox support
- GT-centered support summaries
- wrong-instance competition
- dense-scene duplicate-like failure signals
- FN recoverability and suppression patterns

This layer answers:

- whether the family-native continuity is useful or harmful in final grounding
  behavior

## Execution Plan

### Phase A: family inventory and contract audit

Purpose:

- enumerate the concrete families available under `output/stage1_2b/` and
  selected `output/stage1/`
- confirm which are merged versus adapter-backed
- audit actual infer surfaces and compatible evaluator paths

Required outputs:

- `family_inventory.json`
- `family_contract_audit.md`

For each family, record at least:

- alias
- checkpoint path
- checkpoint type:
  `merged` / `adapter`
- runtime load pattern:
  `model_checkpoint only` / `model_checkpoint + adapter_checkpoint`
- infer mode:
  `coord` / `text` / other
- bbox format:
  `xyxy` / `cxcywh` / `cxcy_logw_logh` / family-specific
- pred coord mode:
  `pixel` / `norm1000`
- supported eval compatibility path
- whether the family is admitted into the 2B headline lane

Exit condition:

- all headline 2B families have audited contracts and a clear runtime path

### Phase B: matched pilot on family-native continuity

Purpose:

- verify that each family can be meaningfully probed in its native slot space
- avoid scaling a comparison that only works for raw-text xyxy

Scale:

- roughly `20` to `40` matched cases spanning:
  - easy
  - crowded same-class
  - model-mined duplicate-prone

Core measurements:

- `mass@1`, `mass@2`, `mass@4`, `mass@8`, `mass@16`
- local expected absolute error
- basin width / half-height width
- target-distance versus score correlation
- family-native slot summaries

Exit condition:

- pilot continuity landscapes are coherent for all admitted headline families

### Phase C: large-scale family comparison

Purpose:

- produce the primary cross-family comparison

Headline cohorts:

1. `val_broad_random_headline`
2. `isolated_easy`
3. `crowded_same_class`
4. `model_mined_duplicate_prone`

Supplemental cohorts:

- a larger `train` slice for power if needed
- a curated deep subset for full sweeps and 2D heatmaps

Preferred default scale:

- headline `val`: about `500`
- supplemental `train`: about `1500`
- deep hard subset: about `100`
- 2D heatmap subset: about `50`

These are targets, not a validity gate. If runtime becomes the bottleneck, keep
the study honest and report the executed scale explicitly.

### Phase D: image grounding controls

Purpose:

- compare whether family continuity is visually grounded

Required image conditions:

- correct image
- swapped image

Optional if interface permits cleanly:

- blank or dummy image

Primary derived metric:

- `Vision Lift = support(target | correct image) - support(target | swapped image)`

### Phase E: bad-basin and perturbation comparison

Purpose:

- compare harmful local-basin behavior across families

Required conditions:

- self-generated or model-prefix replay where supported
- prediction-centered versus GT-centered local scans
- repeated-object hard subset
- prefix geometry perturbation on a smaller deep subset

Required perturbations:

1. delete previous same-class object
2. keep text but swap previous bbox to a neighboring instance
3. linearly interpolate the previous bbox
4. change only previous `x1/y1` or the family-equivalent onset geometry

### Phase F: low-recall mechanism study

Purpose:

- classify why false negatives happen for each family

Evidence chain:

1. **teacher-forced GT support**
   - does the missing GT object receive nontrivial support under correct-image
     teacher forcing?
2. **proposal-conditioned recoverability**
   - if the GT or FN candidate is inserted as a proposal or counterfactual
     object, does support rise?
3. **Oracle-K recoverability**
   - under repeated stochastic decoding, can the missed object be recovered?

Required FN labels:

- `suppressed_fn`
- `weak_visual_fn`
- `competitive_fn`

Interpretation rules:

- strong teacher-forced or proposal support plus recovery under Oracle-K favors
  `suppressed_fn`
- weak support across all conditions favors `weak_visual_fn`
- meaningful support with a stronger same-class competitor favors
  `competitive_fn`

### Phase G: synthesis

Purpose:

- consolidate family-native and canonical evidence
- answer the six primary questions
- provide a decision-oriented summary of which families look healthiest

## Metrics And Evidence Families

### Good-basin metrics

- `mass@1`, `mass@2`, `mass@4`, `mass@8`, `mass@16`
- target-distance versus score correlation
- local expected absolute error
- basin width / half-height width
- slot-wise stratification
- size and crowding stratification

### Bad-basin metrics

- prediction-centered local mass
- GT-centered local mass
- wrong-anchor advantage
- argmax offset relative to GT
- 2D heatmap peak movement
- hard-versus-easy contrast

### Recall-mechanism metrics

- GT support under teacher forcing
- GT support under proposal insertion
- Oracle-K recovery rate
- FN type proportions:
  - suppressed
  - weak visual
  - competitive

### Cross-family comparison metrics

- per-family basin tables
- per-family vision lift tables
- per-family FN mechanism tables
- pairwise family deltas on headline cohorts

## Controls

### Lexical and representation confounds

For family-native slot scoring, include representation-aware covariates where
applicable:

- numeric distance to active center
- numeric distance to GT-equivalent slot target
- character edit distance when serialized textual chunks are relevant
- token edit distance when serialized textual chunks are relevant
- digit-length match when relevant
- token count when relevant
- slot identity
- size bucket
- crowding bucket
- image condition
- model family

Interpretation rule:

- if continuity survives these controls in the family-native representation,
  treat that as evidence of genuine structured local preference rather than
  a formatting artifact

### Image control

At minimum:

- correct image
- swapped image

Interpretation rule:

- positive, repeated `Vision Lift` supports genuine visual grounding rather
  than a pure serialization prior

## Data Cohorts

Construct explicit manifests. Do not rely on ad hoc in-memory sampling.

Required cohort families:

1. `val_broad_random_headline`
2. `train_broad_random_supplemental`
3. `isolated_easy`
4. `crowded_same_class`
5. `model_mined_duplicate_prone`
6. `fn_low_recall_focus`

Prefer reuse of the existing duplication-collapse language:

- `pred_count`
- `max_desc_count`
- same-class overlap counts
- duplicate-like pair counts
- onset-oriented failure cues

## Artifact Contract

Write outputs under a new analysis root:

`output/analysis/coord_family_comparison_probe_<date>/`

Required contents:

- `report.md`
- `summary.json`
- `family_inventory.json`
- `family_contract_audit.md`
- `per_probe_scores.jsonl` or parquet
- `fn_recoverability.jsonl`
- `hard_cases.jsonl`
- cohort manifests
- aggregate metric tables
- family-vs-family comparison tables
- 1D basin plots
- 2D heatmaps
- recall-mechanism plots
- coefficient or comparison plots
- targeted reusable scripts and tests when needed

Artifact structure should separate:

- family inventory and contract audit
- cohort selection
- inference reproduction
- raw probe records
- raw FN recoverability records
- aggregated summaries
- figures
- report synthesis

No notebook-only state may be required for the final conclusions.

## Evidence Mapping

### Question 1

Main evidence:

- family-native continuity probe summaries
- matched pilot and large-scale continuity results

### Question 2

Main evidence:

- shared canonical comparison layer
- GT-centered basin metrics in pixel-space bbox semantics

### Question 3

Main evidence:

- image-control results
- Vision Lift

### Question 4

Main evidence:

- self-prefix hard-case landscapes
- prediction-centered versus GT-centered scans
- 2D heatmaps
- perturbation responses

### Question 5

Main evidence:

- teacher-forced GT support
- proposal-conditioned recoverability
- Oracle-K recoverability
- FN type tables

### Question 6

Main evidence:

- Questions 1 to 5 jointly
- dense-scene failure alignment
- decision-oriented family synthesis

## Verdict Rules

### Strongly supported

Use only if at least two independent evidence chains agree and the key controls
or interventions do not materially refute the claim.

### Partially supported

Use when the signal is real but limited to certain families, cohorts, or
controls.

### Not supported

Use when the expected signal fails to appear or the major evidence chains point
in the opposite direction.

### Inconclusive

Use when evidence is insufficient, controls remain unresolved, or evidence
chains materially disagree.

## Risks And Guardrails

- Do not collapse all families into one probe representation before auditing
  their true contracts.
- Do not let model-scale differences contaminate the headline 2B comparison.
- Do not overclaim that a family with stronger continuity is automatically
  healthier.
- Do not interpret all false negatives as visual failure; keep suppressed,
  weak-visual, and competitive failures distinct.
- Do not create unnecessary merged checkpoints when adapter inference already
  works through the current pipeline.
- Do not let the new family study silently rewrite the conclusions of the
  prior raw-text study.

## Assumptions

- the 2B headline family list above is fixed unless contract audit shows a
  family is not executable or not semantically comparable
- merged and adapter-backed inference are both valid study surfaces under the
  current HF runtime
- `val` remains the primary headline split and `train` remains supplemental
- existing raw-text continuity results are valid and can be treated as a prior
  anchor rather than re-litigated from scratch

## Completion Gate

This design is complete when:

- the spec is approved by the user
- the implementation plan can be written without unresolved scope ambiguity
- the family inventory and contract audit are explicitly part of the planned
  execution, not left as an implicit first step
