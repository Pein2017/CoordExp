# Mixed-Objective SOTA Checkpoint Probe Design

Date: 2026-04-21
Status: proposed
Owner: Codex

## Goal

Design a focused follow-up probe study for the new adapter checkpoint:

- `/data/CoordExp/output_remote/stage1_2b/coco_bbox_max60-hard_ce_soft_ce_w1_gate/epoch_4-from-base-2B/v0-20260227-050057/checkpoint-1332`

This is not a fresh broad family-comparison study. It is a **delta study**
whose job is to explain why this mixed-objective 2B checkpoint reaches
approximately `bbox_AP ~= 0.39` on the remembered `eval200` benchmark and how
its behavior differs from the families already studied:

- `center_parameterization`
- `raw_text_xyxy_pure_ce`
- legacy weak `hard_soft_ce_2b`

The main output is an evidence-backed answer to:

- what this checkpoint is doing better,
- whether its gains come from healthier basin geometry, stronger recall
  realization, or simply a more stable output surface,
- and whether it should replace the prior weak `hard_soft_ce_2b` checkpoint as
  the headline mixed-objective 2B reference.

## Relationship To Prior Studies

This checkpoint probe extends, but does not replace, two archived studies:

1. `docs/superpowers/specs/2026-04-18-raw-text-coord-continuity-probe-design.md`
2. `docs/superpowers/specs/2026-04-20-coord-family-basin-and-recall-comparison-design.md`

Those studies established:

- raw-text continuity is real, visually modulated, and can form both good and
  bad basins,
- `center_parameterization` is the strongest current 2B family,
- `raw_text_xyxy_pure_ce` is the most interesting non-center mechanism case,
- and the previously archived `hard_soft_ce_2b` checkpoint is weak enough that
  its main problems look surface-level rather than subtle.

This new study asks whether the new mixed-objective adapter checkpoint changes
that picture.

## Known Prior Facts

The following facts are already known and do not need to be rediscovered
before the study can begin:

- this new checkpoint is a **runtime-loadable adapter** with
  `adapter_config.json` and `adapter_model.safetensors`
- its `base_model_name_or_path` points at
  `/data/CoordExp/model_cache_remote/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp`
- the historical remembered `~0.389` result was an `eval200`-scope benchmark,
  not a full-val result
- the corresponding merged checkpoint lineage is
  `output/stage1_2b/coco_bbox_max60-hard_ce_soft_ce_w1_gate_merged-1332`

The exact original training config and hyperparameters are useful context but
are **not required** to run this probe study. If they can be recovered cheaply,
include them as provenance notes only. They are not a blocker.

## Primary Questions

The final report must answer each question with one of:

- `strongly supported`
- `partially supported`
- `not supported`
- `inconclusive`

Questions:

1. Does the new mixed-objective SOTA checkpoint behave like a healthy,
   high-performing successor to the older weak `hard_soft_ce_2b` family?
2. Relative to `center_parameterization` and `raw_text_xyxy_pure_ce`, where do
   its gains primarily come from?
3. Does it exhibit stronger GT-centered local support than the old weak
   `hard_soft_ce_2b` checkpoint?
4. Does it avoid or reduce the invalid-geometry / unstable-surface failure
   patterns that characterized the old weak `hard_soft_ce_2b` checkpoint?
5. In low-recall settings, is it closer to:
   - `center_parameterization` style mostly-systematic FN,
   - `raw_text` style larger recoverable gap,
   - or a third distinct mixed-objective mechanism?
6. Should this checkpoint replace the archived weak `hard_soft_ce_2b`
   checkpoint as the canonical mixed-objective 2B comparison point in future
   research?

## Study Scope

### Included checkpoint

- `mixed_objective_sota_adapter`
  - `/data/CoordExp/output_remote/stage1_2b/coco_bbox_max60-hard_ce_soft_ce_w1_gate/epoch_4-from-base-2B/v0-20260227-050057/checkpoint-1332`

### Required reference models

- `center_parameterization`
  - `output/stage1_2b/coco_bbox_max60-1024-lvis_proxy-center_parameterization-ckpt_1564-merged`
- `raw_text_xyxy_pure_ce`
  - `output/stage1_2b/coco_bbox_max60-coco80-desc_first-1024-lvis_proxy-raw_text_xyxy-pure_ce/epoch_4-raw_text_xyxy-pure_ce-coco80-desc_first-1024-lvis_proxy-from-base-2B/v1-20260417-084341/checkpoint-552`
- `hard_soft_ce_2b` archived weak reference
  - `output/stage1_2b/coco_bbox_max60-hard_soft_ce-2b-merged`

### Explicitly excluded from headline synthesis

- `cxcywh_pure_ce`
- `cxcy_logw_logh_pure_ce`

These families already trail the stronger references and are not the decision
focus of this follow-up.

## Reuse-First Constraint

Reuse the existing analysis stack wherever possible. This study should be
assembled mostly by:

- extending the prior coord-family comparison assets,
- reusing raw-text continuity probe components where they remain relevant,
- and reusing inference/eval/oracle workflows that already exist.

Primary reuse targets:

- `scripts/run_infer.py`
- `src/infer/pipeline.py`
- `src/infer/engine.py`
- `scripts/evaluate_detection.py`
- `scripts/evaluate_oracle_k.py`
- `src/analysis/coord_family_contract_audit.py`
- `src/analysis/coord_family_basin_probe.py`
- `src/analysis/coord_family_recall_probe.py`
- `src/analysis/coord_family_comparison_report.py`
- `src/analysis/unmatched_proposal_verifier.py`

## Checkpoint Loading Policy

The new checkpoint must be loaded using **runtime adapter loading**, not by
creating a merged copy solely for this study.

Interpretation rules:

- treat this checkpoint as adapter-backed unless direct evidence shows
  otherwise
- prefer adapter shorthand through `infer.model_checkpoint` if the current
  stack supports it
- otherwise use the verified base-plus-adapter runtime path already established
  in the repository
- do not switch to vLLM for this checkpoint if the active adapter inference
  contract does not support it

## Method Layers

This study should stay focused and high-information rather than broad.

### Phase 0. Contract audit

Confirm and record:

- adapter vs merged load pattern
- base model resolution
- infer mode
- bbox format
- pred coord mode
- any special surface or parser behavior

Deliverable:

- a short contract note and machine-readable manifest row for the new
  checkpoint

### Phase 1. Matched `val200` detection/eval snapshot

Run a matched evaluation snapshot for the new checkpoint with the same framing
used in the prior family comparison.

This phase answers:

- is the remembered `~0.39` signal still reproducible under the current
  runtime path?
- where does the checkpoint rank relative to `center`, `raw_text`, and the old
  weak `hard_soft` reference?

Required outputs:

- `bbox_AP`
- `bbox_AP50`
- `bbox_AP75`
- `AR1`
- `F1@0.50`
- proposal totals and other standard summary rows

### Phase 2. Focused basin probe

Do **not** re-run the entire broad family basin study.

Instead, run a focused basin probe on:

- broad random subset
- crowded same-class subset
- duplicate-prone or hard repeated-object subset

Priority metrics:

- GT-centered local mass and local expected error
- slot-wise basin behavior
- wrong-anchor / pred-centered basin tendencies in hard repeated-object scenes

This phase should be explicitly comparative:

- new mixed-objective SOTA vs old weak `hard_soft_ce_2b`
- new mixed-objective SOTA vs `center_parameterization`
- new mixed-objective SOTA vs `raw_text_xyxy_pure_ce`

### Phase 3. Low-recall mechanism probe

Run the same three-part recall lane already used in the prior family study:

1. verifier-style teacher-forced scoring
2. Oracle-K repeated decoding
3. FN mechanism labeling

At minimum, report:

- baseline `recall_loc`
- Oracle-K `recall_loc`
- recoverable vs systematic FN
- `suppressed_fn_rate`
- `competitive_fn_rate`
- `weak_visual_fn_rate`

This is the most important phase for deciding whether the checkpoint is:

- mostly-systematic like `center`,
- decode-sensitive like `raw_text`,
- or a distinct mixed-objective pattern.

### Phase 4. Failure-family audit

Specifically inspect whether the new checkpoint still exhibits the old weak
`hard_soft` failure signatures:

- invalid geometry
- overflow to image edges
- unstable coordinate surfaces
- pathological repeated-object behavior

If those failures are absent or strongly reduced, that is itself a major
result.

## Data Scope

Recommended minimal scale:

- matched `val200` for the eval snapshot
- `val64` for recall mechanism work
- focused hard subsets for basin work

This should be enough to answer the main questions without reopening the full
family-comparison project.

## Deliverables

Artifacts should live under a new analysis directory, for example:

- `output/analysis/mixed-objective-sota-probe-2026-04-21/`

Required contents:

1. `report.md`
   - final narrative report
2. `summary.json`
   - aggregate machine-readable verdicts and metrics
3. focused basin outputs
   - JSONL and plots as needed
4. recall outputs
   - Oracle-K summaries, verifier summaries, mechanism tables
5. comparison tables
   - explicit comparisons against `center`, `raw_text`, and old weak
     `hard_soft`

Long-term archival copies should later be copied into `progress/diagnostics/`
if the study becomes decision-relevant.

## Decision Criteria

The final synthesis must distinguish among these possible outcomes:

### A. Strong mixed-objective successor

The checkpoint clearly supersedes the old weak `hard_soft_ce_2b` reference and
earns promotion to the canonical mixed-objective 2B slot.

### B. Better metric, same unhealthy mechanism

It achieves strong `eval200` mAP but still exhibits unstable or risky basin /
surface behavior that limits its value as a research baseline.

### C. `center` remains cleaner, but mixed-objective is now credible

It becomes a serious second-line baseline but does not displace `center` as the
cleanest strong family.

### D. `raw_text` remains more interesting mechanistically

It may be strong on metrics but still contributes less to the deeper mechanism
questions than `raw_text`.

### E. Inconclusive because runtime contract drifted

If current adapter runtime behavior no longer reproduces the remembered
performance surface, say so explicitly and separate runtime-contract issues from
model-quality conclusions.

## Working Assumptions

- the exact historical training hyperparameters may remain partially unknown
- this is acceptable as long as the runtime contract is audited and the study
  remains artifact-grounded
- the current question is behavioral and comparative, not a full training
  archaeology exercise
