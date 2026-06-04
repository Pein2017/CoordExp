# Candidate-Field Cardinality Tomography Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` or
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Phase A diagnostic pipeline that tests whether
desc-conditioned pre-x1 coordinate posteriors expose enough distinct same-desc
instance modes before launching painting, ordering, or new training objectives.

**Architecture:** The pipeline uses a project-wise analysis subtree. Pure
schema, case-index, peak, taxonomy, and reporting modules are isolated from
GPU/model-facing posterior, scoring, decode, and attention modules. Artifact
validation is treated as a first-class stage, so every conclusion is traceable
to policy-consistent JSONL rows and an exhaustive case-index denominator.

**Tech Stack:** Python, PyYAML, JSON/JSONL, pytest, PyTorch/Qwen3-VL only
inside GPU-facing functions, tmux shell launchers for post-approval smoke/full
runs.

---

## Approval Gate

This document is a roadmap, not approval to run experiments.  Do not create
pipeline code, launch tmux, run model forwards, or start validation until the
user explicitly approves implementation.

Implementation and GPU execution for the pure-CE contrast were approved by the
user on 2026-06-03.  The approval is scoped to the Phase A candidate-field
diagnostic comparison below; it is not approval for production training,
painting interventions, sorted/random SFT, multiple-positive training, or
object-marginal training.

When implementation is approved, run from:

```bash
cd /data/CoordExp/.worktrees/fn-rescue-attention-probes
```

Do not stage or commit files unless the user asks.

## Pure-CE Contrast Execution Addendum

The first checkpoint comparison uses the completed ET-RMP-CE
`representative8192` run as the reference and launches the matching pure-CE
probe with the same dataset, sampling, shard, and peak policy.

Reference ET-RMP-CE artifact root:

```text
/data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_val200/candidate_field_cardinality_tomography_representative8192
```

Pure-CE checkpoint:

```text
/data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/compact_full_random_sft_bsz1_accum16_4epoch_tokenrows_v4_chatfix_max12k/compact-full-random-sft-bsz1-accum16-4epoch-tokenrows-v4-chatfix-max12k/v0-20260506-021259/checkpoint-3664
```

Pure-CE artifact root:

```text
/data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_purece/candidate_field_cardinality_tomography_representative8192
```

Execution steps:

- [x] Add `configs/analysis/candidate_field_cardinality_tomography/ckpt3664_purece_representative8192.yaml`.
- [x] Dry-run the launcher with the pure-CE config and confirm it resolves
  `case_index,probe_plan,validate`.
- [x] Launch the tmux run with 8 one-GPU shards; this is an analysis run, not
  production training.
- [x] Monitor shard logs and status until `merge,taxonomy,validate,report`
  finish.
- [x] Confirm `summary.json`, `phase_a_case_taxonomy_rows.jsonl`,
  `x1_candidate_field_rows.jsonl`, and `manifest.json` exist and validate.
- [x] Generate a direct ET-RMP-CE vs pure-CE comparison report over matching
  `representative8192` artifacts before drawing mechanism conclusions.

Run command:

```bash
CONFIG=/data/CoordExp/.worktrees/fn-rescue-attention-probes/configs/analysis/candidate_field_cardinality_tomography/ckpt3664_purece_representative8192.yaml \
SESSION=candidate_field_cardinality_purece_representative8192_ckpt3664 \
bash /data/CoordExp/.worktrees/fn-rescue-attention-probes/scripts/analysis/candidate_field_cardinality_tomography/launch_candidate_field_cardinality_tomography_tmux.sh
```

Launched session:

```text
candidate_field_cardinality_purece_representative8192_ckpt3664
```

Launch status at start:

- `probe_plan_summary.json`: `gpu_probe_planned_cases=8192`,
  `num_shards=8`;
- 8 shard processes alive;
- GPUs 0-7 assigned one shard each;
- observed GPU utilization after model load: 70-95%;
- not production training.

Completion status:

- all 8 shards exited 0;
- merged `x1_candidate_field_rows.jsonl`: 8192 rows;
- `phase_a_case_taxonomy_rows.jsonl`: 8192 rows;
- `manifest.json`: `validation_status=ok`;
- comparison report:
  `/data/CoordExp/outputs/analysis/autoreg_object_rollout/candidate_field_cardinality_comparisons/et_rmp_ce_vs_purece_representative8192/comparison_report.md`.

Status command:

```bash
python /data/CoordExp/.worktrees/fn-rescue-attention-probes/scripts/analysis/candidate_field_cardinality_tomography/status.py \
  --artifact-root /data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_purece/candidate_field_cardinality_tomography_representative8192 \
  --log-root /data/CoordExp/.worktrees/fn-rescue-attention-probes/logs/candidate_field_cardinality_purece_representative8192_ckpt3664
```

## Phase A2 Coordinated Dual-Checkpoint Analysis Addendum

The next approved step is to turn the two completed
`representative8192` runs into a deeper paired diagnostic.  This step should
prefer offline artifact analysis.  It may use GPU only if existing rows are
missing the posterior/top-k evidence needed for a conclusion.

Phase A2 artifact root:

```text
/data/CoordExp/outputs/analysis/autoreg_object_rollout/candidate_field_cardinality_comparisons/et_rmp_ce_vs_purece_representative8192/phase_a2_dual_checkpoint_analysis
```

Execution steps:

- [x] Update the design spec to define Phase A2 scope, required artifacts, and
  allowed interpretations.
- [x] Generate paired transition tables over exact matched rows:
  `both_A1`, `ET_only_A1`, `pure_only_A1`, `neither_A1`.
- [x] Compute paired delta metrics for peak count, multi-peak status, coverage
  fraction, valid peak count/share/mass, unmatched peak count/share/mass,
  target rank, and `p_gt_cond`.
- [x] Recompute approximate top-32 sensitivity for merge radius and mass
  threshold choices, labeled explicitly as `top32_approx`.
- [x] Produce same-desc bucket and desc-level tables that distinguish valid
  peak gain from unmatched peak gain.
- [x] Save a Markdown report with multiple plausible explanations and their
  evidence boundaries, without selecting an algorithmic intervention yet.
- [x] Generate an unmatched-peak manual-review sidecar with row-level JSONL,
  spreadsheet template, and a rendered gallery for visual triage.
- [x] Update the progress note with the artifact root, evidence scope, and
  stable findings.
- [x] Verify all Phase A2 JSON/Markdown/plot artifacts are readable.

Expected outputs:

```text
phase_a2_summary.json
phase_a2_report.md
plots/paired_a1_transitions.png
plots/coverage_delta_by_count.png
plots/valid_vs_unmatched_peak_delta.png
plots/sensitivity_grid_a1_rate.png
plots/plot_summary.json
unmatched_review/unmatched_peak_rows.jsonl
unmatched_review/unmatched_peak_summary.json
unmatched_review/unmatched_peak_review.md
unmatched_review/manual_review_template.csv
unmatched_review/gallery/index.md
```

Do not start another full GPU pass unless this offline analysis shows that the
current artifacts are insufficient.

## File Structure

Create this project-wise layout:

```text
src/analysis/candidate_field_cardinality_tomography/
  __init__.py
  config.py
  runner.py
  case_index.py
  probe_plan.py
  prefixes.py
  controls.py
  x1_candidate_field.py
  residual_row_scoring.py
  basin_attraction.py
  attention_components.py
  taxonomy.py
  artifacts.py
  merge.py
  report.py
  plots.py

scripts/analysis/candidate_field_cardinality_tomography/
  run.py
  launch_candidate_field_cardinality_tomography_tmux.sh

configs/analysis/candidate_field_cardinality_tomography/
  ckpt3664_smoke.yaml
  ckpt3664_trainval.yaml

tests/analysis/candidate_field_cardinality_tomography/
  conftest.py
  test_config.py
  test_artifacts.py
  test_case_index.py
  test_probe_plan.py
  test_prefixes.py
  test_controls.py
  test_x1_candidate_field.py
  test_residual_row_scoring.py
  test_basin_attraction.py
  test_attention_components.py
  test_taxonomy.py
  test_merge.py
  test_report.py
  test_runner_cli.py
```

Responsibility split:

| Module | Responsibility |
| --- | --- |
| `config.py` | Parse YAML, resolve paths, validate stage names and policy ids; no torch import. |
| `artifacts.py` | JSONL IO, schema versions, required fields, manifests, row counts, sha256, join validators. |
| `case_index.py` | Exhaustive train/val case universe, desc normalization, same-desc buckets, overlay membership. |
| `probe_plan.py` | Sampling frame, planned probe set, strata-balanced shard assignment, planned skip reasons. |
| `prefixes.py` | Prompt/prefix construction and prompt identity hashing. |
| `controls.py` | Negative-control rows, control pass/fail/missing matrix, headline-control gates. |
| `x1_candidate_field.py` | Coordinate posterior projection, conditional mass, local peaks, merge/sensitivity logic. |
| `residual_row_scoring.py` | Residual-row span scoring definitions and preference-violation metrics. |
| `basin_attraction.py` | Forced-x1 greedy continuation and teacher-forced tail margin metrics. |
| `attention_components.py` | Attention extraction and aggregation; only this module touches attention tensors. |
| `taxonomy.py` | A1/A2/A3a/A3b assignment, invalid flags, secondary flags, reason metric refs. |
| `merge.py` | Shard manifest validation, duplicate active-row policy, merged artifact creation. |
| `report.py` | Summary tables, denominator labels, Markdown report from artifacts only. |
| `plots.py` | Plot data preparation and `plots/plot_manifest.json`; no mechanism assignment. |
| `runner.py` | Stage dispatch, shard orchestration, validation order. |

## Task 1: Config Schema And Project Skeleton

**Files:**

- Create: `src/analysis/candidate_field_cardinality_tomography/__init__.py`
- Create: `src/analysis/candidate_field_cardinality_tomography/config.py`
- Create: `configs/analysis/candidate_field_cardinality_tomography/ckpt3664_smoke.yaml`
- Create: `configs/analysis/candidate_field_cardinality_tomography/ckpt3664_trainval.yaml`
- Create: `tests/analysis/candidate_field_cardinality_tomography/conftest.py`
- Create: `tests/analysis/candidate_field_cardinality_tomography/test_config.py`

- [ ] Define a `CandidateFieldConfig` dataclass with these required sections:

```python
@dataclass(frozen=True)
class CandidateFieldConfig:
    project_id: str
    artifact_root: Path
    checkpoint_path: Path
    train_jsonl: Path
    val_jsonl: Path
    fn_rescue_overlay_root: Path | None
    phase5_overlay_root: Path | None
    stages: tuple[str, ...]
    peak: PeakConfig
    sampling: SamplingConfig
    policies: PolicyConfig
```

- [ ] Add `load_config(path: str | Path) -> CandidateFieldConfig`.
- [ ] Reject unknown stages with `ValueError("unknown stage")`.
- [ ] Reject configs whose `project_id` is not
  `candidate_field_cardinality_tomography`.
- [ ] Keep checkpoint and JSONL paths explicit in both YAML configs.
- [ ] Write tests that assert smoke config parses, unknown stages fail, and no
  GPU/model libraries are imported by `config.py`.
- [ ] Run:

```bash
PYTHONPATH=/data/CoordExp/.worktrees/fn-rescue-attention-probes \
python -m pytest tests/analysis/candidate_field_cardinality_tomography/test_config.py -q
```

Expected: all tests in `test_config.py` pass.

## Task 2: Artifact Schema, Manifest, And Validators

**Files:**

- Create: `src/analysis/candidate_field_cardinality_tomography/artifacts.py`
- Create: `tests/analysis/candidate_field_cardinality_tomography/test_artifacts.py`

- [ ] Define artifact names exactly:

```python
CORE_JSONL = (
    "case_index.jsonl",
    "probe_plan.jsonl",
    "x1_candidate_field_rows.jsonl",
    "residual_row_score_rows.jsonl",
    "basin_attraction_rows.jsonl",
    "attention_component_rows.jsonl",
    "phase_a_case_taxonomy_rows.jsonl",
)
SUPPORT_FILES = (
    "case_index_summary.json",
    "controls_summary.json",
    "summary.json",
    "report.md",
    "resolved_config.yaml",
    "manifest.json",
    "plots/plot_manifest.json",
    "gallery/gallery_rows.jsonl",
)
```

- [ ] Implement JSON-safe row writing and reading.
- [ ] Implement `sha256_file(path)`, `write_manifest(root, files, metadata)`,
  and `validate_manifest(root)`.
- [ ] Implement row validators for unconditional base fields:

```text
schema_version
project_id
phase_id
run_id
checkpoint_id
case_id
case_index_row_id
split
pool_role
source_dataset_jsonl
dataset_manifest_id
dataset_manifest_sha256
fn_rescue_overlay_membership
```

- [ ] Implement conditional validators:

```text
probe_plan_row_id when probe-dependent
prefix_condition when prompt-dependent
prompt_instance_id when prompt-dependent
shard_id when sharded
```

- [ ] Implement table-specific validators for `source_line_idx`, `image_id`,
  `image_path`, `desc_id`, `same_desc_cluster_id`, `gt_idx`,
  `residual_gt_idx`, `region_instance_id`, `x1_peak_id`, and `run_id` whenever
  those keys are required by the table schema.
- [ ] Implement prompt/policy consistency checks for taxonomy metric refs.
- [ ] Implement `probe_plan.jsonl` subset validation: every GPU probe row must
  reference a sampled probe-plan row with matching `sampling_policy_id`,
  `sampling_policy_sha256`, `strata_key`, and planned shard identity.
- [ ] Tests must cover valid rows, missing required fields, duplicate active
  shard rows, missing manifest entries, and a taxonomy row that references
  mismatched prompt ids.
- [ ] Negative tests must fail when a taxonomy row references the right
  `case_id` but wrong `desc_id`, `gt_idx`, `x1_peak_id`, policy id, or
  `probe_plan_row_id`.
- [ ] Run:

```bash
PYTHONPATH=/data/CoordExp/.worktrees/fn-rescue-attention-probes \
python -m pytest tests/analysis/candidate_field_cardinality_tomography/test_artifacts.py -q
```

Expected: all artifact-validator tests pass.

## Task 3: Exhaustive Case Index

**Files:**

- Create: `src/analysis/candidate_field_cardinality_tomography/case_index.py`
- Create: `tests/analysis/candidate_field_cardinality_tomography/test_case_index.py`

- [ ] Implement canonical desc normalization with
  `desc_normalization_policy_id = lower_strip_collapse_ws_v1`:

```python
def canonical_desc(desc: str) -> str:
    return " ".join(desc.strip().lower().split())
```

- [ ] Preserve `desc_text_raw`, `desc_text_stripped`, and
  `desc_text_canonical`.
- [ ] Implement case indexing over train and val JSONL with explicit pool
  roles:

```text
headline_crowded: same_desc_gt_count_annotated >= 3
same_desc_count_1_control: same_desc_gt_count_annotated == 1
same_desc_count_2_control: same_desc_gt_count_annotated == 2
fn_rescue_overlay_linked: linked to overlay artifacts
overlay_only_control: overlay-linked but not headline crowded
```

- [ ] Preserve control slices as first-class `case_index.jsonl` rows with
  `case_id` and `case_index_row_id`.
- [ ] Emit crowding buckets `same_desc_3`, `same_desc_4_5`,
  `same_desc_6_plus`.
- [ ] Emit overlay membership fields without mixing overlay-only rows into the
  crowded headline denominator.
- [ ] Emit `case_index_summary.json` with exhaustive counters by split, desc,
  same-desc bucket, and overlay membership.
- [ ] Tests must use tiny JSONL fixtures with one count-1 control, one count-2
  control, one count-3 headline case, and one overlay-linked case.
- [ ] Tests must include desc strings that differ by case, leading/trailing
  whitespace, and repeated internal whitespace, proving the exact canonical
  grouping policy.
- [ ] Run:

```bash
PYTHONPATH=/data/CoordExp/.worktrees/fn-rescue-attention-probes \
python -m pytest tests/analysis/candidate_field_cardinality_tomography/test_case_index.py -q
```

Expected: case counters match the fixture exactly.

## Task 3A: Probe Plan And Control Matrix

**Files:**

- Create: `src/analysis/candidate_field_cardinality_tomography/probe_plan.py`
- Create: `src/analysis/candidate_field_cardinality_tomography/controls.py`
- Create: `tests/analysis/candidate_field_cardinality_tomography/test_probe_plan.py`
- Create: `tests/analysis/candidate_field_cardinality_tomography/test_controls.py`

- [ ] Build `probe_plan.jsonl` from `case_index.jsonl`; never sample directly
  from raw JSONL in GPU stages.
- [ ] Emit one planned row per indexed case with:

```text
probe_plan_row_id
case_id
case_index_row_id
probe_sampled
sampling_policy_id
sampling_policy_sha256
sampling_seed
sampling_weight
strata_key
planned_shard_id
planned_gpu_id
planned_stage_set
planned_status
planned_skip_reason
```

- [ ] Strata must include split, pool role, same-desc count bucket, desc
  frequency bucket, FN-rescue overlay membership, object-size bucket, and
  overlap bucket.
- [ ] Implement control rows and a control matrix for:

```text
same_desc_count_1_control
same_desc_count_2_control
wrong_desc_same_image
wrong_image_same_desc
x1_projection_collision_slice
gt_x1_jitter
competitor_x1_control
merge_radius_sensitivity
mass_floor_sensitivity
p_cond_vs_coord_vocab_mass
```

- [ ] Implement `control_status_by_type` with values `pass`, `fail`, and
  `missing`.
- [ ] Implement `headline_eligibility_status` gates:

```text
eligible
ineligible_low_coverage
ineligible_missing_controls
ineligible_partial_label_ambiguous
ineligible_sensitivity_unstable
```

- [ ] Tests must fail when a GPU probe row references no planned row, when a
  sampled row has a mismatched sampling-policy hash, when planned rows are
  duplicated, and when a required control is missing but a headline bucket is
  promoted.
- [ ] Run:

```bash
PYTHONPATH=/data/CoordExp/.worktrees/fn-rescue-attention-probes \
python -m pytest \
  tests/analysis/candidate_field_cardinality_tomography/test_probe_plan.py \
  tests/analysis/candidate_field_cardinality_tomography/test_controls.py \
  -q
```

Expected: probe-plan and control-gate fixtures pass.

## Task 4: Prefix And Prompt Identity

**Files:**

- Create: `src/analysis/candidate_field_cardinality_tomography/prefixes.py`
- Create: `tests/analysis/candidate_field_cardinality_tomography/test_prefixes.py`

- [ ] Implement prefix builders for:

```text
teacher_set_empty_prefix
teacher_prefix_at_boundary
self_rollout_prefix
```

- [ ] For every prompt-dependent row, produce:

```text
prompt_instance_id
prefix_condition
prefix_row_count
prefix_text_sha256
prompt_text_sha256
tokenizer_id
tokenizer_sha256
eos_token_id
pad_token_id
prompt_template_id
object_field_order
bbox_format
coord_surface
normalization
```

- [ ] Tests must assert that changing prefix rows changes
  `prefix_text_sha256`, changing desc changes `prompt_text_sha256`, and two
  modules using the same prompt instance receive identical identity fields.
- [ ] Reuse existing CoordExp compact detection row/token helpers where
  applicable instead of reimplementing detection syntax.  Local code may adapt
  those helpers only to add prompt identity hashes.
- [ ] Prefix tests must assert compact markers and coord tokens match existing
  helper behavior for row rendering and coordinate-token round trips.
- [ ] Run:

```bash
PYTHONPATH=/data/CoordExp/.worktrees/fn-rescue-attention-probes \
python -m pytest tests/analysis/candidate_field_cardinality_tomography/test_prefixes.py -q
```

Expected: prompt identity tests pass.

## Task 5: X1 Candidate Field Pure Logic

**Files:**

- Create: `src/analysis/candidate_field_cardinality_tomography/x1_candidate_field.py`
- Create: `tests/analysis/candidate_field_cardinality_tomography/test_x1_candidate_field.py`

- [ ] Implement coordinate-vocab projection over `<|coord_0|>` to
  `<|coord_999|>`.
- [ ] Compute `coord_vocab_mass`, `noncoord_vocab_mass`, and conditional
  `p_cond`.
- [ ] Implement peak extraction with defaults:

```text
absolute_mass_floor = 0.002
relative_floor = 0.10
primary_merge_radius = 24
gt_x1_neighborhood_radius = 24
raw_topk_k = 32
```

- [ ] Implement sensitivity radii `[16, 32]`, top-k `[16, 64]`, and mass-floor
  variants from config.
- [ ] Implement `x1_projection_collision` flags using x1 closeness plus y/center
  separation.
- [ ] Implement `coordinate_channel_low_mass_or_leakage` when
  `coord_vocab_mass < 0.05`.
- [ ] Tests must cover five GT x1 values collapsing into three merged peaks,
  five separated GT x1 values producing five covered peaks, low coordinate
  mass, projection collision, and sensitivity label flips.
- [ ] Tests must include `wrong_desc_same_image` and `wrong_image_same_desc`
  fixtures so generic objectness or desc-prior peaks cannot masquerade as
  desc-conditioned same-image candidate modes.
- [ ] Tests must include `gt_x1_jitter` and `competitor_x1_control` fixtures so
  forced-x1 basin stability is checked before A3 is promoted.
- [ ] Run:

```bash
PYTHONPATH=/data/CoordExp/.worktrees/fn-rescue-attention-probes \
python -m pytest tests/analysis/candidate_field_cardinality_tomography/test_x1_candidate_field.py -q
```

Expected: peak and coverage fixtures classify deterministically.

## Task 6: Residual-Row Logprob Scoring

**Files:**

- Create: `src/analysis/candidate_field_cardinality_tomography/residual_row_scoring.py`
- Create: `tests/analysis/candidate_field_cardinality_tomography/test_residual_row_scoring.py`

- [ ] Define row-score spans:

```text
desc_span
x1_token
bbox_tail_span
complete_row
eos_at_boundary
```

- [ ] Implement `row_score_policy_id = residual_row_mean_v1`.
- [ ] Compute `logp_row_mean`, `logp_desc_span_mean`, `logp_x1_token`,
  `logp_bbox_tail_mean`, `logp_eos_at_boundary`,
  `margin_best_residual_vs_teacher_next`,
  `margin_best_residual_vs_eos`, and `preference_violation`.
- [ ] Tests must use fixed logprob tensors to verify that length-normalized mean
  selects the expected best residual and that summed logprob is not used for
  `preference_violation`.
- [ ] Run:

```bash
PYTHONPATH=/data/CoordExp/.worktrees/fn-rescue-attention-probes \
python -m pytest tests/analysis/candidate_field_cardinality_tomography/test_residual_row_scoring.py -q
```

Expected: scoring spans and margins are exact on the fixture.

## Task 7: Basin Attraction

**Files:**

- Create: `src/analysis/candidate_field_cardinality_tomography/basin_attraction.py`
- Create: `tests/analysis/candidate_field_cardinality_tomography/test_basin_attraction.py`

- [ ] Represent forced-x1 attempts with:

```text
forced_x1_gt_idx
decode_attempt_id
greedy_box
greedy_target_iou
greedy_target_iou_rank
best_same_desc_competitor_iou
teacher_forced_tail_margin_target_vs_best_competitor
target_tail_beats_competitor_tail
decode_policy_id
max_new_tokens
```

- [ ] Implement pure aggregation that distinguishes
  `A3a_decode_basin_failure` from `A3b_tail_representation_failure`.
- [ ] Keep actual greedy generation in a GPU-facing function with lazy model
  imports.
- [ ] Tests must cover target tail wins but greedy binds competitor, target tail
  loses under teacher forcing, and missing tail score yields `A3_unresolved`.
- [ ] Run:

```bash
PYTHONPATH=/data/CoordExp/.worktrees/fn-rescue-attention-probes \
python -m pytest tests/analysis/candidate_field_cardinality_tomography/test_basin_attraction.py -q
```

Expected: A3 subtype fixtures classify deterministically.

## Task 8: Attention Components As Supporting Evidence

**Files:**

- Create: `src/analysis/candidate_field_cardinality_tomography/attention_components.py`
- Create: `tests/analysis/candidate_field_cardinality_tomography/test_attention_components.py`

- [ ] Implement row schema for:

```text
role
layer
head
aggregation_scope
region_kind
target_region_mass
same_desc_competitor_region_mass
background_region_mass
sink_or_special_token_mass
attention_aggregation_policy_id
```

- [ ] Enforce that attention rows cannot assign taxonomy buckets directly.
- [ ] Keep model/attention tensor imports lazy.
- [ ] Tests must verify region-mass normalization, background/sink fields, and
  that taxonomy refuses attention-only assignments.
- [ ] Run:

```bash
PYTHONPATH=/data/CoordExp/.worktrees/fn-rescue-attention-probes \
python -m pytest tests/analysis/candidate_field_cardinality_tomography/test_attention_components.py -q
```

Expected: attention rows validate as supporting evidence only.

## Task 9: Taxonomy And Counters

**Files:**

- Create: `src/analysis/candidate_field_cardinality_tomography/taxonomy.py`
- Create: `tests/analysis/candidate_field_cardinality_tomography/test_taxonomy.py`

- [ ] Implement primary buckets:

```text
A1_cardinality_collapse
A2_readout_or_coverage_failure
A3a_decode_basin_failure
A3b_tail_representation_failure
unassigned_or_inconclusive
invalid_or_uninterpretable
```

- [ ] Implement the continuation/readout gate before A3 assignment.  A3a/A3b
  are legal only when candidate x1 modes cover the target, residual-row
  continuation is viable, EOS is not dominant, and prompt/policy ids are
  consistent.  If EOS is dominant or all residual rows are suppressed, classify
  as `A2_readout_or_coverage_failure`.
- [ ] Implement invalid/guard flags:

```text
policy_mismatch
sensitivity_unstable
partial_label_ambiguous
coordinate_channel_low_mass_or_leakage
attention_only_unassigned
projection_collision_unresolved
control_failed_or_missing
```

- [ ] Implement precedence:

```text
policy/schema/parse mismatch
missing required evidence
coordinate-channel leakage
sensitivity instability
partial-label ambiguity
unresolved projection collision
A1
A2 continuation/readout/coverage failure
A3a/A3b basin/tail failure
unassigned
```

- [ ] Implement macro/micro counter preparation by split, desc, image,
  same-desc bucket, prefix condition, pool role, and overlay membership.
- [ ] Tests must include one A1, one A2, one A3a, one A3b, one
  projection-collision unresolved case, one partial-label ambiguous case, and
  one policy mismatch.
- [ ] Conflict tests must include:

```text
EOS high plus target tail loses -> A2_readout_or_coverage_failure
residual rows low plus greedy binds competitor -> A2_readout_or_coverage_failure
residual viable plus teacher tail loses -> A3b_tail_representation_failure
residual viable plus teacher tail wins but greedy fails -> A3a_decode_basin_failure
```
- [ ] Run:

```bash
PYTHONPATH=/data/CoordExp/.worktrees/fn-rescue-attention-probes \
python -m pytest tests/analysis/candidate_field_cardinality_tomography/test_taxonomy.py -q
```

Expected: every fixture receives the intended primary bucket and flags.

## Task 10: Reports, Plots, And Gallery

**Files:**

- Create: `src/analysis/candidate_field_cardinality_tomography/report.py`
- Create: `src/analysis/candidate_field_cardinality_tomography/plots.py`
- Create: `tests/analysis/candidate_field_cardinality_tomography/test_report.py`

- [ ] Generate `summary.json` with explicit counters:

```text
case_index_total_cases
gpu_probe_planned_cases
gpu_probe_attempted_cases
gpu_probe_valid_cases
taxonomy_assigned_cases
validation_status
headline_eligibility_status
control_status_by_type
```

- [ ] Generate `report.md` with scope labels `smoke`, `trainval`, or
  `stratified_full`.
- [ ] Generate `plots/plot_manifest.json` from artifact rows only.
- [ ] Generate `gallery/gallery_rows.jsonl` with row-keyed review metadata for
  unmatched peaks and representative taxonomy cases.
- [ ] Tests must verify the report refuses to headline a GPU-probed denominator
  as exhaustive and labels partial-label ambiguity explicitly.
- [ ] Tests must include a structurally valid run whose valid probe coverage is
  too low; `validation_status` should be `ok`, while
  `headline_eligibility_status` should block Phase B/C promotion.
- [ ] Tests must include a missing-control run that reports
  `ineligible_missing_controls`.
- [ ] Run:

```bash
PYTHONPATH=/data/CoordExp/.worktrees/fn-rescue-attention-probes \
python -m pytest tests/analysis/candidate_field_cardinality_tomography/test_report.py -q
```

Expected: report fixtures contain denominator labels and gallery metadata.

## Task 11: Runner, CLI, Configs, And Launch Script

**Files:**

- Create: `src/analysis/candidate_field_cardinality_tomography/runner.py`
- Create: `scripts/analysis/candidate_field_cardinality_tomography/run.py`
- Create: `scripts/analysis/candidate_field_cardinality_tomography/launch_candidate_field_cardinality_tomography_tmux.sh`
- Create: `tests/analysis/candidate_field_cardinality_tomography/test_runner_cli.py`

- [ ] Implement stage names:

```text
case_index
probe_plan
x1_candidate_field
residual_row_scoring
basin_attraction
attention_components
taxonomy
merge
report
gallery
validate
```

- [ ] Implement `--config`, `--stages`, `--shard-id`, `--num-shards`,
  `--dry-run`, and `--allow-overwrite`.
- [ ] `--dry-run` must resolve config and print planned artifacts without
  loading a model or writing GPU probe rows.
- [ ] The tmux launcher must support 8 independent GPU shards, but it must not
  describe the run as production training.
- [ ] Sharded GPU stages must write under:

```text
artifact_root/shards/shard_000/
artifact_root/shards/shard_001/
...
```

- [ ] Each shard must write `shard_manifest.json` with `shard_id`, `gpu_id`,
  `case_count`, `strata_histogram`, `stage_status`, `input_case_ids`, and
  `output_row_counts`.
- [ ] Implement `merge` as the only stage that writes authoritative merged
  GPU-probe tables at `artifact_root/*.jsonl`.
- [ ] Duplicate active rows after merge must fail validation unless they are
  exact byte-identical retry outputs and the merged manifest records the
  selected canonical row.
- [ ] Tests must assert dry-run does not create GPU rows, unknown stages fail,
  and overwrite protection prevents accidental artifact clobbering.
- [ ] Tests must create two fake shard directories, merge them, fail on
  duplicate active rows, and pass only when the merged manifest covers the
  planned probe set.
- [ ] Run:

```bash
PYTHONPATH=/data/CoordExp/.worktrees/fn-rescue-attention-probes \
python -m pytest tests/analysis/candidate_field_cardinality_tomography/test_runner_cli.py -q
```

Expected: CLI and overwrite tests pass.

## Task 12: Integration Verification Before Any GPU Run

**Files:**

- Modify only files created in Tasks 1-11.

- [ ] Run the full project test slice:

```bash
PYTHONPATH=/data/CoordExp/.worktrees/fn-rescue-attention-probes \
python -m pytest tests/analysis/candidate_field_cardinality_tomography -q
```

Expected: all candidate-field tests pass.

- [ ] Run py_compile:

```bash
PYTHONPATH=/data/CoordExp/.worktrees/fn-rescue-attention-probes \
python -m py_compile \
  src/analysis/candidate_field_cardinality_tomography/*.py \
  scripts/analysis/candidate_field_cardinality_tomography/run.py
```

Expected: command exits 0.

- [ ] Run config dry-run:

```bash
PYTHONPATH=/data/CoordExp/.worktrees/fn-rescue-attention-probes \
python scripts/analysis/candidate_field_cardinality_tomography/run.py \
  --config configs/analysis/candidate_field_cardinality_tomography/ckpt3664_smoke.yaml \
  --stages case_index \
  --dry-run
```

Expected: resolved config and planned artifacts are printed; no model is
loaded; no GPU artifacts are written.

## Task 13: Post-Approval Smoke And Full Execution

This task is executed only after user approval and after Task 12 passes.

**Files:**

- No new code files.  Use the implemented configs, script, and launcher.

- [ ] Run post-approval, no-model case-index/probe-plan materialization before
  any GPU stage:

```bash
PYTHONPATH=/data/CoordExp/.worktrees/fn-rescue-attention-probes \
python scripts/analysis/candidate_field_cardinality_tomography/run.py \
  --config configs/analysis/candidate_field_cardinality_tomography/ckpt3664_smoke.yaml \
  --stages case_index,probe_plan,validate
```

Expected: `case_index.jsonl`, `case_index_summary.json`,
`probe_plan.jsonl`, `resolved_config.yaml`, and `manifest.json` exist; no
model is loaded.

- [ ] Run smoke with one or more GPUs according to config:

```bash
ALLOW_OVERWRITE=1 \
CONFIG=/data/CoordExp/.worktrees/fn-rescue-attention-probes/configs/analysis/candidate_field_cardinality_tomography/ckpt3664_smoke.yaml \
SESSION=candidate_field_cardinality_smoke_ckpt3664 \
bash /data/CoordExp/.worktrees/fn-rescue-attention-probes/scripts/analysis/candidate_field_cardinality_tomography/launch_candidate_field_cardinality_tomography_tmux.sh
```

Expected: smoke writes all six core JSONL tables, `summary.json`,
`manifest.json`, `report.md`, `plots/plot_manifest.json`, and
`gallery/gallery_rows.jsonl`.

- [ ] Validate smoke artifacts:

```bash
PYTHONPATH=/data/CoordExp/.worktrees/fn-rescue-attention-probes \
python scripts/analysis/candidate_field_cardinality_tomography/run.py \
  --config configs/analysis/candidate_field_cardinality_tomography/ckpt3664_smoke.yaml \
  --stages validate
```

Expected: `summary.json.validation_status == "ok"`.

- [ ] Dry-run full train/val plan:

```bash
DRY_RUN=1 \
CONFIG=/data/CoordExp/.worktrees/fn-rescue-attention-probes/configs/analysis/candidate_field_cardinality_tomography/ckpt3664_trainval.yaml \
SESSION=candidate_field_cardinality_ckpt3664 \
bash /data/CoordExp/.worktrees/fn-rescue-attention-probes/scripts/analysis/candidate_field_cardinality_tomography/launch_candidate_field_cardinality_tomography_tmux.sh
```

Expected: shard plan shows strata-balanced use of available GPUs and does not
start production training.

- [ ] Launch full linked mechanism run:

```bash
CONFIG=/data/CoordExp/.worktrees/fn-rescue-attention-probes/configs/analysis/candidate_field_cardinality_tomography/ckpt3664_trainval.yaml \
SESSION=candidate_field_cardinality_ckpt3664 \
bash /data/CoordExp/.worktrees/fn-rescue-attention-probes/scripts/analysis/candidate_field_cardinality_tomography/launch_candidate_field_cardinality_tomography_tmux.sh
```

Expected: 8 GPUs are used as parallel analysis shards; no production training
job is launched.

- [ ] Merge and validate the full linked run:

```bash
PYTHONPATH=/data/CoordExp/.worktrees/fn-rescue-attention-probes \
python scripts/analysis/candidate_field_cardinality_tomography/run.py \
  --config configs/analysis/candidate_field_cardinality_tomography/ckpt3664_trainval.yaml \
  --stages merge,validate,report,gallery
```

Expected: merged JSONL tables join to `probe_plan.jsonl`; `summary.json` has
`validation_status == "ok"`; `headline_eligibility_status` is explicit even if
the run remains inconclusive.

## Review Checklist

- [ ] Design spec gates A1 away from projection-collision, low coordinate mass,
  sensitivity instability, and partial-label ambiguity.
- [ ] Attention is supporting evidence only.
- [ ] Every prompt-dependent artifact has prompt and policy identity fields.
- [ ] Case-index denominator is exhaustive and separate from GPU-probed rows.
- [ ] `probe_plan.jsonl` is the durable source of truth for planned GPU probes.
- [ ] Count-1/count-2 controls are first-class case-index rows, not side inputs.
- [ ] Desc normalization uses `lower_strip_collapse_ws_v1` consistently.
- [ ] `validation_status` and `headline_eligibility_status` are separate.
- [ ] Shard merge is explicit and validated before report/gallery.
- [ ] Smoke/full reports label scope and denominator.
- [ ] Phase B/C promotion rules depend on Phase A artifacts, not preference.
- [ ] No experiment starts before explicit user approval.
