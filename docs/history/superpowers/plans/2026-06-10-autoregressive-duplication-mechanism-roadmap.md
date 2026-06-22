# Autoregressive Duplication Mechanism Roadmap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a long-term, artifact-backed mechanism diagnosis pipeline for compact-full coordinate-token autoregressive duplication bursts, from Phase 0 onset ledgers through hidden-state/logit probes, attention/routing analysis, causal interventions, coordinate-token atlas synthesis, and final diagnosis.

**Architecture:** Keep Phase 0 CPU-only and artifact-first: it reads existing val128 rollout/eval/token artifacts and writes a joint onset ledger plus exact Phase 1 selected-window manifest. Phase 1 and later GPU phases consume that manifest without reselecting cases, separating hidden-state/logit trajectory, attention/routing, coordinate-token atlas, and causal intervention outputs into independent roots. The roadmap treats Phase 0 as pre-dessert and keeps the main course on hidden-state transitions, coordinate-slot basin attraction, and attention/visual-routing mechanisms.

**Tech Stack:** Python, pytest, YAML/JSONL artifacts, existing CoordExp compact-full rollout artifacts, existing duplicate geometry helpers, Qwen3-VL HF runtime for GPU probes, PyTorch, existing hidden-state/attention analysis surfaces, no upstream HF edits.

---

## Scope Lock

Worktree:

```text
/data/CoordExp/.worktrees/mechanistic-diagnosis-experiments
```

Primary diagnostic synthesis:

```text
/data/CoordExp/.worktrees/mechanistic-diagnosis-experiments/progress/diagnostics/2026-06-12_autoregressive_duplication_causal_chain_synthesis.md
```

The original June 10 mechanism charter was consolidated into that synthesis.

Phase 0 output family:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase0_joint_onset_ledger_<timestamp>/
```

Phase 1+ output family:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_<timestamp>/
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase2_attention_routing_<timestamp>/
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase3_interventions_<timestamp>/
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase4_coord_token_atlas_<timestamp>/
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/final_diagnosis_<timestamp>/
```

Do not:

- edit upstream Hugging Face model files;
- launch GPU probes before Phase 0 validates required cases;
- use guarded eval matches as the canonical model-prefix source;
- treat attention maps alone as causal proof;
- treat coordinate-token smoothness/discontinuity as a failure label without dynamic onset evidence;
- turn this roadmap into OpenSpec unless stable eval/training contracts change.

Dynamic exploration is explicitly welcome. If an intermediate artifact reveals
a promising and attractive path that may substantially change the final
mechanism picture, pause the current checklist long enough to inspect it,
record the reason, and update the next exploration steps. The roadmap is a
mission scaffold, not a ban on deeper dives when the evidence points somewhere
with higher explanatory leverage.

It is also acceptable to adjust the task sequence dynamically when a deeper
path appears to have more potential influence over the final picture behind
duplication bursts. In that case, prefer a short recorded update plus focused
follow-up probe over mechanically completing lower-leverage checklist items.

## Inputs

The Phase 0 config must encode these four rollout roots:

```text
/data/CoordExp/outputs/infer/loss_only_instance_enumeration_ablation_active_vs_none/compact_full_prefix_rollin_balance2_no_aligner_parent_val128_freegreedy_ckpt3668_val128_bsz8_temp0_rp1p10_max3084_chatfix_8gpu
/data/CoordExp/outputs/infer/loss_only_instance_enumeration_ablation_active_vs_none/compact_full_prefix_rollin_balance2_original_latest_aligner_parent_val128_freegreedy_ckpt1824_val128_bsz8_temp0_rp1p10_max3084_chatfix_8gpu
/data/CoordExp/outputs/infer/loss_only_instance_enumeration_ablation_active_vs_none/compact_full_prefix_rollin_balance2_loss_only_aux_latest_aligner_dora_val128_freegreedy_checkpoint-32-inference-clean_val128_bsz8_temp0_rp1p10_max3084_chatfix_4gpu
/data/CoordExp/outputs/infer/loss_only_instance_enumeration_ablation_active_vs_none/compact_full_prefix_rollin_balance2_loss_only_none_latest_aligner_dora_val128_freegreedy_ckpt32_val128_bsz8_temp0_rp1p10_max3084_chatfix_4gpu
```

The config must also encode these checkpoint paths:

```text
/data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/compact_full_fullobj_random_sft_bsz16_4epoch_tokenrows_v2/compact-full-fullobj-random-sft-bsz16-4epoch-tokenrows-v2/v1-20260601-062428/checkpoint-3668
/data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/compact_full_fullobj_random_sft_llm_aligner_lora_packed_bsz16_4epoch_tokenrows_v2/compact-full-fullobj-random-sft-llm-aligner-lora-packed-bsz16-4epoch-tokenrows-v2/v5-20260608-081447/checkpoint-1824
/data/CoordExp/outputs/stage1_2b/loss_only_instance_enumeration_ablation_active_vs_none/random_active_hp02_latest_aligner_dora_4096_128_no_newline_1epoch32_4gpu/loss-only-instance-enum-random-active-hp02-latest-aligner-dora-4096-128-no-newline-1epoch32-4gpu/v0-20260609-115128/checkpoint-32
```

## File Structure

Create the Phase 0 analysis surface:

- Create: `configs/analysis/autoregressive_duplication_mechanism/phase0_joint_val128.yaml`
- Create: `scripts/analysis/run_autoregressive_duplication_phase0.py`
- Create: `src/analysis/autoregressive_duplication_mechanism/__init__.py`
- Create: `src/analysis/autoregressive_duplication_mechanism/config.py`
- Create: `src/analysis/autoregressive_duplication_mechanism/io.py`
- Create: `src/analysis/autoregressive_duplication_mechanism/geometry.py`
- Create: `src/analysis/autoregressive_duplication_mechanism/components.py`
- Create: `src/analysis/autoregressive_duplication_mechanism/matches.py`
- Create: `src/analysis/autoregressive_duplication_mechanism/ledger.py`
- Create: `src/analysis/autoregressive_duplication_mechanism/selection.py`
- Create: `src/analysis/autoregressive_duplication_mechanism/reports.py`
- Create: `src/analysis/autoregressive_duplication_mechanism/runner.py`

Create Phase 0 tests:

- Create: `tests/analysis/autoregressive_duplication_mechanism/test_config.py`
- Create: `tests/analysis/autoregressive_duplication_mechanism/test_geometry_components.py`
- Create: `tests/analysis/autoregressive_duplication_mechanism/test_matches.py`
- Create: `tests/analysis/autoregressive_duplication_mechanism/test_ledger.py`
- Create: `tests/analysis/autoregressive_duplication_mechanism/test_selection_reports.py`

Extend or wrap existing Phase 1+ surfaces instead of duplicating working runtime code:

- Read/modify when needed: `scripts/analysis/run_autoreg_hidden_state_probe.py`
- Read/modify when needed: `src/analysis/autoreg_hidden_state_probe.py`
- Read/modify when needed: `scripts/analysis/run_autoreg_attention_evidence_routing.py`
- Read/modify when needed: `src/analysis/autoreg_attention_evidence_routing.py`
- Read/modify when needed: `scripts/analysis/run_hard_ce_coord_logit_locality.py`
- Read/modify when needed: `src/analysis/hard_ce_coord_logit_locality.py`
- Add configs under: `configs/analysis/autoregressive_duplication_mechanism/`
- Add targeted tests next to the reused surface tests, for example `tests/test_autoreg_hidden_state_probe.py` and `tests/test_autoreg_attention_evidence_routing.py`.

## Phase 0: Joint Onset Ledger And Selected Windows

**Purpose:** Build the CPU-only pre-dessert artifact that makes later GPU probes deterministic.

**Outputs:**

```text
phase0_joint_onset_ledger.jsonl
phase0_summary.json
phase0_component_stats.json
phase0_component_stats.md
phase0_required_cases.md
phase1_selected_windows.jsonl
```

### Task 0.1: Config And Input Audit

- [ ] Create `configs/analysis/autoregressive_duplication_mechanism/phase0_joint_val128.yaml` with four checkpoint records. Each record must include `checkpoint_label`, `checkpoint_path`, `rollout_root`, `aligner_tuned`, `aux_loss_kind`, `training_ordering`, `calibration_role`, and `decode_protocol_id`.

- [ ] Write `tests/analysis/autoregressive_duplication_mechanism/test_config.py` asserting:

```python
from pathlib import Path

from src.analysis.autoregressive_duplication_mechanism.config import load_phase0_config


def test_phase0_config_names_four_checkpoint_records():
    config = load_phase0_config(
        Path("configs/analysis/autoregressive_duplication_mechanism/phase0_joint_val128.yaml")
    )
    assert tuple(row.checkpoint_label for row in config.checkpoints) == (
        "no_aligner_parent_ckpt3668",
        "aligner_parent_ckpt1824",
        "aux_latest_ckpt32",
        "none_latest_ckpt32",
    )
    assert all(row.decode_protocol_id == "val128_freegreedy_temp0_rp1p10_max3084_no_separator" for row in config.checkpoints)
    assert "checkpoint_family" not in config.model_dump()
```

- [ ] Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider \
  tests/analysis/autoregressive_duplication_mechanism/test_config.py -q
```

Expected after implementation: tests pass.

### Task 0.2: Geometry And Component Detection

- [ ] Implement shared duplicate-like geometry in `geometry.py`: bbox IoU above threshold, or similar size with center distance inside local radius.

- [ ] Implement `same_desc_component` and `spatial_basin_component` in `components.py`. `same_desc_component` requires normalized desc equality; `spatial_basin_component` ignores desc and reports desc purity.

- [ ] Write tests in `test_geometry_components.py` that build three rows:

```python
rows = [
    {"desc": "person", "bbox_2d": [10, 10, 30, 30]},
    {"desc": "person", "bbox_2d": [11, 11, 31, 31]},
    {"desc": "dog", "bbox_2d": [12, 12, 32, 32]},
]
```

Expected:

```python
same_desc_largest.size == 2
spatial_basin_largest.size == 3
spatial_basin_largest.desc_purity < 1.0
```

- [ ] Add a tie-break test: when two same-desc components have equal size, primary selection uses earliest size-2 growth row, then earliest seed row, then larger spatial-basin overlap.

- [ ] Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider \
  tests/analysis/autoregressive_duplication_mechanism/test_geometry_components.py -q
```

Expected after implementation: tests pass.

### Task 0.3: Raw Match Context

- [ ] Implement `matches.py` to load `eval/matches.jsonl` at IoU `0.50` as canonical raw match context.

- [ ] Implement fallback local same-desc IoU50 matcher only when raw matches are absent; rows using it must record `match_source=fallback_local_iou50`.

- [ ] Write `test_matches.py` with a raw match row containing `matches`, `unmatched_pred_indices`, and `ignored_pred_indices`. Assert guarded matches are ignored for canonical prefix counters.

- [ ] Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider \
  tests/analysis/autoregressive_duplication_mechanism/test_matches.py -q
```

Expected after implementation: tests pass.

### Task 0.4: Ledger Rows

- [ ] Implement `ledger.py` to emit one row per `(record, checkpoint)` pair, including empty, invalid, and truncated predictions.

- [ ] Required row fields:

```text
record_idx
image_id
image_path
checkpoint_label
checkpoint_path
rollout_root
aligner_tuned
aux_loss_kind
training_ordering
calibration_role
decode_protocol_id
parse_status
pred_count
gt_count
match_source
pair_onset
component_onset
primary_burst_onset_row
same_desc_component
spatial_basin_component
component_growth_rows
prefix_matched_tp_count
prefix_clean_row_count
prefix_basin_seed_count
center_bucket_3x3
false_basin_candidate_score
false_basin_reviewed
exclusion_reason
```

- [ ] Write `test_ledger.py` asserting invalid rows are retained with null onset/component fields and non-null `exclusion_reason`.

- [ ] Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider \
  tests/analysis/autoregressive_duplication_mechanism/test_ledger.py -q
```

Expected after implementation: tests pass.

### Task 0.5: Selected Windows And Reports

- [ ] Implement `selection.py` to write `phase1_selected_windows.jsonl` from a fixed stratified panel:

```text
required_cases
early_primary_burst
late_primary_burst
same_desc_dominant
spatial_basin_before_desc
false_basin_candidate
contrast_case
```

- [ ] Each selected-window row must include:

```text
checkpoint_label
checkpoint_path
rollout_root
record_idx
image_id
window_start_row
window_end_row
primary_burst_onset_row
same_desc_component_id
spatial_basin_component_id
pred_token_trace_path
pred_confidence_path
gt_vs_pred_scored_path
raw_matches_path
```

- [ ] Implement reports in `reports.py`: `phase0_summary.json`, `phase0_component_stats.json`, `phase0_component_stats.md`, and `phase0_required_cases.md`.

- [ ] Write `test_selection_reports.py` asserting selected windows are `primary_burst_onset_row - 3` through `+ 8`, clipped to valid rows, and required records `54`, `33`, `47`, `82`, `96` are present when available.

- [ ] Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider \
  tests/analysis/autoregressive_duplication_mechanism/test_selection_reports.py -q
```

Expected after implementation: tests pass.

### Task 0.6: Phase 0 End-To-End Run

- [ ] Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/analysis/run_autoregressive_duplication_phase0.py \
  --config configs/analysis/autoregressive_duplication_mechanism/phase0_joint_val128.yaml
```

Expected:

- command exits `0`;
- writes the timestamped Phase 0 output root;
- `phase0_joint_onset_ledger.jsonl` has `128 * 4 = 512` rows unless the config explicitly scopes a smaller smoke;
- `phase1_selected_windows.jsonl` is non-empty;
- required cases `54`, `33`, `47`, `82`, and `96` appear in `phase0_required_cases.md`.

- [ ] Verify:

```bash
PYTHONDONTWRITEBYTECODE=1 python - <<'PY'
import json
from pathlib import Path
root = sorted(Path('/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism').glob('phase0_joint_onset_ledger_*'))[-1]
ledger = root / 'phase0_joint_onset_ledger.jsonl'
windows = root / 'phase1_selected_windows.jsonl'
summary = json.loads((root / 'phase0_summary.json').read_text())
assert ledger.exists(), ledger
assert windows.exists(), windows
assert summary['scope']['phase'] == 'phase0_joint_onset_ledger'
assert summary['counts']['ledger_rows'] >= 1
assert summary['counts']['selected_windows'] >= 1
print(root)
PY
```

## Phase 1: Hidden-State And Coordinate-Logit Trajectory

**Purpose:** Identify the forward-propagation transition around `primary_burst_onset_row`.

**Inputs:**

```text
phase1_selected_windows.jsonl
```

**Outputs:**

```text
hidden_state_rows.jsonl
coord_logit_rows.jsonl
row_similarity_rows.jsonl
stop_continue_margin_rows.jsonl
phase1_hidden_logit_summary.json
phase1_hidden_logit_report.md
```

### Task 1.1: Manifest Adapter

- [ ] Add a manifest adapter so `src/analysis/autoreg_hidden_state_probe.py` can consume Phase 0 selected-window rows without inventing a second case selector.

- [ ] Tests should assert the adapter preserves `checkpoint_label`, `record_idx`, `image_id`, `window_start_row`, `window_end_row`, and `primary_burst_onset_row`.

- [ ] Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider \
  tests/test_autoreg_hidden_state_probe.py -q
```

Expected after implementation: existing tests plus manifest-adapter tests pass.

### Task 1.2: Token Phase Capture

- [ ] Capture these phases for each selected row when token traces permit exact anchoring:

```text
row_start
desc_end
box_start/pre_x1
post_x1/pre_y1
post_y1/pre_x2
post_x2/pre_y2
post_y2/row_boundary
next_row_or_stop_decision
```

- [ ] First hidden-state pass samples layers `{0, 4, 8, 12, 16, 20, 24, 28, final}`.

- [ ] Default similarity is cosine on normalized hidden states.

### Task 1.3: Coordinate-Internal Logit Readout

- [ ] Report coordinate-only `p_cond` top-k and basin mass by slot.

- [ ] Because strict schema/type loss already makes coordinate positions stable, treat full-vocab coordinate-token mass as a sanity/regression check.

- [ ] Basin mass windows:

```text
+/-4 bins around component median coordinate
+/-8 bins around component median coordinate
+/-16 bins around component median coordinate
+/-4 bins around source-row coordinate
+/-8 bins around source-row coordinate
+/-16 bins around source-row coordinate
```

- [ ] Report stop/EOS versus object-start margin at row-boundary phases.

### Task 1.4: Phase 1 Smoke And Full Run

- [ ] Create smoke and full configs:

```text
configs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_smoke.yaml
configs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_val128.yaml
```

- [ ] Smoke run command:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/analysis/run_autoreg_hidden_state_probe.py \
  --config configs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_smoke.yaml
```

Expected: no-op or tiny GPU run exits `0`, writes at least one summary row, and records skipped-token-anchor reasons explicitly.

## Phase 2: Attention And Visual-Routing Evidence

**Purpose:** Measure routing to duplicate basins, GT objects, empty controls, and candidate sinks after Phase 1 identifies the important token phases.

**Inputs:**

```text
phase1_selected_windows.jsonl
phase1_hidden_logit_summary.json
```

**Outputs:**

```text
attention_region_rows.jsonl
attention_sink_rows.jsonl
attention_maps/
phase2_attention_summary.json
phase2_attention_report.md
```

### Task 2.1: Region Definitions

- [ ] Define regions per selected window:

```text
duplicate_basin
same_desc_component_envelope
spatial_basin_component_envelope
matched_gt_regions
empty_top_left_control
empty_middle_left_control
rest_of_image
```

- [ ] Region metadata must include normalized bbox, pixel bbox, source component id, and whether the region is manual-review-gated.

### Task 2.2: Attention Readout

- [ ] Reuse or extend `src/analysis/autoreg_attention_evidence_routing.py` to consume Phase 0 selected-window rows.

- [ ] Treat attention as routing evidence only. Reports must not claim attention causality before Phase 3 perturbations.

- [ ] Run existing and new attention tests:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider \
  tests/test_autoreg_attention_evidence_routing.py -q
```

## Phase 3: Causal Interventions

**Purpose:** Test whether candidate regions or coordinate basins actually change continuation behavior.

**Outputs:**

```text
intervention_plan.jsonl
no_op_replay_rows.jsonl
image_region_mask_rows.jsonl
coord_basin_suppression_rows.jsonl
separator_boundary_rows.jsonl
phase3_intervention_summary.json
phase3_intervention_report.md
```

### Task 3.1: No-Op Replay Parity

- [ ] First intervention lane is no-op replay parity. If no-op replay does not reproduce selected continuations under deterministic decode, stop and fix replay before interpreting any intervention.

### Task 3.2: Image-Region Masking

- [ ] Run duplicate-basin image-region masking after no-op parity succeeds.

- [ ] Interpret as visual-region dependence only if masking changes the continuation relative to no-op controls.

### Task 3.3: Coordinate-Basin Suppression

- [ ] Run coordinate-basin logit suppression only after Phase 1 shows repeated basin mass before or at `x1`/`y1`.

- [ ] Suppression is diagnostic, not a production fix.

### Task 3.4: Separator/Newline Boundary Diagnostic

- [ ] Run row separator/newline only after hidden/logit and visual-region lanes have produced interpretable results, because separator changes can hide several mechanisms at once.

## Phase 4: Coordinate-Token Atlas And Special-Token Synthesis

**Purpose:** Connect dynamic coordinate-slot basin attraction to the static geometry, coverage, and norms of `<|coord_0|>` through `<|coord_999|>`.

**Outputs:**

```text
coord_token_frequency_by_slot.json
coord_token_geometry_rows.jsonl
coord_token_nearest_neighbors.jsonl
coord_token_smoothness.json
duplicate_basin_coord_correlation.json
phase4_coord_token_atlas_report.md
```

### Task 4.1: Slot-Separated Coverage

- [ ] Build training-label frequency histograms separated by `x1`, `y1`, `x2`, and `y2`.

- [ ] Build rollout duplicate-bin histograms separated by slot and checkpoint label.

### Task 4.2: Token Geometry

- [ ] For each checkpoint, compute input-row norm, output-row norm, effective post-offset row norm when applicable, nearest neighbors, locality, and smoothness/discontinuity scores for coordinate tokens.

- [ ] Compare ordinary vocab and other special-token norms/biases only as sanity controls. The main analysis remains coordinate-internal.

### Task 4.3: Dynamic Alignment

- [ ] Correlate duplicate-basin coordinates with token coverage and geometry.

- [ ] Interpret static smoothness/discontinuity as suspicious only when it aligns with Phase 1 onset-time coordinate-logit attraction.

## Phase 5: Final Mechanism Diagnosis

**Purpose:** Produce the final mechanism report after the probes and interventions can distinguish direct trigger from amplifiers.

**Outputs:**

```text
final_diagnosis_report.md
final_diagnosis_summary.json
case_gallery/
```

Required report sections:

```text
Executive conclusion
Symptom taxonomy
Evidence table
Probe results
Root-cause assessment
Fix candidates
Verification plan
```

Interpretation rules:

- Direct trigger must be tied to the earliest supported transition in hidden states/logits/routing/intervention behavior.
- Attention maps alone are not causal proof.
- Aggregate AP, duplicate suppression counts, and guarded metrics are context, not mechanism proof.
- Pure-text coordinates, coordinate-token repair, visual-routing fixes, stop/continue fixes, and template/boundary fixes must be ranked by evidence, not prior suspicion.

## Verification Ladder

Run these in order as phases are implemented:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider \
  tests/analysis/autoregressive_duplication_mechanism -q
```

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider \
  tests/test_autoreg_hidden_state_probe.py \
  tests/test_autoreg_attention_evidence_routing.py \
  tests/test_hard_ce_coord_logit_locality.py -q
```

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/analysis/run_autoregressive_duplication_phase0.py \
  --config configs/analysis/autoregressive_duplication_mechanism/phase0_joint_val128.yaml
```

GPU smoke commands for Phase 1+ should run only after the Phase 0 output root contains a valid `phase1_selected_windows.jsonl`.

## Suggested Commit Slices

- Commit 1: Phase 0 config, CPU modules, unit tests.
- Commit 2: Phase 0 reports and end-to-end val128 artifact run.
- Commit 3: Phase 1 hidden-state/logit manifest adapter and smoke config.
- Commit 4: Phase 2 attention/routing manifest adapter and region definitions.
- Commit 5: Phase 3 no-op replay and image-region masking intervention plan.
- Commit 6: Phase 4 coordinate-token atlas and dynamic alignment.
- Commit 7: Phase 5 final diagnosis report skeleton and synthesis tooling.

## Stop Conditions

Stop Phase 0 when:

- required cases render in `phase0_required_cases.md`;
- `phase1_selected_windows.jsonl` exists and includes deterministic row windows;
- invalid/empty/truncated rollouts are retained as ledger rows;
- no GPU code is needed to produce the artifacts.

Stop Phase 1 when:

- hidden-state/logit rows identify whether the transition appears before `x1`, at `x1`, after `x1`, at row boundary, or only after repeated rows;
- coordinate-slot basin mass is available for each selected onset window;
- stop/continue margin is available or explicitly marked unavailable.

Stop Phase 2 when:

- attention/routing rows map selected token phases to duplicate basin, GT, empty controls, and rest-of-image regions;
- the report clearly labels attention as routing evidence, not causal proof.

Stop Phase 3 when:

- no-op replay parity is validated;
- image-region masking has interpretable changed-or-unchanged continuations;
- coordinate suppression is run only for cases where Phase 1 justified it.

Stop the mission when:

- the final report can rank coordinate-token embedding/head geometry, coordinate-slot basin attraction, hidden-state recurrence, visual-routing/sink behavior, stop/continue calibration, and boundary/template looping as direct trigger, amplifier, negative, or unresolved.
