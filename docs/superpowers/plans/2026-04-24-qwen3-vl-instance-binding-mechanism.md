# Qwen3-VL Instance Binding Mechanism Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a first-pass mechanism study for one fixed CoordExp / Qwen3-VL coord-token checkpoint, answering whether same-desc instance binding is present before `x1` or hardens mainly at `x1/y1`.

**Architecture:** Keep the study isolated under a new analysis namespace and a single artifact root. Reuse CoordExp checkpoint resolution, CoordJSON formatting, teacher-forced scoring, and existing duplication/case-mining helpers where they are correct, but add a narrow coord-token hidden-state/probe/patching harness rather than modifying the inference pipeline or adding CLI flags.

**Tech Stack:** Python, PyTorch, Hugging Face Qwen3-VL runtime, existing CoordExp analysis utilities, YAML configs, JSONL artifacts, pytest, and `conda run -n ms python ...` verification.

---

## Current Execution Status

Status as of the first-pass GPU loop:

- Worktree: `/data/CoordExp/.worktrees/qwen3-vl-instance-binding`
- Artifact root: `/data/CoordExp/output/analysis/qwen3-vl-instance-binding-mechanism-20260424`
- Fixed checkpoint: `/data/CoordExp/output_remote/stage1_2b/coco_bbox_max60-hard_ce_soft_ce_w1_gate/epoch_4-from-base-2B/v0-20260227-050057/checkpoint-1332-merged-full`
- Curated subset: `64` cases, with `56` priority same-desc cases and `8` sparse controls
- Rollout execution: 8 tmux shards, batch size `8`, all `64` rows merged
- Donor patching execution: 8 tmux shards, `56` donor-eligible repeated-object cases and `224` span rows merged
- Current conclusion: `converged_first_pass_mixed_soft_pre_x1_coordinate_hardening`

Decision summary:

- Pre-`x1` binding is not absent, but it is weak/partial.
- The pre-`x1` coordinate distribution remains multi-modal in same-desc scenes.
- `x1/y1` is still the hard disambiguation boundary on the difficult cases.
- Schema-context states are causally important: attenuation changes margins, and
  same-case donor copying moves mass toward the donor far more than copying
  current-desc or previous-geometry spans.

Key result artifacts:

- `/data/CoordExp/output/analysis/qwen3-vl-instance-binding-mechanism-20260424/report/report.md`
- `/data/CoordExp/output/analysis/qwen3-vl-instance-binding-mechanism-20260424/report/summary.json`
- `/data/CoordExp/output/analysis/qwen3-vl-instance-binding-mechanism-20260424/rollout_failure_split/summary.json`
- `/data/CoordExp/output/analysis/qwen3-vl-instance-binding-mechanism-20260424/donor_patching/summary.json`

Remaining highest-value controls:

- randomized donor patching to separate identity transfer from positional disruption
- wrong-image schema-context controls
- a wider same-desc rollout split if the conclusion will be promoted to a durable diagnostic note

---

## Scope

Implementation is active in:

`/data/CoordExp/.worktrees/qwen3-vl-instance-binding`

The worktree is source-only. Resolve heavyweight checkpoints, prepared datasets,
and durable outputs through the shared CoordExp root `/data/CoordExp`, not
through worktree-relative paths.

The fixed checkpoint is:

`/data/CoordExp/output_remote/stage1_2b/coco_bbox_max60-hard_ce_soft_ce_w1_gate/epoch_4-from-base-2B/v0-20260227-050057/checkpoint-1332-merged-full`

The primary dataset is:

`/data/CoordExp/public_data/coco/rescale_32_1024_bbox_max60/val.coord.jsonl`

The optional LVIS-proxy case source is:

`/data/CoordExp/public_data/coco/rescale_32_1024_bbox_max60_lvis_proxy/val.coord.jsonl`

Treat it as a merged coord-token model unless the checkpoint audit proves
otherwise. The directory is expected to contain `coord_tokens.json`.

The study is not a benchmark sweep. It should produce a small, decision-facing
mechanism report with evidence, interpretation, and open uncertainty separated.

Execution should assume 8 available GPUs for the real first-pass study. GPU-heavy
stages should shard by case id or stage cell across devices `0..7`. Any
generation or short controlled continuation used for continuation-flip evidence
must use per-GPU generation batch size `8`, with an explicit config override
only if the smoke run shows memory pressure.

## File Structure

### New files

- Create: `src/analysis/qwen3_vl_instance_binding.py`
  - Shared contract layer for path canonicalization, checkpoint audit,
    repeated-desc case mining, token-position inventory validation, sharding,
    pre-x1 distribution summaries, mechanism classification, and stage
    dispatch. Keep the first implementation compact here; split into focused
    submodules only when GPU/model-forward logic makes the file hard to reason
    about.
- Create: `scripts/analysis/run_qwen3_vl_instance_binding_study.py`
  - Config-first entrypoint for individual stages.
- Create: `configs/analysis/qwen3_vl_instance_binding/default.yaml`
  - Full first-pass config.
- Create: `configs/analysis/qwen3_vl_instance_binding/smoke.yaml`
  - CPU/lightweight validation config with tiny fixtures and no model load.
- Create: `tests/test_qwen3_vl_instance_binding_study.py`
  - CPU contract tests for path resolution, audit, case selection, token
    anchors, candidate-distribution alignment, sharding, mechanism
    classification, config parsing, and stage artifact writes.

### Existing files to inspect and reuse

- Reuse: `src/infer/checkpoints.py`
  - Checkpoint resolution and adapter-vs-merged audit patterns.
- Reuse: `src/infer/backends.py`
  - HF generation/logit extraction patterns.
- Reuse: `src/utils/assistant_json.py`
  - CoordJSON serialization rules for bare `<|coord_*|>` tokens.
- Reuse: `src/analysis/unmatched_proposal_verifier.py`
  - Teacher-forced scorer and hidden-state extraction pattern.
- Reuse: `src/analysis/duplication_collapse_analysis.py`
  - Existing duplicate-like mining concepts.
- Reuse: `src/analysis/coord_family_basin_probe.py`
  - Coord-family basin metric conventions.
- Reuse as structural reference only: raw-text mechanism modules under
  `src/analysis/raw_text_*`.
  - Do not inherit digit-token assumptions.

### Existing files to modify only if necessary

- Modify: `src/analysis/unmatched_proposal_verifier.py`
  - Only if the existing scorer cannot expose the needed per-layer hidden states
    and logits without copy-pasting model preparation logic.
- Modify: `tests/test_unmatched_proposal_verifier_scorer.py`
  - Only with a targeted regression if the scorer API changes.

## Artifact Root

Use:

`/data/CoordExp/output/analysis/qwen3-vl-instance-binding-mechanism-20260424/`

Expected outputs:

- `config.resolved.yaml`
- `audit/checkpoint_audit.json`
- `cases/selected_cases.jsonl`
- `cases/summary.json`
- `cases/shard_<idx>-of-<n>.json`
- `token_position_inventory.jsonl`
- `hidden_states/manifest.json`
- `probe_results.json`
- `pre_x1_multimodality.jsonl`
- `patching_results.jsonl`
- `good_bad_case_panels.jsonl`
- `summary.json`
- `report.md`

Promote durable copies to `progress/diagnostics/` only after actual results
exist and have been reviewed.

---

## Task 1: Prepare the Worktree and Contract Audit

**Files:**

- Create: `configs/analysis/qwen3_vl_instance_binding/default.yaml`
- Create: `configs/analysis/qwen3_vl_instance_binding/smoke.yaml`
- Create: `scripts/analysis/run_qwen3_vl_instance_binding_study.py`
- Create: `src/analysis/qwen3_vl_instance_binding.py`
- Test: `tests/test_qwen3_vl_instance_binding_study.py`

- [x] **Step 1: Verify the implementation worktree and shared roots**

Verified:

```bash
git worktree list
test -f /data/CoordExp/output_remote/stage1_2b/coco_bbox_max60-hard_ce_soft_ce_w1_gate/epoch_4-from-base-2B/v0-20260227-050057/checkpoint-1332-merged-full/coord_tokens.json
test -f /data/CoordExp/public_data/coco/rescale_32_1024_bbox_max60/val.coord.jsonl
```

Expected:

- The active worktree exists at `.worktrees/qwen3-vl-instance-binding`.
- The branch name is `codex/qwen3-vl-instance-binding`.
- The real checkpoint/data paths are under `/data/CoordExp`, not the worktree.

- [ ] **Step 2: Write the token/contract tests first**

Create `tests/test_qwen3_vl_instance_binding_study.py` with tests for:

- checkpoint audit detects `coord_tokens.json`
- coord-token ids map `<|coord_0|>` through `<|coord_999|>`
- role positions include all eight required anchors
- position inventory rejects missing `pre_x1` or duplicate anchors

Example assertions:

```python
from pathlib import Path

from src.analysis.qwen3_vl_instance_binding_tokens import (
    REQUIRED_POSITION_ROLES,
    audit_checkpoint_surface,
    validate_position_inventory,
)


def test_required_position_roles_are_complete() -> None:
    assert REQUIRED_POSITION_ROLES == (
        "desc_end",
        "desc_closing_quote",
        "field_delimiter",
        "bbox_key",
        "bbox_open_bracket",
        "pre_x1",
        "post_x1",
        "post_y1",
    )


def test_audit_checkpoint_surface_finds_coord_tokens(tmp_path: Path) -> None:
    ckpt = tmp_path / "checkpoint"
    ckpt.mkdir()
    (ckpt / "coord_tokens.json").write_text("{}", encoding="utf-8")
    (ckpt / "config.json").write_text("{}", encoding="utf-8")

    audit = audit_checkpoint_surface(ckpt)

    assert audit["surface"] == "coord_tokens"
    assert audit["has_coord_tokens_json"] is True


def test_validate_position_inventory_rejects_missing_pre_x1() -> None:
    rows = [
        {"case_uid": "c1", "role": "desc_end", "token_index": 10},
        {"case_uid": "c1", "role": "post_x1", "token_index": 20},
    ]

    try:
        validate_position_inventory(rows)
    except ValueError as exc:
        assert "pre_x1" in str(exc)
    else:
        raise AssertionError("expected missing pre_x1 to fail")
```

- [ ] **Step 3: Run the tests and confirm they fail**

Run:

```bash
conda run -n ms python -m pytest tests/test_qwen3_vl_instance_binding_study.py -q
```

Expected:

- FAIL with module import errors or missing functions.

- [ ] **Step 4: Implement the minimal token/contract module**

Implement `src/analysis/qwen3_vl_instance_binding.py` with:

- `REQUIRED_POSITION_ROLES`
- `audit_checkpoint_surface(checkpoint_path: Path) -> dict`
- `validate_position_inventory(rows: Iterable[Mapping[str, object]]) -> None`
- helper functions for coord-token id lookup to be wired to the tokenizer later

Keep this module lightweight and CPU-testable.

- [ ] **Step 5: Add YAML configs**

Create `configs/analysis/qwen3_vl_instance_binding/default.yaml`:

```yaml
run:
  name: qwen3-vl-instance-binding-mechanism-20260424
  output_dir: output/analysis/qwen3-vl-instance-binding-mechanism-20260424
  stages:
    - audit
    - case_bank
    - positions
    - hidden_states
    - probe
    - multimodality
    - merge_multimodality
    - patching
    - merge_patching
    - report

model:
  checkpoint: /data/CoordExp/output_remote/stage1_2b/coco_bbox_max60-hard_ce_soft_ce_w1_gate/epoch_4-from-base-2B/v0-20260227-050057/checkpoint-1332-merged-full
  expected_surface: coord_tokens
  device: cuda
  dtype: bfloat16

execution:
  gpu_ids: [0, 1, 2, 3, 4, 5, 6, 7]
  shard_count: 8
  per_gpu_generation_batch_size: 8
  per_gpu_teacher_forced_batch_size: 8
  artifact_root: /data/CoordExp/output/analysis/qwen3-vl-instance-binding-mechanism-20260424

data:
  candidate_jsonl:
    - /data/CoordExp/public_data/coco/rescale_32_1024_bbox_max60_lvis_proxy/val.coord.jsonl
    - /data/CoordExp/public_data/coco/rescale_32_1024_bbox_max60/val.coord.jsonl
  exact_desc_priority:
    - book
    - person
    - chair
    - baseball bat
    - bowl
    - sheep

case_bank:
  sparse_controls: 12
  healthy_same_desc: 24
  hard_same_desc: 24
  duplicate_collapse: 12
  min_candidate_count: 2
  max_cases_first_pass: 72

positions:
  roles:
    - desc_end
    - desc_closing_quote
    - field_delimiter
    - bbox_key
    - bbox_open_bracket
    - pre_x1
    - post_x1
    - post_y1
  layer_policy:
    anchors: [early, middle, late, final]
    include_last_n_layers: 6

probe:
  candidate_sets: [remaining_same_desc, full_same_desc]
  box_features: center_log_size
  controls: [order_next_unseen, left_to_right, top_to_bottom, token_only_post_coord]
  split_key: image
  bootstrap_unit: image

multimodality:
  coord_window_radius: 2
  top_k: 10
  slots: [x1, x2]

patching:
  max_sites: 16
  layer_bands: [0.25, 0.50, 0.75, 0.90]
  continuation_batch_size: 8
  spans:
    - current_desc
    - current_schema
    - previous_geometry
    - previous_x1y1
```

Create `configs/analysis/qwen3_vl_instance_binding/smoke.yaml` with the same
shape but `device: cpu`, tiny fixture paths, `gpu_ids: []`,
`per_gpu_generation_batch_size: 1`, and `max_cases_first_pass: 3`.

- [ ] **Step 6: Add the stage runner skeleton**

Create `scripts/analysis/run_qwen3_vl_instance_binding_study.py` with a
config-first CLI that accepts:

```bash
python scripts/analysis/run_qwen3_vl_instance_binding_study.py \
  --config configs/analysis/qwen3_vl_instance_binding/default.yaml \
  --stage audit \
  --shard-index 0 \
  --num-shards 1
```

The runner should dispatch by `--stage`; unsupported stages must fail with a
clear error. `--shard-index` and `--num-shards` should apply to case-level
stages only. CPU stages can accept the flags and no-op them for consistency.

- [ ] **Step 7: Verify the CPU contract tests pass**

Run:

```bash
conda run -n ms python -m pytest tests/test_qwen3_vl_instance_binding_study.py -q
```

Expected:

- PASS.

Commit:

```bash
git add configs/analysis/qwen3_vl_instance_binding scripts/analysis/run_qwen3_vl_instance_binding_study.py src/analysis/qwen3_vl_instance_binding.py tests/test_qwen3_vl_instance_binding_study.py
git commit -m "analysis: scaffold qwen3 vl instance binding study"
```

---

## Task 2: Build the Case Bank

**Files:**

- Create: `src/analysis/qwen3_vl_instance_binding_cases.py`
- Test: `tests/test_qwen3_vl_instance_binding_cases.py`

- [ ] **Step 1: Write case-bank tests**

Cover:

- exact-desc normalization groups `person` with `person`, not category id alone
- remaining-candidate set excludes earlier emitted same-desc objects
- full-candidate set includes all same-desc objects
- quantized duplicate boxes are rejected as ambiguous
- sample quotas produce sparse, healthy, hard, and duplicate buckets

Example test:

```python
from src.analysis.qwen3_vl_instance_binding_cases import build_same_desc_site


def test_remaining_candidates_exclude_previous_same_desc() -> None:
    objects = [
        {"desc": "person", "bbox_2d": [10, 10, 40, 80]},
        {"desc": "person", "bbox_2d": [200, 20, 240, 90]},
        {"desc": "dog", "bbox_2d": [500, 30, 550, 90]},
    ]

    site = build_same_desc_site(
        image_id="img1",
        object_index=1,
        objects=objects,
        already_emitted_indices={0},
    )

    assert [c["object_index"] for c in site.remaining_same_desc_candidates] == [1]
    assert [c["object_index"] for c in site.full_same_desc_candidates] == [0, 1]
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```bash
conda run -n ms python -m pytest tests/test_qwen3_vl_instance_binding_cases.py -q
```

Expected:

- FAIL because the module does not exist yet.

- [ ] **Step 3: Implement case-bank module**

Implement:

- `BindingCase`
- `CandidateInstance`
- `load_coordjson_records(paths: Sequence[Path])`
- `mine_repeated_desc_sites(records, priority_descs, quotas)`
- `build_same_desc_site(...)`
- `reject_ambiguous_quantized_sites(...)`
- `write_case_bank(cases, output_path)`
- `summarize_case_bank(cases)`

Use existing public JSONL records if present. If both configured candidate JSONL
paths are missing, fail with a path-explicit error rather than silently changing
datasets.

- [ ] **Step 4: Add duplicate/hard-case hooks**

Read existing duplication artifacts only as optional candidate generators:

- `progress/diagnostics/2026-04-13_duplication_collapse_final_analysis.md`
- `research/duplication_collapse_focus/**/probe/case_rows.jsonl` if present
- `output/analysis/**/gt_vs_pred.jsonl` only when explicitly configured

The case bank remains authoritative after it is written.

- [ ] **Step 5: Verify case-bank tests pass**

Run:

```bash
conda run -n ms python -m pytest tests/test_qwen3_vl_instance_binding_cases.py -q
```

Expected:

- PASS.

Commit:

```bash
git add src/analysis/qwen3_vl_instance_binding_cases.py tests/test_qwen3_vl_instance_binding_cases.py
git commit -m "analysis: add instance binding case bank"
```

---

## Task 3: Inventory Token Positions and Extract Hidden States

**Files:**

- Create: `src/analysis/qwen3_vl_instance_binding_hidden.py`
- Modify if required: `src/analysis/unmatched_proposal_verifier.py`
- Test: extend `tests/test_qwen3_vl_instance_binding_study.py`

- [ ] **Step 1: Add position-inventory tests**

Use a tiny CoordJSON assistant string such as:

```text
[{"desc":"person","bbox_2d":[<|coord_10|>,<|coord_20|>,<|coord_30|>,<|coord_40|>]}]
```

Assert that the inventory can identify:

- desc content end
- closing quote
- field delimiter
- `bbox_2d` key
- opening bracket
- pre-x1
- post-x1
- post-y1

- [ ] **Step 2: Implement token-position inventory**

Implement tokenizer-backed position extraction that records:

- `case_uid`
- `object_index`
- `role`
- `token_index`
- `token_text`
- `char_span` when available
- `slot` for coord positions

Use exact CoordJSON serialization. Do not prettify or normalize the text after
positions are computed.

- [ ] **Step 3: Add hidden-state extraction API**

Implement:

- `load_fixed_model_for_analysis(config)`
- `extract_hidden_state_batch(model, processor, batch, positions, layer_policy)`
- `write_hidden_state_manifest(...)`

The output manifest should record:

- checkpoint path
- dtype
- device
- model config hash if easy
- layer indices
- position roles
- case ids
- tensor shard paths

Use `output_hidden_states=True`. For first pass, save selected positions and
selected layers only. Do not save full sequence hidden states unless needed for
patching.

Batching rules:

- teacher-forced extraction should default to per-GPU batch size `8`
- generation or controlled continuation should default to per-GPU batch size `8`
- the config must record the actual batch size used in `config.resolved.yaml`
  and each stage manifest

- [ ] **Step 4: Smoke without GPU model load**

Run:

```bash
conda run -n ms python -m pytest tests/test_qwen3_vl_instance_binding_study.py -q
```

Expected:

- PASS.

Commit:

```bash
git add src/analysis/qwen3_vl_instance_binding.py tests/test_qwen3_vl_instance_binding_study.py
git commit -m "analysis: inventory binding probe token positions"
```

---

## Task 4: Implement the Position-Wise Binding Probe

**Files:**

- Create: `src/analysis/qwen3_vl_instance_binding_probe.py`
- Test: `tests/test_qwen3_vl_instance_binding_probe.py`

- [ ] **Step 1: Write probe metric tests**

Test:

- candidate scores are softmaxed per example, not across the batch
- top-1 accuracy handles variable candidate counts
- chance-normalized lift uses `1/K` for each example
- order baselines are reported separately from learned probe metrics
- image-level bootstrap groups by image id

- [ ] **Step 2: Implement candidate-conditioned probe**

Implement a low-capacity retrieval probe:

- input: hidden state `h`
- candidate feature `phi(c)` in center/log-size space
- score: projected dot product or bilinear score
- loss: cross-entropy over the example's candidate set

Train separate readouts for:

- each token role
- each layer anchor
- remaining same-desc candidates
- full same-desc candidates

Keep the first pass simple. A linear PyTorch module is enough.

- [ ] **Step 3: Implement controls**

Add metrics for:

- next-unseen-by-order baseline
- left-to-right baseline
- top-to-bottom baseline
- token-only post-x1/post-y1 baseline
- random-label sanity control

- [ ] **Step 4: Write probe outputs**

Write `probe_results.json` with:

- per-role/layer metrics
- candidate-count strata
- desc strata for priority descs
- healthy vs hard bucket strata
- bootstrap confidence intervals
- interpretation flags:
  - `pre_x1_above_order_control`
  - `largest_jump_at_x1_or_y1`
  - `full_candidate_control_survives`

- [ ] **Step 5: Verify tests**

Run:

```bash
conda run -n ms python -m pytest tests/test_qwen3_vl_instance_binding_probe.py -q
```

Expected:

- PASS.

Commit:

```bash
git add src/analysis/qwen3_vl_instance_binding_probe.py tests/test_qwen3_vl_instance_binding_probe.py
git commit -m "analysis: add instance binding probe metrics"
```

---

## Task 5: Implement Pre-x1 Multi-Modality Metrics

**Files:**

- Create: `src/analysis/qwen3_vl_instance_binding_multimodality.py`
- Test: `tests/test_qwen3_vl_instance_binding_multimodality.py`

- [ ] **Step 1: Write distribution metric tests**

Test:

- entropy is computed over coord-token ids only
- candidate-window mass sums token probabilities within `radius=2`
- target/previous/distractor windows are separately reported
- boundary-style uncertainty is separated from identity uncertainty

- [ ] **Step 2: Implement coord-token distribution extraction**

At `pre_x1`, compute:

- coord-token probability vector for `x1`
- entropy
- top-k coord-token modes
- mass in each candidate x1 window
- target-vs-previous and target-vs-distractor margins

On a smaller subset, repeat for `x2` to distinguish identity uncertainty from
box-boundary tight/loose uncertainty.

- [ ] **Step 3: Implement interpretation flags**

Write flags:

- `multi_candidate_mass`: mass spread across more than one same-desc candidate
- `target_window_dominant`: target window dominates before `x1`
- `boundary_uncertainty`: target-adjacent mass high while competitor mass low
- `coordinate_split_signature`: entropy or target margin changes sharply only
  after forcing or observing `x1/y1`

- [ ] **Step 4: Verify tests**

Run:

```bash
conda run -n ms python -m pytest tests/test_qwen3_vl_instance_binding_multimodality.py -q
```

Expected:

- PASS.

Commit:

```bash
git add src/analysis/qwen3_vl_instance_binding_multimodality.py tests/test_qwen3_vl_instance_binding_multimodality.py
git commit -m "analysis: add pre x1 multimodality metrics"
```

---

## Task 6: Implement Span-Level Causal Patching

**Files:**

- Create: `src/analysis/qwen3_vl_instance_binding_patching.py`
- Test: add focused CPU tests if hookable without loading the model

- [ ] **Step 1: Define patching rows**

Each patching row must include:

- `case_uid`
- `recipient_case_uid`
- `donor_case_uid`
- `donor_role`
- `intervention_span`
- `layer_band`
- `layer_index`
- `readout_role`
- `baseline_target_margin`
- `patched_target_margin`
- `delta_target_margin`
- `baseline_entropy`
- `patched_entropy`
- `continuation_flip`
- `routing_delta`

- [ ] **Step 2: Implement intervention spans**

Support:

- `current_desc`
- `current_schema`
- `previous_geometry`
- `previous_x1y1`
- `desc_plus_schema`

Start with one contiguous span at a time. Do not implement head-level patching
in the first pass.

- [ ] **Step 3: Implement donor/recipient pairing**

Support:

- healthy same-desc A -> B and B -> A
- hard wrong-instance `donor_intended` and `donor_competitor`
- duplicate-collapse `donor_dup` and `donor_missing`
- unrelated schema null donor

Reject cross-image donors for the main estimate unless the config marks the row
as a negative control.

- [ ] **Step 4: Implement readouts**

For each patch:

- compute next-x1 coord-token distribution
- compute target-vs-competitor x1 window margin
- compute intended-instance probe margin if probe weights are available
- compute `P(next token is coord)` as a routing sanity metric
- optionally run a short controlled continuation to test flips

- [ ] **Step 5: Add a small GPU smoke run in the worktree**

Run only after CPU tests pass:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n ms python scripts/analysis/run_qwen3_vl_instance_binding_study.py \
  --config configs/analysis/qwen3_vl_instance_binding/smoke.yaml \
  --stage patching \
  --shard-index 0 \
  --num-shards 1
```

Expected:

- A tiny `patching_results.jsonl` is written.
- No full benchmark or large rollout starts.

Commit:

```bash
git add src/analysis/qwen3_vl_instance_binding_patching.py
git commit -m "analysis: add span patching for instance binding"
```

---

## Task 7: Build the Report

**Files:**

- Create: `src/analysis/qwen3_vl_instance_binding_report.py`
- Test: `tests/test_qwen3_vl_instance_binding_report.py`

- [ ] **Step 1: Write report-decision tests**

Test decision rules:

- no pre-x1 lift and post-x1 jump -> H0-supported conclusion
- weak pre-x1 lift plus larger x1/y1 jump -> mixed conclusion
- strong pre-x1 lift with schema causal deltas -> H1-supported conclusion
- missing controls -> conclusion downgraded to inconclusive

- [ ] **Step 2: Implement summary aggregation**

Aggregate:

- probe role/layer table
- pre-x1 multimodality table
- patching intervention table
- healthy vs bad-basin comparison
- sparse-control sanity table
- open-risk checklist

- [ ] **Step 3: Implement `report.md` generation**

The report must have:

- executive conclusion
- evidence
- interpretation
- open uncertainty
- next-step recommendation

It must not overclaim if evidence is mixed.

- [ ] **Step 4: Verify tests**

Run:

```bash
conda run -n ms python -m pytest tests/test_qwen3_vl_instance_binding_report.py -q
```

Expected:

- PASS.

Commit:

```bash
git add src/analysis/qwen3_vl_instance_binding_report.py tests/test_qwen3_vl_instance_binding_report.py
git commit -m "analysis: add instance binding report synthesis"
```

---

## Task 8: Run the First-Pass GPU Study

**Files:**

- Read/write only under the study artifact root.
- Do not modify source during this task unless a bug blocks the run.
- Use all 8 GPUs for GPU-heavy stages.
- Keep per-GPU generation or controlled-continuation batch size at `8` unless a
  smoke run records a memory-pressure override.

- [ ] **Step 1: Run audit and case bank**

```bash
conda run -n ms python scripts/analysis/run_qwen3_vl_instance_binding_study.py \
  --config configs/analysis/qwen3_vl_instance_binding/default.yaml \
  --stage audit

conda run -n ms python scripts/analysis/run_qwen3_vl_instance_binding_study.py \
  --config configs/analysis/qwen3_vl_instance_binding/default.yaml \
  --stage case_bank
```

Expected:

- `checkpoint_audit.json` confirms coord-token surface.
- `case_bank.jsonl` exists.
- `case_bank_summary.json` shows nonzero healthy and hard same-desc cases.

- [ ] **Step 2: Run token position inventory**

```bash
conda run -n ms python scripts/analysis/run_qwen3_vl_instance_binding_study.py \
  --config configs/analysis/qwen3_vl_instance_binding/default.yaml \
  --stage positions
```

Expected:

- `token_position_inventory.jsonl` contains all eight required roles for each
  included object site.

- [ ] **Step 3: Run hidden-state extraction**

Run 8 shards, one per GPU:

```bash
for gpu in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES="$gpu" conda run -n ms python scripts/analysis/run_qwen3_vl_instance_binding_study.py \
    --config configs/analysis/qwen3_vl_instance_binding/default.yaml \
    --stage hidden_states \
    --shard-index "$gpu" \
    --num-shards 8 &
done
wait
```

Expected:

- `hidden_states/manifest.json` exists.
- Selected layer/position tensor shards exist.
- The merged manifest records 8 shard manifests and the actual per-GPU batch
  size.

- [ ] **Step 4: Run probe and multimodality**

```bash
conda run -n ms python scripts/analysis/run_qwen3_vl_instance_binding_study.py \
  --config configs/analysis/qwen3_vl_instance_binding/default.yaml \
  --stage probe
```

Run multimodality as 8 GPU shards:

```bash
for gpu in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES="$gpu" conda run -n ms python scripts/analysis/run_qwen3_vl_instance_binding_study.py \
    --config configs/analysis/qwen3_vl_instance_binding/default.yaml \
    --stage multimodality \
    --shard-index "$gpu" \
    --num-shards 8 &
done
wait

conda run -n ms python scripts/analysis/run_qwen3_vl_instance_binding_study.py \
  --config configs/analysis/qwen3_vl_instance_binding/default.yaml \
  --stage merge_multimodality
```

Expected:

- `probe_results.json` exists.
- `pre_x1_multimodality.jsonl` exists.

- [ ] **Step 5: Run causal patching**

```bash
for gpu in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES="$gpu" conda run -n ms python scripts/analysis/run_qwen3_vl_instance_binding_study.py \
    --config configs/analysis/qwen3_vl_instance_binding/default.yaml \
    --stage patching \
    --shard-index "$gpu" \
    --num-shards 8 &
done
wait

conda run -n ms python scripts/analysis/run_qwen3_vl_instance_binding_study.py \
  --config configs/analysis/qwen3_vl_instance_binding/default.yaml \
  --stage merge_patching
```

Expected:

- `patching_results.jsonl` exists.
- The number of rows equals the configured patching sites times spans times
  layer bands, minus explicit rejects.
- Any controlled-continuation rows record generation batch size `8` unless an
  explicit memory override was used.

- [ ] **Step 6: Build the report**

```bash
conda run -n ms python scripts/analysis/run_qwen3_vl_instance_binding_study.py \
  --config configs/analysis/qwen3_vl_instance_binding/default.yaml \
  --stage report
```

Expected:

- `summary.json` exists.
- `report.md` exists.
- The report explicitly selects H0, H1, mixed, or inconclusive.

---

## Task 9: Verify and Publish the Findings

**Files:**

- Modify: `progress/diagnostics/README.md` if the final decision note is added.
- Create after results: `progress/diagnostics/2026-04-24_qwen3_vl_instance_binding_mechanism.md`
- Create after results: `progress/diagnostics/artifacts/qwen3_vl_instance_binding_mechanism_20260424/`

- [ ] **Step 1: Run focused tests**

```bash
conda run -n ms python -m pytest \
  tests/test_qwen3_vl_instance_binding_cases.py \
  tests/test_qwen3_vl_instance_binding_study.py \
  tests/test_qwen3_vl_instance_binding_probe.py \
  tests/test_qwen3_vl_instance_binding_multimodality.py \
  tests/test_qwen3_vl_instance_binding_report.py \
  -q
```

Expected:

- PASS.

- [ ] **Step 2: Validate artifacts**

Run a small validation script or stage that checks:

- no missing required files
- `case_bank_summary.json` counts match `case_bank.jsonl`
- all probe roles have entries or an explicit skip reason
- `patching_results.jsonl` contains all configured spans
- `report.md` references the exact checkpoint and case counts

- [ ] **Step 3: Promote only reviewed results**

Copy durable artifacts only after the report is reviewed:

```bash
mkdir -p progress/diagnostics/artifacts/qwen3_vl_instance_binding_mechanism_20260424
cp output/analysis/qwen3-vl-instance-binding-mechanism-20260424/summary.json \
  progress/diagnostics/artifacts/qwen3_vl_instance_binding_mechanism_20260424/summary.json
cp output/analysis/qwen3-vl-instance-binding-mechanism-20260424/report.md \
  progress/diagnostics/artifacts/qwen3_vl_instance_binding_mechanism_20260424/report.md
```

Then write the final diagnostic note under `progress/diagnostics/` with:

- scope and checkpoint
- sample counts
- evidence
- interpretation
- open uncertainty
- next recommendation

- [ ] **Step 4: Commit implementation and findings separately**

Implementation commit:

```bash
git add src/analysis scripts/analysis configs/analysis/qwen3_vl_instance_binding tests
git commit -m "analysis: run qwen3 vl instance binding mechanism study"
```

Findings commit:

```bash
git add progress/diagnostics
git commit -m "docs: record qwen3 vl instance binding mechanism findings"
```

## Stop Conditions

Stop and ask for direction if:

- the checkpoint is not a coord-token merged checkpoint
- configured JSONL data paths are missing and no obvious canonical replacement
  exists
- hidden-state extraction requires editing upstream HF model files
- same-desc case mining cannot produce at least 12 healthy and 12 hard sites
- patching results show large parse/routing corruption on most interventions
- a result looks strong but fails image-swap or full-candidate controls
