# A3.2 Sorted Vs Random No-Newline Phenotype Roadmap And Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an audit-safe A3.2 mechanism experiment comparing the two ckpt3668 full-object pure-CE checkpoints under no-newline compact-full prompting, with prefix-state readouts, native greedy rollout phenotype, and FN-specialized hint-ladder probes.

**Architecture:** A3.2 must be a separate analysis project, not another pile of scripts inside the A3.1 prefix-state tomography surface. Reuse stable utilities from A3.1 only where they are truly generic, such as compact row rendering and JSONL helpers; own A3.2 config, runner, status, artifacts, reports, rollout phenotype, FN universe, and FN probe logic in a new package.

**Tech Stack:** Python, PyTorch/Qwen3-VL HF runtime, JSONL/YAML artifacts, existing CoordExp inference/eval helpers, pytest, tmux, 8 single-GPU shard workers for analysis only.

---

## Scope Lock

This roadmap implements the design recorded in:

- `docs/superpowers/specs/2026-06-04-sorted-vs-random-no-newline-phenotype-design.md`

The implementation scope is:

- Checkpoints:
  - random: `/data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/compact_full_fullobj_random_sft_bsz16_4epoch_tokenrows_v2/compact-full-fullobj-random-sft-bsz16-4epoch-tokenrows-v2/v1-20260601-062428/checkpoint-3668`
  - sorted: `/data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/compact_full_fullobj_sorted_sft_bsz16_4epoch_tokenrows_v2/compact-full-fullobj-sorted-sft-bsz16-4epoch-tokenrows-v2/v1-20260601-062429/checkpoint-3668`
- Local analysis JSONLs:
  - `/data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000/train.coord.jsonl`
  - `/data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000/val.coord.jsonl`
- Image root:
  - `/data/CoordExp/public_data/coco/rescale_32_1024_bbox`
- Evidence labels:
  - prefix readout: `a3_2_prefix4096_hardbiased_len12000_canonical_sorted`
  - native rollout: `a3_2_rollout1024_greedy_len12000_native`
  - FN probe: `a3_2_fn512_greedyfn_len12000`
- Main A3.2 artifact root:
  - `/data/CoordExp/outputs/analysis/autoreg_object_rollout/sorted_random_no_newline_phenotype/fullobj_purece_random_vs_sorted_ckpt3668_phase_a3_2`

Do not describe this as production training, full validation, or exact training-data replay. It is local `len12000` mechanism probing over a clean random-vs-sorted checkpoint pair.

External eval context to preserve in the final report:

- The user's offline `val-200` detector metrics already show `sorted > random` across AP/AR/F1:
  - `AP@[.50:.95]`: `0.4072 - 0.3914 = +0.0158`
  - `AP50`: `0.5642 - 0.5351 = +0.0291`
  - `AP75`: `0.4226 - 0.4054 = +0.0173`
  - `AR100`: `0.4836 - 0.4498 = +0.0338`
  - `F1-ish@0.50`: `0.5892 - 0.4659 = +0.1233`
  - `recall@0.50`: `0.5637 - 0.4952 = +0.0686`
  - `precision@0.50`: `0.6171 - 0.4400 = +0.1771`
  - `FN@0.50`: `630 - 729 = -99`
  - `FP@0.50`: `505 - 910 = -405`
- Treat these numbers as external phenotype context, not a mechanism proof. A3.2 should explain what changes in prefix transition, rollout behavior, and FN rescue evidence are consistent with sorted's observed advantage.

## Subagent Audit Incorporation

Four read-only audit lanes reviewed the design before this roadmap. Their findings become hard gates:

- Research-design gate: data-root mismatch and canonical sorted-prefix conditioning must be explicit in every summary and report.
- Code-integration gate: A3.1 role, phase, run-id, report, and merge hardcoding must not leak into A3.2 artifacts.
- FN-methodology gate: FN accounting must be instance-level and multi-axis, not a desc-only mutually exclusive bucket.
- Runtime/GPU gate: `len12000` JSONLs must use the image-bearing `/data/CoordExp/public_data/coco/rescale_32_1024_bbox` image root, and GPU launch must wait until CPU/index/status gates pass.

Implementation must treat any violation of these gates as a blocker, not as a warning.

Audit finding to roadmap mapping:

| Audit finding | Roadmap gate |
| --- | --- |
| A3.1 launcher/config/role defaults would run ckpt3664 or mislabel ckpt3668 | Tasks 1, 3, 7, 8 require a new A3.2 project, semantic roles, dynamic merge/report labels, and no legacy-label status gate |
| `len12000` JSONLs need the image-bearing root | Task 1 fixes `image_root: /data/CoordExp/public_data/coco/rescale_32_1024_bbox` and adds `data_root_audit.json` |
| Main paired probe is canonical sorted-prefix-conditioned | Tasks 2, 3, 10 persist `prefix_source_policy` and restrict interpretation to canonical sorted teacher-prefix readout |
| A3.1 x-first/mixed prefix policies do not match training sorted semantics | Task 2 uses top-to-bottom then left-to-right `(y1, x1, original_index)` canonical sorted prefixes |
| FN bucket rates would compare different FN populations | Task 5 creates a per-GT `fn_case_universe.jsonl` with shared, random-only, and sorted-only FN membership |
| Desc-only boundary roles cannot identify residual accounting | Tasks 3, 5, 6 store role lists, emitted/residual GT indices, x1 attribution, and multi-axis FN labels |
| Fixed radius 24 would overstate small-object rescue | Task 6 separates strict R95 evidence from broad radius-24 diagnostics |
| Prefix suppression can disappear behind primary buckets | Task 6 reports prefix-flip axes separately from within-prefix primary buckets |
| Status currently checks existence, not semantic consistency | Tasks 7, 8 add row-count, role-label, stale-file, provenance, and full FN artifact gates |

Implementation boundary added during smoke:

- Native rollout is an unconstrained free-text greedy diagnostic.  Local parser
  problems and non-metric-bearing parser policies are recorded as evidence and
  must not fail-fast the A3.2 mechanism run.  Official metric-bearing eval
  remains strict and separate.
- Replayable FN cases preserve raw `fn_bbox` in the source GT/rollout surface.
  For current COCO `len12000` artifacts this is pixel `xyxy`; wide images can
  have coordinates above 1000.  FN hint prompts must derive a separate
  coord-token target box from `fn_bbox`, `width`, and `height`, and use that
  target for hint seeding, generated-box IoU, and slot evidence.
- `rollout_phenotype_rows.jsonl` must forward real native-rollout provenance
  via `source_runtime_kind`, `source_checkpoint_fingerprint`, and
  `source_gpu_id`; status gates must reject placeholder or provenance-free
  mechanism artifacts.

## File Structure

Create a new idea-wise analysis surface:

- Create: `src/analysis/sorted_random_no_newline_phenotype/__init__.py`
- Create: `src/analysis/sorted_random_no_newline_phenotype/config.py`
- Create: `src/analysis/sorted_random_no_newline_phenotype/data_root_audit.py`
- Create: `src/analysis/sorted_random_no_newline_phenotype/prefix_index.py`
- Create: `src/analysis/sorted_random_no_newline_phenotype/boundary_roles.py`
- Create: `src/analysis/sorted_random_no_newline_phenotype/paired_probe.py`
- Create: `src/analysis/sorted_random_no_newline_phenotype/rollout_phenotype.py`
- Create: `src/analysis/sorted_random_no_newline_phenotype/fn_matching.py`
- Create: `src/analysis/sorted_random_no_newline_phenotype/fn_probe.py`
- Create: `src/analysis/sorted_random_no_newline_phenotype/merge_report.py`
- Create: `src/analysis/sorted_random_no_newline_phenotype/gallery.py`
- Create: `src/analysis/sorted_random_no_newline_phenotype/status.py`
- Create: `src/analysis/sorted_random_no_newline_phenotype/runner.py`
- Create: `scripts/analysis/sorted_random_no_newline_phenotype/run.py`
- Create: `scripts/analysis/sorted_random_no_newline_phenotype/status.py`
- Create: `scripts/analysis/sorted_random_no_newline_phenotype/launch_a3_2_tmux.sh`
- Create: `configs/analysis/sorted_random_no_newline_phenotype/fullobj_purece_random_vs_sorted_ckpt3668_phase_a3_2.yaml`
- Create: `configs/infer/recursive_detection_ce/fullobj_random_purece_ckpt3668_a3_2_rollout1024_greedy.yaml`
- Create: `configs/infer/recursive_detection_ce/fullobj_sorted_purece_ckpt3668_a3_2_rollout1024_greedy.yaml`
- Create: `tests/analysis/sorted_random_no_newline_phenotype/`

Allowed reuse:

- `src/analysis/prefix_state_transition_tomography/prefix_rendering.py` for no-newline compact row rendering.
- `src/analysis/prefix_state_transition_tomography/jsonl.py` or an equivalent tiny local JSONL helper.
- `src/analysis/hard_ce_coord_logit_locality.py` model-handle helpers if they avoid import-time heavy GPU work.
- `src/eval/detection_geometry.py` or existing rollout matching helpers for IoU/matching, wrapped with A3.2 same-desc semantics.
- `src/datasets/geometry.py` for bbox validation and IoU where direct geometry primitives are needed.

Avoid changing upstream HF model files. Avoid hidden defaults that silently fall back to A3.1 labels.

## Audit-Gated Execution Model

Use subagents during implementation, but keep write scopes disjoint:

- Agent lane 1: config, provenance, data-root audit, runner/status.
- Agent lane 2: canonical sorted prefix index, boundary roles, paired readout.
- Agent lane 3: rollout phenotype and FN universe matching.
- Agent lane 4: FN hint ladder, instance-level accounting, strict R95 evidence.
- Agent lane 5: reports, galleries, artifact validation.
- Agent lane 6: independent spec/code-quality audit.

Keep active subagents at or below 6. GPU work starts only after local tests, dry-run, CPU index/validate, and status gates pass.

---

### Task 1: Scaffold A3.2 Config, Identity, And Provenance

**Files:**
- Create: `src/analysis/sorted_random_no_newline_phenotype/__init__.py`
- Create: `src/analysis/sorted_random_no_newline_phenotype/config.py`
- Create: `src/analysis/sorted_random_no_newline_phenotype/data_root_audit.py`
- Create: `configs/analysis/sorted_random_no_newline_phenotype/fullobj_purece_random_vs_sorted_ckpt3668_phase_a3_2.yaml`
- Test: `tests/analysis/sorted_random_no_newline_phenotype/test_config.py`
- Test: `tests/analysis/sorted_random_no_newline_phenotype/test_data_root_audit.py`

- [ ] **Step 1: Write config tests first**

Write tests that assert the A3.2 YAML resolves to:

```python
def test_a3_2_config_resolves_semantic_roles(a3_2_config):
    config = load_config(a3_2_config)
    assert config.project_id == "sorted_random_no_newline_phenotype"
    assert config.phase_id == "phase_a3_2"
    assert config.schema_version == "a3.2.v1"
    assert config.run_id == "fullobj_purece_random_vs_sorted_ckpt3668_phase_a3_2"
    assert tuple(config.checkpoints) == (
        "fullobj_random_pure_ce_ckpt3668",
        "fullobj_sorted_pure_ce_ckpt3668",
    )
    assert config.template_contract.row_separator == "none"
    assert config.template_contract.detection_sequence_format == "compact_full"
    assert config.template_contract.coordinate_surface == "coord_token"
    assert config.template_contract.bbox_format == "xyxy"
    assert config.sampling.max_prefix_states == 4096
    assert config.sampling.num_shards == 8
    assert config.rollout.limit_images == 1024
    assert config.fn_probe.max_fn_objects_per_checkpoint == 512
```

Also test rejection of invalid `template_contract.row_separator`, missing checkpoint roles, non-absolute paths, and `image_root` that does not contain the first sampled image.

- [ ] **Step 2: Implement config dataclasses and validation**

Required config objects:

```python
@dataclass(frozen=True)
class TemplateContractConfig:
    detection_sequence_format: str
    coordinate_surface: str
    bbox_format: str
    row_separator: str

@dataclass(frozen=True)
class CheckpointConfig:
    checkpoint_path: Path
    training_ordering: str
    readout_prompt_ordering: str

@dataclass(frozen=True)
class A32Config:
    project_id: str
    phase_id: str
    schema_version: str
    run_id: str
    artifact_root: Path
    train_jsonl: Path
    val_jsonl: Path
    image_root: Path
    checkpoints: dict[str, CheckpointConfig]
    template_contract: TemplateContractConfig
    sampling: SamplingConfig
    rollout: RolloutConfig
    fn_probe: FNProbeConfig
    peak: PeakConfig
```

Validation must fail fast if:

- `row_separator != "none"`;
- any checkpoint path is missing;
- the roles are not exactly the semantic random/sorted role names;
- `image_root` is accidentally set to the `len12000` directory;
- `sampling.num_shards != 8` for the full config.

- [ ] **Step 3: Add `data_root_audit.json` support**

`data_root_audit.py` must produce:

```json
{
  "status": "ok",
  "actual_train_jsonl": ".../rescale_32_1024_bbox_len12000/train.coord.jsonl",
  "actual_val_jsonl": ".../rescale_32_1024_bbox_len12000/val.coord.jsonl",
  "image_root": ".../rescale_32_1024_bbox",
  "evidence_scope": "len12000-jsonl-local-mechanism-probe",
  "row_counts": {"train": 117266, "val": 4952},
  "jsonl_sha256": {"train": "...", "val": "..."},
  "object_count_histogram": {},
  "desc_count_histogram": {},
  "same_desc_multi_instance_count": 0,
  "missing_image_examples": []
}
```

Hashing may be full-file SHA256 or a recorded stable fingerprint if runtime cost is documented. For the full run, prefer full-file SHA256 once.

- [ ] **Step 4: Create the A3.2 YAML**

The YAML must use:

```yaml
project_id: sorted_random_no_newline_phenotype
phase_id: phase_a3_2
schema_version: a3.2.v1
run_id: fullobj_purece_random_vs_sorted_ckpt3668_phase_a3_2
artifact_root: /data/CoordExp/outputs/analysis/autoreg_object_rollout/sorted_random_no_newline_phenotype/fullobj_purece_random_vs_sorted_ckpt3668_phase_a3_2
train_jsonl: /data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000/train.coord.jsonl
val_jsonl: /data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000/val.coord.jsonl
image_root: /data/CoordExp/public_data/coco/rescale_32_1024_bbox
template_contract:
  detection_sequence_format: compact_full
  coordinate_surface: coord_token
  bbox_format: xyxy
  row_separator: none
sampling:
  max_prefix_states: 4096
  num_shards: 8
  seed: 3668
  easy_sanity_max_fraction: 0.20
rollout:
  limit_images: 1024
  decode_policy: free_text_unconstrained_greedy_temp0
  native_prompt_ordering: true
  constraint_policy: none
fn_probe:
  max_fn_objects_per_checkpoint: 512
  hint_policy_id: desc_x1_r95_ladder_v1
  strict_r95_axis_fraction: 0.04
  strict_r95_cap_bins: 8
  broad_x1_radius: 24
checkpoints:
  fullobj_random_pure_ce_ckpt3668:
    training_ordering: random_permutation
    readout_prompt_ordering: sorted
    checkpoint_path: /data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/compact_full_fullobj_random_sft_bsz16_4epoch_tokenrows_v2/compact-full-fullobj-random-sft-bsz16-4epoch-tokenrows-v2/v1-20260601-062428/checkpoint-3668
  fullobj_sorted_pure_ce_ckpt3668:
    training_ordering: sorted
    readout_prompt_ordering: sorted
    checkpoint_path: /data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/compact_full_fullobj_sorted_sft_bsz16_4epoch_tokenrows_v2/compact-full-fullobj-sorted-sft-bsz16-4epoch-tokenrows-v2/v1-20260601-062429/checkpoint-3668
```

- [ ] **Step 5: Verify Task 1**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider \
  tests/analysis/sorted_random_no_newline_phenotype/test_config.py \
  tests/analysis/sorted_random_no_newline_phenotype/test_data_root_audit.py
```

Expected: all tests pass, and config dry-run prints A3.2 paths and semantic role names.

---

### Task 2: Canonical Sorted Prefix-State Index

**Files:**
- Create: `src/analysis/sorted_random_no_newline_phenotype/prefix_index.py`
- Test: `tests/analysis/sorted_random_no_newline_phenotype/test_prefix_index.py`

- [ ] **Step 1: Write ordering tests that catch x-first mistakes**

Use a fixture where `x1` and `y1` order disagree:

```python
objects = [
    {"gt_idx": 0, "desc": "person", "bbox_xyxy": [900, 10, 950, 80]},
    {"gt_idx": 1, "desc": "person", "bbox_xyxy": [20, 500, 80, 580]},
    {"gt_idx": 2, "desc": "chair", "bbox_xyxy": [40, 20, 100, 100]},
]
ordered = canonical_sorted_objects(objects)
assert [obj["gt_idx"] for obj in ordered] == [0, 2, 1]
```

The policy is top-to-bottom then left-to-right: `(y1, x1, original_index)`.

- [ ] **Step 2: Implement canonical sorted state construction**

Each sampled prefix-state row must include:

```json
{
  "project_id": "sorted_random_no_newline_phenotype",
  "phase_id": "phase_a3_2",
  "schema_version": "a3.2.v1",
  "run_id": "fullobj_purece_random_vs_sorted_ckpt3668_phase_a3_2",
  "evidence_scope": "a3_2_prefix4096_hardbiased_len12000_canonical_sorted",
  "prefix_source_policy": "canonical_sorted_teacher_prefix_readout",
  "prefix_order_policy_id": "canonical_sorted_yx_teacher_v1",
  "readout_prompt_ordering": "sorted",
  "template_contract": {"row_separator": "none"},
  "gt_objects": [],
  "canonical_sorted_gt_indices": [],
  "emitted_gt_indices": [],
  "residual_gt_indices": [],
  "candidate_descs": [],
  "candidate_descs_with_roles": []
}
```

- [ ] **Step 3: Implement hard-biased sampling quotas**

At minimum, persist `sample_manifest.json` with:

- available and selected rows by split;
- same-desc count bucket;
- object-count bucket;
- desc-count bucket;
- easy sanity selected fraction;
- underfill reasons.

Keep easy sanity at or below 20 percent.

- [ ] **Step 4: Include no-newline prefix preview hashes**

For sampled rows, render a small preview prefix with `render_teacher_prefix(...)` and assert:

```python
assert "\n" not in rendered_prefix
```

Persist `rendered_prefix_sha256` and `rendered_prefix_char_len` for sampled rows, not full rendered text.

- [ ] **Step 5: Verify Task 2**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider \
  tests/analysis/sorted_random_no_newline_phenotype/test_prefix_index.py
```

Then run CPU index/validate only:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/analysis/sorted_random_no_newline_phenotype/run.py \
  --config configs/analysis/sorted_random_no_newline_phenotype/fullobj_purece_random_vs_sorted_ckpt3668_phase_a3_2.yaml \
  --stages data_root_audit,prefix_state_index,validate
```

Expected artifacts:

- `data_root_audit.json`
- `prefix_state_index.jsonl`
- `prefix_state_sampled_rows.jsonl`
- `prefix_state_index_summary.json`
- `sample_manifest.json`

Reject the run if any row contains `phase_a3_1`, `ckpt3664`, `et_rmp_ce`, or legacy `pure_ce` labels.

---

### Task 3: Rich Boundary Roles And Dynamic Paired Readout

**Files:**
- Create: `src/analysis/sorted_random_no_newline_phenotype/boundary_roles.py`
- Create: `src/analysis/sorted_random_no_newline_phenotype/paired_probe.py`
- Test: `tests/analysis/sorted_random_no_newline_phenotype/test_boundary_roles.py`
- Test: `tests/analysis/sorted_random_no_newline_phenotype/test_paired_probe.py`

- [ ] **Step 1: Write tests for overlapping roles**

The same desc may be both emitted and residual. Store roles as lists:

```python
roles = candidate_roles(
    desc="chair",
    target_fn_desc="chair",
    emitted_same_desc_gt_indices=[1],
    residual_same_desc_gt_indices=[2],
    selected_hard_competitor_desc=None,
)
assert roles == ["target_fn_desc", "emitted_same_desc", "residual_same_desc"]
```

Never collapse this to one scalar role for A3.2.

- [ ] **Step 2: Implement boundary summaries over role lists**

Each boundary summary must include:

```json
{
  "winner_desc": "chair",
  "winner_roles": ["target_fn_desc", "emitted_same_desc", "residual_same_desc"],
  "boundary_winner_class": "residual_same_desc_favored",
  "residual_vs_eos_margin": 0.0,
  "residual_vs_winner_margin": 0.0,
  "low_margin_flag": false,
  "candidate_descs_with_roles": []
}
```

Allowed winner classes:

- `residual_same_desc_favored`
- `residual_other_desc_favored`
- `emitted_same_desc_favored`
- `emitted_other_desc_favored`
- `hard_competitor_favored`
- `other_gt_desc_favored`
- `eos_favored`
- `mixed_tie`
- `no_residual_candidate`

- [ ] **Step 3: Implement dynamic checkpoint roles**

`paired_probe.py` must read ordered checkpoint roles from config. It must not import or reuse `CHECKPOINT_ROLES = ("et_rmp_ce", "pure_ce")`.

Shard manifests must include:

```json
{
  "checkpoint_roles": [
    "fullobj_random_pure_ce_ckpt3668",
    "fullobj_sorted_pure_ce_ckpt3668"
  ],
  "prefix_source_policy": "canonical_sorted_teacher_prefix_readout",
  "readout_prompt_ordering": "sorted"
}
```

- [ ] **Step 4: Preserve A3.2 prompt provenance**

For paired readout, both checkpoints intentionally use canonical sorted prompt/order policy. Persist:

- `checkpoint_training_ordering`;
- `readout_prompt_ordering`;
- `teacher_prefix_ordering`;
- `prefix_source_policy`.

Reports must phrase this as "canonical sorted teacher-prefix readout", not "native rollout behavior".

- [ ] **Step 5: Verify Task 3**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider \
  tests/analysis/sorted_random_no_newline_phenotype/test_boundary_roles.py \
  tests/analysis/sorted_random_no_newline_phenotype/test_paired_probe.py
```

Run one tiny mocked shard test, then a dry run:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/analysis/sorted_random_no_newline_phenotype/run.py \
  --config configs/analysis/sorted_random_no_newline_phenotype/fullobj_purece_random_vs_sorted_ckpt3668_phase_a3_2.yaml \
  --stages paired_checkpoint_probe \
  --shard-id 0 \
  --dry-run
```

Expected: no legacy A3.1 role labels in the dry-run output.

---

### Task 4: Native Greedy Rollout Configs And Phenotype Input Contract

**Files:**
- Create: `configs/infer/recursive_detection_ce/fullobj_random_purece_ckpt3668_a3_2_rollout1024_greedy.yaml`
- Create: `configs/infer/recursive_detection_ce/fullobj_sorted_purece_ckpt3668_a3_2_rollout1024_greedy.yaml`
- Create: `src/analysis/sorted_random_no_newline_phenotype/rollout_phenotype.py`
- Test: `tests/analysis/sorted_random_no_newline_phenotype/test_rollout_phenotype.py`

- [ ] **Step 1: Create native greedy rollout configs**

Both configs must use:

- the matching ckpt3668 checkpoint;
- `temperature: 0.0`;
- deterministic ordinary greedy decode;
- free-text, unconstrained generation: do not enable compact grammar constraints, trie constraints, forced-row constraints, constrained decoding processors, or other decode-time structural guards;
- 1024 image limit;
- no-newline compact-full prompt/template;
- role-specific output roots under the A3.2 artifact root.

Random checkpoint uses its native random-order prompt/order policy for rollout. Sorted checkpoint uses its native sorted prompt/order policy for rollout.

Parsing failures, malformed rows, early EOS, and format drift are part of the rollout phenotype in this study. They must be recorded and analyzed rather than hidden by decode-time grammar constraints.

- [ ] **Step 2: Define the rollout input artifact contract**

`rollout_phenotype.py` must accept either standard infer/eval artifacts or an explicitly normalized A3.2 rollout JSONL. Required normalized fields:

```json
{
  "checkpoint_role": "fullobj_random_pure_ce_ckpt3668",
  "image_id": "000000000009",
  "source_line_idx": 0,
  "raw_output_text_sha256": "...",
  "pred_rows_ordered": [],
  "invalid_pred_rows": [],
  "rollout_stop_reason": "eos",
  "decode_policy": "free_text_unconstrained_greedy_temp0",
  "constraint_policy": "none",
  "native_prompt_ordering": "random_permutation",
  "template_contract": {"row_separator": "none"}
}
```

- [ ] **Step 3: Implement rollout phenotype metrics**

Persist `rollout/rollout_phenotype_rows.jsonl` and `rollout/rollout_summary.json`.

Required metrics:

- class-any recall;
- instance recall;
- class-found-but-instance-missed rate;
- same-desc FN rate;
- same-desc duplication rate;
- predicted row count distribution;
- EOS-after-partial-coverage rate;
- sorted GT order agreement;
- parse invalid count;
- degenerate box count;
- duplicate side-label count.

- [ ] **Step 4: Verify Task 4**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider \
  tests/analysis/sorted_random_no_newline_phenotype/test_rollout_phenotype.py
```

Do not launch the 1024-image GPU rollout until Task 8 gates pass.

---

### Task 5: FN Case Universe And Replayable Matching Ledger

**Files:**
- Create: `src/analysis/sorted_random_no_newline_phenotype/fn_matching.py`
- Test: `tests/analysis/sorted_random_no_newline_phenotype/test_fn_matching.py`

- [ ] **Step 1: Write matching tests**

Main FN definition:

- same-desc greedy one-to-one IoU >= 0.5 is a match;
- if no same-desc match, the GT is FN;
- `near_miss` is side label for same-desc IoU in `[0.3, 0.5)`;
- `wrong_desc_overlap` is side label for non-same-desc IoU >= 0.5;
- `same_desc_duplicate` is side label for same-desc IoU > 0.95;
- side labels do not alter main FN status.

Also test tie-breaking by `(-iou, pred_idx, gt_idx)` to match the eval geometry convention.

- [ ] **Step 2: Implement per-GT FN universe**

Create:

- `fn_probe/fn_case_universe.jsonl`
- `fn_probe/fn_cases.jsonl`
- `fn_probe/fn_matching_summary.json`

Each `fn_case_universe.jsonl` row must include both checkpoints:

```json
{
  "gt_object_key": "split:image_id:gt_idx",
  "is_fn_fullobj_random_pure_ce_ckpt3668": true,
  "is_fn_fullobj_sorted_pure_ce_ckpt3668": false,
  "random_match_pred_idx": null,
  "sorted_match_pred_idx": 3,
  "fn_membership": "random_only_fn",
  "sampled_for_probe": true,
  "sampling_reason": "same_desc_hard_case"
}
```

This prevents comparing different FN populations as if they were the same denominator.

- [ ] **Step 3: Make each FN case replayable**

Minimum `fn_cases.jsonl` fields:

- `fn_case_id`;
- `checkpoint_role`;
- `split`;
- `image_id`;
- `source_line_idx`;
- `image_path`;
- `width`;
- `height`;
- `coord_mode`;
- `bbox_surface`;
- `data_root`;
- `jsonl_sha256`;
- `checkpoint_fingerprint`;
- `decode_policy`;
- `template_contract`;
- `fn_gt_idx`;
- `fn_desc`;
- `fn_bbox`;
- `gt_sorted_rank`;
- `same_desc_gt_count`;
- `object_count`;
- `pred_rows_ordered`;
- `match_policy_id`;
- `match_candidates_same_desc`;
- `accepted_matches`;
- `best_same_desc_iou`;
- `near_miss`;
- `wrong_desc_overlap`;
- `same_desc_duplicate`;
- `duplicate_source`;
- `invalid_pred_count`;
- `sample_stratum`.

- [ ] **Step 4: Verify Task 5**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider \
  tests/analysis/sorted_random_no_newline_phenotype/test_fn_matching.py
```

Add a replay check:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/analysis/sorted_random_no_newline_phenotype/run.py \
  --config configs/analysis/sorted_random_no_newline_phenotype/fullobj_purece_random_vs_sorted_ckpt3668_phase_a3_2.yaml \
  --stages fn_case_index,fn_match_replay \
  --dry-run
```

Expected: the replay stage can regenerate the same FN IDs and side labels from `fn_cases.jsonl`.

---

### Task 6: FN Hint-Ladder Probe With Instance-Level Accounting

**Files:**
- Create: `src/analysis/sorted_random_no_newline_phenotype/fn_probe.py`
- Test: `tests/analysis/sorted_random_no_newline_phenotype/test_fn_probe.py`

- [ ] **Step 1: Write strict R95 tests**

Use the focused Gaussian R95 rule:

```python
def r95(axis_len: int) -> int:
    return math.floor(min(8, 0.04 * axis_len))
```

Test:

```python
assert r95(10) == 0
assert r95(50) == 2
assert r95(100) == 4
assert r95(200) == 8
assert r95(400) == 8
```

Also test that width 10 with peak at `gt_x1 + 1` is:

```python
assert row["x1_broad_near_24"] is True
assert row["x1_strict_r95_hit"] is False
```

- [ ] **Step 2: Implement hint levels**

Hint levels:

- `none`: score boundary and image-local candidate descs;
- `desc`: force FN desc and inspect pre-x1;
- `desc_x1`: force FN desc plus exact GT x1 token and inspect y1/x2/y2;
- `desc_x1_y1`: force desc plus GT x1/y1 tokens and inspect x2/y2.

Persist `hint_policy_id`, `hint_text_sha256`, `hint_coords`, and `valid_parse`.

- [ ] **Step 3: Implement prefix conditions with exact anchor policies**

Required prefix conditions:

- `empty_prefix`;
- `rollout_prefix`;
- `teacher_sorted_prefix`;
- `teacher_oracle_remaining_prefix`;
- `same_desc_removed_prefix`;
- `same_desc_shuffled_prefix`.

Each row must include:

- `anchor_policy`;
- `prefix_gt_indices`;
- `prefix_pred_indices`;
- `prefix_len`;
- `contains_future_gt_after_fn`;
- `rendered_assistant_prefix_sha256`;
- `prefix_objects` with source, gt_idx, pred_idx, desc, bbox, and order_idx.

- [ ] **Step 4: Store instance-level evidence**

Write separate but joinable artifacts:

- `fn_probe/fn_probe_rows.jsonl`
- `fn_probe/fn_candidate_scores.jsonl`
- `fn_probe/fn_slot_evidence.jsonl`

For same-desc emitted-plus-residual cases, keep emitted and residual GT indices separate. Do not let a desc-only `chair` score count as residual accounting success unless x1/slot evidence points to the residual instance rather than the emitted instance.

- [ ] **Step 5: Implement multi-axis bucket predicates**

Use orthogonal booleans plus a primary bucket trace.

Headline axes:

- `not_rescued_under_valid_desc_x1_controls`
- `coord_binding_failure`
- `desc_selection_failure`
- `residual_accounting_failure`
- `prefix_suppression_flip`
- `low_margin_ambiguous`
- `probe_invalid_or_unscored`

Primary bucket labels can still be emitted for table convenience, but reports must include multi-label counts and prefix-flip metrics. Do not call a case `vision_unreachable_fn` unless all probe controls are valid and the stronger evidence really supports that label. Prefer the conservative label `not_rescued_under_valid_desc_x1_controls`.

- [ ] **Step 6: Verify Task 6**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider \
  tests/analysis/sorted_random_no_newline_phenotype/test_fn_probe.py
```

Required fixture coverage:

- emitted and residual same-desc instances exist simultaneously;
- desc is favored but x1 peak points to emitted instance;
- teacher prefix has strict x1 evidence but rollout prefix loses it;
- invalid generated continuation becomes `probe_invalid_or_unscored`, not `vision_unreachable_fn`;
- broad radius-24 hit does not imply strict R95 rescue.

---

### Task 7: Merge, Report, Gallery, And Artifact Status

**Files:**
- Create: `src/analysis/sorted_random_no_newline_phenotype/merge_report.py`
- Create: `src/analysis/sorted_random_no_newline_phenotype/gallery.py`
- Create: `src/analysis/sorted_random_no_newline_phenotype/status.py`
- Test: `tests/analysis/sorted_random_no_newline_phenotype/test_merge_report.py`
- Test: `tests/analysis/sorted_random_no_newline_phenotype/test_status.py`
- Test: `tests/analysis/sorted_random_no_newline_phenotype/test_gallery.py`

- [ ] **Step 1: Merge paired prefix-state rows by dynamic roles**

The pair is:

```python
role_a = "fullobj_random_pure_ce_ckpt3668"
role_b = "fullobj_sorted_pure_ce_ckpt3668"
```

Delta names must be explicit:

- `sorted_minus_random_residual_vs_eos_margin`
- `sorted_minus_random_strict_r95_x1_hit_rate`
- `sorted_minus_random_boundary_residual_favored_rate`

Never emit `pure_minus_et` in A3.2 artifacts.

- [ ] **Step 2: Write report sections**

`report.md` must include:

- scope and evidence labels;
- checkpoint provenance;
- actual local data root and image root;
- external offline `val-200` eval context showing sorted's AP/AR/F1/FN/FP advantage over random, explicitly labeled as phenotype context rather than mechanism proof;
- template contract;
- prefix readout results;
- native rollout phenotype;
- FN universe and shared/random-only/sorted-only FN counts;
- FN multi-axis bucket counts;
- prefix sensitivity;
- metric compatibility with A3.1;
- interpretive caveats.

Ban unqualified causal language in reports. Use "consistent with" or "supports under this evidence scope" unless the experiment contains a direct intervention.

- [ ] **Step 3: Build galleries**

Create:

- `gallery/index.md`
- `gallery/images/*.jpg`
- `fn_probe/gallery/index.md`
- `fn_probe/gallery/images/*.jpg`

Gallery images must visually distinguish:

- GT objects;
- random predictions;
- sorted predictions;
- FN target object;
- emitted same-desc objects;
- residual same-desc objects;
- x1 candidate peaks when available.

Legends must not cover image content. Put legends in side panels or below images.

- [ ] **Step 4: Implement semantic status gates**

Status must reject:

- missing required artifacts;
- legacy A3.1 labels;
- wrong checkpoint roles;
- row-count mismatch between shard summaries and merged rows;
- `*.inprogress` leftovers;
- missing template contract;
- missing data-root audit;
- missing FN universe;
- missing FN probe summaries.

- [ ] **Step 5: Verify Task 7**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider \
  tests/analysis/sorted_random_no_newline_phenotype/test_merge_report.py \
  tests/analysis/sorted_random_no_newline_phenotype/test_status.py \
  tests/analysis/sorted_random_no_newline_phenotype/test_gallery.py
```

Run a mocked end-to-end materialization test over tiny JSONL fixtures and verify `status.py` returns `final_artifacts_present` only when all required A3.2 surfaces exist.

---

### Task 8: Runner, CLI, And Tmux Orchestration

**Files:**
- Create: `src/analysis/sorted_random_no_newline_phenotype/runner.py`
- Create: `scripts/analysis/sorted_random_no_newline_phenotype/run.py`
- Create: `scripts/analysis/sorted_random_no_newline_phenotype/status.py`
- Create: `scripts/analysis/sorted_random_no_newline_phenotype/launch_a3_2_tmux.sh`
- Test: `tests/analysis/sorted_random_no_newline_phenotype/test_runner_cli.py`

- [ ] **Step 1: Implement stage names**

Required stages:

- `data_root_audit`
- `prefix_state_index`
- `validate`
- `paired_checkpoint_probe`
- `prefix_merge`
- `prefix_report`
- `prefix_gallery`
- `native_rollout`
- `rollout_phenotype`
- `fn_case_index`
- `fn_hint_probe`
- `fn_merge`
- `fn_report`
- `fn_gallery`
- `finalize`

Each stage must support `--dry-run`. GPU stages must require explicit shard or launch context.

- [ ] **Step 2: Implement dry-run and CPU gates**

Before any GPU launch:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/analysis/sorted_random_no_newline_phenotype/run.py \
  --config configs/analysis/sorted_random_no_newline_phenotype/fullobj_purece_random_vs_sorted_ckpt3668_phase_a3_2.yaml \
  --stages data_root_audit,prefix_state_index,validate \
  --dry-run
```

Then:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/analysis/sorted_random_no_newline_phenotype/run.py \
  --config configs/analysis/sorted_random_no_newline_phenotype/fullobj_purece_random_vs_sorted_ckpt3668_phase_a3_2.yaml \
  --stages data_root_audit,prefix_state_index,validate
```

Required status before GPU:

```json
{
  "stage_status": "index_ready_pending_gpu",
  "launch_eligible": true,
  "failed_launch_gates": [],
  "expected_shards": 8,
  "planned_prefix_state_rows": 4096
}
```

- [ ] **Step 3: Implement GPU preflight and launch**

`launch_a3_2_tmux.sh` must:

- refuse to launch if target GPUs have compute apps unless explicitly overridden;
- refuse to launch if status gate is not ready;
- refuse to set `ARTIFACT_ROOT` separately from config;
- validate `len(GPU_IDS) == sampling.num_shards`;
- launch paired prefix readout as 8 single-GPU shards;
- launch native rollouts only after prefix shards are healthy or in an explicitly separate window;
- launch FN probes as 8 single-GPU shards over selected FN cases;
- write `shard_pids.tsv`, `shard_status.log`, and per-stage logs.

The default should be conservative while GPUs are currently busy. Do not use `SKIP_GPU_PREFLIGHT=1` for the first real A3.2 run.

- [ ] **Step 4: Suggested 8-GPU resource schedule**

When GPUs are free:

1. Paired prefix readout: 8 shards, one GPU each.
2. Native rollout: run random and sorted concurrently as two 4-GPU jobs if the infer runner supports partitioned `CUDA_VISIBLE_DEVICES`; otherwise run sequential 8-GPU jobs.
3. FN hint-ladder: 8 shards over `(checkpoint_role, fn_case_id)`; balance random/sorted cases across GPUs.

This uses all 8 cards for analysis throughput without launching production training.

- [ ] **Step 5: Verify Task 8**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider \
  tests/analysis/sorted_random_no_newline_phenotype/test_runner_cli.py
```

Dry-run:

```bash
DRY_RUN=1 \
CONFIG=/data/CoordExp/.worktrees/fn-rescue-attention-probes/configs/analysis/sorted_random_no_newline_phenotype/fullobj_purece_random_vs_sorted_ckpt3668_phase_a3_2.yaml \
SESSION=sorted_random_no_newline_a3_2_ckpt3668 \
bash /data/CoordExp/.worktrees/fn-rescue-attention-probes/scripts/analysis/sorted_random_no_newline_phenotype/launch_a3_2_tmux.sh
```

Expected: no GPU process starts; printed plan uses A3.2 root, ckpt3668 roles, 8 shards, and no A3.1 labels.

---

### Task 9: Smoke Then Full A3.2 Run

**Files:**
- Create: `configs/analysis/sorted_random_no_newline_phenotype/fullobj_purece_random_vs_sorted_ckpt3668_phase_a3_2_smoke.yaml`
- No code changes unless smoke exposes a bug.

- [ ] **Step 1: Create a smoke config**

Use the same schema with:

- `sampling.max_prefix_states: 64`
- `rollout.limit_images: 32`
- `fn_probe.max_fn_objects_per_checkpoint: 16`
- `sampling.num_shards: 8`
- separate smoke artifact root ending in `_smoke`

- [ ] **Step 2: Run all non-GPU tests**

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider \
  tests/analysis/sorted_random_no_newline_phenotype
```

Compile:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m py_compile \
  $(find src/analysis/sorted_random_no_newline_phenotype scripts/analysis/sorted_random_no_newline_phenotype -name '*.py' -print)
```

- [ ] **Step 3: Run smoke CPU gate**

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/analysis/sorted_random_no_newline_phenotype/run.py \
  --config configs/analysis/sorted_random_no_newline_phenotype/fullobj_purece_random_vs_sorted_ckpt3668_phase_a3_2_smoke.yaml \
  --stages data_root_audit,prefix_state_index,validate
```

- [ ] **Step 4: Run smoke tmux when GPUs are free**

```bash
CONFIG=/data/CoordExp/.worktrees/fn-rescue-attention-probes/configs/analysis/sorted_random_no_newline_phenotype/fullobj_purece_random_vs_sorted_ckpt3668_phase_a3_2_smoke.yaml \
SESSION=sorted_random_no_newline_a3_2_ckpt3668_smoke \
GPU_IDS="0 1 2 3 4 5 6 7" \
bash /data/CoordExp/.worktrees/fn-rescue-attention-probes/scripts/analysis/sorted_random_no_newline_phenotype/launch_a3_2_tmux.sh
```

Smoke acceptance:

- all stages complete;
- no legacy labels;
- merged row counts match shard row counts;
- no `*.inprogress`;
- `report.md` and galleries exist;
- `fn_probe/fn_cases.jsonl` and `fn_probe/fn_probe_rows.jsonl` exist;
- status reports final A3.2 readiness.

- [ ] **Step 5: Launch full A3.2 after smoke passes**

```bash
CONFIG=/data/CoordExp/.worktrees/fn-rescue-attention-probes/configs/analysis/sorted_random_no_newline_phenotype/fullobj_purece_random_vs_sorted_ckpt3668_phase_a3_2.yaml \
SESSION=sorted_random_no_newline_a3_2_ckpt3668 \
GPU_IDS="0 1 2 3 4 5 6 7" \
bash /data/CoordExp/.worktrees/fn-rescue-attention-probes/scripts/analysis/sorted_random_no_newline_phenotype/launch_a3_2_tmux.sh
```

Do not use `ALLOW_OVERWRITE=1` for the first full run. If a rerun is needed, write the reason in the run notes and verify the root is safe to overwrite.

---

### Task 10: Final Analysis And Decision Report

**Files:**
- Create: `progress/diagnostics/2026-06-04_a3_2_sorted_vs_random_no_newline_results.md`
- Create or update: `docs/superpowers/handoffs/2026-06-04-a3-2-next-direction-brief.md`

- [ ] **Step 1: Analyze prefix-state readout**

Report:

- boundary winner class rates;
- residual-vs-EOS margins;
- residual-vs-winner margins;
- strict R95 x1 evidence;
- broad24-only diagnostic rate;
- same-desc vs different-desc strata;
- sorted-minus-random deltas.

Keep this interpretation scoped to canonical sorted teacher-prefix readout.

- [ ] **Step 2: Analyze native rollout phenotype**

Report:

- class-any recall;
- instance recall;
- class-found-but-instance-missed rate;
- FN rates by same-desc count/object count/area/split;
- duplicate side labels;
- parse/invalid counters;
- EOS-after-partial-coverage.

Keep this interpretation separate from paired prefix-state readout.

- [ ] **Step 3: Analyze FN probe**

Report:

- shared FN, random-only FN, sorted-only FN;
- multi-axis FN mechanism counts;
- prefix-suppression flip matrix;
- residual-accounting failures restricted to same-desc emitted-plus-residual cases;
- desc selection vs coord binding;
- not-rescued-under-valid-desc-x1-controls;
- invalid/skipped probe rate.

Do not overcall `vision_unreachable_fn`; use conservative labels unless all controls are valid.

- [ ] **Step 4: Write algorithm-design implications as hypotheses, not conclusions**

Allowed forms:

- "This evidence is consistent with..."
- "Under the A3.2 scope, the sorted checkpoint shows..."
- "The next algorithm should target... if this pattern holds under a targeted follow-up."

Avoid:

- "proved";
- "root cause";
- "visual encoder failed";
- "sorted solves recall";
- "multiple positive is unnecessary".

- [ ] **Step 5: Final verification**

Run:

```bash
python scripts/analysis/sorted_random_no_newline_phenotype/status.py \
  --artifact-root /data/CoordExp/outputs/analysis/autoreg_object_rollout/sorted_random_no_newline_phenotype/fullobj_purece_random_vs_sorted_ckpt3668_phase_a3_2
```

Expected:

- `stage_status: final_artifacts_present`;
- all required artifact counts nonzero or explicitly explained;
- semantic labels match A3.2;
- no legacy A3.1 labels;
- row-count consistency passes.

## Stop Conditions

Stop and ask for user judgment only if:

- exact checkpoint paths are missing or incompatible;
- local data-root mismatch becomes non-negligible after `data_root_audit.json`;
- GPU availability remains blocked after CPU gates pass and user wants scheduling help;
- FN probe invalid/skipped rate is high enough to threaten interpretation;
- the implementation would require changing training data contracts or upstream model files.

## Done Criteria

A3.2 is done only when all of these exist and status passes:

- `data_root_audit.json`
- `prefix_state_index_summary.json`
- `sample_manifest.json`
- prefix paired readout shard artifacts for 8 shards
- merged prefix `summary.json` and `report.md`
- native rollout artifacts for both checkpoints
- `rollout/rollout_summary.json`
- `fn_probe/fn_case_universe.jsonl`
- `fn_probe/fn_cases.jsonl`
- `fn_probe/fn_probe_rows.jsonl`
- `fn_probe/fn_candidate_scores.jsonl`
- `fn_probe/fn_slot_evidence.jsonl`
- `fn_probe/fn_bucket_summary.json`
- `fn_probe/fn_prefix_sensitivity.json`
- `fn_probe/fn_slot_rescue_summary.json`
- `gallery/index.md`
- `fn_probe/gallery/index.md`
- `progress/diagnostics/2026-06-04_a3_2_sorted_vs_random_no_newline_results.md`

Final report must state:

- changed files;
- tests and commands run;
- smoke artifact root;
- full artifact root;
- tmux session names;
- residual risks;
- whether full GPU run is complete or still running.
