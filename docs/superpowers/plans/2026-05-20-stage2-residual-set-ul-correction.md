# Stage-2 Residual-Set UL Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current partial/legacy Stage-2 residual-set path with the approved offline self-prefix residual-set correction objective, including strict prepared-rollout input, shared target IR atoms, valid-action transitions, dirty-prefix recovery, and conservative unlabeled-object promotion.

**Architecture:** Keep the stable semantics in OpenSpec and implement them through a small number of reusable boundaries: strict config parsing, prepared rollout attempts, a training-side template/span boundary adapter, a residual-state scanner, UL consensus, shared `SupervisionAtom` compilation, and the existing teacher-forcing objective runner. Treat the current residual-set code as migration material, not as final contract, because it still contains old `bbox_tail_from_anchor`, `num_rollouts`, `ul_geometry`, and artifact-policy assumptions.

**Tech Stack:** Python dataclasses, PyTorch logits math, CoordExp Stage-2 AB trainer, shared `src/training/teacher_forcing` IR/probability helpers, Stage-1 detection template/tokenization helpers, YAML config schema, and pytest under `conda run -n ms`.

---

## Scope And Gates

Worktree:

```text
/data/CoordExp/.worktrees/unified-training-infra-refactor
```

Normative OpenSpec:

```text
openspec/changes/add-stage2-residual-set-ul-correction/
```

Decision log:

```text
progress/explorations/2026-05-20_stage2_residual_set_self_prefix_ul_redesign.md
```

Execution rules:

- Work only under `/data/CoordExp/.worktrees/unified-training-infra-refactor`.
- Do not edit `/data/CoordExp` root `main`.
- Do not touch upstream HF/Qwen model files.
- Do not implement code until the user approves this audited plan.
- Do not reintroduce `loss_duplicate_burst_unlikelihood`, `duplicate_unlikelihood`, `bbox_geo`, `bbox_size_aux`, `coord_reg`, `coord_gate`, `text_gate`, coordinate regression, bbox geometry aux, geometry regularizers, or raw-rollout coordinate repair.
- Preserve hard SFT and current `stage2_trie_ce` as baselines.
- `residual_set_correction` is explicit opt-in and consumes offline prepared rollout records.
- New v1 data requires `response_token_ids`; raw-text re-encode is legacy-only.
- `<|im_end|>` is the STOP/EOS token; `<|endoftext|>` is padding only.
- Keep checkpoint-compatible Stage-1 newline/separator rendering in v1; do not force the no-newline grammar in this refactor.
- `target_position` is the supervised label-token position; `logit_position + 1 == target_position` must be verified.
- K rollout attempts are K independent self-prefix samples after exact duplicate dedup; they are not averaged pseudo-labels.
- After each task, run the listed narrow tests before the next task.

Review boundary before implementation:

- This plan intentionally describes work that is not implemented yet.
- Current code is expected to still contain old residual-set behavior before this plan is executed.
- Pre-implementation review should flag a P0/P1 only when this plan omits, weakens, or contradicts a required implementation step.
- Do not treat the current absence of `prepared_rollout_jsonl` runtime wiring, the producer script, schema migration, smoke YAML migration, or per-attempt sequence fan-out as a plan blocker when the plan already assigns those edits and tests.

## File Map

Create:

- `src/training/span_adapters/residual_boundary.py`
  Thin Stage-2 residual-set facade over existing detection rendering/tokenization and `EncodedDetectionView`. This is the one justified new source file because it replaces several ad hoc compact-string span builders without creating a second detection-level span contract.

- `scripts/tools/prepare_stage2_residual_rollouts.py`
  Offline producer for prepared residual-set rollout JSONL. This is the one justified new script because v1 explicitly separates rollout generation from training and the smoke needs a reproducible producer command.

- `tests/test_stage2_residual_boundary_adapter.py`
  Tests assistant span extraction, object/separator/terminal spans, suffix slicing, checkpoint-compatible newline/separator behavior, and no schema duplication against the existing `TokenizedDetectionExample` / `EncodedDetectionView` contract.

Modify:

- `src/config/schema.py`
  Replace old residual-set config keys with strict `prepared_rollout_jsonl` v1 contract and explicit optional keys.

- `src/trainers/teacher_forcing/module_registry.py`
  Keep the teacher-forcing module catalog aligned with the strict residual-set v1 key set so schema defaults and registry validation cannot diverge.

- `configs/stage2_two_channel/smoke/compact_full_residual_set_ckpt3664_hf_1step.yaml`
  Migrate the existing smoke leaf away from `coord_span_policy`, `num_rollouts`, `ul_geometry`, and `artifact_policy`; explicitly disable inherited eval work.

- `src/trainers/stage2_two_channel/rollout_views.py`
  Own prepared rollout record parsing, strict `response_token_ids` validation, exact duplicate dedup, and legacy re-encode diagnostics.

- `src/trainers/stage2_two_channel/residual_set.py`
  Own `ResidualObject`, `ResidualState`, `ValidAction`, row classification, semantic residual scan, dirty-prefix recovery decisions, deterministic suffix ordering, and correction atom draft construction.

- `src/trainers/stage2_two_channel/ul_consensus.py`
  Replace geometry-heavy UL logic with strict same-desc cross-rollout consensus, gray-zone rejection, rollout-local member supervision, and flat artifact rows.

- `src/trainers/stage2_two_channel/teacher_forcing_adapter.py`
  Compile residual correction atom drafts into shared `TeacherForcingTargetIR` with causal alignment checks.

- `src/trainers/stage2_two_channel/target_builder.py`
  Route `residual_set_correction` to prepared-rollout/self-prefix target construction and remove old coordinate-repair target construction from this path.

- `src/trainers/stage2_two_channel.py`
  Bypass live rollout generation in residual-set offline mode, attach prepared attempts to each batch sample, carry compact residual diagnostics, and write `monitor_dumps/ul_clusters.jsonl`.

- `src/trainers/stage2_two_channel/types.py`
  Keep only the minimal residual metadata sidecar fields needed by the trainer and objective runner.

- `src/trainers/teacher_forcing/modules/residual_set_correction.py`
  Ensure the module applies standalone token-type loss plus inner valid-set/hard-path loss and normalizes by per-sequence atom-weighted means.

- `src/trainers/teacher_forcing/module_registry.py`
  Register only the strict v1 residual-set module config keys.

- `src/trainers/teacher_forcing/objective_pipeline.py`
  Preserve explicit module routing and avoid silent fallback to trie/hard-SFT behavior.

- `src/training/teacher_forcing/ir.py` and `src/training/teacher_forcing/validation.py`
  Modify only if the current IR validator lacks a reusable check required by residual-set atoms.

- `tests/test_stage2_ab_config_contract.py`
  Update residual-set config acceptance/rejection tests.

- `tests/test_teacher_forcing_loss_catalog.py`
  Verify residual-set registry keys and removed objective modules stay rejected.

- `tests/test_stage2_residual_boundary_adapter.py`
  Verify the training-side residual boundary adapter against existing tokenization/span contracts.

- `tests/test_stage2_residual_set_correction.py`
  Update state/valid-action/dirty-prefix/correction-event tests.

- `tests/test_stage2_residual_ul_consensus.py`
  Update UL consensus, gray-zone, duplicate-exclusion, rollout-local bbox, and artifact path tests.

- `tests/test_stage2_residual_set_loss_module.py`
  Update residual-set loss normalization, type-loss, valid-set, and alignment tests.

- `tests/test_stage2_ab_training.py`
  Keep only trainer/integration tests that prove metadata propagation, diagnostics, and artifact writing.

## Task 1: Strict Residual Config Contract

**Files:**

- Modify: `src/config/schema.py`
- Modify: `src/trainers/teacher_forcing/module_registry.py`
- Modify: `src/trainers/stage2_two_channel/target_builder.py`
- Modify: `src/trainers/stage2_two_channel.py`
- Modify: `tests/test_stage2_ab_config_contract.py`
- Modify: `tests/test_teacher_forcing_loss_catalog.py`
- Modify: `configs/stage2_two_channel/smoke/compact_full_residual_set_ckpt3664_hf_1step.yaml`

- [ ] **Step 1.1: Write failing config contract tests**

Add or rewrite tests in `tests/test_stage2_ab_config_contract.py`:

```python
def _payload_with_residual_set_config(config: dict) -> dict:
    raw = _make_stage2_training_payload()
    raw["stage2_ab"]["pipeline"]["objective"] = [
        {
            "name": "token_ce",
            "enabled": True,
            "weight": 1.0,
            "channels": ["A"],
            "application": {"preset": "anchor_text_only"},
            "config": {
                "desc_ce_weight": 1.0,
                "rollout_fn_desc_weight": 1.0,
                "rollout_global_prefix_struct_ce_weight": 1.0,
            },
        },
        {
            "name": "residual_set_correction",
            "enabled": True,
            "weight": 1.0,
            "channels": ["B"],
            "application": {"preset": "rollout_self_prefix"},
            "config": dict(config),
        },
    ]
    raw["stage2_ab"]["channel_b"]["pseudo_positive"] = {"enabled": False}
    return raw


def _load_stage2_payload(raw: dict) -> TrainingConfig:
    return TrainingConfig.from_mapping(raw, ConfigLoader.resolve_prompts(raw))


def _find_pipeline_objective(cfg: TrainingConfig, name: str):
    assert cfg.stage2_ab is not None
    matches = [objective for objective in cfg.stage2_ab.pipeline.objective if objective.name == name]
    assert len(matches) == 1
    return matches[0]


def test_residual_set_requires_prepared_rollout_jsonl() -> None:
    raw = _payload_with_residual_set_config(config={})
    with pytest.raises(
        ValueError,
        match=r"stage2_ab\.pipeline\.objective\[name=residual_set_correction\]\.config\.prepared_rollout_jsonl",
    ):
        _load_stage2_payload(raw)


def test_residual_set_accepts_minimal_prepared_rollout_config() -> None:
    raw = _payload_with_residual_set_config(
        config={"prepared_rollout_jsonl": "output/stage2/prepared_rollouts/train8.jsonl"}
    )
    cfg = _load_stage2_payload(raw)
    objective = _find_pipeline_objective(cfg, "residual_set_correction")
    assert objective.config["prepared_rollout_jsonl"] == "output/stage2/prepared_rollouts/train8.jsonl"
    assert objective.config["expected_num_rollouts"] == 4
    assert objective.config["base_seed"] == 17
    assert objective.config["lambda_type"] == 1.0
    assert objective.config["lambda_inner"] == 1.0
    assert objective.config["clean_gt_sft_mix"] == 0


@pytest.mark.parametrize(
    "bad_key,bad_value",
    [
        ("num_rollouts", 4),
        ("coord_span_policy", "bbox_tail_from_anchor"),
        ("coverage_strength", 0.0),
        ("ul_geometry", {"iou_min": 0.9}),
        ("artifact_policy", {"ul_clusters": "monitor_debug_smoke"}),
    ],
)
def test_residual_set_rejects_removed_config_keys(bad_key: str, bad_value: object) -> None:
    raw = _payload_with_residual_set_config(
        config={
            "prepared_rollout_jsonl": "output/stage2/prepared_rollouts/train8.jsonl",
            bad_key: bad_value,
        }
    )
    with pytest.raises(ValueError, match=bad_key):
        _load_stage2_payload(raw)


@pytest.mark.parametrize(
    "removed_module",
    ["loss_duplicate_burst_unlikelihood", "bbox_geo", "bbox_size_aux", "coord_reg", "coord_gate", "text_gate"],
)
def test_residual_set_rejects_removed_live_objective_modules(removed_module: str) -> None:
    raw = _payload_with_residual_set_config(
        config={"prepared_rollout_jsonl": "output/stage2/prepared_rollouts/train8.jsonl"}
    )
    raw["stage2_ab"]["pipeline"]["objective"].append(
        {
            "name": removed_module,
            "enabled": True,
            "weight": 1.0,
            "channels": ["B"],
            "config": {},
        }
    )
    with pytest.raises(ValueError, match=removed_module):
        _load_stage2_payload(raw)
```

- [ ] **Step 1.2: Run tests and confirm failure**

Run:

```bash
conda run -n ms python -m pytest tests/test_stage2_ab_config_contract.py -k 'residual_set and (prepared_rollout_jsonl or removed_config_keys)' -q
```

Expected now: FAIL because the current schema still allows old residual-set keys and does not require `prepared_rollout_jsonl`.

- [ ] **Step 1.3: Implement strict schema**

In `src/config/schema.py` and `src/trainers/teacher_forcing/module_registry.py`:

- Replace `STAGE2_RESIDUAL_SET_CONFIG_KEYS` with the OpenSpec v1 set:
  `prepared_rollout_jsonl`, `expected_num_rollouts`, `base_seed`, `lambda_type`, `lambda_inner`, `fallback_loss_weight`, `lambda_ul_promoted`, `label_conflict_weight`, `commit_iou_threshold`, `duplicate_burst_iou_threshold`, `ul_cluster_iou_threshold`, `ul_gray_iou_low`, `ul_consensus_ratio`, `min_ul_valid_rollouts`, `clean_gt_sft_mix`, `strict_prepared_rollout_tokens`, `legacy_reencode_fallback`, `strict_builder_invariants`.
- Require `prepared_rollout_jsonl` before trainer initialization.
- Default `expected_num_rollouts=4`, `base_seed=17`, `lambda_type=1.0`, `lambda_inner=1.0`, `lambda_ul_promoted=0.5`, `label_conflict_weight=0.25`, `commit_iou_threshold=0.75`, `duplicate_burst_iou_threshold=0.95`, `ul_cluster_iou_threshold=0.9`, `ul_gray_iou_low=0.30`, `ul_consensus_ratio=1.0`, `min_ul_valid_rollouts=2`, `clean_gt_sft_mix=0`, `strict_prepared_rollout_tokens=True`, `legacy_reencode_fallback=False`, and `strict_builder_invariants=True`.
- Delete residual-set ownership of `channel_b.triage_posterior.num_rollouts`; keep that legacy count only for non-residual baseline paths.
- Remove old residual-set readers in `_channel_b_residual_set_correction_options()` and trainer setup. Runtime readers must use `expected_num_rollouts`, flat UL thresholds, and the canonical flat artifact path.
- Assert the parsed `objective.config` contains defaulted v1 keys and does not contain `num_rollouts`, `coord_span_policy`, `coverage_strength`, `ul_geometry`, or `artifact_policy`.

- [ ] **Step 1.4: Migrate the smoke YAML**

In `configs/stage2_two_channel/smoke/compact_full_residual_set_ckpt3664_hf_1step.yaml`:

```yaml
config:
  prepared_rollout_jsonl: output/stage2_ab/prepared_rollouts/train8_ckpt3664.jsonl
  expected_num_rollouts: 4
  base_seed: 17
  lambda_type: 1.0
  lambda_inner: 1.0
```

Remove old keys from the residual objective config: `num_rollouts`, `coord_span_policy`, `ul_geometry`, and `artifact_policy`.

Also override inherited eval work in the smoke leaf:

```yaml
training:
  eval_strategy: "no"
custom:
  val_sample_limit: 0
rollout_matching:
  eval_monitor_dump:
    enabled: false
  eval_detection:
    enabled: false
    materialize_artifacts: false
```

Add a config contract test for this smoke leaf asserting the config uses checkpoint-3664, has `eval_strategy == "no"`, `val_sample_limit == 0`, eval detection disabled, a `prepared_rollout_jsonl` path, and no removed residual-set keys.

- [ ] **Step 1.5: Verify**

Run:

```bash
conda run -n ms python -m pytest \
  tests/test_stage2_ab_config_contract.py \
  tests/test_teacher_forcing_loss_catalog.py \
  -k 'residual_set or removed or duplicate_burst_unlikelihood or bbox_geo or coord_reg' \
  -q
```

Expected: PASS.

## Task 2: Prepared Rollout Attempt Input

**Files:**

- Modify: `src/trainers/stage2_two_channel/rollout_views.py`
- Modify: `src/trainers/stage2_two_channel.py`
- Modify: `src/trainers/stage2_two_channel/target_builder.py`
- Create: `scripts/tools/prepare_stage2_residual_rollouts.py`
- Modify: `tests/test_stage2_residual_set_correction.py`
- Modify: `tests/test_stage2_ab_training.py`

- [ ] **Step 2.1: Add failing tests for prepared records**

Add tests that cover:

```python
def test_prepared_rollout_requires_response_token_ids_in_strict_mode() -> None:
    record = {
        "sample_id": "s0",
        "rollout_id": "r0",
        "raw_text": "<|object_ref_start|>person<|box_start|><|coord_1|><|coord_2|><|coord_3|><|coord_4|>",
        "decode_mode": "greedy",
    }
    with pytest.raises(ValueError, match="response_token_ids"):
        parse_prepared_rollout_attempt(record, strict_prepared_rollout_tokens=True)


def test_prepared_rollout_exact_dedup_uses_response_token_ids() -> None:
    attempts = [
        parse_prepared_rollout_attempt({"sample_id": "s0", "rollout_id": "a", "response_token_ids": [1, 2], "raw_text": "x", "decode_mode": "greedy"}),
        parse_prepared_rollout_attempt({"sample_id": "s0", "rollout_id": "b", "response_token_ids": [1, 2], "raw_text": "x changed", "decode_mode": "sample"}),
        parse_prepared_rollout_attempt({"sample_id": "s0", "rollout_id": "c", "response_token_ids": [1, 3], "raw_text": "y", "decode_mode": "sample"}),
    ]
    kept, stats = dedup_prepared_rollout_attempts(attempts, legacy_reencode_fallback=False)
    assert [attempt.rollout_id for attempt in kept] == ["a", "c"]
    assert stats["exact_duplicate_attempts"] == 1


@pytest.mark.parametrize("missing_key", ["generation_config_hash", "image_id", "image_path"])
def test_prepared_rollout_requires_replay_provenance(missing_key: str) -> None:
    record = {
        "sample_id": "s0",
        "image_id": "image-0",
        "image_path": "images/000000.jpg",
        "rollout_id": "r0",
        "response_token_ids": [1, 2, 3],
        "raw_text": "raw",
        "decode_mode": "greedy",
        "generation_config_hash": "sha256:abc",
    }
    record.pop(missing_key)
    with pytest.raises(ValueError, match=missing_key):
        parse_prepared_rollout_attempt(record, strict_prepared_rollout_tokens=True)
```

- [ ] **Step 2.2: Implement parser/dedup**

In `src/trainers/stage2_two_channel/rollout_views.py`, add or refactor:

- `PreparedRolloutAttempt`
- `parse_prepared_rollout_attempt(record, *, strict_prepared_rollout_tokens: bool)`
- `load_prepared_rollout_jsonl(path, *, strict_prepared_rollout_tokens: bool)`
- `dedup_prepared_rollout_attempts(attempts, *, legacy_reencode_fallback: bool)`

Required fields for new strict data: `sample_id`, `image_id`, `image_path`, `rollout_id`, `response_token_ids`, `raw_text`, `decode_mode`, and `generation_config_hash`. `sampling_seed` and extra decode metadata are optional but preserved when present. Missing detection list for a sample is a drop+diagnose condition.

- [ ] **Step 2.3: Wire prepared attempts into Stage-2 runtime**

Before residual target construction:

- resolve `prepared_rollout_jsonl` from the materialized config;
- load records once per training process;
- group attempts by `sample_id` and/or stable image/sample provenance;
- attach grouped attempts to each B-channel batch sample;
- exact-dedup attempts by `response_token_ids`;
- diagnose `K_total`, `K_after_dedup`, `K_valid`, and dropped reasons;
- enforce `expected_num_rollouts` as a diagnostic/default expectation, not as live rollout ownership;
- bypass `_prepare_samples_for_rollout` and live rollout backend generation when `residual_set_correction` is active.

Fan-out requirement:

- one retained prepared rollout attempt becomes one Channel-B residual training sequence/segment, or an explicitly equivalent per-attempt segment record;
- build residual events per retained attempt, never only from the first `anchor_view`;
- each segment metadata carries `sample_id`, `image_id`, `rollout_id`, dedup status, `K_total`, `K_after_dedup`, and `K_valid`;
- `stage2_ab/channel_b/residual_set/sequence_count` equals the retained prepared-attempt count that produced active residual atoms/sequences.

Add a trainer-level test in `tests/test_stage2_ab_training.py` that stubs `_prepare_samples_for_rollout` and `_rollout_many` to raise, feeds two nonduplicate prepared attempts for one sample, and proves the residual-set path produces two residual sequences with distinct `rollout_id` metadata.

- [ ] **Step 2.4: Implement the offline producer CLI**

Create `scripts/tools/prepare_stage2_residual_rollouts.py` with this operator contract:

```bash
PYTHONPATH=. conda run -n ms python scripts/tools/prepare_stage2_residual_rollouts.py \
  --config configs/stage2_two_channel/smoke/compact_full_residual_set_ckpt3664_hf_1step.yaml \
  --out output/stage2_ab/prepared_rollouts/train8_ckpt3664.jsonl \
  --train-sample-limit 8 \
  --expected-num-rollouts 4 \
  --seed 17 \
  --greedy-rollouts 1 \
  --sampling-rollouts 3 \
  --include-debug-cases invalid_bbox_dirty_prefix,exact_duplicate_attempt
```

The script must insert the repo root into `sys.path` like existing `scripts/tools/*` scripts, or the documented `PYTHONPATH=.` command must be sufficient. It must resolve the model checkpoint and dataset from the materialized config, write required provenance fields, keep checkpoint-compatible newline/template output, and fail before generation if the resolved checkpoint path does not contain `checkpoint-3664` for this smoke config.

- [ ] **Step 2.5: Verify**

Run:

```bash
conda run -n ms python -m pytest \
  tests/test_stage2_residual_set_correction.py \
  tests/test_stage2_ab_training.py \
  -k 'prepared_rollout or dedup or offline_residual_set' \
  -q
```

Expected: PASS.

## Task 3: Residual Boundary Adapter

**Files:**

- Create: `src/training/span_adapters/residual_boundary.py`
- Create: `tests/test_stage2_residual_boundary_adapter.py`
- Modify: `src/trainers/stage2_two_channel/target_builder.py`
- Modify: `src/trainers/stage2_two_channel/teacher_forcing_adapter.py`

- [ ] **Step 3.1: Write failing adapter tests**

Add tests that prove:

- assistant spans are found from Stage-1 rendering/tokenization, not hand-authored strings;
- newline/separator tokens follow `get_detection_template("compact_full")` and the active tokenizer;
- suffix slicing from object boundary removes trailing incomplete object spans;
- no deterministic schema token is duplicated when prefix plus generated suffix are joined;
- adapter spans equal the existing `TokenizedDetectionExample` object/separator/terminal/assistant spans and can be projected into `EncodedDetectionView`.

Use function names:

```python
def test_residual_boundary_adapter_slices_suffix_from_object_boundary() -> None: ...
def test_residual_boundary_adapter_drops_trailing_incomplete_object_span() -> None: ...
def test_residual_boundary_adapter_matches_tokenized_detection_spans() -> None: ...
def test_residual_boundary_adapter_validates_no_schema_duplication() -> None: ...
```

- [ ] **Step 3.2: Implement adapter**

In `src/training/span_adapters/residual_boundary.py`, implement a small API that wraps existing surfaces instead of duplicating them:

```python
@dataclass(frozen=True)
class ResidualBoundarySlice:
    tokenized: TokenizedDetectionExample
    encoded_view: EncodedDetectionView
    suffix_start: int
    suffix_input_ids: tuple[int, ...]
    retained_prefix_input_ids: tuple[int, ...]


class ResidualBoundaryAdapter:
    def __init__(self, *, tokenizer: Any, template_mode: str = "compact_full") -> None: ...
    def render_objects(self, objects: Sequence[Mapping[str, Any]]) -> RenderedAssistantSequence: ...
    def tokenize_rendered(self, rendered: RenderedAssistantSequence) -> TokenizedDetectionExample: ...
    def slice_from_boundary(self, rendered: RenderedAssistantSequence, *, boundary: str, object_index: int | None = None) -> ResidualBoundarySlice: ...
```

Use existing surfaces:

- `src/detection/template.py::get_detection_template`
- `RenderedAssistantSequence.object_entries`
- `RenderedAssistantSequence.separator_spans`
- `src/detection/tokenization.py::tokenize_rendered_detection_conversation`
- `src/training/encoding/view.py::EncodedDetectionView`
- existing compact span adapters under `src/training/span_adapters/`

- [ ] **Step 3.3: Replace ad hoc compact builders**

In `src/trainers/stage2_two_channel/target_builder.py`, remove residual-set dependence on `_render_compact_objects`, `_build_compact_prefix_text_data`, `_compact_object_and_desc_spans`, and ad hoc suffix-token slicing. Keep those helpers only if non-residual legacy paths still call them.

- [ ] **Step 3.4: Verify**

Run:

```bash
conda run -n ms python -m pytest tests/test_stage2_residual_boundary_adapter.py tests/test_stage2_teacher_forcing_adapter_contract.py -q
```

Expected: PASS.

## Task 4: Residual State, ValidAction, And Dirty-Prefix Scan

**Files:**

- Modify: `src/trainers/stage2_two_channel/residual_set.py`
- Modify: `tests/test_stage2_residual_set_correction.py`

- [ ] **Step 4.1: Write failing state-transition tests**

Add tests for:

```python
def test_x1_valid_action_commits_subsequent_bbox_to_same_object() -> None: ...
def test_invalid_bbox_row_is_dirty_context_and_does_not_update_remaining_set() -> None: ...
def test_trailing_incomplete_object_is_removed_to_last_stable_boundary() -> None: ...
def test_spatial_wrong_desc_conflict_emits_low_weight_desc_atom_when_span_reliable() -> None: ...
def test_spatial_wrong_desc_conflict_records_no_atom_reason_when_span_unreliable() -> None: ...
def test_duplicate_burst_is_uncommitted_and_cannot_vote_for_ul() -> None: ...
def test_malformed_span_context_has_no_atoms_or_type_loss() -> None: ...
def test_row_commitment_uses_deterministic_gt_before_ul_tiebreak() -> None: ...
```

Each test must assert the remaining-object set before and after the row scan.

- [ ] **Step 4.2: Refactor `ValidAction`**

In `src/trainers/stage2_two_channel/residual_set.py`:

- Ensure every ambiguous token choice is represented as a `ValidAction`.
- Include materialized transition data to drive later suffix construction. The selected action consumed by atom/suffix builders must expose `next_state` directly or through a `ResolvedValidAction` wrapper that stores the transition result. Bare recomputation from token ids is not allowed downstream.
- Ensure `valid_token_ids == {action.token_id for action in valid_actions}`.
- Ensure selected x1 action filters active candidates before y1/x2/y2 atoms are built.
- Fail fast in strict mode if a selected action has missing, invalid, or empty `next_state` while objects remain.

- [ ] **Step 4.3: Remove coordinate repair semantics**

Delete or deprecate residual correction kinds and code paths that imply raw coordinate repair:

- `matched_object_repair` as an active objective event;
- `bbox_tail_from_anchor`;
- nearest-GT coordinate fixup;
- low-IoU coordinate refinement.
- coordinate-neighborhood tolerance as a repair target;
- sort/clamp repair for invalid xyxy boxes.

Keep invalid bbox diagnostics, but do not emit coordinate-tail supervision from invalid rows.

- [ ] **Step 4.4: Implement dirty-prefix recovery**

Rules:

- legal row + exact normalized desc + IoU `>=0.75`: commit and remove that object;
- duplicate burst: uncommitted, excluded from UL, no unlikelihood;
- invalid geometry: uncommitted dirty context, no IoU;
- malformed middle span with reliable resync: masked dirty context, continue;
- unreliable resync: cut to last stable boundary or drop sample;
- trailing incomplete object: remove the whole incomplete object span from `<|object_ref_start|>`.
- rollout prefix labels are masked by default;
- malformed retained spans get no atoms and no type loss inside the span;
- no atom may cross prompt/assistant boundaries, padding bounds, or an unreliable resync boundary.

Commitment tie-breaks:

- consider labeled GT candidates before promoted UL candidates;
- among passing same-desc candidates, choose IoU descending, center-distance ascending, then stable object id ascending;
- remove exactly the selected object id from the remaining set.

Spatial wrong-description conflicts:

- desc-agnostic IoU `>=0.75` plus desc mismatch is `spatial_wrong_desc_conflict`;
- it is uncommitted, does not update remaining state, and is never a UL candidate;
- reliable divergence emits an earliest-divergence desc atom with `label_conflict_weight=0.25`;
- unreliable divergence emits no atom and records a no-atom diagnostic reason.

- [ ] **Step 4.5: Verify**

Run:

```bash
conda run -n ms python -m pytest tests/test_stage2_residual_set_correction.py -q
```

Expected: PASS.

## Task 5: UL Consensus And Artifacts

**Files:**

- Modify: `src/trainers/stage2_two_channel/ul_consensus.py`
- Modify: `src/trainers/stage2_two_channel.py`
- Modify: `tests/test_stage2_residual_ul_consensus.py`

- [ ] **Step 5.1: Write failing UL tests**

Add or update tests:

```python
def test_ul_consensus_requires_distinct_rollout_ids() -> None: ...
def test_ul_consensus_ratio_one_promotes_all_valid_rollout_members() -> None: ...
def test_promoted_ul_training_uses_rollout_local_member_bbox_not_medoid() -> None: ...
def test_near_gt_gray_zone_rejects_ul_candidate() -> None: ...
def test_duplicate_burst_members_cannot_vote_for_ul() -> None: ...
def test_spatial_wrong_desc_conflict_cannot_vote_for_ul() -> None: ...
def test_ul_clusters_artifact_uses_monitor_dumps_relative_path() -> None: ...
```

- [ ] **Step 5.2: Simplify UL config**

Remove v1 dependence on `ULGeometryConfig` fields that are not in OpenSpec. Keep only:

- `ul_cluster_iou_threshold=0.9`;
- `ul_gray_iou_low=0.30`;
- `ul_consensus_ratio=1.0`;
- `min_ul_valid_rollouts=2`;
- `lambda_ul_promoted=0.5`.

- [ ] **Step 5.3: Implement consensus**

Rules:

- Candidate must be legal, unmatched, non-duplicate, and same normalized desc within its cluster.
- Candidate must not be a `spatial_wrong_desc_conflict`.
- Support must come from distinct rollout ids.
- Denominator is `K_valid`, not raw K when attempts were dropped before UL eligibility.
- Consensus admits the cluster only.
- Each retained rollout attempt trains against its own local promoted UL member bbox/desc.
- Medoid/representative bbox is review metadata only.

- [ ] **Step 5.4: Emit flat review artifact**

Write:

```text
<run_dir>/monitor_dumps/ul_clusters.jsonl
```

Each row contains global step, sample/image provenance, desc, representative bbox, member bboxes, member rollout ids, `K_total`, `K_valid`, support ratio, decision, and rejection/promotion reason. Do not generate PNGs by default.

If current integration still resolves a nested root such as `monitor_dumps/stage2_ul_consensus/step_<global_step>/`, change the trainer root resolver so this writer receives `<run_dir>/monitor_dumps` and carries step/sample provenance inside each JSONL row. Add an integration assertion in `tests/test_stage2_ab_training.py`; a low-level `write_ul_clusters_artifact(root)` unit test is not enough.

- [ ] **Step 5.5: Verify**

Run:

```bash
conda run -n ms python -m pytest tests/test_stage2_residual_ul_consensus.py -q
```

Expected: PASS.

## Task 6: Compile Correction Atoms Into Shared IR

**Files:**

- Modify: `src/trainers/stage2_two_channel/teacher_forcing_adapter.py`
- Modify: `src/trainers/stage2_two_channel/target_builder.py`
- Modify: `src/trainers/stage2_two_channel/types.py`
- Modify: `tests/test_stage2_teacher_forcing_adapter_contract.py`
- Modify: `tests/test_stage2_ab_training.py`

- [ ] **Step 6.1: Write failing IR compilation tests**

Tests must assert:

- `atom.logit_position + 1 == atom.target_position`;
- `input_ids[atom.target_position] == atom.selected_token_id`;
- `selected_token_id in valid_token_ids`;
- `selected_token_role in allowed_token_roles`;
- all eligible non-conflicting correction atoms from one rollout attempt stay in one sequence;
- same-logit identical targets merge provenance;
- conflicting same-logit targets diagnose and do not silently pick a random target.

- [ ] **Step 6.2: Compile from selected `ValidAction`**

For every correction atom draft:

- `valid_token_ids` comes from action token ids;
- `selected_token_id` comes from the selected action;
- later suffix construction uses the selected action transition state;
- singleton branches are represented as singleton valid sets;
- STOP/EOS is a singleton valid set only when no remaining objects exist.

- [ ] **Step 6.3: Preserve sequence boundaries**

Carry rollout-attempt sequence identity through sidecar metadata so the loss module can compute per-sequence weighted means before batch mean. Do not normalize clean/dirty/UL buckets separately.

Required runtime shape:

- a sample with two retained nonduplicate prepared attempts yields two residual IR sidecars/segments;
- each sidecar keeps the originating `rollout_id`;
- artifacts and metrics can be traced back to the originating attempt;
- the first/anchor attempt has no privileged training role after dedup.

- [ ] **Step 6.4: Verify**

Run:

```bash
conda run -n ms python -m pytest tests/test_stage2_teacher_forcing_adapter_contract.py tests/test_stage2_ab_training.py -k 'residual_set or target_ir or logit_position' -q
```

Expected: PASS.

## Task 7: Residual-Set Loss Module And Metrics

**Files:**

- Modify: `src/trainers/teacher_forcing/modules/residual_set_correction.py`
- Modify: `src/trainers/teacher_forcing/module_registry.py`
- Modify: `src/trainers/teacher_forcing/objective_pipeline.py`
- Modify: `tests/test_stage2_residual_set_loss_module.py`
- Modify: `tests/test_teacher_forcing_loss_catalog.py`

- [ ] **Step 7.1: Write failing loss tests**

Update tests so residual-set v1 proves:

- type loss is standalone and default-on through `lambda_type=1.0`;
- inner loss uses valid-set marginal over atom `valid_token_ids`;
- singleton valid set equals hard CE;
- STOP singleton uses STOP role only;
- per-sequence loss is `sum(weight_i * loss_i) / sum(weight_i)`;
- batch loss is mean of sequence losses;
- zero active atoms contribute zero residual-set loss and explicit metrics.

- [ ] **Step 7.2: Implement module config**

Expose only residual-set v1 config keys from Task 1. Do not keep old `coverage_strength`, `coord_span_policy`, `ul_geometry`, or `artifact_policy` as live residual-set config. Add registry/catalog tests proving these removed module names are rejected in residual-set live configs: `loss_duplicate_burst_unlikelihood`, `bbox_geo`, `bbox_size_aux`, `coord_reg`, `coord_gate`, and `text_gate`.

- [ ] **Step 7.3: Implement metrics**

Emit compact keys under `stage2_ab/channel_b/residual_set/`, including:

- `sequence_count`;
- `atom_count`;
- `atom_weight_sum`;
- `raw_atom_loss_sum`;
- `sequence_loss`;
- `type_loss`;
- `inner_loss`;
- `wrong_type_mass`;
- `valid_set_mass`;
- `dirty_prefix_sequence_count`;
- `committed_gt_rows`;
- `committed_ul_rows`;
- `pending_ul_candidates`;
- `promoted_ul_clusters`;
- `uncommitted_invalid_geometry`;
- `uncommitted_malformed`;
- `uncommitted_duplicate`;
- `uncommitted_fp_or_unpromoted`;
- `spatial_wrong_desc_conflict`;
- `label_conflict_atoms`;
- `label_conflict_no_atom`;
- `eos_targets`;
- `continue_targets`;
- decode-mode sliced counts;
- `dirty_prefix_reencoded`;
- `clean_success_skipped`.

- [ ] **Step 7.4: Verify**

Run:

```bash
conda run -n ms python -m pytest tests/test_stage2_residual_set_loss_module.py tests/test_teacher_forcing_loss_catalog.py -q
```

Expected: PASS.

## Task 8: Integration, Docs, And Smoke

**Files:**

- Modify: `src/trainers/stage2_two_channel/target_builder.py`
- Modify: `src/trainers/stage2_two_channel/objective_runner.py`
- Modify: `src/trainers/stage2_two_channel.py`
- Modify: `docs/training/STAGE2_RUNBOOK.md` if stable behavior changed
- Modify: `docs/ARTIFACTS.md` if artifact names changed
- Modify: `openspec/changes/add-stage2-residual-set-ul-correction/tasks.md`
- Create: `scripts/tools/prepare_stage2_residual_rollouts.py`

- [ ] **Step 8.1: Run integrated unit suite**

Run:

```bash
conda run -n ms python -m pytest \
  tests/test_stage2_ab_config_contract.py \
  tests/test_stage2_residual_boundary_adapter.py \
  tests/test_stage2_residual_set_correction.py \
  tests/test_stage2_residual_ul_consensus.py \
  tests/test_stage2_residual_set_loss_module.py \
  tests/test_stage2_teacher_forcing_adapter_contract.py \
  tests/test_stage2_ab_training.py \
  -q
```

Expected: PASS.

- [ ] **Step 8.2: Validate OpenSpec**

Run:

```bash
openspec validate add-stage2-residual-set-ul-correction --type change --strict --no-interactive
```

Expected: `Change 'add-stage2-residual-set-ul-correction' is valid`.

- [ ] **Step 8.3: Prepare and preflight a tiny offline rollout JSONL**

Run the offline prepared-rollout producer from Task 2.4. It must write:

```text
output/stage2_ab/prepared_rollouts/train8_ckpt3664.jsonl
```

Minimum fixture composition:

- one clean committed GT row;
- one invalid bbox dirty-prefix attempt that must not update the remaining set;
- one exact duplicate rollout attempt so dedup is exercised;
- one K-valid sample with no UL promotion;
- one optional sample with a strict UL-consensus candidate when the smoke is intended to write `monitor_dumps/ul_clusters.jsonl`.

Generate K attempts before exact-token dedup. Retained attempts may be `< K`; diagnostics must report both pre-dedup and post-dedup counts.

Run this CPU preflight before GPU smoke:

```bash
conda run -n ms python - <<'PY'
import json
from pathlib import Path
from collections import defaultdict

from src.config.loader import ConfigLoader

path = Path("output/stage2_ab/prepared_rollouts/train8_ckpt3664.jsonl")
config_path = Path("configs/stage2_two_channel/smoke/compact_full_residual_set_ckpt3664_hf_1step.yaml")
if not path.is_file():
    raise SystemExit(f"missing prepared rollout JSONL: {path}")

cfg = ConfigLoader.load_materialized_training_config(str(config_path))
assert cfg.stage2_ab is not None
residual = [obj for obj in cfg.stage2_ab.pipeline.objective if obj.name == "residual_set_correction"]
if len(residual) != 1:
    raise SystemExit("expected exactly one residual_set_correction objective")
configured_path = Path(str(residual[0].config["prepared_rollout_jsonl"]))
if configured_path != path:
    raise SystemExit(f"config prepared_rollout_jsonl={configured_path} does not match preflight path={path}")

rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
if not rows:
    raise SystemExit("prepared rollout JSONL is empty")

required = {"sample_id", "image_id", "image_path", "rollout_id", "response_token_ids", "raw_text", "decode_mode", "generation_config_hash"}
missing = [(i, sorted(required - set(row))) for i, row in enumerate(rows) if required - set(row)]
if missing:
    raise SystemExit(f"missing required keys: {missing[:3]}")

def join_keys(row: dict, fallback_index: int | None = None) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for key in ("sample_id", "image_id", "image_path"):
        value = row.get(key)
        if value is not None:
            out.add((key, str(value)))
    if fallback_index is not None:
        out.add(("base_idx", str(fallback_index)))
    return out

train_path = Path(str(cfg.custom.train_jsonl))
if not train_path.is_file():
    raise SystemExit(f"train_jsonl not found for join preflight: {train_path}")
train_rows = []
limit = int(getattr(cfg.custom, "train_sample_limit", 8) or 8)
for idx, line in enumerate(train_path.read_text(encoding="utf-8").splitlines()):
    if idx >= limit:
        break
    if line.strip():
        train_rows.append(json.loads(line))
train_keys = set()
for idx, row in enumerate(train_rows):
    train_keys.update(join_keys(row, idx))
prepared_keys = set()
for row in rows:
    prepared_keys.update(join_keys(row))
if not train_keys & prepared_keys:
    raise SystemExit("prepared rows do not overlap selected train samples by sample_id/image_id/image_path/base_idx")

expected_k = int(residual[0].config.get("expected_num_rollouts", 4))
by_sample: dict[str, list[dict]] = defaultdict(list)
for row in rows:
    by_sample[str(row["sample_id"])].append(row)
if not any(len(group) == expected_k for group in by_sample.values()):
    raise SystemExit(f"no sample has expected_num_rollouts={expected_k} rows before exact dedup")

if not any(row.get("debug_case") == "invalid_bbox_dirty_prefix" for row in rows):
    raise SystemExit("prepared rollout JSONL must include invalid_bbox_dirty_prefix debug case")
duplicate_rows = [row for row in rows if row.get("debug_case") == "exact_duplicate_attempt"]
if not duplicate_rows:
    raise SystemExit("prepared rollout JSONL must include exact_duplicate_attempt debug case")
tokens_to_rollouts: dict[tuple[int, ...], set[str]] = defaultdict(set)
for row in duplicate_rows:
    tokens_to_rollouts[tuple(int(tok) for tok in row["response_token_ids"])].add(str(row["rollout_id"]))
if not any(len(rollout_ids) >= 2 for rollout_ids in tokens_to_rollouts.values()):
    raise SystemExit("exact_duplicate_attempt rows must include distinct rollout_ids with identical response_token_ids")

print(f"prepared_rows={len(rows)} path={path}")
PY
```

- [ ] **Step 8.4: Run smoke without eval_step**

Use the checkpoint family the user requested:

```text
et-rmp-ce-ckpt-3660+ / checkpoint-3664 base
```

Launch only after confirming the migrated YAML points at a real prepared rollout JSONL:

```bash
config=configs/stage2_two_channel/smoke/compact_full_residual_set_ckpt3664_hf_1step.yaml \
gpus=0,1,2,3 \
conda run -n ms bash scripts/train.sh
```

Expected for the first smoke:

- training starts from the ET-RMP-CE checkpoint-3664 base;
- the materialized config keeps `eval_strategy: "no"` and does not schedule `eval_step`;
- residual-set metric namespace appears under `stage2_ab/channel_b/residual_set/`;
- fixture-specific counters match the prepared input, including nonzero `dirty_prefix_sequence_count` and `uncommitted_invalid_geometry`;
- if the fixture includes a UL candidate, `monitor_dumps/ul_clusters.jsonl` exists and contains rows; otherwise the artifact is absent or empty with `promoted_ul_clusters == 0`;
- live rollout backend generation is not invoked in residual-set offline mode.

Post-smoke artifact check:

```bash
conda run -n ms python - <<'PY'
import json
from pathlib import Path

roots = sorted(Path("output").glob("**/compact_full_residual_set_ckpt3664_hf_1step*/**/logging.jsonl"))
if not roots:
    raise SystemExit("could not find smoke logging.jsonl")
log = roots[-1]
rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
metric_rows = [row for row in rows if isinstance(row, dict) and any(str(k).startswith("stage2_ab/channel_b/residual_set/") for k in row)]
if not metric_rows:
    raise SystemExit(f"missing residual-set metrics in {log}")
print(f"checked_residual_metrics={log}")
PY
```

- [ ] **Step 8.5: Update docs and OpenSpec tasks**

After tests and smoke pass:

- update stable docs only for behavior that is now supported;
- mark implementation tasks complete in `openspec/changes/add-stage2-residual-set-ul-correction/tasks.md`;
- record smoke scope and artifact root in `progress/`, not as an OpenSpec success gate.

## Suggested Subagent Ownership

Use subagent-driven development after user approval. Keep write sets disjoint:

- Agent A, Config: `src/config/schema.py`, `src/trainers/teacher_forcing/module_registry.py`, `tests/test_stage2_ab_config_contract.py`, `tests/test_teacher_forcing_loss_catalog.py`, one smoke YAML.
- Agent B, Template/Span Adapter: `src/training/span_adapters/residual_boundary.py`, `tests/test_stage2_residual_boundary_adapter.py`, adapter call-site migrations.
- Agent C, Residual State: `src/trainers/stage2_two_channel/residual_set.py`, `tests/test_stage2_residual_set_correction.py`.
- Agent D, UL Consensus: `src/trainers/stage2_two_channel/ul_consensus.py`, `tests/test_stage2_residual_ul_consensus.py`.
- Agent E, Prepared Runtime/IR/Loss: `rollout_views.py`, `stage2_two_channel.py`, `teacher_forcing_adapter.py`, `residual_set_correction.py`, objective pipeline tests.
- Main agent, Integration: resolve conflicts, run combined tests, update docs/progress, launch smoke.

## Self-Review

- Spec coverage: Every OpenSpec task maps to at least one plan task.
- Config explosion check: The existing residual smoke YAML is migrated instead of adding many new leaves.
- File creation check: Only one reusable source abstraction and one focused adapter test file are created.
- Deleted behavior check: `bbox_tail_from_anchor`, coordinate repair, `ul_geometry`, `loss_duplicate_burst_unlikelihood`, and duplicate unlikelihood are explicitly rejected.
- Causal alignment check: `logit_position + 1 == target_position` is tested in state, adapter, and integration layers.
- Implementation gate: This plan is for review only; code execution starts only after user approval.
