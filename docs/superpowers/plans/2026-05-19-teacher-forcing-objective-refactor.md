# Teacher-Forcing Objective Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the unified `objective.id: teacher_forcing` framework from `openspec/changes/add-teacher-forcing-objective`, replacing active recursive/ET-RMP and decoded-box auxiliary training surfaces with a shared target IR, typed valid-set objective modules, marker-delimited `compact_full`, and explicit diagnostics.

**Architecture:** Build bottom-up from a reusable `TeacherForcingTargetIR`, then extend the existing `src/training/objectives` semantic runner instead of creating a parallel objective stack. Preserve hard SFT first, reject unsafe packing/cache/logits paths early, and delete old recursive/auxiliary APIs only after replacement tests pass.

**Tech Stack:** Python dataclasses, PyTorch logits/probability math, Hugging Face/Qwen3-VL forward outputs, ms-swift trainer integration, CoordExp YAML config schemas, OpenSpec, pytest under `conda run -n ms`.

---

Date: 2026-05-19

Status: proposal for user review. Do not implement until the user explicitly approves this plan.

Primary OpenSpec change:
`openspec/changes/add-teacher-forcing-objective/`

Decision note:
`progress/explorations/2026-05-19_unified_teacher_forcing_objective_architecture_decisions.md`

## Execution Policy

This is a one-shot end-to-end refactor, but not one giant edit. Execute in task order and keep commits small enough to review. Each task must leave the tree in a testable state.

Use the existing worktree:

```text
/data/CoordExp/.worktrees/unified-training-infra-refactor
```

Binding constraints:

- No production implementation starts until this plan is approved.
- Do not edit upstream Qwen/HF model files.
- Do not silently preserve old training aliases.
- Do not forward `teacher_forcing_target_ir` to model forward.
- Do not use Stage-2 packing for the new objective in v1 unless exact atom-position remapping is implemented and tested. The recommended v1 path is fail-fast rejection.
- Do not enable encoded-sample cache for epoch-varying training roll-in.
- Do not globally mutate historical `compact_full`; make rendering/parsing policy-aware.
- Do not erase current recursive-detection comparator handles before the new teacher-forcing path has smoke evidence and docs route those handles into an explicit legacy/comparator namespace.
- Treat `target_position` as the canonical atom position. `logit_position` is a redundant validation field; objective execution still resolves rows through the existing `LabelLogitRowMap`.

## Planned File Map

Create:

- `src/training/teacher_forcing/__init__.py`
- `src/training/teacher_forcing/constants.py`
- `src/training/teacher_forcing/roles.py`
- `src/training/teacher_forcing/vocab.py`
- `src/training/teacher_forcing/ir.py`
- `src/training/teacher_forcing/validation.py`
- `src/training/teacher_forcing/probabilities.py`
- `src/training/teacher_forcing/metrics.py`
- `src/training/objectives/teacher_forcing.py`
- `src/training_runtime/preflight.py`
- `src/detection/teacher_forcing/__init__.py`
- `src/detection/teacher_forcing/description_tokens.py`
- `src/detection/teacher_forcing/compact_full_policy.py`
- `src/detection/teacher_forcing/rollin.py`
- `src/detection/teacher_forcing/trie.py`
- `src/detection/teacher_forcing/target_builder.py`
- `src/analysis/teacher_forcing_objective_report.py`
- `src/analysis/teacher_forcing_atom_probe.py`
- `src/analysis/compact_full_parse_report.py`
- `tests/test_teacher_forcing_ir_contract.py`
- `tests/test_teacher_forcing_objective_runner.py`
- `tests/test_teacher_forcing_target_builder.py`
- `tests/test_teacher_forcing_sidecar_bridge.py`
- `tests/test_teacher_forcing_config_contract.py`
- `tests/test_teacher_forcing_encoded_cache_contract.py`
- `tests/test_compact_full_marker_policy.py`
- `tests/test_infer_compact_full_policy_contract.py`
- `tests/test_stage2_teacher_forcing_adapter_contract.py`
- `tests/test_teacher_forcing_metric_contract.py`
- `tests/test_teacher_forcing_report_contract.py`
- `tests/test_stage2_launcher_preflight_contract.py`

Modify:

- `src/training/objectives/runner.py`
- `src/training/objectives/types.py`
- `src/training/supervision/distributions.py`
- `src/training/sidecars.py`
- `src/trainers/batch_extras.py`
- `src/data_collators/batch_extras_collator.py`
- `src/training/encoding/model_inputs.py`
- `src/training/bridge/loss_bridge.py`
- `src/detection/dataset.py`
- `src/detection/template.py`
- `src/detection/evaluation.py`
- `src/detection/__init__.py`
- `src/detection/runtime.py`
- `src/detection/objective.py`
- `src/detection/loss.py`
- `src/detection/packing.py`
- `src/common/detection_sequence.py`
- `src/bootstrap/trainer_setup.py`
- `src/trainers/stage2_two_channel.py`
- `src/trainers/teacher_forcing/module_registry.py`
- `src/trainers/teacher_forcing/objective_pipeline.py`
- `src/training_runtime/plan.py`
- `src/sft.py`
- `docs/AGENT_INDEX.md`
- `docs/training/README.md`
- `docs/IMPLEMENTATION_MAP.md`
- `docs/catalog.yaml`
- `tests/test_batch_extras_contract.py`
- `tests/test_model_input_bundle_contract.py`
- `tests/test_trainer_loss_bridge_qwen3vl_contract.py`
- `tests/test_latest_training_config_contract.py`
- `tests/test_stage2_ab_config_contract.py`
- `tests/test_detection_compact_full_template.py`
- `tests/test_detection_template_parsing_eval.py`
- `tests/test_coord_utils.py`
- `tests/test_recursive_detection_ce_loss_adapter.py`
- `tests/test_recursive_detection_ce_sft_wiring.py`
- `tests/test_recursive_detection_ce_target_builder.py`
- `tests/test_detection_training_dataset.py`
- Stage-1 latest compact configs under `configs/stage1/`
- Stage-2 smoke/prod configs under `configs/stage2_two_channel/`

Delete or move to legacy-only namespace after replacement tests pass:

- active use of `recursive_detection_targets`
- active use of `compute_recursive_detection_ce_batch_loss`
- active `recursive_detection_ce` trainer mixin/wiring
- new active/default Stage-1 routes under `configs/stage1/recursive_detection_ce_latest/**`; freeze current comparator configs under an explicit legacy/comparator namespace before removing their active routing docs
- active Stage-2 objective modules `bbox_geo`, `bbox_size_aux`, `coord_reg`, old `token_ce` pipeline semantics, `coord_gate`, or gate weights
- active old aliases or config leaves containing `ET_RMP_CE`, `et_rmp_like`, `typed_trie`, `random_permutation_et_rmp_ce`, `support_balance`, `trie_support`, or `trie_balance`

Historical inference/eval configs for already-trained checkpoints may remain only in an explicitly legacy namespace.

## Task 0: Freeze The Contract And Baseline Current State

**Files:**

- Read: `openspec/changes/add-teacher-forcing-objective/**`
- Read: `docs/superpowers/plans/2026-05-19-teacher-forcing-objective-refactor.md`
- Modify only if validation reveals drift: OpenSpec files above

- [ ] **Step 0.1: Validate OpenSpec before code**

Run:

```bash
openspec validate add-teacher-forcing-objective --strict
openspec status --change add-teacher-forcing-objective --json
```

Expected:

```text
Change 'add-teacher-forcing-objective' is valid
```

- [ ] **Step 0.2: Capture baseline failure/compatibility surface**

Run:

```bash
conda run -n ms python -m pytest \
  tests/test_batch_extras_contract.py \
  tests/test_trainer_loss_bridge_qwen3vl_contract.py \
  tests/test_detection_compact_full_template.py \
  tests/test_latest_training_config_contract.py \
  tests/test_stage2_ab_config_contract.py \
  -q
```

Expected: Record current pass/fail state in the task notes. If failures already exist, do not fix unrelated failures in this task.

- [ ] **Step 0.3: Commit only the planning/spec artifacts if requested**

Run only if the user asks to commit planning artifacts:

```bash
git add openspec/changes/add-teacher-forcing-objective \
  progress/explorations/2026-05-19_unified_teacher_forcing_objective_architecture_decisions.md \
  docs/superpowers/plans/2026-05-19-teacher-forcing-objective-refactor.md
git commit -m "docs: plan teacher-forcing objective refactor"
```

## Task 1: Add Shared Teacher-Forcing IR And Vocabulary Contracts

**Files:**

- Create: `src/training/teacher_forcing/constants.py`
- Create: `src/training/teacher_forcing/roles.py`
- Create: `src/training/teacher_forcing/vocab.py`
- Create: `src/training/teacher_forcing/ir.py`
- Create: `src/training/teacher_forcing/validation.py`
- Create: `tests/test_teacher_forcing_ir_contract.py`

- [ ] **Step 1.1: Write IR contract tests first**

Create tests that assert:

```python
from src.training.teacher_forcing.constants import TEACHER_FORCING_TARGET_IR_KEY
from src.training.teacher_forcing.ir import SupervisionAtom, TeacherForcingTargetIR
from src.training.teacher_forcing.roles import TokenRole
from src.training.teacher_forcing.validation import validate_target_ir


def test_target_ir_key_is_canonical() -> None:
    assert TEACHER_FORCING_TARGET_IR_KEY == "teacher_forcing_target_ir"


def test_atom_requires_causal_next_token_positions() -> None:
    atom = SupervisionAtom(
        batch_index=0,
        logit_position=3,
        target_position=5,
        allowed_token_roles=frozenset({TokenRole.TEXT}),
        selected_token_role=TokenRole.TEXT,
        valid_token_ids=frozenset({101}),
        selected_token_id=101,
        latent_valid_token_ids=frozenset({101}),
        coverage_target_weights=None,
        loss_tags=frozenset({"singleton"}),
        loss_weight=1.0,
        coord_role=None,
        provenance={},
    )
    ir = TeacherForcingTargetIR(schema_version=1, atoms=(atom,), metadata={})

    with pytest.raises(ValueError, match="target_position = logit_position \\+ 1"):
        validate_target_ir(ir, input_ids=torch.tensor([[9, 9, 9, 9, 9, 101]]))


def test_valid_tokens_must_belong_to_allowed_role_vocab() -> None:
    vocab = make_test_role_vocab(
        text_ids={101},
        schema_ids={201},
        coord_ids={301},
        stop_id=401,
    )
    atom = make_atom(
        allowed_token_roles=frozenset({TokenRole.TEXT}),
        selected_token_role=TokenRole.TEXT,
        valid_token_ids=frozenset({101, 301}),
        selected_token_id=101,
    )

    with pytest.raises(ValueError, match="valid_token_ids must be inside allowed role vocab"):
        validate_target_ir(
            TeacherForcingTargetIR(schema_version=1, atoms=(atom,), metadata={}),
            input_ids=torch.tensor([[9, 101]]),
            role_vocab=vocab,
        )
```

- [ ] **Step 1.2: Run failing IR tests**

Run:

```bash
conda run -n ms python -m pytest tests/test_teacher_forcing_ir_contract.py -q
```

Expected: fails because the package and dataclasses do not exist.

- [ ] **Step 1.3: Implement minimal IR package**

Implement:

```python
TEACHER_FORCING_TARGET_IR_KEY = "teacher_forcing_target_ir"
MARGINAL_SCOPE_SAMPLED_PATH_NEXT_TOKEN = "sampled_path_next_token"
```

Implement `TokenRole` with exactly:

```python
class TokenRole(str, Enum):
    SCHEMA = "SCHEMA"
    TEXT = "TEXT"
    COORD = "COORD"
    STOP = "STOP"
```

Implement frozen dataclasses:

```python
@dataclass(frozen=True)
class SupervisionAtom:
    batch_index: int
    logit_position: int
    target_position: int
    allowed_token_roles: frozenset[TokenRole]
    selected_token_role: TokenRole
    valid_token_ids: frozenset[int]
    selected_token_id: int
    latent_valid_token_ids: frozenset[int]
    coverage_target_weights: Mapping[int, float] | None
    loss_tags: frozenset[str]
    loss_weight: float
    coord_role: str | None
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class TeacherForcingTargetIR:
    schema_version: int
    atoms: tuple[SupervisionAtom, ...]
    metadata: Mapping[str, Any]
```

Implement validation:

- `target_position == logit_position + 1`
- nonempty `allowed_token_roles`
- `selected_token_role in allowed_token_roles`
- nonempty `valid_token_ids` for trainable atoms
- `selected_token_id == input_ids[batch_index, target_position]`
- `valid_token_ids` is a subset of `role_vocab(allowed_token_roles)`
- `selected_token_id` is a member of `valid_token_ids`
- unsupported mixed role sets rejected except `{TEXT, SCHEMA}`
- `STOP` token ownership is represented separately from `SCHEMA`
- `STOP` atoms can target only the configured `<|im_end|>` token id
- `target_position` is the canonical token position; `logit_position` is retained as a redundant validation field and must match the causal row later resolved by `LabelLogitRowMap`

- [ ] **Step 1.4: Verify IR tests pass**

Run:

```bash
conda run -n ms python -m pytest tests/test_teacher_forcing_ir_contract.py -q
```

Expected: pass.

- [ ] **Step 1.5: Commit**

```bash
git add src/training/teacher_forcing tests/test_teacher_forcing_ir_contract.py
git commit -m "feat: add teacher-forcing target IR"
```

## Task 2: Extend The Existing Objective Runner With Teacher-Forcing Modules

**Files:**

- Create: `src/training/teacher_forcing/probabilities.py`
- Create: `src/training/teacher_forcing/metrics.py`
- Create: `src/training/objectives/teacher_forcing.py`
- Modify: `src/training/objectives/runner.py`
- Modify: `src/training/objectives/types.py`
- Modify: `src/training/supervision/distributions.py`
- Create: `tests/test_teacher_forcing_objective_runner.py`

- [ ] **Step 2.1: Write runner validation tests**

Tests must cover:

- rank-2 logits rejected even when `batch=1`
- sliced logits rejected when `logits.shape[:2] != input_ids.shape[:2]`
- selected-token mismatch rejected
- valid-token role-vocabulary mismatch rejected before probability math
- masked logit/target position rejected
- mismatched redundant `logit_position` is rejected against `LabelLogitRowMap`
- ambiguous atom with coverage disabled uses valid-set marginal loss only
- singleton atom equals hard CE inside allowed role union
- `coverage_strength=0` disables coverage
- `coverage_strength=1` computes valid-set marginal plus within-valid CE
- `src.training.objectives.runner.ObjectiveRunner` owns execution
- no new parallel runner package is created under `src/objectives`

- [ ] **Step 2.2: Run failing runner tests**

Run:

```bash
conda run -n ms python -m pytest tests/test_teacher_forcing_objective_runner.py -q
```

Expected: fails because runner modules do not exist.

- [ ] **Step 2.3: Implement existing-runner validation and registry extension**

Extend the current semantic runner at `src/training/objectives/runner.py`; do not create a second objective framework. Register the new objective id through the existing `src/training/supervision/distributions.py` and `src/training/objectives/types.py` contracts so the bridge and row-map stay the single owners of label-position to logit-row mapping.

Position authority:

- `target_position` is the canonical label/token position.
- `LabelLogitRowMap` derives the row used for logits.
- `logit_position` in `SupervisionAtom` is checked against the derived row and then treated as validation evidence, not as an alternate indexing authority.
- Teacher-forcing bridge paths reject rank-2 logits before runner execution, even if current non-teacher-forcing paths still accept `[seq, vocab]` for `batch=1`.

Teacher-forcing preconditions:

```python
if logits.ndim != 3:
    raise ValueError("teacher_forcing requires rank-3 logits [batch, seq, vocab]")
if logits.shape[:2] != input_ids.shape[:2]:
    raise ValueError("teacher_forcing requires logits.shape[:2] == input_ids.shape[:2]")
```

Reject `logits_to_keep` in the trainer bridge before forward where possible; reject sliced logits in the runner as the last line of defense.

- [ ] **Step 2.4: Implement probability decomposition**

Implement full-vocab softmax once, then:

```text
P_allowed = sum p(v) over role_vocab(allowed_token_roles)
p_bar(v) = p(v) / P_allowed
L_type = -log(P_allowed)
L_valid = -log(sum p_bar(valid_token_ids))
p_valid(v) = p_bar(v) / sum p_bar(valid_token_ids)
L_coverage = CE(coverage_target_weights_normalized, p_valid)
```

Do not add selected-child hard CE for ambiguous atoms.

- [ ] **Step 2.5: Verify runner tests pass**

Run:

```bash
conda run -n ms python -m pytest tests/test_teacher_forcing_objective_runner.py -q
```

Expected: pass.

- [ ] **Step 2.6: Commit**

```bash
git add src/training/teacher_forcing src/training/objectives src/training/supervision \
  tests/test_teacher_forcing_objective_runner.py
git commit -m "feat: add teacher-forcing objective runner"
```

## Task 3: Make `compact_full` Policy-Aware And Marker-Delimited For New Training

**Files:**

- Create: `src/detection/teacher_forcing/compact_full_policy.py`
- Modify: `src/detection/template.py`
- Modify: `src/detection/evaluation.py`
- Modify: `src/common/detection_sequence.py`
- Modify: `tests/test_detection_compact_full_template.py`
- Modify: `tests/test_detection_template_parsing_eval.py`
- Modify: `tests/test_coord_utils.py`
- Create: `tests/test_compact_full_marker_policy.py`

- [ ] **Step 3.1: Write policy tests**

Required tests:

```python
def test_marker_delimited_render_has_no_newline_between_objects() -> None:
    rendered = render_compact_full(sample_with_two_objects, serialization_policy="marker_delimited")
    assert "\n" not in rendered
    assert rendered.count("<|object_ref_start|>") == 2


def test_strict_marker_parser_rejects_legacy_newline() -> None:
    result = parse_compact_full(text_with_newline_rows, mode="marker_delimited_strict")
    assert result.error_code == "legacy_separator_in_new_format"


def test_legacy_compatible_parser_accepts_newline_for_historical_outputs() -> None:
    result = parse_compact_full(text_with_newline_rows, mode="legacy_compatible")
    assert result.objects
    assert result.mode == "legacy_compatible"


def test_marker_parser_allows_next_object_or_im_end_after_four_coords() -> None:
    result = parse_compact_full(two_marker_objects_then_im_end, mode="marker_delimited_strict")
    assert result.objects[0].description == "person"
    assert result.objects[1].description == "car"
    assert result.terminal_token == "<|im_end|>"
```

- [ ] **Step 3.2: Run failing policy tests**

Run:

```bash
conda run -n ms python -m pytest tests/test_compact_full_marker_policy.py -q
```

Expected: fails because policy-aware rendering/parsing is absent.

- [ ] **Step 3.3: Implement policy-aware rendering/parsing**

Implementation requirements:

- Keep template id `compact_full`.
- New training policy is `serialization_policy: marker_delimited`.
- Historical policy may be `legacy_newline_delimited` or parser mode `legacy_compatible`.
- Update metric-bearing eval parser owners, not only template rendering. At minimum, route `marker_delimited_strict` and `legacy_compatible` through `src/detection/evaluation.py` and stop `src/common/detection_sequence.py` from assuming newline is the only compact object separator.
- In marker-delimited strict mode, after `<|coord_y2|>` the parser accepts only `<|object_ref_start|>` for another object or `<|im_end|>` for STOP. A newline is not a legal object-boundary marker in new outputs.
- Strict parse mode names:
  - `marker_delimited_strict`
  - `legacy_compatible`
- Strict error codes:
  - `empty_output`
  - `legacy_separator_in_new_format`
  - `missing_object_ref_start`
  - `missing_box_start`
  - `empty_description`
  - `forbidden_description_token`
  - `wrong_coord_arity`
  - `invalid_coord_token`
  - `trailing_garbage`
  - `invalid_geometry`

- [ ] **Step 3.4: Verify parser/template tests**

Run:

```bash
conda run -n ms python -m pytest \
  tests/test_compact_full_marker_policy.py \
  tests/test_detection_compact_full_template.py \
  tests/test_detection_template_parsing_eval.py \
  tests/test_coord_utils.py \
  -q
```

Expected: pass, with historical parser tests explicitly selecting legacy mode.

- [ ] **Step 3.5: Commit**

```bash
git add src/detection/template.py src/detection/evaluation.py src/common/detection_sequence.py \
  src/detection/teacher_forcing/compact_full_policy.py \
  tests/test_compact_full_marker_policy.py tests/test_detection_compact_full_template.py \
  tests/test_detection_template_parsing_eval.py tests/test_coord_utils.py
git commit -m "feat: add marker-delimited compact_full policy"
```

## Task 3A: Wire Inference Parser Defaults And Parse Artifacts

**Files:**

- Modify: inference config schema owners discovered by `tests/test_infer_compact_full_policy_contract.py`
- Modify: `src/detection/evaluation.py`
- Modify: inference artifact writers discovered by `tests/test_detection_template_parsing_eval.py`
- Create: `tests/test_infer_compact_full_policy_contract.py`
- Modify: `tests/test_detection_template_parsing_eval.py`

- [ ] **Step 3A.1: Write inference parser contract tests**

Required tests:

```python
def test_new_teacher_forcing_infer_defaults_to_marker_strict() -> None:
    config = load_infer_config("teacher_forcing_marker_smoke")
    assert config.infer.parsing.compact_full.mode == "marker_delimited_strict"
    assert config.infer.generation.compact_grammar.enabled is False


def test_legacy_parser_mode_requires_explicit_legacy_namespace() -> None:
    config = load_infer_config("legacy_recursive_detection_checkpoint")
    assert config.infer.parsing.compact_full.mode == "legacy_compatible"
    assert config.metadata.compatibility_namespace == "legacy"


def test_parse_artifact_records_policy_and_separator() -> None:
    artifact = parse_marker_output_to_artifact(marker_output_with_two_objects)
    assert artifact["parse_mode"] == "marker_delimited_strict"
    assert artifact["serialization_policy"] == "marker_delimited"
    assert artifact["object_separator"] == "<|object_ref_start|>"
```

- [ ] **Step 3A.2: Run failing inference parser tests**

Run:

```bash
conda run -n ms python -m pytest \
  tests/test_infer_compact_full_policy_contract.py \
  tests/test_detection_template_parsing_eval.py \
  -q
```

Expected: fails until inference defaults and artifact fields are wired.

- [ ] **Step 3A.3: Implement inference defaults and artifact fields**

Implementation requirements:

- New teacher-forcing inference configs default to `infer.parsing.compact_full.mode: marker_delimited_strict`.
- `legacy_compatible` is accepted only through explicit legacy configs for already-trained checkpoints.
- Compact grammar remains disabled by default at `infer.generation.compact_grammar.enabled: false` for new teacher-forcing inference; rely on trained grammar unless the user explicitly enables a future constrained-decoding experiment.
- Parse artifacts record at least `parse_mode`, `serialization_policy`, `object_separator`, `terminal_token`, and the concrete parse error code for failures.
- Historical newline outputs remain parseable only when the config says `legacy_compatible`.

- [ ] **Step 3A.4: Verify inference parser contract**

Run the same command as Step 3A.2.

Expected: pass.

- [ ] **Step 3A.5: Commit**

```bash
git add src/detection/evaluation.py tests/test_infer_compact_full_policy_contract.py \
  tests/test_detection_template_parsing_eval.py configs
git commit -m "feat: enforce teacher-forcing inference parser contract"
```

## Task 4: Build Stage-1 Target IR Construction

**Files:**

- Create: `src/detection/teacher_forcing/description_tokens.py`
- Create: `src/detection/teacher_forcing/rollin.py`
- Create: `src/detection/teacher_forcing/trie.py`
- Create: `src/detection/teacher_forcing/target_builder.py`
- Create: `tests/test_teacher_forcing_target_builder.py`

- [ ] **Step 4.1: Write target-builder tests**

Required tests:

- hard SFT profile emits singleton `valid_token_ids`
- pure valid-set profile emits all legal next tokens at ambiguous prefixes
- `{TEXT, SCHEMA}` mixed-role atom for `car` / `carrot`
- same-description repeated objects stay ambiguous until `x1`
- selecting `x1` filters candidate objects before `y1/x2/y2`
- selected token id matches rendered `input_ids[target_position]`
- missing detection list drops sample
- explicit empty COCO object list drops sample
- overlength sample drops before partial target construction
- roll-in seed derives from base seed `17`, epoch, stable sample id, policy name/version

- [ ] **Step 4.2: Run failing target-builder tests**

Run:

```bash
conda run -n ms python -m pytest tests/test_teacher_forcing_target_builder.py -q
```

Expected: fails because builder modules do not exist.

- [ ] **Step 4.3: Implement context-aware description tokenization**

Tokenize:

```text
<|object_ref_start|>{desc}<|box_start|>
```

Reject:

- empty description after strip
- newline/tab/control characters
- reserved schema/control/image/stop marker text
- coordinate-token text
- non-unique marker boundary mapping

- [ ] **Step 4.4: Implement sampled roll-in and trie atoms**

Builder output:

```python
TeacherForcingTargetIR(
    schema_version=1,
    atoms=(
        SupervisionAtom(
            batch_index=0,
            logit_position=12,
            target_position=13,
            allowed_token_roles=frozenset({TokenRole.TEXT}),
            selected_token_role=TokenRole.TEXT,
            valid_token_ids=frozenset({6918}),
            selected_token_id=6918,
            latent_valid_token_ids=frozenset({6918}),
            coverage_target_weights=None,
            loss_tags=frozenset({"description", "singleton"}),
            loss_weight=1.0,
            coord_role=None,
            provenance={"object_id": "obj0"},
        ),
    ),
    metadata={
        "marginal_scope": "sampled_path_next_token",
        "rollin_policy": "random_permutation",
        "rollin_seed": seed,
        "serialization_policy": "marker_delimited",
    },
)
```

Candidate filtering rules:

- selected description token keeps candidates with that token continuation
- selected `<|box_start|>` keeps candidates whose description path is complete
- selected coordinate token keeps compatible object branches
- object is removed from remaining set only after full bbox completion

- [ ] **Step 4.5: Verify target-builder tests**

Run:

```bash
conda run -n ms python -m pytest tests/test_teacher_forcing_target_builder.py -q
```

Expected: pass.

- [ ] **Step 4.6: Commit**

```bash
git add src/detection/teacher_forcing tests/test_teacher_forcing_target_builder.py
git commit -m "feat: build teacher-forcing target IR"
```

## Task 5: Wire Sidecars Through Collation And Strip Before Model Forward

**Files:**

- Modify: `src/trainers/batch_extras.py`
- Modify: `src/data_collators/batch_extras_collator.py`
- Modify: `src/training/sidecars.py`
- Modify: `src/training/encoding/model_inputs.py`
- Modify: `src/training/bridge/loss_bridge.py`
- Modify: `tests/test_batch_extras_contract.py`
- Modify: `tests/test_model_input_bundle_contract.py`
- Modify: `tests/test_trainer_loss_bridge_qwen3vl_contract.py`
- Create: `tests/test_teacher_forcing_sidecar_bridge.py`

- [ ] **Step 5.1: Write sidecar bridge tests**

Tests must prove:

- `teacher_forcing_target_ir` survives dataset item -> collator -> batch extras
- `teacher_forcing_target_ir` is represented in `TrainingSidecars.supervision`
- `teacher_forcing_target_ir` is absent from model-forward kwargs
- objective runner can retrieve the stashed target IR
- flash-attention kwargs remain forwardable:
  - `cu_seq_lens_q`
  - `cu_seq_lens_k`
  - `max_length_q`
  - `max_length_k`
- `logits_to_keep` is rejected or bypassed for teacher-forcing runs

- [ ] **Step 5.2: Run failing bridge tests**

Run:

```bash
conda run -n ms python -m pytest \
  tests/test_teacher_forcing_sidecar_bridge.py \
  tests/test_batch_extras_contract.py \
  tests/test_model_input_bundle_contract.py \
  tests/test_trainer_loss_bridge_qwen3vl_contract.py \
  -q
```

Expected: fails until the new sidecar key is wired.

- [ ] **Step 5.3: Implement sidecar plumbing**

Add `TEACHER_FORCING_TARGET_IR_KEY` to current sidecar/batch-extra owners. The canonical semantic home is `src/training/sidecars.py`; add a typed `teacher_forcing_target_ir` field to `SupervisionSidecars` or an equally explicit payload convention there, then register the raw batch key in `SIDECAR_ONLY_KEYS` and `RUNNER_OWNED_LOSS_STRIPPED_KEYS` as needed. Do not delete `recursive_detection_targets` in this task; keep it until the replacement path passes integration tests.

- [ ] **Step 5.4: Verify bridge tests**

Run the same command as Step 5.2.

Expected: pass.

- [ ] **Step 5.5: Commit**

```bash
git add src/trainers/batch_extras.py src/data_collators/batch_extras_collator.py \
  src/training/sidecars.py src/training/encoding/model_inputs.py \
  src/training/bridge/loss_bridge.py \
  tests/test_teacher_forcing_sidecar_bridge.py tests/test_batch_extras_contract.py \
  tests/test_model_input_bundle_contract.py tests/test_trainer_loss_bridge_qwen3vl_contract.py
git commit -m "feat: bridge teacher-forcing target IR sidecar"
```

## Task 6: Add Config Schema And Migration Errors

**Files:**

- Modify: Stage-1 latest training config schema owners discovered by `tests/test_latest_training_config_contract.py`
- Modify: Stage-2 config schema owners discovered by `tests/test_stage2_ab_config_contract.py`
- Modify: `src/sft.py`
- Modify: `src/bootstrap/trainer_setup.py`
- Create or modify: `tests/test_teacher_forcing_config_contract.py`
- Modify: `tests/test_latest_training_config_contract.py`
- Modify: `tests/test_stage2_ab_config_contract.py`

- [ ] **Step 6.1: Write config tests**

Tests must assert:

- `objective.id: teacher_forcing` accepts profiles:
  - `hard_sft`
  - `pure_valid_set_marginal`
  - `coverage_regularized_valid_set_marginal`
- coverage profile requires explicit positive `objective.modules.within_valid_coverage.coverage_strength`
- pure profile rejects positive coverage
- old ids fail:
  - `recursive_detection_ce`
  - `random_permutation_et_rmp_ce`
  - `prefix_rollin_et_rmp_ce`
  - `ET_RMP_CE`
  - `et_rmp_like`
  - `typed_trie_alpha0_random_rollin`
  - `typed_trie_alpha0p1_random_rollin`
  - `support_balance`
- old Stage-2 modules fail under active teacher-forcing:
  - `bbox_geo`
  - `bbox_size_aux`
  - `coord_reg`
  - `token_ce` when authored as `stage2_ab.pipeline.objective[name=token_ce]`
  - `coord_gate`
  - `text_gate`
- old Stage-2 gate/config keys fail under active teacher-forcing:
  - `coord_gate_weight`
  - `text_gate_weight`
  - `soft_ce_weight`
  - `w1_weight`
- Stage-2 `training.packing=true` fails for `objective.id: teacher_forcing` unless exact mapping is explicitly enabled

- [ ] **Step 6.2: Run failing config tests**

Run:

```bash
conda run -n ms python -m pytest \
  tests/test_teacher_forcing_config_contract.py \
  tests/test_latest_training_config_contract.py \
  tests/test_stage2_ab_config_contract.py \
  -q
```

Expected: fails until config schema is migrated.

- [ ] **Step 6.3: Implement config migration**

Recommended schema shape:

```yaml
objective:
  id: teacher_forcing
  profile: pure_valid_set_marginal
  target_ir:
    rollin_policy:
      name: random_permutation
      base_seed: 17
  modules:
    token_type_mass:
      enabled: true
    conditional_valid_set_likelihood:
      enabled: true
    within_valid_coverage:
      enabled: false
      coverage_strength: 0.0
    continuation_margin:
      enabled: false
```

- [ ] **Step 6.4: Verify config tests**

Run the same command as Step 6.2.

Expected: pass.

- [ ] **Step 6.5: Commit**

```bash
git add <exact config schema files changed in src/> src/sft.py \
  tests/test_teacher_forcing_config_contract.py \
  tests/test_latest_training_config_contract.py tests/test_stage2_ab_config_contract.py
git commit -m "feat: add teacher-forcing objective config"
```

Do not run `git add src` in a dirty tree. Stage exact changed files discovered by the config tests.

## Task 6A: Enforce Teacher-Forcing Encoded Cache Contract

**Files:**

- Modify: `src/training/encoding/model_inputs.py`
- Create: `src/training_runtime/preflight.py`
- Modify: cache key/payload owners discovered by `tests/test_teacher_forcing_encoded_cache_contract.py`
- Create: `tests/test_teacher_forcing_encoded_cache_contract.py`
- Modify: `tests/test_training_runtime_sft_integration.py`

- [ ] **Step 6A.1: Write cache contract tests**

Required tests:

```python
def test_epoch_varying_training_rollin_rejects_encoded_cache() -> None:
    config = make_training_config(objective_id="teacher_forcing", rollin_policy="random_permutation")
    config.training.encoded_cache.enabled = True
    with pytest.raises(ValueError, match="teacher_forcing encoded training cache"):
        validate_training_runtime_preflight(config, runtime_plan=resolve_training_runtime_plan(config.custom.trainer_variant))


def test_encoded_cache_bypass_reason_is_recorded() -> None:
    config = make_training_config(objective_id="teacher_forcing", rollin_policy="random_permutation")
    config.training.encoded_cache.enabled = True
    result = collect_training_runtime_preflight(config, runtime_plan=resolve_training_runtime_plan(config.custom.trainer_variant))
    assert result.encoded_cache.bypass_reason == "teacher_forcing_epoch_varying_rollin"


def test_fixed_eval_probe_cache_key_includes_teacher_forcing_contract() -> None:
    key = build_fixed_eval_probe_cache_key(
        tokenizer_fingerprint="tok",
        chat_template_fingerprint="tmpl",
        serialization_policy="marker_delimited",
        description_normalization_policy="coco80_exact",
        rollin_policy={"name": "canonical_eval", "base_seed": 17, "epoch": 0},
        target_ir_schema_version=1,
        max_length=256,
    )
    assert key["target_ir_schema_version"] == 1
    assert key["serialization_policy"] == "marker_delimited"


def test_cached_teacher_forcing_payload_contains_input_ids_and_ir() -> None:
    payload = build_fixed_eval_probe_payload(input_ids=[1, 2], teacher_forcing_target_ir=minimal_ir())
    assert "input_ids" in payload
    assert "teacher_forcing_target_ir" in payload
```

- [ ] **Step 6A.2: Run failing cache tests**

Run:

```bash
conda run -n ms python -m pytest \
  tests/test_teacher_forcing_encoded_cache_contract.py \
  tests/test_training_runtime_sft_integration.py \
  -q
```

Expected: fails until cache rejection and fixed-cache key validation are wired.

- [ ] **Step 6A.3: Implement cache behavior**

Implementation requirements:

- Epoch-varying training cache is disabled for `objective.id: teacher_forcing` in v1 because random roll-in changes atom positions by epoch.
- The config-aware preflight owner is `src/training_runtime/preflight.py`. Keep `src/training_runtime/plan.py::resolve_training_runtime_plan` variant-only unless a separate design explicitly changes that contract.
- Cache rejection/bypass records `encoded_cache.bypass_reason: teacher_forcing_epoch_varying_rollin` in the runtime plan, manifest, or equivalent preflight result.
- Fixed eval/probe caches may exist only if their key includes tokenizer fingerprint, chat-template fingerprint, serialization policy, description normalization policy, roll-in policy/version/seed/epoch, `TeacherForcingTargetIR.schema_version`, and max length.
- Fixed eval/probe payloads must contain both `input_ids` and `teacher_forcing_target_ir`.
- Cache load validates the IR schema and selected-token alignment before returning the payload.

- [ ] **Step 6A.4: Verify cache contract**

Run the same command as Step 6A.2.

Expected: pass.

- [ ] **Step 6A.5: Commit**

```bash
git add src/training/encoding src/training_runtime/preflight.py \
  tests/test_teacher_forcing_encoded_cache_contract.py tests/test_training_runtime_sft_integration.py
git commit -m "feat: enforce teacher-forcing cache contract"
```

## Task 7: Integrate Stage-1 Training With The New Objective

**Files:**

- Modify: `src/detection/dataset.py`
- Modify: `src/detection/__init__.py`
- Modify: `src/sft.py`
- Modify: Stage-1 config leaves under `configs/stage1/`
- Create/modify: `tests/test_training_runtime_sft_integration.py`
- Modify: `tests/test_latest_detection_view_metadata.py`

- [ ] **Step 7.1: Write Stage-1 integration tests**

Required tests:

- Stage-1 `hard_sft` produces `teacher_forcing_target_ir`
- Stage-1 `pure_valid_set_marginal` produces ambiguous atoms where expected
- rendered marker-delimited input and IR atom positions align
- trainer computes loss through `src/training/objectives/runner.py` and the registered teacher-forcing objective module
- static packing, eval packing, padding-free packing, and encoded training cache fail fast for training v1
- trainable token-row adaptation includes coordinate rows plus marker schema rows `<|object_ref_start|>` and `<|box_start|>` when adaptation is enabled
- STOP atoms target `<|im_end|>`; HF/Qwen config uses `eos_token_id=<|im_end|>` and `pad_token_id=<|endoftext|>`
- Stage-1 runtime keeps `do_resize=false`
- generation contract metadata records STOP token, pad token, serialization policy, parser mode, and compact grammar enabled/disabled state

- [ ] **Step 7.2: Run failing Stage-1 tests**

Run:

```bash
conda run -n ms python -m pytest \
  tests/test_training_runtime_sft_integration.py \
  tests/test_latest_detection_view_metadata.py \
  -q
```

Expected: fails until dataset/runtime integration is wired.

- [ ] **Step 7.3: Wire Stage-1 dataset and trainer**

Replace active recursive sidecar generation with `TeacherForcingTargetIR` for new configs. Keep old recursive paths only long enough for migration-error tests to prove they fail with guidance.

- [ ] **Step 7.4: Verify Stage-1 integration**

Run:

```bash
conda run -n ms python -m pytest \
  tests/test_teacher_forcing_ir_contract.py \
  tests/test_teacher_forcing_target_builder.py \
  tests/test_teacher_forcing_objective_runner.py \
  tests/test_teacher_forcing_sidecar_bridge.py \
  tests/test_teacher_forcing_encoded_cache_contract.py \
  tests/test_training_runtime_sft_integration.py \
  tests/test_latest_detection_view_metadata.py \
  tests/test_infer_compact_full_policy_contract.py \
  -q
```

Expected: pass.

- [ ] **Step 7.5: Commit**

```bash
git add src/detection src/sft.py configs/stage1 tests/test_training_runtime_sft_integration.py \
  tests/test_latest_detection_view_metadata.py
git commit -m "feat: integrate stage1 teacher-forcing objective"
```

## Task 8: Add Stage-2 Adapter And Fail-Fast Packing Guard

**Files:**

- Modify: `src/trainers/stage2_two_channel.py`
- Modify: `src/trainers/teacher_forcing/module_registry.py`
- Modify: `src/trainers/teacher_forcing/objective_pipeline.py`
- Modify: `src/bootstrap/stage2_policy_provenance.py`
- Modify: `src/training_runtime/plan.py`
- Create/modify: `src/training_runtime/preflight.py`
- Modify: `src/sft.py`
- Modify: Stage-2 config leaves under `configs/stage2_two_channel/`
- Create: `tests/test_stage2_teacher_forcing_adapter_contract.py`
- Create/modify: `tests/test_stage2_launcher_preflight_contract.py`
- Modify: `tests/test_stage2_ab_config_contract.py`
- Modify: `tests/test_stage2_two_channel_training.py`
- Modify: `tests/test_training_runtime_sft_integration.py`

- [ ] **Step 8.1: Write Stage-2 adapter tests**

Tests must assert:

- Stage-2 Channel-A can emit GT-context `TeacherForcingTargetIR`
- Stage-2 Channel-B can emit rollout/FN/recovered-FN provenance metadata
- duplicate-certified rollout objects emit zero positive atoms
- pseudo-positive or shielded rollout objects emit zero positive atoms unless explicitly promoted by the Stage-2 triage policy
- ordinary FN objects emit positive atoms with FN provenance
- recovered-FN objects emit positive atoms whose `loss_weight` and `loss_tags` are scoped only to recovered-FN atoms
- Channel-A atoms do not introduce `self_context`
- old `bbox_geo`, `bbox_size_aux`, `coord_reg`, `soft_ce`, `w1`, `token_ce`, `coord_gate`, `text_gate`, and gate-weight configs fail for active teacher-forcing
- `objective.id: teacher_forcing` plus `custom.trainer_variant: stage2_two_channel` plus `training.packing=true` fails in the launch/runtime preflight before trainer model forward
- current non-teacher-forcing Stage-2 packing remains trainer-owned and does not fail through the new teacher-forcing preflight
- the Stage-2 trainer still keeps a defense-in-depth packing guard
- flash-attention kwargs are preserved when sidecars are stripped
- there is one public teacher-forcing objective entrypoint; `src/trainers/teacher_forcing/*` is either a legacy Stage-2 adapter boundary or subordinate bridge, not a peer objective framework

- [ ] **Step 8.2: Run failing Stage-2 tests**

Run:

```bash
conda run -n ms python -m pytest \
  tests/test_stage2_teacher_forcing_adapter_contract.py \
  tests/test_stage2_launcher_preflight_contract.py \
  tests/test_stage2_ab_config_contract.py \
  tests/test_stage2_two_channel_training.py \
  tests/test_training_runtime_sft_integration.py \
  -q
```

Expected: fails until Stage-2 adapter and migration errors are wired.

- [ ] **Step 8.3: Implement Stage-2 adapter**

Recommended v1 behavior:

```text
if objective.id == "teacher_forcing"
and custom.trainer_variant == "stage2_two_channel"
and training.packing is true:
    fail during runtime preflight with guidance to disable packing
```

Do not implement packed atom remapping in this first pass unless the user reopens the decision. Keep the trainer-level guard as a second line of defense, but make the primary error happen before rollout setup or model forward.

Migration boundary:

- `src/training/objectives/runner.py` is the public objective entrypoint.
- `src/trainers/teacher_forcing/*` must be reduced to a Stage-2 adapter/legacy bridge or removed after the shared runner path passes.
- Do not leave `src/trainers/teacher_forcing/objective_pipeline.py` as a second peer teacher-forcing framework.

- [ ] **Step 8.4: Verify Stage-2 adapter**

Run the same command as Step 8.2.

Expected: pass.

- [ ] **Step 8.5: Commit**

```bash
git add src/trainers src/bootstrap src/training_runtime/plan.py src/training_runtime/preflight.py \
  src/sft.py configs/stage2_two_channel \
  tests/test_stage2_teacher_forcing_adapter_contract.py \
  tests/test_stage2_launcher_preflight_contract.py tests/test_stage2_ab_config_contract.py \
  tests/test_stage2_two_channel_training.py \
  tests/test_training_runtime_sft_integration.py
git commit -m "feat: adapt stage2 to teacher-forcing target IR"
```

## Task 9: Add Metrics And Analysis Reports

**Files:**

- Modify: `src/training/teacher_forcing/metrics.py`
- Create: `src/analysis/teacher_forcing_objective_report.py`
- Create: `src/analysis/teacher_forcing_atom_probe.py`
- Create: `src/analysis/compact_full_parse_report.py`
- Create: `tests/test_teacher_forcing_metric_contract.py`
- Create: `tests/test_teacher_forcing_report_contract.py`

- [ ] **Step 9.1: Write metric-contract tests**

Minimum keys:

```text
teacher_forcing/loss/total
teacher_forcing/loss/token_type_mass
teacher_forcing/loss/conditional_valid_set_likelihood
teacher_forcing/loss/within_valid_coverage
teacher_forcing/valid_set/mass
teacher_forcing/coverage/kl
teacher_forcing/continuation/eos_margin
teacher_forcing/ambiguity/coordinate_onset_count
teacher_forcing/ambiguity/mixed_role_count
teacher_forcing/builder/rejected_samples
teacher_forcing/builder/rejection_reason/<code>
teacher_forcing/branch_coherence/rate
teacher_forcing/residual_set/remaining_count
teacher_forcing/permutation_probe/nll_std
teacher_forcing/decode/object_coherence_rate
teacher_forcing/decode/duplicate_rate
teacher_forcing/decode/missed_object_rate
teacher_forcing/decode/malformed_sequence_rate
infer/parse/compact_full/error/<code>
```

- [ ] **Step 9.2: Run failing metric tests**

Run:

```bash
conda run -n ms python -m pytest \
  tests/test_teacher_forcing_metric_contract.py \
  tests/test_teacher_forcing_report_contract.py \
  -q
```

Expected: fails until metric event/report code exists.

- [ ] **Step 9.3: Implement metric emitters and report CLIs**

Analysis scripts should accept JSONL/artifact paths and produce compact JSON/Markdown summaries. They must not require production training artifacts to run in unit tests.

Report fixture requirements:

- coordinate-onset ambiguity is reported separately from text ambiguity;
- mixed-role `{TEXT, SCHEMA}` ambiguity is counted;
- builder rejection counters preserve reason codes such as `missing_detection_list`, `empty_detection_list`, `overlength`, and `description_tokenization_failed`;
- decode-time object coherence, duplicate, missed-object, and malformed-sequence rates are emitted only when generation artifacts are present, with absent-artifact status recorded rather than fabricated zero rates.

- [ ] **Step 9.4: Verify metric tests**

Run:

```bash
conda run -n ms python -m pytest \
  tests/test_teacher_forcing_metric_contract.py \
  tests/test_teacher_forcing_report_contract.py \
  -q
```

Expected: pass.

- [ ] **Step 9.5: Commit**

```bash
git add src/training/teacher_forcing/metrics.py src/analysis \
  tests/test_teacher_forcing_metric_contract.py tests/test_teacher_forcing_report_contract.py
git commit -m "feat: add teacher-forcing diagnostics"
```

## Task 9A: Map And Retire Stage-1 Recursive Owners Safely

**Files:**

- Modify: `src/detection/runtime.py`
- Modify: `src/detection/objective.py`
- Modify: `src/detection/loss.py`
- Modify: `src/detection/packing.py`
- Modify: `src/detection/dataset.py`
- Modify: `tests/test_recursive_detection_ce_loss_adapter.py`
- Modify: `tests/test_recursive_detection_ce_sft_wiring.py`
- Modify: `tests/test_recursive_detection_ce_target_builder.py`
- Modify: `tests/test_detection_training_dataset.py`
- Modify: `docs/AGENT_INDEX.md`

- [ ] **Step 9A.1: Write recursive-owner boundary tests**

Tests must assert:

- `RecursiveDetectionTargets` and `compute_recursive_detection_ce_batch_loss` are not used by new `objective.id: teacher_forcing` configs;
- current recursive-detection comparator configs are either explicitly legacy/comparator-routed or fail with migration guidance when used as new active training defaults;
- dataset-side encoded sidecar shifting does not run for `teacher_forcing_target_ir`;
- recursive-detection packing helpers are not silently reused for the new target IR;
- docs route `compact_full_support2.yaml` and `compact_full_prefix_rollin_balance2.yaml` as legacy/comparator handles until the new teacher-forcing smoke replaces them.

- [ ] **Step 9A.2: Run failing recursive-owner tests**

Run:

```bash
conda run -n ms python -m pytest \
  tests/test_recursive_detection_ce_loss_adapter.py \
  tests/test_recursive_detection_ce_sft_wiring.py \
  tests/test_recursive_detection_ce_target_builder.py \
  tests/test_detection_training_dataset.py \
  -q
```

Expected: fails until recursive owners are routed away from active teacher-forcing.

- [ ] **Step 9A.3: Implement recursive-owner migration boundary**

Implementation requirements:

- `src/detection/runtime.py`, `src/detection/objective.py`, `src/detection/loss.py`, and `src/detection/packing.py` are the old Stage-1 recursive owners and must be handled deliberately before deletion.
- Do not leave importable old training APIs as active defaults.
- Do not delete documented comparator config handles until docs and configs route them into an explicit legacy/comparator namespace.
- Do not use the old recursive packing/target-shift logic for `TeacherForcingTargetIR`.

- [ ] **Step 9A.4: Verify recursive-owner boundary**

Run the same command as Step 9A.2.

Expected: pass, with any remaining recursive detection references explicitly classified as legacy/comparator or migration tests.

- [ ] **Step 9A.5: Commit**

```bash
git add src/detection/runtime.py src/detection/objective.py src/detection/loss.py \
  src/detection/packing.py src/detection/dataset.py docs/AGENT_INDEX.md \
  tests/test_recursive_detection_ce_loss_adapter.py tests/test_recursive_detection_ce_sft_wiring.py \
  tests/test_recursive_detection_ce_target_builder.py tests/test_detection_training_dataset.py
git commit -m "refactor: route stage1 recursive owners to legacy boundary"
```

## Task 10: Delete Old Active Objective Surfaces

**Files:**

- Modify/delete active references found by:

```bash
rg -n "ET_RMP_CE|et_rmp_like|typed_trie|recursive_detection_targets|compute_recursive_detection_ce_batch_loss|recursive_detection_ce|prefix_rollin_et_rmp_ce|random_permutation_et_rmp_ce|support_balance|trie_support|trie_balance|et_rmp/|bbox_geo|bbox_size_aux|coord_reg|loss_duplicate_burst_unlikelihood|stage2_ab\\.pipeline\\.objective.*token_ce|coord_gate|coord_gate_weight" src tests configs docs
```

- [ ] **Step 10.1: Write migration/absence tests**

Tests must assert:

- old objective ids fail with migration guidance
- old Stage-2 modules fail for active teacher-forcing
- old Stage-2 `token_ce` pipeline modules and `coord_gate`/gate-weight keys fail for active teacher-forcing
- old metric aliases are absent from new runs
- old public APIs are either removed or raise explicit migration errors
- old terminology hits are confined to migration tests, explicit legacy docs, or historical inference/eval compatibility namespaces
- current comparator configs are not erased until they are routed under explicit legacy/comparator docs and no longer presented as new active defaults

- [ ] **Step 10.2: Remove active old code and configs**

Remove active use of:

- `recursive_detection_targets`
- `compute_recursive_detection_ce_batch_loss`
- active recursive trainer mixins
- active `bbox_geo`, `bbox_size_aux`, `coord_reg`
- active duplicate unlikelihood
- active recursive/ET-RMP config leaves
- active `ET_RMP_CE`, `et_rmp_like`, `typed_trie`, `random_permutation_et_rmp_ce`, `support_balance`, `trie_support`, `trie_balance`, or `et_rmp/` names
- active Stage-2 `token_ce` pipeline module semantics and `coord_gate`/`coord_gate_weight` keys for `objective.id: teacher_forcing`

Keep historical inference/eval only where explicitly namespaced as legacy.
Keep current comparator config handles only where explicitly namespaced as legacy/comparator and documented as non-default.

- [ ] **Step 10.3: Verify absence and migration tests**

Run:

```bash
conda run -n ms python -m pytest \
  tests/test_teacher_forcing_config_contract.py \
  tests/test_stage2_ab_config_contract.py \
  tests/test_teacher_forcing_metric_contract.py \
  -q
rg -n "ET_RMP_CE|et_rmp_like|typed_trie|recursive_detection_targets|compute_recursive_detection_ce_batch_loss|loss/recursive_detection_ce|recursive_detection_ce|prefix_rollin_et_rmp_ce|random_permutation_et_rmp_ce|support_balance|trie_support|trie_balance|et_rmp/|stage2_ab\\.pipeline\\.objective.*token_ce|coord_gate|coord_gate_weight" src configs tests
rg -n "recursive_detection_ce_latest|compact_full_support2|compact_full_prefix_rollin_balance2" docs configs tests
```

Expected: tests pass; remaining `rg` hits are only explicit legacy/migration tests, historical inference/eval paths, or documented legacy/comparator config handles.

- [ ] **Step 10.4: Commit**

```bash
git add src tests configs docs
git commit -m "refactor: remove active legacy teacher-forcing objectives"
```

## Task 11: End-To-End Smoke Validation

**Files:**

- Modify only if smoke failures expose integration bugs.

- [ ] **Step 11.1: Run focused unit suite**

Run:

```bash
conda run -n ms python -m pytest \
  tests/test_teacher_forcing_ir_contract.py \
  tests/test_teacher_forcing_objective_runner.py \
  tests/test_teacher_forcing_target_builder.py \
  tests/test_teacher_forcing_sidecar_bridge.py \
  tests/test_teacher_forcing_config_contract.py \
  tests/test_teacher_forcing_encoded_cache_contract.py \
  tests/test_compact_full_marker_policy.py \
  tests/test_infer_compact_full_policy_contract.py \
  tests/test_stage2_teacher_forcing_adapter_contract.py \
  tests/test_stage2_launcher_preflight_contract.py \
  tests/test_teacher_forcing_metric_contract.py \
  tests/test_teacher_forcing_report_contract.py \
  tests/test_detection_template_parsing_eval.py \
  tests/test_latest_detection_view_metadata.py \
  tests/test_training_runtime_sft_integration.py \
  tests/test_recursive_detection_ce_loss_adapter.py \
  tests/test_recursive_detection_ce_sft_wiring.py \
  tests/test_recursive_detection_ce_target_builder.py \
  tests/test_detection_training_dataset.py \
  -q
```

Expected: pass.

- [ ] **Step 11.2: Run config/materialization smoke**

Run the narrowest config parse/materialization commands already used by `tests/test_latest_training_config_contract.py` and `tests/test_stage2_ab_config_contract.py`.

Expected: new Stage-1 and Stage-2 teacher-forcing configs parse; old active ids fail.

- [ ] **Step 11.3: Run tiny train-side smoke only after unit gates pass**

Use the new smoke config created by Task 7 or Task 8. Label evidence `tiny`, not `val200` or full validation.

Expected:

- model forward receives no `teacher_forcing_target_ir`
- objective runner receives target IR
- loss keys use concrete `teacher_forcing/` leaves such as
  `teacher_forcing/loss/total`
- parser metrics use concrete `infer/parse/compact_full/` leaves such as
  `infer/parse/compact_full/error/legacy_separator_in_new_format`
- no recursive metric aliases emitted

- [ ] **Step 11.4: Commit final cleanup**

```bash
git status --short
git add <only files changed by smoke fixes>
git commit -m "test: validate teacher-forcing objective smoke"
```

## Task 12: Review And Approval Gate Before Production Runs

**Files:**

- Update: `docs/training/README.md`
- Update: `docs/IMPLEMENTATION_MAP.md`
- Update: `docs/catalog.yaml`
- Update: relevant progress note under `progress/`

- [ ] **Step 12.1: Request code review**

Use `superpowers:requesting-code-review` with:

```text
DESCRIPTION: Unified teacher-forcing objective refactor.
PLAN_OR_REQUIREMENTS: docs/superpowers/plans/2026-05-19-teacher-forcing-objective-refactor.md and openspec/changes/add-teacher-forcing-objective.
BASE_SHA: merge-base with main before implementation.
HEAD_SHA: current implementation head.
```

- [ ] **Step 12.2: Fix Critical and Important review findings**

Do not proceed to production training with unresolved Critical or Important findings.

- [ ] **Step 12.3: Produce implementation report**

Report must include:

- changed files
- tests run
- skipped checks and why
- tiny/smoke artifact roots
- old active surfaces removed
- remaining legacy inference/eval compatibility
- known risks before `val200`

- [ ] **Step 12.4: Ask for production-scale approval**

Do not launch `val200`, long training, or production-scale Stage-2 jobs until the user explicitly approves.

## Recommended Commit Slices

1. `feat: add teacher-forcing target IR`
2. `feat: add teacher-forcing objective runner`
3. `feat: add marker-delimited compact_full policy`
4. `feat: enforce teacher-forcing inference parser contract`
5. `feat: build teacher-forcing target IR`
6. `feat: bridge teacher-forcing target IR sidecar`
7. `feat: add teacher-forcing objective config`
8. `feat: enforce teacher-forcing cache contract`
9. `feat: integrate stage1 teacher-forcing objective`
10. `feat: adapt stage2 to teacher-forcing target IR`
11. `feat: add teacher-forcing diagnostics`
12. `refactor: route stage1 recursive owners to legacy boundary`
13. `refactor: remove active legacy teacher-forcing objectives`
14. `test: validate teacher-forcing objective smoke`

## Self-Review

Spec coverage:

- `teacher-forcing-objective`: Tasks 1, 2, 4, 5, 6, 7, 9, 9A, 10.
- `stage1-latest-detection-objectives`: Tasks 3, 6, 7, 9A, 10.
- `stage2-ab-training`: Tasks 6, 8, 10, 11.
- `teacher-forcing-unified-loss-registry`: Tasks 2, 8, 10.
- `teacher-forcing-objective-pipeline`: Tasks 2, 5, 8, 10.
- `encoded-training-cache`: Tasks 6A, 7, 11.
- `inference-pipeline`: Tasks 3, 3A, 9, 11.
- `trainer-metrics-components`: Tasks 9, 11.

Ambiguity decisions:

- V1 Stage-2 packing recommendation is fail-fast, not packed atom mapping.
- V1 training encoded cache is disabled for epoch-varying roll-in; fixed eval/probe cache is allowed only with teacher-forcing cache-key and payload validation.
- Old ET-RMP comparator is structural coverage profile, not an executable alias.
- Hard SFT is preserved as `objective.id: teacher_forcing`, `profile: hard_sft`.
- Objective execution extends the existing `src/training/objectives` and `src/training/supervision` framework; no parallel `src/objectives` framework should be created.

No implementation has started from this plan.
