# Bidirectional Type Gating Losses Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Promote `objective.terms.token_type_mass` on `main` as the stable bidirectional exclusive type-family loss for compact Stage-1 research teacher-forcing, independent from the experimental coverage-ledger branch.

**Architecture:** Reuse the existing Stage-1 research teacher-forcing objective path and replace the old allowed-type-mass helpers with a four-family schema/coord/desc/stop mass classifier. Keep the term additive to hard CE or valid-set likelihood, keep config YAML-first, and emit raw plus contribution metrics with no `_weighted` names.

**Tech Stack:** Python, PyTorch, pytest, OpenSpec, CoordExp `DetectionTrainingConfig`, `RoleVocab`, `TeacherForcingObjective`, Qwen3-VL compact special tokens.

---

## Source Of Truth

- OpenSpec change: `openspec/changes/bidirectional-type-gating-losses/`
- Existing config schema owner: `src/config/schema.py`
- Existing compact family builder: `src/detection/token_types.py`
- Existing role vocabulary: `src/training/teacher_forcing/vocab.py`
- Existing atom probability math: `src/training/teacher_forcing/probabilities.py`
- Existing objective reducer: `src/training/objectives/teacher_forcing.py`
- Existing trainer bridge: `src/trainers/metrics/teacher_forcing.py`
- Existing metric helpers: `src/training/teacher_forcing/metrics.py`
- Existing tests to update:
  - `tests/test_compact_type_gate.py`
  - `tests/test_teacher_forcing_config_contract.py`
  - `tests/test_teacher_forcing_objective_runner.py`
  - `tests/test_teacher_forcing_metric_contract.py`
  - `tests/test_detection_training_config_contract.py`
  - `tests/test_metric_events.py`

## Review Convergence Log

- 2026-06-24 initial split decision: stable bidirectional type-family gating is
  promoted on `main`; coverage ledger remains experimental in the ledger
  worktree.
- Main checkout state before drafting: `main` aligned with `origin/main`
  after `git fetch origin`; existing unrelated dirty files were preserved.
- OpenSpec CLI state: `openspec` binary was not found on PATH, so strict CLI
  validation is recorded as blocked until the tool is installed.
- Review closure rule: required lanes are OpenSpec governance, implementation
  plan/code fit, metric semantics, and workflow safety. A lane must report
  `approved`, `approved with P2`, or `blocked`. Timeout or disconnection is
  unresolved, not approval. Any open P0/P1 blocks source implementation.
- Round 1 review results:
  - OpenSpec governance: blocked with P1s; accepted fixes are additive
    requirements, no global `struct/eos` rename, and no partial objective-id
    migration.
  - Implementation/code fit: blocked with P1s; accepted fixes are no ledger
    worktree mutation in this plan and explicit metric-contract/runtime-payload
    tests.
  - Metric semantics: blocked on implementation/test specificity; accepted
    fixes are exact denominator, contribution, family-mass, and out-of-family
    tests.
  - Workflow safety: blocked with P1s; accepted fixes are explicit reviewer
    closure and path-scoped dirty-main guards.
- Round 2 review is required after these revisions before source edits begin.

## Direct Main Safety Gate

This plan intentionally edits `/data/CoordExp` on `main` because the user
explicitly requested no new worktree for the stable loss split. The checkout is
dirty with unrelated local files; preserve them.

Implementation target paths:

```text
src/config/schema.py
src/sft.py
src/detection/token_types.py
src/training/teacher_forcing/probabilities.py
src/training/objectives/teacher_forcing.py
src/trainers/metrics/teacher_forcing.py
src/training/teacher_forcing/metrics.py
docs/training/METRICS.md
tests/test_compact_type_gate.py
tests/test_teacher_forcing_config_contract.py
tests/test_teacher_forcing_objective_runner.py
tests/test_teacher_forcing_metric_contract.py
tests/test_detection_training_config_contract.py
tests/test_metric_events.py
openspec/changes/bidirectional-type-gating-losses
docs/superpowers/plans/2026-06-24-bidirectional-type-gating-losses.md
```

Before source edits run:

```bash
git status --short --branch
git status --short -- src/config/schema.py src/sft.py src/detection/token_types.py src/training/teacher_forcing/probabilities.py src/training/objectives/teacher_forcing.py src/trainers/metrics/teacher_forcing.py src/training/teacher_forcing/metrics.py docs/training/METRICS.md tests/test_compact_type_gate.py tests/test_teacher_forcing_config_contract.py tests/test_teacher_forcing_objective_runner.py tests/test_teacher_forcing_metric_contract.py tests/test_detection_training_config_contract.py tests/test_metric_events.py openspec/changes/bidirectional-type-gating-losses docs/superpowers/plans/2026-06-24-bidirectional-type-gating-losses.md
```

If any target path is already dirty, inspect its scoped diff before editing and
work with the existing change rather than reverting it. After implementation,
run the same path-scoped status plus:

```bash
git diff -- src/config/schema.py src/sft.py src/detection/token_types.py src/training/teacher_forcing/probabilities.py src/training/objectives/teacher_forcing.py src/trainers/metrics/teacher_forcing.py src/training/teacher_forcing/metrics.py docs/training/METRICS.md tests/test_compact_type_gate.py tests/test_teacher_forcing_config_contract.py tests/test_teacher_forcing_objective_runner.py tests/test_teacher_forcing_metric_contract.py tests/test_detection_training_config_contract.py tests/test_metric_events.py openspec/changes/bidirectional-type-gating-losses docs/superpowers/plans/2026-06-24-bidirectional-type-gating-losses.md
```

Do not stage or commit unless the user asks. If a later commit is requested,
use pathspec-only staging for the reviewed target paths; do not use `git add -A`
or `git commit -a`.

## File Structure

- Modify `src/config/schema.py`
  - Add `TeacherForcingTokenTypeMassConfig(enabled: bool = False, weight: float = 1.0)`.
  - Parse `objective.terms.token_type_mass` strictly with only `enabled` and `weight`.
  - Let `hard_sft` enable `token_type_mass`; keep the other target-IR-only term rejections.
- Modify `src/sft.py`
  - Include `token_type_mass.weight` in `_detection_objective_runtime_payload`.
- Modify `src/detection/token_types.py`
  - Include closed compact structural tokens when present.
  - Add family-mass vocabulary helpers while preserving
    `allowed_type_token_ids_for_target` and `combine_main_and_type_losses` as
    backward-compatible exports for legacy/comparator recursive-detection paths.
- Modify `src/training/teacher_forcing/probabilities.py`
  - Add a bidirectional four-family mass helper returning type loss, target mass, and family name.
  - Keep the existing valid-set and coverage math intact.
- Modify `src/training/objectives/teacher_forcing.py`
  - Read `token_type_mass_enabled` and `token_type_mass_weight` from `ObjectiveSpec.config`.
  - Add the type-mass contribution only when enabled.
  - Aggregate raw/contribution and family mass metrics.
- Modify `src/trainers/metrics/teacher_forcing.py`
  - Pass `token_type_mass_enabled` and `token_type_mass_weight` from `teacher_forcing_objective_cfg` to `ObjectiveSpec`.
- Modify `src/training/teacher_forcing/metrics.py`
  - Add metric-event helpers or direct events for raw, contribution, and family mass means.
  - Retire or realign the existing diagnostic-only
    `teacher_forcing/loss/token_type_mass` emission so it cannot collide with
    the promoted stable non-diagnostic objective metric identity.
- Modify `docs/training/METRICS.md`
  - Document the promoted raw, contribution, and family-mass metric keys.
- Update active tests named in each task.

## Task 1: Config Contract

**Files:**
- Modify: `src/config/schema.py`
- Modify: `src/sft.py`
- Test: `tests/test_teacher_forcing_config_contract.py`
- Test: `tests/test_detection_training_config_contract.py`

- [x] **Step 1: Update config tests for hard SFT plus token type mass**

In `tests/test_teacher_forcing_config_contract.py`, change the current
`test_hard_sft_rejects_target_ir_only_terms` parametrization so
`token_type_mass` is no longer rejected. Add a positive test:

```python
def test_hard_sft_accepts_token_type_mass_with_weight() -> None:
    payload = _latest_teacher_payload(
        profile="hard_sft",
        terms={
            "token_type_mass": {"enabled": True, "weight": 1.0},
            "conditional_valid_set_likelihood": {"enabled": False},
            "within_valid_coverage": {"enabled": False, "coverage_strength": 0.0},
            "continuation_margin": {"enabled": False},
        },
    )

    cfg = DetectionTrainingConfig.from_mapping(payload)

    assert cfg.objective.profile == "hard_sft"
    assert cfg.objective.terms.token_type_mass.enabled is True
    assert cfg.objective.terms.token_type_mass.weight == 1.0
```

Update the runtime payload test so a non-default weight round-trips:

```python
def test_sft_runtime_payload_preserves_token_type_mass_weight() -> None:
    from src.sft import _detection_objective_runtime_payload

    terms = _teacher_forcing_objective()["terms"]
    terms = {
        **terms,
        "token_type_mass": {"enabled": True, "weight": 0.2},
    }
    cfg = DetectionTrainingConfig.from_mapping(_latest_teacher_payload(terms=terms))

    payload = _detection_objective_runtime_payload(cfg)

    assert payload is not None
    assert payload["terms"]["token_type_mass"]["enabled"] is True
    assert payload["terms"]["token_type_mass"]["weight"] == 0.2
```

Add strict-key tests:

```python
@pytest.mark.parametrize("payload", ({"mode": "allowed_type_mass"}, {"extra": True}))
def test_token_type_mass_rejects_unknown_keys(payload: dict[str, object]) -> None:
    objective = _teacher_forcing_objective()
    objective["terms"] = {
        **objective["terms"],
        "token_type_mass": {"enabled": True, "weight": 1.0, **payload},
    }
    with pytest.raises(ValueError, match=r"objective\.terms\.token_type_mass"):
        DetectionTrainingConfig.from_mapping(_latest_teacher_payload(terms=objective["terms"]))
```

- [x] **Step 2: Run config tests and confirm the expected failure**

Run:

```bash
python -m pytest tests/test_teacher_forcing_config_contract.py -q
```

Expected before implementation: failure on missing `weight` field support or
`hard_sft` rejection.

- [x] **Step 3: Implement strict token type mass config**

In `src/config/schema.py`, replace the `TeacherForcingEnabledModuleConfig`
field for `token_type_mass` with:

```python
@dataclass(frozen=True)
class TeacherForcingTokenTypeMassConfig:
    enabled: bool = False
    weight: float = 1.0

    def __post_init__(self) -> None:
        _detection_validate_bool(
            self.enabled,
            path="objective.terms.token_type_mass.enabled",
        )
        if isinstance(self.weight, bool) or not isinstance(self.weight, (int, float)):
            raise TypeError("objective.terms.token_type_mass.weight must be numeric")
        value = float(self.weight)
        if not math.isfinite(value):
            raise ValueError("objective.terms.token_type_mass.weight must be finite")
        if value < 0.0:
            raise ValueError("objective.terms.token_type_mass.weight must be >= 0")
        object.__setattr__(self, "weight", value)
```

Update `TeacherForcingModulesConfig.token_type_mass` to use that dataclass and
parse it with `parse_dataclass_strict(..., path="objective.terms.token_type_mass")`.

- [x] **Step 4: Update hard-SFT validation**

In `TeacherForcingObjectiveConfig.__post_init__`, remove
`objective.terms.token_type_mass.enabled` from the `hard_sft_module_checks`
tuple. Keep the checks for `conditional_valid_set_likelihood`,
`within_valid_coverage`, `coverage_strength`, and `continuation_margin`.

- [x] **Step 5: Persist the weight in runtime payloads**

In `src/sft.py::_detection_objective_runtime_payload`, add:

```python
"weight": _get_section_value(token_type_mass_cfg, "weight")
```

inside the existing `payload["terms"]["token_type_mass"]` mapping.

- [x] **Step 6: Verify config contract**

Run:

```bash
python -m pytest tests/test_teacher_forcing_config_contract.py tests/test_detection_training_config_contract.py -q
```

Expected after implementation: pass.

## Task 2: Family Vocabulary

**Files:**
- Modify: `src/detection/token_types.py`
- Test: `tests/test_compact_type_gate.py`

- [x] **Step 1: Update token group tests**

Extend `TypeGateTokenizer` in `tests/test_compact_type_gate.py` with:

```python
"<|object_ref_end|>": 8,
"<|box_end|>": 9,
```

Assert those IDs are in `groups.struct`, while `<|im_end|>` is in `groups.eos`
and `<|endoftext|>` / `<|end_of_text|>` are not.

- [x] **Step 2: Rename tests toward family mass**

Keep the existing backward-compatible helper tests and add new explicit tests
named like:

```python
def test_compact_token_groups_include_closed_schema_tokens() -> None:
    tokenizer = TypeGateTokenizer()
    groups = build_compact_token_type_groups(tokenizer)

    assert tokenizer.convert_tokens_to_ids("<|object_ref_end|>") in groups.struct
    assert tokenizer.convert_tokens_to_ids("<|box_end|>") in groups.struct
```

Do not remove `allowed_type_token_ids_for_target` or
`combine_main_and_type_losses`; legacy recursive-detection comparator code still
imports those helpers.

- [x] **Step 3: Run the focused test and confirm failure**

Run:

```bash
python -m pytest tests/test_compact_type_gate.py -q
```

Expected before implementation: failure because closed structural tokens are
not yet added to `_COMPACT_STRUCT_TOKENS`.

- [x] **Step 4: Implement the vocabulary update**

In `src/detection/token_types.py`, import the closed-token constants from the
current compact row/template module if available. If the constants are not
exported, define local strings:

```python
OBJECT_REF_END_TOKEN = "<|object_ref_end|>"
BOX_END_TOKEN = "<|box_end|>"
_COMPACT_STRUCT_TOKENS = (
    OBJECT_REF_START_TOKEN,
    OBJECT_REF_END_TOKEN,
    BOX_START_TOKEN,
    BOX_END_TOKEN,
)
```

Keep `_resolve_known_tokens` best-effort so older lightweight tokenizers that
do not know closed tokens still work.

- [x] **Step 5: Verify family vocabulary tests**

Run:

```bash
python -m pytest tests/test_compact_type_gate.py -q
```

Expected after implementation: pass.

## Task 3: Bidirectional Family-Mass Probability Math

**Files:**
- Modify: `src/training/teacher_forcing/probabilities.py`
- Test: `tests/test_teacher_forcing_objective_runner.py`

- [x] **Step 1: Add probability tests**

Add tests that compare low loss when the selected family has high mass and high
loss when a competing family has high mass. These tests must assert exact
four-family softmax values, not only relative ordering:

```python
def test_token_type_mass_rewards_selected_family_over_competing_families() -> None:
    role_vocab = _role_vocab(text_ids={1, 2}, schema_ids={3}, coord_ids={4}, stop_id=5)
    atom = _atom(
        allowed_token_roles=frozenset({TokenRole.TEXT}),
        selected_token_role=TokenRole.TEXT,
        valid_token_ids=frozenset({1}),
        selected_token_id=1,
    )
    good_logits = torch.tensor([0.0, 5.0, 4.0, -2.0, -2.0, -2.0], dtype=torch.float32)
    bad_logits = torch.tensor([0.0, -2.0, -2.0, 5.0, 4.0, 3.0], dtype=torch.float32)

    good = teacher_forcing_atom_loss(
        good_logits,
        atom=atom,
        role_vocab=role_vocab,
        coverage_strength=0.0,
        token_type_mass_enabled=True,
        token_type_mass_weight=1.0,
    )
    bad = teacher_forcing_atom_loss(
        bad_logits,
        atom=atom,
        role_vocab=role_vocab,
        coverage_strength=0.0,
        token_type_mass_enabled=True,
        token_type_mass_weight=1.0,
    )

    good_expected_masses = _manual_family_masses(good_logits, role_vocab)
    bad_expected_masses = _manual_family_masses(bad_logits, role_vocab)
    assert good.token_type_mass.item() < bad.token_type_mass.item()
    assert good.target_family == "desc"
    assert good.token_type_mass.item() == pytest.approx(
        -torch.log(good_expected_masses["desc"]).item()
    )
    assert good.target_family_mass.item() == pytest.approx(
        good_expected_masses["desc"].item()
    )
    assert bad.token_type_mass.item() == pytest.approx(
        -torch.log(bad_expected_masses["desc"]).item()
    )
    assert bad.target_family_mass.item() == pytest.approx(
        bad_expected_masses["desc"].item()
    )
    for family_name in ("schema", "desc", "coord", "stop"):
        assert good.family_masses[family_name].item() == pytest.approx(
            good_expected_masses[family_name].item()
        )
        assert bad.family_masses[family_name].item() == pytest.approx(
            bad_expected_masses[family_name].item()
        )
```

Add an exact denominator-exclusion test:

```python
def test_token_type_mass_excludes_out_of_family_logits_from_denominator() -> None:
    role_vocab = _role_vocab(text_ids={1}, schema_ids={2}, coord_ids={3}, stop_id=4)
    atom = _atom(
        allowed_token_roles=frozenset({TokenRole.TEXT}),
        selected_token_role=TokenRole.TEXT,
        valid_token_ids=frozenset({1}),
        selected_token_id=1,
    )
    base_logits = torch.tensor([0.0, 1.0, 0.5, -0.25, -1.0, -20.0], dtype=torch.float32)
    high_out_of_family = base_logits.clone()
    high_out_of_family[5] = 100.0

    base = teacher_forcing_atom_loss(
        base_logits,
        atom=atom,
        role_vocab=role_vocab,
        coverage_strength=0.0,
        token_type_mass_enabled=True,
        token_type_mass_weight=1.0,
    )
    shifted = teacher_forcing_atom_loss(
        high_out_of_family,
        atom=atom,
        role_vocab=role_vocab,
        coverage_strength=0.0,
        token_type_mass_enabled=True,
        token_type_mass_weight=1.0,
    )

    family_logits = torch.stack(
        [
            torch.logsumexp(base_logits[torch.tensor([2])], dim=0),
            torch.logsumexp(base_logits[torch.tensor([1])], dim=0),
            torch.logsumexp(base_logits[torch.tensor([3])], dim=0),
            torch.logsumexp(base_logits[torch.tensor([4])], dim=0),
        ]
    )
    expected = -torch.log_softmax(family_logits, dim=0)[1]
    assert base.token_type_mass.item() == pytest.approx(expected.item())
    assert shifted.token_type_mass.item() == pytest.approx(expected.item())
```

Add a stop-vs-schema test:

```python
def test_stop_token_uses_stop_family_not_schema_family() -> None:
    role_vocab = _role_vocab(text_ids={1}, schema_ids={2}, coord_ids={3}, stop_id=4)
    atom = _atom(
        allowed_token_roles=frozenset({TokenRole.STOP}),
        selected_token_role=TokenRole.STOP,
        valid_token_ids=frozenset({4}),
        selected_token_id=4,
    )
    logits = torch.tensor([0.0, -3.0, 5.0, -3.0, 1.0], dtype=torch.float32)

    loss = teacher_forcing_atom_loss(
        logits,
        atom=atom,
        role_vocab=role_vocab,
        coverage_strength=0.0,
        token_type_mass_enabled=True,
        token_type_mass_weight=1.0,
    )

    assert loss.target_family == "stop"
    assert loss.family_masses["schema"].item() > loss.family_masses["stop"].item()
    assert loss.token_type_mass.item() > 1.0
```

- [x] **Step 2: Run the focused probability tests and confirm failure**

Run:

```bash
python -m pytest tests/test_teacher_forcing_objective_runner.py -q
```

Expected before implementation: failure because `teacher_forcing_atom_loss`
does not expose token-type-mass arguments or return fields.

- [x] **Step 3: Extend the atom loss dataclass**

In `src/training/teacher_forcing/probabilities.py`, extend
`TeacherForcingAtomLoss` with:

```python
token_type_mass: torch.Tensor
token_type_mass_contribution: torch.Tensor
target_family_mass: torch.Tensor
target_family: str | None
family_masses: Mapping[str, torch.Tensor]
```

Use immutable mappings for returned `family_masses`.

- [x] **Step 4: Add the family-mass helper**

Implement a helper that:

1. maps `TokenRole.SCHEMA -> "schema"`, `TokenRole.TEXT -> "desc"`,
   `TokenRole.COORD -> "coord"`, and `TokenRole.STOP -> "stop"`;
2. gathers `role_vocab.schema_token_ids`, `text_token_ids`,
   `coord_token_ids`, and `stop_token_ids`;
3. computes full-vocab `log_probs` once;
4. computes one `logsumexp` per family;
5. applies `torch.log_softmax(torch.stack(family_logits), dim=0)`;
6. returns `-family_log_probs[target_index]`, the selected family mass, and all
   family masses.

- [x] **Step 5: Wire the helper into `teacher_forcing_atom_loss`**

Add keyword-only arguments:

```python
token_type_mass_enabled: bool = False
token_type_mass_weight: float = 1.0
```

When disabled, return graph-anchored zeros for `token_type_mass` and
`token_type_mass_contribution`. When enabled, validate nonnegative finite
`token_type_mass_weight`, compute the raw family loss, and add
`token_type_mass_weight * token_type_mass` to `total`.

- [x] **Step 6: Verify probability tests**

Run:

```bash
python -m pytest tests/test_teacher_forcing_objective_runner.py -q
```

Expected after implementation: pass.

## Task 4: Objective And Trainer Wiring

**Files:**
- Modify: `src/training/objectives/teacher_forcing.py`
- Modify: `src/trainers/metrics/teacher_forcing.py`
- Modify: `src/training/teacher_forcing/metrics.py`
- Test: `tests/test_teacher_forcing_objective_runner.py`

- [x] **Step 1: Add objective-level metric tests**

Add a runner test that enables the term through `ObjectiveSpec.config` and
asserts exact raw/contribution values:

```python
def test_teacher_forcing_objective_reports_token_type_mass_metrics() -> None:
    role_vocab = _role_vocab(text_ids={1}, schema_ids={2}, coord_ids={3}, stop_id=4)
    atom = _atom(
        allowed_token_roles=frozenset({TokenRole.TEXT}),
        selected_token_role=TokenRole.TEXT,
        valid_token_ids=frozenset({1}),
        selected_token_id=1,
    )
    result = ObjectiveRunner().run(
        logits=torch.zeros((1, 2, 6), dtype=torch.float32),
        supervision=SupervisionBatch(spans=(_span(_ir(atom)),), batch_id="batch-1"),
        objectives=(
            ObjectiveSpec(
                "teacher_forcing",
                config={
                    "input_ids": torch.tensor([[0, 1]], dtype=torch.long),
                    "role_vocab": role_vocab,
                    "token_type_mass_enabled": True,
                    "token_type_mass_weight": 0.2,
                },
            ),
        ),
        sample_id_to_batch_index={"sample-1": 0},
    )
    metrics = {event.key: event for event in result.metric_events}

    assert "teacher_forcing/loss/token_type_mass" in metrics
    assert "teacher_forcing/loss/token_type_mass/contribution" in metrics
    raw = metrics["teacher_forcing/loss/token_type_mass"]
    contribution = metrics["teacher_forcing/loss/token_type_mass/contribution"]
    assert contribution.numerator == pytest.approx(raw.numerator * 0.2)
    assert contribution.denominator == raw.denominator
    assert all(not key.endswith("_weighted") for key in metrics)
```

In `tests/test_teacher_forcing_metric_contract.py`, update
`REQUIRED_TEACHER_FORCING_METRIC_KEYS` coverage and add reducer/flattening
assertions for:

```text
teacher_forcing/loss/token_type_mass
teacher_forcing/loss/token_type_mass/contribution
teacher_forcing/type/schema_mass_at_schema
teacher_forcing/type/coord_mass_at_coord
teacher_forcing/type/desc_mass_at_desc
teacher_forcing/type/stop_mass_at_stop
```

Use two weighted-mean event groups with unequal active atom counts to prove
reduction uses active atom denominators, not an unweighted mean of batch means.
Also assert that a missing family denominator omits the corresponding flattened
key instead of emitting `0.0`.

Also add a mixed-event identity regression test: combine the promoted
token-type-mass objective metric events with the remaining diagnostic events and
assert there is no duplicate `teacher_forcing/loss/token_type_mass`
diagnostic-only identity. The promoted key must flatten with
`metric_surface="objective_loss"` and `diagnostic_only=False`.

- [x] **Step 2: Implement objective config reads**

In `TeacherForcingObjective.run`, read:

```python
token_type_mass_enabled = config_bool(spec.config, "token_type_mass_enabled", default=False)
token_type_mass_weight = config_float(spec.config, "token_type_mass_weight", default=1.0, minimum=0.0)
```

If `config_bool` does not exist, add a small local helper matching the existing
`config_float` style.

- [x] **Step 3: Pass config into atom loss**

Pass the two config values to `teacher_forcing_atom_loss` and aggregate:

- raw `atom_loss.token_type_mass`
- contribution `atom_loss.token_type_mass_contribution`
- target-family mass buckets by `atom_loss.target_family`

- [x] **Step 4: Emit metrics**

Use `MetricEvent` helpers in `src/training/teacher_forcing/metrics.py` or local
objective events to emit:

```text
teacher_forcing/loss/token_type_mass
teacher_forcing/loss/token_type_mass/contribution
teacher_forcing/type/schema_mass_at_schema
teacher_forcing/type/coord_mass_at_coord
teacher_forcing/type/desc_mass_at_desc
teacher_forcing/type/stop_mass_at_stop
```

Use denominator counts from the matching active atoms. Omit family mass metrics
when no atom targets that family.

Emit these as stable flattened metric keys with ordinary weighted means and
`diagnostic_only=False`. Use `metric_surface="objective_loss"` for raw and
contribution, and `metric_surface="type_family_mass"` for family-mass gauges.
Do not leave a second diagnostic-only writer for
`teacher_forcing/loss/token_type_mass`; remove that diagnostic slot or route it
through the same promoted identity.

- [x] **Step 5: Wire trainer config**

In `src/trainers/metrics/teacher_forcing.py`, add helpers:

```python
def _token_type_mass_enabled(objective_cfg: Any) -> bool:
    cfg = _token_type_mass_cfg(objective_cfg)
    if cfg is None:
        return False
    value = _cfg_get(cfg, "enabled", False)
    if type(value) is not bool:
        raise TypeError("token_type_mass.enabled must be a bool")
    return value

def _token_type_mass_weight(objective_cfg: Any) -> float:
    cfg = _token_type_mass_cfg(objective_cfg)
    if cfg is None:
        return 1.0
    value = _cfg_get(cfg, "weight", 1.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("token_type_mass.weight must be a finite numeric scalar")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("token_type_mass.weight must be finite")
    if parsed < 0.0:
        raise ValueError("token_type_mass.weight must be >= 0")
    return parsed
```

Pass both values into `ObjectiveSpec.config` next to `coverage_strength`.

- [x] **Step 6: Verify objective and trainer tests**

Run:

```bash
python -m pytest tests/test_teacher_forcing_objective_runner.py tests/test_teacher_forcing_metric_contract.py tests/test_teacher_forcing_config_contract.py -q
```

Expected after implementation: pass.

## Task 5: Docs And Final Verification

**Files:**
- Modify: `docs/training/METRICS.md`
- Verify target source and test files only.

- [x] **Step 1: Update training metrics docs**

Add the promoted metric keys to the Stage-1 teacher-forcing metric section of
`docs/training/METRICS.md`, with these semantics:

```text
teacher_forcing/loss/token_type_mass: raw mean type-family loss.
teacher_forcing/loss/token_type_mass/contribution: weight * raw mean.
teacher_forcing/type/schema_mass_at_schema: schema-family mass at schema targets.
teacher_forcing/type/coord_mass_at_coord: coord-family mass at coord targets.
teacher_forcing/type/desc_mass_at_desc: desc-family mass at desc targets.
teacher_forcing/type/stop_mass_at_stop: stop-family mass at stop targets.
```

- [x] **Step 2: Run targeted tests**

Run:

```bash
python -m pytest tests/test_compact_type_gate.py tests/test_teacher_forcing_objective_runner.py tests/test_teacher_forcing_metric_contract.py tests/test_teacher_forcing_config_contract.py tests/test_detection_training_config_contract.py -q
python -m pytest tests/test_objective_runner_math.py tests/test_metric_events.py -q
```

- [x] **Step 3: Run diff hygiene and scoped dirty-tree checks**

Run:

```bash
git diff --check -- src/config/schema.py src/sft.py src/detection/token_types.py src/training/teacher_forcing/probabilities.py src/training/objectives/teacher_forcing.py src/trainers/metrics/teacher_forcing.py src/training/teacher_forcing/metrics.py docs/training/METRICS.md tests/test_compact_type_gate.py tests/test_teacher_forcing_config_contract.py tests/test_teacher_forcing_objective_runner.py tests/test_teacher_forcing_metric_contract.py tests/test_detection_training_config_contract.py tests/test_metric_events.py openspec/changes/bidirectional-type-gating-losses docs/superpowers/plans/2026-06-24-bidirectional-type-gating-losses.md
git status --short -- src/config/schema.py src/sft.py src/detection/token_types.py src/training/teacher_forcing/probabilities.py src/training/objectives/teacher_forcing.py src/trainers/metrics/teacher_forcing.py src/training/teacher_forcing/metrics.py docs/training/METRICS.md tests/test_compact_type_gate.py tests/test_teacher_forcing_config_contract.py tests/test_teacher_forcing_objective_runner.py tests/test_teacher_forcing_metric_contract.py tests/test_detection_training_config_contract.py tests/test_metric_events.py openspec/changes/bidirectional-type-gating-losses docs/superpowers/plans/2026-06-24-bidirectional-type-gating-losses.md
git diff -- src/config/schema.py src/sft.py src/detection/token_types.py src/training/teacher_forcing/probabilities.py src/training/objectives/teacher_forcing.py src/trainers/metrics/teacher_forcing.py src/training/teacher_forcing/metrics.py docs/training/METRICS.md tests/test_compact_type_gate.py tests/test_teacher_forcing_config_contract.py tests/test_teacher_forcing_objective_runner.py tests/test_teacher_forcing_metric_contract.py tests/test_detection_training_config_contract.py tests/test_metric_events.py openspec/changes/bidirectional-type-gating-losses docs/superpowers/plans/2026-06-24-bidirectional-type-gating-losses.md
```

- [x] **Step 4: Record OpenSpec CLI state**

Run:

```bash
command -v openspec
```

If available, also run:

```bash
openspec validate bidirectional-type-gating-losses --type change --strict
```

If unavailable, record the missing binary in the final report and do not claim
OpenSpec CLI validation passed.

## Stop Condition

This implementation plan stops after stable loss implementation and
verification on `/data/CoordExp` `main`. Any follow-up outside the main checkout
is not part of this plan and requires a separate scoped step.
