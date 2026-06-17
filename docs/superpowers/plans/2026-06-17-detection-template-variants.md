# Detection Template Variants Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement semantic detection template variants so `detection_template.id` is the single training, inference, artifact, token-row, and evaluator contract.

**Architecture:** Add a small template contract resolver that maps semantic ids to render, parse, prompt, token-row, and provenance behavior. Update existing strict template, teacher-forcing, prompt, inference, and evaluator surfaces to consume that resolver instead of independently authored compact format knobs. Keep standard post-hoc mAP on normalized `gt_vs_pred.pred` objects while using template metadata for inference materialization and evaluation preflight.

**Tech Stack:** Python, pytest, OpenSpec, Qwen3-VL tokenizer special tokens, CoordExp detection template/rendering stack, PEFT coord-offset adapter metadata.

---

## Source Of Truth

- OpenSpec change: `openspec/changes/detection-template-variants/`
- Normative implementation tasks: `openspec/changes/detection-template-variants/tasks.md`
- Current strict template owner: `src/detection/template.py`
- Compatibility facade boundary: `src/common/detection_sequence.py`
- Low-level compact row helpers: `src/common/detection_compact_rows.py`
- Native special-token id constants: `src/tokens/qwen_native.py`
- Teacher-forcing target owner: `src/detection/teacher_forcing/target_builder.py`
- Config schema owner: `src/config/schema.py`
- Prompt owner: `src/config/prompts.py`
- Inference shared seam: `src/infer/runtime.py`
- Adapter checkpoint validator: `src/infer/checkpoints.py`
- Evaluator parsing/preflight owner: `src/detection/evaluation.py`

## Review Convergence

- 2026-06-17 subagent self-review completed with three read-only reviewers:
  OpenSpec coverage, implementation ownership, and verification sufficiency.
- No reviewer found a P0 blocker.
- Accepted P1/P2 findings have been folded into this roadmap:
  - canonical authoring is `detection_template.id` for both training and
    inference configs; old inference compact knobs are rejection-only migration
    diagnostics, not aliases;
  - training cache fingerprints and training resolved-config provenance must
    include `detection_template.id`;
  - token-row tests must assert exact structural row identity, not just
    1002/1003/1004 counts;
  - teacher-forcing template choice must propagate from dataset/runtime call
    sites into target building;
  - backend prompt parity must compare final prompt payload bytes and template
    metadata at the request boundary;
  - post-hoc mAP metadata preflight must be covered in dependency-light tests,
    with pycocotools used only for metric equality checks;
- active residue scans should focus on current code/config/docs/tests/scripts and
  classify `configs/archive/` and `docs/history/` as historical unless a current
  entrypoint imports them.
- 2026-06-17 convergence recheck found no P0/P1 blockers after these revisions;
  `openspec validate detection-template-variants --type change --strict` passed.
- Per user approval, implementation may start after this revised roadmap and
  OpenSpec validation pass.

## File Structure

- Create `src/detection/template_contracts.py`
  - Owns semantic template ids, structural token constants, required row ids, compact row rendering policy, prompt row pattern, parser dispatch metadata, and migration predicates.
- Modify `src/common/detection_compact_rows.py`
  - Adds `OBJECT_REF_END_TOKEN` and `BOX_END_TOKEN` constants and generic row rendering helper support.
- Modify `src/tokens/qwen_native.py`
  - Adds `OBJECT_REF_END_TOKEN`, `BOX_END_TOKEN`,
    `EXPECTED_OBJECT_REF_END_ID = 151647`, and
    `EXPECTED_BOX_END_ID = 151649`.
- Modify `src/detection/teacher_forcing/compact_full_policy.py`
  - Keeps existing compact parser behavior available while adding template-aware render/parse helpers or delegating to the new contract resolver.
- Modify `src/detection/template.py`
  - Exposes `compact`, `compact_box_closed`, `compact_object_box_closed`, and `compact_object_box_closed_lines` through `get_detection_template`.
- Modify `src/config/schema.py`
  - Replaces `compact_full` authored schema values with semantic ids and derives token-row validation.
- Modify `src/detection/runtime.py`
  - Resolves compact runtime support from semantic template ids and records the
    template id in training provenance/fingerprints.
- Modify `src/detection/teacher_forcing/target_builder.py`
  - Builds object branches and target IR from selected template contract.
- Modify `src/detection/teacher_forcing/description_tokens.py`
  - Tokenizes description contexts with optional object-ref-end and box-end structure.
- Modify `src/config/prompts.py`
  - Builds prompt examples and prompt hashes from `detection_template.id`.
- Modify `src/infer/pipeline.py`, `src/infer/runtime.py`, `src/infer/artifacts.py`, and `src/infer/checkpoints.py`
  - Removes independent compact parse/row-separator source-of-truth behavior, records template metadata, and validates adapter rows.
- Modify `src/detection/evaluation.py`
  - Uses template metadata for post-change artifact preflight and keeps mAP scoring on normalized `gt` and `pred` objects.
- Modify active config/docs paths that currently present `compact_full` as current guidance.
- Add or modify tests under `tests/` named in each task below.
  - New current-surface tests and test functions should use `compact` or
    `detection_template` naming. Keep `compact_full` in test names only for
    explicit legacy rejection, migration diagnostics, or compatibility shims
    that have not yet been renamed.

## Task 1: Semantic Template Contract Resolver

**Files:**
- Create: `src/detection/template_contracts.py`
- Modify: `src/common/detection_compact_rows.py`
- Modify: `src/tokens/qwen_native.py`
- Test: `tests/test_detection_template_variants.py`

- [x] **Step 1: Write resolver tests**

Create `tests/test_detection_template_variants.py` with tests that assert:

```python
import pytest

from src.detection.template_contracts import (
    COMPACT_TEMPLATE_IDS,
    BOX_END_TOKEN,
    OBJECT_REF_END_TOKEN,
    required_trainable_token_count,
    required_trainable_token_row_ids,
    render_compact_contract_row,
    resolve_detection_template_contract,
)


def test_semantic_template_ids_are_exact() -> None:
    assert COMPACT_TEMPLATE_IDS == (
        "compact",
        "compact_box_closed",
        "compact_object_box_closed",
        "compact_object_box_closed_lines",
    )


@pytest.mark.parametrize(
    ("template_id", "expected"),
    [
        (
            "compact",
            "<|object_ref_start|>cat<|box_start|><|coord_1|><|coord_2|><|coord_3|><|coord_4|>",
        ),
        (
            "compact_box_closed",
            "<|object_ref_start|>cat<|box_start|><|coord_1|><|coord_2|><|coord_3|><|coord_4|><|box_end|>",
        ),
        (
            "compact_object_box_closed",
            "<|object_ref_start|>cat<|object_ref_end|><|box_start|><|coord_1|><|coord_2|><|coord_3|><|coord_4|><|box_end|>",
        ),
        (
            "compact_object_box_closed_lines",
            "<|object_ref_start|>cat<|object_ref_end|><|box_start|><|coord_1|><|coord_2|><|coord_3|><|coord_4|><|box_end|>\n",
        ),
    ],
)
def test_contract_row_rendering(template_id: str, expected: str) -> None:
    contract = resolve_detection_template_contract(template_id)
    row = render_compact_contract_row(
        contract,
        desc="cat",
        bbox_tokens=("<|coord_1|>", "<|coord_2|>", "<|coord_3|>", "<|coord_4|>"),
    )
    assert row == expected


@pytest.mark.parametrize(
    ("template_id", "count"),
    [
        ("compact", 1002),
        ("compact_box_closed", 1003),
        ("compact_object_box_closed", 1004),
        ("compact_object_box_closed_lines", 1004),
    ],
)
def test_required_trainable_token_count(template_id: str, count: int) -> None:
    assert required_trainable_token_count(template_id) == count


@pytest.mark.parametrize(
    ("template_id", "expected_structural_ids"),
    [
        ("compact", (151646, 151648)),
        ("compact_box_closed", (151646, 151648, 151649)),
        ("compact_object_box_closed", (151646, 151647, 151648, 151649)),
        ("compact_object_box_closed_lines", (151646, 151647, 151648, 151649)),
    ],
)
def test_required_trainable_token_row_ids(
    template_id: str,
    expected_structural_ids: tuple[int, ...],
) -> None:
    row_ids = required_trainable_token_row_ids(template_id)
    assert row_ids[: len(expected_structural_ids)] == expected_structural_ids
    assert row_ids[len(expected_structural_ids) :] == tuple(range(151670, 152670))


def test_closed_token_constants_are_available() -> None:
    assert OBJECT_REF_END_TOKEN == "<|object_ref_end|>"
    assert BOX_END_TOKEN == "<|box_end|>"


def test_compact_full_is_rejected() -> None:
    with pytest.raises(ValueError, match="compact_full"):
        resolve_detection_template_contract("compact_full")
```

- [x] **Step 2: Run the new tests and confirm they fail**

Run:

```bash
python -m pytest tests/test_detection_template_variants.py -q
```

Expected: import failure for `src.detection.template_contracts`.

- [x] **Step 3: Implement `src/detection/template_contracts.py`**

Add a frozen dataclass `DetectionTemplateContract` with fields for `template_id`, `is_compact`, `include_object_ref_end`, `include_box_end`, `row_separator`, `canonical_final_separator`, `required_structural_tokens`, `required_structural_token_ids`, and `prompt_pattern`. Implement `resolve_detection_template_contract`, `render_compact_contract_row`, `required_trainable_token_count`, `required_trainable_token_row_ids`, and `is_compact_template_id`.

The implementation must reject `compact_full` and unknown ids.

- [x] **Step 4: Add closed-token constants and native token ids**

Export from `src/common/detection_compact_rows.py`:

```python
OBJECT_REF_END_TOKEN = "<|object_ref_end|>"
BOX_END_TOKEN = "<|box_end|>"
```

Keep existing `OBJECT_REF_START_TOKEN` and `BOX_START_TOKEN` imports stable.

Export from `src/tokens/qwen_native.py`:

```python
OBJECT_REF_END_TOKEN = "<|object_ref_end|>"
BOX_END_TOKEN = "<|box_end|>"
EXPECTED_OBJECT_REF_END_ID = 151647
EXPECTED_BOX_END_ID = 151649
```

`required_trainable_token_row_ids(template_id)` must use these vetted native ids plus `range(EXPECTED_COORD_START_ID, EXPECTED_COORD_END_ID + 1)`. Tests that have a tokenizer available must also verify these token strings encode to exactly those single ids.

- [x] **Step 5: Run resolver tests**

Run:

```bash
python -m pytest tests/test_detection_template_variants.py -q
```

Expected: all tests pass.

## Task 2: Strict Template Rendering And Parsing

**Files:**
- Modify: `src/detection/template.py`
- Modify: `src/detection/teacher_forcing/compact_full_policy.py`
- Test: `tests/test_detection_compact_full_template.py`
- Test: `tests/test_detection_template_registry.py`
- Test: `tests/test_detection_template_parsing_eval.py`

- [x] **Step 1: Add parametrized render and parse tests**

Extend `tests/test_detection_compact_full_template.py` so existing compact tests run for all semantic compact ids. Assert exact rendered text, strict parse roundtrip, structural token spans, separator spans, and terminal close spans.

Use these exact expected strings for the two-object sample:

```python
EXPECTED_COMPACT = (
    "<|object_ref_start|>traffic light<|box_start|>"
    "<|coord_10|><|coord_20|><|coord_30|><|coord_40|>"
    "<|object_ref_start|>person<|box_start|>"
    "<|coord_100|><|coord_200|><|coord_300|><|coord_400|>"
)
EXPECTED_BOX_CLOSED = (
    "<|object_ref_start|>traffic light<|box_start|>"
    "<|coord_10|><|coord_20|><|coord_30|><|coord_40|><|box_end|>"
    "<|object_ref_start|>person<|box_start|>"
    "<|coord_100|><|coord_200|><|coord_300|><|coord_400|><|box_end|>"
)
EXPECTED_OBJECT_BOX_CLOSED = (
    "<|object_ref_start|>traffic light<|object_ref_end|><|box_start|>"
    "<|coord_10|><|coord_20|><|coord_30|><|coord_40|><|box_end|>"
    "<|object_ref_start|>person<|object_ref_end|><|box_start|>"
    "<|coord_100|><|coord_200|><|coord_300|><|coord_400|><|box_end|>"
)
EXPECTED_OBJECT_BOX_CLOSED_LINES = (
    "<|object_ref_start|>traffic light<|object_ref_end|><|box_start|>"
    "<|coord_10|><|coord_20|><|coord_30|><|coord_40|><|box_end|>\n"
    "<|object_ref_start|>person<|object_ref_end|><|box_start|>"
    "<|coord_100|><|coord_200|><|coord_300|><|coord_400|><|box_end|>\n"
)
```

- [x] **Step 2: Add strict rejection tests**

Add tests asserting:

```python
with pytest.raises(ValueError):
    get_detection_template("compact").parse_assistant(EXPECTED_BOX_CLOSED)

with pytest.raises(ValueError):
    get_detection_template("compact_box_closed").parse_assistant(EXPECTED_COMPACT)

with pytest.raises(ValueError):
    get_detection_template("compact_object_box_closed").parse_assistant(
        EXPECTED_OBJECT_BOX_CLOSED_LINES
    )
```

- [x] **Step 3: Run template tests and confirm failures**

Run:

```bash
python -m pytest tests/test_detection_template_registry.py tests/test_detection_compact_full_template.py tests/test_detection_template_parsing_eval.py -q
```

Expected: failures for unknown semantic ids and missing parser behavior.

- [x] **Step 4: Implement semantic compact template classes**

Refactor `CompactFullTemplate` into a contract-backed compact template implementation. Keep a compatibility class name if local imports still need it, but set the current semantic template id to `compact`.

`get_detection_template` must accept:

```python
"stage1_json_pretty"
"compact"
"compact_box_closed"
"compact_object_box_closed"
"compact_object_box_closed_lines"
```

and must reject `compact_full`.

- [x] **Step 5: Implement template-aware compact parsing**

Update compact parsing helpers so each compact template enforces its exact structure:

- `compact`: no object-ref-end, no box-end, no newline separator.
- `compact_box_closed`: box-end required after four coord tokens.
- `compact_object_box_closed`: object-ref-end required before box-start and box-end required after coords.
- `compact_object_box_closed_lines`: same as object-box-closed plus one final newline per row.

- [x] **Step 6: Run template tests**

Run:

```bash
python -m pytest tests/test_detection_template_registry.py tests/test_detection_compact_full_template.py tests/test_detection_template_parsing_eval.py -q
```

Expected: all selected tests pass.

## Task 3: Config Schema, Runtime, Token Rows, And Active Rename

**Files:**
- Modify: `src/config/schema.py`
- Modify: `src/detection/runtime.py`
- Modify: active compact configs under `configs/stage1/detection_teacher_forcing/`
- Modify: current docs that present `compact_full` as active guidance
- Test: `tests/test_detection_training_config_contract.py`
- Test: `tests/test_detection_training_dataset.py`
- Test: `tests/test_compact_full_encoding_contract.py`

- [x] **Step 1: Add schema tests for accepted and rejected ids**

Extend `tests/test_detection_training_config_contract.py` to assert:

```python
for template_id in [
    "stage1_json_pretty",
    "compact",
    "compact_box_closed",
    "compact_object_box_closed",
    "compact_object_box_closed_lines",
]:
    cfg = make_detection_training_config(template_id=template_id)
    assert cfg.detection_template.id == template_id

with pytest.raises(ValueError, match="compact_full"):
    make_detection_training_config(template_id="compact_full")
```

Use the existing config factory or fixture in that test file rather than adding a second config loader.

- [x] **Step 2: Add exact token-row identity tests**

Add non-skipped tests that load compact configs with token rows enabled and assert the resolved row contract requires the exact template-derived row id set:

- `compact`: coord rows `<|coord_0|>` through `<|coord_999|>` plus `<|object_ref_start|>` and `<|box_start|>`.
- `compact_box_closed`: the `compact` rows plus `<|box_end|>`.
- `compact_object_box_closed` and `compact_object_box_closed_lines`: coord rows plus `<|object_ref_start|>`, `<|object_ref_end|>`, `<|box_start|>`, and `<|box_end|>`.
- `stage1_json_pretty`: no compact structural token-row adaptation requirement.

Also assert missing closure rows, extra rows, duplicate rows, and `compact` bypass attempts fail in training/config resolution, not only in checkpoint validation.

- [x] **Step 3: Add training cache/provenance tests**

Add tests in `tests/test_detection_training_config_contract.py` or another non-skipped training encoding/provenance file asserting:

- encoded-sample cache or packing fingerprints change when only `detection_template.id` changes,
- the materialized training resolved-config/provenance carrier records the resolved `detection_template.id`,
- current tests do not depend on module-skipped `tests/test_detection_training_dataset.py` unless that skip is intentionally removed.

- [x] **Step 4: Run config tests and confirm failures**

Run:

```bash
python -m pytest tests/test_detection_training_config_contract.py tests/test_compact_full_encoding_contract.py -q
```

Expected: failures for unknown ids and old hard-coded 1002 row checks.

- [x] **Step 5: Update typed config schema**

Update `DetectionTemplateConfig.id`, evaluation expected-template validation, token-row validation, and runtime compact-family checks to consume `resolve_detection_template_contract`.

Reject independent compact parse mode, serialization policy, and row separator fields when authored as config source-of-truth.

- [x] **Step 6: Rename active config/doc references and assert comparator semantics**

Rename active current guidance from `compact_full` to `compact`. Do not edit historical docs or archived progress notes unless they are current entrypoints.

The active comparator path should move from:

```text
configs/stage1/detection_teacher_forcing/prod/compact_full_support2.yaml
```

to:

```text
configs/stage1/detection_teacher_forcing/prod/compact_support2.yaml
```

If the old filename must remain temporarily for operator convenience, make it fail with a clear message or remove it from active guidance.

Add or update a config test that parses `configs/stage1/detection_teacher_forcing/prod/compact_support2.yaml` and asserts the comparator contract from OpenSpec:

- `objective.id: teacher_forcing`
- `objective.variant: random_permutation_et_rmp_ce`
- `detection_template.id: compact`
- `data.object_ordering: random_permutation`
- `objective.state_weighting: uniform_permutation`
- `objective.normalization: semantic_image_bucket_balanced`
- trie support and balance weights come from top-level `objective.trie_support_weight` and `objective.trie_balance_weight`

- [ ] **Step 7: Add prefix-rollin compact-family schema tests**

Current implementation note: the strict active `DetectionTrainingConfig` path
rejects the retired `recursive_detection_ce` objective before prefix-rollin
variant validation. Keep this as a Stage-2/rollout boundary check unless a live
prefix-rollin config route is explicitly restored.

Add config/schema tests, not checkpointed rollout behavior tests, asserting `objective.variant: prefix_rollin_et_rmp_ce`:

- accepts `compact`, `compact_box_closed`, `compact_object_box_closed`, and `compact_object_box_closed_lines`,
- rejects `stage1_json_pretty`,
- rejects obsolete flat trie aliases with errors pointing to nested `objective.target` and `objective.boundary`.

- [x] **Step 8: Run config tests**

Run:

```bash
python -m pytest tests/test_detection_training_config_contract.py tests/test_compact_full_encoding_contract.py -q
```

Expected: all selected tests pass.

## Task 4: Teacher-Forcing Token And IR Alignment

**Files:**
- Modify: `src/detection/teacher_forcing/target_builder.py`
- Modify: `src/detection/teacher_forcing/description_tokens.py`
- Modify: `src/detection/tokenization.py` only if role labels need a new structural role
- Test: `tests/test_recursive_detection_ce_target_builder.py`
- Test: `tests/test_compact_et_rmp_span_contract.py`
- Test: `tests/test_detection_template_span_alignment.py`
- Test: `tests/test_token_span_masks_from_templates.py`

- [x] **Step 1: Add per-template teacher-forcing assertions**

Extend existing teacher-forcing tests with a parametrized compact template id. Each case must assert:

- rendered assistant text equals the template renderer output,
- supervised structural token ids include exactly the template-required markers,
- description context includes object-ref-end only for object-box-closed variants,
- box-end labels are supervised for box-closed variants,
- target IR atom count and positions are derived from the same resolved contract as rendered text,
- final newline belongs to row serialization and is separate from `<|im_end|>`.

- [x] **Step 2: Run teacher-forcing tests and confirm failures**

Run:

```bash
python -m pytest tests/test_recursive_detection_ce_target_builder.py tests/test_compact_et_rmp_span_contract.py tests/test_detection_template_span_alignment.py tests/test_token_span_masks_from_templates.py -q
```

Expected: failures where target builder still hard-codes `compact_full`.

- [x] **Step 3: Make target building contract-aware**

Replace hard-coded object branch construction with calls through the selected template contract. Make the target builder accept the resolved template or contract and use it for:

- entry text,
- structural token ids,
- branch role sequences,
- target IR atom positions,
- separator spans,
- terminal and newline handling.

- [x] **Step 4: Propagate template choice from runtime/dataset call sites**

Update the public target-building entrypoints and their callers so the same resolved `detection_template.id` or contract reaches `TeacherForcingTargetBuilder`:

- `build_teacher_forcing_target`,
- `TeacherForcingTargetBuilder.__init__` or `build`,
- detection runtime/dataset preparation call sites that currently assume the old compact branch,
- any fake-tokenizer test helpers that construct compact examples directly.

Do not let the builder default silently to `compact`; callers that build compact target IR must pass or resolve the template explicitly.

- [x] **Step 5: Update description-token context**

Update `tokenize_description_context` so the context is:

- `object_ref_start + desc + box_start` for `compact` and `compact_box_closed`,
- `object_ref_start + desc + object_ref_end + box_start` for object-box-closed variants.

Do not include `box_end` in description context.

- [x] **Step 6: Run teacher-forcing tests**

Run:

```bash
python -m pytest tests/test_recursive_detection_ce_target_builder.py tests/test_compact_et_rmp_span_contract.py tests/test_detection_template_span_alignment.py tests/test_token_span_masks_from_templates.py -q
```

Expected: all selected tests pass.

## Task 5: Offset Adapter And Checkpoint Validation

**Files:**
- Modify: `src/infer/checkpoints.py`
- Modify: `src/config/schema.py`
- Test: `tests/test_infer_checkpoint_resolution.py`
- Test: `tests/test_inject_coord_offsets_script.py`

- [x] **Step 1: Add adapter validation tests**

Extend checkpoint tests to cover exact row sets. Use an existing checkpoint fixture if the file already has one; otherwise add a test-local helper named `_adapter_checkpoint_with_coord_ids(tmp_path, coord_ids) -> ResolvedInferenceCheckpoint` that writes the minimal adapter metadata and tensors, resolves them through `resolve_inference_checkpoint(...)` when needed, and returns the resolved checkpoint object consumed by `validate_compact_coord_token_adapter_contract`.

```python
@pytest.mark.parametrize(
    ("template_id", "expected_count"),
    [
        ("compact", 1002),
        ("compact_box_closed", 1003),
        ("compact_object_box_closed", 1004),
        ("compact_object_box_closed_lines", 1004),
    ],
)
def test_compact_adapter_checkpoint_accepts_exact_template_rows(
    tmp_path: Path,
    template_id: str,
    expected_count: int,
) -> None:
    coord_ids = required_trainable_token_row_ids(template_id)
    checkpoint = _adapter_checkpoint_with_coord_ids(tmp_path, coord_ids)
    validate_compact_coord_token_adapter_contract(
        checkpoint,
        detection_template_id=template_id,
    )
    assert len(coord_ids) == expected_count
```

Also add rejection tests for missing `<|box_end|>`, missing `<|object_ref_end|>`, duplicate ids, extra ids, tensor shape mismatch, and `compact` not bypassing validation.

- [x] **Step 2: Run checkpoint tests and confirm failures**

Run:

```bash
python -m pytest tests/test_infer_checkpoint_resolution.py tests/test_inject_coord_offsets_script.py -q
```

Expected: failures because validation still keys on `detection_sequence_format == "compact_full"` and 1002 rows.

- [x] **Step 3: Implement template-derived row validation**

Update `validate_compact_coord_token_adapter_contract` to accept `detection_template_id`. If an adapter checkpoint has `coord_offset_adapter`, validate:

- exact required row id set,
- no missing ids,
- no extra ids,
- no duplicates,
- coord id count matches embedding offset row count,
- untied head offset row count matches when present,
- `modules_to_save` contains `coord_offset_adapter` when adapter rows are expected.

Keep full or merged checkpoint exemption for tensor row validation, but require template metadata elsewhere.

- [x] **Step 4: Run checkpoint tests**

Run:

```bash
python -m pytest tests/test_infer_checkpoint_resolution.py tests/test_inject_coord_offsets_script.py -q
```

Expected: all selected tests pass.

## Task 6: Prompt, Inference, Shared Runtime, And Artifact Metadata

**Files:**
- Modify: `src/config/prompts.py`
- Modify: `src/infer/pipeline.py`
- Modify: `src/infer/runtime.py`
- Modify: `src/infer/artifacts.py`
- Modify: `src/infer/prompt.py`
- Test: `tests/test_infer_compact_full_policy_contract.py`
- Test: `tests/test_infer_artifact_metadata.py`
- Test: `tests/test_unified_infer_pipeline.py`
- Test: `tests/test_infer_pipeline_shared_decode_request.py`
- Test: `tests/test_run_infer_legacy_shared_runtime.py`

- [x] **Step 1: Add inference YAML rejection tests**

Add tests asserting the canonical inference authoring shape is top-level `detection_template.id`, resolved once and passed into runtime. The following old fields must raise actionable config errors when authored:

- `infer.detection_sequence_format`
- `infer.row_separator`
- `infer.compact_full_parse_mode`
- `infer.parsing.compact_full`
- compact `serialization_policy` or parser-policy fields outside the resolved template contract

Do not accept these as aliases. They may appear only in rejection tests, migration diagnostics, compatibility helpers, or derived metadata after resolution.

- [x] **Step 2: Add prompt and backend parity tests**

Add or update tests asserting:

- prompt row pattern changes with `detection_template.id`,
- prompt hash changes when only `detection_template.id` changes,
- HF, local vLLM, and server-backed vLLM request builders use byte-equivalent final prompt text for the same config,
- the final captured request payload or bridge kwargs contain the same `detection_template_id`, prompt hash, and prompt text bytes,
- closure-token and final-newline instructions are present or absent according to the selected template,
- prompt hash is computed after template-specific prompt text is resolved and before backend-specific request dispatch.

In current tests, the bridge boundary is `run_offline_inference(inference_kwargs=..., generation_kwargs=...)`; extend that capture or instrument the final backend request builders so the test compares actual `messages` or prompt-string payload bytes, not only pre-adapter policy metadata.

- [x] **Step 3: Add artifact metadata tests**

Add or update tests asserting:

- `resolved_config.json` and `summary.json` record `detection_template.id`,
- every `gt_vs_pred.jsonl` row contains `detection_template_id`,
- legacy parser names may appear only as derived diagnostics, not as parser policy.

- [x] **Step 4: Run inference metadata tests and confirm failures**

Run:

```bash
python -m pytest tests/test_infer_compact_full_policy_contract.py tests/test_infer_artifact_metadata.py tests/test_unified_infer_pipeline.py tests/test_infer_pipeline_shared_decode_request.py tests/test_run_infer_legacy_shared_runtime.py -q
```

Expected: failures for old `detection_sequence_format`, row-separator, parse-mode, and missing template metadata assumptions.

- [x] **Step 5: Update prompt API**

Update dense prompt functions to accept `detection_template_id` and derive compact pattern plus line handling from the template contract. Keep `stage1_json_pretty` on the JSON prompt path.

Prompt hash payload must include `detection_template_id`.

- [x] **Step 6: Update inference config loading and runtime bridge**

Resolve template id once from `detection_template.id` during config loading. Remove live runtime fields whose only purpose was independently selecting compact format policy:

- `detection_sequence_format`,
- `row_separator`,
- `compact_full_parse_mode`,
- `parsing.compact_full`.

If internal compatibility names remain during the refactor, keep them private and derive their value from `detection_template.id`; do not let authored config select them.

- [x] **Step 7: Update runtime parsing/materialization**

Resolve template id once during config loading and pass the resolved id through:

- prompt construction,
- backend request preparation,
- generated text parsing,
- materialization into normalized `gt_vs_pred.pred`.

Strict parsing must be selected by the template contract. Standard materialization must not fall back to diagnostic salvage parsing when the selected compact variant fails.

- [x] **Step 8: Update artifact metadata**

Persist the resolved template id through:

- `resolved_config.json`,
- `summary.json`,
- shared runtime parser/provenance metadata,
- prompt fingerprints,
- backend request provenance when available,
- per-row `detection_template_id` in `gt_vs_pred.jsonl`.

Parser ids may remain only as derived metadata.

- [x] **Step 9: Run inference metadata tests**

Run:

```bash
python -m pytest tests/test_infer_compact_full_policy_contract.py tests/test_infer_artifact_metadata.py tests/test_unified_infer_pipeline.py tests/test_infer_pipeline_shared_decode_request.py tests/test_run_infer_legacy_shared_runtime.py -q
```

Expected: all selected tests pass.

## Task 7: Evaluator Post-Hoc mAP Preflight

**Files:**
- Modify: `src/detection/evaluation.py`
- Modify: `src/eval/artifacts.py` only if evaluator artifact loading needs source metadata
- Test: `tests/test_detection_eval_output_parity.py`
- Test: `tests/test_detection_eval_ingestion_diagnostics.py`
- Test: `tests/test_evaluate_detection_provenance.py`
- Test: `tests/test_stage1_detection_eval.py`

- [x] **Step 1: Add evaluator tests**

Add tests asserting:

- post-change artifacts missing `detection_template_id` fail before scoring,
- post-hoc mAP scores normalized `pred` objects without reparsing raw compact text,
- equivalent normalized predictions produce equal mAP across compact variants,
- raw compact text without normalized `pred` objects is not accepted by standard post-hoc mAP.

Put missing-template-id and raw-compact-without-normalized-pred rejections in dependency-light tests such as `tests/test_evaluate_detection_provenance.py` or `tests/test_infer_artifact_metadata.py`. Keep actual equal-mAP metric assertions in pycocotools-backed tests.

- [x] **Step 2: Run evaluator tests and confirm failures**

Run:

```bash
python -m pytest tests/test_evaluate_detection_provenance.py tests/test_infer_artifact_metadata.py -q
python -m pytest tests/test_detection_eval_output_parity.py tests/test_detection_eval_ingestion_diagnostics.py tests/test_stage1_detection_eval.py -q
```

Expected: failures for missing template metadata preflight.

- [x] **Step 3: Implement evaluator preflight**

Update evaluator artifact loading to require template metadata for post-change compact artifacts. Keep metric computation on normalized `gt` and `pred` arrays. Do not add template fields to `metrics.json`, `per_image.json`, `matches.jsonl`, or `per_class.csv`.

- [x] **Step 4: Run evaluator tests**

Run:

```bash
python -m pytest tests/test_evaluate_detection_provenance.py tests/test_infer_artifact_metadata.py -q
python -m pytest tests/test_detection_eval_output_parity.py tests/test_detection_eval_ingestion_diagnostics.py tests/test_stage1_detection_eval.py -q
```

Expected: all selected tests pass.

## Task 8: Legacy Rollout Codec Boundary And Final Verification

**Files:**
- Modify only files needed to keep Stage-2 imports/config resolution working.
- Test: `tests/test_stage2_rollout_import_boundaries.py`
- Test: `tests/test_stage2_rollout_template_policy.py`
- Test: `tests/test_stage2_compact_full_rollout_io.py`

- [x] **Step 1: Run legacy rollout import/config and codec regression guards**

Run:

```bash
python -m pytest tests/test_stage2_rollout_import_boundaries.py tests/test_stage2_rollout_template_policy.py tests/test_stage2_compact_full_rollout_io.py -q
```

Expected: import/config and existing compact codec regression tests pass or fail only on renamed compact ids that need source updates. Do not treat this as behavior validation for new checkpointed rollout variants.

- [x] **Step 2: Fix import/config or legacy codec breakage only**

If these tests fail because code still branches on `compact_full`, update that branch to use the semantic compact family or a private compatibility shim. Do not add new per-template rollout behavior assertions for `compact_box_closed` or object-box variants in this change.

- [x] **Step 3: Run the full targeted suite**

Run:

```bash
python -m pytest \
  tests/test_detection_template_variants.py \
  tests/test_detection_template_registry.py \
  tests/test_detection_compact_full_template.py \
  tests/test_detection_template_parsing_eval.py \
  tests/test_detection_training_config_contract.py \
  tests/test_compact_full_encoding_contract.py \
  tests/test_recursive_detection_ce_target_builder.py \
  tests/test_compact_et_rmp_span_contract.py \
  tests/test_detection_template_span_alignment.py \
  tests/test_token_span_masks_from_templates.py \
  tests/test_infer_checkpoint_resolution.py \
  tests/test_inject_coord_offsets_script.py \
  tests/test_infer_compact_full_policy_contract.py \
  tests/test_infer_artifact_metadata.py \
  tests/test_unified_infer_pipeline.py \
  tests/test_infer_pipeline_shared_decode_request.py \
  tests/test_run_infer_legacy_shared_runtime.py \
  tests/test_detection_eval_output_parity.py \
  tests/test_detection_eval_ingestion_diagnostics.py \
  tests/test_evaluate_detection_provenance.py \
  tests/test_stage1_detection_eval.py \
  tests/test_stage2_rollout_import_boundaries.py \
  tests/test_stage2_rollout_template_policy.py \
  tests/test_stage2_compact_full_rollout_io.py \
  -q
```

Expected: all targeted tests pass.

- [x] **Step 4: Validate OpenSpec and whitespace**

Run:

```bash
openspec validate detection-template-variants --type change --strict
openspec validate --changes --strict
git diff --check
```

Expected: all validation commands pass.

- [x] **Step 5: Review active docs/config references**

Run:

```bash
rg -n "compact_full|detection_sequence_format|compact_full_parse_mode|parsing\\.compact_full|row_separator|serialization_policy" \
  configs/stage1/detection_teacher_forcing \
  configs/infer \
  docs/AGENT_INDEX.md \
  docs/catalog.yaml \
  docs/IMPLEMENTATION_MAP.md \
  docs/PROJECT_CONTEXT.md \
  docs/SYSTEM_OVERVIEW.md \
  src/detection \
  src/common \
  src/config \
  src/infer \
  openspec/changes/detection-template-variants

rg -n "compact_full|detection_sequence_format|compact_full_parse_mode|parsing\\.compact_full|row_separator|serialization_policy" \
  tests \
  scripts
```

Expected: any remaining active `compact_full` or old compact-knob references are compatibility code with explicit migration errors, test fixtures asserting rejection, derived metadata, explicit legacy codec fixtures, or current OpenSpec/roadmap text documenting the migration. `configs/archive/`, `docs/history/`, `progress/`, and `openspec/archive/` are historical unless a current entrypoint imports them.

## Self-Review Checklist

- [x] Every OpenSpec requirement in `openspec/changes/detection-template-variants/specs/detection-template-variants/spec.md` maps to Tasks 1, 2, 3, 5, 6, or 7.
- [x] `stage1-detection-objectives` deltas map to Tasks 3, 4, and 8.
- [x] `coord_offset` deltas map to Task 5.
- [x] `dataset-prompt-variants`, `inference-engine`, and `shared-inference-runtime` deltas map to Task 6.
- [x] `detection-evaluator` deltas map to Task 7.
- [x] No implementation starts before this plan and OpenSpec docs converge through subagent review.
- [x] Implementation happens on `/data/CoordExp` main, then verified changes are synced only to the prefix-denoising worktree/branch after main is complete.
