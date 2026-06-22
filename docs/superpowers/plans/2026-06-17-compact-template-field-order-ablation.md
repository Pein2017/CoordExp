# Compact Template Field-Order Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement two-knob compact detection templates so bbox-first and desc-first rich compact rows can be trained and smoke-tested through standard Stage-1 SFT.

**Architecture:** Keep compact template family and structural rows in the semantic template registry, and pass `custom.object_field_order` alongside the template contract anywhere compact assistant bytes, parser policy, prompt hashes, cache fingerprints, or artifacts are produced. The renderer and parser should be strict for the configured pair, while mAP remains over normalized prediction objects.

**Tech Stack:** Python, pytest, OpenSpec, Qwen3-VL special tokens, CoordExp Stage-1 SFT data builders, static packing cache, inference/eval artifact contracts.

---

## Source Of Truth

- OpenSpec change: `openspec/changes/compact-template-field-order-ablation/`
- Design note: `docs/superpowers/specs/2026-06-17-compact-template-field-order-ablation-design.md`
- Current compact registry: `src/detection/template_contracts.py`
- Strict template owner: `src/detection/template.py`
- Compatibility facade: `src/common/detection_sequence.py`
- Object field order helper: `src/common/object_field_order.py`
- Prompt resolver: `src/config/prompts.py`
- Config schema/loader: `src/config/schema.py`, `src/config/loader.py`
- Standard SFT data path: `src/datasets/builders/jsonlines.py`, `src/datasets/dense_caption.py`, `src/sft.py`
- Packing/cache: `src/detection/packing.py`, `src/datasets/encoded_sample_cache.py`, `src/datasets/wrappers/packed_caption.py`
- Inference/eval provenance: `src/infer/`, `src/eval/`, `src/detection/evaluation.py`

## Review Convergence

- Mode: docs/spec/plan.
- Allowed mutation: OpenSpec and roadmap documentation only; no code/config
  implementation and no smoke launches before user approval.
- Review execution: local independent lanes. Subagents were not launched because
  this tool environment requires an explicit delegation request for subagents.
- OpenSpec governance lane:
  - Finding P1: older `detection-template-variants` docs still said
    `detection_template.id` was the only authored serialization source.
  - Resolution: added supersession notes and rewrote the strict sentence so
    template id owns wrapper/closure/separator/parser/token-row policy while
    `custom.object_field_order` owns desc-first versus geometry-first row layout
    in this amendment.
- Implementation-surface lane:
  - Finding P1: current compact registry lacks `compact_object_closed`, and
    current compact render/parse paths are desc-first.
  - Resolution: OpenSpec and tasks require `compact_object_closed`, strict
    template id x field-order render/parse tests, span updates, and standard
    SFT data-path plumbing.
- Reproducibility/eval lane:
  - Finding P1: desc-first and bbox-first runs can contaminate each other if
    cache fingerprints or artifact metadata omit field order.
  - Resolution: specs/tasks require encoded-sample cache, static packing,
    prompt hash, inference artifacts, shared runtime provenance, and evaluator
    preflight to include both axes.
- Convergence state: no remaining P0/P1 findings in the planning artifacts;
  implementation remains gated on explicit user approval.

## Task 1: Contract Tests First

**Files:**
- Modify: `tests/test_detection_compact_full_template.py`
- Modify: `tests/test_detection_template_variants.py` if present, otherwise create it
- Modify: `tests/test_prompt_variants.py`

- [ ] **Step 1: Add render oracle tests for both field orders**

Add parametrized expectations for `compact_object_box_closed`:

```python
EXPECTED_DESC_FIRST_RICH = (
    "<|object_ref_start|>cat<|object_ref_end|>"
    "<|box_start|><|coord_1|><|coord_2|><|coord_3|><|coord_4|><|box_end|>"
)
EXPECTED_GEOMETRY_FIRST_RICH = (
    "<|box_start|><|coord_1|><|coord_2|><|coord_3|><|coord_4|><|box_end|>"
    "<|object_ref_start|>cat<|object_ref_end|>"
)
```

Run:

```bash
python -m pytest tests/test_detection_compact_full_template.py -q
```

Expected before implementation: tests for `geometry_first` and
`compact_object_closed` fail because the renderer/parser is desc-first only and
the object-only closure id is not registered.

- [ ] **Step 2: Add prompt hash tests**

Add a test asserting prompt hashes differ when only `object_field_order` changes
under `detection_template_id="compact_object_box_closed"`.

Run:

```bash
python -m pytest tests/test_prompt_variants.py -q
```

Expected before implementation: the new compact prompt expectation fails if the
prompt example still ignores compact field order.

## Task 2: Template Registry and Strict Parser

**Files:**
- Modify: `src/detection/template_contracts.py`
- Modify: `src/detection/template.py`
- Modify: `src/common/detection_sequence.py`

- [ ] **Step 1: Add `compact_object_closed` to the registry**

Register the id with object-ref-end included, box-end excluded, no newline, and
1003 trainable rows.

- [ ] **Step 2: Add field-order-aware compact segment rendering**

Make the compact row renderer accept `object_field_order` and build these
segments:

```python
object_segment = OBJECT_REF_START_TOKEN + desc + optional_object_ref_end
box_segment = BOX_START_TOKEN + coords + optional_box_end
```

Return `object_segment + box_segment` for `desc_first` and
`box_segment + object_segment` for `geometry_first`, then append the contract
final separator.

- [ ] **Step 3: Update strict parsing**

Parse the configured segment order and reject the opposite order.  Keep
`compact_object_box_closed_lines` newline handling template-specific.

Run:

```bash
python -m pytest tests/test_detection_compact_full_template.py tests/test_detection_template_variants.py -q
```

Expected after implementation: all render/parse and row-count tests pass.

## Task 3: Standard SFT Data Path

**Files:**
- Modify: `src/config/schema.py`
- Modify: `src/config/loader.py`
- Modify: `src/datasets/builders/jsonlines.py`
- Modify: `src/datasets/dense_caption.py`
- Modify: `src/sft.py`
- Modify: `tests/test_dense_caption_prompt_override.py`
- Modify: `tests/test_stage1_static_packing_runtime_config.py`

- [ ] **Step 1: Route `custom.object_field_order` into compact rendering**

Keep `custom.object_field_order` required and normalized.  Pass it into compact
assistant rendering whenever `custom.detection_template_id` selects a compact
template.

- [ ] **Step 2: Prove standard SFT is not rollout-only**

Add a config/data-builder test that builds a standard SFT conversation with
`custom.detection_template_id: compact_object_box_closed` and
`custom.object_field_order: geometry_first`, then asserts the assistant text is
the rich bbox-first row.

Run:

```bash
python -m pytest tests/test_dense_caption_prompt_override.py tests/test_stage1_static_packing_runtime_config.py -q
```

Expected after implementation: geometry-first compact SFT target text passes and
no `TeacherForcingRollin` dependency is required.

## Task 4: Cache and Packing Fingerprints

**Files:**
- Modify: `src/detection/packing.py`
- Modify: `src/sft.py`
- Modify: `tests/test_packing_cache_fingerprints.py`
- Modify: `tests/test_stage1_static_packing_runtime_config.py`

- [ ] **Step 1: Include both axes in fingerprints**

Ensure encoded-sample and static-packing fingerprints include:

```text
detection_template_id
object_field_order
object_ordering
prompt_hash
global_max_length / packing_length
tokenizer or chat template identity
```

- [ ] **Step 2: Add fingerprint tests**

Assert fingerprints change when only `object_field_order` changes, when only
template id changes, and when only packing length changes.

Run:

```bash
python -m pytest tests/test_packing_cache_fingerprints.py tests/test_stage1_static_packing_runtime_config.py -q
```

Expected after implementation: all cache identity tests pass.

## Task 5: Inference, Eval, and Docs

**Files:**
- Modify: relevant `src/infer/*.py` and `src/eval/*.py` provenance/materialization files found by `rg -n "detection_template_id|object_field_order|gt_vs_pred" src/infer src/eval src/detection`
- Modify: `tests/test_infer_artifact_metadata.py`
- Modify: `tests/test_parser_policy_parity.py`
- Modify: `docs/data/PACKING.md`
- Modify: `docs/training/STAGE1_OBJECTIVE.md`

- [ ] **Step 1: Persist both axes**

Record `detection_template_id` and `object_field_order` in compact inference
resolved config, summary, parser policy, and `gt_vs_pred.jsonl` rows.

- [ ] **Step 2: Keep mAP on normalized predictions**

Add evaluator preflight tests for missing `object_field_order` and missing
`detection_template_id`; do not reparse raw compact text in standard mAP.

Run:

```bash
python -m pytest tests/test_infer_artifact_metadata.py tests/test_parser_policy_parity.py -q
```

Expected after implementation: provenance and parser-policy tests pass.

## Task 6: Ablation Configs and Smoke Runs

**Files:**
- Create: `configs/stage1/profiles/2b/pure_ce_coco80_geometry_first_1024_object_ref_close_box_close_sorted_packed_natural_adjacent.yaml`
- Create: `configs/stage1/smoke/pure_ce_coco80_geometry_first_1024_object_ref_close_box_close_sorted_packed_natural_adjacent_tiny.yaml`
- Modify existing desc-first sibling configs if implementation changes the final naming or inherited facets
- Modify: `tests/test_stage1_static_packing_runtime_config.py`

- [ ] **Step 1: Add bbox-first config pair**

Mirror the desc-first natural-adjacent config and set:

```yaml
custom:
  object_field_order: geometry_first
  object_ordering: sorted
  detection_template_id: compact_object_box_closed
global_max_length: 12000
model:
  model: model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent
```

- [ ] **Step 2: Config-load verification**

Run:

```bash
python -m pytest tests/test_stage1_static_packing_runtime_config.py -q
```

Expected after implementation: desc-first and bbox-first production/smoke leaves
resolve to the same checkpoint, sorted ordering, `global_max_length: 12000`,
static packing, and LLM-only trainability, with intentional field-order/run-path
differences.

- [ ] **Step 3: Smoke launches after approval**

Run the two tiny smoke configs only after user approval.  Capture log paths,
output dirs, cache dirs, parse/drop counters, and final train/eval loss lines.

## Task 7: Final Verification

- [ ] **Step 1: OpenSpec validation**

```bash
openspec validate compact-template-field-order-ablation --type change --strict
```

- [ ] **Step 2: Diff whitespace check**

```bash
git diff --check
```

- [ ] **Step 3: Report gate**

Report changed files, tests run, skipped expensive checks, smoke artifact roots,
and residual production-launch risks.  Stop before production training unless
the user explicitly approves it.
