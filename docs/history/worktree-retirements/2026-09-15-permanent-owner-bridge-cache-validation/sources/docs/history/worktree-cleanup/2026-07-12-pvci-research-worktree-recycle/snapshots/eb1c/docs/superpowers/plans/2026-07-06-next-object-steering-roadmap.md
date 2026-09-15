# Next-Object Steering Implementation Roadmap

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this roadmap wave-by-wave. Each wave has its own tests, review gate, and commit boundary. Do not start source implementation until the user explicitly approves implementation.

**Goal:** Build the approved `next_object_steering` path for the painted-GT research branch: matched `text_image` and `image_only` training materialization, rollout, metrics, reports, configs, and tiny-gate execution.

**Architecture:** Reuse the existing painted-GT substrate and add the narrow next-object surfaces around materialization, label masking, prompt identity, config dispatch, rollout control, and gate reports. Keep ordinary one-shot inference and the historical paint-all / teacher-prefix routes stable unless a shared helper must be extended backward-compatibly.

**Tech Stack:** Python 3.12, PyTorch, Transformers Qwen3-VL, PEFT DoRA, Pillow, pytest, OpenSpec, existing CoordExp-Swift `src/painted_gt`, `src/templates`, `src/inference`, `src/packing`, `src/training`, `src/eval`, and `configs/coordexp_swift`.

## Global Constraints

- Effective object-step labels MUST be exactly `row(X) + <|object_ref_start|>`.
- Effective terminal-step labels MUST be exactly `<|im_end|>`.
- Object-step structural chat-template `<|im_end|>\n` MUST be ignored in labels.
- No third target form is allowed.
- V1 schedule is `geo_sorted`.
- V1 steering policies are `text_image` and `image_only`.
- V1 prompt identity is `next_unmarked_object`.
- V1 painter is `outline_center_flat_v1`: magenta outline, yellow center, no fill, no opacity accumulation, no overlap-depth visual encoding.
- Inference commits the first valid row only, strips the trailing continuation sentinel from committed rows, and does not suppress duplicates.
- First-token `<|im_end|>` is the only semantic terminal-stop authority.
- Backend generation stop is `generation_stop_reason`; semantic controller result is `controller_outcome`.
- Cross-step KV cache reuse is forbidden because the image changes after each commit.
- Default next-object rollout cap is GT-independent `max_rollout_steps=64`.
- Standard one-shot autoregressive detection remains a reference baseline, not the central ablation.

---

## Source Of Truth

Read these before implementing and treat them in this order:

1. `openspec/changes/add-painted-gt-transcription-probe/specs/coordexp-swift-painted-gt-materialization/spec.md`
2. `openspec/changes/add-painted-gt-transcription-probe/specs/coordexp-swift-painted-gt-decode-eval/spec.md`
3. `openspec/changes/add-painted-gt-transcription-probe/specs/coordexp-swift-painted-gt-launch-gates/spec.md`
4. `docs/superpowers/specs/2026-07-06-next-object-steering-design.md`
5. This roadmap.
6. `docs/superpowers/plans/2026-07-06-next-object-steering-plan.md`
7. `research/ideas/qwen3-vl-painted-gt-transcription-probe/overview.md`

When a source file or artifact disagrees with the OpenSpec, stop and repair the
plan/spec or ask the user before implementing behavior drift.

## Existing Owner Map

CodeGraph/read evidence before this roadmap identified the current owner seams:

- `src/painted_gt/materialization.py`
  - Current owners: `build_slice_manifest`, `build_painted_plan`,
    `materialize_painted_image`, painted-condition writers, geometry audit,
    schedule/materialized examples.
  - Add next-object materialization here first.
  - Extract `src/painted_gt/painting.py` only if the shared painter starts to
    make `materialization.py` harder to read; that extraction must be a separate
    small commit with parity tests.
- `src/templates/renderer.py`
  - Current owner for compact row rendering, assistant suffix, spans, and
    template fingerprint.
  - Extend through small helpers instead of duplicating compact-row rendering in
    `src/painted_gt`.
- `src/inference/prompt.py`
  - Current owner for `PromptRecord` and assistant-prefix continuation. The
    empty-prefix `image_only` case must use this route and keep one open
    assistant turn.
- `src/inference/pipeline.py` and `src/config/inference.py`
  - Current config-driven inference entry and schema. Add steering dispatch here
    only after the controller is test-backed.
- `src/inference/artifacts.py`
  - Current owner for inference artifact writing. Add next-object step-trace
    persistence here so rollout rows, evaluator rows, and report inputs share
    one artifact contract.
- `src/painted_gt/decode.py`
  - Current painted decode/controller owner. Add the next-object controller here
    unless implementation proves a split is clearer.
- `src/painted_gt/reports.py` and `src/painted_gt/metrics.py`
  - Current report/metric owners for painted-GT gates.
- `src/training/pack_cache.py`
  - Generic cache owner. It must consume opaque materialization identity and must
    not import `src.painted_gt`.

## Commit And Review Policy

- Commit after each wave that passes its local verification.
- Stage only files owned by the wave. Preserve unrelated dirty work.
- Run `git diff --cached --check` before every commit.
- Run `openspec validate add-painted-gt-transcription-probe --strict` after any
  contract, config, materialization, decode, metric, or report wave.
- Run review-convergence after Wave 2, Wave 5, and Wave 7 before launching any
  training or rollout evidence claim.
- Stop and ask the user before:
  - changing the two-form target grammar;
  - using GT count/order/boxes to stop rollout;
  - adding syntax-constrained decoding to primary evidence;
  - changing the painter style;
  - changing the baseline adapter/source identity;
  - launching a larger-than-tiny training run;
  - deleting or rewriting existing experiment artifacts.

## Wave 0: Pre-Implementation Lock

**Purpose:** Ensure implementation starts from a known state and does not absorb
unrelated dirty files.

**Files:**
- Read: `git status --short --branch`
- Read: `openspec/changes/add-painted-gt-transcription-probe/review-triage-next-object-steering.md`
- No source modifications.

**Steps:**

- [ ] Record current branch and dirty files.

```bash
git status --short --branch
git log --oneline -5
```

- [ ] Confirm the two planning commits are present.

Expected latest planning commits:

```text
f2775299 docs: converge next object steering review
c7fe0839 docs: define next object steering contract
```

- [ ] Run the contract parser before code edits.

```bash
openspec validate add-painted-gt-transcription-probe --strict
```

Expected: `Change 'add-painted-gt-transcription-probe' is valid`.

**Exit:** proceed only when the branch, OpenSpec, and dirty-file boundary are
understood. Do not stage unrelated pre-existing source/config/probe changes.

## Wave 1: Target Grammar And Label Contract Tests

**Purpose:** Make the two legal supervised target forms fail first.

**Files:**
- Create or modify: `tests/painted_gt/test_next_object_steering_materialization.py`
- Modify if needed: `tests/painted_gt/test_materialization.py`
- Modify if needed: `tests/templates/test_renderer.py`
- Modify if needed: `tests/painted_gt/test_label_dump.py`
- No production code until failing tests are written and run.

**Interfaces To Lock:**
- `step_target_kind: "object_continuation" | "terminal_stop"`
- `steering_context_policy: "text_image" | "image_only"`
- Non-ignored object labels: `[row_tokens..., object_ref_start_id]`
- Non-ignored terminal labels: `[im_end_id]`

**Steps:**

- [ ] Add failing tests for object-continuation labels.

Test intent:

```python
def test_next_object_object_continuation_supervises_row_plus_start_token():
    result = build_two_object_next_object_fixture(
        step_target_kind="object_continuation",
        steering_context_policy="text_image",
    )
    assert result.non_ignored_token_texts == [
        "<|object_ref_start|>",
        "person",
        "<|object_ref_end|>",
        "<|box_start|>",
        "<|coord_100|>",
        "<|coord_200|>",
        "<|coord_300|>",
        "<|coord_400|>",
        "<|box_end|>",
        "<|object_ref_start|>",
    ]
```

- [ ] Add failing tests for terminal labels.

Test intent:

```python
def test_next_object_terminal_supervises_only_im_end():
    result = build_two_object_next_object_fixture(
        step_target_kind="terminal_stop",
        steering_context_policy="image_only",
    )
    assert result.non_ignored_token_texts == ["<|im_end|>"]
```

- [ ] Add rejection tests for the forbidden effective targets.

Forbidden forms:

```text
row(X)
row(X) + <|im_end|>
row(X) + <|object_ref_start|> + <|im_end|>
multiple complete rows
empty object target
```

- [ ] Run the failing test slice.

```bash
pytest tests/painted_gt/test_next_object_steering_materialization.py -q
```

Expected before implementation: tests fail because next-object materialization
or validation does not exist yet.

**Exit:** failing tests demonstrate the exact target grammar and no production
source has been edited yet.

## Wave 2: Materialization, Label Dumps, And Parity

**Purpose:** Implement next-object materialized examples and prove labels survive
through rendering, encoding, packing, and micro-step assembly.

**Files:**
- Modify: `src/painted_gt/materialization.py`
- Modify: `src/inference/prompt.py`
- Modify if needed: `src/templates/renderer.py`
- Modify if needed: `src/qwen/encoding.py`
- Modify if needed: `src/packing/supervision.py`
- Modify if needed: `src/supervision/tokens.py`
- Modify if needed: `src/training/pack_cache.py`
- Test: `tests/painted_gt/test_next_object_steering_materialization.py`
- Test: `tests/painted_gt/test_label_dump.py`
- Test: `tests/inference/test_prompt_image.py`
- Test: `tests/training/test_pack_cache.py`
- Test: `tests/training/test_pipeline_assembly.py`

**Interfaces To Produce:**
- `NEXT_UNMARKED_OBJECT_PROMPT_ID = "next_unmarked_object"`
- `build_next_unmarked_object_prompt_identity(prompt_text: str) -> dict[str, str]`
- `materialize_next_object_steering_condition(...)`
- `validate_next_object_effective_labels(...)`
- `build_painted_plan(..., allow_empty_painted_ids=True)` or a narrower
  `build_unpainted_next_object_plan(...)` for the first next-object step only.
- `materialization_identity` containing mode, policy, prompt id/fingerprint,
  target grammar id, painter id, schedule id, source JSONL SHA256, and example
  payload SHA256.
- `parity_summary.json` proving `text_image` and `image_only` row-set parity.

**Steps:**

- [ ] Implement the shared prompt identity before finalizing materialization
  identity.

Prompt identity rule:

```text
prompt_id = "next_unmarked_object"
prompt_fingerprint = sha256 of the exact prompt text/prompt payload
one artifact payload is shared by materialization and rollout
```

- [ ] Implement next-object materialization for object steps.

Object step semantics:

```text
image_t = source image painted with GT objects before t
text_image prefix = GT rows before t
image_only prefix = empty string
target = row(current GT object) + <|object_ref_start|>
```

- [ ] Add the first-step unpainted visual state.

Required behavior:

```text
step 0 image has no painted pixels
step 1 image paints only object 0 under the schedule
terminal image paints all scheduled objects
```

The existing paint-all and teacher-prefix callers must keep rejecting empty
painted-object plans unless they explicitly request the next-object first-step
path.

- [ ] Implement terminal materialization.

Terminal semantics:

```text
image_T = source image painted with all GT objects
text_image prefix = all GT rows
image_only prefix = empty string
target = <|im_end|>
```

- [ ] Emit label-dump artifacts for sample rows.

Each dump must include:

```text
example id
policy
step_target_kind
input token texts
label token texts
ignored prefix labels
ignored structural chat-close labels
non-ignored target labels
painted_plan_id
prompt id and fingerprint
visual audit status
geometry audit status
Qwen no-resize image-plan preflight status
```

- [ ] Emit parity summary before packing.

The summary must compare:

```text
(image_id, step_index, target_object_id, terminal)
source image ids
schedule id
prompt identity
painter identity
terminal-step counts
dropped-example counters
target-span truncation counters
non-ignored label counts
token-length distributions
```

- [ ] Emit pre-GPU materialization receipts before packing.

Required files:

```text
slice_manifest.json
painted_plan.jsonl
schedule_manifest.<schedule_id>.json
condition_manifest.json
visual_audit/index.json
geometry_audit_summary.json
qwen_no_resize_preflight.json
materialization_identity.json
```

Training launch remains blocked if the visual audit, geometry audit, or Qwen
no-resize image-plan preflight is missing or failing for either steering policy.

- [ ] Implement generic materialization identity for cache determinants.

Pack-cache rule:

```text
src/training/pack_cache.py consumes materialization_identity as an opaque value.
It must not import src.painted_gt.
It must not parse slice_manifest.json, painted_plan.jsonl, or schedule manifests.
```

- [ ] Run the materialization and packing checks.

```bash
pytest tests/painted_gt/test_next_object_steering_materialization.py tests/painted_gt/test_label_dump.py tests/inference/test_prompt_image.py -q
pytest tests/training/test_pack_cache.py tests/training/test_pipeline_assembly.py -q
openspec validate add-painted-gt-transcription-probe --strict
git diff --check
```

**Exit:** both policies materialize with exact labels, parity evidence exists,
and cache identity changes only for semantic determinants.

**Review Gate:** run a review-convergence loop focused on target grammar,
label masks, materialization identity, and cache boundaries before proceeding.

## Wave 3: Prompt Identity And Context Policy Plumbing

**Purpose:** Make training and inference share the same `next_unmarked_object`
prompt and make `image_only` a true empty-prefix continuation.

**Files:**
- Modify: `src/inference/prompt.py`
- Modify if needed: `src/templates/renderer.py`
- Modify: `src/painted_gt/materialization.py`
- Test: `tests/inference/test_prompt_image.py`
- Test: `tests/painted_gt/test_next_object_steering_materialization.py`

**Interfaces To Produce:**
- Prompt id: `next_unmarked_object`
- Prompt fingerprint recorded in materialization and rollout artifacts.
- `PromptRecord` artifact includes prompt id/fingerprint if current artifact
  shape lacks it.

**Steps:**

- [ ] Add prompt identity tests.

Required prompt meaning:

```text
predict exactly one next visible object instance
marked magenta rectangles and yellow center points are already committed
do not emit marked objects again
overlapping or partially occluded uncommitted objects should still be emitted
return <|im_end|> only when every visible object instance is already committed
```

- [ ] Add empty-prefix continuation test for `image_only`.

Assertions:

```text
assistant_prefix_text == ""
exactly one final assistant turn exists
final assistant turn is open for generation
rendered chat does not close the assistant with <|im_end|>\n before generation
prompt tokenization uses continue_final_message=True or equivalent
```

- [ ] Route materialization and rollout prompt construction through the same
  prompt owner.

- [ ] Run prompt tests.

```bash
pytest tests/inference/test_prompt_image.py tests/painted_gt/test_next_object_steering_materialization.py -q
```

**Exit:** both policy prompts are byte/fingerprint compatible between
materialization and rollout.

## Wave 4: Shared Painter And Overlap Diagnostics

**Purpose:** Keep train/infer visual state transitions identical and record
overlap features without adding overlap-depth visual coding.

**Files:**
- Modify: `src/painted_gt/materialization.py`
- Create only if justified: `src/painted_gt/painting.py`
- Modify: `src/painted_gt/decode.py`
- Test: `tests/painted_gt/test_materialization.py`
- Test: `tests/painted_gt/test_decode_controllers.py`

**Interfaces To Produce:**
- Painter identity: `outline_center_flat_v1`
- Provenance fields: `style_id`, `paint_style`, `uses_fill=false`,
  `uses_overlap_depth_encoding=false`.
- Geometry overlap fields for targets and predictions.

**Steps:**

- [ ] Decide painter location.

Default: keep the current painter in `src/painted_gt/materialization.py`.
Extract `src/painted_gt/painting.py` only if the same function must be reused by
decode and materialization and the extraction is tested in isolation.

- [ ] Add parity tests proving training and rollout use the same painter style.

- [ ] Add target overlap diagnostics.

Required fields:

```text
target_overlaps_committed_region
target_center_inside_committed_region
target_committed_overlap_area_ratio
```

- [ ] Add prediction overlap diagnostics.

Required fields:

```text
prediction_overlaps_committed_region
prediction_center_inside_committed_region
prediction_committed_overlap_area_ratio
```

- [ ] Run painter tests.

```bash
pytest tests/painted_gt/test_materialization.py tests/painted_gt/test_decode_controllers.py -q
```

**Exit:** rendered-pixel overlap is optional audit evidence, but geometry-based
overlap fields are present and style identity is shared.

## Wave 5: Next-Object Rollout Controller

**Purpose:** Implement autonomous predicted-box feedback rollout without GT
stop leakage.

**Files:**
- Modify: `src/painted_gt/decode.py`
- Modify: `src/inference/artifacts.py`
- Modify: `src/inference/backend.py` only if current `DecodeResult` cannot
  expose raw token trace / backend stop facts needed by the controller.
- Test: `tests/painted_gt/test_decode_controllers.py`
- Test: `tests/inference/test_artifacts.py`
- Test if backend touched: `tests/inference/test_backend_trace.py`

**Interfaces To Produce:**
- `run_next_object_steering_controller(...)`
- `painted_gt_step_trace.jsonl` written by `src/inference/artifacts.py`.
- Links from committed prediction rows and evaluator-input rows back to
  `step_trace_id`.
- Step artifact fields:
  - `raw_generated_text`
  - `raw_generated_token_ids = DecodeResult.generated_token_ids`
  - `generation_stop_reason = DecodeResult.stop_reason`
  - `controller_outcome`
  - `parsed_candidates`
  - `dropped_malformed_candidates`
  - `committed_row`
  - `ignored_extra_rows`
  - `paint_source`
  - `prefix_source`
  - `image_transition`
  - `max_rollout_steps`

**Steps:**

- [ ] Add fake-backend tests for raw-token-first classification.

Required cases:

```text
first-token <|im_end|> -> controller_outcome=terminal_stop
row + <|object_ref_start|> -> commit row only
row + <|im_end|> -> generation_stop_reason=im_end, controller_outcome=post_row_im_end_violation, commit row, continue
malformed + valid row -> drop malformed and commit valid row
no valid row twice -> no_progress_limit_stop
duplicate row -> commit, paint, and flag duplicate without suppression
two valid rows -> commit first and record multi_row_violation
max_rollout_steps=64 -> step_cap_stop without reading GT count
```

- [ ] Implement controller state transitions.

State transition rule:

```text
state_0 = source image, empty committed predictions
after committed row: paint predicted bbox into next image state
text_image prefix: append canonical committed prediction row only
image_only prefix: remain empty
malformed no-progress: image and prefix unchanged
```

- [ ] Preserve raw traces and parser text separately.

Parser strip rule:

```text
strip exactly one terminal <|im_end|> for parser text
do not strip inner or repeated stop tokens
do not let stripping hide post-row <|im_end|> violations
```

- [ ] Persist step traces through the inference artifact writer.

Artifact rule:

```text
src/painted_gt/decode.py returns step trace rows
src/inference/artifacts.py writes painted_gt_step_trace.jsonl
prediction rows reference step_trace_id
evaluator input rows reference step_trace_id when available
```

- [ ] Add a GT-isolation test for controller execution.

Test rule:

```text
fake GT accessors raise during run_next_object_steering_controller(...)
GT boxes/categories are passed only to scorer/report code after rollout
```

- [ ] Run controller tests.

```bash
pytest tests/painted_gt/test_decode_controllers.py tests/inference/test_artifacts.py -q
```

**Exit:** rollout can be tested entirely with fake backend outputs before any
GPU inference.

**Review Gate:** run a review-convergence loop focused on rollout semantics,
stop behavior, raw trace preservation, duplicate handling, and GT leakage.

## Wave 6: Config Dispatch And Primary Decode Identity

**Purpose:** Make next-object steering reachable from normal inference configs
without disturbing one-shot inference.

**Files:**
- Modify: `src/config/inference.py`
- Modify: `src/inference/pipeline.py`
- Modify if needed: `src/inference/artifacts.py`
- Add configs under: `configs/coordexp_swift/infer/painted_gt/next_object_steering/`
- Test: `tests/inference/test_pipeline.py`
- Test: `tests/inference/test_data_parallel_runtime.py` if dispatch affects data parallel paths.
- Test: `tests/painted_gt/test_decode_controllers.py`

**Interfaces To Produce:**
- `decode.controller: one_shot | next_object_steering`
- `decode.steering_context_policy: null | text_image | image_only`
- `decode.max_rollout_steps: null | positive int`
- Decode identity includes `generation.max_new_tokens=384`,
  `decode.max_rollout_steps=64`,
  `generation_stop_reason` convention, parser version, prompt id/fingerprint,
  painter id, and scoreability policy.

**Steps:**

- [ ] Add schema tests for valid and invalid steering configs.

Valid:

```yaml
decode:
  surface: free_raw_generation
  controller: next_object_steering
  steering_context_policy: text_image
  max_rollout_steps: 64
generation:
  temperature: 0
  top_p: 1.0
  repetition_penalty: 1.10
  max_new_tokens: 384
```

Invalid:

```text
decode.controller=next_object_steering with steering_context_policy omitted
steering_context_policy unknown
max_rollout_steps <= 0
detect-all prompt selected for primary steering
```

Ordinary one-shot configs must keep:

```yaml
decode:
  controller: one_shot
  steering_context_policy: null
  max_rollout_steps: null
```

- [ ] Add dispatch tests.

Assertions:

```text
next_object_steering config routes to run_next_object_steering_controller
standard one-shot config still routes to the existing inference path
data-parallel inference does not change controller semantics
```

- [ ] Add primary decode identity checks.

Reject or mark ineffective config-only knobs that are not passed to the backend.

- [ ] Run dispatch tests.

```bash
pytest tests/inference/test_pipeline.py tests/painted_gt/test_decode_controllers.py -q
openspec validate add-painted-gt-transcription-probe --strict
git diff --check
```

**Exit:** configs can launch the controller, and old inference configs do not
route through it.

## Wave 7: Metrics, Reports, And Launch Gates

**Purpose:** Make evidence interpretable before any training run is trusted.

**Files:**
- Modify: `src/painted_gt/metrics.py`
- Modify: `src/painted_gt/reports.py`
- Modify if needed: `src/eval/*`
- Test: `tests/painted_gt/test_reports.py`
- Test: existing eval tests if metric owners change.

**Interfaces To Produce:**
- `PRIMARY_METRICS_BY_MODE` or equivalent explicit registry.
- Gate reports for `next_object_steering.text_image` and
  `next_object_steering.image_only`.
- Report fields for terminal, duplicate, overlap, invalid-output, scoreability,
  and parity diagnostics.
- Detection metrics (`mAP`, `mRecall`, `precision`, `recall`, `F1`,
  `debug_detection_f1`) remain owned by the evaluator/evaluator-consumer path.
- Steering diagnostics remain owned by `src/painted_gt/metrics.py` and
  `src/painted_gt/reports.py`.

**Steps:**

- [ ] Add report tests for required top-level metrics.

Required metrics:

```text
mAP
mRecall
precision
recall
F1
debug_detection_f1
```

- [ ] Add steering diagnostic tests.

Required diagnostics:

```text
step_index_quality
first_token_im_end_rate
emitted_objects_per_image
malformed_no_progress_rate
exact_row_duplicate_rate
near_prediction_duplicate_rate
same_gt_duplicate_rate
matched_expected_next_gt_count
matched_previous_committed_gt_count
matched_future_scheduled_gt_count
matched_same_description_competitor_count
matched_best_image_gt_count
no_gt_match_count
first_drift_step
downstream_contaminated_step_count
multi_row_violation_rate
post_row_im_end_violation_rate
terminal_correct
terminal_premature
remaining_gt_at_terminal
matched_gt_at_terminal
coverage_at_terminal
stop_without_any_commit
step_cap_stop
step_cap_stop_rate
no_progress_limit_stop
no_progress_limit_stop_rate
generation_stop_reason_distribution
controller_outcome_distribution
overlap_conditioned_metrics
```

- [ ] Add denominator tests.

Malformed, invalid, and unparseable outputs must remain in precision/recall
denominators; they may appear in parseable subtables only as a separate view.

Normalization rule:

```text
dropped_malformed_candidates, no-valid-row attempts, invalid geometry,
unparseable steps, and unknown categories are denominator-bearing failed
prediction attempts
each no-valid generation step contributes one failed prediction attempt
unmatched GT objects after rollout are false negatives
generation truncation is recorded as a separate counter and never hidden
```

- [ ] Add compatibility rejection tests.

Reports must reject mismatched:

```text
slice id
schedule id
prompt id/fingerprint
painter id
decode identity
stop/EOS policy
scoreability policy
materialization identity
steering context policy for matched-policy claims
```

- [ ] Run report tests.

```bash
pytest tests/painted_gt/test_reports.py -q
openspec validate add-painted-gt-transcription-probe --strict
git diff --check
```

**Exit:** reports can block bad comparisons before model behavior is interpreted.

**Review Gate:** run review-convergence focused on metric validity, denominator
semantics, compatibility gates, and launch-readiness reporting.

## Wave 8: Configs And CPU Preflight Artifacts

**Purpose:** Prove materialization and packing are ready before GPU work.

**Files:**
- Add: `configs/coordexp_swift/painted_gt/next_object_steering/text_image_geo_sorted_gate256_*.yaml`
- Add: `configs/coordexp_swift/painted_gt/next_object_steering/image_only_geo_sorted_gate256_*.yaml`
- Add: `configs/coordexp_swift/infer/painted_gt/next_object_steering/text_image_geo_sorted_gate256_*.yaml`
- Add: `configs/coordexp_swift/infer/painted_gt/next_object_steering/image_only_geo_sorted_gate256_*.yaml`
- Modify research notes only after artifacts exist.

**Steps:**

- [ ] Add matched materialization/training configs for both policies.

Both configs must share:

```text
fixed 256-image slice
geo_sorted schedule
next_unmarked_object prompt
outline_center_flat_v1 painter
warm_start_expand_dora seed
same base model and repaired embedding payload
```

- [ ] Add matched inference configs for both policies.

Both configs must share:

```text
temperature=0
repetition_penalty=1.10
max_new_tokens=384
max_rollout_steps=64
free_raw_generation
same parser and evaluator identity
```

- [ ] Run CPU/materialization preflight.

Use the existing preflight entrypoint if available; otherwise run the targeted
pytest and materialization commands added in earlier waves. Do not invent a CLI
dry-run flag.

Required evidence:

```text
slice_manifest.json
painted_plan.jsonl
schedule_manifest.<schedule_id>.json
condition_manifest.json
visual_audit/index.json
materialization_identity.json
materialized examples JSONL
parity_summary.json
label_dumps/*.json
pack_cache_receipt.json
qwen_no_resize_preflight.json
geometry_audit_summary.json
target-span truncation counters are zero
pack-cache determinant includes materialization identity
```

- [ ] Run verification.

```bash
pytest tests/painted_gt/test_next_object_steering_materialization.py tests/painted_gt/test_label_dump.py -q
pytest tests/training/test_pack_cache.py tests/training/test_pipeline_assembly.py -q
openspec validate add-painted-gt-transcription-probe --strict
git diff --check
```

**Exit:** both policies have artifact-backed CPU preflight and packing evidence.

## Wave 9: Tiny Matched Training And Rollout

**Purpose:** Run the first actual evidence-producing experiment only after all
contract gates pass.

**Files:**
- Add or modify final launch configs under:
  `configs/coordexp_swift/painted_gt/next_object_steering/`
- Add or modify final inference configs under:
  `configs/coordexp_swift/infer/painted_gt/next_object_steering/`
- Write results under:
  `research/ideas/qwen3-vl-painted-gt-transcription-probe/`

**Steps:**

- [ ] Confirm all prior waves are committed and reviewed.

```bash
openspec validate add-painted-gt-transcription-probe --strict
git status --short --branch
```

- [ ] Launch tiny matched training only after user approval.

Default tiny budget:

```text
slice size = 256 images
initial training = 2 epochs
extension ladder = 4 epochs, then 8 epochs
extension criterion = intermediate reports improve but do not yet pass
beyond 8 epochs = blocked until review decision and user approval
GPU cap = up to 8 GPUs, recorded in launch manifest
eval/checkpoint cadence = step 1, midpoint, final
selected checkpoint = best predeclared primary metric; ties choose earliest
```

Abort if nonfinite loss, target-span truncation, missing checkpoint/final
metrics, missing frozen baseline, missing parity/visual audit evidence, artifact
compatibility failure, or GT leakage evidence appears.

Use available GPUs responsibly and record GPU count, backend, exact config,
commit, artifact root, all evaluated checkpoints, selected-checkpoint reason,
and selected-checkpoint provenance.

- [ ] Run matched rollouts for both policies.

Required:

```text
expanded warm-start seed -> text_image rollout
expanded warm-start seed -> image_only rollout
text_image-trained adapter -> text_image rollout
image_only-trained adapter -> image_only rollout
```

The frozen baseline is the exact expanded warm-start seed used for training:
base model plus expanded all-tower DoRA initialization plus repaired selected
embedding payload. Direct rollout from the historical step-917 source adapter is
optional diagnostic evidence, not a replacement for the frozen warm-start
baseline. Cross-policy runs are diagnostic only.

- [ ] Produce matched gate reports.

Reports must include:

```text
artifact roots
source adapter identity
checkpoint identity
materialization identity
prompt/decode/painter identity
primary metrics
diagnostic metrics
parity summary path
visual audit status
failure taxonomy
pass/gray/hard-fail classification
planned and actual epochs
evaluated checkpoint table
frozen baseline roots
trained rollout roots
loss trajectory
compatibility verdict
```

- [ ] Write the research result note.

The note must state whether evidence supports:

```text
visual ledger + language ledger
visual ledger only
neither
inconclusive due to artifact or metric failure
```

**Exit:** tiny matched next-object steering evidence exists, or a blocked report
explains exactly why it cannot be trusted.

## Final Acceptance Before Larger Runs

Larger training or mechanistic claims remain blocked until:

- all P0/P1 review findings are resolved or explicitly rejected with evidence;
- matched `text_image` and `image_only` tiny gates have valid reports;
- no target-span truncation occurred;
- report compatibility checks passed;
- terminal, duplicate, overlap, and invalid-output diagnostics are present;
- model behavior is interpreted at the correct evidence scope;
- the user explicitly approves the next launch.

## Self-Review Checklist

- Spec coverage: every next-object materialization, decode/eval, and launch-gate
  requirement maps to Waves 1-9.
- Placeholder scan: this roadmap avoids unfinished-marker placeholders and names
  concrete files, commands, and artifact fields.
- Simplicity check: no learned slots, no syntax-constrained primary decoding, no
  cross-step KV reuse, no overlap-depth color coding, and no duplicate
  suppression are introduced.
- Accuracy check: all fragile behavior is tested before launch: labels, prompt
  continuation, EOS/stop split, GT-independent cap, cache identity, policy
  parity, and denominator-bearing metrics.
