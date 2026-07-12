# Next-Object Steering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved `next_object_steering` revision for the painted-GT research branch, with matched `text_image` and `image_only` training/inference policies and an exact two-form supervised target grammar.

**Architecture:** Reuse the existing CoordExp-Swift painted-GT substrate and patch the narrow seams that define materialization, label masking, prompt identity, painter parity, rollout control, and reports. The implementation must make target grammar validation fail-fast before packing/GPU launch and must keep the old ordinary one-shot detector path unchanged.

**Tech Stack:** Python 3.12, PyTorch, Transformers Qwen3-VL, PEFT DoRA, Pillow, pytest, OpenSpec, existing `src/painted_gt`, `src/templates`, `src/inference`, `src/packing`, `src/training`, `src/eval`.

**Execution Roadmap:** Use
`docs/superpowers/plans/2026-07-06-next-object-steering-roadmap.md` for
implementation ordering, review gates, commit boundaries, and launch stop
conditions. This file remains the compact Superpowers task plan; the roadmap is
the handoff-grade execution guide.

## Global Constraints

- Effective object-step labels MUST be exactly `row(X) + <|object_ref_start|>`.
- Effective terminal-step labels MUST be exactly `<|im_end|>`.
- Object-step structural chat-template `<|im_end|>\n` MUST be ignored in labels.
- No third target form is allowed.
- V1 schedule is `geo_sorted`.
- V1 policies are `text_image` and `image_only`.
- V1 painter is `outline_center_flat_v1` with no overlap-depth color encoding.
- Inference commits first valid row only, strips trailing sentinel from committed rows, and does not suppress duplicates.
- First-token `<|im_end|>` is the only terminal stop authority.
- Cross-step KV cache reuse is forbidden because the image changes after each commit.
- Standard one-shot autoregressive detection remains a reference baseline, not the central ablation.

---

## File Structure

- Modify `src/painted_gt/materialization.py`: add next-object steering materialization, target-kind metadata, context-policy metadata, terminal rows, and target grammar validation.
- Modify `src/templates/renderer.py`: make supervision spans express object-continuation sentinel labels and terminal-stop labels without supervising structural object-step EOS.
- Modify `src/qwen/encoding.py`, `src/packing/supervision.py`, and `src/supervision/tokens.py` as needed so encoded and packed labels preserve the two-form target contract exactly.
- Modify `src/inference/prompt.py`: add or route `next_unmarked_object` prompt identity and ensure empty-prefix continuation remains a single assistant turn.
- Modify `src/config/inference.py` and `src/inference/pipeline.py` as needed so config-driven inference can dispatch to the next-object steering controller without disturbing ordinary one-shot inference.
- Modify `src/inference/artifacts.py`: write `painted_gt_step_trace.jsonl` and link committed/evaluator rows to step traces.
- Modify `src/painted_gt/materialization.py` as the current painter owner, or deliberately extract `src/painted_gt/painting.py`: expose shared `outline_center_flat_v1` for both training and inference and record painter provenance and overlap diagnostics.
- Modify `src/painted_gt/decode.py`: add next-object steering rollout controller for `text_image` and `image_only`.
- Modify `src/painted_gt/metrics.py` and `src/painted_gt/reports.py`: add steering diagnostics and overlap-conditioned summaries.
- Add configs under `configs/coordexp_swift/painted_gt/next_object_steering/`.
- Add tests under `tests/painted_gt/` and targeted template/supervision tests if existing boundaries require it.

## Task 1: Target Grammar And Label-Mask Contract

**Files:**
- Modify: `src/painted_gt/materialization.py`
- Modify: `src/templates/renderer.py`
- Modify if needed: `src/qwen/encoding.py`
- Modify if needed: `src/packing/supervision.py`
- Modify if needed: `src/supervision/tokens.py`
- Test: `tests/painted_gt/test_next_object_steering_materialization.py`
- Test: `tests/templates/test_renderer.py` or the current renderer test owner
- Test: current qwen/packing/supervision label test owners if labels are transformed after rendering

**Interfaces:**
- Produces: `step_target_kind: Literal["object_continuation", "terminal_stop"]`
- Produces: `steering_context_policy: Literal["text_image", "image_only"]`
- Produces: materialized rows whose non-ignored label ids are exactly `[row_tokens..., object_ref_start_id]` or `[im_end_id]`
- Produces: sample dumps proving the same non-ignored ids after rendering, Qwen encoding, packed supervision, and final training micro-step assembly

- [ ] **Step 1: Write failing tests for the two legal target forms**

Create tests that materialize a two-object fake image schedule and assert:

```python
def test_object_continuation_labels_end_with_object_ref_start():
    row = build_fake_next_object_row(step_target_kind="object_continuation")
    supervised = non_ignored_token_texts(row)
    assert supervised == [
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

def test_terminal_stop_labels_only_im_end():
    row = build_fake_next_object_row(step_target_kind="terminal_stop")
    assert non_ignored_token_texts(row) == ["<|im_end|>"]
```

Run:

```bash
pytest tests/painted_gt/test_next_object_steering_materialization.py -q
pytest tests/painted_gt/test_label_dump.py tests/training/test_pipeline_assembly.py -q
```

Expected before implementation: failures because the builder/metadata does not exist or still supervises the old row-only/row-plus-EOS form.

- [ ] **Step 2: Add fail-fast grammar validation**

Implement validation that rejects these non-terminal effective label forms:

```text
row(X)
row(X) + <|im_end|>
row(X) + <|object_ref_start|> + <|im_end|>
multiple complete rows
empty object target
```

Add tests that each invalid form raises the repo's existing template/materialization contract error.

- [ ] **Step 3: Implement terminal rows for both policies**

Add terminal materialization rows where all GT objects are painted. Assert:

```text
text_image terminal prefix = all canonical GT rows
image_only terminal prefix = empty
```

Run:

```bash
pytest tests/painted_gt/test_next_object_steering_materialization.py -q
pytest tests/templates/test_renderer.py -q
pytest tests/painted_gt/test_label_dump.py tests/training/test_pipeline_assembly.py -q
```

Expected after implementation: pass.

## Task 2: Prompt Identity And Context Policies

**Files:**
- Modify: `src/inference/prompt.py`
- Modify: `src/painted_gt/materialization.py`
- Test: `tests/inference/test_prompt_image.py`
- Test: `tests/painted_gt/test_next_object_steering_materialization.py`

**Interfaces:**
- Produces: prompt id `next_unmarked_object`
- Produces: prompt fingerprint shared by training and rollout artifacts
- Produces: first-step unpainted visual state for next-object steering without weakening existing non-empty painted-plan guards

- [ ] **Step 1: Add failing prompt identity tests**

Assert that both policies use a prompt containing the meaning:

```text
next visible object instance that has not already been marked
marked rectangles and center points are already committed
overlapping or partially occluded uncommitted objects should still be emitted
<|im_end|> only when every visible object is committed
```

Also assert that the old `Detect all visible objects` prompt is rejected for primary steering.

- [ ] **Step 2: Implement shared prompt routing**

Add the prompt identity in the existing prompt/template owner, not in an ad hoc rollout script. Keep training and inference artifacts recording the same prompt id and fingerprint.

- [ ] **Step 3: Add the next-object first-step visual state**

Implement the first steering step as an unpainted image with an empty committed
set. Keep existing paint-all and historical stepwise materialization callers
rejecting empty painted-object plans unless they explicitly use the
next-object first-step path.

- [ ] **Step 4: Verify single-turn continuation**

Add tests proving generated chat text has one open assistant turn:

```text
<|im_start|>assistant
[optional prefix]
[model continues here]
```

and does not close the assistant before generation.

The test MUST include the empty-prefix case used by
`steering_context_policy=image_only` or `assistant_prefix_text=""`. It must prove
the final assistant turn is still open, the rendered text does not end with a
structural `<|im_end|>\n` before generation, and the prompt path uses
`continue_final_message=True` or an equivalent single-turn continuation contract.

Run:

```bash
pytest tests/inference/test_prompt_image.py tests/painted_gt/test_next_object_steering_materialization.py -q
```

## Task 3: Shared Flat Painter And Overlap Diagnostics

**Files:**
- Modify: `src/painted_gt/painting.py` or current painter owner
- Modify: `src/painted_gt/materialization.py`
- Modify: `src/painted_gt/decode.py`
- Test: `tests/painted_gt/test_materialization.py`
- Test: `tests/painted_gt/test_decode_controllers.py`

**Interfaces:**
- Produces: `paint_style="outline_center_flat_v1"`
- Produces: overlap diagnostics fields in painted plans and rollout steps

- [ ] **Step 1: Write failing painter parity test**

Assert training materialization and rollout call the same painter function/style and record:

```json
{
  "paint_style": "outline_center_flat_v1",
  "style_id": "outline_center_flat_v1",
  "uses_fill": false,
  "uses_overlap_depth_encoding": false
}
```

- [ ] **Step 2: Implement flat painter provenance**

Use fixed magenta outline and yellow center point with no fill, alpha accumulation, darkening, or brightening. Later marks draw on top normally.

- [ ] **Step 3: Add overlap diagnostics**

Record at least:

```text
target_overlaps_committed_region
target_center_inside_committed_region
target_committed_overlap_area_ratio
prediction_overlaps_committed_region
prediction_center_inside_committed_region
prediction_committed_overlap_area_ratio
```

These are geometry-based bbox-union diagnostics. Rendered-pixel overlap may be
recorded separately for visual audit, but it must not replace the geometry
fields above.

Run:

```bash
pytest tests/painted_gt/test_materialization.py tests/painted_gt/test_decode_controllers.py -q
```

## Task 4: Next-Object Rollout Controller

**Files:**
- Modify: `src/painted_gt/decode.py`
- Modify if needed: `src/config/inference.py`
- Modify if needed: `src/inference/pipeline.py`
- Modify: `src/inference/artifacts.py`
- Test: `tests/painted_gt/test_decode_controllers.py`
- Test: `tests/inference/test_artifacts.py`
- Test: config-load or inference-dispatch tests for next-object steering mode

**Interfaces:**
- Produces: `run_next_object_steering_controller(...)`
- Consumes: `steering_context_policy`
- Produces: step trace rows with committed row, raw text, drops, image transition, prefix transition, `generation_stop_reason`, `controller_outcome`, and stop/violation diagnostics
- Produces: `painted_gt_step_trace.jsonl` with `raw_generated_token_ids = DecodeResult.generated_token_ids` and `generation_stop_reason = DecodeResult.stop_reason`
- Produces: config-driven dispatch into the next-object controller while ordinary one-shot inference remains unchanged

- [ ] **Step 1: Write failing controller tests**

Test these cases:

```text
first-token <|im_end|> -> terminal_stop
row + <|object_ref_start|> -> commit row only
row + <|im_end|> -> backend generation_stop_reason=im_end, commit row, controller_outcome=post_row_im_end_violation, continue
malformed + valid row -> drop malformed and commit valid row
no valid row -> unchanged state until no_progress_limit=2, then no_progress_limit_stop
duplicate row -> commit and flag duplicate without guard
two valid rows -> commit first and record multi_row_violation
image_only -> prefix stays empty
text_image -> prefix appends canonical committed row only
max_rollout_steps=64 default -> step_cap_stop without consulting GT count
```

Raw-token-first classification is required: inspect the first generated token
before parser stripping. Parser text may strip one terminal `<|im_end|>`, but it
must not hide `row + <|im_end|>` violations or convert backend EOS into semantic
terminal authority.

- [ ] **Step 2: Implement controller with full prefill per step**

Do not reuse cross-step KV. Rebuild image plan and prompt each committed step because the painted image changes.

Add inference dispatch only at the normal config/pipeline seam: a config that
declares next-object steering should route to this controller, while ordinary
standard inference still routes to the existing one-shot path.

- [ ] **Step 3: Add GT-isolation and trace-artifact tests**

Assert fake GT accessors raise if touched during controller execution. GT boxes
and categories may be passed only to scorer/report code after rollout.

Persist step traces through `src/inference/artifacts.py`, with committed
prediction rows and evaluator-input rows referencing `step_trace_id`.

- [ ] **Step 4: Verify controller artifacts**

Run:

```bash
pytest tests/painted_gt/test_decode_controllers.py tests/inference/test_artifacts.py -q
```

## Task 5: Configs, Preflight, Materialization Identity, And Reports

**Files:**
- Add: `configs/coordexp_swift/painted_gt/next_object_steering/text_image_geo_sorted_gate256_*.yaml`
- Add: `configs/coordexp_swift/painted_gt/next_object_steering/image_only_geo_sorted_gate256_*.yaml`
- Modify: `src/painted_gt/materialization.py`
- Modify if needed: `src/training/pack_cache.py`
- Modify: `src/painted_gt/reports.py`
- Modify: `src/painted_gt/metrics.py`
- Test: `tests/painted_gt/test_reports.py`
- Test: `tests/training/test_pack_cache.py`

**Interfaces:**
- Produces: matched-policy gate reports for `next_object_steering.text_image` and `next_object_steering.image_only`
- Produces: condition/materialization identity that includes next-object mode, context policy, prompt id/fingerprint, target-grammar id, painter id, schedule id, source JSONL SHA, and materialized example payload SHA
- Produces: `PRIMARY_METRICS_BY_MODE` or equivalent explicit metric registration for report validation

- [ ] **Step 1: Add report tests**

Assert reports include:

```text
mAP
mRecall
precision
recall
F1
first_token_im_end_rate
emitted_objects_per_image
malformed_no_progress_rate
step_index_quality
duplicate_rate
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
overlap_conditioned_metrics
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
```

Invalid generation normalization:

```text
dropped malformed candidates, no-valid-row attempts, invalid geometry,
unparseable steps, and unknown categories stay in denominator-bearing failed
prediction attempts;
each no-valid generation step contributes one failed prediction attempt;
unmatched GT objects after rollout are false negatives;
generation truncation is reported separately.
```

- [ ] **Step 2: Add preflight configs and identity checks**

Create matched configs for tiny materialization/preflight and packing checks, one
per policy. Config names must include `next_object_steering`, the policy,
`geo_sorted`, and the slice/limit.

Add tests proving:

```text
different prompt id/fingerprint, target grammar id, painter id, schedule id,
policy, or materialized JSONL sha changes the materialization identity;
worker/cache/timestamp metadata does not change semantic identity;
src/training/pack_cache.py consumes the materialization identity opaquely and
does not import src.painted_gt or parse painted-GT manifests.
```

- [ ] **Step 3: Run preflight validation**

Run:

```bash
pytest tests/painted_gt/test_next_object_steering_materialization.py tests/painted_gt/test_label_dump.py -q
pytest tests/training/test_pack_cache.py tests/training/test_pipeline_assembly.py -q
pytest tests/painted_gt/test_reports.py -q
openspec validate add-painted-gt-transcription-probe --strict
git diff --check
```

Expected: preflight/materialization tests prove target labels are exactly one of
the two legal forms before any training launch. Do not use a nonexistent
training CLI dry-run flag unless a real dry-run surface is intentionally
implemented and tested.

Required pre-GPU receipts:

```text
slice_manifest.json
painted_plan.jsonl
schedule_manifest.<schedule_id>.json
condition_manifest.json
visual_audit/index.json
geometry_audit_summary.json
qwen_no_resize_preflight.json
materialization_identity.json
materialized examples JSONL
label_dumps/*.json
pack_cache_receipt.json
```

- [ ] **Step 4: Emit paired policy parity artifacts**

For matched `text_image` and `image_only` materialization, write a parity summary
and fail if any of these differ unexpectedly:

```text
(image_id, step_index, target_object_id, terminal) row set
target-span truncation counters
dropped-example counters
terminal-step counts
```

Also record token-length distributions and non-ignored label counts by policy.
Target-span truncation blocks training.

## Task 6: Tiny Matched Training And Rollout Gate

**Files:**
- Add or modify: final tiny configs under `configs/coordexp_swift/painted_gt/next_object_steering/`
- Modify: research notes after metrics exist

**Interfaces:**
- Produces: artifact roots for matched `text_image` and `image_only` training and rollout

- [ ] **Step 1: Launch tiny overfit runs**

Use all available GPUs responsibly. Train the matched `text_image` and
`image_only` adapters from the approved warm-start seed.

Budget rule:

```text
slice size = 256 images
initial training = 2 epochs
extension ladder = 4 epochs, then 8 epochs
extension only if intermediate reports improve but do not yet pass
beyond 8 epochs requires review decision and user approval
eval/checkpoint cadence = step 1, midpoint, final
checkpoint selection = best predeclared primary metric, earliest on ties
```

- [ ] **Step 2: Run matched rollouts**

Run rollouts with:

```text
temperature=0
repetition_penalty=1.10
max_new_tokens=384
max_rollout_steps=64
free_raw_generation
```

Required rollout matrix:

```text
expanded warm-start seed -> text_image rollout
expanded warm-start seed -> image_only rollout
text_image-trained adapter -> text_image rollout
image_only-trained adapter -> image_only rollout
```

The frozen baseline is the exact expanded warm-start seed used for training,
including the repaired selected-token embedding payload. Direct step-917 source
rollout is optional diagnostic evidence, not a replacement.

- [ ] **Step 3: Write result note**

Record artifact roots, planned/actual epochs, evaluated checkpoint table,
selected checkpoint reason, frozen baseline roots, trained rollout roots,
metrics, compatibility verdict, and conclusion in
`research/ideas/qwen3-vl-painted-gt-transcription-probe/`. Do not claim a
larger mechanism than the evidence supports.

- [ ] **Step 4: Review gate**

Run a review-convergence loop before launching any larger run or making a mechanistic conclusion.

## Self-Review

- Spec coverage: this plan covers target grammar, prompt identity, context policies, painter parity, rollout semantics, reports, dry runs, tiny training, and review.
- Placeholder scan: no placeholder markers are used.
- Type consistency: the plan uses `next_object_steering`, `text_image`, `image_only`, `object_continuation`, `terminal_stop`, and `outline_center_flat_v1` consistently with the OpenSpec revision.
