# Segment-Aware Packing V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Segment-Aware Packing V1 for Stage-1 teacher-forcing training so `training.packing: true` can use static long packed physical rows while preserving segment-correct FlashAttention boundaries, position resets, image ownership, and rebased teacher-forcing sidecars.

**Architecture:** Keep the existing static packing membership engine. Add a compact `PackedSegmentMap` sidecar and pure validation/rebase helpers under `src/training/packing/`. Integrate those helpers through the existing data-collator/enricher and batch-extras path so `TrainerLossBridge` and `ObjectiveRunner` continue consuming normal teacher-forcing IRs. Explicitly reject unsupported packed surfaces with reason codes.

**Tech Stack:** Python, PyTorch, Transformers Qwen3-VL, FlashAttention v2, ms-swift static packing conventions, pytest, existing CoordExp batch extras/model-input-bundle contracts.

---

## Scope Lock

Implement only the V1 supported surface:

- Stage-1 teacher-forcing sidecar supervision.
- Static length-fill packing.
- Qwen3-VL-style forward path.
- FlashAttention v2 with explicit `cu_seq_lens_q`, `cu_seq_lens_k`, `max_length_q`, and `max_length_k`.
- Full logits only.
- One physical packed row per per-rank forward.
- Homogeneous teacher-forcing supervision profile inside each physical pack.

Do not implement in V1:

- row-coverage packing
- Qwen2/Qwen2.5 support
- non-FA2 packed sidecar training
- dynamic packing
- logits projection or `logits_to_keep`
- multi-row physical forward batches
- atom-density optimized repacking
- same-base-image as a semantic requirement

## Approval Gate

This document is a pre-implementation plan. Do not start production code changes until the user approves this plan after review.

Stable docs and OpenSpec should not be promoted during planning. After implementation evidence exists, supported `training.packing: true` behavior for teacher-forcing sidecars is a stable training/config/loss contract change and must be promoted deliberately rather than implied by the code diff.

## Target Files

Create:

- `src/training/packing/__init__.py`
- `src/training/packing/segment_map.py`
- `tests/training/packing/test_segment_map.py`
- `tests/training/packing/test_teacher_forcing_rebase.py`
- `tests/training/packing/test_segment_aware_collator.py`
- `tests/training/packing/test_segment_aware_objective_parity.py`
- `tests/training/packing/test_segment_aware_fa2_smoke.py`

Modify:

- `src/config/schema.py`
- `src/detection/runtime.py`
- `src/data_collators/enrichers.py`
- `src/trainers/batch_extras.py`
- `src/trainers/metrics/teacher_forcing.py`
- `src/training/bridge/loss_bridge.py`
- `src/training/encoding/model_inputs.py`
- `src/trainers/metrics/batch_contract.py`
- `src/trainers/metrics/aggregate_tokens.py`
- `tests/test_model_input_bundle_contract.py`
- `tests/test_batch_extras_contract.py`
- `tests/test_train_batch_contract.py`
- `tests/test_teacher_forcing_sidecar_bridge.py`
- `tests/test_trainer_loss_bridge_qwen3vl_contract.py`
- `tests/test_detection_training_config_contract.py`
- `tests/test_training_config_strict_unknown_keys.py`
- `tests/test_packing_attention_backend_gate.py`
- `tests/test_aggregate_token_metric_events.py`
- `tests/test_trainer_metrics_payload_contract.py`
- `tests/detection/coverage/test_config.py`
- `tests/test_training_architecture_golden_thread.py`
- `docs/data/PACKING.md`
- `docs/IMPLEMENTATION_MAP.md`

Do not modify `src/training/objectives/runner.py` or `src/training/objectives/teacher_forcing.py` unless tests prove a packing-unaware physical `SupervisionBatch` and `LabelLogitRowMap` cannot preserve their current contract.

## Data Contracts

Add these constants and sidecar keys:

```python
PACKED_SEGMENT_MAP_KEY = "packed_segment_map"
SEGMENT_AWARE_PACKING_STATUS_KEY = "segment_aware_packing_status"
```

`packed_segment_map` is batch-extras-only and sidecar-only. It must not enter model kwargs.

Pack-level teacher-forcing contract:

```text
teacher_forcing_target_ir:
  length = logical_segment_count

sample_id:
  length = logical_segment_count

V1 physical_batch_size:
  1

For each logical segment:
  one rebased TeacherForcingTargetIR

PackedSegmentMap:
  maps many logical sample ids to physical row 0
```

FlashAttention contract:

```text
cu_seq_lens_q[0] == 0
cu_seq_lens_q[-1] == input_ids.shape[1]
cu_seq_lens_q == cu_seq_lens_k
max_length_q == max(diff(cu_seq_lens_q))
max_length_k == max(diff(cu_seq_lens_k))
pack_num_samples == len(cu_seq_lens_q) - 1
packed_segment_map.segment_count == len(cu_seq_lens_q) - 1
packed_segment_map.sample_id_to_physical_row[sample_id] == 0 for V1
```

## Failure Reason Codes

Use stable string reason codes in exceptions, metrics, and compact failure artifacts:

```text
row_coverage_unsupported_v1
mixed_supervision_profile_unsupported_v1
ordinary_sft_mixed_with_teacher_forcing_unsupported_v1
model_family_unsupported_v1
flash_attention_required_v1
missing_cu_seq_lens_v1
invalid_cu_seq_lens_v1
logits_to_keep_unsupported_v1
multi_physical_row_unsupported_v1
dynamic_packing_unsupported_v1
teacher_forcing_sidecar_missing_v1
packed_segment_map_missing_v1
teacher_forcing_atom_out_of_segment_bounds_v1
segment_length_mismatch_v1
image_token_ownership_mismatch_v1
sample_id_owner_mismatch_v1
selected_token_mismatch_v1
attention_mask_unsupported_v1
use_cache_unsupported_v1
per_device_batch_gt_one_unsupported_v1
multiple_planned_packs_unsupported_v1
packing_component_sidecars_lost_v1
position_ids_qwen3_mrope_mismatch_v1
```

## Implementation Tasks

### Task 1: Add Pure Segment Map Types

- [ ] Create `src/training/packing/__init__.py`.
- [ ] Create `src/training/packing/segment_map.py`.
- [ ] Define frozen dataclasses:

```python
@dataclass(frozen=True)
class PackedSegment:
    physical_row: int
    logical_segment_index: int
    source_batch_index: int
    source_sample_id: str | None
    start: int
    end: int
    selected_token_positions: Sequence[int] = ()
    base_image_id: str | None = None
    supervision_profile: str = "teacher_forcing"

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class PackedSegmentMap:
    physical_batch_size: int
    sequence_length: int
    segments: Sequence[PackedSegment]
    support_status: Mapping[str, object] = field(default_factory=dict)

    def sample_id_to_physical_row(self) -> dict[str, int]:
        mapping: dict[str, int] = {}
        for segment in self.segments:
            if segment.source_sample_id is None:
                continue
            existing = mapping.get(segment.source_sample_id)
            if existing is not None and existing != segment.physical_row:
                raise SegmentAwarePackingError(
                    reason_code="sample_id_owner_mismatch_v1",
                    details={
                        "sample_id": segment.source_sample_id,
                        "first_physical_row": existing,
                        "second_physical_row": segment.physical_row,
                    },
                )
            mapping[segment.source_sample_id] = segment.physical_row
        return mapping
```

- [ ] Add helpers:

```text
build_packed_segment_map(
    *,
    packed_component_rows: Sequence[Mapping[str, object]],
    collated_input_ids_shape: tuple[int, int],
    sample_ids: Sequence[str | None] | None,
    base_image_ids: Sequence[str | None] | None,
    supervision_profiles: Sequence[str],
) -> PackedSegmentMap

validate_packed_segment_map(
    packed_map: PackedSegmentMap,
    *,
    require_single_physical_row: bool = True,
) -> None

cu_seq_lens_from_segment_map(
    packed_map: PackedSegmentMap,
) -> tuple[list[int], int]

validate_segment_map_against_collated_inputs(
    packed_map: PackedSegmentMap,
    *,
    input_ids: torch.Tensor,
    labels: torch.Tensor | None,
    text_position_ids: torch.Tensor | None,
    cu_seq_lens_q: torch.Tensor | None,
    cu_seq_lens_k: torch.Tensor | None,
    max_length_q: int | torch.Tensor | None,
    max_length_k: int | torch.Tensor | None,
    pack_num_samples: torch.Tensor | None,
) -> None
```

- [ ] Validation must reject non-contiguous segments, non-positive lengths, out-of-range spans, overlapping spans, row count mismatch, multi-row physical batches in V1, and mixed profiles.
- [ ] Component segment spans must be derived from raw packed component rows before the upstream `packing_row` step, using component `attention_mask`, `input_ids`, or `labels` lengths in that order.
- [ ] Do not use existing `dataset_segments` as a segment-span source; it is a per-physical-row length summary. Treat `pack_num_samples` only as a sanity count.
- [ ] Validate derived spans against final `input_ids`, labels, explicit `cu_seq_lens_*`, `text_position_ids`, and `pack_num_samples`.
- [ ] Keep exceptions typed and reason-coded, for example:

```python
raise SegmentAwarePackingError(
    reason_code="segment_length_mismatch_v1",
    details={"segment_index": 1, "expected_length": 7, "actual_length": 6},
)
```
- [ ] Add a negative test where `pack_num_samples=2` but component lengths disagree with `cu_seq_lens_q`, and require `segment_length_mismatch_v1` or `invalid_cu_seq_lens_v1`.

Run:

```bash
python -m pytest tests/training/packing/test_segment_map.py
```

### Task 2: Add Teacher-Forcing IR Rebase Helpers

- [ ] In `src/training/packing/segment_map.py`, add:

```text
rebase_teacher_forcing_irs_for_pack(
    *,
    target_irs: Sequence[TeacherForcingTargetIR],
    packed_map: PackedSegmentMap,
) -> Sequence[TeacherForcingTargetIR]
```

- [ ] Preserve atom fields other than `batch_index`, `logit_position`, and `target_position`.
- [ ] Set rebased atom coordinates:

```text
batch_index = segment.physical_row
logit_position = segment.start + atom.logit_position
target_position = segment.start + atom.target_position
```

- [ ] Reject atoms that fall outside their original logical segment length.
- [ ] Produce exactly one rebased `TeacherForcingTargetIR` per logical segment, preserving the one-to-one relation with `sample_id`.
- [ ] Merge metadata with explicit provenance:

```python
metadata = {
    "segment_aware_packing_active": True,
    "logical_segment_index": segment.logical_segment_index,
    "source_sample_id": segment.source_sample_id,
    "physical_row": segment.physical_row,
    "support_status": {"segment_aware_packing_supported_surface": True},
}
```

- [ ] Add `build_packed_teacher_forcing_supervision` or an equivalent helper in `src/trainers/metrics/teacher_forcing.py` that consumes `PackedSegmentMap` and emits:

```python
supervision = SupervisionBatch(
    spans=tuple(spans),
    batch_id="teacher_forcing_segment_aware_packed",
)
sample_id_to_batch_index = {
    source_sample_id: segment.physical_row
    for segment in packed_map.segments
}
```

- [ ] This helper must not call `_with_batch_index` with the logical enumerate index for packed IRs; rebased atom `batch_index` is already the physical row.
- [ ] Keep `ObjectiveRunner` and `TeacherForcingObjective` packing-unaware by passing them physical coordinates and the correct row map.
- [ ] Add an asymmetric golden fixture with distinct sample ids, segment-start atoms, segment-end atoms, terminal STOP atoms, and unique selected token ids/logit rows.
- [ ] Assert the rebased atom table exactly: source segment, sample id, physical row, `logit_position`, `target_position`, selected token, label row, and denominator.

Run:

```bash
python -m pytest tests/training/packing/test_teacher_forcing_rebase.py
```

### Task 3: Register Batch Extras and Model Input Bundle Contracts

- [ ] In `src/trainers/batch_extras.py`, add `PACKED_SEGMENT_MAP_KEY` and `SEGMENT_AWARE_PACKING_STATUS_KEY`; include both in `BATCH_EXTRAS_KEYS`.
- [ ] Extend the batch-extras dataclass with `packed_segment_map` and `segment_aware_packing_status`.
- [ ] Ensure `maybe_pop_and_stash_batch_extras` removes both keys from model inputs.
- [ ] In `src/training/encoding/model_inputs.py`, add `packed_segment_map` and `segment_aware_packing_status` to sidecar-only keys.
- [ ] Keep `cu_seq_lens_q`, `cu_seq_lens_k`, `max_length_q`, and `max_length_k` in forwarded model-input keys.
- [ ] Extend `tests/test_model_input_bundle_contract.py` so `ModelInputBundle.from_mapping` rejects `packed_segment_map` and `segment_aware_packing_status` as sidecar-only if they reach model-input construction.
- [ ] Extend `tests/test_batch_extras_contract.py` so `maybe_pop_and_stash_batch_extras` stashes/removes both keys before `ModelInputBundle` construction.
- [ ] Add a bridge/model-call spy proving neither key reaches `model(**kwargs)`.

Run:

```bash
python -m pytest tests/test_model_input_bundle_contract.py tests/test_batch_extras_contract.py
```

### Task 4: Integrate Segment-Aware Enrichment in the Collator Path

- [ ] In `src/data_collators/enrichers.py`, replace the current packed teacher-forcing hard failure with segment-aware construction for supported V1 packs.
- [ ] Keep the current hard failure for row coverage packed sidecars and update the message to include `row_coverage_unsupported_v1`.
- [ ] Snapshot raw packed component rows before calling or relying on the upstream packed output, because ms-swift `packing_row` only carries a narrow key set and can drop local semantic sidecars.
- [ ] Reject packed teacher-forcing V1 if `len(raw_batch) != 1`, because ms-swift may flatten multiple planned packs into one physical row when `per_device_train_batch_size > 1`.
- [ ] Detect packed batches from the raw batch shape and existing metadata, but derive component spans from raw packed component rows rather than `dataset_segments`.
- [ ] For packed teacher-forcing batches, produce:

```python
collated["packed_segment_map"] = packed_map
collated["teacher_forcing_target_ir"] = tuple(rebased_irs)
collated["sample_id"] = tuple(segment.source_sample_id for segment in packed_map.segments)
collated["segment_aware_packing_status"] = status
```

- [ ] Ensure the rebased IR list length matches the logical segment count.
- [ ] Ensure the `sample_id` sidecar length matches the logical segment count.
- [ ] Ensure V1 rejects physical batch size greater than 1.
- [ ] Ensure ordinary unpacked teacher-forcing batches keep current behavior byte-for-byte at the contract level.
- [ ] Add collator tests covering:

```text
packed two-segment teacher-forcing batch succeeds
unpacked teacher-forcing batch is unchanged
row coverage packed batch fails closed
mixed profiles fail closed
ordinary SFT mixed with teacher-forcing fails closed
two planned pack-lists in one collator call fail closed
raw component sidecars missing after packing fails closed
multi-physical-row pack fails closed
```

Run:

```bash
python -m pytest tests/training/packing/test_segment_aware_collator.py tests/training/packing/test_segment_aware_fail_closed.py
```

### Task 5: Enforce Explicit FA2 Boundary and Position Contracts

- [ ] Use `cu_seq_lens_from_segment_map` to derive or validate `cu_seq_lens_q`, `cu_seq_lens_k`, `max_length_q`, and `max_length_k`.
- [ ] Do not rely on Transformers position-id inference for V1.
- [ ] Emit `cu_seq_lens_q` and `cu_seq_lens_k` as `torch.int32` tensors and normalize `max_length_q` and `max_length_k` to Python `int`, unless the smoke proves tensor scalars are required.
- [ ] Require padding-free training prefill: no real padding mask forwarded to Qwen3-VL, `past_key_values is None`, and `use_cache` false.
- [ ] In `src/trainers/metrics/batch_contract.py`, extend packed validation to assert:

```text
explicit cu_seq_lens_* are present when segment-aware packing is active
text_position_ids resets to 0 at each segment boundary
pack_num_samples equals segment count
cu_seq_lens match PackedSegmentMap boundaries when sidecar is present
physical batch size is one
attention_mask is absent from forwarded model kwargs
past_key_values and use_cache are absent/false
```

- [ ] Keep Qwen3 4-row `position_ids` preparation in `prepare_forward_inputs`; do not duplicate Qwen model logic in packing code.
- [ ] When `packed_segment_map` is active, instantiate or configure `TrainerLossBridgeSettings(packing_enabled=True)` so Qwen packed-position checks execute.
- [ ] Add model-family-specific Qwen3 tests:

```text
row 0 text_position_ids reset points equal cu_seq_lens_q[:-1]
rows 1-3 preserve MRoPE shape and image placeholder alignment
3-row multimodal position_ids plus text_position_ids become 4-row before model
```

- [ ] Add tests for valid and invalid `cu_seq_lens_*`, missing explicit `cu_seq_lens_*`, text-position reset mismatches, non-`None` padding masks, cache/generation inputs, and physical batch size greater than 1.

Run:

```bash
python -m pytest \
  tests/training/packing/test_segment_aware_collator.py \
  tests/test_train_batch_contract.py \
  tests/test_trainer_loss_bridge_qwen3vl_contract.py \
  tests/test_training_architecture_golden_thread.py
```

### Task 6: Reject Sliced Logits and Unsupported Model/Attention Surfaces

- [ ] Reuse existing `TrainerLossBridgeSettings.allow_logits_projection=False`.
- [ ] Add an earlier reason-coded rejection when packed segment-aware sidecars and `logits_to_keep` coexist.
- [ ] Reject `args.use_logits_to_keep=True` before or during trainer input preparation, not only inside `TrainerLossBridge`, because ms-swift can rewrite labels and add `inputs["logits_to_keep"]`.
- [ ] Fail fast if any `logits_to_keep` key remains after trainer input preparation when `packed_segment_map` is active.
- [ ] Add a supported-surface check that requires Qwen3-VL-style model metadata and FA2 attention implementation before activating segment-aware packing.
- [ ] If reliable model metadata is not available at collator time, perform the model/attention check in the training pipeline before dataloader construction and store the result in a small capability object or config-derived status passed to the collator.
- [ ] In `src/config/schema.py` and `src/detection/runtime.py`, replace the blanket teacher-forcing `training.packing=true` rejection with a narrow V1 capability gate:

```text
allow only teacher_forcing + static packing + Qwen3-VL + FA2 + padding-free + per_device_train_batch_size=1 + full logits
keep row coverage and all other packed sidecar surfaces fail-closed
```

- [ ] Require `per_device_train_batch_size == 1` for the V1 supported surface.
- [ ] Tests must prove unsupported surfaces fail closed instead of falling back to wrong standard packing.

Run:

```bash
python -m pytest \
  tests/training/packing/test_segment_aware_collator.py \
  tests/training/packing/test_segment_aware_fail_closed.py \
  tests/test_detection_training_config_contract.py \
  tests/detection/coverage/test_config.py \
  tests/test_training_config_strict_unknown_keys.py \
  tests/test_packing_attention_backend_gate.py \
  tests/test_teacher_forcing_sidecar_bridge.py
```

### Task 7: Objective Parity Test

- [ ] Build a deterministic CPU fixture with two logical teacher-forcing examples.
- [ ] Use asymmetric atom rows and token ids so scalar loss cannot pass accidentally:

```text
segment A:
  unique non-terminal atom near segment start
  terminal STOP atom near segment end
segment B:
  unique non-terminal atom at a different offset
  terminal STOP atom at a different offset
```

- [ ] Compute teacher-forcing loss in the current unpacked shape.
- [ ] Pack the same examples into one physical row.
- [ ] Rebase their IRs using `rebase_teacher_forcing_irs_for_pack`.
- [ ] Feed full logits and the rebased IR through the existing teacher-forcing objective path.
- [ ] Assert matching loss numerator and denominator within floating-point tolerance.
- [ ] Assert denominator equals supervised atoms after remap.
- [ ] Assert distinct logical `sample_id`s remain distinct, both map to physical row `0`, and each span resolves to the expected physical logit rows.
- [ ] Assert selected tokens match the physical labels/input ids at rebased target positions.

Run:

```bash
python -m pytest tests/training/packing/test_segment_aware_objective_parity.py
```

### Task 8: Observability and Failure Artifacts

- [ ] Add bounded metrics/status fields:

```text
segment_aware_packing_active
segment_aware_packing_supported_surface
unsupported_reason_codes
validated_packs
failed_packs
failure_issue_counts
physical_packed_rows
logical_segments
base_images_per_pack
supervised_teacher_forcing_atoms
supervised_label_tokens_per_second
supervised_atoms_per_second
logical_segments_per_second
text_tokens
visual_tokens
pack_fill_ratio
padding_slack
```

- [ ] Reuse existing metric/event plumbing where possible, especially `src/metrics/events.py` and trainer metrics helpers.
- [ ] Define per-update and per-second scopes explicitly so throughput claims use logical supervised exposure, not raw physical rows/sec alone.
- [ ] DDP/gradient-accumulation aggregation must be deterministic and must not double-count logical segments.
- [ ] Add compact rank-0 failure artifact emission only if the existing training diagnostics pattern already has an artifact sink in this path.
- [ ] Failure artifacts must include issue code, pack id, sample ids, segment spans, and shape summary; they must not dump full tensors.
- [ ] Add unit tests for counter aggregation and capped failure records.

Run:

```bash
python -m pytest \
  tests/training/packing/test_segment_aware_metrics.py \
  tests/test_aggregate_token_metric_events.py \
  tests/test_trainer_metrics_payload_contract.py
```

### Task 9: Tiny Qwen3-VL FlashAttention v2 Smoke

- [ ] Add a hardware-gated smoke test or script entrypoint that uses the local `conda ms` environment packages already available in this workspace.
- [ ] The smoke must instantiate the Qwen3-VL training forward path with:

```text
attn_implementation = flash_attention_2
one physical packed row
two logical segments
explicit cu_seq_lens_q/k
4-row Qwen3 position_ids after prepare_forward_inputs
full logits
rebased teacher-forcing IR
no logits_to_keep
no real padding mask
no cache/past_key_values
```

- [ ] Mark the test skipped in automated CI unless CUDA, FlashAttention v2, and a usable tiny/local model fixture are available.
- [ ] A skipped smoke is not acceptance. If this test skips, the implementation status remains "not production eligible" and no production-scale throughput claim is allowed.
- [ ] The manual evidence must record package versions, fixture/model id, exact command, whether the test skipped, and the non-skipped result.
- [ ] Record the manual command in the final implementation report.

Run as the manual launch gate when hardware/model fixture is available:

```bash
python -m pytest tests/training/packing/test_segment_aware_fa2_smoke.py -s -rs
```

### Task 10: Promote Stable Docs and OpenSpec After Evidence

- [ ] Keep current stable docs/specs fail-fast language unchanged until CPU parity and tiny FA2 smoke evidence exist.
- [ ] Open an OpenSpec change before declaring segment-aware teacher-forcing packing supported under `training.packing: true`.
- [ ] Update the relevant OpenSpec contracts for static-only support, `packed_segment_map`, full-logits requirement, explicit FA2 varlen boundaries, sidecar fail-closed exceptions, loss denominator semantics, and metric/artifact deltas.
- [ ] Update `docs/data/PACKING.md` to state the V1 segment-aware teacher-forcing contract with evidence scope.
- [ ] Update `docs/IMPLEMENTATION_MAP.md` with the new packing module, tests, and entrypoints.
- [ ] Keep `progress/directions/2026-06-07_segment_aware_packing_infra.md` as the decision/provenance note.
- [ ] Update `docs/standards/UPSTREAM.md` only after the FA2/Qwen3 local preflight and tiny smoke are reverified in the implementation environment.

Run:

First validate the concrete OpenSpec change id created for this promotion:

```bash
openspec validate segment-aware-packing-v1 --strict
```

Then run the active test bundle:

```bash
python -m pytest \
  tests/training/packing/test_segment_map.py \
  tests/training/packing/test_teacher_forcing_rebase.py \
  tests/training/packing/test_segment_aware_collator.py \
  tests/training/packing/test_segment_aware_fail_closed.py \
  tests/training/packing/test_segment_aware_objective_parity.py \
  tests/training/packing/test_segment_aware_metrics.py \
  tests/test_model_input_bundle_contract.py \
  tests/test_batch_extras_contract.py \
  tests/test_train_batch_contract.py \
  tests/test_teacher_forcing_sidecar_bridge.py \
  tests/test_trainer_loss_bridge_qwen3vl_contract.py \
  tests/test_detection_training_config_contract.py \
  tests/detection/coverage/test_config.py \
  tests/test_training_config_strict_unknown_keys.py \
  tests/test_packing_attention_backend_gate.py \
  tests/test_aggregate_token_metric_events.py \
  tests/test_trainer_metrics_payload_contract.py \
  tests/test_training_architecture_golden_thread.py
python -m pytest tests/training/packing/test_segment_aware_fa2_smoke.py -s -rs
```

## Implementation Order

1. Pure packing data structures and validation.
2. Pure teacher-forcing IR rebase plus logical sample ownership map.
3. Batch extras and model-input contract registry.
4. Config/runtime V1 capability gate for static Qwen3-VL FA2 packing.
5. Collator integration for one planned packed row and raw component sidecar snapshotting.
6. FA2 boundary, padding-free/no-cache, Qwen3 position, and full-logits validation.
7. Unsupported-surface fail-closed tests.
8. Asymmetric objective parity.
9. Observability and logical-throughput metrics.
10. Non-skipped tiny Qwen3-VL FA2 smoke.
11. Stable docs and OpenSpec promotion after evidence.

## Review Checklist Before Production Training

- [ ] `training.effective_batch_size` is still physical packed-sequence budget.
- [ ] `training.packing: true` is the only normal user-facing switch.
- [ ] Segment correctness, not same-base-image grouping, is the architecture invariant.
- [ ] `per_device_train_batch_size == 1` and one planned pack-list per collator call are enforced for V1.
- [ ] `PackedSegmentMap` is built from raw packed component rows plus final tensors, not from `dataset_segments`.
- [ ] `packed_segment_map` is never forwarded into the model.
- [ ] `teacher_forcing_target_ir` remains one-per-logical-segment and is rebased before objective code.
- [ ] Distinct logical `sample_id`s map to the correct physical row.
- [ ] `ObjectiveRunner` and `TeacherForcingObjective` remain packing-unaware.
- [ ] Explicit `cu_seq_lens_*` are present and validated.
- [ ] No materialized padding mask, cache path, or `logits_to_keep` reaches the active V1 forward.
- [ ] Qwen3 4-row position IDs are validated.
- [ ] `logits_to_keep` is rejected for active segment-aware sidecars.
- [ ] Row coverage remains fail-closed.
- [ ] Mixed profiles remain fail-closed.
- [ ] CPU parity passes.
- [ ] Tiny real Qwen3-VL FA2 smoke is non-skipped and passes before production-scale claims.
