# Segment-Aware Packing V1 Review Log

Date: 2026-06-07
Status: draft review resolutions; implementation still requires final user approval

This log records the parallel review loop for the Segment-Aware Packing V1 design and implementation plan.

Primary docs:

- `docs/superpowers/specs/2026-06-07-segment-aware-packing-v1-design.md`
- `docs/superpowers/plans/2026-06-07-segment-aware-packing-v1.md`
- `progress/directions/2026-06-07_segment_aware_packing_infra.md`

## Review Agents

Four read-only review agents audited independent risk surfaces:

- Architecture/code-boundary review: existing owners, `PackedSegmentMap`, teacher-forcing supervision ownership, config/runtime gates.
- Upstream compatibility review: local `transformers`, FlashAttention v2, Qwen3-VL, and ms-swift packing behavior.
- Test/verification review: parity, fail-closed, model boundary, batch contract, metrics, and launch gates.
- Docs/governance review: approval gate, routing, stable-doc/OpenSpec promotion boundary, and provenance placement.

No reviewer found a fatal architecture contradiction. All reviewers found plan refinements that must be incorporated before implementation.

## Resolved P1 Findings

### Approval Gate

Finding: the frozen progress note used language that could be read as permission to begin production code changes.

Resolution:

- Added an explicit approval gate to `progress/directions/2026-06-07_segment_aware_packing_infra.md`.
- Updated the architecture-freeze wording to "architecture-ready for implementation planning".
- Added approval gates to the V1 spec and plan.

Implementation consequence: code changes, production training, stable-doc promotion, and OpenSpec promotion require explicit user/owner approval after review triage.

### Dedicated Planning Surface

Finding: segment-aware packing should not be folded into the row-conditioned visual coverage prototype, which is explicitly unpacked-only in V1.

Resolution:

- Created a dedicated design spec.
- Created a dedicated implementation plan.
- Kept the June 7 progress note as provenance/history rather than the task list.

Implementation consequence: Phase 1 is teacher-forcing static packing remap only. Row coverage remains Phase 2 and fail-closed.

### Routing

Finding: machine-readable routes did not include the June 7 segment-aware packing note or the new planning surfaces.

Resolution:

- Added the progress direction note to `progress/index.yaml`.
- Added the progress direction note, V1 spec, and V1 plan to `docs/catalog.yaml`.
- Kept the route status non-normative: `emerging-direction` and `draft-for-approval`.

Implementation consequence: future agents can find the planning docs without treating them as current supported behavior.

### Stable Docs and OpenSpec Boundary

Finding: supporting `training.packing: true` for teacher-forcing sidecars changes stable training/config/loss semantics and cannot be promoted silently.

Resolution:

- Added a Stable Docs and OpenSpec Boundary section to the spec.
- Updated Task 10 in the plan to require an OpenSpec change before declaring supported behavior.
- Left current stable docs/specs fail-fast behavior untouched during planning.

Implementation consequence: stable `docs/data/PACKING.md`, `docs/IMPLEMENTATION_MAP.md`, and OpenSpec promotion happen only after CPU parity and a non-skipped FA2 smoke.

### Segment Map Source of Truth

Finding: the existing `dataset_segments` and `pack_num_samples` extras are not enough to build semantically correct segment spans.

Resolution:

- Updated the spec to say `PackedSegmentMap` must be built from raw packed component rows plus final collated tensors.
- Updated the plan so `dataset_segments` is not a segment-span source.
- Added validation against final `input_ids`, labels, `cu_seq_lens_*`, `text_position_ids`, and `pack_num_samples`.

Implementation consequence: `pack_num_samples` is only a sanity count; component lengths must be captured before upstream packed-row collation loses local sidecar structure.

### Teacher-Forcing Ownership

Finding: a single physical-row-merged `TeacherForcingTargetIR` can make scalar loss run but risks losing logical sample ownership.

Resolution:

- Changed the contract to one rebased `TeacherForcingTargetIR` per logical segment.
- Added explicit `sample_id` length equal to logical segment count.
- Added `PackedSegmentMap.sample_id_to_physical_row`, where many logical sample ids map to physical row `0` in V1.
- Required the teacher-forcing supervision builder to avoid overwriting packed atom `batch_index` by logical enumeration.

Implementation consequence: `ObjectiveRunner` and `TeacherForcingObjective` should remain packing-unaware; they receive physical coordinates and a correct `sample_id_to_batch_index` map.

### Config and Runtime Enabling

Finding: current schema/runtime guardrails reject teacher-forcing `training.packing: true` before collator logic can run.

Resolution:

- Added `src/config/schema.py` and `src/detection/runtime.py` to the plan target files.
- Added a narrow V1 capability gate: teacher-forcing, static packing, Qwen3-VL, FlashAttention v2, padding-free, full logits, and `per_device_train_batch_size == 1`.
- Kept row coverage and other packed sidecar surfaces fail-closed.

Implementation consequence: the old blanket rejection becomes a supported-surface gate, not an unconditional unlock.

### ms-swift Planned-Pack Flattening

Finding: ms-swift can flatten multiple planned pack-lists into one physical padding-free row if per-device batch size is greater than one.

Resolution:

- Added hard V1 requirements for `per_device_train_batch_size == 1`, `input_ids.shape[0] == 1`, and exactly one planned pack-list per collator call.
- Added negative test requirements for multiple planned pack-lists in one collator call.

Implementation consequence: the V1 physical budget remains one planned static pack per per-rank forward.

### Sidecar Preservation Before Upstream Packing

Finding: upstream packing carries a narrow key set and may drop local semantic sidecars.

Resolution:

- Added a collator requirement to snapshot raw packed component rows before relying on the upstream packed output.
- Added positive tests requiring distinct sample ids and target IRs to survive into rebased sidecars.
- Added model-call spy requirements proving packing sidecars never reach model kwargs.

Implementation consequence: semantic ownership is recovered from raw component rows, then validated against final tensors.

### FlashAttention v2 Forward Contract

Finding: explicit FA2 varlen boundaries are only meaningful on the padding-free/no-cache path; padding masks or cache paths can bypass the intended segment-isolated varlen behavior.

Resolution:

- Added support gates requiring padding-free training prefill, no real padding mask, no cache/past, and explicit `cu_seq_lens_*`.
- Added `TrainerLossBridgeSettings(packing_enabled=True)` as the path that must run Qwen packed-position checks.
- Added Qwen3-specific tests for row-0 text boundary resets and rows 1-3 MRoPE/image-placeholder alignment.

Implementation consequence: V1 is not generic FA2; it is the explicit Qwen3-VL padding-free prefill varlen path.

### Full Logits

Finding: ms-swift can insert `logits_to_keep` and mutate labels/logit axes before CoordExp loss code sees the batch.

Resolution:

- Added early rejection for `args.use_logits_to_keep=True`.
- Added fail-fast if any `logits_to_keep` key remains after trainer input preparation when `packed_segment_map` is active.
- Kept bridge full-logits validation as a second line of defense.

Implementation consequence: V1 uses full sequence logits only.

### Active Fail-Closed Tests

Finding: one originally routed legacy recursive-detection test is globally skipped, so it cannot serve as a V1 verification anchor.

Resolution:

- Added `tests/training/packing/test_segment_aware_fail_closed.py` to the target test set.
- Routed fail-closed checks through active tests: sidecar bridge, batch extras, config/runtime, attention backend, batch contract, and model-input bundle.

Implementation consequence: unsupported states must fail with reason codes instead of silently falling back to standard wrong packing.

### Objective Parity

Finding: scalar loss parity alone can miss ownership or off-by-one bugs.

Resolution:

- Added an asymmetric golden fixture requirement with distinct sample ids, unique token/logit rows, segment-start atoms, segment-end atoms, and terminal STOP atoms.
- Required exact rebased atom-table assertions in addition to numerator/denominator parity.

Implementation consequence: parity must prove ownership and row mapping, not only a matching scalar.

### Metrics and Throughput

Finding: counters were listed but not tied tightly enough to the primary win semantics.

Resolution:

- Added candidate logical-throughput names: `supervised_label_tokens_per_second`, `supervised_atoms_per_second`, `logical_segments_per_second`, `pack_fill_ratio`, and `padding_slack`.
- Added metric-route tests for event payloads, per-update/per-second scopes, DDP/grad-accum aggregation, and capped failure records.

Implementation consequence: throughput claims must report supervised/logical exposure, not raw physical rows/sec alone.

### FA2 Smoke Acceptance

Finding: the tiny Qwen3-VL FA2 smoke could be skipped in CI and still appear absent from the final acceptance bundle.

Resolution:

- Made the tiny real Qwen3-VL FA2 smoke a hard manual launch gate.
- Added explicit language: skipped smoke is not acceptance and leaves the implementation not production-eligible.
- Added required evidence: package versions, fixture/model id, exact command, skip status, non-skipped result, explicit varlen kwargs, four-row Qwen3 position ids, full logits, and finite loss.

Implementation consequence: CPU tests can approve local mechanics, but production/training-throughput claims require non-skipped FA2 evidence.

## Remaining Approval Condition

The planning docs are ready for user review after the sanity checks pass. Real implementation must not start until the user gives final approval.
