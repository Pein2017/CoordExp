# Stage-1 Static Packing Exact Atom-Position Mapping Audit

**Reviewed artifact:** `docs/superpowers/plans/2026-06-25-stage1-static-packing-exact-atom-position-mapping.md`

**Mode:** docs/spec/plan review-convergence loop.

**Mutation scope:** docs-only. No code/config implementation was started.

**Corrected user contract:** each original training sample has exactly one image
and one GT assistant response. Stage-1 static packing may concatenate multiple
such samples into one long sequence and one model forward, so a packed forward
may contain multiple images. This should behave like batch-like loss
accumulation over packed segments, not like a single sample containing multiple
images or a video. It must also preserve segment-aware attention isolation:
packed samples must not attend across segment boundaries, and loss/sidecar maps
must agree with the same boundary contract.

**Final docs/spec verdict:** review-hold findings were accepted into the
roadmap. Implementation remains gated until the user explicitly approves
code/config changes, and the first implementation gate must prove static-packed
attention isolation before sidecar remapping proceeds.

## Review Lanes

- Main loop: corrected the original mistaken "one image per forward" audit
  interpretation, routed through `coordexp-router-context`, and revised the
  plan in docs/spec mode.
- Lovelace: Qwen/media capture and coverage-ledger bridge semantics.
- Aristotle: config/schema/runtime packing boundaries and topology.
- Linnaeus: tests, metrics, provenance shifting, and launch/preflight gates.

## Rejected Prior Finding

The previous blocking finding that "multiple images in one forward violates
single-image input" is rejected as a false interpretation. The intended
constraint is one image per original sample. Multiple one-image samples in one
packed forward are allowed and are the reason this feature exists.

## Accepted Findings And Revisions

### Resolved: packed Qwen media validation must support multiple one-image samples

Evidence:

- Current `src/training/coverage_ledger/qwen_capture.py::_validate_image_grid_thw_v0`
  rejects anything except `image_grid_thw.shape == (1, 3)`.
- Qwen-style packed multimodal batches can carry multiple image-grid rows, but
  this feature must still reject videos, multi-frame rows, and multi-image
  original samples.

Revision:

- The plan now requires `CoverageLedgerForwardCapture` to accept
  `(num_segments, 3)` only when every row has `T == 1`.
- It keeps `pixel_values_videos`, `video_grid_thw`, missing grids, malformed
  grids, and sidecar/grid count mismatch rejected.
- It requires fake-Qwen capture tests for the positive packed multi-image case
  and the negative video/multiframe/malformed cases.

### Resolved: production bridge path must set `packing_enabled`

Evidence:

- `TeacherForcingObjectiveMixin` currently constructs
  `TrainerLossBridgeSettings` with `coverage_ledger=...` only.
- `prepare_forward_inputs` enforces Qwen 4-row packed `position_ids` only when
  `packing_enabled=True`.

Revision:

- The plan now derives `packing_enabled` from
  `extras.packed_segment_offsets is not None`, not from YAML alone.
- It adds production mixin tests proving malformed packed Qwen metadata fails
  through the actual `TeacherForcingObjectiveMixin.compute_loss` path and valid
  4-row metadata is accepted.

### Resolved: static packing must preserve attention isolation, not only loss ownership

Evidence:

- `docs/data/PACKING.md` states that packed/padding-free runs using
  FlashAttention should materialize `cu_seq_lens_q`, `cu_seq_lens_k`,
  `max_length_q`, and `max_length_k` and should not rely on a plain 2D
  `attention_mask` to represent multiple packed examples inside one row.
- `src/trainers/metrics/batch_contract.py` already validates cumulative
  sequence boundaries against `input_ids`, `text_position_ids` reset points,
  and `pack_num_samples` when `cu_seq_lens_q` is present.
- `src/sft.py::_validate_attention_backend_for_packing` requires
  `model.attn_impl` to be a FlashAttention backend when `training.packing=true`.

Revision:

- The plan now says a packed row must not behave like one natural sequence with
  cross-sample attention.
- The collator/forward-prep alignment test must verify varlen boundaries when
  present, or explicitly validate the Qwen packed `position_ids`/`text_position_ids`
  boundary-inference path against `packed_segment_offsets`.
- Exact atom-position mapping is framed as the loss/sidecar layer on top of the
  standard segment-isolated packed forward contract.

### Resolved: visual-token offsets have one owner

Evidence:

- The earlier draft put `visual_token_start=0` on every `PackedSegmentOffset`,
  while the bridge also computed cumulative visual starts from sidecar order.

Revision:

- `PackedSegmentOffset` is now text/sample-local only.
- The plan explicitly forbids storing `visual_token_start` there.
- Visual starts are computed bridge-locally from validated sidecar and
  `image_grid_thw` order.
- A bridge test now must use distinguishable visual slices so a missing
  `offset_visual_token_region` call cannot pass.

### Resolved: provenance rewriting must not shift vocabulary token ids

Evidence:

- The original plan included `bbox_positive_area_valid_token_ids` and
  `bbox_positive_area_invalid_token_ids` in the shift whitelist.
- These values are vocabulary ids, not sequence positions.

Revision:

- The plan removes token-id fields from `_POSITION_KEYS`.
- It adds a deny rule for `*_token_ids`, `stop_token_id`, and
  `continuation_token_ids`.
- It adds test expectations that geometry-valid token-id payloads remain
  unchanged after packed position rewriting.

### Resolved: coverage-ledger packed metrics must aggregate per forward

Evidence:

- `teacher_forcing/loss/coverage_ledger_auxiliary/contribution` is documented
  as a `last` reducer reporting the exact scalar added to forward loss.
- Raw concatenation of segment metric events could log only the final segment's
  contribution.

Revision:

- The plan now forbids raw concatenation for `last` reducer events.
- It now also rejects raw summation of segment-local mean losses as the default
  training scale. Packed coverage-ledger contribution must be normalized to the
  same semantic scale as the unpacked objective, and the logged contribution
  must equal the exact scalar added to forward loss.
- It requires a packed-vs-unpacked equivalence test, plus metric tests where
  diagnostic counts and weighted means/ratios reduce by documented
  `MetricEvent` semantics.

### Resolved: second review found missing config/preflight/bridge contracts

Evidence:

- `_detection_validate_teacher_forcing_coverage_ledger_contract` independently
  rejected `training.packing`, `training.eval_packing`, and
  `packing.static_packing` for coverage-ledger configs.
- `run_coverage_ledger_preflight` currently iterates unpacked dataset rows and
  cannot prove the packed alignment contract it claims to gate.
- `TrainerLossBridge.compute_loss` already receives `batch_extras`, but the
  earlier plan only derived a boolean `packing_enabled` and did not validate
  actual packed segment offsets against sidecar/media order.

Revision:

- The roadmap now names both schema guards and allows
  `training.eval_packing=true` for the standard SFT eval-step path, guarded by a
  regression test proving eval reaches the same packed `compute_loss` bridge.
- It requires the bridge to consume `batch_extras.packed_segment_offsets` and
  validate count, sample id, segment index, sidecar order, and image-grid row
  order before accepting multi-sidecar coverage-ledger inputs.
- It adds preflight code/tests to materialize or fixture a deterministic
  two-segment static pack and persist packed offsets, shifted labels,
  image-grid order, placeholder counts, Qwen position metadata, and attention
  boundary evidence.

### Resolved: packed sidecar validation remains strict

Evidence:

- Current unpacked `CoverageLedgerSidecarEnricher` validates unsupported row
  sidecar fields before aggregation.
- The draft packed branch originally extracted only ledger payloads and could
  silently drop other row-sidecar data.

Revision:

- The packed branch now reuses `_validate_aggregatable_row_sidecars`.
- For this first implementation, packed coverage-ledger rows intentionally
  reject non-ledger `supervision.payloads` rather than preserving arbitrary
  payloads.
- Tests must cover unsupported diagnostics/dataset/stage2 row sidecar fields
  and additional non-ledger payload rejection.

### Resolved: helper API snippets match current `SupervisionBatch.spans`

Evidence:

- Current `_build_teacher_forcing_supervision` returns
  `SupervisionBatch(spans=...)`, not `.atoms`.
- Existing `_with_batch_index` would overwrite already-shifted packed IR batch
  indices.

Revision:

- The plan now asserts through `SupervisionBatch.spans`.
- The implementation sketch appends `SupervisionSpan` values that wrap the
  already shifted IR and explicitly avoids `_with_batch_index` for packed IRs.

### Resolved: schema/runtime guard changes are scoped

Evidence:

- The draft schema text risked broad generic teacher-forcing packing enablement.
- The Stage-2 rejection test targeted a helper whose signature and ownership did
  not match the intended Stage-2 preflight boundary.

Revision:

- The plan now says not to replace the generic guard with a bare return.
- It loosens only the exact Stage-1 detection
  `objective.id=research_teacher_forcing` static-packing route.
- It keeps `padding_free_packed` rejected, Stage-2 self-rollout rejected, and
  existing Stage-2 trainer-owned post-rollout packing separate.
- It routes Stage-2 teacher-forcing rejection through runtime preflight and
  keeps a positive test for normal Stage-2 rollout-correction trainer packing.

### Resolved: encoded sample cache strictness is decided

Decision:

- Authored `training.encoded_sample_cache.enabled=true` is rejected for this
  packed latest-teacher-forcing feature, even when another runtime path would
  bypass an ineligible cache.

Revision:

- The plan records the authored-fail-fast rule and adds a bypass-negative test.

### Resolved: launch gate is no longer config-only

Evidence:

- Config loading cannot prove packed media/order/sidecar alignment.
- Existing coverage-ledger launch planning requires preflight artifacts and
  overlay checks before interpreting smoke results.

Revision:

- The final gate keeps the config topology probe but adds a no-training packed
  coverage-ledger preflight.
- The gate must materialize at least one deterministic two-segment static pack,
  record packed offsets, shifted label positions, `image_grid_thw` order,
  placeholder counts, Qwen 4-row `position_ids`, `ledger/selected_samples.json`,
  `ledger/alignment_debug.jsonl`, and `ledger/overlays/index.json`.
- Smoke and production training remain separate user approval gates.

## Confirmed OK Checks

- The corrected contract is now explicit in both plan and review: one image plus
  one GT assistant response per original sample; multiple such samples/images
  may be packed into one forward.
- Current `docs/data/PACKING.md` disables latest compact detection packing
  because sidecar/target-position rewriting is missing, not because
  multi-image packed forwards are categorically invalid.
- Sidecar-local image identity is the right boundary:
  `CoverageLedgerObjectEntry.image_index == 0` remains local to each sidecar.
- Keeping explicit `packing.padding_free_packed=true` rejected while validating
  the existing static-packing template metadata path is the right distinction.
- Stage-2 self-rollout remains out of scope; existing Stage-2 trainer-owned
  post-rollout packing remains separate.

## Remaining Gates

- User final approval is required before code/config implementation starts.
- After implementation, targeted pytest, config materialization, no-training
  packed preflight, and later smoke training approval are separate gates.

## Next State

Ready for user approval. Do not implement until the user explicitly says to
start implementation.
