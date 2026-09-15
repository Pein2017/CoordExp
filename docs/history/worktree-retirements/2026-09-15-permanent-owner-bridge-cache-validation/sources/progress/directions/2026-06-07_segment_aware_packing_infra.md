# Segment-Aware Packing Infra Purpose

## 0. One-sentence summary

Design future Stage-1-style detection teacher-forcing and row-coverage packing
around **segment-correct packing**: physical packed sequences keep the old
`global_max_length` budget semantics, while segment metadata makes target,
label, visual-token, row-state, sample-id, and loss ownership remapping exact.

Evidence scope: `none-yet`. This note records an architecture decision and
purpose statement from a grill discussion. It is not an implementation claim,
stable config contract, or throughput benchmark.

Approval gate: this architecture freeze authorizes implementation planning
only. Production code changes, stable-doc promotion, OpenSpec promotion, and
production training require explicit user/owner approval after parallel review
findings are triaged. Phase 1 scope is limited to teacher-forcing static packing
remap unless a later approved plan expands it.

## 1. Purpose

The purpose is to maximize training throughput for compact detection
teacher-forcing and row-conditioned visual coverage without weakening the
research meaning of the objective.

The current safe runtime is unpacked. It rejects packing because teacher-forcing
target atoms and row-coverage sidecars are not yet remapped into packed physical
sequences. That guardrail is correct for the current implementation, but the
future target should not be "same-base-image packing." The future target should
be segment-correct packing.

## 2. Core semantic invariant

The invariant is:

```text
packing is valid iff every logical supervision segment remains exact after
placement into a physical packed sequence
```

Each segment must carry enough metadata to answer:

```text
which logical sample or row state owns this segment?
where are this segment's text tokens in the physical sequence?
where are this segment's supervised label positions?
where are this segment's visual tokens or image-feature rows?
which teacher-forcing atoms belong to this segment?
which row-coverage state belongs to this segment?
which loss normalization rule applies to this segment?
```

Same-base-image-only packing may be useful as a debug mode or first validation
profile, but it must not be a semantic assumption in the architecture.

## 3. Budget accounting decision

Future segment-aware packing should preserve the old packed-training accounting
routine:

```text
training.effective_batch_size = physical packed-sequence budget
```

In packed mode, the optimizer-step budget counts accumulated long packed
sequences capped by `global_max_length` / `template.max_length`. It does not
directly count logical row states, base images, target atoms, or segment count.

Conceptually:

```text
one optimizer update
= N physical packed sequences accumulated across devices/grad-accum
= each physical sequence <= global_max_length
```

Segment-aware packing changes what can safely live inside each physical packed
sequence:

```text
pack = segment_0 + segment_1 + ... + segment_n
```

The physical pack remains the budget unit. Logical segment density is measured
as throughput and equivalence metadata, not used as the gradient-accumulation
source of truth.

V1 forward staging rule:

```text
one model forward / per-rank microbatch = one physical packed sequence
one optimizer update = many physical packed sequences via devices + accumulation
```

V1 should allow many logical segments inside one long physical row, capped by
`global_max_length` / `template.max_length`, but should not yet put multiple
physical packed rows into the same model forward. This keeps the first
coordinate remap simple:

```text
segment-local position -> physical row 0 absolute position
```

while preserving the old physical-budget semantics through distributed devices
and gradient accumulation. The effective batch still counts accumulated
physical packed sequences, not logical segments.

Multi-physical-row forwards are a V2 optimization. They will need explicit
`physical_row_index`, row-local and flattened offsets, per-row `cu_seq_lens`
agreement, padding-free output reshape/revert tests, and loss-side parity tests
before they can be treated as equivalent.

## 4. Required segment map

The future packer should promote segment metadata to the authoritative runtime
map, not merely cache/provenance data.

The agreed runtime owner is a first-class `packed_segment_map` batch extra,
produced by the packing/collator bridge and consumed by teacher-forcing
supervision plus row-coverage visual-state code. It should not be hidden inside
`TeacherForcingTargetIR`, and it should not overload existing
`dataset_segments` or `pack_num_samples` fields.

`TeacherForcingTargetIR` should remain a logical/local supervision object. The
packed bridge is the only layer allowed to convert segment-local positions into
physical packed positions.

Minimum segment fields:

```text
pack_id
segment_id
sample_id
row_state_id or none
base_image_id
text_token_range
label_range
visual_token_range
image_grid_range or image_feature_range
target_position_delta
logit_position_delta
row_coverage_state or reference
loss_normalization_scope
```

For teacher-forcing atoms:

```text
packed_target_position = segment.text_start + local_target_position
packed_logit_position  = packed_target_position - 1
packed_batch_index     = physical packed row index
```

If a future padding-free or cu-seqlens path resets sequence coordinates per
segment, the segment map must encode that coordinate system explicitly instead
of assuming one global contiguous row.

## 5. Remapping boundary decision

The resolved architecture boundary is:

```text
packer / collator
  -> builds and validates packed_segment_map
  -> transports it as BatchExtras sidecar

teacher-forcing supervision bridge / TrainerLossBridge
  -> consumes packed_segment_map
  -> converts logical/local target IR and row-coverage state into physical rows

ObjectiveRunner / objective modules
  -> consume already-physical SupervisionBatch and LabelLogitRowMap
  -> never reinterpret packing metadata

model(**inputs)
  -> receives only upstream-compatible model tensors
```

Component-sample `TeacherForcingTargetIR` starts logical/local. The packed
bridge materializes the physical view: batch indices, target positions, shifted
logit positions, span label positions, sample-id ownership, and any row-coverage
visual descriptor ranges. When packing is accepted, the collator/bridge should
attach the rebased physical target IR under the existing
`teacher_forcing_target_ir` key, plus the compact `packed_segment_map` as
sidecar/debug metadata.

The stable loss input is the rebased physical IR. `TrainerLossBridge`,
`ObjectiveRunner`, and objective modules should not consume both local and
physical coordinate systems, and they should not branch on packing mode. If local
IR provenance is needed for debugging, keep it in construction scope or emit a
bounded summary in the failure artifact rather than forwarding a second payload
through the loss stack.

Do not put `packed_segment_map` in `ObjectiveSpec.config`, and do not make
`TeacherForcingObjective` responsible for segment remapping. Objective code
should keep validating that already-remapped atoms match `input_ids`, labels,
and resolved causal logit rows.

This boundary matches the current typed training shape: the trainer mixin builds
`SupervisionBatch`, `TrainerLossBridge` strips non-model sidecars and constructs
bridge-level coordinates before `ObjectiveRunner`, and objective modules compute
loss over resolved rows. Putting remap inside the objective would couple loss
math to packing policy, upstream padding-free metadata, and visual-sidecar
ownership.

Evidence handles for the boundary are:

```text
src/trainers/metrics/teacher_forcing.py::_build_teacher_forcing_supervision
src/training/bridge/loss_bridge.py::TrainerLossBridge.compute_loss
src/training/encoding/model_inputs.py::ModelInputBundle
src/training/objectives/runner.py::ObjectiveRunner.run
src/training/objectives/teacher_forcing.py::TeacherForcingObjective
```

For upstream compatibility, `packed_segment_map` must remain a CoordExp sidecar.
It must never be forwarded to Qwen/ms-swift/HF model calls. Segment-map
validation must agree with the physical tensors and loss-side tensors it
indexes, including `input_ids`, labels when present, `position_ids` /
`cu_seq_lens_*` when present, and `image_grid_thw` / `pixel_values` visual
ordering.

The bridge should fail closed when:

```text
sample_id ownership is ambiguous
target/logit positions do not match input_ids and labels after rebasing
visual_token_range does not match the physical image_grid_thw split
row_coverage_state.sample_id does not match its segment owner
multiple row states for the same base image share one physical visual range
logits_to_keep would slice away any supervised physical logit row
packed_segment_map appears in forwarded model kwargs
```

V1 row-coverage visual ownership rule:

```text
one row-coverage segment
= one physical visual occurrence
= one row_coverage_state
= one visual_token_range / image_grid_range owner
```

The same `base_image_id` may appear multiple times in one physical pack, but
each row-coverage appearance must own a distinct physical visual occurrence.
`base_image_id` is for metrics, diagnostics, and future deduplication analysis;
it must not imply shared visual-token ownership in V1.

The rejected V1 alternative is sharing one visual occurrence across multiple row
states from the same base image. That would make only one coverage descriptor
semantically correct while allowing tensor shapes to pass. A later shared-image
optimization would need a separate mechanism that applies row-state descriptors
per segment after shared visual encoding, with explicit tests for that contract.

V1 packing-mode scope:

```text
segment-aware packing V1 supports static dataset packing only
dynamic packing remains out of scope for V1
```

This inherits the current Stage-1 guardrail in `docs/data/PACKING.md`: dataset
packing requires `training.packing_mode: static`, while dynamic Stage-1 packing
is deprecated/unsupported and fails fast. Static packing gives V1 deterministic
pack plans, cache fingerprints, DDP alignment, equivalence fixtures, and exact
sidecar replay. Dynamic packing can be revisited only after
`packed_segment_map` equivalence, row-coverage visual ownership, and loss
normalization are validated under static plans.

V1 pack-selection objective:

```text
primary objective = fill physical token length under global_max_length
not an atom-density, image-diversity, or row-state-diversity optimizer
```

This should preserve the original Stage-1/static-packing behavior. The packer
should choose groups by physical length fit under `global_max_length` /
`template.max_length`; segment-aware packing adds correctness metadata and
validation for sidecars, not a new sampling policy.

Logical density counters such as `supervised_atoms_per_pack`,
`base_images_per_pack`, `row_states_per_pack`, and
`supervised_label_tokens_per_second` are required observability metrics, but
they are not V1 packing objectives. Optimizing for supervision density or image
diversity would change research exposure at the same time as the coordinate
remap, making parity and throughput interpretation harder.

Future V2 policies such as `supervision_dense`, `base_image_diverse`, or
`row_state_balanced` would be real sampling-policy changes. They need explicit
config, metrics, and interpretation notes before being compared to the standard
Stage-1-style length-fill baseline.

V1 segment-map source of truth:

```text
static pack plan = membership only
encoded component samples = local tensor/sidecar truth
collator bridge = packed_segment_map construction
post-collation checks = final physical truth
```

V1 should reuse the existing Stage-1/static length-based pack planner as the
membership engine. The planner chooses which encoded samples share one physical
pack by length-fill under `global_max_length`; it should not become the semantic
owner of sidecars, row states, objectives, supervision density, or physical
target ranges.

Physical ranges depend on encoded `input_ids`, labels, visual placeholder
expansion, image ordering, Qwen position metadata, and ms-swift padding-free
flattening. Therefore `packed_segment_map` must be built from encoded component
samples during collation and validated against the final collated tensors before
teacher-forcing supervision or row-coverage visual descriptors are remapped.

This stage should be deliberately cautious. V1 must fail closed instead of
inferring missing ranges from raw dataset records, length-cache entries, or pack
plan lengths. A map derived only from cached lengths can be shape-compatible but
semantically wrong when template expansion, special tokens, image placeholders,
or sidecar ownership change the final physical layout.

Implementation should keep the code reusable and concise: centralize segment
map construction, validation, and remap helpers in a small CoordExp-owned module
or bridge layer, rather than scattering one-off offset arithmetic through
objectives, enrichers, metrics, or upstream wrappers. Replacing the pack planner
is out of scope for V1 because it would add cache semantics, DDP reproducibility
risk, and sampling-policy drift before packed-vs-unpacked parity is proven.

Reusable V1 module boundary:

```text
small typed PackedSegmentMap module
pure construction / validation / remap helpers
thin collator bridge
```

The reusable boundary should be a small typed module, for example
`src/training/packing/segment_map.py`, with names still flexible. It should own
types such as `PackedSegmentMap`, `PackedSegment`, `VisualOccurrenceRange`, and
`SegmentValidationIssue`, plus pure helpers such as
`build_packed_segment_map`, `validate_packed_segment_map`,
`rebase_teacher_forcing_ir`, `derive_or_validate_fa_varlen_boundaries`, and
`summarize_packing_throughput`.

The collator bridge should stay thin: call the base ms-swift/template collator,
build and validate the typed segment map, attach it as a sidecar, attach rebased
teacher-forcing targets when supported, and leave model-forward keys
upstream-compatible. ObjectiveRunner and objectives remain packing-unaware;
metrics and debug artifacts read the typed map instead of recomputing offsets.

`PackedSegmentMap` should stay compact. It should store explicit ownership and
range facts such as `pack_id`, physical batch row, physical sequence length,
segment ids, sample ids, base-image ids, supervision profile, text and label
ranges, local-to-physical offsets, visual occurrence / image-grid ranges, and
optional row-state ids.

It should not store full `input_ids`, labels, `pixel_values`,
teacher-forcing payload copies, row-coverage payload copies, logits, attention
tensors, or other large/derived payloads. Validation can inspect those tensors
and sidecars while building the map, and failure artifacts can emit compact
summaries. The map itself is a coordinate and ownership index, not a second
batch object.

V1 rollout order:

```text
phase 1: teacher-forcing segment remap only
phase 2: row-coverage visual descriptor remap
phase 3: combined tiny training smoke with throughput counters
```

Teacher-forcing remap should be validated first because it already covers the
core segment map invariants: target positions, shifted logit positions,
physical batch row, sample-id ownership, and exact `input_ids` / labels
matching. Row coverage adds an independent visual ownership surface:
`row_coverage_state`, `visual_token_range`, `image_grid_thw` split, descriptor
painting, and same-base-image repeated occurrences. Enabling both at once would
make equivalence failures harder to localize.

Phase 1 acceptance does not imply row-coverage support. A packed
teacher-forcing surface may be production-eligible while packed row coverage
still fails closed until the visual occurrence and descriptor-remap fixtures pass
in phase 2.

V1 user-facing config default:

```text
do not add a separate public opt-in knob for segment-aware sidecar remap
training.packing: true remains the user-facing packing switch
```

The default behavior should be simple for users. After V1 is implemented and
validated for a supported surface, enabling `training.packing: true` should use
the segment-aware sidecar remap automatically when teacher-forcing or
row-coverage sidecars are present. Before that implementation exists for a
surface, the current fail-fast behavior remains the default safety behavior.

Internal development gates, debug modes, and validation rungs may exist in tests
or private harnesses, but they should not become extra stable YAML knobs unless
a real user-facing choice emerges. This keeps config authoring simple while
still requiring the code path itself to prove `packed_segment_map` correctness
before packed sidecars are accepted.

V1 supported-surface registry:

```text
training.packing: true
+ supported model family
+ supported objective/supervision profile
+ static packing
+ explicit packed_segment_map
+ explicit cu_seq_lens_q/k and max_length_q/k
+ full logits
+ homogeneous supervision profile
=> segment-aware remap allowed
```

The support check should be an internal capability validator, not a new public
YAML switch. Unsupported combinations should fail closed with issue codes and
reasons, not silently fall back to partial behavior. This preserves simple user
configuration while making implementation coverage explicit.

Initial V1 support should be narrow and evidence-gated. A Qwen3-VL-style model
with `flash_attention_2`, static teacher-forcing sidecars, explicit FA varlen
boundaries, full logits, and one physical packed row per forward is the intended
first supported surface. Qwen2/Qwen2.5, row coverage, multi-physical-row
forwards, dynamic packing, `logits_to_keep`, and mixed supervision profiles
should become supported only after their own parity fixtures and tiny FA2 smoke
tests pass.

V1 homogeneous supervision profile:

```text
one physical pack = one supervision profile
```

For V1, every component sample inside a physical pack should use the same active
supervision profile. Supported V1 examples are "all segments carry
`teacher_forcing_target_ir`" and, after phase 2, "all segments carry
`teacher_forcing_target_ir` plus `row_coverage_state`". V1 should not mix
teacher-forcing sidecar segments with ordinary SFT-only segments, or mix
unrelated objective families inside the same physical pack.

This is a correctness and diagnosability rule, not a same-base-image rule. The
segment map may still contain multiple base images or repeated base images, as
long as each segment has clear ownership and the same supervision/profile
contract. Mixed-profile packing can become a later optimization only after
explicit per-segment objective routing, atom ownership, loss ownership, and
observability are designed.

The Stage-2 residual-set path has historical examples of loss-side remapping,
but that is not the desired future boundary for the typed ObjectiveRunner path.
Future implementation should adapt or compare against that path only as a
parity reference.

## 6. Observability requirements

Physical budget compatibility is not enough. When segment-aware packing is
active, logical-throughput counters are mandatory training metrics, not optional
debug logs. Every such run must report the logical density that explains its
actual training exposure:

```text
physical_packed_sequences_per_update
avg_segments_per_pack
logical_segments_per_update
base_images_per_update
row_states_per_update
supervised_atoms_per_update
supervised_label_tokens_per_update
text_tokens_per_update
visual_tokens_per_update
pack_fill_ratio
padding_slack
```

For row-conditioned visual coverage, report both base-image and row-state
exposure. An `effective_batch_size` of 128 in unpacked row-state training is
not 128 base images; packed training must make the same distinction explicit.

These counters are part of the V1 trust contract. A packed run that records only
physical `effective_batch_size` is not interpretable against an unpacked Stage-1
run, because two runs with the same physical budget may see different numbers
of segments, row states, base images, supervised atoms, and visual tokens.

Throughput interpretation rule:

```text
primary win metric = correct logical supervision throughput
```

After objective parity passes, the primary packing benchmark should report
whether the system delivers more correct logical supervision per unit wall-clock
and memory budget. Primary metrics are `supervised_label_tokens_per_second`,
`supervised_atoms_per_second`, `logical_segments_per_second`,
`row_states_per_second` when row coverage is active, and `pack_fill_ratio` /
`padding_slack`.

Auxiliary metrics such as `physical_packed_sequences_per_second`, raw
`tokens_per_second`, `optimizer_steps_per_second`, and GPU memory use are still
useful, but they are not sufficient to claim that ideal packing wins. A packed
run that produces more long containers while reducing supervised exposure is
not a throughput win for the research objective.

Benchmark baseline rule:

```text
compare only semantically correct training paths
```

For sidecar-bearing teacher-forcing, ordinary static packing without segment
remap is not a competing correct algorithm. It can choose length-fill membership,
but it cannot preserve target/logit ownership, causal segment isolation, or
sidecar coordinates. Therefore it should not be used as an official throughput
baseline against segment-aware packing.

The required benchmark comparison is the current correct unpacked
teacher-forcing path versus the new correct segment-aware packed path. Standard
Stage-1 packing remains the membership/planner component that V1 reuses, and it
may be used for implementation profiling on correctness-equivalent no-sidecar
surfaces, but not as an algorithmic source of truth for sidecar surfaces.

Support-status reporting:

```text
segment_aware_packing_active
segment_aware_packing_supported_surface
unsupported_reason_codes
validated_packs
failed_packs
failure_issue_counts
```

When `training.packing: true` is the simple user-facing switch, run metadata must
make it clear whether segment-aware packing was actually active and which
support gate was applied. Production training should fail fast on invalid packs
instead of skipping and continuing, but preflight, smoke, and debug artifacts
should still report counts and reason codes so unsupported surfaces are
diagnosable.

Segment-map validation failures must raise explicit issues and reasons. A V1
failure should not be a generic "packing invalid" exception; it should name the
failed invariant and the segment or pack that failed.

On validation failure:

```text
raise fail-fast exception with issue list and reason codes
emit compact rank-0 debug artifact for the first N failing packs
include IDs, ranges, token ids, shape summaries, and validation reason
exclude full image tensors and large raw payloads
```

Minimum failure artifact fields:

```text
pack_id
segment_id
sample_id
row_state_id or none
base_image_id
issue_code
reason
local_target_position / local_logit_position when relevant
physical_target_position / physical_logit_position when relevant
input_token_id / label_token_id at checked positions when relevant
text_token_range
visual_token_range
image_grid_range
image_grid_thw_summary
position_boundary_summary or cu_seq_lens_summary
```

This artifact is for debugging trust failures, not for training evidence. It
should be bounded, rank-0 only, and compact enough to inspect without dumping
images or full tensors.

V1 full-logits requirement:

```text
segment-aware packing V1 requires full, unsliced logits in production
logits_to_keep / sliced logits remain rejected while packed sidecar remap is active
```

The production V1 path should perform only one coordinate remap:

```text
segment-local sidecar positions -> absolute physical packed positions
```

`logits_to_keep` adds a second coordinate remap:

```text
absolute physical packed causal row -> sliced logits row
```

Upstream support does not remove that risk. Qwen3-VL and Qwen2.5-VL can slice
hidden states before `lm_head`, and ms-swift has boolean-mask and integer-tail
`logits_to_keep` modes. Those modes change the output logits time axis and do
not return a CoordExp sidecar-aware projection map. `position_ids` and
`cu_seq_lens_*` describe packed sequence boundaries; they do not prove that
each supervised sidecar row is present exactly once in sliced logits.

Future support should be a separate V2 contract, not a V1 optimization. It
would need an explicit `LogitProjectionMap` or equivalent:

```text
physical_logit_position -> sliced_logit_row
```

and tests for integer-tail slicing, tensor/mask slicing, missing supervised
rows, duplicate projection rows, multi-segment packs, causal-shift alignment,
and packed-vs-full-logit objective parity.

## 7. FlashAttention-2 forward compatibility

Pre-analysis source: local `conda ms` environment on 2026-06-07:

```text
transformers==4.57.1
flash_attn==2.8.3
ms-swift==4.2.2
```

For V1, segment-aware packing must be compatible with the upstream
FlashAttention varlen contract used by Transformers and ms-swift. The
`packed_segment_map` remains a CoordExp sidecar for supervision remap,
validation, and debug artifacts. It must not be forwarded to
`model(**inputs)`, and it must not replace the upstream attention-boundary
tensors.

Local upstream observations:

- ms-swift static packing sets `template.padding_free = True`, and the SFT
  argument checks require a FlashAttention-family `attn_impl` when `packing` or
  `padding_free` is enabled.
- Transformers FlashAttention dispatch uses `flash_attn_varlen_func` when
  explicit `cu_seq_lens_q`, `cu_seq_lens_k`, `max_length_q`, and `max_length_k`
  are provided, or when packed `position_ids` reveal reset boundaries. Its own
  utility notes that preparing cumulative lengths at collator stage is ideal.
- Qwen3-VL treats 4-row `position_ids` specially: row 0 is the text position
  sequence used to build the causal mask and passed to attention layers; rows
  1-3 are multimodal RoPE geometry. Segment boundaries therefore belong to the
  row-0 text positions / `cu_seq_lens_*` contract, while visual geometry must
  remain correct in rows 1-3.
- Qwen2-VL, Qwen2.5-VL, and Qwen3-VL vision towers use `cu_seqlens` with
  non-causal FlashAttention for variable-length visual chunks. This is a
  separate image-grid correctness surface from language-token isolation.
- ms-swift adjusts `cu_seqlens` when `logits_to_keep` slices rows. That is
  another reason V1 should require full logits while segment-aware sidecar remap
  is active.

V1 compatibility rule:

```text
attention isolation is represented by upstream-compatible forward tensors
supervision ownership is represented by CoordExp sidecars
```

Concretely, V1 should pass only supported model-forward keys such as
`attention_mask`, `position_ids`, `cu_seq_lens_q`, `cu_seq_lens_k`,
`max_length_q`, and `max_length_k`. It should not invent a custom attention mask
path, patch upstream HF model files, or rely on `packed_segment_map` inside the
model forward. The sidecar map and FA varlen tensors must be cross-validated
before the forward pass.

V1 explicit varlen boundary rule:

```text
segment-aware packing active
=> collator must emit cu_seq_lens_q/k and max_length_q/k explicitly
```

Transformers can infer packed varlen boundaries from `position_ids` resets, but
that should be treated as an upstream fallback, not as the CoordExp V1 contract.
The packer/collator is the layer that sees physical pack membership,
`packed_segment_map`, `pack_num_samples`, row-state ownership, visual occurrence
ranges, and loss sidecars at the same time. It should therefore produce the
canonical FA varlen tensors directly, and validation should compare all other
boundary views against them.

Required agreement:

```text
text_position_ids reset points == cu_seq_lens_q[:-1]
Qwen3 position_ids row 0 reset points == cu_seq_lens_q[:-1]
packed_segment_map text ranges == adjacent cu_seq_lens_q intervals
pack_num_samples == len(cu_seq_lens_q) - 1 for each physical packed row
max_length_q/k == max(diff(cu_seq_lens_q/k))
```

If any of these disagree, the pack is invalid before model forward. This keeps
attention isolation and supervision ownership anchored to one checked boundary
map instead of hoping that upstream inference reconstructs the same boundaries.

V1 causal-isolation rule:

```text
one logical segment = one independent causal sequence
one cu_seq_lens interval = exactly one logical segment
no cross-segment attention, even for the same base image
```

The physical packed row is only a storage and throughput container. Attention
semantics must remain segment-local:

```text
physical row: [segment A tokens][segment B tokens][segment C tokens]
attention:    A only attends within A; B only within B; C only within C
```

This is mandatory for same-base-image repeated row states as well. Shared
`base_image_id` is not shared context; each segment may have different
`row_coverage_state`, descriptor painting, target atoms, and supervision
ownership. Cross-segment causal bleed would silently change the training
distribution and make packed-vs-unpacked parity invalid.

Therefore these four boundary views must agree:

```text
packed_segment_map text ranges
== cu_seq_lens intervals
== text_position_ids / Qwen3 row-0 reset intervals
== supervision remap ownership intervals
```

Implementation-time tests should cover:

```text
cu_seq_lens_q/k start at 0, end at physical seq_len, and are strictly increasing
max_length_q/k equal max segment length
text_position_ids reset points match cu_seq_lens starts
Qwen3 4-row position_ids keep row 0 for text boundaries and rows 1-3 for RoPE
pack_num_samples / packed_segment_map segment count matches cu_seq_lens-1
image_grid_thw count matches image placeholder tokens
vision occurrence ranges match image_grid_thw slices
logits_to_keep remains rejected when segment-aware remap is active
```

The first implementation pass can make most of these CPU/local contract tests.
Hardware-gated smoke tests should then run one tiny packed forward with
`attn_impl=flash_attention_2` and compare it against an unpacked/eager or
unpacked/FA2 parity fixture before any throughput benchmark.

## 8. Loss-scale rule

Loss scale must be controlled by explicit objective normalization, not by the
accidental number of logical segments that fit into a physical pack.

For V1, segment-aware packing preserves the current teacher-forcing atom
normalization:

```text
loss denominator
= supervised teacher-forcing atoms after physical remap
!= physical packed sequences
!= logical segments
!= row states
!= base images
```

This matches `TeacherForcingObjective`, which accumulates one loss term per
supervised atom and divides by the atom count. Packing may change throughput and
padding waste; it must not silently change objective weighting because more
segments fit inside one physical pack.

If future work wants per-image, per-row-state, or per-segment balancing, that
must be an explicit objective option with its own metrics and tests. It must not
be introduced as a side effect of packing.

## 9. Validation ladder

V1 parity gate:

```text
no throughput benchmark or production-scale training before packed-vs-unpacked
objective parity passes
```

Before throughput is interpreted, the same logical samples must be run unpacked
and packed, and the objective surface must match. The gate should assert parity
for supervised atom count, label token positions, shifted logit positions,
per-atom ownership, total objective loss within tolerance, and the debug
ownership map.

The positive fixture should include at least two logical segments in one
physical pack, non-terminal and terminal supervised atoms, distinct sample IDs,
and repeated-base-image cases once row coverage enters phase 2. Negative
fixtures should prove fail-closed behavior for a missing map, wrong owner,
mismatched token, sliced logits, and mixed supervision profile.

Phase 1 production-eligibility gate:

```text
CPU/unit objective parity fixtures
+ one real tiny Qwen3-VL flash_attention_2 smoke
```

The unit fixtures must prove `PackedSegmentMap` construction, teacher-forcing IR
rebasing, `input_ids` / labels / shifted-logit validation, `cu_seq_lens_q/k` and
text-position reset agreement, and fail-closed handling for missing maps, wrong
owners, mixed profiles, and sliced logits.

The real tiny smoke must exercise the actual upstream stack: Qwen3-VL-style
4-row `position_ids`, `attn_impl=flash_attention_2`, explicit FA varlen
boundaries, full unsliced logits, model-key stripping, `ObjectiveRunner` loss
parity against the unpacked fixture, and logical-throughput counters. Without
both the pure parity fixtures and this hardware-gated FA2 smoke, the
teacher-forcing packed surface is not production-eligible.

Use same-base-image-only packing as a validation rung, not as the design target:

```text
1. unpacked-vs-packed objective parity fixtures
2. same-base-image packed fixtures
3. mixed-image packed fixtures
4. row-coverage visual-range remap fixtures
5. tiny packed training smoke
6. throughput benchmark with logical exposure counters
```

Required positive tests should prove that mixed-image packing remaps:

```text
teacher-forcing atom positions
labels and shifted logits
sample_id to physical row / segment ownership
visual-token ranges
row_coverage_state ownership
terminal-state supervision
```

Required negative tests should fail on missing segment ranges, mismatched
sample IDs, invalid visual split sizes, and any atom whose remapped target token
does not match `input_ids` / `labels`.

V1 architecture freeze:

```text
freeze V1 as architecture-ready for implementation planning after the decisions above
remaining details move to implementation plan, tests, and smoke evidence
```

The V1 implementation target is now stable enough to plan: reuse the existing
Stage-1/static length-fill planner; add a compact typed `PackedSegmentMap`
bridge; support teacher-forcing sidecars first; require explicit FA2 varlen
boundaries, full logits, strict per-segment causal isolation, one physical
packed row per forward, and packed-vs-unpacked parity; compare only
semantically correct paths. New architecture branches should be opened only if
implementation or upstream-library evidence contradicts this contract.

## 10. Consequence

The future architecture should extend the current static packing machinery with
a segment-aware sidecar bridge instead of special-casing row coverage or
same-image batches.

Until that bridge is implemented and validated, the current fail-fast behavior
for teacher-forcing packing and row-coverage packing remains correct. Detailed
handles and troubleshooting live in companion pages.
