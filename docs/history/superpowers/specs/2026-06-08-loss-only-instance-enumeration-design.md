# Loss-Only Instance Enumeration Design

Status: draft research design; not an OpenSpec contract.

Date: 2026-06-08

Branch: `codex/loss-only-instance-enumeration`

Fork base: `codex/row-conditioned-visual-coverage`

## Decision

Build the next object-enumeration prototype as a **loss-only explicit
instance-enumeration** mechanism.

The active path should return to standard Qwen3-VL image/text forwarding and
standard decode behavior:

```text
image + prompt + teacher-forced row prefix
-> standard Qwen3-VL forward
-> standard CE objective
-> training-only instance-enumeration auxiliary losses
```

The active path must not:

```text
modify raw pixels
modify visual features
modify Q/K/V
add attention-logit bias
require output_attentions
invalidate or complicate KV cache at inference
```

The current feature-level visual coverage painter remains useful as historical
evidence and legacy reference code, but it should be disabled or deprecated from
the active training/inference route for this branch.

## Why This Fork Base

This branch intentionally forks from `codex/row-conditioned-visual-coverage`
rather than `main`.

Reuse from the coverage branch only where it supports the V1a full-sequence
loss path:

- object-boundary update semantics;
- `TeacherForcingTargetIR` alignment to Swift-encoded token positions;
- sidecar and collator transport patterns;
- parser/rollout fixes that exposed duplicate bursts;
- visual-token lattice geometry helpers, after renaming or isolating them from
  feature painting semantics.

Do not keep as active behavior:

- dynamic feature residual painting;
- `RowCoverageFeatureTuner` as a training dependency;
- re-prefill paths whose purpose is only to apply changed visual features;
- pixel-painted coverage paths;
- feature-residual magnitude logging as a primary success metric.

The fork-base trade-off is deliberate. Starting from `main` would avoid legacy
coverage code, but it would also require reimplementing the target-alignment and
sidecar transport substrate, which is the part most likely to create silent
credit-assignment bugs.

Coverage painter isolation is an explicit branch rule. The new
`instance_enumeration_aux` path may reuse geometry ideas after moving or
renaming them into neutral helpers, but it must not call painter wrappers,
feature tuners, pixel-painting code, or row re-prefill paths. In this worktree,
coverage painter code should remain isolated legacy/reference code, not an
active dependency of the loss-only implementation.

## Core Hypothesis

Regular CE is too implicit for autoregressive VLM object enumeration:

```text
given image + prefix, predict the next token
```

CE can reward the correct serialized object row without explicitly forcing the
row representation to bind to the visual region that row describes.

The new hypothesis is:

```text
For object row k:
  the row anchor should bind to object bbox k;
  the row anchor should prefer bbox k over previously enumerated boxes;
  after all objects are enumerated, ordinary assistant-stop CE should remain
  the first stop signal.
```

This should test whether better training-time credit assignment can teach the
model to enumerate distinct instances without dynamic visual-state mutation or
runtime attention surgery.

## Terminology

Use **instance-enumeration auxiliary loss** for this branch.

Avoid calling this the "visual coverage painter" or "attention proxy" in new
config and logs. The loss is probe-based and training-only. It may use previous
object regions as negatives, but it does not paint features or directly change
attention.

Where the discussion says "proxy loss", interpret it narrowly as:

```text
a training-only probe loss intended to shape row-to-region credit assignment
```

not as:

```text
a runtime attention-control mechanism
a guarantee that generation-time attention will be suppressed
a replacement metric for rollout duplicate rate
```

## Existing Integration Points

### Teacher-Forced Prefix Semantics

Current branch path:

- `src/detection/dataset.py::DetectionTrainingDataset.__getitem__`
- `src/detection/coverage/row_state_dataset.py::RowCoverageTrainingDataset`
- `src/detection/dataset.py::DetectionTrainingDataset.encode_teacher_forcing_row_state`
- `src/detection/dataset.py::DetectionTrainingDataset._row_coverage_state`
- `src/detection/teacher_forcing/target_builder.py::_build_atoms`

V1a has one teacher-forcing path:

```text
full-sequence auxiliary path:
  one standard teacher-forced sample per image
  all object anchors are present in one causal sequence
  previous objects for row k are derived from rendered object-row order
```

For a causal decoder, the hidden state at `<object_ref_start>` for row `k`
cannot attend to future rows.
If the auxiliary loss uses the anchor hidden state and constructs previous
masks from rendered objects before row `k`, full-sequence training remains
faithful to teacher-forced online prefix semantics while avoiding the cost of
row-state expansion.

For Stage-1 GT teacher-forcing, `<object_ref_start>` is mandatory. The GT
sequence is the canonical supervision source, so a missing or duplicated
object-start anchor is a prompt/template/IR alignment bug, not a row to drop.
V1 should fail fast for Stage-1 when rendered object rows do not have exactly
one aligned `<object_ref_start>` anchor each.

Do not implement, expose, or validate an explicit row-state dataset/wrapper in
V1a. The active loss should not depend on the old `row_coverage_state` name.

In full-sequence mode, "previous" must mean previous in rendered row order, not
numeric object id. The order source is:

```text
TeacherForcingTargetIR.metadata["selected_normalized_object_indices"]
```

For an object at rendered order position `r`:

```text
target object index = selected_normalized_object_indices[r]
previous object indices = selected_normalized_object_indices[:r]
future object indices = selected_normalized_object_indices[r + 1:]
```

Future object indices must never contribute to current or previous masks.

For the V1a random-vs-sorted ablation, the lane name must describe the actual
rendered target-IR order:

```text
random_order_sft:
  data.object_ordering = random_permutation
  objective.target_ir.rollin_policy.name = random_permutation

sorted_sft:
  data.object_ordering = sorted
  objective.target_ir.rollin_policy.name = sorted
  selected_normalized_object_indices is sorted/rendered order
```

Do not treat `instance_enumeration_aux.enabled=true` as a reason to keep
`target_ir.rollin_policy.name="random_permutation"` while labeling the run
sorted. The auxiliary loss consumes `selected_normalized_object_indices`, so a
random target-IR order would make the sorted ablation mislabeled.

### Target IR Alignment

Current branch path:

- `src/detection/dataset.py::_align_teacher_forcing_target_to_encoded`

`TeacherForcingTargetIR` atoms are shifted to Swift-encoded positions. After
alignment:

- `atom.target_position` is the encoded input position of the supervised token;
- `atom.logit_position` is `atom.target_position - 1`;
- `atom.provenance["object_index"]` identifies the source object;
- `atom.provenance["branch_position"]` identifies token position inside the
  object row.

This distinction is critical.

For the V1a grounding anchor, use the hidden state at the encoded
`<object_ref_start>` input position. Identify object anchors by object
provenance and branch position:

```text
anchor position = atom.target_position
where:
  atom.provenance["object_index"] == current target object index
  atom.provenance["branch_position"] == 0
```

Do not use `labels != -100` alone. Do not use the previous causal row
`atom.logit_position` as the anchor hidden state.

The current token role vocabulary labels both `<object_ref_start>` and
`<box_start>` as `TokenRole.SCHEMA`, so V1 must not identify the object anchor
only by token role. Use `branch_position == 0` plus object-row provenance.

Stage-1 anchor policy:

```text
expected anchors = TeacherForcingTargetIR.metadata["selected_normalized_object_indices"]
discovered anchors = branch_position == 0 atoms grouped by object_index

if any expected object has zero or multiple discovered anchors:
  raise a hard error before computing auxiliary loss

if any discovered anchor is outside expected rendered order:
  raise a hard error before computing auxiliary loss
```

The error should include sample id, rendered object order, missing object
indices, duplicate object indices, and aligned target positions where present.
Silent partial auxiliary supervision is not allowed in V1 Stage-1 because it
would make loss curves and monitoring easier than the real GT contract.

### Qwen3-VL Forward Constraints

Installed local Transformers source:

```text
/root/miniconda3/envs/ms/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py
```

Important upstream behavior:

- `Qwen3VLModel.get_image_features()` returns post-vision/projector
  `image_embeds` split by image and DeepStack visual embeddings.
- `Qwen3VLModel.forward()` concatenates `image_embeds`, validates placeholder
  counts, and scatters them into `inputs_embeds` at positions where
  `input_ids == image_token_id`.
- `Qwen3VLModel.forward()` calls `language_model(...)` and returns
  `Qwen3VLModelOutputWithPast(last_hidden_state=...)`.
- `Qwen3VLForConditionalGeneration.forward()` calls `self.model(...)`, takes
  `hidden_states = outputs[0]`, computes logits through `lm_head`, and returns
  `Qwen3VLCausalLMOutputWithPast`.
- Although `Qwen3VLCausalLMOutputWithPast` has optional `hidden_states` and
  `attentions` fields, the current forward path does not populate them.

Therefore V1 must not assume:

```python
outputs.hidden_states is not None
outputs.attentions is not None
```

V1 should also avoid `output_attentions=True` and avoid attention-backend
patching. The loss needs only:

- final sequence hidden states, preferably captured at the input of `lm_head`;
- original post-projector image embeddings, captured from `get_image_features`
  without modifying them;
- `input_ids`, `image_grid_thw`, target IR, and full-sequence scene metadata.

V1 also requires full sequence logits/hidden states. The existing
`TrainerLossBridge` already rejects `logits_to_keep` projection and validates
that logits preserve the `input_ids` time dimension; the instance-enumeration
capture should inherit that contract and additionally validate that captured
`lm_head` input hidden states have the same `[batch, seq]` prefix as
`input_ids`. This protects object anchors that may occur far before the final
token window.

The visual-key source for V1 is the post-projector `image_embeds` returned by
`get_image_features`, before Qwen scatters those vectors into `inputs_embeds`.
The scattered `inputs_embeds` image-token positions are an alignment check, not
the primary loss source. Final LLM hidden states at visual-token positions are
also not V1 visual keys.

V1 uses only the main post-projector `image_embeds` as visual keys. It must not
include Qwen DeepStack visual embeddings in `L_pos` or `L_neg`. DeepStack
features are injected separately into the text model and have layer-specific
semantics, so adding them to the key population would break the simple
probability-mass interpretation over image-placeholder-aligned visual tokens.
They may be captured for shape/count debugging only.

Independent validation of the upstream Qwen3-VL path confirms this as the least
invasive V1 source: `get_image_features()` is the boundary where the visual
branch has produced LLM-ready image embeddings, while `masked_scatter` is input
assembly. Capturing this tensor avoids touching raw pixels, Q/K/V, attention
backends, KV cache, or decode semantics. The intentional caveat is that
`detach_visual_keys=true` prevents gradients flowing into the visual/projector
path through the captured key tensor; V1 trains the row-anchor side and the tiny
probe.

Capture precision policy:

```text
captured hidden states: keep native model dtype/device
captured post-projector image embeddings: keep native model dtype/device
probe linear projections: normal training dtype/autocast
visual region masks: fp32 on the score device
q/k normalization, score matmul, score scaling: fp32
log_softmax, logsumexp, log/softplus loss math: fp32
```

Do not force the full Qwen forward, captured tensors, or probe projection
layers into fp32. The fp32 boundary should be local to the small score and
probability-mass computation where numerical stability matters. Do not change
global autocast/default dtype settings to implement this auxiliary loss.

No upstream Qwen3-VL files may be edited.

## Target Architecture

Add a new detection-local module family:

```text
src/detection/instance_enumeration/
  config.py
  types.py
  state.py
  visual_regions.py
  capture.py
  loss.py
```

The module should own the new terminology and avoid extending the
feature-painter API.

`types.py` should define the canonical geometry sidecar type as:

```text
Norm1000XYXYBox = tuple[int, int, int, int]
```

with `x1, y1, x2, y2` in the repo-default pre-normalized `0..999` coordinate
space. Pixel-space floats are derived values and must not be stored in row
state.

`types.py` should define `InstanceEnumerationState` as a frozen dataclass with
tuple fields. The sidecar is a CPU/Python metadata object, not a tensor
container; do not store torch tensors, projected features, masks, or
device-specific values in it.

Use explicit debug serialization helpers, for example `to_debug_dict()` or
`to_jsonable()`, only for logging and error messages. Do not use plain dicts as
the primary runtime representation of the loss-critical sidecar.

V1a should not introduce a row-expanded dataset wrapper. It should attach
full-sequence `InstanceEnumerationState` to the standard detection
teacher-forcing samples.

### Training Data State

Introduce a new sidecar or metadata adapter:

```python
InstanceEnumerationState
```

Required fields for full-sequence V1:

```text
sample_id
base_idx
ordered_object_indices: tuple[int, ...]
object_boxes_norm1000_xyxy_by_index: mapping[int, Norm1000XYXYBox]
image_width
image_height
ordering_strategy
ordering_seed
```

Rows are derived views, not stored sidecar data. Boxes live once in
`object_boxes_norm1000_xyxy_by_index`; the sidecar must not duplicate
`target_box`, `previous_boxes`, or stored row objects. The mask builder resolves
derived row indices through that mapping and fails if any required index lacks a
box. The sidecar mapping stores frozen integer norm1000 xyxy boxes only;
conversion to pixel-space floats happens inside `visual_regions.py` when
constructing visual-token masks.

The loss-critical sidecar is geometry-and-order only. It must not include
object labels, descriptions, category names, or decoded text as required inputs
to `loss.py`. Textual object semantics stay in the ordinary teacher-forced CE
path. Optional label/description metadata may be added later for monitoring,
but it must not be required to compute V1 `L_pos` or `L_neg`.

The loss or a tiny helper derives row views from `ordered_object_indices`:

```text
for r, target_object_index in enumerate(ordered_object_indices):
    previous_object_indices = ordered_object_indices[:r]
    future_object_indices = ordered_object_indices[r + 1:]
```

Rows do not store anchor token positions in V1a. The loss resolves each current
anchor from the aligned `TeacherForcingTargetIR` using the derived
`target_object_index` and `branch_position == 0`, then reads
`atom.target_position` as the encoded input position.

Tests should include a nonmonotonic order such as `[2, 0, 1]` and assert that
derived rows are `(2, ())`, `(0, (2,))`, and `(1, (2, 0))`. This keeps
random-vs-sorted ordering honest without storing duplicate row state.

Do not infer the current target box from labels, decoded text, or parsed
assistant payloads. The dataset already has the scene object geometry; put the
object boxes in `object_boxes_norm1000_xyxy_by_index` explicitly and derive
current/previous masks from row indices plus that mapping.

### Dataset Wrapping

For V1, do not add a row-state-expanded dataset wrapper. Add a dataset-owned
builder to the standard detection teacher-forcing dataset. It should attach
`instance_enumeration_state`, not `row_coverage_state`, as the active sidecar
for the new loss.

Ownership rule:

```text
dataset / teacher-forcing IR path:
  constructs InstanceEnumerationState
  validates rendered order, anchors, boxes, image size, and sample ids

collator / batch extras path:
  transports already-constructed instance_enumeration_state sidecars
  requires sidecar presence for every unpacked aux-enabled sample
  rejects packed sidecars until V1b segment-offset support exists

loss path:
  validates sidecar-vs-batch consistency
  derives row views from sidecar order and consumes sidecar boxes
  does not reconstruct object order or boxes from text/token heuristics
```

The dataset path is the semantic owner because it already has the canonical
scene, rendered object order, aligned `TeacherForcingTargetIR`, normalized
object geometry, image size, sample id, and ordering seed. The collator should
behave like the existing teacher-forcing sidecar enrichers: preserve and group
metadata, not invent it.

Full-sequence V1 must validate that:

- `TeacherForcingTargetIR.metadata["selected_normalized_object_indices"]`
  exists;
- every object anchor atom's `object_index` appears exactly once in that order;
- `InstanceEnumerationState.ordered_object_indices` exactly equals that metadata;
- derived target/previous/future indices are consistent with
  `ordered_object_indices`;
- previous masks are derived from earlier order positions, not numeric ids;
- future rendered objects are excluded from previous masks;
- all selected object indices have sidecar boxes.

The aligned `TeacherForcingTargetIR` is the eligibility gate for auxiliary
supervision:

```text
eligible object rows =
  object_index values with exactly one aligned atom where branch_position == 0
```

In normal full-sequence V1, every rendered object should be eligible:

```text
eligible object indices == selected_normalized_object_indices
```

Default V1 should fail closed if this equality does not hold. Silent partial
auxiliary supervision would make the experiment difficult to interpret.

For Stage-1 GT, fail-closed is mandatory rather than advisory. Do not add a
drop-invalid fallback, partial-row mode, or row-state wrapper to V1a.

### Packing Compatibility

V1a auxiliary training is unpacked full-sequence Stage-1 only. Standard static
SFT packing may continue for runs that do not enable `instance_enumeration_aux`,
but aux-enabled batches must fail closed when packed sidecars appear.

Current codebase fact:

```text
StaticPackedCaptionDataset.__getitem__ returns list[dict]
TeacherForcingTargetIREnricher rejects packed sidecars until
target-position offsets are preserved
```

Keep one active V1a mode:

```text
unpacked_strict:
  required first smoke path
  one batch row = one detection sample = one image
```

The auxiliary sidecars are not offset-safe yet, so V1a should not carry partial
packed-batch logic or packed offset repair.

### Forward Capture

The active Qwen forward should still run once through the standard
`Qwen3VLForConditionalGeneration.forward()` path.

Use a scoped, non-mutating capture context:

```text
InstanceEnumerationForwardCapture
```

It should collect:

- `lm_head_input_hidden_states`: the tensor passed into `lm_head`, shape
  `[batch, seq, hidden]`;
- `image_embeds_by_sample`: primary unpacked loss input, containing
  post-projector visual tokens returned by `get_image_features`, split in the
  same sample/image order used by Qwen;
- `flat_image_embeds` and `visual_token_offsets_by_sample`: validation-only
  views for count checks and diagnostics;
- optional DeepStack shape metadata for validation/debugging only, not for
  probe keys or loss computation;
- optional shape metadata for validation and logging.

The split structure is the semantic source for the loss. A flattened tensor is
useful for validation because Qwen concatenates image features before
`masked_scatter`, but the loss must not rely on hand-computed flat offsets as
its primary data model.

Allowed implementation approaches:

- a temporary forward hook on `core_model.lm_head` to capture its input;
- a temporary wrapper around `core_model.model.get_image_features` that records
  returned tensors and then returns them unchanged.
- a resolver that mirrors or reuses the existing coverage wrapper traversal over
  `.module`, `.base_model`, and `.model` to find the actually invoked forward
  model, `lm_head` owner, and Qwen visual owner under PEFT/ms-swift wrappers.

Forbidden implementation approaches:

- editing upstream Qwen3-VL files;
- modifying visual embeddings before scatter;
- using scattered `inputs_embeds` as the primary visual-key source;
- using final visual-token hidden states as V1 visual keys;
- replacing `Qwen3VLTextAttention`;
- relying on `output_attentions=True`;
- relying on `outputs.hidden_states` from `Qwen3VLForConditionalGeneration`;
- deriving packed segment mappings from token text heuristics instead of
  collator/dataset metadata.

The capture context must restore original methods/hooks even if forward raises.

### Loss Placement

The existing `ObjectiveRunner` computes CE-style objective losses from logits.
The new auxiliary loss requires hidden states and visual features, so V1 should
not force it into the current logits-only objective interface.

Preferred V1 placement:

```text
TrainerLossBridge.compute_loss()
  -> prepare inputs
  -> run standard model forward inside capture context if aux enabled
  -> extract logits
  -> run existing ObjectiveRunner for CE
  -> run InstanceEnumerationAuxLoss with capture artifacts and sidecars
  -> total_loss = objective_result.loss + aux_result.weighted_loss
```

The bridge result should expose auxiliary metric events or a small metrics map
without pretending the loss came from the CE objective runner.

Longer-term, the objective interface can be generalized to accept auxiliary
forward artifacts, but V1 should minimize blast radius.

### PEFT / Adapter Lifecycle

Local library versions inspected for V1:

```text
peft 0.17.1
transformers 4.57.1
trl 0.23.1
```

CoordExp already has a related `modules_to_save` pattern for
`coord_offset_adapter` in `src/sft.py` and `src/tokens/row_offsets.py`. The
instance-enumeration probe should follow the same adapter-topology discipline,
with one important difference: the probe must be used by the loss directly, so
the loss must resolve the active post-PEFT copy.

PEFT behavior confirmed from the installed local source:

```text
peft.utils.other.ModulesToSaveWrapper:
  deep-copies modules_to_save into modules_to_save[adapter_name]
  freezes the original module
  trains the active copied module

peft.utils.save_and_load.get_peft_model_state_dict:
  walks AuxiliaryTrainingWrapper modules
  includes modules_to_save tensors in the PEFT adapter state dict
```

Therefore the V1 lifecycle should be:

```text
before sft.prepare_model / PEFT wrapping:
  attach InstanceEnumerationProbe at model.instance_enumeration_probe
  append "instance_enumeration_probe" to train_args.modules_to_save

after sft.prepare_model / PEFT wrapping:
  resolve the active probe from ModulesToSaveWrapper.modules_to_save[active]
  store that active probe handle in a small training context
  fail if the resolver returns the frozen original module

during forward/loss:
  standard PEFT-wrapped model forward computes anchor hidden states
  anchor hidden states remain attached to the graph
  visual keys are detached
  active registered probe computes scores/loss

backward:
  gradients flow into the active probe parameters
  gradients also flow through anchor hidden states into trainable LoRA/adapters

checkpoint/resume:
  PEFT adapter checkpoints include probe tensors through modules_to_save
  resume from an aux-enabled checkpoint must restore probe state

final inference/export:
  generation must not require the probe
  V1a rejects adapters that still declare or contain instance_enumeration_probe
  deployable strip/export tooling is out of scope until explicitly approved
```

Optimizer grouping:

```text
VIT / vision tower:
  use training.vit_lr

projector / aligner:
  use training.aligner_lr

LLM / LoRA / remaining trainable text-side params:
  use training.learning_rate

instance_enumeration_probe:
  use instance_enumeration_aux.probe_lr
  use an explicit probe bucket so it is not dropped by multimodal grouping

all optimizer groups in V1 instance-enumeration runs:
  use weight_decay = 0.0
```

The probe should not default to `vit_lr` because visual keys are detached and
the loss must not update the vision tower through the key path. It should not
default to `aligner_lr` because it is not part of the projector or visual-to-LLM
aligner. Its LR should be an explicit scalar, not a ratio. Recommended V1a
starting value is equal to the configured LLM/adapter `training.learning_rate`,
but the config should spell out the value directly so experiment records remain
unambiguous after inheritance resolution.

When `instance_enumeration_aux.enabled=true`, `probe_lr` is required:

```text
probe_lr missing -> config error
probe_lr is null -> config error
probe_lr <= 0 -> config error
```

Do not silently derive the probe LR from `training.learning_rate` during config
resolution. Matching the LLM/adapter LR is a recommended V1a value, but it must
be written explicitly in the resolved config.

Local source caveat: ms-swift's plain `multimodal` optimizer groups only
vision, aligner, and language-model prefix parameters. A PEFT
`modules_to_save` probe may not match any of those prefixes. V1 must therefore
extend the CoordExp optimizer path to place `instance_enumeration_probe`
parameters exactly once, with LR `probe_lr` and weight decay `0.0`.

Do not set weight decay on any V1 instance-enumeration optimizer group:
vision, aligner, LLM/LoRA, coord-offset modules if present, and
`instance_enumeration_probe` all use `weight_decay=0.0`. This keeps the first
readout focused on CE plus the auxiliary objective rather than interactions
between small auxiliary probes, LoRA weights, and regularization.

Do not instantiate the probe inside `TrainerLossBridge`. The bridge is created
inside the loss path and is not a stable model owner; a probe there would not be
seen reliably by the optimizer or checkpointing code.

Local toy evidence scope:

```text
toy PEFT model + modules_to_save["instance_enumeration_probe"]:
  active probe params require_grad=True under modules_to_save.default
  original probe params require_grad=False
  PEFT state dict contains probe q/k tensors
  synthetic aux loss from PEFT-wrapped hidden state gives nonzero probe grad
  synthetic aux loss gives nonzero grad on at least one LoRA parameter
```

This toy check proves the adapter mechanics, not Qwen3-VL task quality.

Artifact-mode policy:

```text
training_aux_adapter:
  may contain instance_enumeration_probe in modules_to_save and adapter tensors
  must restore probe tensors for aux-enabled resume
  missing requested probe tensors are a resume error

inference_adapter/export:
  must not contain instance_enumeration_probe config or tensors
  V1a inference checkpoint resolution fails fast if a training-only probe is present
  no stripped/exported adapter claim is made by V1a
```

This keeps the first implementation training-correct without adding an
under-tested export transformer. A future deployment task can add explicit strip
tooling and tests once the loss-only mechanism is worth keeping.

### Mandatory Qwen-Path Gradient Smoke Gate

Before any real V1 training run is trusted, run a tiny Qwen3-VL teacher-forced
batch with `instance_enumeration_aux.enabled=true` and valid object rows. This
is a wiring trust gate, not a quality or mAP test.

The smoke must fail if any of these assertions fail:

```text
active PEFT ModulesToSave probe copy is resolved, not the frozen original
aux loss is finite
aux loss is nonzero when valid rows exist
probe q/k parameters receive nonzero gradients
at least one trainable adapter/LoRA parameter receives nonzero aux-path gradient
post-projector visual key source receives no aux gradient when detach_visual_keys=true
captured lm_head-input hidden state has [batch, seq] matching input_ids
captured image features match image-token placeholder counts
captured image features match image_grid_thw-derived post-merge token counts
```

The smoke should isolate the auxiliary route enough to prove participation. If
the implementation measures gradients after the combined CE+aux backward, also
include an aux-only backward or equivalent hook-based check so nonzero adapter
gradients cannot be attributed to CE alone.

Do not interpret this smoke as evidence that the objective improves rollout. It
only proves that the active Qwen/PEFT training path is capable of applying the
auxiliary signal to the intended trainable modules while keeping visual keys
detached.

## Loss Formulation

### Inputs Per Object Row

For each object-row sample:

```text
h_k = final hidden state at <object_ref_start> input position
z_j = original post-projector visual token feature j
m_cur_j = support-style loss mask for current bbox
m_prev_j = max-union support-style loss mask for previous boxes
m_neg_j = m_prev_j * (1 - m_cur_j)
```

`h_k` is taken from final hidden states after the standard causal LLM forward.
Because the model is causal, the anchor hidden state at `<object_ref_start>`
may see the prompt, previous rows, and the anchor token itself, but not the
future desc/bbox tokens in the current row.

Do not use span pooling in V1. Pooling over the teacher-forced object span can
see current desc and coordinate tokens, making the auxiliary loss much easier
and less meaningful as a test of row-anchor binding.

### Probe Scores

Use a lightweight probe:

```python
q_k = normalize_or_project(h_k)
k_j = normalize_or_project(z_j)
score_kj = q_k @ k_j / sqrt(d)
```

V1 requirements:

```text
detach_visual_keys = true
detach_anchor_hidden = false
probe = one shared trainable linear q + trainable linear k
d_probe = small fixed hidden_dim
probe sharing = shared across rows, samples, and steps
per-layer probes = disabled
per-head probes = disabled
row-conditioned probe parameters = disabled
MLP probe = disabled
normalize q/k = true
score_dtype = fp32
probability_dtype = fp32
```

Rationale:

- detaching visual keys prevents the auxiliary loss from rewriting visual
  memory or projector features as a shortcut, so `detach_visual_keys=true` is
  mandatory in V1 rather than merely a default;
- keeping anchor hidden states attached lets the loss shape the row
  representation and trainable adapters/token rows;
- a tiny trainable probe avoids assuming final text-decoder hidden states and
  post-projector image embeddings are already dot-product comparable;
- running probe projections in normal training dtype keeps compute cost local,
  while casting projected q/k and masks to fp32 for normalization, scoring, and
  probability-mass losses keeps the numerically sensitive path stable;
- a small `d_probe`, normalization, and no MLP reduce the chance that the probe
  alone solves the auxiliary task while generation behavior remains unchanged.
- sharing a single probe across all object rows and samples keeps V1 focused on
  whether row-anchor representations bind to visual regions. Per-layer,
  per-head, or row-conditioned probes are later experiments after the shared
  route proves wiring and rollout value.

V1a uses a fixed positive `score_scale` over normalized cosine probe logits.
The default is `1.0`, preserving the unsharpened baseline; larger values are an
explicit hyperparameter, not a trainable temperature. This keeps score-scale
experiments visible in config while avoiding Q/K/V edits or trainable attention
temperature.

### Positive Region Binding

Use region-mass binding as the V1 positive loss:

```python
log_p_j = log_softmax(score_scale * score_kj over visual tokens)
log_mass_cur = logsumexp(log_p_j + masked_log(m_cur_j))
L_pos = -log_mass_cur
```

Here `masked_log(0) = -inf`; `eps` must not add fake probability mass to
zero-mask visual tokens.

This is intentionally weaker than a full KL target over bbox interior tokens.
It asks the row anchor to put probability mass inside the current object
region without forcing a uniform or overly precise distribution over all
tokens inside the bbox.

Full soft KL can be a later experiment:

```python
target_j = m_cur_j / sum(m_cur)
L_kl = KL(target || p)
```

but it is not the first claim path because bbox interiors are coarse and may
include background, occluders, or multiple overlapping instances.

### Previous-Region Exclusion

Use an overlap-aware probability-mass margin:

```python
log_p_j = log_softmax(score_scale * score_kj over visual tokens)
log_mass_cur = logsumexp(log_p_j + masked_log(m_cur_j))
log_mass_neg = logsumexp(log_p_j + masked_log(m_neg_j))
L_neg = softplus(margin + log_mass_neg - log_mass_cur)
```

with:

```python
m_prev_j = 0.0 if no previous objects else max(m_prev_object_masks[:, j])
m_neg_j = clamp(m_prev_j * (1.0 - m_cur_j), 0.0, 1.0)
```

Previous-object masks are unioned with `max`, not summed. Each previous object
gets its own raw/support mask first; the previous-region mask is the per-token
maximum across those objects, then clamped to `[0, 1]`. This keeps `L_neg`
about whether a visual token was previously covered, not how many previous
boxes overlap that token.

Conceptually, `L_pos` and `L_neg` both operate on the same normalized
visual-token distribution. The margin is a log-ratio margin:

```text
mass_cur / mass_neg > exp(margin)
```

This is preferred over a raw score-margin in V1 because the monitored values
have direct semantics:

```text
current-region probability mass
previous-region probability mass
current-vs-previous mass ratio
positive-vs-previous margin satisfaction rate
```

Use the stable `log_softmax + logsumexp` computation above. Do not implement
this as naive `softmax -> sum -> log`.

If there are no previous boxes or the negative mask has negligible total mass,
materialize `L_neg = 0` for that row for tensor-shape and metric bookkeeping,
but exclude it from the `sample_neg` and `batch_neg` reducers. Log the zero
denominator/count explicitly. This is normal for the first object row and for
near-total current/previous overlap cases. It is not analogous to an empty
current mask, which is a hard geometry error for valid Stage-1 GT objects.

Do not treat previous boxes as hard negatives where they strongly overlap the
current target. Overlap handling is part of the loss contract, not a later
metric tweak.

### Loss Aggregation

V1 uses sample-weighted auxiliary aggregation:

```text
sample_pos = mean(L_pos over valid object rows in sample)
sample_neg = mean(L_neg over rows with usable negative masks in sample)
batch_pos = mean(sample_pos over samples with at least one valid object row)
batch_neg = mean(sample_neg over samples with at least one usable negative row)
weighted_aux = pos_weight * batch_pos + neg_weight * batch_neg
```

Do not sum losses over rows in V1. A crowded image should provide more
row-level diagnostic examples, but it should not automatically dominate the
batch objective because it has many ground-truth objects.

For `L_neg`, rows with empty or tiny negative masks are excluded from
`sample_neg`; if a sample has no usable negative rows, that sample does not
contribute to `batch_neg`. This is normal for one-object samples and
near-total-overlap cases. It must still contribute to `batch_pos` when its
current masks are valid.

Log both sample-weighted and row-weighted summaries:

```text
loss values used for optimization = sample-weighted
row-weighted values = diagnostics only
```

This makes crowded-scene behavior inspectable without changing the V1 gradient
scale.

### Zero Valid Positive Rows

For Stage-1 V1a, an aux-enabled training batch must contain at least one valid
positive object row.

Behavior:

```text
batch valid positive row count == 0 -> hard error
```

The hard error should include batch/sample ids when available, sidecar presence,
rendered object counts, anchor counts, invalid/degenerate-box counts, and
geometry-error counts. A zero-row batch can indicate that the sidecar was not
attached, anchors failed, packing dropped metadata, or geometry filtering
silently removed all objects.

Do not add an empty-object policy to V1a. If a future dataset intentionally
contains empty-object images, that is a separate variant with separate tests and
interpretation.

Because score scale still affects the softmax distribution, V1 keeps the
following controls tied to the loss contract:

```text
q/k normalization
fixed positive score_scale from config
tiny linear probe
detach_visual_keys=true
small auxiliary weights
global LR scheduler warmup
```

V1a should not add a separate auxiliary-loss warmup. Use the existing global LR
scheduler warmup for all optimizer groups. This means CE and auxiliary gradients
are present from step zero, but the scheduler scales update magnitudes during
warmup:

```text
effective_vit_lr     = lr_scheduler_scale_t * training.vit_lr
effective_aligner_lr = lr_scheduler_scale_t * training.aligner_lr
effective_llm_lr     = lr_scheduler_scale_t * training.learning_rate
effective_probe_lr   = lr_scheduler_scale_t * instance_enumeration_aux.probe_lr
```

The first version should avoid an independent loss-ramp schedule because that
would add another interpretation axis before the probe, masks, and gradient
route are validated. If early instability appears, a future V1b
`aux_loss_schedule` may tie the auxiliary-loss scale to the same global LR
warmup progress rather than introducing a separate step count.

### Row Normalization

Within each sample reducer, each usable object row has equal weight. Do not
weight row losses by bbox area, token count, or mask mass:

```python
sample_pos = mean(L_pos over valid object rows in sample)
sample_neg = mean(L_neg over rows with usable negative masks in sample)
```

Each object row is one enumeration decision. A small object and a large object
should contribute equally in V1 once their masks are valid. Object-size,
mask-mass, or token-count weighting would bias the auxiliary signal toward
large regions and should be reserved for explicit later experiments. Across a
batch, V1 still uses the sample-weighted aggregation contract above.

### Terminal / Stop Rows

V1 should keep terminal supervision as ordinary assistant-stop CE.

Do not add EOS auxiliary loss in the first implementation. A stop-related
probe is underdefined because there is no single "current visual target" after
all objects are exhausted. Instead, log terminal-row CE/EOS behavior and
rollout early-stop/max-row behavior.

EOS auxiliary losses can be revisited after object-row binding is validated.

## Visual Token Region Masks

The loss needs bbox-to-visual-token soft masks. This should reuse the proven
geometry ideas from the coverage painter but be factored as region-mask logic,
not feature painting.

Inputs:

```text
image_grid_thw
model/processor spatial_merge_size
image width and height
norm1000 xyxy boxes
```

`visual_regions.py` owns conversion from `Norm1000XYXYBox` to pixel-space and
visual-token overlap weights, using the repo geometry convention such as
`src.datasets.geometry.norm1000_xyxy_to_pixel_xyxy`. This keeps image-size
dependent calculations out of sidecar construction and makes mask diagnostics
the source of truth for geometry expansion behavior.

The visual-token lattice source of truth is Qwen's actual `image_grid_thw`
combined with the model/processor `spatial_merge_size`. Image width and height
are used only to convert norm1000 boxes into pixel-space coordinates. Do not
reconstruct the visual grid from image dimensions alone, because that would
silently assume processor resizing, patching, and merge details that may drift
from the actual post-projector image embeddings.

The coordinate-frame source of truth for `Norm1000XYXYBox` conversion is the
dataset scene's stored image width and height: the same original-image frame in
which the repo's norm1000 boxes are defined. This relies on the active training
path keeping `do_resize=false`. If a future config enables processor/template
resizing or any image transform after box normalization, V1 must fail until an
explicit box-to-processed-image transform is designed and tested.

Keep these two frames separate:

```text
box coordinate frame:
  dataset scene original image width/height

visual token lattice frame:
  Qwen image_grid_thw plus spatial_merge_size
```

V1 assumes the existing detection dataset invariant of one image per sample:

```text
one detection scene
one image per sample
one image_grid_thw row per sample
no video tokens
```

Batched unpacked training is supported, but each unpacked batch row must
correspond to exactly one image-backed detection scene. If a sample violates the
one-image invariant, treat it as a dataset/config error. Reject video samples
for `instance_enumeration_aux`.

Output per image:

```text
visual token footprints on post-merge lattice
current bbox raw overlap mask raw_cur in [0, 1]
previous-box max-union raw overlap mask raw_prev in [0, 1]
current bbox support loss mask m_cur in [0, 1]
previous-box support loss mask m_prev in [0, 1]
```

Rules:

- use the post-merge visual lattice length:
  `t * (h // merge_size) * (w // merge_size)`;
- use `image_grid_thw` as the authoritative source for `t`, `h`, and `w`;
- do not reconstruct the lattice from image width/height alone;
- require `t == 1` for V1 image detection samples;
- reject grids where `h` or `w` is not divisible by `merge_size`;
- reject mask length mismatches against captured post-projector image embeds;
- reject image-token count mismatches against `input_ids == image_token_id`;
- reject video token/features in V1;
- compute overlap as area fraction of each visual token footprint;
- keep raw area-fraction masks for geometry diagnostics and visualization;
- convert raw area-fraction masks into support-style loss masks before
  computing `mass_cur` or `mass_neg`;
- clamp all masks to `[0, 1]`;
- do not globally normalize masks before the loss formulas above;
- normalize only inside optional diagnostics that explicitly require a
  probability distribution.

V1 loss-mask conversion:

```python
raw_cur_j = area(token_j intersect current_bbox) / area(token_j)
raw_prev_object_o_j = area(token_j intersect previous_bbox_o) / area(token_j)
raw_prev_j = max(raw_prev_object_o_j over previous objects)

m_cur_j = clamp(raw_cur_j / support_floor, 0.0, 1.0)
m_prev_j = clamp(raw_prev_j / support_floor, 0.0, 1.0)
m_neg_j = clamp(m_prev_j * (1.0 - m_cur_j), 0.0, 1.0)
```

Do not sum previous-object masks. Summing would make crowded or overlapping
history more negative solely because several previous boxes cover the same
token, changing the meaning of `L_neg`. Log crowding/overlap separately instead
of baking it into V1 mask weights.

Default:

```yaml
region_masks:
  raw_overlap: area_fraction
  loss_weighting: support_clipped
  support_floor: 0.10  # experimental v1 default
  current_mask_min_sum: 1.0e-6
  negative_mask_min_sum: 1.0e-6
```

`support_floor=0.10` is an experimental V1 default, not a stable geometry
contract. Keep it fixed for the first run so mask semantics are inspectable
and comparable; revisit it only after logging shows whether it over-expands
large boxes or under-supports small objects.

Do not raise the V1a default to `0.25` for geometric neatness. Tiny-object
trainability is a first-run priority, and post-merge visual tokens can cover
much larger regions than tiny object boxes. If later evidence shows that `0.10`
over-expands support too aggressively, change `support_floor` as an explicit
hyperparameter update and record it as a new experiment, not as a V1a mechanism
mode.

Rationale:

Post-merge visual tokens are coarse. If a small object covers only a small
fraction of one visual token, raw fractional overlap would cap `mass_cur` even
when all probability is assigned to the correct token. Support-style masks use
raw overlap for geometry alignment but avoid making sub-token object size an
irreducible positive-loss penalty.

Compute `L_pos` for every valid Stage-1 GT object row. Do not skip rows merely
because an object is small. In V1a, an empty current support mask for a valid GT
object is a hard geometry error:

```python
if valid_stage1_gt_box and m_cur.sum() < current_mask_min_sum:
    raise GeometryMappingError
```

This condition indicates broken bbox normalization, image-size metadata,
`image_grid_thw` interpretation, spatial merge handling, visual-token footprint
math, or xyxy clipping. Skipping would hide exactly the geometry bug the loss
depends on.

Degenerate or invalid boxes should be excluded before
`InstanceEnumerationState` construction using the dataset's existing validity
contract, with counts logged separately. Future corrupted-prefix or rollout
modes may use skip semantics, but Stage-1 V1a should fail fast.

## Config Direction

Add a new top-level detection config section. The schema default is disabled:

```yaml
instance_enumeration_aux:
  enabled: false
```

A V1a experiment config uses this compact surface:

```yaml
instance_enumeration_aux:
  enabled: true
  version: v1
  probe_lr: 5.0e-5  # required explicit scalar; usually match training.learning_rate
  positive_weight: 0.05
  negative_weight: 0.02
  negative_margin: 0.5
  score_scale: 1.0
  support_floor: 0.10
```

Scalar training hyperparameters stay configurable. Structural choices are
hardcoded and validated, not exposed as user-facing toggles. Required
invariants:

```text
training path: full-sequence unpacked Stage-1 teacher forcing only
state owner: dataset / teacher-forcing IR path
collator: require all unpacked sidecars, reject packed sidecars
terminal/eos tokens: no aux loss
anchors: object_ref_start via aligned TeacherForcingTargetIR target_position
anchor hidden: final lm_head-input hidden, not detached
visual keys: main post-projector Qwen image_embeds only, detached
DeepStack keys: excluded from L_pos/L_neg
region masks: image_grid_thw lattice, dataset original image frame
previous masks: max union, then current-region overlap removal
probe: one shared linear q/k probe, hidden_dim=256, q/k normalized
precision: projection in model default dtype, score/probability math in fp32
score_scale: fixed positive scalar from config, default 1.0
aggregation: sample mean for optimization, row-weighted diagnostics only
aux schedule: no separate aux warmup; use the global LR scheduler
weight decay: 0.0 for every V1 instance-enumeration optimizer group
```

The resolved `instance_enumeration_aux` config must be propagated through the
detection runtime shim into `DetectionDatasetRuntimeConfig`; otherwise the
dataset cannot know when to attach `instance_enumeration_state`. Do not rely on
the obsolete `custom` bucket or on loss-time text reconstruction to decide
whether the sidecar should exist.

The detection config order validator must not treat `instance_enumeration_aux`
as a coverage-style exception for mismatched ordering. For V1a, the random and
sorted lanes must have matching dataset and target-IR order:

```text
data.object_ordering == objective.target_ir.rollin_policy.name
```

The implementation must add a minimal `sorted` teacher-forcing roll-in policy so
the sorted smoke config materializes sorted
`TeacherForcingTargetIR.metadata["selected_normalized_object_indices"]`. The old
coverage-painter exception for `data.object_ordering="sorted"` plus
`rollin_policy.name="random_permutation"` remains scoped to coverage painters
only and must not apply to this branch.

If a future experiment needs to change any invariant above, it should create a
new explicit variant after V1a is trusted, not add dormant toggles to the first
implementation.

Keep these separate:

```yaml
row_conditioned_visual_coverage:
  enabled: false
```

Do not enable both `row_conditioned_visual_coverage.enabled=true` and
`instance_enumeration_aux.enabled=true` in the first implementation. A later
experiment may combine feature painting and grounding, but the first loss-only
experiment should not mix mechanisms.

`custom` remains obsolete for detection configs, so this should be a strict
typed top-level section rather than a free-form `custom` bucket.

V1a hardcodes strict anchor coverage and unpacked auxiliary batches. Do not add
`anchor_policy`, `packing_policy`, or `padding_free_packed` controls to the
first implementation.

## Ablation Matrix

V1a has one objective:

```text
standard CE + L_pos + L_neg
```

The only V1a ablation axis is rendered object order:

```text
random_order_sft
sorted_sft
```

Both use the same objective, global LR scheduler, scheduler warmup, data scope,
checkpoint family, and metric schema. Do not add objective-control runs,
scheduler variants, auxiliary-warmup variants, painter variants, score-scale
sweeps, support-floor sweeps, or margin sweeps to the V1a matrix.

Default V1a hyperparameters:

```text
positive_weight = 0.05
negative_weight = 0.02
negative_margin = 0.5
support_floor = 0.10
probe_lr = 5.0e-5  # usually match training.learning_rate
```

`L_pos` is the row-to-current-region binding term. `L_neg` is the previous
region exclusion term. Their weights and `negative_margin` are scalar
hyperparameters for the fixed V1a objective, not ablation-mode switches.

The default `L_neg` margin is a log-ratio margin. `margin=0.5` asks for:

```text
mass_cur / mass_neg > exp(0.5) ~= 1.65
```

This keeps the current-region preference active without demanding the stronger
`exp(1.0) ~= 2.72` ratio before the mask geometry and overlap slices are
validated.

Random ordering is the production-relevant stress case when active configs use
random object rows. Sorted ordering is also required because it gives a
deterministic sanity lane for inspecting previous/current/future masks and
catching accidental numeric-id ordering.

For this worktree, keep the coverage painter ignored by active configs. The
loss-only route should use new `instance_enumeration_aux` names and logs; any
geometry code borrowed from painter work must be factored into a neutral module
before use.

## Logging and Health Gates

The ordinary trainer should keep logging total loss, CE loss, and optimizer
learning rates. V1a adds a compact default `instance_enum/` health surface; it
does not add logging-mode knobs to the public config.

Default training metrics:

```text
instance_enum/loss
instance_enum/loss_pos
instance_enum/loss_neg
instance_enum/weighted_loss_pos
instance_enum/weighted_loss_neg
instance_enum/valid_pos_row_count
instance_enum/valid_neg_row_count
instance_enum/skipped_neg_empty_rate
instance_enum/zero_valid_positive_batch_count
instance_enum/current_mask_geometry_error_count
instance_enum/invalid_or_degenerate_box_excluded_count
instance_enum/current_mask_sum
instance_enum/negative_mask_sum
instance_enum/current_mass
instance_enum/previous_mass
instance_enum/mass_ratio_cur_to_neg
instance_enum/margin_satisfied_rate
instance_enum/probe_entropy
instance_enum/probe_effective_tokens
instance_enum/probe_top1_in_current
instance_enum/probe_top5_current_mass
instance_enum/probe_grad_norm
instance_enum/adapter_grad_norm
instance_enum/adapter_nonzero_grad_param_count
instance_enum/aux_to_ce_loss_ratio
instance_enum/weighted_aux_to_total_loss_ratio
instance_enum/lr_vit
instance_enum/lr_aligner
instance_enum/lr_llm
instance_enum/lr_probe
```

Resolved run constants should be logged once with the config/artifact metadata:

```text
positive_weight
negative_weight
negative_margin
support_floor
probe_lr
probe_hidden_dim = 256
score_scale = 1.0
detach_visual_keys = true
do_resize = false
packed_auxiliary = false
visual_key_source = qwen_main_image_embeds
```

Debug-only metrics may be enabled for geometry or probe autopsies, but they are
not part of the public V1a config surface:

```text
raw_current_mask_sum
raw_previous_mask_sum
raw_to_support_gain_current
raw_to_support_gain_previous
current_support_token_count
previous_support_token_count
visual_grid_token_count
image_embed_token_count
image_placeholder_token_count
box_frame_image_width
box_frame_image_height
probe_q_norm
probe_k_norm
probe_q_std
probe_k_std
probe_score_mean
probe_score_std
probe_score_max
probe_score_min
anchor_hidden_norm
anchor_hidden_grad_norm
visual_key_norm
row_weighted_loss_pos
row_weighted_loss_neg
per_size_bucket_mass
per_overlap_bucket_mass
```

Training-health logs should be interpreted before rollout metrics:

- `aux_to_ce_loss_ratio` and `weighted_aux_to_total_loss_ratio` catch
  auxiliary loss domination.
- `probe_entropy`, `probe_effective_tokens`, and score extrema catch saturated
  or collapsed probe distributions.
- `probe_top1_in_current` and `probe_top5_current_mass` distinguish useful
  region binding from diffuse loss reduction.
- `probe_grad_norm` and `adapter_grad_norm` catch silent no-gradient and
  exploding-gradient failures.
- `adapter_nonzero_grad_param_count` confirms the aux path can influence the
  trainable LoRA/adapter side, not only the probe.
- `visual_key_norm` is monitor-only in V1 because visual keys are detached.

Required rollout/eval metrics:

```text
duplicate box rate
same-label same-region repeat rate
max-row hit rate
recall
precision
early EOS rate
overlap-object recall/precision slice
invalid/parse drop counters
```

Wiring gate before trusting any training curve:

- finite nonzero auxiliary loss on a tiny aux-enabled batch with valid rows;
- active PEFT `modules_to_save` probe copy is resolved after wrapping;
- active probe gradients are nonzero;
- at least one trainable adapter/LoRA parameter receives aux-attributable
  gradient;
- detached visual keys receive no aux gradient;
- image placeholder count, captured image embed count, and `image_grid_thw`
  post-merge count agree;
- row-conditioned visual coverage and feature painting are disabled.

Tiny-training health gate:

- losses are finite;
- valid positive row count is nonzero;
- geometry hard-error counters remain zero;
- auxiliary-to-CE ratios stay within the predeclared smoke tolerance;
- probe entropy/effective-token metrics do not collapse immediately;
- top-k current-region metrics are nonzero on examples with valid masks;
- probe and adapter gradient norms remain finite.

Rollout interpretation gate:

Do not claim success because `L_pos` or `L_neg` decreases. The loss is only a
mechanism diagnostic unless rollout duplicate/repeat metrics improve without
recall or precision collapse, without invalid/parse/max-row/early-EOS regression
beyond a predeclared tolerance, and with the overlap-object slice reported.

## Validation Plan

Unit tests before training:

1. Config parsing:
   - default disabled;
   - strict unknown-key rejection;
   - mutual exclusion with active feature painting;
   - scalar validation for `probe_lr`, `positive_weight`, `negative_weight`,
     `negative_margin`, and `support_floor`;
   - rejects separate aux-loss warmup, auxiliary schedule knobs, probe-topology
     knobs, row-state mode knobs, and attention/feature-painter knobs in V1a;
   - requires explicit positive `instance_enumeration_aux.probe_lr` when aux is
     enabled;
   - requires `do_resize=false` for V1a instance-enumeration aux runs;
   - rejects any nonzero V1 instance-enumeration optimizer-group weight decay;
   - rejects packed auxiliary sidecars in V1a;
   - rejects `padding_free_packed` for aux-enabled V1a runs;
   - allows `objective.target_ir.rollin_policy.name="sorted"` for the sorted
     V1a lane;
   - rejects `data.object_ordering="sorted"` with
     `objective.target_ir.rollin_policy.name="random_permutation"` for
     aux-enabled V1a runs;
   - materialized random and sorted smoke configs have matching
     `data.object_ordering` and `objective.target_ir.rollin_policy.name`;
   - the sorted smoke config materializes sorted
     `selected_normalized_object_indices`.

2. Sidecar construction:
   - `InstanceEnumerationState` is constructed by the dataset/teacher-forcing IR
     path, not reconstructed in the collator or loss path;
   - `InstanceEnumerationState` is a frozen dataclass with tuple fields;
   - sidecars contain CPU/Python metadata only, not tensors, masks, projected
     features, or device-specific values;
   - state carries `ordered_object_indices` from
     `selected_normalized_object_indices` and boxes in
     `object_boxes_norm1000_xyxy_by_index`;
   - dataset-side registration keeps `instance_enumeration_state` as a
     trainer-only sidecar and strips it from model inputs;
   - state does not store row objects, anchor token positions, target boxes, or
     previous boxes;
   - a nonmonotonic order such as `[2, 0, 1]` derives rows `(2, ())`,
     `(0, (2,))`, and `(1, (2, 0))`;
   - missing boxes for any derived row index raise a hard error;
   - future rendered objects never enter previous masks;
   - loss code rejects missing sidecars instead of reconstructing boxes/order
     from decoded text, labels, token roles, or assistant payloads.

3. Collator sidecar transport:
   - unpacked aux-enabled batches require every sample to carry
     `instance_enumeration_state`;
   - unpacked collator output carries a tuple of per-sample states aligned to
     batch order;
   - partial sidecar presence in an unpacked aux-enabled batch raises a hard
     error;
   - packed sidecars are rejected.

4. Anchor extraction:
   - finds exactly one `branch_position == 0` atom for object rows;
   - resolves anchors from the aligned `TeacherForcingTargetIR` using
     the derived `target_object_index`;
   - uses aligned `atom.target_position`;
   - rejects missing or multiple anchors with sample/order diagnostics;
   - rejects silent partial anchor coverage;
   - Stage-1 GT hard-fails on any expected/discovered anchor mismatch.

5. Visual mask geometry:
   - visual mask length equals image token count;
   - visual lattice derives from Qwen `image_grid_thw`, not reconstructed image
     dimensions;
   - norm1000 boxes convert to pixel-space using the dataset scene's stored
     original image width/height;
   - V1a rejects `do_resize=true` or any post-normalization image transform;
   - each row has exactly one image grid and no video grid;
   - image-grid temporal length is one;
   - `h` and `w` are divisible by `spatial_merge_size`;
   - mask length equals captured post-projector image embedding length;
   - mask length equals `input_ids == image_token_id` placeholder count;
   - mask length equals
     `t * (h // spatial_merge_size) * (w // spatial_merge_size)`;
   - current bbox mask lands on expected post-merge cells;
   - asymmetric non-square image/grid tests prove flatten order;
   - edge boxes test the norm1000 `999` endpoint convention;
   - valid Stage-1 GT box with empty current support mask raises a geometry error;
   - invalid or degenerate boxes are excluded before aux state construction and
     counted separately;
   - previous-object masks are max-unioned, not summed;
   - previous negative excludes current overlap;
   - empty or tiny negative mask materializes `L_neg=0` for bookkeeping,
     excludes the row from negative reducers, and increments empty-negative
     counters while preserving `L_pos`;
   - raw overlap masks are preserved for diagnostics;
   - support-style masks avoid sub-token positive-loss caps for small objects;
   - empty previous mask gives zero negative loss.

6. Loss formulas:
   - high current score lowers `L_pos`;
   - high previous score raises `L_neg`;
   - overlapping current/previous regions do not create hard false negatives;
   - no NaNs for empty negative masks or mixed precision inputs;
   - valid Stage-1 GT box with empty current support mask is a geometry error,
     not a skipped small object;
   - the high-level aux API consumes sidecars, target IR, captured hidden states,
     captured image embeddings, `image_grid_thw`, `input_ids`, active probe, and
     aux config, then returns weighted/unweighted loss and `instance_enum/*`
     metrics;
   - bridge integration calls that high-level API instead of reimplementing row
     iteration, anchor lookup, or mask construction.

7. PEFT/probe lifecycle:
   - attaches `instance_enumeration_probe` before `sft.prepare_model`;
   - appends `instance_enumeration_probe` to PEFT `modules_to_save`;
   - after PEFT wrapping, resolves the active `modules_to_save.default` probe
     rather than the frozen original module;
   - active probe parameters are in `model.named_parameters()` with
     `requires_grad=True`;
   - probe topology is one shared linear q/k probe with fixed `hidden_dim`;
   - per-layer, per-head, row-conditioned, and MLP probe variants are disabled
     in V1a;
   - optimizer groups active probe parameters exactly once;
   - probe LR equals explicit `instance_enumeration_aux.probe_lr`, not `vit_lr`
     or `aligner_lr`;
   - missing or null `probe_lr` is rejected when aux is enabled;
   - every optimizer group uses `weight_decay=0.0` for V1 instance-enumeration
     runs;
   - PEFT adapter state dict contains probe q/k tensors;
   - aux-enabled resume fails if requested probe tensors are missing instead of
     silently using a freshly initialized probe;
   - inference checkpoint resolution rejects unstripped training aux adapters that
     still declare or contain `instance_enumeration_probe`;
   - aux-only toy PEFT backward gives nonzero active-probe grad and nonzero grad
     on at least one LoRA parameter;
   - Qwen-path smoke resolves the active `ModulesToSaveWrapper` probe copy, not
     the frozen original;
   - Qwen-path smoke reports finite nonzero auxiliary loss when valid rows
     exist;
   - Qwen-path smoke reports nonzero active probe q/k gradients;
   - Qwen-path smoke reports nonzero adapter/LoRA gradient attributable to the
     aux path, not CE alone;
   - Qwen-path smoke verifies the post-projector visual key source receives no
     aux gradient when `detach_visual_keys=true`;
   - before resume claims, training checkpoints include probe tensors;
   - V1a makes no deployable export/strip claim until a future task adds explicit
     stripping tests.

8. Qwen capture:
   - captured `lm_head` input shape matches logits time dimension;
   - captured `lm_head` input `[batch, seq]` prefix matches `input_ids`;
   - rejects `logits_to_keep` or any sliced-logits/sliced-hidden path;
   - split-per-sample image features are the primary unpacked loss input;
   - main post-projector `image_embeds` are the only V1 visual-key source;
   - DeepStack visual embeddings are not included in `L_pos` or `L_neg`;
   - captured image feature count matches `input_ids == image_token_id`;
   - captured image feature count matches `image_grid_thw` post-merge token
     count;
   - post-projector `get_image_features` output is the loss source;
   - scattered `inputs_embeds` image-token positions are used only for
     alignment validation;
   - capture returns unchanged model outputs;
   - original methods/hooks are restored after exceptions;
   - PEFT/ms-swift wrapper traversal finds the actual forward model, `lm_head`
     owner, and visual owner under `.module`, `.base_model`, and `.model` stacks.

9. Monitoring:
   - logs the default training metrics listed above;
   - logs run-resolved constants once with the config/artifact metadata;
   - can enable debug-only geometry/probe metrics without changing config
     semantics;
   - does not report visual-key gradient norms as trainable signal when
     `detach_visual_keys=true`.

10. Health gates:
    - zero-valid-positive-row batch hard-errors in Stage-1 V1a;
    - wiring gate proves finite nonzero aux loss, active probe resolution,
      nonzero active-probe grad, nonzero aux-attributable adapter grad, detached
      visual keys, and image/grid/token count agreement;
    - tiny-training gate proves finite losses, nonzero valid rows, zero geometry
      hard errors, sane aux-to-CE ratios, noncollapsed probe entropy, nonzero
      top-k current-region signal, and finite gradient norms;
    - rollout interpretation gate uses duplicate/repeat, recall, precision,
      max-row, early-EOS, invalid/parse, and overlap-object metrics.

Narrow smoke commands:

```text
python -m pytest tests/detection/instance_enumeration -q
python -m pytest tests/test_teacher_forcing_sidecar_bridge.py -q
python -m pytest tests/test_teacher_forcing_target_builder.py -q
```

Training smoke should be tiny only, with explicit evidence scope. Do not launch
production training until config materialization, sidecars, masks, loss terms,
and metrics are verified.

## Risks

### Probe Shortcut

The probe may learn a private alignment task without improving generation.

Mitigations:

- keep probe small;
- detach visual keys by default;
- keep anchor hidden states attached so gradients can reach LoRA/adapters;
- use small auxiliary weights and the global LR scheduler warmup;
- require nonzero adapter-gradient evidence in smoke tests, not only nonzero
  probe gradients;
- require rollout duplicate metrics for success claims.

### Frozen Original Probe

PEFT `modules_to_save` freezes the original module and trains a copied active
module. If the loss accidentally uses the pre-wrap original
`model.instance_enumeration_probe`, the probe may be frozen or stale. The loss
must resolve the active `ModulesToSaveWrapper.modules_to_save[active]` copy
after `sft.prepare_model` and fail closed if it cannot.

### Inference Adapter Pollution

Saving the probe through PEFT `modules_to_save` is useful for training resume,
but final generation should not require the training-only probe. Export must
strip `instance_enumeration_probe` from inference adapter payloads. Do not make
rollout or vLLM serving depend on constructing the probe.

### Current-Row Leakage

Pooling over teacher-forced desc/bbox tokens would let the representation see
the answer. V1 uses only the `<object_ref_start>` hidden state. In
full-sequence mode, this anchor remains causal: it can see the image, prompt,
previous rendered rows, and the anchor token itself, but not future tokens in
the current row or later rows.

### Rendered-Order Drift

Full-sequence V1 is only valid if "previous object" means previous in rendered
teacher-forced order. Numeric object ids are not a causal order. The
implementation must derive previous/current/future sets from
`TeacherForcingTargetIR.metadata["selected_normalized_object_indices"]` and
must fail closed if that metadata is absent or inconsistent with object-anchor
atoms.

### Anchor Contract Drift

Stage-1 GT training requires `<object_ref_start>` for every rendered object row.
If future prompt variants make object starts optional, they are not compatible
with V1 strict Stage-1 aux training until they provide an equivalent explicit
anchor in `TeacherForcingTargetIR`.

### Qwen Output Assumption Drift

Current Qwen3-VL causal-LM output does not populate hidden states. V1 must use
capture hooks/wrappers and test them against the installed local Transformers
version.

### Geometry Misalignment

If bbox masks do not align with post-merge visual tokens, the loss optimizes
the wrong region. Geometry visualization/debug tests block interpretation.

### Overlap Penalty

Crowded or occluded objects may overlap. V1 negative masks must subtract or
downweight the current target region from previous coverage.

### Config Confusion

The old `row_conditioned_visual_coverage` name implies feature painting. The
new active section should be `instance_enumeration_aux` to avoid treating the
new result as painter behavior.

## Minimal Implementation Plan

1. Add `instance_enumeration_aux` config dataclasses and strict parser hooks.
2. Add full-sequence `InstanceEnumerationState` in the dataset/teacher-forcing
   IR path and an unpacked collator enricher that only transports it.
3. Factor visual-token region mask geometry from painter code into a
   non-painting helper.
4. Add `InstanceEnumerationProbe` installation before PEFT wrapping, append it
   to `modules_to_save`, and resolve the active wrapped probe after
   `sft.prepare_model`.
5. Extend the CoordExp optimizer grouping so active probe parameters are grouped
   exactly once at explicit `probe_lr`, and force `weight_decay=0.0` for every
   V1 instance-enumeration optimizer group.
6. Add Qwen forward capture context with tests.
7. Add anchor extraction and rendered-order previous/current mapping from
   aligned `TeacherForcingTargetIR`.
8. Add `L_pos`, `L_neg`, global-LR-scheduler metrics, and aux health metrics.
9. Integrate after CE in `TrainerLossBridge` without changing upstream Qwen.
10. Add paired random-order and sorted-order smoke configs for the fixed
   standard CE + `L_pos` + `L_neg` objective.
11. Run unit tests and tiny unpacked random+sorted training smokes before
   interpreting any rollout.
12. Add checkpoint/export tests: training checkpoints include the probe for
    resume, while inference/export adapters strip it.
