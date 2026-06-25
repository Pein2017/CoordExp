# Coverage Ledger Auxiliary Loss Design

## Objective

Design a Stage-1 research teacher-forcing auxiliary objective that tests
whether Qwen3-VL autoregressive detection hidden states can encode object
coverage state during normal teacher-forced training. V0 is training-only: it
must not change inference or decoding behavior, must not allocate new special
tokens, and must not change production launch defaults. It deliberately selects
an existing closed compact detection template that emits both
`<|object_ref_end|>` and `<|box_end|>` so row-completion states are real tokens.

The implementation state after this design packet is `unit-implementation
scope only`. Unit tests can support implementation confidence, but they do not
claim preflight success, smoke success, production readiness, or stable
contract promotion. Production training remains blocked until an asset-backed
preflight and the approved smoke packet both pass.

## Source Of Truth

- Research idea overview:
  `research/ideas/ledger-auxiliary-loss/overview.md`
- Research discussion decisions:
  `research/ideas/ledger-auxiliary-loss/discussion.md`
- Canonical Stage-1 route:
  `docs/training/STAGE1_OBJECTIVE.md`
- Stage-1 code routing:
  `docs/IMPLEMENTATION_MAP.md`
- Training metrics contract:
  `docs/training/METRICS.md`
- Upstream boundary:
  `docs/standards/UPSTREAM.md`
- Qwen-VL upstream notes:
  `docs/standards/upstream/QWEN_VL.md`
- Current loss bridge:
  `src/training/bridge/loss_bridge.py`
- Model-input sidecar boundary:
  `src/training/encoding/model_inputs.py`
- Training sidecar containers:
  `src/training/sidecars.py`
- Stage-1 teacher-forcing config schema:
  `src/config/schema.py`
- Metric event helpers:
  `src/metrics/events.py`

## Scope

The first implementation scope is a Stage-1 detection teacher-forcing research
pilot under the public route:

```yaml
pipeline:
  id: stage1_research_teacher_forcing
objective:
  id: research_teacher_forcing
```

The new objective is a training-only coverage-ledger term with two subterms:

- coverage-state BCE over prompt-end and row-completion states;
- one-vs-all row-object binding at row `<box_start>` states.

The implementation must be config-first, strict once enabled, and limited to
single-image detection examples in v0.

V0 template requirement: use `compact_object_box_closed`. Both ending wrapper
tokens are mandatory:

```text
<|object_ref_start|>{desc}<|object_ref_end|><|box_start|>x1 y1 x2 y2<|box_end|>
```

`compact_object_box_closed_lines` is not the default v0 route because newline
behavior adds an avoidable variable. Plain `compact` is invalid for the ledger
pilot because it lacks `<|box_end|>`. `compact_full` is a historical chat
template/schema label used by older recursive-detection artifacts and must not
be used as the v0 Stage-1 `detection_template.id`.

## Non-Goals

- Do not change inference behavior.
- Do not change decoding or generation constraints.
- Do not invent new detection special tokens or change tokenizer allocation.
- Do not silently change global production defaults; the v0 pilot must set the
  closed compact template explicitly in its own smoke/baseline configs.
- Do not patch installed Hugging Face or ms-swift upstream files.
- Do not recompute the vision tower for the production loss path.
- Do not support video or multi-image samples in v0.
- Do not add rollout, Stage-2, GRPO, no-return, stable, commit, router,
  duplicate-unlikelihood, or frontier losses.
- Do not create an OpenSpec change until the loss/config semantics are promoted
  into a stable compatibility-sensitive contract.
- Do not run production training until the 128-sample smoke/overfit artifacts
  verify state, geometry, and visual-token alignment.

## Philosophy

Ledger loss imposes an internal object-coverage inductive bias. It encourages
the autoregressive hidden state after each completed object span to encode
which annotated objects have already been emitted. This may help if duplication
and low recall arise from weak prefix-side coverage memory. However, it does
not by itself guarantee that the model will use this coverage state to select a
new valid object, nor does it solve visual perception, small-object
localization, missing annotations, or router/commitment failures.

The first pilot therefore measures mechanism health before claiming detection
quality improvement.

## Template And Baseline Contract

The ledger pilot depends on completed-object states, so template selection is
part of the experiment contract, not an incidental training knob.

V0 target template:

- `detection_template.id: compact_object_box_closed`;
- `include_object_ref_end: true`;
- `include_box_end: true`;
- required structural adapter rows include `<|object_ref_start|>`,
  `<|object_ref_end|>`, `<|box_start|>`, and `<|box_end|>`;
- no new special tokens are introduced.

The v0 comparison shape is:

```text
closed-wrapper hard-SFT baseline
vs.
closed-wrapper hard-SFT + coverage ledger
```

Both runs must use the same template, object field order, data rows, sample
seed, processor resize policy, packing policy, model/checkpoint, and training
profile. The resolved-config diff for a smoke comparison should be limited to
ledger enablement, ledger weights, run identity/artifact roots, and ledger
debug artifact settings. Any quality claim against the active
`pure_valid_set_marginal` Stage-1 route requires a separate paired
closed-wrapper `pure_valid_set_marginal` baseline and ledger run; do not compare
hard-SFT-plus-ledger directly against the current plain-`compact`
`pure_valid_set_marginal` production route.

Exact planned smoke config paths:

- baseline:
  `configs/stage1/detection_teacher_forcing/smoke/coverage_ledger_closed_hard_sft_128_baseline.yaml`
- ledger:
  `configs/stage1/detection_teacher_forcing/smoke/coverage_ledger_closed_hard_sft_128.yaml`

These paths are reserved by this design but must not be created until the user
approves implementation.

## Config Contract

Use the active Stage-1 `objective.terms` convention instead of reviving retired
`objective.modules` authoring.

Proposed v0 authoring:

```yaml
pipeline:
  id: stage1_research_teacher_forcing
detection_template:
  id: compact_object_box_closed
token_embeddings_adapter:
  enabled: true
  groups:
    coord_geometry:
      role: coord_geometry
      start_token: "<|coord_0|>"
      end_token: "<|coord_999|>"
    compact_structure:
      role: structural_ce_only
      tokens:
        - "<|object_ref_start|>"
        - "<|object_ref_end|>"
        - "<|box_start|>"
        - "<|box_end|>"
objective:
  id: research_teacher_forcing
  profile: hard_sft
  terms:
    coverage_ledger:
      enabled: false
      coverage_weight: 0.1
      region_anchor_weight: 0.1
      ledger_projection_dim: 256
      temperature: 0.2
      normalize_eps: 1.0e-6
      pos_weight: 1.0
      log_auc: true
      log_accuracy: true
      overlay_sample_count: 16
      smoke_sample_count: 128
      smoke_sample_seed: 20260623
```

Schema validation requirements:

- `enabled` must be boolean.
- `coverage_weight` and `region_anchor_weight` must be finite floats `>= 0`.
- `ledger_projection_dim` must be a positive integer.
- `temperature` must be finite and `>= 0.05`.
- `temperature < 0.1` should emit a debug warning or diagnostic event.
- `normalize_eps` must be finite and `>= 1e-8`, defaulting to `1e-6`.
- `pos_weight` must be finite and `> 0`, defaulting to `1.0`.
- `smoke_sample_count` must be `128` for the first smoke/overfit recipe.
- `overlay_sample_count` must be `16` for the first smoke/overfit recipe.
- Strictness is not configurable: when `coverage_ledger.enabled: true`,
  unexpected missing metadata or malformed alignment is a hard failure.
- `coverage_ledger.enabled: true` must reject `detection_template.id: compact`
  and any template contract with `include_object_ref_end: false` or
  `include_box_end: false`.
- `coverage_ledger.enabled: true` must reject `detection_template.id:
  compact_full`; that name belongs to older chat-template/analysis artifacts,
  not the current Stage-1 semantic template registry.
- `coverage_ledger.enabled: true` must reject packing modes until sidecar
  offset rewriting is proven. V0 uses `per_device=1`, one unpadded physical
  sequence per forward pass, `packing: false`, `static_packing: false`, and
  `padding_free_packed: false`.

## Data And Sidecar Contract

Structured rendered/tokenized object-entry metadata is the source of truth for
object identity, object order, and state positions. Token-stream parsing is
allowed only as an assertion/debug fallback.

The implementation should add the smallest explicit typed sidecar payload needed
to carry coverage-ledger metadata through the existing non-forwarded sidecar
boundary. This payload belongs in `TrainingSidecars.supervision.payloads`, not
in generic metadata, so the bridge can type-check it and reject duplicate or
conflicting payloads.

```text
TrainingSidecars
  supervision.payloads[]
    CoverageLedgerSidecar
      sample_id
      prompt_end_position
      object_entries[]
        object_instance_id
        source_object_index
        emitted_order_index
        image_index
        bbox_norm1000_xyxy
        box_start_position
        coord_label_positions[4]
        object_ref_end_position
        box_end_position
      image_grid_thw
      processed_width
      processed_height
```

The sidecar dataclasses should be shallow-immutable after construction and
defensively copied at the collator/sidecar boundary. The bridge must find
exactly one `CoverageLedgerSidecar` per sample when the term is enabled. Missing
payloads, duplicate payloads, or disagreement between explicit sidecars and raw
batch sidecars are hard failures.

Required state positions:

- prompt-end state: the final prompt token before the assistant response;
- row-completion state: the `<box_end>` token for each completed object row;
- region-anchor state: the `<box_start>` token for each object row.

Required sidecar checks:

- The template contract must have both `include_object_ref_end=True` and
  `include_box_end=True`.
- `input_ids[box_start_position]` must be `<|box_start|>`.
- `labels[box_start_position + 1]` must be the first coordinate token for the
  same object row.
- `input_ids[object_ref_end_position]` must be `<|object_ref_end|>`.
- `input_ids[box_end_position]` must be `<|box_end|>`.
- Each object row must have exactly one `object_ref_end` control span and
  exactly one `box_end` control span.
- Every object row must have exactly four coordinate label positions.
- Object ids and emitted-order indices must be unique inside a sample.
- Zero-object samples must fail fast when the ledger term is enabled.

The new payload must remain sidecar-only. It must not be forwarded into
`model(**inputs)`.

Sidecar extraction rules:

- derive `box_start_position` from the structured `bbox_start_span`;
- derive `coord_label_positions` from structured `coord_spans`;
- derive `object_ref_end_position` and `box_end_position` by selecting
  `control_spans` with labels `object_ref_end` and `box_end`, not by raw token
  string search;
- reject rows where either ending wrapper label is absent, duplicated, or
  outside the object entry span;
- derive `prompt_end_position` as the final prompt-side token index before the
  first supervised assistant/object token, using structured assistant/prompt
  spans before any packing or padding rewrite;
- token-id checks are assertions after structured span derivation, not the
  canonical way to find positions.

V0 rejects multi-sample packing and padding-free packed offset rewrites. Those
can be revisited only after a separate test proves sample/object ownership and
all ledger positions are rewritten correctly.

## Forward Integration

Attach the loss to the current single-forward `TrainerLossBridge` path:

```text
raw batch + sidecars
-> ModelInputBundle strips sidecar-only keys
-> prepare_forward_inputs keeps Qwen-compatible forwarded tensors
-> one Qwen3-VL forward pass
-> logits + hidden states + projected image_embeds
-> coverage-ledger objective math
-> ObjectiveRunner/bridge loss aggregation and MetricEvent emission
```

V0 should compute the coverage-ledger auxiliary in the bridge-owned Stage-1
teacher-forcing path after the single model forward, rather than forcing it
through the current `ObjectiveRunner.run` signature. The generic runner accepts
logits and supervision spans; coverage ledger also needs hidden states,
projected `image_embeds`, and sidecar geometry. A future refactor may promote a
richer runner contract, but v0 should keep the integration honest and local to
the bridge/trainer seam.

The v0 seam should be an explicit bridge-local helper such as
`CoverageLedgerForwardCapture`:

- input: resolved `core_model` and Qwen-compatible `inputs_for_model`;
- unwrap the current trainable Qwen3-VL conditional-generation model without
  losing adapter/LoRA/token-row hooks;
- temporarily wrap the lower-level `model.model.get_image_features` method
  during this forward, call the original method when Qwen invokes it, capture
  the returned projected post-merger `image_embeds`, and restore the original
  method in `finally`;
- call the lower-level Qwen3-VL language/vision model exactly once to obtain
  final language hidden states from the same pass that produced the captured
  visual embeddings;
- apply the same `lm_head` path as the normal conditional-generation forward to
  produce full logits;
- return logits, final hidden states, captured `image_embeds`, and call-count
  diagnostics to the bridge;
- never call `get_image_features` or the visual tower a second time for the
  production ledger loss.

The helper should not rely on `output_hidden_states=True` alone. In the current
local Qwen3-VL implementation, the top-level conditional-generation forward
does not expose projected `image_embeds`, and hidden-state exposure is not a
stable enough contract for this loss. If the helper cannot obtain full logits,
final hidden states, and projected `image_embeds` from the same lower-level
forward without patching installed Transformers files, `enabled: true` must fail
fast.

Capture parity requirements:

- baseline normal-forward logits and capture-helper logits match within a
  documented tolerance for the same input batch;
- `get_image_features` is called exactly once;
- final hidden-state sequence length matches the full logits time axis;
- captured projected `image_embeds` count matches the post-merge image
  placeholder count;
- behavior is tested under bf16 autocast, gradient checkpointing, eager/flash
  attention selection where available, and DDP/wrapped-model access where
  available.

Do not register `coverage_ledger` as a new `ObjectiveRunner` objective in v0.
The closed runner should continue to reject unsupported objective ids. The
bridge should first compute the existing runner-owned CE/objective result, then
add the bridge-owned auxiliary scalar to the returned bridge loss and append
typed metric events. A test must prove:

- `ObjectiveRunner` still rejects `coverage_ledger`;
- bridge `result.loss == objective_result.loss + coverage_ledger_loss`;
- bridge metric events include both runner events and ledger events.

Required Qwen/forward checks:

- exactly one image is present;
- `pixel_values_videos` and `video_grid_thw` are absent;
- `image_grid_thw` exists and has one row;
- the model returns full unsliced logits;
- hidden states come from the same forward pass as the CE logits;
- projected visual object embeddings are detached before ledger loss math.

## Coverage Ledger Head Ownership

The trainable projections live in a model-registered module, not inside the
transient bridge/helper.

Required module contract:

```text
CoverageLedgerHead(nn.Module)
  coverage_state_projection
  region_anchor_state_projection
  object_projection
```

Ownership requirements:

- canonical module name: `coverage_ledger_head`;
- registered on the actual trainable model object seen by the Trainer before
  optimizer construction;
- visible in `model.named_parameters()` when `coverage_ledger.enabled: true`;
- absent, or present with no trainable parameters, when the feature is disabled;
- moved with the model to the correct device and dtype;
- participates in DDP/PEFT/Swift wrapping without being hidden from optimizer
  parameter discovery;
- saved and restored in restartable checkpoints and final adapter/model
  artifacts;
- never recreated per `compute_loss` call.

Optimizer requirements:

- v0 does not add a public head-LR or head-weight-decay knob;
- ledger-head parameters use the main training learning rate and weight decay
  unless an implementation review finds the active optimizer cannot express
  that safely;
- the `multimodal_token_embeddings_adapter` optimizer path must explicitly keep
  `coverage_ledger_head` parameters in optimizer groups, even when the model
  architecture split only names vision/aligner/language-module prefixes;
- tests must prove at least one ledger-head parameter changes after a
  backward/optimizer step.

Gradient path:

- detach projected visual object embeddings before `object_projection`, so no
  gradient flows into the visual encoder or aligner through the ledger object
  embeddings;
- allow gradients into the ledger projection heads;
- allow gradients from coverage/anchor losses into language-side hidden states
  and the trainable language/adapter parameters that produced them.

## Visual Region Mapping

Use the projected post-merger `image_embeds` that are scattered into the LLM
input stream. Do not use raw pre-merger final vision-block patch states.

For each object:

```text
image_grid_thw[0] = [T, H_patch, W_patch]
patch_size = model.config.vision_config.patch_size
merge = model.config.vision_config.spatial_merge_size

H_grid = H_patch * patch_size
W_grid = W_patch * patch_size
H_post = H_patch // merge
W_post = W_patch // merge
cell_h = patch_size * merge
cell_w = patch_size * merge
```

V0 requires `T == 1`. The sidecar `processed_width` and `processed_height`
from the actual processed image are the source of truth for bbox projection.
Before selecting visual tokens, assert:

```text
processed_width  == W_grid
processed_height == H_grid
```

This keeps `do_resize=false` alignment explicit and catches image/grid drift
before loss math. V0 routes norm1000 `xyxy` endpoint conversion through the
shared CoordExp geometry helper (`norm1000_bbox_to_pixel_bbox` /
`denorm_and_clamp`). That helper uses the `v / 999 * (dim - 1)` mapping and then
clamps and rounds to integer pixel endpoints. Loss pooling and overlay artifacts
must use the same helper so debug images faithfully show the region used by the
auxiliary loss.

```text
x1_px = clamp_and_round(x1 / 999 * (processed_width  - 1), 0, processed_width  - 1)
x2_px = clamp_and_round(x2 / 999 * (processed_width  - 1), 0, processed_width  - 1)
y1_px = clamp_and_round(y1 / 999 * (processed_height - 1), 0, processed_height - 1)
y2_px = clamp_and_round(y2 / 999 * (processed_height - 1), 0, processed_height - 1)
```

Select the minimal half-open post-merge visual-token rectangle:

```text
col0 = floor(x1_px / cell_w)
col1 = ceil (x2_px / cell_w)
row0 = floor(y1_px / cell_h)
row1 = ceil (y2_px / cell_h)

col0 = clamp(col0, 0, W_post - 1)
row0 = clamp(row0, 0, H_post - 1)
col1 = clamp(col1, col0 + 1, W_post)
row1 = clamp(row1, row0 + 1, H_post)
```

Projected local visual-token indices are row-major:

```text
local_idx(row, col) = row * W_post + col
```

Object visual embedding for v0 is mean pooling over the selected projected
visual-token embeddings. It is visual-only; do not concatenate class text or
description embeddings.

Fail-fast mapping checks:

- `H_patch % merge == 0` and `W_patch % merge == 0`;
- placeholder count equals `T * (H_patch // merge) * (W_patch // merge)`;
- sidecar processed dimensions match `image_grid_thw * patch_size` under
  `do_resize=false`;
- selected region is non-empty and inside the single image placeholder block;
- bbox satisfies `0 <= x1 < x2 <= 999` and `0 <= y1 < y2 <= 999`.

The implementation must include a same-forward Qwen parity probe: for a
single-image sample, locate the image placeholder span in `input_ids`, compute
the post-merge local visual indices, and assert that local visual index `i`
corresponds to placeholder offset `span_start + i` and the same captured
`image_embeds[i]`. This probe must also assert that production ledger loss did
not call `get_image_features` a second time.

## Objective Semantics

### Coverage-State BCE

Coverage states are:

- state 0: prompt end;
- state k: row k `<box_end>` for every emitted object row.

For a sample with `N` GT objects and `K` rendered object rows, score all
state-object pairs. Targets are:

```text
prompt end:
  all objects uncovered -> y = 0

after row k <box_end>:
  objects emitted in rows <= k -> y = 1
  objects emitted in rows >  k -> y = 0
```

The pair computation should be ragged and flattened by sample, not padded into
a large `B x Kmax x Nmax` tensor.

### Region-Anchor Binding

For every rendered object row, the row `<box_start>` state scores against every
annotated visual object. The current row object is positive and all other
objects are negative:

```text
row k target for object j:
  y = 1 if j == k
  y = 0 otherwise
```

This is a row-object binding diagnostic, distinct from the cumulative coverage
inventory target. Its purpose is to bind the unique teacher-forced object
instance to the language-side coordinate-start state and selected visual region;
it still does not guarantee that free rollout will select a new valid object.

Region-anchor uses:

- separate trainable state projection;
- shared visual object projection;
- one-vs-all BCE over the current-row/object matrix.

Use `coverage_ledger_head.region_anchor_state_projection` for the
region-anchor state side and `coverage_ledger_head.object_projection` for the
shared visual object side.

Row-object binding AUC and thresholded accuracy are diagnostic-only and are
computed over the one-vs-all binding matrix. Do not interpret them as rollout
recall or mAP.

## Numerical Formula

Compute the ledger head, normalization, dot product, temperature division, BCE,
and reductions in the repo's explicit fp32 objective precision policy, such as
`ObjectivePrecisionPolicy` or the equivalent local helper, with autocast
disabled:

```python
with torch.autocast(device_type=h.device.type, enabled=False):
    h32 = h.float()
    e32 = e.float()

    q_raw = coverage_ledger_head.coverage_state_projection(h32)
    z_raw = coverage_ledger_head.object_projection(e32.detach())

    q = F.normalize(q_raw, p=2, dim=-1, eps=normalize_eps)
    z = F.normalize(z_raw, p=2, dim=-1, eps=normalize_eps)

    score = (q * z).sum(dim=-1)
    logits = score / temperature
    loss_items = F.binary_cross_entropy_with_logits(
        logits,
        targets.float(),
        reduction="none",
        pos_weight=coverage_pos_weight_tensor_or_none,
    )
```

`pos_weight` applies only to the coverage-state BCE term. The row-object
binding subterm must not silently reuse `pos_weight` unless a later design
introduces a named binding weighting rule.

Because projections are L2-normalized, `score` is bounded near `[-1, 1]` and
logit magnitude is bounded near `1 / temperature`. The implementation must fail
fast on non-finite raw projections, normalized vectors, scores, logits, or
losses. Internal debug assertions may check normalized score bounds, but V0 does
not require score-bound metrics in the logged contract.

Total auxiliary loss:

```text
coverage_ledger_loss =
  coverage_weight * coverage_bce_loss
  + region_anchor_weight * region_anchor_loss
```

This auxiliary loss is added to the existing runner-owned CE/objective loss.

## Metrics

Emit typed `MetricEvent`s and flatten through the existing observability path.
Metric keys use the shorter `teacher_forcing/ledger/*` namespace even though
config authoring uses `objective.terms.coverage_ledger`.

Required metric contract:

| key | reducer | unit | numerator / denominator | notes |
| --- | --- | --- | --- | --- |
| `teacher_forcing/loss/coverage_ledger_auxiliary_weighted` | `last` | `batch` | value only | exact scalar added to runner CE/objective loss |
| `teacher_forcing/ledger/coverage_ledger_auxiliary_pair_normalized` | `weighted_mean` | `object` | component weighted pair sum / valid ledger pair count | diagnostic-only count-weighted view |
| `teacher_forcing/ledger/coverage_bce` | `weighted_mean` | `object` | coverage BCE mean with valid coverage-pair count weight | diagnostic-only |
| `teacher_forcing/ledger/row_object_binding_bce` | `weighted_mean` | `object` | one-vs-all row-object BCE mean with valid binding-pair count weight | diagnostic-only |
| `teacher_forcing/ledger/coverage_auc` | `ratio` | `object` | tie-aware positive-negative rank numerator / comparable pairs | omit when one class is absent |
| `teacher_forcing/ledger/coverage_accuracy` | `ratio` | `object` | correct thresholded predictions / valid coverage pairs | diagnostic-only |
| `teacher_forcing/ledger/row_object_binding_auc` | `ratio` | `object` | tie-aware binding positive-negative rank numerator / comparable pairs | omit when one class is absent |
| `teacher_forcing/ledger/row_object_binding_accuracy` | `ratio` | `object` | correct thresholded predictions / valid binding pairs | diagnostic-only |
| `teacher_forcing/ledger/coverage_state_count` | `sum` | `span` | value only | coverage-state spans |
| `teacher_forcing/ledger/coverage_pair_count` | `sum` | `object` | value only | coverage-state valid state-object pairs |
| `teacher_forcing/ledger/object_count` | `sum` | `object` | value only | rendered objects |
| `teacher_forcing/ledger/row_object_binding_pair_count` | `sum` | `object` | value only | current-row/object binding pairs |

Every event must set `objective_id="coverage_ledger"`,
`metric_surface="coverage_ledger_auxiliary"`, `stage="teacher_forcing"`, and
`diagnostic_only=true`, except
`teacher_forcing/loss/coverage_ledger_auxiliary_weighted`, which is the exact
scalar the bridge adds to the runner-owned loss and must use
`diagnostic_only=false`. Raw and weighted subterm metrics remain diagnostic
observability, even when their weighted values are components of the auxiliary
scalar. Do not add a new `MetricUnit`; use existing `slot`, `object`, and
`batch` units.

AUC must be exact local rank AUC with tie handling and no `sklearn` dependency.
If a batch has no positives or no negatives for coverage-state or row-object
binding pairs, omit that AUC or emit a zero-denominator event; do not log `0.0`.
Positive-negative tied scores receive `0.5` credit, equivalent to the standard
average-rank AUC convention.

Thresholded accuracy uses logit threshold `0.0`, equivalent to sigmoid
probability `0.5`.

Score means, covered-minus-uncovered margins, score bounds, and logit-bound
gauges are future optional diagnostics. They are not V0 contract metrics and
must not be used as required smoke or production gates unless implemented and
documented in `docs/training/METRICS.md`.

Producer examples for `MetricEvent` helpers:

```python
# For BCE-style weighted means:
# loss_sum is already summed over valid items; valid_count is the denominator.
value = loss_sum / valid_count
event = weighted_mean_event(key, value=value, weight=valid_count, ...)

# Equivalent direct event form:
event = MetricEvent(
    key=key,
    numerator=float(loss_sum),
    denominator=float(valid_count),
    value=None,
    reducer="weighted_mean",
    unit="slot",
    ...
)

# For AUC:
if n_pos > 0 and n_neg > 0:
    comparable_pairs = n_pos * n_neg
    event = weighted_mean_event(
        "teacher_forcing/ledger/auc_batch",
        value=auc,
        weight=comparable_pairs,
        unit="slot",
        ...
    )
else:
    # omit, or emit zero-denominator weighted_mean; never coerce to 0.0
    event = None
```

Do not pass `loss_sum` as the helper `value` with `valid_count` as `weight`;
that would store `loss_sum * valid_count` as the numerator.

## Debug And Smoke Artifacts

The first 128-sample smoke/overfit phase must emit:

- canonical run artifacts already expected by the Stage-1 route:
  `resolved_config.json`, `effective_runtime.json`,
  `experiment_manifest.json`, `run_metadata.json`,
  `train_data_provenance.json`, `runtime_env.json`, and config source copies
  when present;
- `pipeline_manifest.json` only if the runtime already writes it for this route;
  do not fabricate a new canonical-looking manifest;
- `ledger/selected_samples.json` from `debug.train_sample_selection` using
  random seed `20260623`;
- `ledger/alignment_debug.jsonl` for all 128 samples;
- `ledger/overlays/` containing 16 rendered overlay samples;
- `ledger/smoke_interpretation.md` with a smoke-scoped interpretation;
- `ledger/smoke_manifest.json` if `experiment_manifest.json` cannot yet point
  directly to the ledger artifacts.

Either `experiment_manifest.json` or `ledger/smoke_manifest.json` must contain
relative pointers to every ledger smoke artifact and to the canonical run
artifacts used for interpretation.

JSONL rows must include:

- sample id;
- object id;
- emitted order index;
- image index;
- `image_grid_thw`;
- `patch_size`;
- `spatial_merge_size`;
- processed width and height;
- norm1000 bbox;
- rounded pixel bbox;
- selected post-merge rectangle;
- selected visual-token count;
- image placeholder span;
- prompt-end position;
- `<box_start>` position;
- first-coordinate label position;
- `<box_end>` position;
- coverage-state count;
- coverage pair count;
- row-object binding pair count;
- tensor shapes used by the loss.

Overlay images must be rendered from the actual processed image with:

- GT bbox;
- post-merge visual-token grid;
- selected minimal token rectangle.

The overlay gallery is required before interpreting the 128-sample run. It is
not optional for the smoke/overfit phase.

The mandatory all-128 `ledger/alignment_debug.jsonl` is independent of any
future first-batch debug dump. Do not gate this smoke artifact behind a
first-batch-only debug switch.

Before launching the smoke/overfit trainer, run a ledger preflight over the
randomly selected 128 examples. The preflight must build sidecars and visual
token rectangles without model training, and it must fail before launch if any
selected example violates single-image, grid divisibility, placeholder count,
processed-dimension, bbox, or object-row alignment requirements.

`ledger/selected_samples.json` schema:

- schema version;
- source JSONL path (`source_jsonl_path`), repo-relative source identity when
  available (`source_jsonl_repo_path`), resolved host path
  (`source_jsonl_resolved_path`), and digest;
- selected row indices;
- selected sample ids when available;
- dataset id and split when available;
- random seed `20260623`;
- selection algorithm name;
- `detection_template.id`;
- object field order;
- tokenizer id and model id;
- processor resize policy, including `do_resize=false`;
- image-grid metadata version;
- per-sample object count;
- per-sample `image_grid_thw`, processed width, processed height, and image
  identity/digest when available.

The preflight writer must fail fast if `ledger/` or `ledger/overlays/` already
exists with stale files. Do not merge new preflight output into an existing
non-empty ledger tree.

The implementation must provide a preflight command before the trainer launch
path is considered usable. Planned command shape:

```bash
python scripts/training/coverage_ledger_preflight.py \
  --config configs/stage1/detection_teacher_forcing/smoke/coverage_ledger_closed_hard_sft_128.yaml \
  --baseline-config configs/stage1/detection_teacher_forcing/smoke/coverage_ledger_closed_hard_sft_128_baseline.yaml \
  --output-root <run-artifact-root>
```

The preflight command must write `ledger/selected_samples.json` and
`ledger/alignment_debug.jsonl` without training, and it must fail before model
launch on invalid template, sidecar, grid, bbox, image, or packing state.

## Implementation Decomposition

The implementation should be split into these independently testable units:

1. `coverage_ledger` config schema under Stage-1 `objective.terms`.
2. Closed-template smoke and baseline config pair using
   `compact_object_box_closed`.
3. Ledger sidecar dataclasses and collator propagation.
4. State-position builder for prompt end, `<box_start>`, `<|object_ref_end|>`,
   and `<box_end>`.
5. Registered `CoverageLedgerHead(nn.Module)` installation, optimizer, and
   checkpoint integration.
6. Bbox-to-post-merge visual-token mapping helper.
7. Qwen forward helper or wrapper seam for same-forward hidden states and
   projected `image_embeds`.
8. Coverage-ledger loss module with fp32 normalized projections.
9. AUC, accuracy, and typed metric-event emitters.
10. JSONL and overlay debug artifact writer.
11. Random 128-sample smoke manifest builder.
12. Smoke/overfit preflight command and launch recipe.

The implementation plan should stop after unit tests, config/schema checks, and
the 128-sample smoke/overfit launch recipe. Production training config belongs
to a later phase after smoke artifacts are reviewed.

## Required Tests

Config/schema tests:

- accept disabled `objective.terms.coverage_ledger`;
- accept enabled v0 defaults;
- accept `objective.profile: hard_sft` with enabled
  `objective.terms.coverage_ledger`, because the ledger term is bridge-local
  auxiliary loss rather than a new semantic target distribution;
- accept only `compact_object_box_closed` for the v0 ledger route;
- reject plain `compact`, `compact_full`, `compact_box_closed`, and any template
  lacking either `<|object_ref_end|>` or `<|box_end|>`;
- validate closed-template token adapter rows include `<|object_ref_start|>`,
  `<|object_ref_end|>`, `<|box_start|>`, and `<|box_end|>`;
- reject retired `objective.modules.coverage_ledger`;
- reject unknown keys;
- reject invalid temperatures, epsilons, weights, and projection dims.

Sidecar/state tests:

- dataset emits object-entry metadata for prompt end, `<box_start>`,
  coordinate labels, `<|object_ref_end|>`, and `<box_end>`;
- selected closed template passes sidecar extraction in both desc-first and
  geometry-first field orders;
- sidecar extraction finds `object_ref_end` and `box_end` by structured control
  span labels, not token-string scanning;
- plain `compact` sidecar construction fails because row-completion states are
  unavailable;
- `CoverageLedgerSidecar` travels through
  `TrainingSidecars.supervision.payloads` and never through model-forwarded
  inputs;
- duplicate, missing, or conflicting coverage-ledger sidecar payloads fail;
- zero-object sample fails when enabled;
- malformed object row fails when enabled;
- static packing and padding-free packing fail in v0 until offset rewriting has
  dedicated tests.

Head/optimizer/checkpoint tests:

- enabled config registers `coverage_ledger_head` on the trainable model before
  optimizer construction;
- ledger-head parameters appear in `model.named_parameters()`;
- ledger-head parameters are present exactly once in optimizer param groups;
- one backward/optimizer step changes at least one ledger-head parameter;
- checkpoint save/load preserves and restores ledger-head state;
- disabled config does not leave stray trainable ledger-head parameters.

Visual mapping tests:

- full image;
- exact cell boundary;
- tiny in-cell object;
- object straddling a post-merge boundary;
- right/bottom edge object;
- degenerate bbox rejection;
- non-square image;
- placeholder count mismatch rejection;
- processed-dimension mismatch rejection;
- same-forward placeholder/index parity for captured projected `image_embeds`.

Forward integration tests:

- `CoverageLedgerForwardCapture` requests or validates hidden states only when
  the ledger term is enabled;
- capture-helper logits match baseline normal-forward logits within tolerance;
- same-forward final hidden states are used;
- captured projected `image_embeds` come from the same lower-level Qwen forward;
- `get_image_features` call count is exactly one;
- full logits are still required;
- visual embeddings are detached for the auxiliary loss;
- missing `image_grid_thw` fails;
- multi-image and video samples fail in v0;
- production ledger loss does not recompute the vision tower or call
  `get_image_features` a second time.

Bridge/objective-runner tests:

- `ObjectiveRunner` still rejects unsupported `coverage_ledger`;
- bridge-local aggregation adds the weighted auxiliary scalar to the existing
  runner-owned loss;
- ledger metric events are appended without weakening runner metric events or
  semantic distribution validation.

Loss tests:

- coverage-state targets match prompt-end and row-completion semantics;
- region-anchor / row-object binding targets mark current-row object positive
  and all other annotated objects negative;
- active bf16 autocast does not leak into the ledger projection math: the
  objective returns fp32 loss, finite gradients, and no dtype mismatch;
- fp32 loss path rejects non-finite values;
- score bounds hold after normalization.

Metric tests:

- coverage metrics use `teacher_forcing/ledger/*`;
- synthetic two-batch reducer test proves each flat metric equals the
  hand-computed numerator/denominator value;
- helper-based weighted-mean producer tests pass `value=sum/count` and
  `weight=count`, never `value=sum` with `weight=count`;
- rank AUC handles ties;
- tied positive-negative logits give `0.5` AUC credit;
- AUC is omitted for single-class batches;
- zero-valid metrics are omitted or zero-denominator events, not coerced to
  `0.0`;
- accuracy uses threshold `0.0`;
- metric events set `diagnostic_only=false` only for
  `teacher_forcing/loss/coverage_ledger_auxiliary_weighted`;
- row-object binding metrics emit diagnostic-only AUC and accuracy when both
  classes are present.

Debug artifact tests:

- JSONL dump contains required fields;
- overlay renderer draws bbox, post-merge grid, and selected rectangle;
- selected-sample manifest records seed `20260623`;
- selected-sample manifest records source JSONL digest, row indices, template
  id, tokenizer/model id, processor resize policy, and image-grid metadata;
- smoke manifest or experiment manifest points to the selected samples,
  alignment JSONL, overlays, interpretation note, and canonical run artifacts.
- resolved-config diff test proves baseline and ledger smoke configs differ
  only in allowed ledger/run-identity fields.

## Smoke And Production Workflow

### Phase 1: Unit And Schema Verification

Run targeted tests for config parsing, sidecar construction, visual mapping,
loss math, metrics, and artifact writers. Do not launch training until these
tests pass.

### Phase 2: 128-Sample Smoke/Overfit

Use random 128-sample selection with seed `20260623`, `per_device=1`, and one
long unpadded physical sequence per forward pass. The 128-sample smoke configs
author `effective_batch_size: 1`, so they must be launched on a single rank; an
8-rank launch is an intentional fail-fast topology mismatch. Static packing and
padding-free packing remain disabled, with a pinned initial budget of
`max_steps=256` optimizer steps for both the no-ledger baseline and ledger run.
The run is a debug and efficiency check, not validation evidence.

Production ledger training uses
`configs/stage1/detection_teacher_forcing/prod/coverage_ledger_closed_hard_sft.yaml`.
That config authors `per_device_train_batch_size: 1` and
`effective_batch_size: 32`; on the intended 8-GPU topology the derived
`gradient_accumulation_steps` is `4`.

The smoke comparison packet must include:

- closed-template hard-SFT baseline config and resolved config;
- closed-template hard-SFT plus-ledger config and resolved config;
- resolved-config diff showing the comparison pair did not accidentally change
  template, profile, data, seed, packing, model/checkpoint, or batch settings
  outside intended ledger/run-identity fields;
- identical selected 128-row manifest for baseline and ledger runs;
- metric reducer output for both runs.

Expected mechanism observables:

- raw and weighted coverage-state losses decrease;
- raw and weighted row-object binding losses decrease;
- coverage and row-object binding AUC/accuracy improve when both classes are
  present;
- positive and negative coverage-state counts are nonzero for interpretable AUC
  windows;
- overlay gallery confirms bbox-to-visual-token mapping.

### Phase 3: Review Gate Before Production Training

Production training is blocked until the smoke packet includes:

- resolved config and canonical run metadata artifacts;
- selected-sample manifest at `ledger/selected_samples.json`;
- JSONL alignment dump at `ledger/alignment_debug.jsonl`;
- 16-sample overlay gallery at `ledger/overlays/`;
- metric stream with coverage-state, row-object binding, auxiliary scalar,
  count, AUC, and accuracy metrics;
- failure-free strict validation;
- nonzero positive and negative coverage-state pair counts for interpretable AUC
  windows;
- no nonfinite emitted metrics;
- written interpretation that stays within smoke evidence scope.

The production phase should be specified separately after the smoke packet is
reviewed.

## Review Convergence

Mode: `docs/spec/plan`.

Allowed mutation: documentation and research knowledge artifacts only. Do not
implement source/config/test changes and do not launch training before user
approval.

Review lanes for this design:

- code-boundary and integration reviewer: no unresolved P0/P1 findings;
- visual geometry and upstream-Qwen reviewer: no unresolved P0/P1 findings;
- metrics and numerical-stability reviewer: no unresolved P0/P1 findings;
- smoke/artifact and research-validity reviewer: no unresolved P0/P1 findings.

Timeouts, disconnected subagents, or missing lane outputs are unresolved review
state, not approval. If a lane times out, either relaunch that lane or mark the
document `hold` with the missing review called out explicitly.

Round-two review status:

| lane | status | accepted follow-up |
| --- | --- | --- |
| code-boundary and integration | ready; no P0/P1/P2 blockers | none |
| visual geometry and upstream-Qwen | ready; no P0/P1/P2 blockers | implementation must still prove same-forward capture and selected-sample grid validity |
| metrics and numerical stability | ready; no P0/P1 blockers | pinned AUC tie credit, fp32 autocast test, and `diagnostic_only` identity |
| smoke/artifact and research validity | ready; no P0/P1 blockers | removed first-batch dump ambiguity and made all-128 alignment JSONL mandatory |

Post-review critical audit response:

| audit item | status in this revision |
| --- | --- |
| P0: row-completion impossible under plain `compact` | resolved by requiring `compact_object_box_closed` with both `<|object_ref_end|>` and `<|box_end|>` |
| P1: trainable projections could live outside optimizer/checkpoint path | resolved by requiring model-registered `CoverageLedgerHead(nn.Module)` and optimizer/checkpoint tests |
| P1: Qwen same-forward capture too abstract | resolved by specifying a lower-level Qwen capture helper with one `get_image_features` call and logits parity tests |
| P1: profile/template baseline confound | resolved by requiring paired closed-template hard-SFT baseline vs ledger comparison and forbidding direct quality comparison to active plain-`compact` `pure_valid_set_marginal` |
| P2: metric producer semantics | resolved with producer examples and reducer tests |
| P2: sidecar derivation | resolved with structured control-span label extraction rules |
| P2: smoke reproducibility | resolved with planned config paths, selected-sample manifest schema, preflight command shape, and fixed smoke training knobs |

Convergence stop state for this document: `ready for user approval after audit
refinement`.

## Current Approval State

Approved for review convergence:

- drafting this design spec;
- editing research/design documentation;
- read-only subagent review.

Not approved:

- source code implementation;
- config implementation;
- test implementation;
- smoke or production training launch;
- OpenSpec promotion.
