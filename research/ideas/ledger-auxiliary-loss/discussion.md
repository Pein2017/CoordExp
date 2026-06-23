---
type: idea
title: Ledger Auxiliary Loss Discussion
description: Durable discussion decisions and open design gates for the ledger auxiliary loss pilot.
tags: [stage1, compact-detection, teacher-forcing, auxiliary-loss, design-gates]
state: active
updated: 2026-06-23
---

# Ledger Auxiliary Loss Discussion

## 2026-06-23 Scope Decision

The first implementation scope is a Stage-1 detection teacher-forcing research
pilot.

## Rationale

The mechanism being tested is prefix-side coverage memory during normal
teacher-forced autoregressive detection. Stage-1 teacher forcing is the narrow
surface that can expose prompt-end and row-completion hidden states without
changing rollout policy, inference, decoding, or production eval behavior.

Stage-2 rollout correction, inference-time decoding, and stable OpenSpec
promotion are deferred until v0 produces enough evidence to justify broader
contract work.

## Philosophy

Ledger loss imposes an internal object-coverage inductive bias. It encourages
the autoregressive hidden state after each completed object span to encode
which annotated objects have already been emitted. This may help if duplication
and low recall arise from weak prefix-side coverage memory. However, it does
not by itself guarantee that the model will use this coverage state to select a
new valid object, nor does it solve visual perception, small-object
localization, missing annotations, or router/commitment failures.

## Consequences

- Keep v0 training-only.
- Keep inference and decoding unchanged.
- Use research and super-power docs before implementation.
- Do not create an OpenSpec change yet.
- Treat ledger diagnostics as mechanism evidence, not as a direct guarantee of
  lower duplication or higher recall.

## 2026-06-23 Metadata Source Decision

Structured rendered/tokenized object-entry metadata is the source of truth for
ledger state positions and emitted-object identity. Token-stream parsing may be
used only as a debug/assert fallback.

### Rationale

The current Stage-1 rendering and tokenization path already carries object
identity and order through `RenderedObjectEntry` and `TokenizedObjectEntry`,
including `object_instance_id`, `object_index`, `source_object_index`,
`desc_span`, `bbox_start_span`, `bbox_span`, `coord_spans`, and
`control_spans`. Compact full-wrapper rendering labels structural control spans
such as `object_ref_end` and `box_end`.

Using this metadata keeps ledger targets tied to the same object-ordering and
template machinery as the CE target. Parsing token streams directly would make
the auxiliary objective more fragile to tokenizer behavior, wrapper variants,
and future template changes.

### Consequences

- The design should add the smallest explicit ledger sidecar or projection
  needed to expose prompt-end and row-completion state positions.
- Missing required metadata should skip or hard-fail according to a configurable
  debug/strictness policy, rather than silently guessing from raw token IDs.
- Tests should cover metadata-derived state construction and the fallback/debug
  assertions separately.

## Open Design Gates

1. Locate the exact projected visual-token access point through CoordExp,
   ms-swift, or Transformers-supported model outputs/wrapper seams.
2. Decide the precise small-dataset overfit recipe and expected ledger-effect
   observations before the first training launch.

## 2026-06-23 Batch 1 Decisions

### Upstream And Visual-Token Access

Use `docs/standards/UPSTREAM.md`,
`docs/standards/upstream/QWEN_VL.md`, and
`docs/standards/upstream/TRAINING_ECOSYSTEM.md` before choosing the visual-token
access point. The upstream rules say CoordExp must not edit installed Hugging
Face generated files such as `modeling_qwen3_vl.py`; integration should happen
through CoordExp, ms-swift, trainer, wrapper, callback, loss, or supported
Transformers surfaces.

The working hypothesis is that projected visual embeddings are available
through the active Transformers Qwen3-VL path. The implementation spec must
verify the actual local signature and wrapper behavior before adding a local
integration seam.

Local upstream inspection on 2026-06-23 found that
`transformers/models/qwen3_vl/modeling_qwen3_vl.py` exposes
`get_image_features(pixel_values, image_grid_thw)`, where
`self.visual(pixel_values, grid_thw=image_grid_thw)` returns `image_embeds` and
deepstack image embeddings. The forward path concatenates `image_embeds`,
checks the placeholder mask, and masked-scatters those embeddings into
`inputs_embeds`. The design must attach to the normal Qwen3-VL/ms-swift
forward convention through a supported helper, wrapper, or hook. It must not
recompute the vision tower for the production ledger loss, and it must not patch
installed Hugging Face files.

### Geometry And Object Order

Bbox-to-visual-grid mapping must route through existing CoordExp geometry and
image-size metadata helpers. Ledger target order follows the realized
teacher-forced object order from structured entry metadata, not category labels,
source-order guesses, or IoU matching.

### State Metadata Strictness

If prompt-end metadata is not already carried, add an explicit sidecar field
rather than inferring prompt end from chat or template token ids.

When ledger loss is enabled, unexpected missing prompt-end metadata, missing
row-completion metadata, malformed object rows, or zero-object training examples
should fail fast. This supersedes the earlier "skip ledger loss with diagnostic
counter" fallback proposal for the active v0 path.

### Full Scoring And Sequence Ownership

Full state-object scoring remains the initial path. Implement the computation
as flattened ragged `(state, object)` pairs for the single physical sequence in
the forward pass, rather than depending on a large padded `B x Kmax x Nmax`
tensor. V0 follows the one-long-unpadded-sequence rule and rejects static
packing or padding-free packed offset rewrites until dedicated tests prove
ledger positions and sample/object ownership are rewritten correctly.

### Gradient Boundary And Weight

Detach visual object embeddings by default. The ledger loss should train the
state projection, object projection, and language hidden-state pathway without
initially pushing gradients into the visual tokens.

Use `lambda_ledger = 0.1` as the first default. The reason is observability:
the first pilot should make the ledger effect visibly positive or negative
rather than hiding it behind an extremely small coefficient.

### Evaluation And Launch Philosophy

Do not define a heavy research acceptance gate before implementing the feature.
Still keep ordinary code/config verification and a small launch-health path.

The first training launch should use a small dataset and test whether the model
can overfit the ledger objective and show the expected covered-vs-uncovered
separation.

## 2026-06-23 Batch 2 Decisions

### Visual Patch Pooling

The v0 object visual embedding should come from the minimal enclosing set of
visual patch embeddings whose grid cells contain the object pixels. This
replaces the earlier "center-in-bbox with nearest-top-k fallback" framing.

The design should not assume that a valid annotated bbox corresponds to no
visual embedding under aligned-image assumptions. The important issue is not
fallback to arbitrary nearest patches; it is the exact mapping between image
pixels/bboxes and the model's visual patch grid when the two are not perfectly
aligned.

For v0, the intended policy is:

- map the bbox from image pixel coordinates into the visual patch grid;
- select the minimal contiguous post-merge visual-cell rectangle that encloses
  the bbox footprint;
- average those selected projected visual-token embeddings;
- fail fast if the image/grid/bbox metadata cannot support that mapping.

Source verification on 2026-06-23: Qwen3-VL first embeds pre-merge image
patches, runs the vision blocks, and then applies `Qwen3VLVisionPatchMerger`.
The returned `image_embeds` are split according to
`image_grid_thw.prod(-1) // spatial_merge_size**2` and are the embeddings
masked-scattered into the LLM input stream. Therefore the ledger object
embedding should use these post-merger `image_embeds`, not raw
`vision.blocks[-1]` states. Raw final vision-block states are pre-merger and
live at the finer patch-grid resolution; using them would not match the
convention of "visual embeddings that enter the LLM."

For grid mapping, use `image_grid_thw` and `spatial_merge_size` to derive:

```text
visual_grid_h = image_grid_h / spatial_merge_size
visual_grid_w = image_grid_w / spatial_merge_size
expected_visual_tokens = grid_t * visual_grid_h * visual_grid_w
```

The selected token rectangle should use half-open bbox geometry: floor the
left/top boundary into the post-merge visual grid, ceil the right/bottom
boundary, clamp to the grid, and require a non-empty selection. This is the
minimal set of post-merge visual tokens whose cells enclose the object pixels.
Validate that the visual token span length equals `expected_visual_tokens`.

### Hidden-State Source And Loss Surface

Use the final-layer hidden states from the same forward pass. Prompt state uses
the hidden state at the last prompt token before assistant response. Row state
uses the hidden state at the row `<box_end>` token position, meaning the state
after consuming the completed object span in teacher forcing.

The first implementation should live in the Stage-1 teacher-forcing
trainer/loss bridge rather than the generic objective runner. Ledger loss needs
model hidden states and visual embeddings, so the trainer bridge is the more
honest v0 integration point.

### Single-Image Detection Rule

V0 supports exactly one image in the input. Video and multi-image examples are
out of scope and should fail fast in the ledger path. This should be promoted
as a repo-wise detection rule if current docs or configs do not already make it
explicit.

### Config Simplicity And Strictness

The feature remains config-first, but the active config should not expose knobs
for choices that v0 does not actually vary. The original candidate config name
was `objectives.ledger_inventory`, but the public/config surface should avoid
`inventory`. The active Stage-1 teacher-forcing schema uses `objective.terms`,
so use `objective.terms.coverage_ledger`: enable/disable, loss weights, ledger
projection dimension, and score temperature. Do not revive retired
`objective.modules` authoring. Drop unnecessary knobs when only one choice is
available.

Use a clearer semantic name than `aux_dim`; prefer `ledger_projection_dim` in
the design unless an existing config naming convention suggests a better local
term.

Strictness is hardcoded when the ledger loss is enabled. No `strict` knob is
needed.

Temperature should be configurable, but the design must include the numerical
stability analysis and exact scoring formula. The preferred direction is
normalized state/object projections, an explicit positive temperature with a
minimum validation floor rather than a runtime clamp, and `BCEWithLogitsLoss`
on the resulting logits.

### Diagnostics

The first smoke/overfit phase should write mandatory JSONL alignment rows for
every selected smoke sample under the run artifact directory, not only for the
first batch. Each entry should include compact handles for sample id, state
positions, object order, covered-label shape or labels, selected visual patch
ranges/counts, and tensor shapes. The 16-sample overlay gallery remains the
visual companion artifact for inspecting bbox-to-visual-token mapping.

`ledger_auc` was initially pending because its implementation cost and benefit
for tiny overfit runs were unclear. The marked numerical review below resolves
the main question: it is available from the same forward pass if ledger logits,
binary labels, and the valid-pair mask are already materialized. It should be
monitoring-only, not a gate.

For the first smoke/overfit run, required mechanism observables should include
ledger loss, covered/uncovered score means, covered-minus-uncovered margin,
thresholded accuracy if easy, and batch/window AUC if it can be emitted without
a new dependency.

Follow-up analysis: `ledger_auc` and thresholded accuracy are available from
one forward pass if the ledger implementation already materializes detached
ledger logits and binary covered labels for all state-object pairs. They do not
require another model forward. They should be included as monitoring metrics if
they can be implemented locally without adding a production dependency; they
are not training gates.

### Smoke And Overfit Pilot

Use a random 128-training-sample smoke/overfit pilot. Treat it as both a
runtime debug run and an efficiency check before larger training. The goal is
to verify that configs resolve, the ledger sidecars align, the runtime path is
efficient enough, and the expected ledger effect is observable.

Follow the applicable repo golden rule for this pilot: `per_device=1`, with
each forward pass consuming one long sequence only and no padding within the
physical batch.

### Production Boundary

Avoid a formal research acceptance gate. Still require implementation
verification and the 128-sample smoke/overfit launch-health check before
production training.

## 2026-06-23 Batch 3 Decisions

### Naming And Objective Shape

The object-region alignment force belongs inside the same ledger objective,
not as a separate standalone training objective. The public/config name should
avoid `inventory`; use `coverage_ledger` for the config-facing block and
describe the mechanism as a coverage ledger or coverage state.

### Pooling And State Coverage

Use mean pooling over the selected minimal post-merge visual-token region for
v0. Do not add attention pooling until the simple alignment path is verified.

Include the prompt-end state in v0 scoring. Prompt-end labels all objects as
uncovered and provides the cleanest baseline negative state before any object
row has been consumed.

### Metrics And First Run

Implement both AUC and thresholded accuracy as monitoring metrics. Compute
them from the same forward-pass ledger logits, binary labels, and valid-pair
mask used by the loss.

Use metric keys under `teacher_forcing/ledger/*`, even though the config block
is `objective.terms.coverage_ledger`. The shorter metric namespace is preferred
for dashboards and logs.

Select the 128-sample smoke/overfit pilot randomly. Do not stratify by object
count or crowding for the first plumbing run. Use seed `20260623` and write a
manifest of selected sample ids under the run artifact root.

### Debug Artifacts

The required v0 debug artifacts are JSONL numeric alignment handles for all 128
smoke samples and a 16-sample overlay visualization gallery during the
smoke/overfit phase. An overlay is a rendered visual debug image showing the
processed input image with the GT bbox, post-merge visual-token grid, and
selected token rectangle drawn on top. Overlays are required for the first
smoke/overfit launch-health inspection because they catch roundoff and
frame-alignment mistakes that are hard to trust from numeric dumps alone.

### Implementation Boundary

The first implementation plan should stop after unit tests, config/schema
checks, and a 128-sample smoke/overfit launch recipe. Production training
config and production-scale launch should be a separate phase after JSONL and
overlay artifacts verify geometry and state alignment.

### Strictness And Dependency Boundary

All strict coverage-ledger validation should run before auxiliary-loss
aggregation and before any partial optimizer step. When enabled, a bad sample
should abort the step rather than silently dropping the ledger term.

AUC should be exact local rank AUC with tie handling, implemented without a
new `sklearn` dependency.

### Region-Anchor Target Policy

The region-anchor force remains inside the same ledger objective and should be
reported as a named subterm, e.g. `region_anchor_loss`, with raw and weighted
metrics. Object embeddings stay visual-only in v0; do not concatenate class
text or object description embeddings before the visual alignment path is
trusted.

For v0 simplicity, the row `<box_start>` region-anchor subterm should bind the
only current row object instance as positive and mask all non-current objects.
Do not punish previous or future objects in this subterm. The goal is to bind
the unique teacher-forced object row to its language-side coordinate-start
state and selected visual region, without turning the anchor term into a
full one-vs-all object classifier.

Use a separate trainable state projection for the region-anchor subterm while
sharing the visual object projection with the main coverage-ledger scoring
path. The state-side question differs between "which object is current for this
row?" and "which objects are already covered by this prefix?", but both terms
should point into the same visual object embedding space in v0.

Use separate configurable loss weights for the two subterms rather than a
single shared objective weight:

```yaml
objective:
  terms:
    coverage_ledger:
      coverage_weight: 0.1
      region_anchor_weight: 0.1
```

Both default to `0.1` for the first observable pilot. The implementation should
also report raw and weighted losses for both subterms so the balance is visible
during the 128-sample smoke/overfit run.

Consequences:

- Region-anchor pairs are positive-only in v0.
- Region-anchor has its own state projection and reuses the shared visual
  object projection.
- Coverage and region-anchor subterms have separate configurable weights.
- AUC and thresholded accuracy belong to the main coverage-ledger state/object
  pairs, not to the positive-only region-anchor subterm.
- Region-anchor monitoring should focus on raw loss, weighted loss, positive
  score/logit mean, pair count, and non-finite/shape checks.

## 2026-06-23 Marked Review A: Bbox To Visual Tokens

### Verdict

Use the existing Qwen3-VL helper/hook path that already materializes projected
`image_embeds`; do not recompute the vision tower for the production ledger
loss. Map object bboxes into the LLM-facing post-merge visual-token grid
derived from `image_grid_thw` and `spatial_merge_size`, then pool the minimal
post-merge token rectangle that encloses the object pixels.

This decision is based on the Qwen3-VL/Transformers convention that the text
placeholder count is `image_grid_thw.prod() // spatial_merge_size**2`, and the
forward path splits and scatters the same projected `image_embeds` into the
LLM stream. Primary source handles:

- Hugging Face Qwen3-VL forward and `get_image_features`:
  <https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L1050-L1143>
- Hugging Face Qwen3-VL processor image-token expansion:
  <https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_vl/processing_qwen3_vl.py#L1202-L1218>
- Hugging Face Qwen3-VL mRoPE post-merge visual ordering:
  <https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L987-L1001>
- PyTorch `masked_scatter_` source-order semantics:
  <https://docs.pytorch.org/docs/2.12/generated/torch.Tensor.masked_scatter_.html>

### Mapping Contract

For each v0 object:

```text
image_index = 0
bbox_2d = [x1, y1, x2, y2] in CoordExp norm1000 integer xyxy
image_grid_thw[0] = [T, H_patch, W_patch]
patch_size = model.config.vision_config.patch_size
merge = model.config.vision_config.spatial_merge_size

H_proc = H_patch * patch_size
W_proc = W_patch * patch_size
H_post = H_patch // merge
W_post = W_patch // merge
cell_h = patch_size * merge
cell_w = patch_size * merge
```

For v0, require `T == 1`; video is not supported. Convert norm1000 endpoints
to processed-image pixel floats without early rounding:

```text
x1_px = x1 / 999 * (W_proc - 1)
x2_px = x2 / 999 * (W_proc - 1)
y1_px = y1 / 999 * (H_proc - 1)
y2_px = y2 / 999 * (H_proc - 1)
```

Select the minimal half-open post-merge cell rectangle:

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

The local projected visual-token indices are row-major:

```text
local_idx(row, col) = row * W_post + col
region_local_indices =
  [row * W_post + col
   for row in range(row0, row1)
   for col in range(col0, col1)]
```

For v0 single-image detection, the region indices should map directly into
the single image placeholder block. If a later helper is generalized to
multi-image, offset each image by the cumulative post-merge token count of
prior images; do not silently support multi-image training in the active
ledger path.

### Box-Start Alignment Anchor

When applying a direct object-region force, anchor it at the hidden state of
the row's `<box_start>` token, because that is the causal state that predicts
the first coordinate token. The ledger path must verify the sidecar alignment:

```text
input_ids[box_start_pos] == box_start_id
labels[box_start_pos + 1] == coord_x1_id_for_same_object
```

The row-completion ledger state remains the hidden state at `<box_end>` for
the completed object span. The `<box_start>` anchor is the object-region
alignment check, not a replacement for row-completion ledger state.

### Fail-Fast Requirements

The active ledger path should fail fast if any of these checks fail:

- `image_grid_thw` is missing, has more than one image, or has `T != 1`.
- `H_patch` or `W_patch` is not divisible by `spatial_merge_size`.
- The processed image dimensions derived from `image_grid_thw * patch_size`
  do not match the record's post-preprocessing dimensions under `do_resize=false`.
- The number of image-token placeholders is not exactly
  `T * (H_patch // merge) * (W_patch // merge)`.
- The selected region is empty or indexes outside the single image placeholder
  block.
- The object bbox is not strict half-open `xyxy` within norm1000 bounds:
  `0 <= x1 < x2 <= 999` and `0 <= y1 < y2 <= 999`.
- The sidecar object id, `<box_start>` position, coordinate token positions,
  and `<box_end>` row-completion state do not agree.

### Required Tests And Debug Artifacts

The implementation plan should include pure mapping tests for full-image,
exact-boundary, tiny-in-cell, straddling-boundary, right/bottom-edge, and
degenerate-bbox cases. It should also include a Qwen parity probe proving that
selected `region_local_indices` address the same `image_embeds` that are
scattered into the corresponding image-token placeholder positions.

The first smoke run should write a small JSONL debug dump under the run
artifact directory with sample id, object id, `image_grid_thw`, patch/merge
values, processed image size, norm1000 bbox, pixel-float bbox, selected
post-merge rectangle, selected-token count, image placeholder span, `<box_start>`
position, first-coordinate label position, and `<box_end>` state position.

The first smoke/overfit phase must also write a 16-sample overlay gallery
rendered from the actual processed image, with the GT bbox, the post-merge
visual-token grid, and the selected minimal token rectangle drawn on top. This
visual check is required before interpreting the first 128-sample overfit run,
not a later nice-to-have.

## 2026-06-23 Marked Review B: Stable Score And AUC

### Verdict

Use normalized state/object projections with a temperature-scaled logit and
`BCEWithLogitsLoss`. Compute the ledger head, normalization, dot product,
temperature division, BCE, and reduction in an explicit fp32 objective island.
Do not apply sigmoid before the loss.

Primary source handles:

- PyTorch `BCEWithLogitsLoss` numerical-stability note:
  <https://docs.pytorch.org/docs/2.12/generated/torch.nn.BCEWithLogitsLoss.html>
- PyTorch `normalize` API:
  <https://docs.pytorch.org/docs/2.12/generated/torch.nn.functional.normalize.html>
- PyTorch AMP BCE guidance:
  <https://docs.pytorch.org/docs/2.12/amp.html#prefer-binary-cross-entropy-with-logits-over-binary-cross-entropy>

### Formula

Use a small fp32 head:

```python
with torch.autocast(device_type=h.device.type, enabled=False):
    h32 = h.float()
    e32 = e.float()

    q_raw = W_state(h32)
    z_raw = W_obj(e32)

    q = F.normalize(q_raw, p=2, dim=-1, eps=normalize_eps)
    z = F.normalize(z_raw, p=2, dim=-1, eps=normalize_eps)

    score = torch.einsum("...d,...d->...", q, z)
    logits = score / temperature
    loss_items = F.binary_cross_entropy_with_logits(
        logits,
        targets.float(),
        reduction="none",
        pos_weight=pos_weight_tensor_or_none,
    )
    ledger_loss = masked_weighted_mean(loss_items, valid_pair_mask)
```

Because `q` and `z` are L2-normalized, the score is bounded near `[-1, 1]`.
The logit magnitude is therefore bounded near `1 / temperature`, so projection
norms cannot inflate the logits. This is the main numerical-stability reason
to prefer normalized dot products over raw dot products for v0.

### Defaults And Guards

Keep the user-selected first-pilot weight:

```yaml
objective:
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
```

Validation should reject non-finite values, reject `temperature < 0.05`, and
warn or debug-flag `temperature < 0.1`. A `temperature` of `0.2` gives max
logit magnitude about `5`; `0.1` gives about `10`; `0.05` gives about `20` and
should be treated as the hard lower edge for v0. `normalize_eps` should default
to `1e-6` and reject values below `1e-8`.

Do not auto-balance positives in v0. Keep `pos_weight: 1.0`, log the class
balance, and revisit weighting only if the first overfit run shows a clear
imbalance failure.

### Monitoring Metrics

`ledger_auc` is available from one forward pass if the ledger path already
materializes detached ledger logits, binary targets, and a valid-pair mask. It
does not require another model call and should be included as a monitoring
metric if it can be implemented without adding a new dependency. It is not an
acceptance gate.

Prefer an explicit scoped metric key such as
`teacher_forcing/ledger/auc_batch` unless docs define `ledger_auc` as a local
batch/window metric. If a batch has no positives or no negatives, omit the AUC
metric or emit a zero-weight metric event; do not log `0.0`.

Thresholded accuracy should also be implemented for v0, using the same logits,
targets, and valid-pair mask. The default threshold is logit `0.0`, equivalent
to sigmoid probability `0.5`.

Other first-run metrics:

- `teacher_forcing/loss/coverage_state`
- `teacher_forcing/loss/coverage_state_weighted`
- `teacher_forcing/loss/region_anchor`
- `teacher_forcing/loss/region_anchor_weighted`
- `teacher_forcing/loss/coverage_ledger_auxiliary_weighted`
- `teacher_forcing/ledger/pair_count`
- `teacher_forcing/ledger/positive_count`
- `teacher_forcing/ledger/negative_count`
- `teacher_forcing/ledger/positive_frac`
- `teacher_forcing/ledger/covered_score_mean`
- `teacher_forcing/ledger/uncovered_score_mean`
- `teacher_forcing/ledger/covered_minus_uncovered_margin`
- `teacher_forcing/ledger/accuracy`
- `teacher_forcing/ledger/auc_batch`
- `teacher_forcing/ledger/logit_abs_max`
- `teacher_forcing/ledger/score_min`
- `teacher_forcing/ledger/score_max`
- `teacher_forcing/ledger/score_abs_max`

The implementation should fail fast on non-finite raw projections, normalized
vectors, logits, or loss, and should debug-fail if `score_abs_max > 1.001`
because that usually means normalization or masking is wrong.

## 2026-06-23 Audit Refinement: Closed Template Requirement

A critical audit found that the first reviewed design spec was internally
inconsistent: it required row-completion ledger states at `<|box_end|>` while
also saying v0 would not change template bytes. The active plain `compact`
Stage-1 route does not emit `<|object_ref_end|>` or `<|box_end|>`, so it cannot
support the requested ledger semantics.

Resolved decision:

- v0 ledger uses `compact_object_box_closed`;
- both `<|object_ref_end|>` and `<|box_end|>` are mandatory;
- `compact_object_box_closed_lines` is not the default because newline behavior
  adds a variable;
- plain `compact`, `compact_box_closed`, and historical `compact_full` are not
  valid v0 ledger templates;
- `compact_full` refers to older chat-template/schema artifacts and should not
  be used as the current Stage-1 semantic `detection_template.id`;
- the pilot must include a paired no-ledger closed-template baseline with the
  same profile, data, seed, processor policy, packing policy, and model/checkpoint.

The super-power spec was refined to add a `CoverageLedgerHead(nn.Module)`
ownership contract, concrete Qwen same-forward capture requirements, metric
producer examples, structured sidecar extraction rules, and a reproducible
128-sample smoke recipe.
