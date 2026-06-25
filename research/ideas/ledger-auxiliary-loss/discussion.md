---
type: idea
title: Ledger Auxiliary Loss Discussion
description: Durable discussion decisions and open design gates for the ledger auxiliary loss pilot.
tags: [stage1, compact-detection, teacher-forcing, auxiliary-loss, design-gates]
state: active
updated: 2026-06-24
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

Stage-2 rollout correction and inference-time decoding are deferred until v0
produces enough evidence to justify broader contract work. For the 2026-06-24
loss/config/metric/artifact surface, OpenSpec is a pre-implementation governance
choice rather than an implicit deferral: choose the OpenSpec path or record an
explicit experiment-only exception before code edits.

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
- Do not create an OpenSpec change unless the user chooses the OpenSpec path;
  otherwise record an explicit experiment-only exception before code edits.
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

The 2026-06-24 implementation refinement supersedes the earlier positive-only
anchor note: for row `<box_start>` of object `k`, the region-anchor subterm uses
a one-vs-all row-object binding target. Object `k` is positive and every other
annotated object is negative. The goal is to bind the unique teacher-forced
object row to its language-side coordinate-start state and selected visual
region, while keeping this binding term distinct from the cumulative coverage
inventory target.

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

- Region-anchor / row-object binding pairs are one-vs-all in v0: current object
  positive, all other annotated objects negative.
- Region-anchor has its own state projection and reuses the shared visual
  object projection.
- Coverage and region-anchor subterms have separate configurable weights.
- AUC and thresholded accuracy are logged separately for cumulative coverage and
  row-object binding; both are diagnostic-only.
- Region-anchor monitoring should focus on raw loss, weighted loss, binding
  AUC/accuracy when comparable classes exist, pair count, and non-finite/shape
  checks.

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
through the shared CoordExp rounded pixel helper used by both loss pooling and
overlay artifacts:

```text
x1_px = clamp_and_round(x1 / 999 * (W_proc - 1), 0, W_proc - 1)
x2_px = clamp_and_round(x2 / 999 * (W_proc - 1), 0, W_proc - 1)
y1_px = clamp_and_round(y1 / 999 * (H_proc - 1), 0, H_proc - 1)
y2_px = clamp_and_round(y2 / 999 * (H_proc - 1), 0, H_proc - 1)
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

## 2026-06-24 Adopted Design: Exclusive Type Partition And Continuation Loss

### Keep Stop/EOS Separate From Schema

The compact teacher-forcing vocabulary should preserve four mutually exclusive
supervised token families:

- `SCHEMA` / `STRUCT`: compact structural tokens such as
  `<|object_ref_start|>`, `<|object_ref_end|>`, `<|box_start|>`,
  `<|box_end|>`, plus template separators when present.
- `COORD`: the 1000 coordinate tokens `<|coord_0|>` through
  `<|coord_999|>`.
- `TEXT` / `DESC`: free description vocabulary after excluding schema,
  coordinate, stop, pad, and other control tokens.
- `STOP` / `EOS`: the semantic assistant stop token `<|im_end|>`.

`<|im_end|>` must not be folded into schema, even though operators may
informally describe the compact control-token set as structural tokens plus
the semantic stop token. The model has to choose between continuing with
another object and stopping at object-list boundaries; merging `<|im_end|>`
with schema would hide that decision from the type-family objective.

Qwen text terminators such as `<|endoftext|>` and `<|end_of_text|>` remain
excluded or pad/control tokens for this surface. They are not semantic
detection EOS targets.

### Rationale

The current teacher-forcing path already represents compact token roles as
`SCHEMA`, `TEXT`, `COORD`, and `STOP`. `RoleVocab` rejects a stop token that
also appears in schema, text, or coordinate vocabularies. The target builder
appends `<|im_end|>` as a dedicated stop atom after all rendered object
entries. Preserving that invariant keeps semantic stop supervision distinct
from ordinary compact structure supervision.

Functionally, the object-count decision is not "is this a control token?" It
is "should the next action start another object or terminate the assistant
detection sequence?" That decision crosses token families and should remain
visible to a separate boundary objective.

### Consequence

- The planned bidirectional exclusive type loss should be a four-way
  family-mass objective over `schema`, `coord`, `desc`, and `stop`.
- At a schema position, the objective should raise schema mass and suppress
  coord, desc, and stop mass.
- At a stop position, the objective should raise stop mass and suppress
  schema, coord, and desc mass.
- Existing control/pad tokens outside these families should remain excluded
  from the trainable family partition.

### Replace The Current Decomposition With A Mandatory Added Type Term

The exclusive type-family objective should be mandatory for the active
teacher-forcing surface. It should not remain an optional experiment-only
module and should not be represented as a mere decomposition of ordinary
token-level cross entropy.

The current teacher-forcing objective computes a type component and a
conditional valid-token component whose sum algebraically collapses to ordinary
full-vocabulary valid-token negative log-likelihood. That decomposition is
useful for diagnostics, but it is not stronger than pure CE. The intended new
behavior is an added supervised family-partition pressure, for example:

```text
loss = valid_token_nll + lambda_type * exclusive_type_partition_nll + ...
```

where `exclusive_type_partition_nll` is the four-way mass objective over
`schema`, `coord`, `desc`, and `stop`. The exact implementation can still log
conditional-within-type likelihoods for analysis, but the optimization target
must include extra type-family pressure beyond the raw token CE term.

### Consequence

- Active teacher-forcing profiles should treat the type partition as mandatory.
- A pure-CE or no-type-loss run should be an explicit ablation/comparator, not
  the default active objective semantics.
- Config/schema work should either remove the active on/off meaning of
  `objective.terms.token_type_mass.enabled` or reject `false` for the new
  active profile while preserving a clearly named ablation path.
- Metrics should separately report raw valid-token NLL, exclusive type
  partition NLL, and total weighted loss so improvements are not mistaken for a
  bookkeeping decomposition.

### Mixed-Role Target Policy

For the current `hard_sft` / sorted / 128-sample overfit pilot, the mandatory
exclusive type-family objective should use the one-hot `selected_token_role` as
the target. This is the strongest version of the type consolidation signal and
matches the realized teacher-forced sequence.

For `pure_valid_set_marginal` and other valid-set profiles, do not silently use
one-hot selected-role type targets on mixed-role atoms. The current target IR
allows the specific mixed role set `TEXT + SCHEMA`, for example when one valid
object branch can continue a description token while another can close the
description and emit a schema token. For those atoms, one-hot selected-role
pressure would penalize alternate valid objects and partially undo the
marginal valid-set semantics.

The valid-set path must either derive a soft family target from valid
token/candidate weights or explicitly skip/fail/diagnose mixed-role atoms until
that target is designed. The initial hard-SFT pilot does not need to solve the
mixed-role marginal case before launch.

### Add Boundary-Specific Continue-Vs-Stop Calibration

Object-list continuation should be trained or at least measured as a separate
boundary objective, not by collapsing stop into schema or by relying only on
the generic type-family loss.

At semantic object-list boundaries, compare full-vocabulary probability mass
for the valid continuation action against `<|im_end|>`:

```text
L_continue = logsumexp(logits over valid next-object opener tokens)
L_stop     = logit(<|im_end|>)
```

Use `CONTINUE` when objects remain and `STOP` when the object list is complete.
For the common `desc_first` `compact_object_box_closed` surface, the
continuation action is `<|object_ref_start|>`, so the boundary is
`<|object_ref_start|>` versus `<|im_end|>`.

Do not hard-code `<|object_ref_start|>` as the only continuation action across
all compact variants. In `geometry_first`, a new object can begin with
`<|box_start|>`. In `compact_object_box_closed_lines`, the free autoregressive
boundary can be newline versus `<|im_end|>`, while the after-separator
diagnostic boundary is `<|object_ref_start|>` versus `<|im_end|>`.

### Consequence

- The continuation term should resolve its positive continuation set from the
  active template, field order, prefix state, and target IR rather than from a
  global literal token id.
- The existing `objective.terms.continuation_margin` name is the natural
  config slot for this behavior, but implementation must audit whether it is
  metric-only, margin-only, or binary-CE in the first version.
- Boundary metrics should report continue mass, stop mass, and
  continue-minus-stop margin separately from generic type-family loss.
- Inference should not require a ledger head for this mechanism. The intended
  effect is internalized in LLM hidden states and LoRA/token-row updates during
  training.

### Evidence Handles

- `docs/training/STAGE1_OBJECTIVE.md`
- `src/training/teacher_forcing/vocab.py`
- `src/training/teacher_forcing/roles.py`
- `src/trainers/metrics/teacher_forcing.py`
- `src/training/objectives/teacher_forcing.py`
- `src/training/teacher_forcing/probabilities.py`
- `src/detection/teacher_forcing/target_builder.py`
- `src/detection/template_contracts.py`

## 2026-06-24 Adopted Design: Hard Geometry Penalty

Use a simple hard valid-region penalty for rectangular bbox geometry. Do not
start with a pairwise distribution regularizer, decoded argmax penalty, or
other complex geometry-aware objective.

For the initial hard-SFT pilot, apply the geometry term only to tail-coordinate
slots:

```text
x2 slot: valid bins are x2 > x1
y2 slot: valid bins are y2 > y1
```

Equivalently, for a selected/teacher bbox `(x1, y1, x2, y2)` in coord-token
bins, the auxiliary punishes coordinate probability mass on:

```text
x2 <= x1
y2 <= y1
```

This keeps the loss causal and simple: at the `x2` position, `x1` is already
in the teacher-forced prefix; at the `y2` position, `y1` is already in the
teacher-forced prefix. Do not add symmetric penalties at `x1` or `y1` against
future `x2` or `y2` in v0.

### Suggested Loss Shape

Use the coordinate-vocabulary conditional distribution for the active coord
slot, since the mandatory type-family objective already handles coord-vs-
noncoord pressure:

```text
L_x2_valid = -log sum_{bin > x1} P_coord_x2(bin)
L_y2_valid = -log sum_{bin > y1} P_coord_y2(bin)
```

Skip the term when the required prefix coordinate cannot be resolved from the
structured teacher-forcing target metadata. For the active v0 hard-SFT pilot,
that should be treated as a construction bug rather than normal behavior.

### Consequence

- Geometry v0 is a hard support-mass penalty, not a soft IoU/Gibbs/CIoU-style
  coordinate loss.
- The term reinforces valid positive-area rectangles while preserving ordinary
  hard coordinate CE for the exact target bin.
- The first implementation should normalize by eligible tail-coordinate atoms
  or completed objects and log eligible/skip counts.
- Valid-set marginal and mixed-candidate geometry semantics remain deferred
  unless the builder can provide unambiguous selected-object prefix coordinates
  for the tail slot.

## 2026-06-24 Batch Decisions: V0 Loss Launch Shape

The first implementation target is the standard SFT-style hard teacher-forced
surface: `research_teacher_forcing` with `profile: hard_sft`, sorted object
order, and the closed compact template required by the ledger pilot. This is
the "standard sft" scope for these decisions; it does not mean the older
`objective.id: standard_ce` path unless a later decision explicitly changes
the implementation surface.

Resolved v0 decisions:

- Use the existing `objective.terms.token_type_mass` key for the mandatory
  exclusive type-family term.
- Do not add a `mode` key for type-family behavior in v0; the exclusive
  partition is the default behavior for this active surface.
- Use `lambda_type = 1.0`.
- Use one-hot `selected_token_role` targets for hard-SFT type-family
  supervision.
- Use hard geometry penalty with `lambda_geometry = 0.1`.
- Use coord-conditional probability for geometry support mass.
- Train `objective.terms.continuation_margin` in v0 as object-list
  continue-vs-stop calibration with `lambda_continuation = 0.2`.
- Add diagnostic post-hoc invalid-span salvage that drops invalid object spans
  instead of dropping the whole row, while keeping strict row-level metrics
  separate and official.
- For the first benchmark run, use only the ledger arm with mandatory type
  loss, geometry loss, and continuation loss. Do not run the full
  baseline/ledger/no-loss/loss matrix as the first pass.
- Tiny/train128 success criteria are mechanism-scoped: improved F1 versus the
  matching baseline, fewer strict malformed rows, better salvaged diagnostic
  F1, clearer type-family metrics, lower bbox-positive-area invalid mass,
  separated continuation/stop margins, and saved-adapter reload for inference.
- Implement in staged slices on the same branch: type partition, continuation,
  geometry, config/preflight smoke, then train128 comparison.

`objective.terms.continuation_margin` means object-list boundary calibration:
compare the model's mass for valid continuation actions against semantic stop
`<|im_end|>`, train/measure `CONTINUE` while objects remain and `STOP` when
the object list is complete.

## 2026-06-24 Batch Decisions: Implementation Contracts

The v0 objective is additive. The intended loss shape is:

```text
total =
  valid_token_nll
  + 1.0 * token_type_mass
  + 0.2 * continuation_margin
  + 0.1 * bbox_positive_area
  + coverage_ledger terms when enabled
```

Do not fold auxiliary terms into the valid-token CE denominator.

Use the existing `objective.terms` family for config:

```yaml
objective:
  terms:
    token_type_mass:
      enabled: true
      weight: 1.0
    continuation_margin:
      enabled: true
      weight: 0.2
    bbox_positive_area:
      enabled: true
      weight: 0.1
```

`objective.terms.token_type_mass.enabled=false` should be rejected for the new
active v0 config surface, but old configs should not be globally broken during
the first implementation pass. Pure CE or no-type-loss behavior must be a
clearly named ablation/comparator, not the default.

Continuation eligibility:

- apply only at object-list boundary atoms where the target IR knows whether
  objects remain;
- `CONTINUE` when a valid next-object opener is expected;
- `STOP` for the final `<|im_end|>` atom;
- do not apply this term to every schema token.

Geometry eligibility:

- apply only to `coord_role == "x2"` and `coord_role == "y2"` atoms;
- require selected-object bbox metadata;
- fail fast in the hard-SFT v0 pilot when required metadata is absent.

Metric naming should follow the existing unsuffixed convention. Do not add
parallel metric names ending in `_weighted`. Use canonical component names such
as:

```text
teacher_forcing/loss/token_type_mass
teacher_forcing/loss/continuation_margin
teacher_forcing/loss/bbox_positive_area
teacher_forcing/type/schema_mass_at_schema
teacher_forcing/type/coord_mass_at_coord
teacher_forcing/type/desc_mass_at_desc
teacher_forcing/type/stop_mass_at_stop
teacher_forcing/continuation/continue_minus_stop_margin
teacher_forcing/continuation/continue_accuracy
teacher_forcing/geometry/bbox_positive_area_invalid_mass
teacher_forcing/geometry/bbox_positive_area_eligible_count
```

If raw-vs-contribution accounting is needed, keep the distinction in the
metric payload/reduction metadata or an explicit non-`_weighted` name rather
than creating a second `_weighted` metric family.

Diagnostic post-hoc span salvage should use the named view
`compact_span_drop_salvage`. It should report strict row-level metrics
separately from salvaged span-drop diagnostics and include counters such as:

```text
rows_strict_malformed
rows_salvaged
objects_dropped_invalid_span
objects_kept_valid_span
```

The first training config should stay aligned with the existing smoke YAML
naming family instead of stacking every enabled term into the filename. The
existing local anchors are:

```text
configs/stage1/detection_teacher_forcing/smoke/coverage_ledger_closed_hard_sft_128.yaml
configs/stage1/detection_teacher_forcing/smoke/coverage_ledger_closed_hard_sft_128_baseline.yaml
```

During implementation, derive the active v0 config name from this pair and keep
it concise. Avoid names like
`coverage_ledger_closed_hard_sft_128_type_geom_cont.yaml`. Preserve comparator
provenance explicitly if the existing file is migrated.

The first smoke should require saved adapter reload before inference
comparison. Implement in staged order: token type loss and tests first,
continuation loss and tests second, geometry loss and tests third, then config
parse/preflight and the train128 comparison.

## 2026-06-24 Governance Decision: Split OpenSpec And Experiment

stable contract path: bidirectional type-family gating loss
experiment-only path: coverage ledger, continuation, bbox positive-area penalty, saved-adapter train128 smoke, and diagnostic span salvage
production eligible: no
required follow-up before promotion: archive/sync the type-gating OpenSpec after implementation evidence; separately promote any non-type ledger mechanisms before production use
approval: user correction after initially selecting experiment-only exception

The user corrected the governance split: bidirectional type gating losses should
be promoted through OpenSpec, while the ledger mechanism remains experimental.
This means the four-family exclusive type gate (`schema`, `coord`, `desc`,
`stop`) is a stable compatibility-sensitive contract. The coverage ledger,
continuation boundary loss, hard bbox positive-area penalty, saved-adapter train128
smoke, and `compact_span_drop_salvage` diagnostic view remain research-only
unless separately promoted.

## 2026-06-25 Grill Resolution: Simple Type Surface And Bbox Positive Area

Decision: keep the config surface simple. `objective.terms.token_type_mass`
does not get a `mode` key in v0. Under the active hard-SFT ledger surface,
`token_type_mass.enabled: true` means the bidirectional four-family exclusive
type objective over `schema`, `coord`, `desc`, and `stop`, trained with
one-hot `selected_token_role` targets. The public optimization stack is:

```text
sorted hard CE
+ token_type_mass
+ continuation_margin
+ bbox_positive_area
+ coverage_ledger
```

Do not enable `conditional_valid_set_likelihood` or `within_valid_coverage`
for this stack.

Rename the planned geometry term from `geometry_valid_tail` to
`bbox_positive_area`. The term enforces the positive-area xyxy invariant
`x2 > x1` and `y2 > y1` with a coord-conditional support-mass penalty only at
tail-coordinate slots:

```text
x2: -log P_coord(coord > x1)
y2: -log P_coord(coord > y1)
```

It should fail fast in the active hard-SFT pilot when required selected-bbox or
prefix-coordinate metadata is missing. Use initial weights:

```yaml
token_type_mass:
  enabled: true
  weight: 1.0
continuation_margin:
  enabled: true
  weight: 0.2
bbox_positive_area:
  enabled: true
  weight: 0.1
coverage_ledger:
  coverage_weight: 0.1
  region_anchor_weight: 0.1
```

Consequences: update the implementation plan, config schema, target-builder
metadata, objective metrics, smoke configs, and preflight allowlists to use
`bbox_positive_area`. Do not add `mode: bidirectional_family_ce`; the stronger
exclusive type-family semantics are the canonical `token_type_mass` behavior
for this active surface.
