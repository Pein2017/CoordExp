# Next-Object Steering Design

## Objective

Revise the painted-GT / predicted-ledger exploration into a consistent
training and inference contract for next-object steering. The central question
is whether an external visual ledger, optionally paired with a language ledger,
helps Qwen3-VL continue object enumeration without collapsing into premature
`<|im_end|>` or drifting under language priors.

This document is a planning artifact only. It authorizes OpenSpec and
implementation planning updates, not source implementation or GPU launch by
itself.

## Canonical Unit

The canonical unit is `NextObjectSteeringStep`.

Each step gives the model:

- a current image state;
- a current committed-object state;
- a `steering_context_policy`;
- one prompt asking for the next visible object instance that has not already
  been marked.

The model must produce exactly one of two supervised target forms:

```text
object_step:
  row(X) + <|object_ref_start|>

terminal_step:
  <|im_end|>
```

There is no third legal effective supervised target. For object steps, any
structural chat-template `<|im_end|>\n` that exists only to close the training
message must be masked from loss.

`row(X)` is the complete CoordExp compact row:

```text
<|object_ref_start|>desc<|object_ref_end|><|box_start|>coords<|box_end|>
```

The trailing `<|object_ref_start|>` after `row(X)` is a supervised lookahead
continuation sentinel. It is a real Qwen vocabulary token with a unique id. It
is not part of the committed ledger row.

## Context Policies

Two policies are first-class and must share prompt, painter, target grammar,
parser, metrics, and artifact schema:

- `text_image`: the model sees the current painted image plus canonical
  committed rows as assistant prefix.
- `image_only`: the model sees the current painted image with an empty
  assistant prefix at every step.

Primary results must train and evaluate matching policies:

```text
text_image training -> text_image rollout
image_only training -> image_only rollout
```

Cross-policy runs are diagnostic only.

## Prompt Contract

The V1 prompt must ask for one next unmarked object, not all objects. It must
state that magenta rectangles and yellow center points mark already committed
objects, that marked objects must not be emitted again, and that overlapping or
partially occluded uncommitted objects should still be emitted when visibly
distinguishable.

The prompt must define terminal behavior:

```text
Return <|im_end|> only when every visible object instance has already been
committed.
```

The same prompt identity must be used for training and inference unless a run
is explicitly labeled as a prompt sensitivity panel.

## Training Pipeline

For each image, V1 uses `geo_sorted` schedule order.

For object step `t`:

```text
image_t = source image painted with committed GT objects before t
prefix_t =
  text_image: canonical GT rows before t
  image_only: empty
target_t = row(current GT object) + <|object_ref_start|>
labels on = row(current GT object) and trailing <|object_ref_start|>
labels off = prefix text, structural chat-close <|im_end|>, structural newline
```

For terminal step `T`:

```text
image_T = source image painted with all GT objects
prefix_T =
  text_image: all canonical GT rows
  image_only: empty
target_T = <|im_end|>
labels on = <|im_end|>
labels off = prefix text and following newline if present
```

Materialization must fail fast when an effective target is anything other than
the two legal forms above.

## Inference Pipeline

Inference uses model-predicted boxes for state transition. It must not use GT
object order, GT count, or GT boxes to decide what to paint or when to stop.
GT is used only for scoring after rollout.

Each committed step performs full prefill because the image changes after
painting. KV cache may be used inside a single step's token generation, but not
across committed steps.

At each step:

```text
prefix =
  text_image: concat(canonical committed prediction rows)
  image_only: empty

decode one steering step with max_new_tokens=384
```

Controller rules:

- raw generated tokens are inspected before parser stripping;
- first generated token `<|im_end|>` stops the image with
  `controller_outcome=terminal_stop`;
- backend generation stop is recorded separately as
  `generation_stop_reason=im_end|length|...`;
- otherwise parse the generated stream;
- drop malformed candidates;
- commit the first valid complete row if one exists;
- strip and record any trailing `<|object_ref_start|>` sentinel;
- if `row + <|im_end|>` appears, commit the row and record
  `controller_outcome=post_row_im_end_violation`, but do not treat it as
  terminal authority even when `generation_stop_reason=im_end`;
- if no valid row exists and first token is not `<|im_end|>`, record
  `no_progress_malformed` and continue with unchanged state up to
  `no_progress_limit=2`;
- duplicate rows are committed, painted, and plotted; no duplicate guard is
  applied in V1;
- if multiple valid rows are emitted, commit only the first valid row and
  record `multi_row_violation`.

Rollout MUST also declare a GT-independent `max_rollout_steps` before decoding.
V1 defaults to `64` steps per image unless an explicitly stricter value is
recorded in the config and decode identity. Hitting this cap records
`controller_outcome=step_cap_stop`. Exhausting `no_progress_limit` records
`controller_outcome=no_progress_limit_stop`. Neither cap may depend on the GT
object count for the image.

The final prediction set for one image is exactly the committed rows.

## Painter Contract

V1 uses shared `outline_center_flat_v1` for training and inference:

- opaque magenta rectangle outline;
- opaque yellow center point;
- no fill;
- no opacity accumulation;
- no darkening or brightening in overlapping regions;
- no visible class text, object id, or order cue.

Overlap is handled by prompt and diagnostics, not by visual intensity coding.
Artifacts should record geometry-based overlap features so metrics can be
sliced by whether the next target or committed prediction overlaps previously
committed painted regions. Required V1 overlap fields are:

- target/prediction bbox overlap with the committed bbox union;
- target/prediction center inside the committed bbox union;
- target/prediction committed-overlap area ratio.

Rendered-pixel overlap is a separate visual-audit diagnostic and must not be
used as a substitute for geometry-based overlap fields.

## Metrics And Artifacts

Primary image-level metrics remain mAP, mRecall, precision, recall, and F1 over
committed rows. Steering diagnostics are required:

- step index quality;
- emitted objects per image;
- first-token `<|im_end|>` rate;
- terminal step distribution;
- malformed/no-progress rate;
- multi-row violation rate;
- duplicate rate;
- post-row `<|im_end|>` violation rate;
- overlap-conditioned metrics.

Duplicate diagnostics distinguish at least:

- `exact_row_duplicate`: identical normalized compact row text;
- `near_prediction_duplicate`: same normalized description and bbox IoU above
  the configured duplicate threshold;
- `same_gt_duplicate`: post-hoc multiple predictions assigned to the same GT
  object by the scorer.

Terminal diagnostics distinguish a correct end from premature coverage collapse:
`terminal_correct`, `terminal_premature`, `remaining_gt_at_terminal`,
`matched_gt_at_terminal`, `coverage_at_terminal`, `stop_without_any_commit`,
`step_cap_stop`, and `no_progress_limit_stop`.

Every rollout artifact must preserve raw generated text, parsed candidates,
dropped malformed candidates, committed row, ignored extra rows, paint source,
prefix source, image transition, and the resolved prompt/decode identities.
It must also preserve both `generation_stop_reason` and `controller_outcome`.

Matched `text_image` and `image_only` runs must emit parity evidence before
training or rollout claims: identical `(image_id, step_index, target_object_id,
terminal)` row sets for materialization, token-length distributions, target-span
truncation counters, dropped-example counters, non-ignored label counts, and
terminal-step counts. Target-span truncation blocks the run.

## Non-Goals

- Do not rerun ordinary one-shot autoregressive detection as the central
  comparison; it is a reference baseline only.
- Do not add overlap-depth color encoding in V1.
- Do not add duplicate suppression or salvage as primary behavior.
- Do not introduce cross-step KV reuse.
- Do not use syntax-constrained decoding for primary evidence.

## Approval State

The user approved:

- `geo_sorted` V1 schedule;
- `object_step = row(X) + <|object_ref_start|>` with structural `<|im_end|>`
  masked;
- `terminal_step = <|im_end|>`;
- first-class `text_image` and `image_only` policies;
- shared flat painter;
- malformed stripping with bounded no-progress continuation;
- duplicate plotting without guard;
- first-valid commit for multi-row outputs;
- step-local decode cap `384`;
- GT-independent `max_rollout_steps=64` default for next-object rollout;
- separate `generation_stop_reason` and `controller_outcome` artifact fields;
- overlap handled through prompt and diagnostics, not color-depth encoding.
