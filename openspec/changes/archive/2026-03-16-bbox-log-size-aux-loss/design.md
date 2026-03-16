## Context

Repo exploration surfaced four concrete integration handles:

- shared decoded-box math:
  - `src/trainers/teacher_forcing/geometry.py`
- Stage-2 decoded-box supervision module:
  - `src/trainers/teacher_forcing/modules/bbox_geo.py`
- Stage-2 two-channel trainer wiring:
  - `src/trainers/stage2_two_channel.py`
- Stage-1 auxiliary-loss wiring pattern:
  - `src/trainers/metrics/mixins.py`
  - `src/sft.py`

Two existing representation contracts must remain fixed:

- coord tokens stay on the current `0..999` grid with the `k/999` decode rule,
- bbox stays in the current `bbox_2d: [x1, y1, x2, y2]` expression, where the
  expression layer is top-left then bottom-right and canonicalization remains
  an internal robustness step rather than a format change.

Current behavior is already close to the requested feature on the Stage-2 side:

- `_decode_groups(...)` in `bbox_geo.py` decodes predicted boxes from coord
  logits, reconstructs target `xyxy` boxes from target bins, and compares them
  group-by-group.
- `canonicalize_bbox_xyxy(...)` and `bbox_smoothl1_ciou_loss(...)` already
  provide canonicalization and numerical-stability rules for decoded box loss.
- `stage2_two_channel` Channel-A and Channel-B both converge on `bbox_geo`.
- `stage2_rollout_aligned` uses the same teacher-forcing pipeline and therefore
  can reuse the same `bbox_geo` extension.

The Stage-1 side is the only non-trivial design point:

- Stage-1 currently adds optional aux loss through mixins attached in
  `src/sft.py`.
- The current Stage-1 aux pattern (`coord_soft_ce_w1`) consumes logits + labels.
- Stage-1 does not currently carry `bbox_groups_prefix` / `bbox_groups_fn`
  metadata, and the repo permits non-bbox geometry modes.

## Goals

- Keep `xyxy` as the only bbox parameterization.
- Keep the current coord-token vocabulary and current
  `bbox_2d=[x1,y1,x2,y2]` top-left/bottom-right expression authoritative.
- Compute the new term from decoded continuous boxes after canonicalization.
- Reuse one shared helper across Stage-1, Stage-2, and rollout-aligned paths.
- Keep the new bbox-size loss plugin/module-based across all training stages.
- Preserve the current Stage-2 pipeline-first config model.
- Keep oversize regularization opt-in and weak.

## Non-Goals

- No switch to `cxcywh`.
- No tokenizer, coord-vocab, or decode-protocol changes.
- No broad trainer refactor.
- No silent weight change to existing SmoothL1 / CIoU / coord terms.
- No default all-boxes-should-be-small prior.

## Decisions

### 1. Shared math should live in `teacher_forcing/geometry.py`

Recommended helper surface:

- `compute_bbox_log_size_loss(...)`
- `compute_bbox_oversize_penalty(...)`

Why this location:

- it already owns canonicalized decoded-box loss math,
- it is already imported by both `bbox_geo` and Stage-2 trainer helpers,
- and it keeps the new loss independent from any one trainer variant.

Recommended helper semantics:

- canonicalize both predicted and target boxes first,
- compute `w = clamp(x_hi - x_lo, min=eps)` and
  `h = clamp(y_hi - y_lo, min=eps)`,
- use `torch.nan_to_num(...)` on returned scalars,
- accept an optional mask / weights tensor,
- return both scalar loss tensors and lightweight summary stats for logging.

### 2. Stage-2 should add `bbox_size_aux` as a separate plugin module

Recommendation:

- keep `bbox_geo` focused on the current SmoothL1 + CIoU geometry objective,
- add a new plugin-style objective module `bbox_size_aux`,
- let `bbox_geo` publish decoded canonicalized box state,
- let `bbox_size_aux` consume that state and emit the new size atoms.

Authoring surface:

- for `custom.trainer_variant: stage2_two_channel`, the authored YAML knob is
  `stage2_ab.pipeline.objective` with a `name: bbox_size_aux` entry,
- for `custom.trainer_variant: stage2_rollout_aligned`, the mirror surface
  remains `rollout_matching.pipeline.objective`.

Why:

- it matches the user's plugin-first requirement more closely,
- it keeps the new loss independently enable-able and easier to revert,
- it avoids bloating `bbox_geo` into a catch-all module,
- and `stage2_two_channel` plus `stage2_rollout_aligned` can still share the
  same implementation through the existing pipeline surface.

Recommended `bbox_size_aux.config` additions:

- `log_wh_weight`
- `log_area_weight`
- `oversize_penalty_weight`
- `oversize_area_frac_threshold`
- `oversize_log_w_threshold`
- `oversize_log_h_threshold`
- `eps`

Channel scoping remains the existing pipeline contract:

- `channels: [A]`
- `channels: [B]`
- or `channels: [A, B]`

That avoids inventing new Stage-2 flat booleans while keeping the new size loss
as its own authored objective plugin.

### 3. Stage-1 uses a lightweight plugin host and is bbox-only in v1

Locked v1 direction:

- keep the current single Stage-1 forward,
- reuse the canonical `bbox_size_aux` plugin implementation through a thin
  adapter / host after the base forward,
- keep the Stage-1 scope bbox-only for this plugin path.

Why:

- it preserves the plugin/module requirement across stages,
- it avoids inventing a second divergent Stage-1-only loss definition,
- and it keeps the v1 data contract simple.

V1 fail-fast rule:

- if `custom.bbox_size_aux.enabled=true` and non-bbox geometry is encountered,
  the Stage-1 path errors immediately instead of trying to infer mixed-geometry
  grouping semantics.

### 4. Stage-1 config should follow the existing nested-aux pattern

Recommendation:

- add a nested config block under `custom`:
  - `custom.bbox_size_aux`
- implement Stage-1 as a thin adapter that reuses the same canonical
  `bbox_size_aux` plugin logic rather than creating a second bespoke size-loss
  implementation.

Recommended keys:

- `enabled`
- `log_wh_weight`
- `log_area_weight`
- `oversize_penalty_weight`
- `oversize_area_frac_threshold`
- `oversize_log_w_threshold`
- `oversize_log_h_threshold`
- `eps`

Why not flat `stage1_enable_bbox_log_wh_loss`-style keys:

- `CustomConfig` already groups Stage-1 auxiliary objectives as nested blocks,
- the repo strongly prefers config-first typed sections over loose flag
  proliferation,
- and execution can still be plugin/module-based even if Stage-1 authoring
  stays nested under `custom` in v1.

### 5. Logging should expose the same geometry atom family across stages

Recommended Stage-2 objective atoms from the `bbox_size_aux` plugin:

- `bbox_log_wh`
- `bbox_log_area`
- `bbox_oversize`

Why:

- it keeps Stage-2 additivity auditable in the existing objective-atom contract,
- and it lets Stage-1 emit recognizable plugin-owned geometry atoms instead of
  inventing a second naming scheme for the same loss math.

Recommended key shape:

- Stage-1 single-forward host:
  - `loss/geo/{bbox_log_wh,bbox_log_area,bbox_oversize}`
- Stage-2 / rollout-aligned:
  - existing provenance-split plugin atoms such as
    `loss/A2_coord/{bbox_log_wh,bbox_log_area,bbox_oversize}` and
    `loss/B_coord/{bbox_log_wh,bbox_log_area,bbox_oversize}`

### 6. The external bbox contract must stay the current tl-br expression

Recommendation:

- keep the current coord tokens `<|coord_k|>` authoritative,
- keep the current `bbox_2d=[x1,y1,x2,y2]` expression authoritative,
- allow internal canonicalization for loss stability only,
- do not introduce `cxcywh` or any reordered bbox slot contract.

Concrete handles:

- `docs/data/CONTRACT.md`
- `openspec/specs/coord-utils/spec.md`
- `src/datasets/geometry.py`

## Locked V1 Choices

### 1. Stage-1 bbox-only scope

The spec now assumes bbox-only Stage-1 usage for this plugin.
Mixed bbox/poly Stage-1 support is not part of the locked v1 scope.

### 2. Immediate A1 support

The spec now assumes immediate optional A1 support for `bbox_size_aux`.

Recommended `bbox_size_aux.config` additions for this:

- `a1_log_wh_weight`
- `a1_log_area_weight`
- `a1_oversize_penalty_weight`

Why this shape:

- it mirrors the repo's existing explicit A1 geometry/coord ablation pattern,
- it keeps A1 opt-in,
- and it avoids broadening the main `A2/B` weights to cover anchor-forward
  behavior accidentally.

### 3. Naming recommendation

Keep the plugin/module identity as `bbox_size_aux`.

Reason:

- `area_loss` is too narrow because the plugin is not only area,
- the design includes width/height, area, and optional oversize terms,
- and `bbox_size_aux` stays formulation-agnostic while the atom names remain
  precise:
  - `bbox_log_wh`
  - `bbox_log_area`
  - `bbox_oversize`
