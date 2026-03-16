## Why

CoordExp already has a stable decoded-box geometry path for Stage-2:

- decoded continuous `xyxy` boxes come from the existing coord decode path,
- canonicalization already happens before IoU-sensitive geometry loss,
- and the `bbox_geo` module already owns matched box supervision in the
  teacher-forcing pipeline.

What the stack does not currently supervise is **box size agreement in
log-space**.
That leaves a gap the user request is trying to close:

- keep the current `xyxy` parameterization and decode format unchanged,
- keep the current coord-token vocabulary and current
  `bbox_2d=[x1,y1,x2,y2]` top-left/bottom-right expression unchanged,
- keep existing SmoothL1 / CIoU / coord losses unchanged,
- but add a small auxiliary term that explicitly prefers matched predicted
  widths / heights (and optionally area) to agree with GT or matched targets
  after decode.

The repo exploration shows the Stage-2 side is straightforward:

- `src/trainers/teacher_forcing/geometry.py` already owns canonicalized decoded
  box math,
- `src/trainers/teacher_forcing/modules/bbox_geo.py` already has predicted and
  target `xyxy` boxes at the right abstraction level,
- `stage2_two_channel` and `stage2_rollout_aligned` already reuse the same
  `bbox_geo` pipeline module.

The only real ambiguity is Stage-1.
Stage-1 currently does not carry the same bbox-group metadata that Stage-2 uses,
and the repo supports both bbox and polygon-style geometry.
So a safe Stage-1 implementation needs one explicit contract decision before
coding:

- either Stage-1 is allowed to assume bbox-only supervision for the target
  datasets,
- or Stage-1 must gain explicit bbox grouping metadata so the new loss never
  guesses box boundaries from raw coord-token counts.

## What Changes

This change proposes a conservative, modular decoded-box size auxiliary with
three layers:

1. shared decoded-box helper(s) in `src/trainers/teacher_forcing/geometry.py`
2. a new plugin-style loss module `bbox_size_aux` for pipeline-driven trainers
3. a thin Stage-1 plugin host / adapter that reuses that same plugin
   implementation once bbox grouping is unambiguous

Recommended semantics:

- primary matched term:
  - `log_wh_weight`
  - SmoothL1 on `log(width)` and `log(height)` after canonicalization
- optional matched area term:
  - `log_area_weight`
  - SmoothL1 on `log(width * height)`
- optional weak oversize regularizer:
  - thresholded only,
  - off by default,
  - never a default global small-box prior
- current expression and coord-token contract stay unchanged:
  - bbox remains `bbox_2d: [x1, y1, x2, y2]`
  - the plugin consumes the current four coord slots in that order
  - coord tokens remain `<|coord_k|>` for `k ∈ [0, 999]`
  - norm1000 semantics remain `(0, 0)=top-left`, `(999, 999)=bottom-right`
  - canonicalization is internal-only for stable loss computation

Recommended config shape:

- Stage-1:
  - `custom.bbox_size_aux.*`
- Stage-2 two-channel authored YAML knob:
  - `stage2_ab.pipeline.objective`
  - add an entry with `name: bbox_size_aux`
- rollout-aligned authored YAML knob:
  - `rollout_matching.pipeline.objective`
  - add an entry with `name: bbox_size_aux`

This intentionally differs from adding new flat top-level knobs for Stage-2.
The repo already treats Stage-2 objective behavior as pipeline-owned, and the
proposal keeps that contract intact while also keeping the new size loss as its
own reversible plugin module rather than hardwiring more behavior into
`bbox_geo`.

## Locked V1 Decisions

This draft now assumes:

1. `Stage-1 is bbox-only for this plugin path`
   - the Stage-1 adapter may rely on bbox-only grouping,
   - enabling the plugin on non-bbox geometry remains a fail-fast condition.

2. `Channel-A A1 support lands immediately`
   - the plugin covers:
     - Channel-A self-context (`A2`)
     - Channel-A anchor forward (`A1`) when explicit A1 weights are non-zero
     - Channel-B rollout / Hungarian-matched supervision
     - rollout-aligned matched supervision

## Impact

If approved, this change keeps the patch intentionally narrow:

- no tokenizer or sequence-format changes,
- no coord-vocab changes,
- no decode-order changes,
- no replacement of existing geometry losses,
- no strong small-box prior by default,
- no large trainer refactor.

The main user-visible outcome is:

- a new opt-in decoded-box size plugin that is shared across training flows,
- the same plugin/module semantics across Stage-1, Stage-2, and rollout-aligned
  training,
- strict config surfaces for Stage-2 pipeline users,
- and a Stage-1 integration path that is explicit about the bbox-grouping
  assumption instead of silently guessing.
