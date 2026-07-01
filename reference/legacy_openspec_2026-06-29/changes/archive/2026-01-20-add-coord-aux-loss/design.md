## Context
Stage-1 SFT already teaches JSON-format output with coord tokens. We add auxiliary geometry losses to provide continuous gradients while keeping the Qwen3-VL SFT trainer/backbone and packing intact. No detection head or rollout matching is introduced.

## Goals / Non-Goals
- Goals:
  - Provide L1 + GIoU supervision on coord-token outputs for bbox_2d and poly (poly→bbox) with L1-only for line.
  - Support packed and non-packed batches.
  - Provide coord vs non-coord CE weighting via loss_scale.
  - Log the same loss components in train and eval.
- Non-Goals:
  - True polygon IoU/GIoU (requires new geometry kernels or differentiable rasterization).
  - Hungarian matching or rollout-based alignment (Stage-2 only).

## Decisions
- **Top-k expectation decoding**: Use ordered coord token ids (<|coord_0|>.. <|coord_999|>) and compute E(top-k) over bins 0..999. `top_k` accepts a fraction (0<k<1) or integer.
- **Geometry targets**: L1 uses normalized targets in [0,1] from coord tokens. GIoU uses bbox (x1,y1,x2,y2) with min/max reordering.
- **Poly handling**: convert poly points to a bbox and apply bbox GIoU ("GIoU-style"), while L1 remains per vertex.
- **Line handling**: only L1; GIoU is skipped.
- **CE weighting**: use loss_scale to apply separate coord/non-coord weights.

## Algorithm Sketch
1) Build coord_token_ids (ordered 0..999) once from tokenizer.
2) Compute coord_position_mask from labels (coord token ids).
3) Apply CE weighting via loss_scale (coord_weight / non_coord_weight).
4) For all coord positions:
   - Gather coord logits, apply top-k softmax (temperature).
   - Compute expected bin value and normalize to [0,1].
   - L1 = |E - target| per coord token.
5) For each object span:
   - If bbox_2d: group 4 coords into box and compute GIoU.
   - If poly: group coords into points, compute bbox via min/max, then GIoU.
   - If line: skip GIoU.
6) Sum losses with weights and log in train/eval.

## Risks / Trade-offs
- Poly→bbox GIoU is not true polygon IoU; gradients affect extreme vertices only.
- Top-k expectation may be sensitive to k; defaults and logging must be clear.

## Migration Plan
- Disabled by default; enable via `custom.coord_loss.enabled: true` in YAML.
- Existing configs continue to run unchanged.

## Open Questions
- None (requirements confirmed in this change).
