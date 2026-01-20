# Change: Adopt loss_scale-based coord/text CE weighting

## Archive Note (2026-01-20)
This change targets the legacy `custom.coord_loss` coord/text CE weighting mode. Current Stage-1 training uses
`custom.coord_soft_ce_w1` (distribution-based supervision), and `custom.coord_loss` is removed from the runtime
config surface (configs fail fast in `src/config/schema.py`).

Accordingly, this change is archived as legacy and does not update current specs.

## Why
Manual coord/text CE weighting recomputes cross-entropy from logits and bypasses ms-swift's native per-token loss path. This adds extra compute and diverges from the upstream loss_scale mechanism that is already integrated with the trainer.

## What Changes
- Use ms-swift `loss_scale` to assign per-token CE weights for coord vs non-coord tokens.
- Emit `loss_scale` only when `coord_loss` is enabled and coord/text weights differ, preserving the default fast path when weights are 1.0.
- Disable manual coord/text CE recomputation when loss_scale-based weighting is active to prevent double weighting.
- Keep auxiliary coord losses (L1/GIoU/poly) coord-only and weighted independently (no change to their semantics).

## Impact
- Affected specs: `coord-token-mode`
- Affected code:
  - `src/data_collators/dataset_metrics.py` (attach loss_scale based on coord token IDs)
  - `src/metrics/dataset_metrics.py` (skip manual CE weighting when loss_scale is active)
- External behavior:
  - When coord/text CE weights differ, ms-swift will compute per-token loss and apply loss_scale; Liger kernel CE is not used in this mode.
  - When weights are 1.0, the default CE path remains unchanged.
