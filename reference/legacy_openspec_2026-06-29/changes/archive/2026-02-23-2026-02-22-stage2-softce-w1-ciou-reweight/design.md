## Design

### Objective Composition

Stage-2 AB already computes coord-slot distributions for supervised bbox groups in `_decode_groups(...)`. The change reuses that exact distribution tensor (`coord_logits`) to compute:
- Gaussian soft-target cross-entropy (`soft_ce`)
- 1D Wasserstein distance via CDF (`w1`)

using existing shared CoordExp utilities (`src/coord_tokens/soft_ce_w1.py`).

This keeps all coord losses on one consistent token-indexing contract (causal shift, supervised groups only) and avoids introducing a second coord-label pipeline.

### Scope Boundaries

- Apply only on Stage-2-supervised bbox groups (`bbox_groups_prefix`, `bbox_groups_fn`).
- Do not supervise FP-only/unmatched predicted coord slots.
- Keep Channel-B FP-neutral semantics unchanged.

### Config Strategy

Use existing `custom.coord_soft_ce_w1` typed config for soft-CE/W1 parameters and weights (no new CLI flags, no ad-hoc runtime knobs).

Canonical Stage-2 defaults are split by surface:
- `configs/stage2_ab/base.yaml` keeps fallback/smoke defaults (`bbox_ciou_weight: 0.5`, `soft_ce_weight: 0.02`, `w1_weight: 0.02`).
- `configs/stage2_ab/prod/*.yaml` explicitly pins production overrides (`bbox_ciou_weight: 0.2`, `soft_ce_weight: 0.2`, `w1_weight: 0.2`).
