---
doc_id: docs.training.stage1-et-rmp-ce
layer: docs
doc_type: historical-note
status: superseded
domain: training
summary: Superseded Stage-1 continuation objective note. Active implementation now lives in latest compact detection planning and code.
updated: 2026-05-07
---

# Superseded Stage-1 Continuation Objective

This page is intentionally historical-only.

The previous continuation implementation has been removed from the active code,
config, runtime, and test surface. Provenance remains available through git
history, archived progress notes, and stored run artifacts, but this file no
longer provides a runnable route or recommended current training profile.

Current Stage-1 objective routing lives in:

- `docs/training/README.md`
- `docs/training/STAGE1_OBJECTIVE.md`
- `configs/stage1/recursive_detection_ce_latest/`
- `src/detection/runtime.py`
- `src/detection/objective.py`
- `src/detection/loss.py`

Do not use this page as the route for `prefix_rollin_et_rmp_ce`. The current
compact-only ablation route is documented in `STAGE1_OBJECTIVE.md` and
`docs/catalog.yaml`, with config
`configs/stage1/recursive_detection_ce_latest/ablation/compact_full_prefix_rollin_balance2.yaml`.
It must stay in the latest compact detection stack rather than reviving this
retired continuation implementation. Old branch-balance/support knobs and the
retired continuation trainer path must remain absent from active configs,
runtime routing, tests, and docs recommendations.
