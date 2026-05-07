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

The next compact-only prefix roll-in multi-positive objective is tracked by the
repo-local super-power plan/spec under `docs/superpowers/`. It must be
implemented through the latest compact detection stack rather than by reviving
this retired continuation implementation.
