---
doc_id: progress.audits.index
layer: progress
doc_type: router
status: canonical
domain: research-history
summary: Router for structured audit notes and decision-focused reviews.
tags: [progress, audits, review]
updated: 2026-05-03
---

# Audits Index

Use this folder when the main question is:

- what did a focused audit conclude about a training or logging surface?
- which historical review note drove a later config or implementation correction?

Prefer `progress/diagnostics/` for open-ended mechanism work and failure
investigation. Use `progress/audits/` when the note is a tighter review or
decision record.

## Current Contents

- [2026-05-14-instance-trie-gaussian-post-implementation-audit.md](2026-05-14-instance-trie-gaussian-post-implementation-audit.md)
  - concluded implementation-contract audit for Instance-Trie Gaussian SoftCE before target-shape audit and smoke
- [2026-05-14-instance-trie-gaussian-smoke-behavior-audit.md](2026-05-14-instance-trie-gaussian-smoke-behavior-audit.md)
  - concluded smoke and DDP8 preflight behavior audit for Instance-Trie Gaussian SoftCE launch readiness
- [2026-05-03_type_schema_architecture_audit.md](2026-05-03_type_schema_architecture_audit.md)
  - active type-system and schema-boundary audit for raw domain containers
- [2026-01-22_stage1_softce_logging.md](2026-01-22_stage1_softce_logging.md)
  - historical audit of Stage-1 SoftCE logging, scaling, and efficiency
- [2026-02-25_stage2_channel_a_coord_loss.md](2026-02-25_stage2_channel_a_coord_loss.md)
  - historical audit / decision note for Channel-A coord-loss behavior
