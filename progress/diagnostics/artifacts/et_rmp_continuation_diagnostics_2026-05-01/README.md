---
doc_id: progress.diagnostics.artifacts.et_rmp_continuation_diagnostics_2026_05_01
layer: progress
doc_type: artifact-index
status: supporting
domain: stage1-et-rmp-ce
summary: Copied summary artifacts for the old ET-RMP continuation, repetition-penalty, FN latent, length-bias, and stop-control diagnostics.
tags: [stage1, et-rmp-ce, artifacts, diagnostics]
updated: 2026-05-01
---

# ET-RMP Continuation Diagnostics Artifact Copies

Parent note:

- [../../2026-04-29_et_rmp_rp_continuation_bias_hypothesis.md](../../2026-04-29_et_rmp_rp_continuation_bias_hypothesis.md)

These files are copied summaries from `temp/` and `output_remote/`-adjacent
diagnostic runs. The parent note is the canonical source. These artifacts are
kept so the parent note remains auditable if scratch directories are cleaned.

## Files

- `core6_deterministic_sweep_summary.md/json`
  - deterministic core-6 base-vs-ET RP/temp sweep.
- `core6_stochastic_sweep_summary.md/json`
  - stochastic core-6 base-vs-ET RP/temp/seed sweep.
- `latent_probe_summary.md/json`
  - FN continuation and visual-mask sensitivity summaries.
- `length_bias_summary.md`
  - close-vs-continue boundary length/count analysis.
- `length_bias_rows.json`
  - row-level cache behind the length-bias summary.
- `stop_control_summary.md/json`
  - strict parse/eval result for hard stop-control variants.
- `stop_control_salvage_summary.md/json`
  - diagnostic-only salvage result from raw token traces.
