# Qwen3 Vision-Language Dense Enumeration Bottleneck

This investigation asks why object-level recognition and localization do not
reliably yield complete, non-repeating dense-scene autoregressive enumeration.

## Reading Path

- [Current research compass](compass.md) — current questions, bounded findings,
  and next discriminators.
- [Historical hypothesis and evidence atlas](overview.md) — mechanism candidates,
  terminology, and the earlier evidence chain.
- [Experiment router](experiments/index.md) — original protocols, results, and
  partial technical records, with each unit retaining its own evidence scope.
  The [experiment directory](experiments/) retains supplementary unit files.
- [Dated July 13–16 research report](2026-07-13-to-2026-07-16-weekly-research-report.md)
  — integrated historical interpretation and handoff for that sequence.

## Upstream Evidence

- [Autoregressive binding template study](../../archive/autoregressive-binding-template-study/)
- [Painted Ground-Truth transcription probe](../../ideas/qwen3-vl-painted-gt-transcription-probe/)
- [Separate selection, transcription, commit, and stop](../../decisions/separate-selection-transcription-commit-and-stop.md)
- [Require target-specific causal consumption](../../decisions/require-target-specific-causal-consumption.md)
- [Let architecture emerge from hypothesis gates](../../decisions/let-architecture-emerge-from-hypothesis-gates.md)
- [Use bagging as an object-support probe](../../decisions/use-bagging-as-an-object-support-probe.md)

## Authority

Everything in this investigation is non-normative research. Runtime artifacts
remain under `outputs/research/`; reusable implementation contracts, if later
justified, require a separate OpenSpec change.
