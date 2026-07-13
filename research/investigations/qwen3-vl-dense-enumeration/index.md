# Qwen3 Vision-Language Dense Enumeration Bottleneck

This investigation asks why a geometry-sorted Qwen3 Vision-Language
(`Qwen3-VL`) detector can describe and localize many individual Common Objects
in Context 80-category (`COCO-80`) objects, yet becomes conservative and
unstable when it must enumerate dense scenes in one autoregressive rollout.

The immediate target is diagnosis, not a final architecture. We first separate
three policy families that ordinary tiled inference confounds:

1. input-level spatial restriction;
2. cumulative accepted-row prefix policy;
3. matched-call repeated sampling or bagging gain.

These policy contrasts do not yet isolate post-vision candidate competition or
pure language-history length. Those require same-feature spatial intervention
and content-matched prefix experiments, respectively.

## Reading Path

1. [Overview and hypothesis map](overview.md)
2. [Experiment units](experiments/)
3. [Masked spatial-policy and accepted-row prefix-policy unit](experiments/2026-07-13-spatial-scope-history-disentanglement/unit.md)
4. [Readiness amendment](experiments/2026-07-13-spatial-scope-history-disentanglement/readiness-amendment.md)
5. [Independent-review synthesis and revision gate](experiments/2026-07-13-spatial-scope-history-disentanglement/review.md)

## Upstream Evidence

- [Autoregressive binding template study](../autoregressive-binding-template-study/)
- [Painted Ground-Truth transcription probe](../../ideas/qwen3-vl-painted-gt-transcription-probe/)
- [Separate selection, transcription, commit, and stop](../../decisions/separate-selection-transcription-commit-and-stop.md)
- [Require target-specific causal consumption](../../decisions/require-target-specific-causal-consumption.md)
- [Let architecture emerge from hypothesis gates](../../decisions/let-architecture-emerge-from-hypothesis-gates.md)

## Authority

Everything in this investigation is non-normative research. Runtime artifacts
remain under `outputs/research/`; reusable implementation contracts, if later
justified, require a separate OpenSpec change.
