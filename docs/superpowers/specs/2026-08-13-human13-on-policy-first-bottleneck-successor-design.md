# Human-13 On-Policy First-Bottleneck Successor Design

This design is approved for implementation by the user's instruction to
continue the Human-13 overfit route.  Scientific meaning and completion are
owned by the
[research unit](../../../research/investigations/qwen3-vl-dense-enumeration/experiments/2026-08-13-human13-on-policy-first-bottleneck-successor/unit.md)
and [OpenSpec change](../../../openspec/changes/add-human13-on-policy-first-bottleneck-successor/).

The implementation extends the experiment-local Human-13 spine with four deep
modules rather than a new trainer:

1. a content-addressed current-frontier builder;
2. a packed-prefilter/HF-decision candidate scorer and forced-continuation
   selector;
3. full-row and first-bottleneck loss adapters over the existing packed runner;
4. an in-memory training transaction that accepts or restores the complete
   optimizer state after fresh clean-greedy evaluation.

The vertical slice is the first real execution.  It must include a deliberate
rollback drill.  Only then may two independent safe arms run for at most eight
attempts.  No ungated arm, positive terminal target, K-miss target, validation
run, checkpoint promotion, or generalized framework is part of this change.

Fable-5-xhigh reviewed the decision and returned conditional PASS after two
mandatory repairs now incorporated here: scientific selection occurs on HF
fp32/SDPA rather than packed BF16/FA2, and owner preservation is a direct
clean-greedy transaction gate that restores AdamW moments on rejection.
