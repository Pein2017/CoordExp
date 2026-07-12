# Qwen3-VL Painted-GT Transcription Probe

## Main Reading Path

- [Overview](overview.md)
- [Experiment Plan](experiment-plan.md)
- [Experiment Units](experiments/index.md)
- [Review Log](review-log.md)

## Current Decisions

- [Visual designation is a causal teacher, not the final interface](../../decisions/use-visual-designation-as-a-causal-teacher.md)
- [Selection, transcription, commit, and stop require separate evidence](../../decisions/separate-selection-transcription-commit-and-stop.md)
- [A representation must be consumed target-specifically](../../decisions/require-target-specific-causal-consumption.md)
- [Specialization must preserve native capability](../../decisions/protect-native-capability-during-specialization.md)
- [Architecture remains downstream of hypothesis gates](../../decisions/let-architecture-emerge-from-hypothesis-gates.md)

This idea remains open and does not yet have a final `conclusion.md`. The
temporary implementation branches have been retired as implementation bases;
their durable evidence is preserved here so future probes can be reimplemented
against canonical CoordExp-Swift infrastructure.

The July 5-10 records establish the painted-mark, anti-copy, feature-delta, and
residual-control evidence chain. The July 10-12 records then test native commit
state, sequential control, and the causal proposal bridge. Read the combined
[experiment index](experiments/) chronologically; later units may narrow or
demote claims from the original plan.

Raw source bytes, branch identities, historical plans, and locally available
probe receipts are preserved in the
[2026-07-12 worktree recycle bundle](../../../docs/history/worktree-cleanup/2026-07-12-pvci-research-worktree-recycle/).
