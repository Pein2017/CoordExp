# Benchmark Interpretation

Use this when comparing checkpoints, score policies, model variants, or keep/drop decisions.

## Scope Labels

Always record checkpoint, dataset slice, limit/full scope, bbox format, prompt/template, decode surface, repetition penalty, temperature, GPU/shard shape, and whether evidence is final, partial, repaired, or a temporary probe.

## Metric Decomposition

Report raw and guarded metrics separately. Do not collapse AP, recall, and duplication into one "better" verdict.

Include when relevant:

- AP/AP50/AP75/AR;
- recall/FN/F1-ish for leakiness questions;
- predicted object count;
- parse/drop/materialization validity;
- duplicate suppression totals and p95/p99/max predictions per image;
- confidence/scoring provenance.

## Scoring Coverage

Before interpreting a score-fusion knob, verify the artifact coverage it depends on. For example, do not claim `desc+bbox` effects unless desc scores are actually non-null in the scored artifacts.

## Stop Rules

- `enough for algorithm design`: bottleneck is localized and next action is a small objective/probe.
- `needs one narrow probe`: one discriminating measurement remains.
- `do not interpret yet`: artifact validity, baseline parity, or scope labels are unresolved.
- `archive/pause`: direction is noncompetitive or moves difficulty without closing the measured gap.

## Negative Direction Archive

Before cleanup, preserve metrics, implemented surfaces, failure diagnosis, reopen criteria, and whether the branch or worktree was kept.
