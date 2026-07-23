# Physical-Owner Duplication Goal Checkpoint

Verified at 2026-07-22T17:17:09Z in
`/data/CoordExp/.worktrees/research-probes`.

## User steering

- Official unmatched predictions are not hallucinations by default.
- Verified false negatives and confirmed physical-owner duplication are
  actionable training signals.
- Run prefix causality analysis and direct duplication training in parallel;
  do not use one as a gate that cancels the other.
- Use the whole `valid -> duplicate burst -> valid recovery` trajectory:
  penalize entry and continuation of the burst, and retain trustworthy recovery
  and later valid rows in a separate duplicate-cleaned counterfactual arm.
- Long training may be launched with a continuation receipt and left for the
  user to report; short training should be monitored through evaluation.

## Frozen research and implementation handles

- Research unit:
  `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-22-physical-owner-duplication-causality-and-training-treatment/unit.md`
- OpenSpec change:
  `openspec/changes/add-physical-owner-duplicate-rejection-and-recovery-training/`
- Historical handoff:
  `handoff/2026-07-22-prefix-counterfactual-and-owner-duplication-penalty.md`

## Evidence census

The canonical Source train-256 rollout contains 3,290 valid prediction rows.
An annotation-anchored census found 113 repeated-owner rows in 38 trajectories;
34 trajectories later recover an uncovered annotated owner. Thirty-four
near-exact repeated rows in 12 images form the safest automatic pilot subset.
The existing 144 ambiguous geometry-derived overlap candidates are excluded
without crop review. Unmatched official predictions remain unresolved.

Initial mechanism roster:

- primary: 455649, 55232, 351528, 187464;
- no-recovery control: 505243;
- irregular or dense stress only: 57676, 424960.

## Treatment comparison

Evaluate frozen Source, Source-preservation-only, recovery-positive-only, local
complete-row duplicate rejection and recovery, duplicate-cleaned trajectory
imitation, and their matched-dose combination. The primary causal comparison
is local duplicate rejection minus recovery-positive-only. Duplicate reduction
without unique-owner growth is symptom suppression, not success.

Counterfactual cleaned prefixes must retain separate source and replay hashes,
operate by exact complete-row deletion, and never be described as naturally
sampled trajectories.

## Parallel execution status

Bounded workers own core training changes, the experiment-local StateBank
assembler, the training-free prefix probe, and independent visual owner review.
No duplication GPU job has launched yet.

The older 2,432-image breadth-panel workers were not running at this checkpoint;
their logs ended after model loading and no sampled JSON shard existed. Re-audit
before resuming that separate unit.
