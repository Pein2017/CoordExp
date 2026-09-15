---
title: Exact Greedy Prefix Uncovered-Object Row Training Screen Results
description: Bounded collection and one-event training evidence closing terminal-prefix rescue as too sparse and too late for a viable 256-image training bank.
type: investigation
role: results
authority: non_normative_research
unit_id: 2026-07-21-exact-greedy-prefix-uncovered-object-row-training-screen
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified_bounded_negative
updated: 2026-07-21
---

# Exact Greedy Prefix Uncovered-Object Row Training Screen Results

## Verdict

The terminal-prefix local treatment is formally closed as too sparse and too
late for a viable 256-image training bank. This does **not** establish that a
terminal `STOP` token is never causally relevant. It establishes only that the
screened local treatment cannot supply enough safe events, and that its one
event smoke is behaviorally unsafe. The evidence therefore points to the main
bagging gains arising from earlier branches or prefixes, where the model can
still change later rows.

## Collection receipt

The source was the geometry-sorted, description-first pure cross-entropy plus
token-type-gated Weight-Decomposed Low-Rank Adaptation checkpoint at step
4,887. The collector artifact is:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-07-21-exact-greedy-terminal-rescue-training-screen/
collector-shards-k128-v1/
```

This collection used 128 seeds per selected prefix, exceeding the unit's
planned 16-sample cap; the extra sampling strengthens the scarcity conclusion
and is not a training authorization.

- 6,436 sampled attempts produced 6,400 materialized outputs: 5,752 literal
  terminal stops and 648 complete rows.
- The complete rows contain 634 unique row texts.
- 65 attempts were accepted, but they collapse to only three unique
  image-events: image `034855` (31 attempts), image `160996` (17), and image
  `334139` (17).
- Rejections were 6,335 `no_unique_complete_target_row`, 35
  `candidate_exceeds_total_trajectory_budget`, and 1
  `greedy_terminal_replay_parity_failed`.

The independent crop-assisted review of all 583 rejected complete rows found
only 17 images represented: 134 were duplicates or plausibly already covered,
360 were same-class official entities with unusable geometry, and 89 had no
same-class ground-truth entity. No reviewed case was a confirmed safe
unlabeled positive. Relaxing review therefore cannot produce a viable
256-image training bank.

## One-event smoke: image 034855

The accepted event targets person owner `2001626` at the exact greedy prefix
immediately before the native terminal token. The training smoke moved the
target margin from `-0.25` before the update to `+1.00` after one optimizer
update. The subsequent clean greedy replay emitted the target person, so the
signal reaches model behavior rather than remaining a fixed-prefix diagnostic.

The same rollout emitted a broad person repetition burst: ten person rows
around the target region after the two vase rows. This is unsafe and
underspecified despite the target entering the rollout. The event is therefore
mechanistically informative but not a training-positive pattern.

The smoke artifacts are under:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-07-21-exact-greedy-terminal-rescue-training-screen/
smoke-image-34855-v2/
```

## Decision and continuation

Do not launch the planned 256-image terminal-rescue training run or promote
this local treatment. Preserve the terminal event as evidence that a terminal
branch can be causally actuated, while moving the next treatment search to
earlier branches and prefixes with enough downstream horizon to test unique
owner value and repetition safety.
