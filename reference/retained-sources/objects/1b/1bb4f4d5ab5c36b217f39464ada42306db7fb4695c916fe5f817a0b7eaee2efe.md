# Numerical recurrence feedback — candidate, awaiting root acceptance

The fixed panel supports a small, role-selective recent-coordinate response, but not a recurrence-specific maintenance mechanism. Same-role crossed margins usually shift toward the substituted value; comparable or larger effects also occur at healthy boundaries. Free original-greedy continuation rarely copies the replacement and commonly returns to the native pattern or another repeating pattern. Numerical escape is not physical recovery.

## Frozen population and executed contrasts

Seven images, two mature training packages. Reused16 compatible focus outputs and generated12 new focus outputs (three new images × two packages × two readouts). Eleven original trajectories qualify: six untied+axis literal runs, four tied literal runs, one tied near run. All selected source rows happen to be geometrically valid; invalid native rows were not filtered by the rule. Tied7511 and both bird309264 trajectories have no qualifying episode; retain these negatives. Fourteen nonrecurrent healthy boundaries are comparison proxies, not randomized or exactly matched controls.

The completed first row of the earliest triple is the source; fixed target rows are source+1,+2,+4. All four coordinate roles and all276 frozen feasible replacements are retained. Twenty-four opposite-direction candidates are impossible at coordinate endpoints; no alternate search fills them. Validity is preserved in every admitted replacement, while79 replacements change the recorded x1/y1 ordering stratum. These mismatches are disclosed, not posthoc exclusions.

## Immediate crossed margins

K>0 means substituting b for a shifts relative a-versus-b support toward b at the SAME fixed-suffix target position. Entries below are medians of per-episode replacement medians, not pooled independent-token estimates.

| Package / boundary | Episodes | Same-role K | Cross-role K | Same-role equal-norm shadow K |
|---|---:|---:|---:|---:|
| untied / failure | 6 | 0.03959 | 0.00011 | 0.04008 |
| untied / healthy | 7 | 0.04950 | 0.00207 | 0.04926 |
| tied / failure | 5 | 0.02556 | 0.00050 | 0.02517 |
| tied / healthy | 7 | 0.00538 | 0.00251 | 0.00529 |

All11 failure episodes have positive immediate same-role median K. Effects generally diminish across fixed delayed rows, but are not universally monotonic; healthy effects and realization variation prevent a selective-failure interpretation. The equal-norm shadow scarcely changes these aggregate crossed margins. This does not mean normalization cannot change absolute winners or long trajectories; it is a conditional difference-of-margins observation. Every source/candidate rank, full-vocabulary log probability, embedding/bin distance, all1000 logit deltas, head inputs, EOS probabilities and full fixed-suffix likelihood remains reconstructable.

## Full free consequences

| Package / boundary | Replacements | First same-role=a / b | Any native / substituted exact pattern | Other near-repeat | EOS |
|---|---:|---:|---:|---:|---:|
| untied / failure | 66 | 59 / 1 | 59 / 2 | 61 | 17 |
| untied / healthy | 78 | 41 / 1 | 20 / 2 | 60 | 28 |
| tied / failure | 55 | 43 / 2 | 38 / 1 | 48 | 13 |
| tied / healthy | 77 | 25 / 3 | 10 / 0 | 45 | 27 |

Return categories overlap. Replacements are correlated interventions on25 selected boundaries, not independent population trials. Supplied coordinates receive no autonomous discovery credit. Native baselines reuse the same saved greedy suffix with the32-row/512-token/EOS truncation; one actual native/no-op release reproduces288 saved tokens exactly. All276 substituted releases end at32 complete rows or natural EOS, with no512-token truncations. Per-trajectory invalidity, malformed fragments, longest exact/near run, fixed-suffix score changes and return positions are in `reduction.json`.

## Technical gates, failures and optional accounting

All source-slot trace comparisons meet the frozen0.0002 bound; all paired native/no-op logits and choices are exact. Full heterogeneous companions and positions are preserved. Full cache is rebuilt for each changed prefix, with no stale KV, parameter updates, repetition penalty or hidden output mask. Later target fields retain their native earlier fields. Native self-comparisons qualify execution, not causal explanation.

Four initial new-image loads failed before any forward because public pixel boxes were mistakenly treated as coordinate bins; preserved failed-input-v1. The single input correction uses the existing canonical xy-sorted coordinate JSONL resolving the identical image bytes. A missing new-source lookup caused one zero-forward capture failure, corrected once. Both failures remain in cost and provenance. The earlier draft singleton replay was corrected before model use.

Optional CPU accounting over prior442 saved states exactly reconstructs (alpha−1)z=(alpha−1)(mu·h)+(alpha−1)(z−mu·h), with mu the mean effective coordinate row. Among coordinate flips, the common term alone reverses the selected pair in24/48 tied and37/55 untied cases; centered term alone in14/48 and30/55. These overlapping arithmetic counts depend on this centering convention and do not identify native modules or recurrence origin. No extra model operator or rollout was launched.

## Limits and next decision

The strongest surviving account is ordinary recent-coordinate continuity combined with persistent image/context preference, not a failure-specific copying circuit. Same-role sensitivity can contribute without making the substituted value the greedy winner. Spatial/order displacement, low-probability synthetic histories and direct rereading of the changed row remain alternatives. A separately released relay contrast would need to distinguish persistent access to the original modified row from propagation through newly emitted coordinates; this package does not launch it. No general physical FN, precision, training-origin or deployment claim follows.

## Execution scope and scheduling

One bs3 native/no-op release qualification matches288 saved original tokens. Other native baselines are reused saved outputs, not independently regenerated per boundary. Earlier companions may already have EOS and padded suffixes; right-padding warnings are preserved. Root independently localized warnings to companions in116/276 releases; every target is unpadded. All free-release claims and metrics apply ONLY to the target. Companion continuations are uninterpretable and retained solely as raw execution evidence. Pair scoring uses the identical frozen companion and padding policy; exact native-slot and no-op checks establish the measured surfaces, not every possible release trajectory. Four GPUs, not all eight, ran the paired scaleout. No operational resource conflict was documented; this scheduling underutilization is reported rather than retroactively justified or rerun.

## Reproduction and closure

CPU reproduction: `PYTHONPATH=. python probes/training_set_completion/numerical_feedback/reduce.py --output /tmp/numerical-feedback-recheck.json`. [Result](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-numerical-recurrence-feedback/result.json), [verification](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-numerical-recurrence-feedback/verification.json), [artifact map](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-numerical-recurrence-feedback/ARTIFACTS.md).

Charged 79606 model forwards, 1142 vision passes, 13515.049 allocated GPU-seconds including failures; 5086213022 actual tensor-file bytes. No active owned producers. Candidate only; root owns acceptance and successors.
