# Treatment Gates, Route Training, and the July 22 Completion Study

Source session: 019f4a19-d81c-75a2-84b0-2c20379e686e, principally Jul 20 to Jul 22 2026.

## Why this chapter matters

Several treatment ideas produced genuine local changes. None yet supplied a
safe, scalable greedy-enumeration method. This note prevents a future agent
from mistaking fixed-prefix learnability, selected-route gain, or a small mean
average precision change for final-set improvement.

## Current baseline choice

The practical mechanism baseline is geometry-sorted pure cross-entropy
step-4887. Random-order training changed routes but did not create useful
covered-set invariance; it collapsed to a habitual successor on the key
image-2299 panel. Historical and modern random/sorted pairs are useful
comparators, but must not be silently treated as identical checkpoints.

Gaussian coordinate smoothing plus ordered cumulative-distribution penalty
lost the modern matched comparison and worsened selected coordinate behavior
under the current sampler. Keep it as an ablation, not the default treatment.

## Coordinate-treatment gate

The coordinate screen established a narrow fact:

    fixed-prefix coordinate margins are learnable.

It did not establish:

    improving those margins improves a self-generated trajectory.

The eight-event smoke and later 256-image one-epoch screen improved many
fixed-prefix events, yet free rollout failed to improve against matched
controls and could worsen geometry, duplicates, invalidity, and mean average
precision. The lower learning rate was less disruptive but still not a
promotion. An old-and-refreshed-prefix follow-up did not isolate a convincing
prefix-only treatment effect.

Do not run more seeds, a larger learning-rate sweep, or scale this coordinate
objective unchanged. If revisited, a new cohort must use a downstream
physical-owner and whole-geometry value gate.

## Route support and positive imitation

The 12 manually refined images are development and validation cases, never
training data. Their high-value role is to distinguish:

* a physically new entity from a duplicate or unknown row;
* entity discovery from geometry quality;
* an individual sampled route that is better than greedy from a union that is
  merely complementary across many routes.

Trajectory audits found both individually better routes and cross-route
complementarity. Forced paths showed that sampled fragments can sometimes
unlock a missed owner from the identical greedy prefix. Their depth is not
universal, and one current-row success can displace a later owner.

The 256-image single-route imitation screen was a valid eight-GPU treatment
experiment. It shifted greedy output toward selected route families. Step 15
recovered 16 route-added owners but lost 15 ordinary owners, held total
training-image coverage flat, and regressed outside the admitted set. Most
gain occurred in non-direct owners on the same selected routes. This is
route-conditioned redistribution, not direct owner imitation or safe
set-expansion.

The next testable treatment family is:

    Source
      -> single-route only
      -> single-route plus Source preservation
      -> multi-route plus Source preservation.

No new launch is implied by that idea. Require both route-added gain and
ordinary-owner retention before scaling.

## The July 22 active diagnostic study

The user rejected the literal “health bar” or fixed-count conservation model.
The useful question became whether a frozen model can finish the remaining
trusted set when supplied a correct prefix, and what minimal assistance exposes
the first failure mode.

The authored research unit is:

    research/investigations/qwen3-vl-dense-enumeration/experiments/
    2026-07-22-human-refined-greedy-set-completion-conditions/unit.md

It uses the twelve refined validation images, geometry-sorted pure
cross-entropy step-4887, full-model float32, physical batch one, repetition
penalty 1.0, and a token ceiling that must never cause conclusion-bearing
truncation.

Its data surface is intentionally broad before it becomes prescriptive:

* ground-truth prefix schedules and separate same-remaining-set order controls;
* remaining-object depths N, 16, 8, 4, 2, and 1;
* native greedy suffix;
* one-time and repeated STOP suppression diagnostics;
* force one trusted remaining row then return to native greedy;
* strict and relaxed row budgets;
* raw tokens, termination reason, global physical-owner assignment, geometry,
  duplicates, invalid rows, unknown rows, and review-needed rows.

Teacher-forced scores are explanatory only. Forced rows are context, never
model-discovered objects. The study should first collect completion curves and
failure families, then run only causal replays that distinguish early STOP,
selection, owner replacement, route change, or geometry failure. It is not a
training authorization.

## Post-transcript live state

The transcript snapshot ended while the runner was being prepared, before July
22 inference began. The live follow-on now has thirteen JSON output receipts
under `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-22-human-refined-greedy-set-completion-conditions/`.
They cover all twelve designated images and 380 arms. Every arm records
`status: valid` and `token_limit_invalid_evidence: false`; no final aggregate
or report exists. Some arms have dropped or wholly dropped parsed spans, so
receipt health does not certify semantic completion, matching, or geometry.

The unit's front matter still says `evidence_status: none`, and that metadata
must not be read as proof that the post-transcript executions never happened.
The immediate task is receipt review and aggregation, not repair of the stale
`DecodeExecutionReceipt` import claim or a duplicate grid. Preserve unrelated
dirty work and do not attribute every untracked file to the July 22 study.

Formal handles:

* research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-20-own-prefix-entity-transition-and-coordinate-boundary-calibration-training-screen/
* research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-21-256-image-coordinate-boundary-training-screen/
* research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-21-individual-trajectory-versus-union-support-audit/
* research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-21-greedy-prefix-forced-owner-path-intervention/
* research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-21-best-sampled-trajectory-positive-row-imitation-screen/
* research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-22-human-refined-greedy-set-completion-conditions/
