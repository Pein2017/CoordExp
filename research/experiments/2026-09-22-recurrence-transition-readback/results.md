# Natural recurrence transitions: accepted bounded readback

Date: 2026-09-22. **Lead-accepted; CPU-only package closed.** Three native
subagents supplied extraction, visual evidence and a mathematical check. This
is new analysis of preserved natural outputs, not new model generation or a
population mechanism claim. Model/GPU calls and GPU time: zero.

[Acceptance receipt](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-transition-readback/lead-acceptance.json)
binds data, source captures and independent checks. [Unit](unit.md) owns scope.
Candidate detail: [dynamics](supporting/dynamics-candidate.md),
[visual review](supporting/visual-candidate.md),
[mathematical constraints](supporting/model-constraints.md).

## Population and outcomes

Four images were selected before step-level analysis: the two exact-repeat
images in the prospective 128-image untied-original cohort, and the longest
eligible capped and uncapped runs in the mature 145-image cohort. All bind the
same mature untied+axis001 step-2444 adapter, effective input/output rows,
independent deltas, FP32/SDPA and original policy. Selection is retrospective,
phenotype-enriched and includes an exposed anchor; it estimates no prevalence.

The four images contain 16 maximal exact complete-row runs and 1,688 coordinate
role points. These are repeated measurements, not independent replicates.
Exactness includes the entire serialized row. Longest runs, zero-based:

| Image | Exact run | End | Physical evidence |
|---|---|---|---|
| train:269858 | rows16–19, 4 | x1 350→348 at row20; row21 returns | Extent change in same crowded patch; unique owner HOLD. Later clear bench/person localizations. |
| train:196924 | rows19–20, 2 | EOS | Reflective tableware; precise owner/category HOLD. No later recovery. |
| train:351017 | rows58–341, 284 | cap with incomplete tail | Every repeated box is invalid [0,0,0,0]; no physical owner. |
| val:7511 | rows27–88, 62 | x2 38→999 at row89 | Repeated person box is water-only. First exit spans background/multiple people; clear kite appears at row93. |

The two longest cases are not established repetitions of an actual individual
object. Formal geometry validity does not ensure physical validity. Numerical
exit, EOS and physical progress remain separate; no pooled recovery rate or
first-ever-owner count is estimated.

## Opposing coordinate dynamics beneath identical rows

For train:269858's four identical rows, x1's repeat-versus-best-other margin is
0.067574, 0.013580, 0.013847, 0.010876, then -0.028833 at its first changed x1.
Meanwhile x2's fixed-rival margin rises from 0.018772 to 0.278236. Roles do not
undergo uniform loss of selection advantage. The x1 runner-up changes, so its
top-two envelope cannot be treated as a fixed-pair curve.

For val:7511, exact-row log probability rises overall from -12.0298 at row27 to
-11.4529 at row88, yet the next row changes. x1 also strengthens, from 0.968594
to 1.461838. These observations challenge an assumed global confidence-depletion
coordinate; they do not identify the causal source.

## Counterexample at the coordinate that actually exits

In val:7511 rows77–88, x2 has the same winner (bin38), runner-up (bin999),
and literal within-row prefix. At row89 x2 is the first differing token and
these two tokens exchange winner/runner-up. Their signed margin is recoverable:

| Row | x2 margin: bin38 minus bin999 |
|---|---:|
| 82 | +0.077303 |
| 83 | +0.162321 |
| 84 | +0.044655 |
| 85 | +0.123299 |
| 86 | +0.157436 |
| 87 | +0.195662 |
| 88 | +0.206703 |
| 89 | -0.006075 |

The first four points have slopes +,-,+; minimum absolute step is 0.078644.
The last four pre-exit steps rise before the sign crossing. The lead rebuilt
these values from raw token positions and top-two logits independently of the
producer's margin helper.

**Conditional mathematical conclusion:** the stationary direct mapping
`d[n] = d_inf + A*lambda**n + B*rho**n`, with positive rates below one,
permits at most one first-difference sign reversal. It cannot reproduce these
four fixed-pair values exactly. Any such curve must incur pointwise absolute
error at least 0.039322 on one of these points: smaller errors preserve all
three alternating slopes. This is an algebraic approximation bound on saved
values, not an experimentally measured numerical replay-error interval.

Long fixed-pair x1 windows corroborate the narrower descriptive pattern. At
deadbands 1e-4 / 1e-3 / 1e-2, val:7511 has 26 / 26 / 12 slope turns; the capped
invalid case has 150 / 144 / 95. Thresholds are sensitivity summaries, not
confidence intervals. The exit-relevant x2 witness is the more direct result
and does not depend on an invalid-box episode.

This rejects the literal stationary two-decay mapping to that observed margin.
It does not reject two hidden variables with nonlinear readout, generic adaptive
history mechanisms, or approximate macro-models with declared error tolerance.
Growing position/context remain alternatives. The adviser gives a positional
attention countermodel without adaptive exclusion that also produces exact
repetition then numerical exit; it demonstrates non-identification, not its
operation in this checkpoint.

## Interpretation and next decision

A useful description must retain which token role is near losing, against
which competitor, and how that fixed competition changes. Greedy repeats a
fixed row while all sequential token decisions keep winning; individual roles
can strengthen and weaken simultaneously. After a role changes, later tokens
have a different prefix. This explains compatibility of unequal role dynamics
with unchanged rows, not the learning origin or responsible circuit.

A next causal proposal should distinguish positional competition from
content-dependent history updating on an explicitly chosen phenotype. Do not
reuse water/invalid-box runs as owner-exclusion evidence, fit two time constants
to the four-point anchor, or score first numerical change as set completion.
No controlled response experiment or parameter scan follows automatically.

## Acceptance and reproduction

Both selfchecks passed: signed margins/below-top-two bounds, corrupt alignment,
batch identity and norm1000-versus-999 geometry. Lead CPU replay left selection,
episodes and TSV byte-identical; all 12 visual figures reproduced identically.
Rendered boxes match token-parsed rows. The lead inspected full/local visual
context, source bindings, shared effective model identities and the fixed-pair
witness. Producers and used parser sources were captured outside outputs.

```bash
python -B -m probes.training_set_completion.recurrence_transition_readback --selfcheck
python -B -m probes.training_set_completion.recurrence_transition_readback
python -B -m probes.training_set_completion.recurrence_transition_visuals --self-check
```

The visual candidate gives its full rendering command. Machine-readable checks
are bound from the acceptance receipt. All children and CPU jobs are terminal.
CPU wall time was not comprehensively instrumented. No forecast was fitted or
prospectively validated. No training, new generation, label edits or publication.
