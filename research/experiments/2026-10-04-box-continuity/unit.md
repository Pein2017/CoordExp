# B16 free box continuation and clamped-coordinate contrast

## Question, authority and advisor ruling

The user authorized an advisor round, Lead selection of the next research probe,
and implementation **and execution** by the persistent worker. This unit is the
selected bounded probe; native execution still requires the exact Lead release
after CPU qualification. Root owns the release and scientific acceptance.
Persistent worker session `01a101ac-9ec8-7862-bdb6-38b9cd154673`
(`gpt-6.1-sol`, `xhigh`) owns implementation and the released execution package.
`/root/coord_binding_reasoning` owns the scientific brief and interpretation
support. No training is authorized.

Question: does a one-bin coordinate input change produce a larger change in the
remaining freely generated box, and how much of that change survives when native
intermediate coordinates are restored? Separate a change in the conditional
probability distribution, a near-tie greedy winner change, propagation through
later generated coordinates, and a change in an identifiable physical instance.

The fresh Astra advisor recommended the smaller thirteen-request contrast below
over a broader nineteen-request sweep. It adds current-B16 free continuation and
matched clamping to the accepted [coordinate audit](../2026-10-04-coordinate-readout-audit/results.md).
That audit forced intervening coordinates and therefore did not measure these
free branches. Historical free continuations already exist and were often
numerically modest; this is not a first general test of coordinate feedback
([question owner](../../questions/capacity-and-readout.md#phaserole-continuation-distinguishes-token-stability-from-recurrence)).
The previous A0/A16 first-person GT-history intervention also failed to recover
its early bottle target before any generated wrong coordinate; bad generated
history is not assumed to be a universal root cause.

AR factorization already defines a joint distribution. This experiment tests
selected learned conditionals and greedy continuations, not whether AR can
represent joint boxes in principle. The image stays fixed: changing x1 does not
move the physical object, and a correct response need not translate every corner.
No result here establishes superiority of exact corner CE over soft targets.

## Frozen inputs and request matrix

Use only B16 from
`outputs/research/physical-fn-recovery/2026-10-03/rule-stability-iou90/native-primary-B-01/checkpoint-16`.
Saved raw/analysis records are `rank-3/version-16/greedy-13348{,-analysis}.json`,
`rank-4/version-16/greedy-7511{,-analysis}.json` and
`rank-7/version-16/greedy-351017{,-analysis}.json` under that run.
Use their original prompt, image, media/grid, tokenizer and median-policy
identities. Labels remain
`research/experiments/2026-10-02-full-label-self-rollout-fit/inputs/full-labels.json`.
Preparation binds these sources and literal selectors; historical run commands
do not supply launch authority.

All P indices and actions are zero-based. Force the saved literal history through
the intervention action, replacing only the specified coordinate. Then release
the stated suffix. Each clamped within-box condition additionally forces the two
native intermediate coordinates before release. There is no new cue or prompt
format. The maximum actions include the entire forced prefix.

| Site and condition | Forced intervention and additional clamping | Released suffix | Requests × maximum actions |
|---|---|---|---:|
|13348 P25, free |x1/action231 =543,544,545;544 is the natural control |y1, x2, y2 and row closure, nominal actions232..235 |3×236|
|13348 P25, clamped |x1/action231 =543 or545; force native y1/action232 =632 and x2/action233 =559 |y2 and closure, nominal actions234..235 |2×236|
|7511 P46, free |x1/action419 =683,684,685;684 is the natural control |y1, x2, y2 and closure, nominal actions420..423 |3×424|
|7511 P46, clamped |x1/action419 =683 or685; force native y1/action420 =577 and x2/action421 =684 |y2 and closure, nominal actions422..423 |2×424|
|351017 first bottle P1, free |x2/action16 =22,23,24;23 is the natural control |All subsequent actions17..29, covering the next native bottle and its following boundary |3×30|

Exactly thirteen requested cells, at most3390 emitted actions:1180+2120+90.
Run the three natural controls first. Within-box conditions produce **four x1
contrasts**; the bottle conditions produce **two preceding-row x2 contrasts**.
Keep these populations and their outcome counts separate.

Native boxes anchoring preparation are normal P25 `[544,632,559,682]`, zero-width
P46 `[684,577,684,597]`, first bottle P1 and next bottle P2 both `[0,0,23,86]`.
The normal worker is the already verified annotation191150, category `person`,
GT box `[546,633,556,682]`; bind that supplied mapping and exact annotation
identity, rather than introducing a new automatic owner matcher. Its GT width
is10bins, so ±1 is10% of that width. The invalid row and boundary bottles remain
unowned. The bottle's23-bin width is a generated-box scale, not a physical-object
width; the zero-width row has no valid width denominator. Report absolute-bin
displacement for all cases and label unavailable or proxy scale normalization.

For the bottle's clamped comparison, reuse only the two accepted
`B16/copy--1/351017` and `B16/copy-+1/351017` records from
`outputs/research/physical-fn-recovery/2026-10-04/coordinate-readout-audit/native-01/`.
Those records forced the remaining original history through action28 and freed
action29. They are not newly executed cells. Before consuming them, verify their
accepted artifact/checkpoint/runtime/prefix/processor-policy identities and the
fresh natural control's unchanged relevant raw/policy observations. Bind the
reuse comparison and numeric identity evidence explicitly. Unresolved identity
or numerical incompatibility HOLDs that reused contrast; it never authorizes
rerunning or silently replacing the old cells.

## Computation and observations

Reuse maintained native loading and cached singleton generation in eval/inference
mode with BF16 autocast and `MedianPolicy`; do not replace cached execution by a
full-prefix replay or alter the model's precision. Use greedy decoding, no
repetition penalty beyond1, and no decoding geometry mask. There is one B16 load
and no optimizer construction.

Capture actual raw scores before normalization and actual median-policy scores
before forcing at every relevant free or clamped coordinate, and at encountered
row-boundary/closure events needed to explain the continuation. Preserve complete
1000-coordinate score vectors with ordering, full-vocabulary normalizers,
non-coordinate winners/top scores, actual emitted tokens, pre-force winners and
forced-token likelihood/rank. Retain actual prefix identities. Correct the raw
selected-token likelihood bookkeeping for **every free action**: a raw-channel
argmax can differ from the token selected by the median policy. Never replace
the captured raw argmax with the selected token or label forced likelihood as
natural likelihood. Prefix validation applies through each forcing boundary;
the free suffix is allowed to differ from saved tokens.

Report coordinate-family probability mass separately from TV and one-dimensional
Wasserstein distance between the coordinate-conditional distributions. Preserve
top-two margins/ties, adjacent and distant competing bins, and the first free
divergence. Distribution similarity and greedy-output similarity are distinct
outcomes; no arbitrary distance threshold is an acceptance gate.

Parse the **actual** emitted rows and coordinate roles. Nominal action positions
in the matrix identify the native reference, not guaranteed slots after a free
divergence. A changed category/header, type escape, early EOS, row crossing,
incomplete span or missing homologous coordinate remains an explicit event.
Do not compare misaligned slots or expand the request to obtain a complete row.
The literal token budgets bound every branch, including unexpected structure.

Report each available complete box, coordinate/center/width/height displacement,
ordering and zero-area validity, completion, and numerical recurrence. For the
normal worker, retain overlap to the named annotation and relevant alternative
annotation overlaps; an IoU winner change is not automatically a physical-owner
switch. Invalid and bottle branches acquire no GT owner. Keep free-box outcomes
separate from observed winners at clamped positions and from the first bottle's
effect on the following row.

The within-box clamp restores **both y1 and x2**. It identifies the combined
contribution of these intervening coordinates; it cannot attribute mediation to
y1 alone. Comparisons require the shared prefix up to the intervention and the
declared restored coordinates, with later differences treated as outcomes.

## Predictions and limits

- **Small-margin greedy amplification:** close conditional distributions and an
  early winner change precede a larger free-path difference; restoring native
  intermediate coordinates reduces the downstream difference.
- **Direct conditional sensitivity:** later-coordinate distribution changes
  remain after those intermediate coordinates are restored. This does not by
  itself identify an internal instance or slot representation.
- **Stable localization/order failure:** free boxes remain nearby, or the same
  invalid preference persists. This weakens a large within-box jump explanation
  at these sites, without establishing global smoothness or correct geometry.
- **Alternative coherent instance mode:** a distant completed box can be another
  mode. Geometry alone cannot decide whether that is wrong binding, especially
  for unowned rows.

This selected panel does not estimate population prevalence, isolate numeric
precision, determine why training changed natural trajectories, establish loop
escape or physical-FN recovery, or compare training objectives. No automatic
successor follows any result branch.

## Qualification, resources and stopping

The three fresh natural controls must reproduce every saved token and incoming
median-policy argmax throughout their entire budgets, including the freely
generated portions: exactly zero mismatches. Capture unchanged pre-intervention
scores and the relevant natural-score identity comparison for prior-cell reuse.
Saved/current likelihood differences are explicit evidence, not an invented
scientific tolerance. A failed control HOLDs its dependent cells; preserve it
and continue only independent cells. Changed counterfactual winners, invalid
boxes, incomplete rows and no observed improvement are scientific outcomes,
not fidelity failures.

Wrong source/input/runtime identity, malformed selectors, unsupported policy,
nonfinite evidence, mismatched forced prefixes, ambiguous declared normal-owner
binding, execution failure or resource excess produces an explicit technical
HOLD without retry. Preserve original artifacts and all attempted-cell status.

Bounds: one serial GPU, one checkpoint load, singleton requests, at most13
requests/3390 emitted actions,424 actions per request and1744 prompt-plus-generated
tokens;15minutes producer wall,32GiB peak RSS,12GiB peak CUDA allocation and32MiB
retained diagnostic payload. Check resource limits at request boundaries and
stop starting requests after an excess. Zero training, optimizer/backward,
warmup, extra model controls, retry, dose extension, export or successor probe.

Implementation belongs in `probes/box_continuity.py` and focused tests, reusing
maintained helpers. New artifacts belong under
`outputs/research/physical-fn-recovery/2026-10-04/box-continuity/`.
CPU qualification must exercise multiple-free-action likelihood alignment,
forced/free prefix boundaries, actual-row parsing after divergence, the paired
consumer and dependency HOLD behavior with meaningful negative cases. It does
not establish native fidelity. Lead binds the exact clean execution source and
input/runtime packet before the one authorized native invocation. Preserve the
unrelated dirty `AGENTS.md` navigation change; never commit or overwrite another
owner's work merely to satisfy clean-source qualification.

Stop after this frozen matrix, its saved consumer report and Lead acceptance or
partial/HOLD ruling. Worker continuation through the already authorized package
does not authorize another research unit or a training trial.
