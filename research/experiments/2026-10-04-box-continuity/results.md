# Box continuity: local winner changes without a selected within-box cascade

Lead-accepted and closed. Four within-box x1 contrasts produce at most2bins of
change in a freely generated corner. Two preceding-bottle x2 contrasts move the
next bottle's right edge by6bins while changing its coordinate-conditional
distribution much less. The evidence supports conditional score competition and
locally fragile greedy winners; it does not establish a general box-fragmentation
mechanism or a training-objective remedy.

## Execution and acceptance

The [frozen unit](unit.md) records the advisor ruling, selected13-request matrix,
authority, identities and stop boundary. The original persistent worker
`01a101ac-9ec8-7862-bdb6-38b9cd154673` implemented and ran this package on the
current integrated codebase, as subsequently required by the user. Execution
source was the clean commit `106aaed8558265a9554c057620d2444e93b37da4`, including
the completed current-stack migration at
`2e444da87783c7ce3bf81f405ce275ca3adc3104`; no older checkout or compatibility
fallback was substituted.

One native invocation completed13requests/3390actions:3319forced and71free,
including13prefills and3377cached prediction steps. There was one B16 load and
zero optimizer, backward, training, replay, warmup or export. The three natural
controls reproduced all690saved token and incoming median-policy decisions with
zero mismatches. Raw selected-token log probabilities were identical; the maximum
median log-probability difference was1.036e-6, retained descriptively. Pre-edit
prefixes and observed raw/median scores were exact in all six free contrasts.
Fresh natural bottle observations also matched the accepted previous audit
exactly, admitting its two retained clamped cells without another model request.

The actual producer exit was0. Producer wall time was356.82s, owner wall358.42s;
peak RSS12.86GiB, CUDA allocation4.96GiB and retained native payload7.92MiB.
No resource limit was exceeded. Producer and owner-wrapper PIDs settled, and the
execution source remained unchanged and clean through acceptance.

CPU qualification covered15distinct test nodes across the recorded initial and
affected reruns, including meaningful failures and corrections for multiple-free
likelihood alignment, actual-row parsing and dependency HOLDs. The final
current-source entry check passed, and final preparation confirmed the affected
source paths unchanged. Lead reused those checks and directly verified every
saved native artifact identity, forced prefix, free selected token, reported box,
first divergence, natural control, reused-cell identity, denominator and exit
receipt. This is technical and bounded scientific acceptance, not user scientific
acceptance or a promoted checkpoint.

Artifact root: `outputs/research/physical-fn-recovery/2026-10-04/box-continuity/`.
Primary records are `prepared-04/input-packet.json`, `native-release-01.json`,
`native-01/readback.json`, all13condition JSONs, `native-01/terminal.json`,
`native-owner-01-terminal.json`, `native-summary-01.json`,
`native-settlement-01.json` and `lead-consumer-check-01.json`. Readback SHA256:
`24b2e54bd9762262fb02ebf1ca8a2ab6f5f53cc91010c9b8560984e2e8ed8a19`.
Original CPU and source-integration receipts remain under `cpu-qualification-01/`,
`integration-01/` and `prepared-04/`; historical failed attempts are preserved.

## Four within-box contrasts

All row/action indices are zero-based. The natural normal P25 box is
`[544,632,559,682]`; the natural zero-width P46 box is `[684,577,684,597]`.
Only x1 is initially edited. The paired clamp restores both native y1 and x2;
those forced corners are interventions, not model corrections.

| Site and x1 edit | Freely continued box | Box with native y1+x2 restored | First free divergence |
|---|---|---|---|
|Normal,−1|`[543,632,559,682]`|same|none|
|Normal,+1|`[545,632,558,682]`|`[545,632,559,682]`|action233: x2,559→558|
|Zero-width,−1|`[683,577,684,599]`|same|action422: y2,597→599|
|Zero-width,+1|`[685,577,684,599]`|same|action422: y2,597→599|

Every target row completes. The normal free branches move the center by
`[-0.5,0]`/`[0,0]` and change width by+1/−2bins; height stays50. The named
annotation191150 has GT width10bins, so the one-bin corner displacement is10%
of that width and the+1branch's width reduction is20%. These changes are not
negligible merely because their absolute values are small. Named-owner IoU is
0.6125/0.75385 versus baseline0.65333. Maximum alternative-annotation overlap
remains below0.0021; no instance switch is demonstrated.

The zero-width free branches retain x2=684 while y2 becomes599. Width becomes1
or−1 and height22: the−1branch is geometrically valid, while the+1branch is
reversed. The valid branch owes its width to the forced x1 intervention; it is
not learned repair or physical-owner recovery. Both rows remain unowned and
have no valid original-width denominator.

At normal+1 x2, median coordinate-conditional TV is0.03560 and Wasserstein
distance0.16950bin. The natural top-two margin is0.05883 and the intervened
margin0.06422; the neighboring winner changes559→558. Restoring x2=559 changes
the subsequent y2 distribution relative to the free branch (TV0.05853,
Wasserstein0.28091bin), but both still select682. Feedback can change a
distribution without changing its winner.

At zero-width y2, the two edits give TV0.03899/0.04303. The new winner599 leads
597 by only0.02450/0.02305. Free and clamped y2 score vectors are exactly equal:
y1 and x2 already stayed native in the free branches, so clamping them changes
no intervening emitted token. The y2 change therefore does not require an
earlier greedy coordinate error.

## Two preceding-bottle contrasts

The first and next natural bottles are both `[0,0,23,86]`. Editing the first
bottle's x2 to22or24 leaves its freely generated y2=86. In both branches the
next freely generated complete bottle becomes `[0,0,29,86]`: x2 increases6bins,
center x increases3bins, and height remains86. This is26.1% of the native
generated23-bin width, which is a numerical proxy rather than physical-object
scale. No bottle is assigned an annotation owner.

The first free divergence is action26, the next bottle's x2. All prior free
tokens remain native. At this position median coordinate-conditional TV is
0.03433/0.02226 and Wasserstein distance0.56776/0.21478bin. The natural winner23
has top-two margin0.04538; winner29's margins are0.03338/0.05955 after the edits.
Small probability redistribution can therefore produce a six-bin argmax jump.
Both edit directions select29 rather than copying22or24.

The accepted historical clamped cells have exactly the same action26 scores.
The new evidence is that this conditional winner is realized as a freely
generated complete row. Subsequent y2 stays86, although restoring the next x2
to23 changes its distribution. Action29 remains object-start in both free and
historical clamped branches. Its coordinate-family winner is not the emitted
token: actual full-vocabulary winners and tokens establish that boundary result.

The strict-IoU>.9 duplicate event decreases1→0 because widths differ: the first
and next boxes have IoU22/29=0.7586 or24/29=0.8276. The output remains in the
same unowned boundary region. This change is numerical de-duplication, not
established physical de-duplication, a new owner, or escape from a later loop.

## Interpretation and stop

The first changed free winner in every affected branch occurs before any
intervening free token has departed from the native sequence. These first
changes are not mediated by earlier changed greedy token IDs. The original edit
can still propagate through hidden states and the KV cache; this experiment
does not localize an internal circuit or rule out hidden-state propagation.
Later emitted-coordinate changes can affect later scores, as the normal+1 y2
contrast demonstrates, without causing another output change.

For these sites, the evidence weakens a large within-box greedy-token cascade
explanation and supports nearby or competing coordinate modes with fragile
winners. It neither establishes global smoothness nor rules out cascades in
other histories. The image remains fixed, so a correct continuation need not
translate all corners together. AR chain factorization can represent a joint
distribution; the observed issue concerns the learned conditional distributions
and the choices made by greedy decoding.

Coordinate-family mass is approximately1 at the relevant coordinate decisions.
Retained reconstructions can exceed1 by about1.7e-6 because the recorded full
normalizer is rounded; no clipping or new gate was introduced. Distribution
distances condition on the coordinate family and are separate from this mass.
Header positions and the final censored opener are not treated as homologous
coordinate slots. All branches stop at their literal budgets; later EOS,
recurrence and full-scene coverage are unmeasured.

No comparison of soft targets with exact corner CE was performed. Similar
embedding rows, sensitivity to a token edit, and greedy discontinuity do not
alone determine which objective learns better localization or owner coverage.
This one-checkpoint selected panel establishes neither prevalence nor a unique
training cause. The unit is closed after the frozen matrix and Lead acceptance;
no training, precision intervention or successor probe is scheduled.
