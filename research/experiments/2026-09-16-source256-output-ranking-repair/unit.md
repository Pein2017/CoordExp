# Source256 complete-output ranking checkpoint repair

## Frozen question and authority

From the same accepted Bnormalized64 checkpoint, does a frozen complete-output
ranking surrogate improve natural-greedy repair beyond positive CE on exactly
the same preferred outputs? User authorized one paired16-update study, including
qualification and evaluation. This is checkpoint repair, not proof of correct
local token/owner credit or prevention from Source. Predecessor studies remain
closed: [normalization](../2026-09-16-source256-completion-ce-normalization/results.md),
[bounded diagnosis](../2026-09-16-source256-completion-ce-normalization/credit-diagnosis.md),
and [original B](../2026-09-16-source256-fixed-prefix-completion/results.md).
No handoff, publication, new labels, review, pairs, refresh, weak-bank or census.

## Inputs and objective

Frozen15 strict known-owner/debt dominance pairs from bound diagnosis.json under
2026-09-16-source256-completion-ce-normalization/lead/credit-diagnosis-v1.
Preferred=A64; rejected=Bnormalized64, exact original image/prompt and observed
action tokens. All80 annotation-unmatched preferred rows remain UNKNOWN and
are intentionally consumed in full positive CE in BOTH arms for this authorized
weak-ranking pilot. Their gradients are not neutral; preferred is not verified GT.
Preserve original processed train256/dev128, bank and empty-prefix natural metric.

ell_theta is SUM action-token log probability excluding prompt/pad. Frozen
reference ell_0 is cached once from starting Bnormalized64, with no gradients.
d_i=max(observed preferred length, observed rejected length), common to both.
rank_i=softplus(-((ell_theta(y+)-ell_theta(y-))-(ell_0(y+)-ell_0(y-)))/d_i).
Lambda=1; no sweep. P=.5 mean canonical CE +.5 mean preferred CE + common
geometry term; R=P+.5 mean(rank_i) over pair presentations. CE remains
sample-equal active-token mean. Same raw-axis geometry weight.01 on canonical
and preferred coordinate positions in BOTH arms; no geometry on rejected.
Geometry validity is not certification of unknown preferred owners. Preserve
observed EOS. The capped rejected action has no fabricated EOS and is not dropped.
This is a length-normalized ranking surrogate, not exact sequence probability
optimization, QP guarantee or causal local-negative assignment.

Image548337 has177 preferred vs3084 rejected action tokens,292/293 rejected
parser drops and7/26 known-match advantage. Other14 rejected lengths total2162.
Keep full15 primary and descriptive14 sensitivity; no third arm or selected endpoint.

## Frozen execution constants and bounds

Common Bnormalized64 adapter and original paired frozen embeddings; fresh identical
AdamW per arm, seed19, LR1e-5, betas(.9,.999),eps1e-8,weight_decay0,foreachFalse,
clip1,cosine16 to0,no warmup. Same fp32/SDPA, DoRA rank16 alpha32 dropout0,
frozen base/vision/readout/embeddings as predecessor. Activation checkpointing on.
Four ranks per arm,microbatch2,global64=32 canonical+32 pair presentations/update.
Canonical uses two seed19 shuffled passes over256 images (512 presentations).
Pairs use a seed19 shuffled cyclic15-image order over512 slots:34 or35 per image.
The exact schedule is prepared once and shared. Each rank receives8 presentations
per branch. Cross-rank gradient SUM uses local denominators32 and branchweight.5;
no extra world-size divide.16 applied updates per arm, save only fixed endpoint.

Training logical forwards/calls: P1024/512, R1536/768 (rejected adds512/256).
Cache30 unique reference sequences once, micro2 on four ranks: at most16 batched
calls. Endpoint likelihood reads repeat these30 per arm (at most16 calls each).
One R one-update qualification with the same16-step scheduler, then cold reload
and four-case bs4 consumer; main resets to original anchor/fresh optimizer.
Qualification gradient/call budget R96/48, plus bounded scoring/readback checks.
Reuse unchanged qualification evidence. Wall guard4h per arm; evaluate each arm
on train256/dev128 with8 existing shards,bs4,RP1,cap3084:768 total requests,
192 generation batches,32 shards. Two independent4GPU arms concurrently if
supported; then all8 GPUs for evaluation. Root owns durable tmux/one wake monitor.
No new controller or orchestration abstraction. Preserve unrelated dirty work.

Qualification must falsify wrong sign, per-member length division, missing.5,
world-size scaling, reference gradients, prompt/pad scoring, fake capped EOS,
and rejected geometry. Check actual added rank gradient, fixed token positions,
identity, save/reload and consumer path. Real memory/call bounds are recorded;
a concrete operational conflict permits one bounded affected-stage repair,
not silent scientific or population change.

## Evaluation, acceptance and stop

Reuse hash-verified Source/A/B/normalized controls. Report fulltrain FN/G/L,
selected15,outside-reference train241,dev128,all debt categories and paired
owner identities. Track survival of ACTUAL91 starting Source-new owners,
losses and replacements separately from final G>=91. Report preferred/rejected
likelihood deltas; ranking adds positive learning, so R superiority alone cannot
isolate useful negative credit. Strongest alternative is reference imitation,
changed positive learning, or suppressing just the single long loop.

Repair gates: R train FN<P and<652; Source-old losses<55; no debt category worse
than P; dev FN<=P and<=276; outside-reference coverage>=P. Report original gain
count gate>=91 and identity turnover separately, never call swapped sets intact.
Promotion additionally requires train FN<628,old loss<=35,dev FN<=271,dev old
loss<=26,and each debt category no worse A. Reference-only or loop-only improvement
is bounded repair, not generalized credit success. One seed/historical dev limits
claims. At fixed16 endpoint publish technical acceptance separately from science,
then STOP. No extra seed,dose,coefficient,refresh,architecture or memory/Git publication.
