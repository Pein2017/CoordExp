# Positive32 closeout: local learning, absent negative signal, preservation damage

## Status and current stop

Technical: both independent eight-rank32-update runs, exact-score cold reloads,
and the amended shared-checkpoint384+6 endpoint are lead-accepted. The user's
discussion pause was honored and then explicitly lifted for the overnight goal.
Scientific: the intended negative-gradient contrast was not realized, and the
shared learned checkpoint fails joint owner/burden quality. This phase is now
closed with no checkpoint promotion; next probes have separate units.

The second network interruption removed native agent handles but did not stop
the original full-B producer. A fresh Sol-high owner joined that same process
and ran its previously authorized cold reload, without retraining.

## Actual natural endpoint and concentration

A was actually evaluated once; B was not independently executed or relabeled,
under the explicit identical-payload execution amendment. The actual Stable50
baseline is retained in the frozen packet, not Source or the older dedup32 arm.

|natural population|Stable50 TP50 -> A|F1 change|owner gains/losses|strict repeats|
|---|---:|---:|---:|---:|
|union384|1899->1874|+0.00609|78/103|582->432|
|train256|1285->1281|+0.04931|61/65|495->109|
|dev128|614->593|-0.08087|17/38|87->323|

Union raw starts4213->3461, parsed3421->3277, dropped792->184. Actual dropped
geometry-invalid rows788->159; other malformed/incomplete4->25. A score field
of zero invalid *parsed* predictions must not hide these parser drops.
Old cap images351017/417044/477415/502725 no longer hit3084tokens;39654 newly
caps. This is not elimination of all old loops:351017 still has52 strict
repeats (previous136). Cases417044/477415 gain10/14 annotated owners with no
old IoU50 loss, while502725 gains1/loses2.

The new39654 cap accounts for82.2% of net added dev repeats and98.6% of net
added dev drops, but only10/38 dev owner losses. Excluding it posthoc still
leaves dev127 F1 down0.02324 and net11 owner losses. Excluding the four old
train caps leaves train252 F1 down0.02570 and net38 owner losses. These are
diagnostic exclusions, **not** substitute acceptance populations.

Most decision-bearing for the next route: the same56 KL-protected reference
images lose33/gain6 IoU50 owners (416->389TP; F1.66295->.61551). Therefore
insufficient reference-image coverage alone cannot explain preservation failure.
Root replayed the full concentration reducer and its parser/owner identities.

The six conditional reads select exact trained c under all3old h. Under h+c,
the first successor is exact w for417044/477415;351017 emits a nearby wineglass
box rather than exact w. Source-image inspection supports the same held glass,
not arbitrary full-suffix cleanliness. The351017 h-only/h+c continuations still
contain56 strict repeats each despite ending at EOS. Local ability is real;
complete and reliably preserved detection is not established.

Root also inspected the actual39654 source image: the new tiny banana box at
the upper-left lies on dark round fruit rather than visible bananas, then
repeats/drifts. This is not an assertion that every unmatched prediction is a
hallucination or that all original loop seeds lack physical support.

## Executed contrast

A used three visually admitted complete positive rows plus compatible
conditional/normal KL protection. B used the same objective and fresh sampled
next-action strict-repeat credit, under the unchanged unit contract. Each arm
started from unchanged Stable50 and fresh AdamW;32 updates, no dose extension.

| Evidence | A | B |
|---|---:|---:|
| Updates |32|32|
| Negative sampled actions |0|768|
| Strict-repeat D=1 samples |n/a|0|
| Model forwards, all ranks |3,510|53,162|
| Image forwards, all ranks |3,510|5,046|
| Maximum rank elapsed, seconds |866.496|1,518.569|
| Summed terminal lifecycle rank-GPU hours |1.92343|3.37123|

These costs exclude separate smokes, failed preparation attempt, and cold loads.
Both cold loads reproduced all three final positive scores exactly. Root freshly
replayed both sealed receipt consumers, source/init identities and all32 update
records. A/B adapters are identical after **every** update and at export:
`8c7d9e841be36fcdadb546d3024a32938124da2b1e12c3dd3ea80769a194dc68`.

## What positive learning established

At the final teacher-forced fixed h/c route, every target token of all three
complete rows is argmax. Minimum target-minus-best-alternative margins are
0.59035(table),0.58709(donut),0.65622(chair). This is more than a first-token
improvement, but **not** a fresh native decode or original-prompt owner-recovery
result by itself. The endpoint above supplies the distinct natural and
conditional evidence and reveals the preservation cost.

## Why the negative arm was inactive

All32 B steps had denominator24 and D=1 count0. The768 retained/replayed
outcomes were751 valid nonduplicates,10 geometry-invalid,6 malformed and1 EOS.
Each case contributed256 samples; all had zero event loss. Therefore the
extra term supplied no gradient and B reproduced A exactly. This cannot decide
whether sufficiently supplied repeat-event learning would help or hurt.

A CPU-only check of the existing Stable50 h-only records rules out the simplest
alternative that these prefixes are merely too early in the loop:

| Fixed h | First free greedy row, native pixel bbox | Earlier h row index | IoU |
|---|---|---:|---:|
|351017-c01|bottle `[0,0,25,72]`|1|1.0|
|417044-c01|donut `[0,221,70,236]`|7|1.0|
|477415-c02|chair `[0,651,165,831]`|5|1.0|

Root verified source hashes, literal h IDs, first-free placement, raw-row
identity and native-pixel IoU. At the initial Stable50 checkpoint, all three
greedy continuations immediately repeat, yet the same initial B step's24
raw-softmax samples produce no strict repeat. The complete768 sample census
spans changing parameters; do not pretend all draws share one stationary
distribution or infer a single repeat-probability confidence interval.

Supported inference: the authorized raw-softmax event acquisition did not
exercise the deployment greedy failure. Greedy argmax selection and sampled
complete-row event mass are different surfaces. The data do not identify the
underlying attention/KV mechanism or prove that all stochastic decodes avoid
repetition. Distinguishing temperature/coordinate spread is still a hypothesis,
not a completed explanation.

## Next decision and separate bounded probes

The shared-checkpoint read is complete. Root now separates two questions:
[checkpoint versus first-row history](../2026-09-11-checkpoint-history-cross/unit.md)
tests conditional loop seeding with a fixed16-cell cross;
[greedy preservation microscope](../2026-09-11-greedy-preservation-microscope/unit.md)
localizes reference KL versus actual greedy-margin crossings on the56 protected
images. These are new bounded inference/scoring units, not another training
arm or unrecorded extension of this contrast.

If natural loops remain, the next bounded negative-learning design should first
demonstrate nonzero strict-repeat acquisition at deployment-relevant conditions.
Greedy-mined current-policy hard negatives are a candidate, but that is a
different surrogate from an unbiased raw-softmax event-risk gradient. Keep
IoU>0.95; multiplying the current zero term by a larger coefficient cannot help.
No extra training, temperature sweep, matcher, prefix refresh or KV intervention
is authorized by this checkpoint. This acquisition/estimand judgment belongs
to the root research lead, not to an alleged Sol/Luna implementation failure.

## Evidence and delegation

Raw root: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-11-positive-branch-vs-repeat-event`.
`full-A-admission.json` and `full-B-admission.json` bind exact training/cold
receipts, acceptance and signal-location evidence; `endpoint-admission.json`
binds the fresh384+6 read and exact concentration replay. `unit.md` owns the frozen
question and pending endpoint scope. See [delegation evidence](delegation-evidence.md)
for the role-specific Sol/Luna findings; they are not a controlled effort study.
