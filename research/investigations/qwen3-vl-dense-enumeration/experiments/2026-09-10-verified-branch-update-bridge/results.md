# Useful entrances improve slightly; conditional continuation is preserved

Scientific disposition: **positive but insufficient entrance movement, with
unchanged conditional behavior on two verified useful branches**. Technical
disposition: **lead-accepted**. The bounded bridge and its stop rule are complete;
no additional training, sampling, intervention or confirmation read is launched.

## Why this read was needed

The prior fixed-witness probe scored original complete sampled trajectories.
The row probe instead established two useful local connections from exact
Source natural histories. Those are different conditioning surfaces. This
bridge measures the actual target entrance and compares Source/round1
continuations under identical forced tokens, rather than substituting a whole
sample's unrelated first fork.

Cases remain image368/person2022537 and image7116/boat181378. They inherit the
earlier annotation-backed moderate-confidence visual admission, including heavy
occlusion and projected prior-box overlap. No case, box or description was
reselected. The models are Source step2444 and its saved first RLOO update,
with fixed base/selected embeddings, native media/prompt, FP32/SDPA, patch
linearization and unmerged DoRA.

## Actual target entrance at the fixed Source history

The decision state consists of the Source completed-row prefix plus the
unchanged object-row header/description before x1. The action-token index is97
for the person and22 for the boat. A is the previously verified useful x1;
B is the original Source next-row x1.

| Case / specific action | Source probability | Round1 probability | Full-vocabulary rank, Source→round1 | A-minus-best-other margin, Source→round1 |
|---|---:|---:|---|---|
|368: x1=455 rather than B=562 |0.00135382952 |0.00164195507 |130→128 |-3.215862→-2.927181 |
|7116: x1=291 rather than B=571 |0.00000168645328 |0.00000176945643 |690→679 |-11.088956→-11.043614 |

The original B remains the actual full-vocabulary top1 in every view, so the
A-vs-B and A-vs-best-other margins coincide here. The target ranks have no tie
ambiguity. Fixed-state log-probability changes are+0.192950 nats and+0.048045
nats; pairwise margin changes are+0.288681 and+0.045341.

These are probabilities and ranks of **one specified coordinate token at one
specified history**, not the probability that a person/boat exists, not a sum
over all same-owner boxes and not the total probability of every useful route.
Alternative coordinate entrances may also be valid; this experiment does not
enumerate or identify them. Do not extrapolate a required number of updates by
dividing the remaining gap by this single observed step.

## Exact-state visits and natural-context control

Neither exact entrance state occurs in its original same-image K4 training
bank:0/4 exact visits and0/4 exact A state/action occurrences for both cases.
This is an exact-token exposure statement, not absence of semantically nearby
supervision or prior SFT. The probability increases despite zero exact visits
also rule out treating exact state replay as a necessary condition for any
functional movement.

For image368, round1's retained natural prefix differs at just action91:
coord451 becomes452. Its eight prefix-covered GT owners and current-row header
remain unchanged. Thus the Source token prefix is not reached verbatim, but
calling the semantic state absent would overstate the evidence. Image7116's
entrance history is token-exact across models and has the same two covered
owners.

The already-budgeted native-greedy parity forwards also provide the target's
entry logits under each model's own natural context, without new forwards:

- Person round1 native-context probability0.00164588723, rank128,
  margin-2.923450, versus0.00164195507/rank128/-2.927181 at fixed Source P.
- Boat round1 native-context probability0.00000176945686, rank679 and
  margin-11.043614, agreeing with the fixed-history conclusion.

These native-context reads mix parameter and history changes where the history
differs, so they are a supplementary relevance check, not the primary controlled
parameter contrast. They show that the useful token remains low-ranked in the
actual native context too. Equal matched prefix sets are not a certificate of
identical internal state or identical behavior after every possible action.

## Four paired continuations: no observed update-induced damage

Only four new round1 continuations were generated: partial A through x1 and
complete A for each case. The Source counterparts were reused from the accepted
row probe with identical prefix/forced tokens and remaining budgets.

| Case / branch | Source conditional TP50 /60 /80 | Round1 conditional TP50 /60 /80 | FP50 / FN50, both models | F1@50, both models |
|---|---|---|---|---:|
|368 partial A |13 /13 /8 |13 /13 /8 |2 /0 |0.928571 |
|368 full A |13 /13 /8 |13 /13 /8 |2 /0 |0.928571 |
|7116 partial A |5 /5 /3 |5 /5 /3 |0 /1 |0.909091 |
|7116 full A |5 /5 /2 |5 /5 /2 |0 /1 |0.909091 |

**Every free suffix is token-exact between Source and round1.** Because the
prefix and forced tokens also match, all four complete action sequences are
identical. In each:

- A is realized in the current row, not only by a later suffix prediction;
- B is recovered freely in the suffix;
- all old Source IoU50 owners are retained;
- EOS is natural, with no strict repeats, parser drops or caps;
- IoU50/60/80 owner sets and matched geometry agree across checkpoints.

For comparison, the unforced Source and round1 outcomes are12/12/8 for368 and
4/4/1 for7116. Conditional gains are therefore still present, but the x1 or full
row was supplied externally. Partial A supplies a substantial location cue;
full forced A is never counted as autonomous discovery. The boat's full-A
IoU80 gain over native is an old owner's threshold crossing; partial A also
realizes the new target at80. The earlier geometry qualifications remain.

## What we learned, and what remains unidentified

This bridge separates two observations at the **same verified useful branch**:

1. The actual update moves the target entrance in the favorable direction,
   but does not make this specific action a greedy winner.
2. The update does not damage current-row realization or the free successor
   on these four paired conditional continuations; it leaves them exactly
   unchanged.

Thus these cases do not require a newly broken successor state to explain the
absence of natural target recovery. They also do not support a blanket claim
that useful entrance probabilities failed to move. The remaining concrete gap
is natural selection of useful actions, with the source of insufficient
preference still unresolved.

We have not identified whether state distribution, credit allocation,
normalization, functional update dose or parameter coupling primarily limits
that preference. Neither exact-state absence nor a low single-token rank is
an exclusive causal diagnosis. No claim about all owners, all coordinate
realizations, a missing global ledger, or unseen-image transfer follows.

A future training or calibration proposal would need an explicit changed
objective and preservation scope; these findings do not authorize it. The
current question is answered at the frozen two-case boundary.

## Execution, evidence and reproduction

One bounded GPU0 invocation: two model loads, eight score forwards, four new
round1 continuations,120 new tokens,128 total model forwards and12 image
forwards;27.782761 seconds including loading. Allocated model time is0.007717
GPU-hours, not the total CPU preparation/coding/review wall time. Peak CUDA
allocation9.39GB and peak RSS12.01GB; about6.18MB artifacts at execution end.
The900-second and12336-new-token limits were not approached. GPU0 returned to
its3MiB idle baseline and no owned model process remains.

Four own-model native-greedy references passed full-token winner parity. All
eight primary/native-context full-vocabulary arrays are stored as FP32 NPY;
the reported log-normalizer/probability arithmetic uses float64. Root freshly
passed five focused tests, replayed the raw consumer/reduction, and independently
recomputed every rank/margin/log-probability using PyTorch float64 rather than
the producer's NumPy arithmetic. Root also independently checked all four
complete action sequences against their Source counterparts. No extra model
calls were required for acceptance.

The producer's launch code hash and the final code hash are separately saved:
the only post-run addition was a CPU-only verification CLI, not a change to
model outputs or the scientific reduction.

- [Primary reduction](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-verified-branch-update-bridge/execution/reduction.json).
- [Raw round1 trajectories](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-verified-branch-update-bridge/execution/rows.jsonl).
- [Native consumer](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-verified-branch-update-bridge/execution/consumer.json).
- [Exact exposure and reachability](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-verified-branch-update-bridge/cpu-state-exposure.json).
- [Terminal resource receipt](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-verified-branch-update-bridge/execution/terminal.json).
- [Worker verification](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-verified-branch-update-bridge/worker-verification.json).
- [Frozen protocol](unit.md).

CPU-only reproduction from the research-probes worktree:

```bash
python -m pytest -q probes/dora_owner_learning/tests/test_branch_bridge.py
python -m probes.dora_owner_learning.branch_bridge verify
```
