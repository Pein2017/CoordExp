# History location and interacting cached support at one numerical exit

2026-09-22. **All three stages lead-accepted; this investigation converged and
closed.** The [location protocol](unit.md),
[cache partition](cache-partition.md) and [native bridge](native-row-mass.md)
own the successive prospective contrasts. The user explicitly requested
autonomous continuation until convergence, without another approval pause.

All stages use mature untied+axis step2444 and one exposed val:7511 trajectory.
The water-person template and first long multi-owner rectangle are both
user-adjudicated bad predictions. This investigation concerns a numerical
decision; it does not establish physical-owner recovery or a population effect.

## A: identical row counts, different locations

At native row88 x2, change one historical x2 from38 to640. Every edited arm has
the same complete-row and generated-token multiset. Current S, sequence length,
image, batch companions, MRoPE positions and actual mask remain fixed.
Let d=z38-z999; the vocabulary winner is measured separately.

| Source of the edit | Complete intervening rows | d | Effect versus native | Global winner |
|---|---:|---:|---:|---:|
| Native, no edit | n/a | +0.206699 | 0 | 38 |
| Row87 | 0 | -2.623060 | -2.829760 | 999 |
| Row86 | 1 | -0.540208 | -0.746907 | 999 |
| Row72 | 15 | -0.030537 | -0.237236 | 999 |
| Row27 | 60 | +0.032948 | -0.173752 | 38 |

All between-location differences from row87 exceed the0.001 numerical guard.
A position-insensitive count-only account does not describe this contrast.
Influence decreases over these four selected locations, while an effect remains
at the oldest one. This is not a fitted decay law: relative position, preceding
context and propagation through intervening tokens change together; row27 also
sits at the plateau entrance. An unchanged winner would not imply no effect.

The frozen material-effect rule selects the nearest older location, row86,
for the causal follow-up: its2.082852 difference exceeds the0.282976 threshold.
It was not selected as the most extreme old location.

## B: source token and subsequent history do not add independently

At each selected location, prefill actual native and edited histories. Partition
their cached K/V at all28 layers into the changed x2 token D and later historical
positions H ending before current S. Keep earlier positions and companions
native, and freshly recompute all six S tokens under each mixed cache.
H contains2 tokens for row87 and11 for row86; these are route partitions, not a
comparison normalized by the number of cached tokens.

| Cached history used | d, edit at row87 | d, edit at row86 |
|---|---:|---:|
| All native | +0.206694 | +0.206694 |
| Edited source token only, D | -1.931947 | -0.651796 |
| Edited downstream history only, H | +0.869978 | +0.729065 |
| Both, full edited history | -2.623043 | -0.540218 |

The near-minus-older contrast is-2.082825 in the full edited condition,
-1.280150 in D-only, and+0.140913 in H-only. The interaction difference is
-0.943587. Neither single route preserves the full contrast under the frozen
0.208282 tolerance; H-only's location contrast is negligible under that same
criterion. This does not mean H has no effect in either location: both of its
individual effects exceed+0.52.

An algebraic consequence of the saved near-location factorial is especially
useful. Adding edited H when D is native changes d by+0.663284, toward38.
Adding the same edited H when D is edited changes d by-0.691096, toward999.
Its effect reverses sign across the other factor. Thus the two routes cannot be
represented as fixed additive contributions to this readout. This is a retained-
state causal interaction, not localization of a particular attention head or
proof that attention alone, rather than the subsequent nonlinear computation,
produces it. The D route includes processing within freshly recomputed current S.

The decomposition is exact algebra, not a fraction of a natural mechanism
explained. Surgical cache hybrids establish what these retained states can do
in the declared context; they do not establish exclusive natural mediation.

## C: the native exit depends strongly on which keys the added row supplies

Return to the actual native row89 x2 transition, without any640 edit. The added
row88 is an exact token repeat of row87. Keep the current row at its native late
positions and freshly recompute all of S under each intervention. M masks the
added row; B copies the previous row's exact post-RoPE K and V into its cache
slots. K and V copy only the named component. The source block is unchanged.

| Added historical row | d | Global winner | Actual winner gap |
|---|---:|---:|---:|
| Native K and V, N | -0.006084 | 999 | 0.006084 |
| Masked, M | +1.133520 | 38 | 0.344784 |
| Previous-row K and V, B | +0.751610 | 38 | 0.330095 |
| Previous-row K, native V | +0.748022 | 38 | 0.330410 |
| Native K, previous-row V | -0.003321 | 999 | 0.003321 |

Native N reproduces the accepted L_L vector; masked M reproduces E_L at the same
current positions. Exact duplicated previous-row K/V does **not** reproduce the
native999 winner. It changes d by-0.381910 relative to masking the added row,
so it has an effect, but is insufficient to produce this exit. The natural
added row changes d by-1.139605 relative to M.

Replacing K changes d by+0.754107 with native V and+0.754930 with old V.
Replacing V changes d by+0.002764 with native K and+0.003588 with old K.
The factorial interaction is+0.000824. This particular older-key replacement
accounts for nearly all the joint replacement's+0.757694 shift; this particular
older-value replacement retains the native winner. This is not a claim that
values are generally unimportant or that every different key/value donor behaves
similarly. Native and V-only winner gaps are small but exceed the frozen0.001
guard and retain their reference parity; all conclusions stay within this
FP32/SDPA context.

The computation implicated by the K-only intervention is concrete. It changes
historical K while historical V remains native. In attention
softmax(qK^T/sqrt(d_head)+mask)V, this enters through the attention weights and
then subsequent residual/query/readout computation. Simply adding another exact
copy of the previous row's cached evidence is insufficient here. However, the
copied keys retain the previous row's rotary phase as well as its contextual key
features. The experiment therefore does not separate positional phase from
contextual key content, or identify the responsible layer/head.

## Converged judgment

The count-only prediction, a fixed additive source/downstream account of the
640 location contrast, and the sufficiency of this exact old-cache duplicate
for the native exit are all unsupported. The positive boundary is narrower and
more useful: influence depends on historical location; source and downstream
states can interact with a sign reversal; and the tested natural transition is
strongly changed by the added row's post-RoPE key state while retaining the exit
under the specified older-value substitution.

The lead therefore prioritizes history-conditioned attention weighting as the
computational explanation to refine, rather than interpreting emitted numeric
repetition as direct copying or treating a relative logit change as absolute
probability amplification. Positional phase versus contextual key content is the
remaining fork; it is not resolved by these results. This is a local causal
boundary at one transition, not an explanation of the entire62-row burst, a
general adaptive accumulator, or a physical-owner repair. No further donor,
row, layer, head or K/V scan follows this converged investigation.

## Evidence and execution boundary

Root independently reconstructed Stage A native and edited input tensors from
literal prompt/raw tokens, verified equal token and complete-row multisets,
checked consumed hashes and recomputed full-vocabulary probabilities and logits.
R and row87 match their previously accepted full vectors exactly.

For Stage B, root inspected the actual graft/crop/restore caller, reran its CPU
sensitivity check and verified all28 layers' consumed partition hashes and the
four exact restoration digests. Native and full-edited cached routes match
Stage A within4.7684e-05,3.7193e-05 and3.5286e-05, with identical global winners.
All seven S-endpoint vectors and their factorial reductions were independently
recomputed. Stages A/B verified31/35 referenced source/vector/capture bindings.
Two offset/scheduling defects were caught and fixed before Stage B's first
model call; they caused no failed GPU attempt or hidden extra forward.

Accepted receipts:
[Stage A](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-history-location/lead-checks/stage-A-readback.json),
[Stage B](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-history-location/lead-checks/stage-B-readback.json).
The [native bridge acceptance](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-history-location/lead-checks/native-bridge-readback.json)
verifies36 referenced bindings, all five full vectors, all28 actual per-layer
source/destination K/V assignments and mask membership, unchanged current S and
positions, and exact restoration. N/M maximum full-vector errors are4.52995e-05
and4.76837e-05 against L_L/E_L; both reference vocabulary winners agree.

The bridge's first attempt failed after three model invocations (the third
partial) because an instrumentation check assumed additive floating masks; the
installed caller returns boolean masks. It produced no M/B/K/V scientific
endpoint. Its receipt, source capture and qualified N vector remain immutable.
Root reproduced the fault through the installed create_causal_mask CPU caller,
then accepted the strict boolean correction with wrong-mask sensitivity checks.
The second attempt repeated the same five conditions and passed; cumulative
bridge cost exactly meets its nine-invocation cap. Failed evidence is not pooled
with the scientific result.

| Attempt | Model invocations | Vision calls | Execution-phase seconds | Elapsed seconds |
|---|---:|---:|---:|---:|
| A, location | 5 | 5 | 16.086 | 22.621 |
| B, cache partition | 10 | 3 | 54.687 | 61.644 |
| C001, instrumentation failure | 3 | 1 | not separately recorded | 23.486 |
| C002, qualified native bridge | 6 | 1 | 24.690 | 30.653 |

Total24 model invocations and10 vision calls on physical GPU4, including the
failed attempt. Successful attempts have95.463 execution-phase seconds; total
recorded elapsed across all attempts is138.404 seconds. These timers include
instrumentation and are not isolated CUDA-kernel time. Maximum recorded peak reserved memory is
15,575,547,904 bytes; the failed attempt did not record a peak. All four producer processes are terminal. There was no
training or free generation and no hidden extra model replay for acceptance.

Maintained producers support `--selfcheck` via their module entries:
`probes.training_set_completion.recurrence_history_location`,
`probes.training_set_completion.recurrence_history_cache_partition`, and
`probes.training_set_completion.recurrence_native_row_mass`. Their source captures
and full-vector artifacts are bound in the final
[lead receipt](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-history-location/lead-acceptance.json).
