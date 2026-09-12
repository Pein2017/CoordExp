# Matched progress supports reference protection, not uniform owner recovery

**Completed; technically lead-accepted; no checkpoint promotion.** One fixed
A17 reconstruction (D), one cold reload and one384natural+6conditional endpoint
were executed. The [frozen unit](unit.md) stop is reached. No dose/coefficient
sweep, new dataset, further training or active GPU worker remains.

## Selection and exact reproduction

Before a D endpoint existed, the frozen scalar-only rule selected the archived
A state closest to C32's sum of three complete-positive-row NLLs. D is **after
17 updates**, measured by A's score reads **before update18**, not A18.

| Positive case | C32 NLL | D17 NLL | D minus C | C / D argmax hits |
|---|---:|---:|---:|---|
| 351017-c01 | 7.720066071 | 7.362805367 | -0.357260704 | 11 / 11 |
| 417044-c01 | 6.024546146 | 6.332001686 | +0.307455540 | 9 / 9 |
| 477415-c02 | 7.870375156 | 7.758611202 | -0.111763954 | 9 / 9 |

Q_C=21.614987373352,
Q_D=21.453418254852; relative residual
**0.747487%**, within the predeclared10% feasibility bound.
Per-case relative differences are -4.63%,+5.10%,-1.42%; token-hit counts match,
but this does not match complete conditional distributions.

All17 reconstructed adapter, reduced-gradient, optimizer hashes and pre-step
positive scalars match the archived A trajectory. The final state is
`80c9c39c42e9e76dfeda8b10abf7a66928bbf4fd1e1a7c1df79af6a47e1c5dd4`.
Cold reload reproduces all three score dictionaries exactly. This is a real
intermediate A model, not weight interpolation, a selected new sample, or a
C checkpoint relabeled D. No-margin means the original A objective still
includes positive learning plus conditional KL10 and normal KL100.

## Primary natural comparison

All384 images use the unchanged original prompt, image bytes, greedy/RP1,
cap3084, GT, geometry, parser, global assignment and pixel-IoU>.95 later-row
repeat rule. All owner/FP metrics are annotation-relative. Raw result orientation
is **D_after_vs_C_before**; gained means D-only and lost means C-only.

| Union384 | Stable50 | C32 margin | D17 no margin |
|---|---:|---:|---:|
| TP50 | 1899 | 1923 | 1896 |
| FP50 | 1522 | 1127 | 1479 |
| FN50 | 947 | 923 | 950 |
| F1@50 | 0.606032 | 0.652307 | 0.609548 |
| TP60 | 1787 | 1812 | 1779 |
| F1@60 | 0.570289 | 0.614654 | 0.571934 |
| TP80 | 1354 | 1366 | 1347 |
| F1@80 | 0.432105 | 0.463365 | 0.433049 |
| Row starts | 4213 | 3078 | 3813 |
| Parsed predictions | 3421 | 3050 | 3375 |
| Strict repeats | 582 | 153 | 469 |
| Geometry-invalid drops | 788 | 28 | 436 |
| Other malformed drops | 4 | 0 | 2 |
| Length caps | 4 | 0 | 2 |

C versus D recovers68/losses41/retains1855 IoU50 owners, net **+27**; it is
not loss-free. D versus Stable50 gains88/loses91, net-3. C's primary aggregate
advantage therefore survives matching the achieved positive-learning scalar.
This weakens the simplest “C only helps because the positives were learned
less” explanation, but the location of the benefit is decisive.

### Paired owner identities (D gained / lost / retained versus C)

| Panel | IoU50 | IoU60 | IoU80 |
|---|---|---|---|
| reference56 | 11/42/374 | 17/39/342 | 14/32/234 |
| train256 | 26/55/1256 | 32/60/1172 | 34/54/870 |
| dev128 | 15/13/599 | 11/16/564 | 13/12/430 |
| union384 | 41/68/1855 | 43/76/1736 | 47/66/1300 |

### Where the advantage lives

| Population | Images | C / D TP50 | C / D repeats | C / D parser drops |
|---|---:|---|---|---|
| reference56 | 56 | 416 / 385 | 0 / 246 | 1 / 417 |
| train256 | 256 | 1311 / 1282 | 31 / 305 | 11 / 429 |
| exposed dev128 | 128 | 612 / 614 | 122 / 164 | 17 / 9 |
| outside reference | 328 | 1507 / 1511 | 153 / 223 | 27 / 21 |

C's +27 aggregate TP50 decomposes into **+31 on the56 protection references,
-4 on the other328 images**. Reference56 overlaps train256 and the union;
these rows are not extra observations. Outside references C still has better
F1/repeat burden and +11/+1 TP60/80, but more geometry-invalid drops (27vs21).
This is mixed transfer, not “no benefit of any kind outside references” and
not uniform missing-owner improvement. Exposed dev128 is not fresh holdout.

D's two new caps are both protected references:

| Image | C / D TP50 | D strict repeats | D parser drops |
|---|---|---:|---:|
| 25274 | 12 / 0 | 124 | 198 |
| 511251 | 8 / 0 | 113 | 219 |

**Posthoc sensitivity, not a changed primary denominator:** excluding these
same two images from both models leaves382 images, C TP50+7 and79 fewer
repeats, but28vs21 parser drops. Within the remaining54 references C still
has TP50+11. Thus the owner advantage is not entirely two catastrophes, while
all417 D reference parser drops occur in those two cases. Keep both the full
ledger and this descriptive concentration visible.

## Positive repairs and conditional behavior remain tradeoffs

| Natural diagnostic | C TP50 | D TP50 |
|---|---:|---:|
| 351017 | 14 | 14 |
| 417044 | 10 | 11 |
| 477415 | 15 | 17 |
| 39654, not a positive target | 9 | 11 |

D has one more donut owner and two more chair owners on the latter two positive
cases despite matched literal positive token-hit counts. C is not a universal
winner. Both models terminate on all four named natural diagnostics.351017 D
has4 repeats and one geometry drop versus C's zero;39654 D has no repeats and
one geometry drop, and its former A32 cap is absent.

| Supplied history | C valid free / repeats / geometry-invalid / other malformed | D valid free / repeats / geometry-invalid / other malformed |
|---|---|---|
| 351017 h_only | 80 / 47 / 20 / 7 | 94 / 53 / 0 / 25 |
| 351017 h_plus_c | 79 / 47 / 20 / 7 | 93 / 53 / 0 / 25 |
| 417044 h_only | 27 / 0 / 0 / 0 | 23 / 0 / 0 / 0 |
| 417044 h_plus_c | 23 / 0 / 0 / 0 | 22 / 0 / 0 / 0 |
| 477415 h_only | 23 / 0 / 0 / 0 | 23 / 0 / 0 / 0 |
| 477415 h_plus_c | 22 / 0 / 0 / 0 | 22 / 0 / 0 / 0 |

All six D branches reach EOS; forced rows are excluded from free credit.
For351017 C has fewer repeats (47vs53) but more total drops (27vs25), including
20 invalid geometries versus D's zero. Natural improvement is not elimination
of the bad supplied-history basin. Valid free rows are not certified new owners;
no new conditional positive labels or physical-owner adjudication are claimed.

## Root visual checks

Root personally used `view_image` on four actual C/D comparisons after source
and consumer verification. No model was rerun or FP/GT label changed.

- **25274:** D starts from a small left-edge person box then produces repeated,
  broad crowd-like boxes and caps. C traverses the crowd and traffic lights.
  Many C red boxes have visible physical/depicted-person support; these are
  not uniformly hallucinations.
- **511251:** D repeats tiny upper-left background chair boxes and caps. C
  generates useful background detections, but the foreground suitcase remains
  GT-unmatched with a much larger C box: escaping repetition is not clean
  full-scene detection.
- **417044:** D restores donut owner1079910, best same-class IoU0.0761(C)
  ->0.9623(D), a substantial coverage difference rather than a threshold jitter.
- **477415:** D corrects chair1598977's extent (IoU0.3534->0.5844) and adds
  a separate matching box for1599562 (0.0141->0.6731). Owner count changes
  can reflect both extent and actual row availability, not one “forgetting” type.

The [root diagnostics](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-11-positive-progress-matched-control/root-diagnostics.json) preserve exact geometry,
source/PNG hashes and selected IDs. Visualization duplicate hints count pairs,
not the primary later-row denominator. These selected images are not an FP
population audit. The root's initial CPU visual glue passed a list to a helper
expecting a tuple; it failed before rendering. Partial adapters are preserved
in `visual-review/`; a corrected local call produced `visual-review-v2/` and
all four accepted PNGs. No shared renderer or scientific artifact was changed.

## Decision and remaining alternative

**Supported:** at nearly equal achieved positive-row learning, C preserves the
reference behavior substantially better than A17. Eligible literal argmax flips
are143/6030 in C versus442/6030 in D; reference TP is416vs385. Matching the
scalar dose did not explain away C's aggregate/reference advantage.

**Not established:** that the margin penalty has a uniquely necessary geometry,
or transfers to general owner recovery. C32 and D17 do not match total optimizer
steps or normal/conditional KL exposures; those training-path differences are
the strongest remaining confound. Scalar/per-case NLL and discrete hit counts
also do not equate full distributions or reached natural histories. This is a
bounded preservation result, not a proof of a KV circuit, owner ledger,
special-token mechanism, or why pretraining produced the bad states.

**Next recommendation, not a launch grant:** for practical value, freeze one
fresh, unselected evaluation of retained Stable50/C32/D17 before more training
or promotion. An equal-update/equal-KL-exposure control is a separate option if
mechanism specificity becomes the decision; do not append an endless control
or coefficient sweep to this completed unit. The current question is answered
at its declared scalar-matched level, with transfer limitations explicit.

## Execution and delegation acceptance

Root passed13 focused tests, current prepared-input validation, native D17
receipt verification, cold identity/vector equality and fresh actual endpoint
`merge(..., verify=True)`.17/17 archived update checks passed on the first real
reconstruction route. Full D used1961 model/image forwards,1768 backwards,
136 synchronizations,80 initial reference and56 final-reference reads,8 loads,
zero sampling; cold added one load/three score reads. Final diagnostic margin
has689 active floors/54images, raw R0.27697595 and raw KL0.00541736, but zero
margin training force. These counts are not a claim that each flip loses an owner.

The endpoint used390 image/continuation calls,8 loads and39,299 model forwards;
maximum rank7286/15000 forwards and617.891/1500 seconds. CUDA allocated/reserved
and RSS stayed below24GiB/rank. One launch, all8 outer exits zero, no retry;
root checked the endpoint PIDs absent. Complete logs/resources remain bound.
Summed allocated rank lifecycle plus the one cold read was2.033129 GPU-hours
(training3871.800268s, cold8.909345s, endpoint3438.555705s); this excludes CPU
preparation/idle wall time and is not measured device compute utilization.

Flat L1 only: Luna-max's read-only scalar extraction was independently accepted;
Sol-high's D17 and Sol-xhigh's endpoint both passed the first real route and
root acceptance without a worker semantic correction in this unit. They owned
disjoint write surfaces and did not launch before grants. This is observed task
success, not a controlled effort/family ranking. Root's tuple-adapter error is
lead-side glue rework, not a worker failure.

One new wake registration attempt failed after10.2s at
`resolve_delivery_target / app-server thread/read timed out`; no monitor was
armed. Root remained active with long native event waits, not short polling.

## Authoritative artifacts

- [Root acceptance](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-11-positive-progress-matched-control/root-acceptance.json) owns technical/scientific/stop status.
- [Selection](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-11-positive-progress-matched-control/selection.json) owns all33 score/state sources and the frozen k17 choice.
- [D17 receipt](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-11-positive-progress-matched-control/D17/receipt.json) and [cold reload](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-11-positive-progress-matched-control/D17/cold-check.json).
- [Actual endpoint packet](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-11-positive-progress-matched-control/endpoint-preparation/packet.json), [result](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-11-positive-progress-matched-control/endpoint-D/result.json),
  [natural consumer](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-11-positive-progress-matched-control/endpoint-D/consumer.json) and [conditional consumer](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-11-positive-progress-matched-control/endpoint-D/conditional-consumer.json).
- [Diagnostic slices/visual review](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-11-positive-progress-matched-control/root-diagnostics.json) and [render manifest](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-11-positive-progress-matched-control/visual-review-v2/images/manifest.json).

No new GT, reward, architecture, special token, independent holdout or promoted
checkpoint was introduced. The one-control stop is complete.
