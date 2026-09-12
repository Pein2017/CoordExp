# Decision-aware preservation improves the aggregate, not every repair

Status: **completed; technically lead-accepted; promising tradeoff, not all
scientific gates passed; no checkpoint promotion**. This closes the selected
overnight portfolio. No coefficient, dose, reference-refresh, rescue branch or
additional inference is pending under this unit.

## What was actually changed

C starts from unchanged Stable50 and uses exactly A's three verified complete-row
positives, conditional KL, and 56 normal KL references. It adds one one-sided
worst-token margin floor per reference image, averaged across those 56 images.
The [frozen unit](unit.md) owns the formula, coefficient 10, eligible-token
selection, optimizer, resource bounds and stop. There is no new GT, matching
rule, architecture, token, or negative-learning event in C.

The new term protects **discrete next-token decisions** on original reference
prefixes rather than assuming a low average KL preserves greedy output.
It covers all 6,030 eligible source-argmax positions, not only A's first flips
or owner-loss images. The 17 near-ties retain KL only; nine originally masked
invalid-row tokens stay excluded. This is not owner-semantic memory and cannot
guarantee preservation under newly reached histories.

## Natural endpoint: complete ledger, not F1 alone

Exactly 384 original-input natural greedy reads, with unchanged RP1, cap 3084,
prompt, geometry, GT, parser and scoring. Stable50 and A are retained endpoints;
only C was newly executed. All owner/FP metrics are annotation-relative.
Reference56 is protection training data; dev128 is already exposed, not a fresh
holdout. Reference56 overlaps the union and must not be added as extra images.

| Union384 metric | Stable50 | Positive32 A | Margin-preserved C |
|---|---:|---:|---:|
| TP50 | 1899 | 1874 | 1923 |
| FP50 | 1522 | 1403 | 1127 |
| FN50 | 947 | 972 | 923 |
| F1@50 | 0.606032 | 0.612118 | 0.652307 |
| TP60 | 1787 | 1761 | 1812 |
| F1@60 | 0.570289 | 0.575208 | 0.614654 |
| TP80 | 1354 | 1342 | 1366 |
| F1@80 | 0.432105 | 0.438347 | 0.463365 |
| Raw row starts | 4213 | 3461 | 3078 |
| Parsed predictions | 3421 | 3277 | 3050 |
| Strict later-row repeats, pixel IoU > .95 | 582 | 432 | 153 |
| Geometry-invalid parser drops | 788 | 159 | 28 |
| Other malformed parser drops | 4 | 25 | 0 |
| Length caps | 4 | 1 | 0 |
| Natural EOS | 380 | 383 | 384 |

All 28 C parser drops are geometry-invalid. The scorer's post-parser
`invalid_predictions=0` does **not** mean no invalid boxes were generated.
C has 70 gained / 46 lost owners versus Stable50 at IoU50, net +24; it is not
loss-free recovery. C versus A is 87 gained / 38 lost, net +49.

### Paired owner identities across all registered thresholds

Each entry is **gained / lost / retained**; no FP row is silently discarded.

| Panel | Comparison | IoU50 | IoU60 | IoU80 |
|---|---|---|---|---|
| reference56 | C vs Stable50 | 2/2/414 | 1/2/380 | 4/6/262 |
| reference56 | C vs A | 32/5/384 | 30/10/351 | 22/12/244 |
| train256 | C vs Stable50 | 51/25/1260 | 51/28/1181 | 40/30/884 |
| train256 | C vs A | 52/22/1259 | 62/29/1170 | 51/37/873 |
| dev128 | C vs Stable50 | 19/21/593 | 20/18/560 | 18/16/424 |
| dev128 | C vs A | 35/16/577 | 37/19/543 | 28/18/414 |
| union384 | C vs Stable50 | 70/46/1853 | 71/46/1741 | 58/46/1308 |
| union384 | C vs A | 87/38/1836 | 99/48/1713 | 79/55/1287 |

| Panel | TP50 Stable / A / C | F1@50 Stable / A / C |
|---|---|---|
| reference56 | 416 / 389 / 416 | 0.662948 / 0.615506 / 0.660842 |
| train256 | 1285 / 1281 / 1311 | 0.591348 / 0.640660 / 0.670245 |
| dev128 | 614 / 593 / 612 | 0.639250 / 0.558380 / 0.616935 |

The reference owner *count* returns to 416, but two old owners are exchanged
for two others, and reference F1 is slightly below Stable50. Dev128 remains
TP50 -2 and lower F1 than Stable50, despite a large recovery relative to A.
Do not claim independent generalization or universally restored quality.

## Per-case no-weakening gate: failed

| Fixed diagnostic | Stable50 TP50 | A TP50 | C TP50 |
|---|---:|---:|---:|
| 351017 | 1 | 12 | 14 |
| 417044 | 1 | 11 | 10 |
| 477415 | 2 | 16 | 15 |
| 39654, not a positive target | 10 | 0 | 9 |

All four C diagnostics terminate naturally with no strict repeats. This repairs
A's new 39654 cap without adding that image to the protection set. However,
417044 and 477415 each lose one A-matched annotated owner. Summing the three
positive cases gives 39 for both A and C and would conceal that gate failure.

Root personally inspected all four actual A-versus-C `view_image` PNGs after
freshly verifying the C consumer and source/manifest/PNG hashes:

- **39654:** A's upper-left tiny-banana repeat pile-up disappears; C detects
  people and several bottles. C still begins with a broad group-like banana
  box, which is not promoted to a correct instance target.
- **351017:** the natural repeat pile-up largely disappears, with people,
  glasses and table retained. Background bottle annotation status remains
  unresolved; this does not validate all red FP boxes as hallucinations.
- **417044:** lost donut owner `1079910` has best same-class IoU only 0.076107
  in C, versus 0.962264 in A. This is substantial local coverage loss, not a
  near-threshold scoring accident. Many other red boxes visibly cover donuts;
  unmatched is still not equivalent to hallucination.
- **477415:** lost chair owner `1598977` has a nearby C chair box extended
  downward to the image boundary: best IoU 0.584423 -> 0.353376. This is an
  extent regression rather than proof that the physical instance was forgotten.
  It nevertheless fails the frozen annotation-relative no-weakening gate.

The [visual review](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-11-margin-preserved-positive-branch/visual-review.json) records exact geometry and
image identities. Four selected diagnostics are not a representative FP audit.
Renderer `dup-cand` counts overlapping pairs; the primary table instead counts
strict later rows. Those denominators must not be interchanged.

## Fixed-history conditional read: natural escape is not basin removal

Forced candidate rows receive no free credit. All six C continuations reach
EOS; additional valid rows are not automatically new owners.

| Case / branch | A valid free / repeats / geometry-invalid / other malformed | C valid free / repeats / geometry-invalid / other malformed |
|---|---|---|
| 351017 h_only | 97 / 56 / 1 / 20 | 80 / 47 / 20 / 7 |
| 351017 h_plus_c | 96 / 56 / 1 / 20 | 79 / 47 / 20 / 7 |
| 417044 h_only | 22 / 0 / 0 / 0 | 27 / 0 / 0 / 0 |
| 417044 h_plus_c | 21 / 0 / 0 / 0 | 23 / 0 / 0 / 0 |
| 477415 h_only | 23 / 0 / 0 / 0 | 23 / 0 / 0 / 0 |
| 477415 h_plus_c | 22 / 0 / 0 / 0 | 22 / 0 / 0 / 0 |

351017's supplied old-history chain remains bad even though its original-input
natural C output is much cleaner. In **each** conditional branch, total drops
increase 21 -> 27 (geometry-invalid 1 -> 20, other malformed 20 -> 7), while
strict repeats fall 56 -> 47. Reporting geometry alone or repeats alone would
misdescribe this tradeoff. The conditional preservation gate is not passed.

## Training signal and technical acceptance

- Paired two-update, eight-rank smokes were lead-accepted. Weight zero exactly
  reproduced retained A2; weight ten had zero initial margin force and an
  actually consumed nonzero gradient once floors were violated.
- One full C32 completed, followed by exact cold reload of three positive
  scores. Actual post-update reads of all 56 references, not pre-step32 logs,
  give 143 / 6,030 eligible flips versus A's 576 / 6,030. The A count 586 is
  over all 6,047 protected tokens and is not the fair eligible denominator.
- Final C `R=0.0259664`, mean normal KL `0.00424505`, 30 active images and
  296 violated floors. Floor violations and argmax flips are different counts.
- C achieves less positive NLL reduction than A; one of ten literal candidate
  tokens on 417044 is not argmax. Fixed-row token realization and natural owner
  preservation are separate outcomes, not interchangeable success labels.
- Full C: 8 loads, 3,566 model/image forwards, 3,328 backwards, 256 gradient
  synchronizations, zero sampling. No shared trainer rewrite or source adapter
  overwrite. Full training plus cold reload used 1.941405 allocated GPU-hours.
- Endpoint: 8 loads, 390 image/continuation calls,
  32,508 model forwards/tokens; maximum rank
  5,383 / 15,000 tokens and
  470.947 s / 1,500 s.
  Peak allocated/reserved CUDA and RSS were within 24 GiB/rank.
  Allocated rank time was 2979.909316 GPU-seconds
  (0.827753 GPU-hours). One launch, eight zero
  exits, no retry; root checked all eight recorded worker PIDs were released.

Root freshly ran `merge(packet, endpoint_C, verify=True)`, recomputing parsed
natural scores, conditional partition, exact identities, complete counters,
resource limits and all four merged outputs. Previous smoke/tests/cold checks
remain accepted; no duplicate audit or new inference was used for closeout.

## Portfolio synthesis and next recommendation

**Observation:** the [history cross](../2026-09-11-checkpoint-history-cross/results.md)
shows 39654's cap/owner loss following the supplied first row under both
checkpoints, while 417044/477415 outcomes follow checkpoint under either row.
Thus entry choice can expose an already available bad conditional trajectory;
other cases require changing continuation behavior, not only the entry row.

**Observation:** the [microscope](../2026-09-11-greedy-preservation-microscope/results.md)
finds small average KL alongside many actual greedy flips. C's new margin term
substantially reduces eligible flips and reference-owner loss and improves
aggregate burden/coverage, but does not preserve every trained repair.

**Supported inference:** decision-aware preservation is a more promising local
lead than increasing the already-closed coordinate-unlikelihood penalty. The
small-owner/drift symptom is not one established universal circuit: fragile
entry decisions and history-conditioned continuation failure coexist. C can
avoid an old bad history naturally without making that supplied history safe.

**Strongest remaining alternative:** C may partly work by reducing effective
positive-learning dose, rather than by a uniquely useful preservation geometry.
The worse achieved positive NLL and per-case regressions keep this explanation
open. This experiment does not identify a universal KL limitation, a necessary
margin mechanism, owner-ledger implementation, KV circuit, special-token role,
or why the original training produced these conditional states.

**Next proposal, not an authorized continuation:** prioritize one positive-only
control preselected from archived A training progress to match C's achieved
positive learning as closely as possible. Freeze the selection rule before
reading that control's natural endpoint; no coefficient/dose sweep. Compare
complete natural owner gains/losses, all parser/repeat/cap burden and each repair,
not only fixed-prefix NLL. If C's advantage survives this cheapest specificity
check, use a fresh independently frozen evaluation before any promotion.
Actual-prefix recursive owner-preserving training remains a candidate, not a
validated or launched algorithm. Do not add an unverified matcher/Co-DETR reward
or start KV surgery to explain the current finite-panel result.

The previously closed strict-event B arm had zero detected events and exactly
A's weights; it was a signal-acquisition null, not a successful dedup intervention.
The earlier coordinate UL reduced repeats while increasing invalid boxes. Neither
result justifies blindly strengthening a negative-learning coefficient.

## Authoritative artifacts and closure

- [Root acceptance receipt](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-11-margin-preserved-positive-branch/endpoint-admission.json) owns technical,
  scientific, promotion and stop statuses and binds all exact hashes.
- [Result](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-11-margin-preserved-positive-branch/endpoint-C/result.json) owns complete numerical reductions;
  [natural consumer](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-11-margin-preserved-positive-branch/endpoint-C/consumer.json) and
  [conditional consumer](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-11-margin-preserved-positive-branch/endpoint-C/conditional-consumer.json) preserve rows.
- [Training receipt](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-11-margin-preserved-positive-branch/full-C/receipt.json),
  [cold reload](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-11-margin-preserved-positive-branch/full-C/cold-check.json) and
  [endpoint packet](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-11-margin-preserved-positive-branch/endpoint-preparation/packet.json) bind execution.
- [Delegation closeout](../2026-09-11-positive-branch-vs-repeat-event/delegation-evidence.md)
  includes the actual Sol/Luna corrections and the unauthorized early local
  render incident; root accepted existing images only after source verification.

The frozen portfolio is complete, with useful all-eight-GPU production stages
and long event-driven worker waits. Wake registration failed and was never
armed; no automatic wake/background-monitor success is claimed. No more jobs
are pending and no held-out, checkpoint, GT, architecture or publication
promotion has occurred. Negative gates do not extend the stop rule.
