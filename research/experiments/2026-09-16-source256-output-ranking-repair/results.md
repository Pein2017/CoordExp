# Paired16 output ranking repair: accepted execution, repair gates unmet

The fixed P/R16 experiment is technically accepted. R improves full-train and dev known-owner coverage over P and the Bnormalized64 start, but fails the frozen repair conjunction: outside-reference coverage falls and training repeat burden rises. Neither arm passes promotion. Stop at this endpoint.

P uses canonical replay plus complete preferred-output CE; R adds the frozen reference-anchored, length-normalized ranking surrogate. Both start from the same Bnormalized64 adapter and paired frozen embeddings, with fresh optimizers. The [protocol](unit.md) owns the unchanged objective and dose.

## Natural greedy outcome

| Endpoint | Train FN | Train G/L vs Source | Dev FN | Dev G/L vs Source |
|---|---:|---:|---:|---:|
| Source0 | 688 | — | 271 | — |
| A64 | 628 | 95/35 | 280 | 17/26 |
| B64 | 667 | 96/75 | 284 | 24/37 |
| Bnormalized64 | 652 | 91/55 | 276 | 20/25 |
| P16 | 638 | 99/49 | 284 | 21/34 |
| R16 | 632 | 90/34 | 265 | 23/17 |

Processed rescale_32_1024_bbox_len12000_xy_sorted; frozen train256/1,988 owners and dev128/891 owners. Empty-prefix natural greedy, bs4, RP1, cap3084; fixed class-agnostic one-to-one IoU50. G/L is matched-identity change, not physical discovery/forgetting.

## Concentration and identities

| Endpoint | Selected15 FN | Descriptive14 FN | Image548337 FN | Outside241 FN |
|---|---:|---:|---:|---:|
| Bnormalized64 | 116 | 99 | 17 | 536 |
| P16 | 101 | 84 | 17 | 537 |
| R16 | 89 | 80 | 9 | 543 |

R versus P gains 12 matches on selected15: 8 on image548337 and 4 on the other14; outside241 loses 6. Relative to the start, R gains 27 selected matches and loses 7 outside. The dev improvement is real within this frozen evaluation, but does not erase the failed outside-reference gate. The 14-image slice is descriptive, not another arm or a replacement endpoint.

| Endpoint | Actual starting91 retained | Lost | Different replacement gains | Final G |
|---|---:|---:|---:|---:|
| P16 | 75 | 16 | 24 | 99 |
| R16 | 59 | 32 | 31 | 90 |

R fails the separate original gain-count diagnostic (90 < 91); neither arm preserves all starting gains. P’s G99 also does not mean its original gains are intact. Exact survived/lost/replacement identities and per-owner endpoint comparisons are retained in the result and receipt.

## Output burden

| Endpoint/split | Malformed | Repeat proxy | Cap | EOS debt | Invalid geometry counter | Annotation-unmatched UNKNOWN |
|---|---:|---:|---:|---:|---:|---:|
| A64/train | 152 | 432 | 2 | 2 | 0 | 983 |
| A64/dev | 18 | 96 | 0 | 0 | 0 | 454 |
| Bnormalized64/train | 660 | 303 | 3 | 3 | 0 | 842 |
| Bnormalized64/dev | 4 | 140 | 0 | 0 | 0 | 532 |
| P16/train | 1148 | 263 | 4 | 4 | 0 | 857 |
| P16/dev | 298 | 120 | 1 | 1 | 0 | 469 |
| R16/train | 578 | 659 | 4 | 4 | 0 | 1268 |
| R16/dev | 16 | 111 | 0 | 0 | 0 | 477 |

R improves malformed rows over P but train repeat proxy rises 263→659; cap remains4. Annotation-unmatched rises857→1268 train and469→477 dev; these are UNKNOWN, not confirmed FP. Even excluding unmatched counts from an error interpretation, repetition independently fails the debt gate. The repeat metric is box overlap IoU>0.95 against earlier valid rows, not verified duplicate identity. The legacy invalid-geometry counter is zero; geometry-related parser drops can still be included in malformed rows.

## Frozen-output likelihood changes

Image-equal means relative to the frozen anchor. Δlogp/token divides each member by its own observed token length for description only; training uses the same fixed max-length denominator for both members. Anchored margin is Δ(ell+−ell−)/d.

| Arm/slice | Preferred Δlogp/token | Rejected Δlogp/token | Anchored margin Δ |
|---|---:|---:|---:|
| P/descriptive14 | 0.170088 | 0.140856 | 0.030382 |
| P/full15 | 0.171162 | 0.140892 | 0.019641 |
| P/image548337 | 0.186188 | 0.141406 | -0.130720 |
| R/descriptive14 | 0.160506 | 0.067398 | 0.093001 |
| R/full15 | 0.161311 | -0.020799 | 0.171165 |
| R/image548337 | 0.172587 | -1.255555 | 1.265461 |

R separates the frozen pairs more strongly. On the other14, rejected likelihood still increases on average; the full15 negative rejected delta is driven by548337. That image’s rejected SUM delta is −3872.13 in R versus +436.10 in P; preferred SUM delta is +30.55 versus +32.96. Full per-pair SUM and token-normalized changes are in the receipt. This supports an effective ranking intervention, not isolated useful negative token/owner credit.

## Acceptance, interpretation and stop

Repair checks pass for train FN versus P/start, old-owner loss reduction (55→34), and dev FN versus P/start. They fail for outside-reference coverage and output debt. Promotion also fails: R train FN632 is not <628 and debt is worse than A; R does satisfy old loss≤35, dev FN≤271 and dev old loss≤26. P fails promotion as well.

The strongest supported interpretation is bounded checkpoint repair with substantial concentration and identity turnover, not generalized credit success. Complete-output ranking also adds positive learning; this contrast does not isolate negative credit or prove local token/action correctness. All80 annotation-unmatched preferred rows remained UNKNOWN while contributing positive CE in BOTH arms. No claim of neutral gradients, exact whole-sequence probability optimization, QP guarantee, or prevention from Source is made. One seed, a selected reference population and previously used dev limit generalization.

No additional dose, seed, coefficient, refresh, weak-bank work, visual census or training is authorized by this result. Original B and normalization results remain unchanged.

## Technical evidence

Both arms completed16 applied updates from the original anchor, not qualification state. Each consumed512 canonical and512 pair presentations; canonical images appear twice, pair images34/35 times with identical P/R schedules. P used512 model calls/1024 logical forwards; R768/1536. Reference caching and each endpoint likelihood scored30 routes in16 calls. All32 natural readback shards (768 images,192 bs4 batches) passed admission.

Qualification exercised R gradients, save/reload and a cold bs4 consumer. One qualification readback validation defect was repaired and only that readback rerun; original failure evidence remains. Main evaluation was independently recomputed from saved rows with exact result equality. Lead checks verified bindings, checkpoint identity, all update schedules and ranking accounting, fixed lengths/EOS, likelihood token identities and actual91 set differences.

- [Result](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-source256-output-ranking-repair/runtime/main-v1/result.json), SHA256 `26f59a8409a64c23089b9bbfa4e5e3bb8af6b3f9de1971227fb55c17840fe783`.
- [Lead acceptance receipt](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-source256-output-ranking-repair/lead/final-acceptance-v1.json): config/checkpoint/source digests, exact counts, gates, likelihood deltas and owner identities.
- [Acceptance check](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-source256-output-ranking-repair/lead/check-acceptance.py) and [independently recomputed result](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-source256-output-ranking-repair/lead/recomputed-result.json).
- [Qualification acceptance](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-source256-output-ranking-repair/lead/qualification-acceptance.json); [launch](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-source256-output-ranking-repair/runtime/main-v1/run.sh).
