# Frozen blind32 physical-owner accounting

Status: **candidate exact CPU join; root acceptance pending**. This independently
recomputed the frozen comparison from the eight decision files plus the now-
unsealed source map. It used no image viewing, relabeling, model/GPU work, or
new benchmark/statistical gate.

Receipt:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-label-vs-compilation/physical/result.json`

The previously accepted accounting is preserved byte-for-byte as
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-label-vs-compilation/physical/result-v1.json`
(SHA256 `b5fa958691f55af35c477e3a1965a327045cd0be785409d8e0bbb55b433e83f0`).
The current receipt is v2 and adds only exact source-prediction GT50
stratification.

Reproduce:

```bash
python research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-14-label-vs-compilation/physical/audit.py
```

## Exact result

All 32 image IDs match the manifest, 32 queue records, 607 flattened queue
proposals, 607 source-map rows, and the selected rows from each 896-row GT
source. All 607 proposals are assigned exactly once across atomic owner,
dense-group, unresolved, or non-owner evidence. The independent arithmetic
exactly matches frozen comparison SHA256
`b4da082ae70e63cdc0e6001b26e937966993bd14b006c8f6ea384ec27c21cbaf`.

### GT50 on exactly the same 32 IDs

Each arm has the same 201 annotated-owner denominator.

| Arm | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| N16-anchor | 139 | 71 | 62 | 0.661905 | 0.691542 | 0.676399 |
| A | 137 | 60 | 64 | 0.695431 | 0.681592 | 0.688442 |
| B | 133 | 67 | 68 | 0.665000 | 0.661692 | 0.663342 |

### Reviewed physical atomic owners

| Surface | N16-anchor | A | B | N16→A gain/loss/net | N16→B gain/loss/net | A→B gain/loss/net |
|---|---:|---:|---:|---:|---:|---:|
| All 198 reviewed atomic clusters | 187 | 167 | 170 | 5 / 25 / **-20** | 10 / 27 / **-17** | 11 / 8 / **+3** |
| Descriptive sensitivity: exclude every cluster carrying any caveat (61 retained; 137 excluded) | 57 | 48 | 50 | 1 / 10 / **-9** | 3 / 10 / **-7** | 6 / 4 / **+2** |

The caveat-free exclusion preserves the descriptive ordering and directions,
but is not a new benchmark or acceptance threshold. It changes the population
from 198 to 61 clusters and must not replace the primary ledger.

The GT50 and physical surfaces do not make the same decision: A has higher
GT50 F1 than N16 because its FP reduction outweighs two fewer annotated TPs,
while the reviewed finite physical ledger has 20 fewer atomic owners. This is
an observed measurement-surface divergence on the panel—not proof of missing-
label causality, and not evidence that every GT-unmatched proposal is an
unannotated object.

## Exact source-prediction GT50 stratification

For each changed atomic cluster, the audit follows its participating proposal
through `blind-review-source-map.json.source_prediction_index` to the exact
bound natural row, then tests membership in that row's stored
`score["50"]["matches"][].pred_index`. It performs no approximate box
rematching. All 607 source-map predictions have exact row/index
correspondence: 409 are GT50 matched and 198 are GT50 unmatched; none are
unidentifiable.

For a lost cluster, classification uses the participating N16 source
prediction(s). For a gained cluster, it uses the destination A or B source
prediction(s). `has_GT50_matched_source_prediction` means at least one exact
participating prediction matched GT50. `supported_but_GT50_unmatched` means all
exact correspondences exist but none matched GT50.

| Contrast / changed cluster side | Has GT50-matched source prediction | Supported but GT50-unmatched | Unidentifiable |
|---|---:|---:|---:|
| N16→A: 25 lost, classified on N16 | 6 | **19** | 0 |
| N16→A: 5 gained, classified on A | 2 | **3** | 0 |
| N16→B: 27 lost, classified on N16 | 7 | **20** | 0 |
| N16→B: 10 gained, classified on B | 2 | **8** | 0 |

Thus the cluster-level net partitions are:

- N16→A physical net −20 = **−4** among clusters with a GT50-matched
  participating source prediction plus **−16** among supported-but-GT50-
  unmatched clusters.
- N16→B physical net −17 = **−5** matched plus **−12** supported-but-GT50-
  unmatched.

The same pattern survives the existing caveat-free sensitivity:

| Caveat-free contrast | Lost matched / unmatched | Gained matched / unmatched | Net matched / unmatched |
|---|---:|---:|---:|
| N16→A | 3 / 7 | 1 / 0 | −2 / **−7** |
| N16→B | 3 / 7 | 0 / 3 | −3 / **−4** |

Observation: 19/25 A losses and 20/27 B losses were physical clusters whose
participating N16 predictions did not contribute a stored GT50 match. This
directly explains why GT TP can mask much of the reviewed physical loss.
It does **not** show that those clusters are truly unlabeled: class, extent,
IoU threshold, or assignment can produce `supported_but_GT50_unmatched`.

The matched-cluster net (−4 for A) is not required to equal the aggregate GT TP
change (−2). Physical clusters and GT assignments are different units;
retained physical clusters can change which exact prediction matches GT, and
one cluster may contain multiple predictions. This stratification therefore
locates metric visibility, not an additive decomposition of TP or a causal
estimate of annotation incompleteness.

## Per-image physical deltas

Format is `gained/lost/net`; the receipt contains all 32 images, exact owner
keys, caveat-free deltas, group coverage, unresolved counts, and GT50 rows.
The 13 images below carry every atomic change; the other 19 have zero gain and
zero loss for all three comparisons.

| Image | N16 / A / B owners | N16→A | N16→B | A→B |
|---:|---:|---:|---:|---:|
| 109798 | 3 / 4 / 4 | 1/0/+1 | 1/0/+1 | 0/0/0 |
| 227765 | 18 / 12 / 15 | 2/8/-6 | 4/7/-3 | 6/3/+3 |
| 269314 | 5 / 1 / 1 | 0/4/-4 | 0/4/-4 | 0/0/0 |
| 276707 | 3 / 1 / 2 | 0/2/-2 | 0/1/-1 | 1/0/+1 |
| 288762 | 12 / 12 / 10 | 0/0/0 | 0/2/-2 | 0/2/-2 |
| 312213 | 3 / 2 / 2 | 0/1/-1 | 0/1/-1 | 0/0/0 |
| 314182 | 11 / 7 / 7 | 0/4/-4 | 0/4/-4 | 0/0/0 |
| 341973 | 5 / 4 / 4 | 0/1/-1 | 0/1/-1 | 0/0/0 |
| 372307 | 2 / 1 / 1 | 0/1/-1 | 0/1/-1 | 0/0/0 |
| 447342 | 13 / 12 / 12 | 1/2/-1 | 1/2/-1 | 0/0/0 |
| 463199 | 12 / 12 / 12 | 1/1/0 | 2/2/0 | 2/2/0 |
| 495732 | 7 / 6 / 6 | 0/1/-1 | 0/1/-1 | 0/0/0 |
| 559099 | 17 / 17 / 18 | 0/0/0 | 2/1/+1 | 2/1/+1 |

## Uncertainty and dense groups

- Nine proposals remain unresolved and neutral: exactly three per arm, on
  images 463199 (3) and 495732 (6). They are not converted to owners, gains,
  losses, or non-owner evidence.
- A conservative proposal-level bound that treats every unresolved proposal as
  a distinct arm-only owner gives net intervals: N16→A **[-23,-17]**,
  N16→B **[-20,-14]**, A→B **[0,+6]**. This is deliberately loose: proposals
  may alias each other or an existing owner, and it is not an exhaustive-recall
  confidence interval. Under this bound the N16→A/B signs remain negative;
  A→B need not remain strictly positive.
- Thirteen dense-group clusters remain separate. Presence is N16/A/B =
  **11/11/10**. Their independent changes are N16→A 2 gained/2 lost,
  N16→B 1/2, and A→B 0/1. They are never added to atomic totals.
- Fourteen proposals are explicit non-owner evidence; this category also is not
  added to owners or treated as an annotation-derived negative training label.

## Boundary and stop

This finite union measures physical-owner presence among generated proposals
on the frozen 32-image panel. It cannot see owners missed by all three arms,
does not estimate exhaustive recall or all 896 images, and does not identify
training-label causality. Exact bounded accounting passed; no further review,
relabeling, inference, or expansion was started.
