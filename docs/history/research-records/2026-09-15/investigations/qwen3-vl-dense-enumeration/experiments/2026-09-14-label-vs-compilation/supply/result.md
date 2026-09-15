# Frozen4096 supply evidence audit

Status: **candidate CPU-only existing-evidence accounting**. Root owns acceptance
and any decision to request external review. This audit ran no model, viewed no
image, changed no labels, and did not inspect the current blind32 queue or source
map.

Machine-readable receipt: [result.json](result.json). Reproduce it with:

```bash
python research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-14-label-vs-compilation/supply/audit.py \
  --output research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-14-label-vs-compilation/supply/result.json
```

## Decision-bearing result

The frozen supply bank reaches **53 packages / 39 images**, exceeding the
32-package / 16-image launch floor but missing the 64-image aim by **25 distinct
images**. Existing evidence does **not** identify any of those 25 as a specific
unannotated physical owner. The large upstream loss is under the GT-backed
nomination contract, but `no GT-backed nomination` is not an annotation verdict.

At the distinct-image level, the exact staged waterfall is
`4096 frozen -> 198 nominated -> 186 with some candidate immediate w -> 39 with
a root-admitted c/w group -> 39 selected`. These are conditional acquisition
stages, not prevalence estimates of real missing objects.

| Frozen stage / reason | Exact denominator and result | Effect on distinct-image supply | What it identifies | What it does not identify |
|---|---:|---:|---|---|
| No GT-backed nomination | 3,898 / 4,096 images | 3,898 never enter conditional review | No job under the frozen strict-repeat-history plus supported-not-yet-covered-GT rule. Of these, 3,896 have no recorded candidate-filter hold and 2 have only `neutral_absence_or_budget_not_clear` holds. | An unlabeled owner, an empty image, or a useful counterfactual row. The original-image population was not censused. |
| Nominated, but no candidate immediate witness | 12 / 198 nominated images; 87 / 589 jobs | 12 images leave no candidate-w row | Job failures are 68 no valid free row, 16 strict-duplicate first row, and 3 nonliteral first content. | Whether a different label or candidate would have yielded a useful complete route. No retry/backfill was allowed. |
| Candidate immediate w, but no credible root c/w admission | 147 / 186 candidate-w images; 370 / 429 exact visual groups are HOLD | 147 images leave no admitted group | 369 groups retain unresolved joint physical c/w trust; 1 targeted audit found the apparent w was the same c person, not a distinct successor. | A negative label. The heterogeneous review records do not support a cleaner c-versus-w split, so all 370 remain unknown-neutral. |
| Root admission then max-two/image selection | 59 admitted groups on 39 images -> 53 packages on the same 39 images | 0 images; 6 extra admitted groups omitted | Frozen selection-policy effect only (`PAM-0101`, `PAM-0120`, `PAP-0016`, `PAR-0052`, `PAR-0053`, `PAR-0064`). | A credibility, label, or algorithm failure. |
| Later-continuation burden | 138 / 502 candidate-w rows have a machine burden; 7 / 53 selected packages do | **0 packages excluded** under the frozen immediate-w admission contract | Existing suffixes contain a later strict repeat, geometry-invalid row, other malformed content, or cap according to the stored ledger. All 53 selected suffixes reach EOS; none cap. | Semantic badness, physical owner loss, or a missing-label cause. Later suffix quality was not physically adjudicated for admission. |
| Technical/persistence incidents | 2 interrupted incomplete attempts and 156 uncommitted forwards; final pending natural=0, conditional=0, review groups=0, jobs=0; alias repair passed | **0 current packages lost** | Historical cost/failure receipts and a completed corrected join. | A scientific null or supply deficit. |

The 1,928 recorded `neutral_absence_or_budget_not_clear` GT-filter rows are
candidate-level holds across 188 images and can coexist with successful
nominations on the same image. They therefore must not be added to the 3,898
no-nomination images as another population loss.

## Conservative counterfactual-gap statement

Observation: the bank is 25 distinct images below its 64-image aim. Inference:
the frozen nomination contract is restrictive because only 198/4,096 images
produce any GT-backed conditional job. **Not identified:** how many of the
3,898 non-nominated images contain an unlabeled, conditionally useful owner, or
how many such owners would survive full-continuation review and training. Thus
the only defensible missing-label counterfactual from these artifacts is
**unknown (not zero, and not 3,898)**. No prediction-union count is converted
into an original-image census.

The current evidence instead directly identifies two later bottlenecks within
the nominated path: 12 nominated images lack any immediate candidate witness,
and 147 candidate-w images lack a root-credible c/w group. This shows where the
observed pipeline contracts, but it does not make those causes mutually
exclusive with historical label incompleteness.

## Small existing continuation-review list

There is **no supported missing-owner list**: creating one would require the
forbidden inference from non-nomination or prediction union to unseen owners.
There is, however, one complete, externally reviewable existing list of the
seven *selected* packages whose stored free suffix has machine burden. These
are suffix-quality review candidates, not new label candidates and not bank
rejections:

| Package | Existing visual group | Stored machine flag |
|---|---|---|
| `568202:h0:c0` | `PAP-0063` | 2 later strict repeats |
| `444315:h1:c0` | `PAR-0004` | 1 later strict repeat |
| `418115:h0:c0` | `PAR-0027` | 6 later strict repeats; 1 geometry-invalid row |
| `381458:h0:c0` | `PAR-0041` | 1 later strict repeat |
| `429065:h0:c0` | `PAR-0065` | 1 geometry-invalid row |
| `381458:h0:c1` | `PAR-0042` | 1 later strict repeat |
| `360767:h1:c1` | `PAR-0075` | 5 later strict repeats |

`result.json` binds each entry to its existing representative card and exact
`ordered-results.json` selector. Reviewing this list could adjudicate whether
machine burden is semantically harmful; it cannot estimate missing-owner
prevalence or authorize candidate expansion.

## Source boundary

The receipt binds the prior unit/result plus the exact completion, image-record,
ordered-job, ordered-result, corrected review-index, root-decision, transport-
proof, and final physical-bank files by absolute path, byte size, and SHA256.
Load-bearing identities include completion
`d2db76808104049317fda9d7f60de082a98ac2f562df2c4d149ac4f833f5014f`,
corrected review index
`7696236c72cd5f5e84dbd872069e2cfad98ccfa9a103a787ef8c00d8d5ea56c2`,
root decisions
`255c3ca9a432548db7a45d7fb6cf6d3a083874ccabdb7513c0ea0a95222a09c1`,
and physical bank
`93b17148731a315cafec6adbc59e336994e39503e6a2db0b367839fd055de4bd`.

Stop reached: exact frozen-source accounting is complete. No external review,
proposal-blind census, candidate expansion, label decision, or model work was
started.
