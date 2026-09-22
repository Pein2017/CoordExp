# Paired natural results, physical review pending

Status: both natural producers and the paired cold consumer completed. Each
arm has896 images from8 successful112-image shards, fixed natural3084 decode.
The original detached chain failed only in the subsequent CPU blind renderer
entry (`ModuleNotFoundError: No module named 'probes'`). Preserve inference and
consumer artifacts; a separately logged invocation-only recovery is underway.

Root independently replayed the existing full cold consumer into
`evaluation/paired-root-verification-v1`. All scientific fields for both arms
matched, and the frozen blind32 queue was byte-identical. Original result
SHA256:580e65bc9e095e7c6fd6a97c59ece59b775241f17368a7807010d91ec6b81143.

Source: `evaluation/paired-consumer-v1/result.json` under this experiment's
output root. A is common credible continuation learning; B adds the frozen
greedy-repeat pairwise margin. Both start from the same N16 anchor.

## Observations

All896 combines640 previously exposed regression images with the new independent
confirmation256. It is not an896-image independent validation set.

| Panel/metric | N16 | A | B |
|---|---:|---:|---:|
| All896 TP@50 |4189|3993|3958|
| All896 GT-relative FP@50 |2304|1700|1694|
| All896 F1@50 |0.643867|0.653947|0.650398|
| All896 strict later-row repeats |199|29|20|
| All896 parser drops |69|43|48|
| All896 generated tokens |62505|54871|54532|
| Confirmation256 TP@50 |1189|1127|1110|
| Confirmation256 GT-relative FP@50 |577|477|480|
| Confirmation256 F1@50 |0.649195|0.643816|0.636650|
| Confirmation256 strict later-row repeats |22|4|6|
| Confirmation256 parser drops |37|12|12|

All arms terminate with EOS on all896 images; no token caps. Strict repeat
means each later valid row counted once at any-class native-pixelIoU>0.95.
It is a geometry diagnostic, not physical-owner adjudication.

Annotated-owner changes relative to N16 atIoU0.5:

- A/all896: gained142, lost338, retained3851; net−196.
- B/all896: gained128, lost359, retained3830; net−231.
- A/confirmation256: gained48, lost110, retained1079; net−62.
- B/confirmation256: gained45, lost124, retained1065; net−79.

The54 reference images have net TP397→401/399, but the old11 positive images
have101→89/77. Net TP is not a statement that every incumbent survived.
The matched A→B aggregate has9 fewer strict repeats and35 fewer TP across896;
on independent256, B has2 more strict repeats and17 fewer TP than A.

The existing native image-paired bootstrap was also applied directly to the
same accepted A/B rows (`direct-A-to-B-v1.json` in the original consumer root).
A→B/all896 gained63/lost98 annotated owners; confirmation256 gained17/lost34.
The10000-replicate95% intervals for deltaTP are[-72,-1] and[-37,0],
respectively. Both deltaF1 intervals include0. These are image-resampling
intervals for one trained pair, not training-seed uncertainty or proof of
universal inferiority of the extra objective.

## Bounded interpretation

The current recipe suppresses repetition but does not establish enumeration
improvement. The aggregate F1 increase must not obscure annotated-owner losses
or the independent confirmation regression. Common learning alone explains
most of the observed repetition reduction; the extra margin shows no favorable
confirmation tradeoff in this single matched comparison.

Output shortening is an alternative to better instance-completion behavior:
fewer tokens/predictions and fewer repeats coexist with loss of known owners.
This observation does not prove an EOS mechanism, a learned ledger, or that
all forms of deduplication learning are ineffective. Reference-local protection
is not a guarantee of natural-path preservation elsewhere.

GT-relative FP is NOT hallucination count. Unlabeled and ambiguous physical
owners remain neutral. The frozen32-image mixed-source queue contains607
proposals and no visual labels yet. Its physical-owner and dense-group review
is required before the round's final interpretation. No checkpoint promotion,
new training arm, threshold change, data-role change, or publication follows
from these preliminary numbers.
