# Numerical-return attrition in the frozen 273 sources — CPU candidate

**Status:** retrospective CPU candidate for lead review, 2026-09-24. This audits support for the frozen screen; it does not select replacement images or test a model mechanism. The [full machine record](numerical-return-attrition-v1.json) retains every qualifying earlier row and the per-pair triple checks. The [independent CPU receipt](numerical-return-attrition-verify-v1.json) reproduces all 31 count fields, all 29 image-level earliest returns, the known book diagnostic and the unchanged R0/D2 screen.

## Bound source and calculation

The source set remains **273 original-policy images** (145 mature, 128 prospective), in frozen order. All three previously excluded images remain in these diagnostic denominators. There are 3,467 complete canonical rows: 2,643 valid-geometry and 824 invalid-geometry. A numerical return is a later same-description row with an earlier **identical full serialized row**, or a different full row whose native boxes are both valid and have coordinate-bin IoU ≥0.5. The latter is only a numerical-nearness definition, not physical identity or the annotation threshold. Invalid exact repeats are separate. A return row can have several qualifying predecessors and several partner kinds; all are retained. Its displayed earliest predecessor is chronological shorthand, not an owner assignment.

| Unit and predicate | Count | Distinct images | Adjacent pairs | Nonadjacent pairs |
| --- | ---: | ---: | ---: | ---: |
| Any numerical return | 1,512 later rows | 29 | — | — |
| Valid exact full-row repeat | 557 later rows | 11 | 534 | 37,307 |
| Invalid-geometry exact repeat | 806 later rows | 3 | 806 | 100,851 |
| Valid nonexact near box | 356 later rows | 28 | 162 | 2,688 |

Partner-kind row and image counts **overlap**: 207 return rows have multiple kinds, so columns must not be summed. There are 1,502 return rows with an adjacent qualifying predecessor and 10 with only nonadjacent predecessors. Among the 29 earliest numerical returns, 28 are valid nonexact near pairs and one is a valid exact pair; none first appear as an invalid exact repeat. Repetitive tails create 142,348 qualifying pairs across the three types, but they remain only 29 source images. Pair counts are unsuitable as independent-case or recurrence-rate denominators.

## Attrition at the original adjacent-triple screen

There are 696 adjacent **valid-geometry** return pairs. Of these, 691 have a preceding third row and five do not. For each possible `(j−2,j−1,j)` triple, the record keeps continuity, class-token equality, each row's validity, unrounded best and runner-up same-class annotation IoU, group flags, the ≥0.75/<0.25 proxy gate, and the A≠B=B proxy pattern and chronology guard when defined. The table below uses the frozen conjunct order; it assigns each pair to its **first** failure only.

| First failing conjunct | Pairs / 696 | Images whose first adjacent valid pair fails here / 29 |
| --- | ---: | ---: |
| No preceding third row | 5 | 5 |
| Invalid geometry in A | 2 | 2 |
| Description class mismatch | 4 | 3 |
| A best annotation IoU <0.75 | 680 | 13 |
| A runner-up IoU ≥0.25 | 1 | 0 |
| B best annotation IoU <0.75 | 3 | 3 |
| B runner-up IoU ≥0.25 | 1 | 1 |
| No adjacent valid pair in image | — | 2 |

Those first-failure pair counts sum to 696; the image column sums to 29 and gives each source at most one first-adjacent-pair diagnosis. Failures overlap before ordering: best IoU below 0.75 occurs in A/B/C for 684/687/688 pairs, runner-up IoU at or above 0.25 in 38/38/35, class mismatch in four, invalid A geometry in two, and missing third row in five. These are separate flags on the same 696 pairs, not additional exclusions. Annotation-missing, explicit group flags, A≠B=B, chronology and earlier-visit fields are retained per triple in JSON; a downstream field is undefined if a prior proxy gate fails. The absence of a particular flag in this valid-pair subset does not certify annotations or owners across all numerical-return rows.

The known book `new:151704`, rows 3→4, is included only as a diagnostic. Its valid nonexact native boxes have IoU **0.6616438356164384** and the preceding row 2 is present, contiguous and book-class. The three best annotation overlaps for rows 2/3/4 are **0.6845238095 / 0.6045602606 / 0.5296052632**, all below 0.75. Thus the frozen proxy identities and A≠B=B/chronology checks are undefined here. The previously visually reviewed B revisit is not an invalid model proposal merely because this annotation-localization screen excludes it.

**Interpretation:** the strict annotation-localization gate is a major *observed attrition point* among numerical near returns, including one independently reviewed physical revisit. That does not establish why the accepted screen had R0: numerical nearness is not physical recurrence, the candidate source pool is fixed, and an absent or ambiguous annotation has different meaning from an invalid box. These data justify considering a separately contracted, annotation-independent visual review for support; this audit chooses no images or thresholds and makes no physical first-visit, owner-novelty, population-rate or mechanism claim. The accepted D2 screen and its novelty HOLD remain unchanged.

## Provenance and resource closure

| Artifact | SHA-256 |
| --- | --- |
| [Lead brief](lead-attrition-brief-v1.md) | `d7e0754bc05caee29b732dfec756e40dbcc533e522d03d638df0e2aa0b2f98fd` |
| [Lead CPU acceptance](../lead-cpu-acceptance-v1.json) | `68d23d266693abff9727706b2263942a4f4c080e3c4e165d54e80553606b503f` |
| [Source freeze](source-screen-freeze-v1.json) | `d17da9af0e727449e8f65f0390057bc34eb1d5ea8ae49688f05e084184beaae9` |
| [Frozen screen registry](cpu-candidate-registry.json) | `e79234c7360427508d9846b6290087231ca1ea05d57f12e6246f14d3e054652d` |
| [Frozen screen helper](../../../../probes/training_set_completion/recurrence_grounded_window_transfer/screen.py) | `cb135f584db3309b594ec041cb63132e932727b74e6fc8e88d369cc48ea7d1e8` |
| [Audit helper](../../../../probes/training_set_completion/recurrence_grounded_window_transfer/attrition.py) | `2162a843367f97f12e9c877fcd5703331e4baea2efc5dc404111558e10e2189a` |
| [Machine record](numerical-return-attrition-v1.json) | `6313558c0a84ca42861ed32a7afd7821d00ce347f5fea016370e99d4f7e1bac8` |
| [Independent CPU receipt](numerical-return-attrition-verify-v1.json) | `1c3e88fcf11911e6372a6c0800e1409fb9393c8b20887a390d5e0f3a1ebb31b8` |

Selfcheck exercised exact, nonexact-near, invalid-exact, nonadjacent, and multiple-earlier-partner cases. Independent reconstruction used the original source rows and maintained coordinate-bin IoU, then replayed the frozen screen read-only: **R0/D2 unchanged**. Model loads **0**, CUDA calls **0**, model forwards **0**, vision forwards **0**, generated tokens **0**, GPU hours **0**. No GPU job was launched; no further work is admitted by this candidate.
