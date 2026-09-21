# A: same-prefix downstream-value probe

Bounded CPU reduction of the sealed baseline/branch artifacts. This record reports within-image paired diagnostics; it does not claim a best-of-K or union improvement.

## Denominators
- Baseline images: 16; valid complete rows (metadata): 130
- Branch rows: 64 (expected 64); complete four-branch images: 16
- Immediate TP50 / final TP50: 212 / 362 (branch-row sums)

## Paired continuation evidence
- Equal immediate, different final unordered pairs: 16
- Strict immediate-to-final ranking reversals: 2

## Outcome diagnostics
- Action stop reasons: `{"box_end": 64}`
- Final stop reasons: `{"im_end": 64}`
- Final parse statuses: `{"accepted": 63, "accepted_with_drops": 1}`
- EOS action rows: 0; final EOS rows: 64; action cap rows: 0; final cap rows: 0
- Malformed action rows: 0; malformed final rows: 1
- Next-action support rows: 64 / 64; final rows with valid predictions: 64
- Distinct sampled action count per image: [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]; terminal valid-prediction ranges: [[6, 7], [7, 9], [6, 6], [17, 18], [14, 15], [1, 31], [2, 2], [2, 2], [6, 7], [5, 5], [6, 6], [3, 4], [8, 15], [32, 36], [2, 2], [6, 7]]

## Schema assumptions
- owner_primitives.matched_owner_ids is the producer's category-consistent IoU>=0.50 owner assignment.
- midpoint.prefix_owner_primitives is the exact frozen prefix reference; branch owner sets are full row assignments.
- incremental counts are arithmetic TP differences; gained/lost IDs are set comparisons, and all cross-image owner keys use example_id::owner_id.
- geometry repeats use valid pixel-space pred bbox pairs with strict IoU >0.95 and ignore category.
- all branch rows remain outcomes, including EOS, length-cap, and malformed parses.
