# Repeat multiplicity: lead-accepted result

## Outcome

On the two visually admitted cases (`9813`, `417044`), reweighting the older
A/B block consistently changed the exact-row A-versus-B score, but in the
direction opposite the registered positive-count monotonic account: **more old
B rows made exact A relatively more likely than exact B**. Older order and
content remain coupled despite reversal pairing. No free continuation ever
returned to A or B, so this panel does not establish which already-covered
instance is selected again during free generation.

This is bounded evidence of synthetic-history reweighting. It is not a pure
multiplicity mechanism, a natural-policy result, or evidence of a physical
owner ledger. Root accepted this exact bounded claim and the finite-panel stop.

## Frozen contrast and admission

Each conditional history was `[A,B] + V + [A,B,C]`, with `V` one of paired
reversals `AAAB/BAAA`, `AABB/BBAA`, or `ABBB/BBBA`. Within each case, all six
prefixes had the same literal unique-row set, token count, most-recent A and B
positions, and final C row. A/B serialized lengths were equal (nine tokens on
`9813`, ten on `417044`). Every cell scored exact full rows A/B/C plus EOS with
native causal replay, then generated at most 512 free greedy tokens.

Root admitted the two person instances in `9813` and two donut instances in
`417044`. `158044` stayed HOLD because its boxes denote book groups/stacks;
`502725` stayed HOLD because A covers an overlapping knife/utensil group. There
was no backfill, and artificial prefix rows received zero natural owner credit.

## Exact-row scores

Reversal-pair means of summed `log P(A row) - log P(B row)`, ordered as
A-heavy `5:3`, balanced `4:4`, B-heavy `3:5`:

| case | A-heavy | balanced | B-heavy | direct A-count direction |
|---:|---:|---:|---:|---|
| 9813 | 1.8522 | 2.6493 | 4.0052 | falsified |
| 417044 | -2.7179 | -1.9937 | -1.3737 | falsified |

Both sequences increase monotonically as old B multiplicity increases. The
absolute forward-versus-reverse gaps are `0.043/0.344/0.426` on `9813` and
`0.045/0.083/0.065` on `417044`; thus the registered direct-count direction is
not rescued by either reversal mate. Summed likelihood and per-token mean are
stored separately. Their A/B ordering agrees only because A and B were forced
to equal serialized length.

**Observation:** exact A-versus-B row preference is sensitive to the older
synthetic A/B block despite an identical recent suffix.

**Supported inference:** older output history conditionally reweights these
two fixed exact rows.

**Strong alternative:** contextual inverse-frequency avoidance, or another
interaction with older order/content, rather than an independent multiplicity
counter. Paired reversal and a common suffix reduce two obvious confounds but
do not identify multiplicity against every function of older history.

## Free behavior

Across all 12 conditional cells:

- `12/12` reached native EOS; `0/12` reached the 512-token horizon.
- There were 124 complete free rows, zero geometry-invalid complete rows, zero
  parser drops, and zero class-blind strict repeats under native projected
  pixel IoU `>0.95` against prefix or earlier free rows.
- `0/12` first free rows, and zero later free rows, matched A or B either
  exactly or by same-description native-pixel IoU `>0.95`.

The figures show the fixed A/B boxes and the first free row (magenta):

- [9813 first-free panel](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-11-repeat-multiplicity/free-row-overlays/case-9813-first-free.png)
- [417044 first-free panel](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-11-repeat-multiplicity/free-row-overlays/case-417044-first-free.png)

The free outputs often move to other visible people or donuts. This separates
the exact-row score shift from an actual A/B recurrence event. Fixed-row
likelihood is not mass over an IoU neighborhood, and no candidate-score change
is promoted to free owner selection.

The extremely thin first row in `9813/a3_b5_rev` remains visually uncertain
and is not assigned an owner label. More generally, zero geometry-invalid rows
does not mean every valid-coordinate box is visually correct.

## Technical validity and cost

Both fresh natural anchors exactly reproduced their saved Stable50 token IDs.
All 12 prefixes, 48 score replays, raw continuations, native parses, resource
counters, and cold readback records passed. The post-review primary repeat and
candidate incidence uses the existing native coord-bin-to-pixel projection;
coord-bin IoU is retained only as an explicitly auxiliary sensitivity field.

Actual execution used 14 generation calls, 48 score replays, 4,415 generated
tokens, 4,463 model forwards, 62 image forwards, and two model loads. Summed
worker wall time was 388.88 seconds (`0.1080` assigned GPU-hours). Maximum
per-worker resources were 9,944,960,512 CUDA allocated bytes,
10,758,389,760 CUDA reserved bytes, and 11,701,538,816 RSS bytes, below the
12-GiB CUDA and 16-GiB RSS gates. There was no training or retry.

## Evidence owner and stop

The authoritative candidate reduction is
[`reduction.json`](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-11-repeat-multiplicity/reduction.json),
SHA256 `58de6fee48209b48ed3e572c324287b7d77d5f1854861b3dae58d2881ccd0d0a`.
It binds packet-v2, both raw JSONL records, terminals, cold readbacks, model and
batch receipts, and the two overlay hashes.

Lead acceptance is recorded at
[`lead-acceptance.json`](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-11-repeat-multiplicity/lead-acceptance.json),
SHA256 `b51a2b17756b5004c581f339b23c8340a3ce40de33dc37650460f20ecc6be24d`.

The frozen finite panel is complete. Stop here: no image swap, donor sweep,
extra cohort or regime, training, special-token/KV patch, circuit claim, or
novelty claim is authorized.
