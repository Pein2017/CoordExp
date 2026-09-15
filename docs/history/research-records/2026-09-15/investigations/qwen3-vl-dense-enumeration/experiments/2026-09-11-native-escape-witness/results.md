# Native escape witness result

## Outcome

The fixed panel acquired **6/7 nonempty, visually useful local releases across
3/4 frozen cases** after supplying one complete candidate row under the exact
original-image native prefix `h`. This is positive evidence that a supplied
different-object row can locally release useful free continuation. It is not a
clean complete repair: none of the seven candidate suffixes is certified as a
globally duplicate-free, owner-complete enumeration, and case 502725 is an
EOS-only non-witness.

This supports retaining a positive-plus-negative learning contrast as a
candidate next decision. It does not establish that the model naturally selects
the supplied row, that all earlier native owners are preserved, that coordinate
order is repaired, or that the behavior is learnable. No training is performed
or authorized here.

## Fixed candidate observations

The forced candidate is excluded from every free count below. `repeat` is the
frozen once-per-later-valid-row class-blind IoU `> 0.95` metric. Complete
geometry-invalid rows, other complete malformed rows, and incomplete fragments
remain separate burdens.

| candidate | free tokens | valid free | repeat | geom invalid | other malformed | visual ruling |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 351017-c01 table | 327 | 33 | 0 | 0 | 0 | useful: real free wineglass and bottles; residual likely same-bottle reboxing below the strict threshold |
| 351017-c02 wine glass | 1340 | 90 | 60 | 0 | 44 | useful local table release, but heavy recurrence and malformed burden |
| 417044-c01 donut | 221 | 22 | 0 | 0 | 0 | useful: multiple distinct real donuts; suffix labels are not entirely clean |
| 417044-c02 donut | 191 | 19 | 0 | 0 | 0 | useful: multiple distinct real donuts; includes a thin partial donut and a huge overextended donut |
| 477415-c01 chair | 226 | 24 | 3 | 1 | 0 | useful: real stage woman and male speaker; some person/chair boxes group or overextend |
| 477415-c02 chair | 190 | 21 | 0 | 0 | 0 | useful: real stage woman and male speaker; not every suffix row is owner-certified |
| 502725-c01 cake | 1 | 0 | 0 | 0 | 0 | EOS-only, no free successor; not a useful-suffix witness |

Four of seven candidates have nonempty free suffixes and zero registered
repeat/geometry/malformed counters (351017-c01, both 417044 rows, and
477415-c02). This is deliberately not called four clean repairs: the visual
counterexamples show that the frozen strict metric can miss likely same-owner
reboxing, partial boxes, or overextended boxes. The empty 502725 suffix is not
counted as a vacuous zero-burden success. EOS is descriptive rather than a
positive outcome or intrinsic failure proof.

## Controls and identity

All four fresh natural anchors exactly equal their frozen Stable50 baseline
actions. All four `h_only` full actions also exactly equal those baselines, with
their prefixes byte-for-byte equal to the original native `h` and their free
tokens equal to the saved natural suffix after `h`. Every candidate action has
prefix exactly `h + c`, and its remainder is freshly generated; translated
history was never injected. The task-local batch identity is constant within
each original-image case. The seven forced rows are recorded separately and
receive no native-owner or free-suffix credit.

Literal preservation of `h` and preservation of the saved natural suffix are
mechanical control facts. They do not prove that every physical owner denoted by
the earlier native text remains visually preserved after a forced candidate.

The release is also not a clean sorted-target promotion. Generated rows can move
backward in x after `c`: for 351017-c01 the forced table starts at coordinate
277 and a later bottle starts at 276; for 477415-c01 the forced chair starts at
249 and the first free chair starts at 217. The result is therefore a
conditional local repair witness, not an order-compatible complete policy.

## Metric counterexample

In 351017-c01, fresh bottle rows at generated orders 25 and 26 have coordinate
boxes `[670,0,697,106]` and `[670,0,697,99]`. Their class-blind IoU is
`0.9339622641509434`, so neither triggers the frozen strict `> 0.95` recurrence
metric, yet the original-image overlay indicates they are likely the same
physical bottle. The bounded late-suffix overlay is preserved without changing
the metric or adding model calls.

## Technical closure and cost

- Fixed scientific panel: 15/15 generation invocations, 15 image forwards,
  26,962 generated/model-forward tokens, four accepted model loads, zero score
  replays, and zero training steps.
- Frozen selected-token upper bound: 45,619; observed tokens remained below it.
- Per-process maxima: 584.54 s, 9,957,252,608 bytes CUDA allocated,
  10,882,121,728 bytes CUDA reserved, and 11,767,713,792 bytes host RSS, all
  within the 1,500 s / 12 GiB / 16 GiB bounds.
- The preserved smoke-01 technical failure adds one model load and 10.53 s but
  zero continuations. Total physical model loads were five. It is not a
  scientific cell or null result.
- Ranks 1--3 exited zero under one joined invocation; all 15 records passed cold
  readback. No retry, replacement, backfill, extra candidate, or extra model
  call followed.

## Durable evidence

- Immutable execution packet: `packet-v3.json`, SHA256
  `de4eec6b0db7ae9d066f8746179ad024aadf34d65bf50df093d76c949f846fe3`.
- Combined raw records: `full/records.jsonl`, SHA256
  `62ffdfbd310e5ef6d8c4cf44ed80d54039c189009e7876f784f55e77ea57361e`.
- Cold technical/parity receipt: `full/analysis.json`, SHA256
  `939dd81e2e1a3b83cacf6ab85005f7cea045d8a455024001b71cec9ae0283cbb`.
- Full reducer output: `full/reduction.json`, SHA256
  `d9b189b8111cfe716b11e6918701bb0726dc7c2fd425768c9ca8365e4676cce7`.
- Joined process exits: `full/process-exits.json`, SHA256
  `908d36da7df905cdb153144c5109b0ce2e84eb27092fb5fc6f1ed3396c8bed80`.
- Root-adjudicated first-free-row overlays are under
  `smoke-02-successor-visualizations/` and
  `full/rank-{1,2,3}-successor-visualizations/`.
- The bounded late-bottle record is
  `smoke-02-successor-visualizations/late-counterexample.json`, SHA256
  `e9297dd11e28fe68f80fa7a4c0a33e850896f73443b9a6afc739500efde278bc`.
- Root lead acceptance: `lead-acceptance.json`, SHA256
  `69d2c42fbeda276cb4a5b195551980f2efc3dec1c654682732ffe93036b290bb`.

Root's receipt freshly replays and binds the source hashes, all prefix/action
partitions, merged records, reducer, runtime costs, visual adjudication, and
claim boundary.

## Stop

The frozen admission question is answered for this cohort. Stop here: no
training, dose selection, new candidate source, threshold sweep, candidate
expansion, scientific retry, or automatic continuation.

The lead receipt's contemporaneous user-pause clause was superseded after the
network switch. That correction allows parent-owned research to continue under
a new scope; it does not reopen or expand this completed fixed panel.
