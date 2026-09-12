# Dynamics: heterogeneous coordinate-branch escape and return

Status: **candidate; finite execution and fresh CPU consumer verification complete; awaiting root scientific acceptance.** Slice mechanics were independently lead-accepted before the remaining-panel grant. No checkpoint, training method, or universal mechanism is promoted.

## Decision-bearing result

One-token x1 interventions separate three behaviors: short apparent escape followed by a loop, full-horizon conditional escape, and state-dependent return to an already-covered row. Early nonrepeat alone is not an adequate outcome. High-probability alternatives and geometric neighbors are different interventions, although their selected token sets sometimes overlap.

## Repeat-boundary outcomes

The first row contains forced x1 and earns **zero autonomous owner credit**. The autonomous suffix starts only after that complete row; all supplied history is also excluded. Repeats count any earlier valid supplied/mixed/free row once under class-blind native-pixel IoU>.95. Geometry-invalid rows and other parser drops remain explicit.

| Image | x1 bins | First mixed row repeats? | Autonomous strict repeats | Geometry-invalid rows | Stop | Autonomous TP50/60/80 | TP50 gains/losses vs self |
|---|---|---|---|---|---|---|---|
| 351017 | 0 | true | 135 | 161 | length | 0/0/0 | 0/0 |
| 351017 | 1 | false | 134 | 162 | length | 0/0/0 | 0/0 |
| 351017 | 2 | false | 134 | 162 | length | 0/0/0 | 0/0 |
| 351017 | 3 | false | 134 | 162 | length | 0/0/0 | 0/0 |
| 351017 | 4 | false | 134 | 162 | length | 0/0/0 | 0/0 |
| 417044 | 0 | true | 290 | 0 | length | 0/0/0 | 0/0 |
| 417044 | 1 | false | 0 | 0 | im_end | 10/10/8 | 10/0 |
| 417044 | 2 | false | 0 | 0 | im_end | 10/10/8 | 10/0 |
| 417044 | 3 | false | 0 | 0 | im_end | 10/10/8 | 10/0 |
| 417044 | 13 | false | 0 | 0 | im_end | 10/10/8 | 10/0 |
| 417044 | 23 | false | 0 | 0 | im_end | 10/10/8 | 10/0 |
| 417044 | 30 | false | 0 | 0 | im_end | 10/10/8 | 10/0 |
| 477415 | 0 | true | 3 | 331 | length | 1/1/1 | 0/0 |
| 477415 | 1 | true | 3 | 331 | length | 1/1/1 | 0/0 |
| 477415 | 2 | true | 3 | 331 | length | 1/1/1 | 0/0 |
| 477415 | 3 | true | 3 | 331 | length | 1/1/1 | 0/0 |
| 477415 | 92 | false | 0 | 0 | im_end | 18/18/10 | 18/1 |

- **351017:** all four distinct alternatives remove first-row strict recurrence, and the next row is also nonrepeat. Nevertheless recurrence returns by the fourth posthistory complete row; all alternatives cap with134 autonomous repeats and162 invalid rows. The native self has136 posthistory repeats versus135 autonomous repeats; these denominators are not interchangeable. All capped cells retain one additional non-geometry parser drop.

- **417044:** all six alternatives (geometric1/2/3; probability13/23/30) reach native EOS with zero strict repeats, invalid rows or parser drops through their complete suffix. Each recovers10/10/8 autonomous annotated owners at IoU50/60/80 versus0/0/0. Autonomous valid output counts range33–36, so absence of strict recurrence is not complete-scene precision or visual-correctness proof.

- **477415:** geometric1/2/3 remain recurrent and cap with331 invalid rows; probability92 instead reaches EOS with zero strict repeats/invalid/drops and18/18/10 autonomous owners. The near perturbations switch the first mixed box to another earlier emitted chair box, rather than merely leaving the original box nearly unchanged. Their complete action differs at five token positions, then matches the baseline exactly after zero-based action token78.

The477415 autonomous-suffix ledger reports18 gained and1 lost TP50 owner for x1=92. The lost suffix owner is still supplied in the history: full-action descriptive matching has20TP50 versus2 baseline and no old-owner loss. That full-action number is **not** autonomous recovery credit, and the mixed row is never credited as a freely generated owner.

## Nonrepeat-boundary control

Controls are the latest saved-native nonrepeat row before each original h, not selected after these outcomes. They change both boundary and natural history, and are not an isolated history-length causal estimate.

| Image | Distinct non-native control alternatives | Native-self autonomous repeats / invalid | Alternatives reaching EOS with zero strict repeats, invalid rows and drops | Alternative autonomous TP50 range |
|---|---:|---|---:|---|
| 351017 | 4 | 136 / 161 | 0/4 | 0–0 |
| 417044 | 6 | 291 / 0 | 6/6 | 10–11 |
| 477415 | 4 | 4 / 331 | 4/4 | 18–19 |

The strongest state-dependent contrast is477415: x1=1/2/3 at the one-row-earlier nonrepeat boundary also produces an EOS suffix with zero strict repeats/invalid/drops, whereas the same coordinate values at the later repeat boundary return to covered rows and collapse. This supports boundary-dependent susceptibility, not a universal copying rule or a proved internal coverage ledger.

## What this changes—and what remains open

Observed: a legal one-token coordinate change can be sufficient for a full-horizon conditional owner-set improvement on two of these three deliberately selected loop images. The first-row intervention is not a natural-policy treatment. Nearby coordinate alternatives can also be insufficient, or merely postpone recurrence.

Supported inference: the current loop cohort is heterogeneous; a single duplicate-token probability or uniformly missing visual evidence is not a sufficient description. The control/result differences make native entry state and later routing relevant. They do not uniquely identify which attention, visual or positional mechanism causes those differences.

Strong alternatives: an IoU-threshold boundary, changed row extent/owner binding, and globally altered successor routing can all contribute. Displaced-but-valid predictions remain annotation-relative or unknown-neutral, not automatically verified new objects. All candidate probabilities/ranks are diagnostics at a fixed native x1 site, not complete-owner probability mass.

## Counts, technical verification and measured cost

- 34 unique executed cells; memberships {'geometric_neighbor': 18, 'high_probability': 18, 'native_self': 6}. Overlapping probability/geometric tokens execute once, so membership counts are not independent replications.
- All three natural anchors and six native-self rows reproduce saved Stable50 complete3084-token actions exactly. All six scoring replays reproduce native x1 as full-vocabulary argmax. First-row repeat/control admission and native EOS/cap accounting pass.
- CPU cold reparse reproduces all34 native parser outputs, mixed-row exclusions, pixel repeat ledgers and global IoU50/60/80 matching exactly. Seven focused CPU tests include strict>.95 sensitivity, native-pixel-vs-bin disagreement, invalid/malformed accounting, family dedup, control selection and a mixed-row fullTP1/autonomousTP0 counterexample.
- 37 generation calls including3 natural anchors, 6 scores, 66386 generated tokens, 66392 model forwards, 43 image forwards; 4 model loads including the reused first-slice load.
- Summed producer time 4958.86s = 1.3775 assigned GPU-hours on physical GPU4. Peak CUDA allocated/reserved 9956485632/10756292608 bytes; peak RSS 11784253440 bytes. No failures, retries, training or hidden repeated slice cells.
- Reused slice anchor/selection/two cells are copied with immutable source hashes into the completed351017 record. Resource totals include the slice exactly once.

## Evidence and stop

- [Frozen packet](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-12-parallel-owner-research/dynamics/packet.json), SHA256 `3b836b8d5685c46741fdac942e66bf8d1e419b53c6246da29ef778b5b8380d40`.
- [Authoritative reduction](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-12-parallel-owner-research/dynamics/reduction-v1.json), SHA256 `03240dbfcc37b81429ab3050f874add51d9774b60b0d912806fa8ebbb9e0a8d3`; exact owner sets, gains/losses, per-cell counts and raw-run receipts.
- [Cold native-consumer verification](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-12-parallel-owner-research/dynamics/cold-reparse-v1.json).
- [Closeout resource/provenance receipt](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-12-parallel-owner-research/dynamics/closeout-v1.json).
- Complete raw tokens, parser ledgers, prompts/grids, choices, logs and terminals remain in the four slice/full directories under that root.

Finite first wave is complete. Stop—no extra cases, x1 values, coordinate slots, temperatures, checkpoint arms or training. GPU4 is released to root. Coordinate-frame transport was not launched. A later optional design would hold content pixels/object size/grid fixed while translating image and history consistently, with a padded-baseline recurrence admission; these results do not make that second question necessary before root integrates the independent lanes.
