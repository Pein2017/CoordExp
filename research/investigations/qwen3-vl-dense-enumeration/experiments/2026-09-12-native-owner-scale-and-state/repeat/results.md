# Lane B CPU result: deployment-greedy repeat supply

Status: `lead-accepted_closed`; no model call or GPU was used.

Lead-acceptance evidence: root independently reran the census consumer and
seven tests, verified all 1,164 original-source character/prefix bounds, and
verified the 15 frozen review IDs/cards plus the exact-128-record raw128
projection. The physical audit remains a selected-15-card audit, not a
prevalence estimate or negative-fit efficacy result.

## Mechanical census

Source is the frozen 384-record Stable50 endpoint packet
(`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-11-positive-progress-matched-control/endpoint-preparation/packet.json`, SHA-256
`0255e2ec783e49421ee74461296956e20a9c3bb58a4d8024335cdd64e1dbbad0`). The
consumer re-enumerates every later valid row once against all earlier valid
rows in native pixels, class-blind, with the unchanged exclusive predicate
IoU `> 0.95`.

A's final-v2 raw128 identity is frozen at
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-12-native-owner-scale-and-state/scale/preparation/selection-v2-remainder.json`
(SHA-256
`30ec2bd91606df0bdb133ca3cfcbc285b999a4bc4b9345c288a0a54d4e4bbca5`). B
uses this only for the bounded identity projection below; it does not
nominate alternatives or expand the census population.

| quantity | count |
|---|---:|
| source / eligible / existing executed records | 384 / 384 / 384 |
| valid complete rows | 3,421 |
| invalid geometry rows (including parser geometry-invalid drops) | 788 |
| strict later-row repeat rows | 582 |
| images with at least one strict repeat | 13 |
| parser geometry-invalid drops | 788 |
| parser malformed drops | 4 |
| parser drops total | 792 |
| capped outputs (`max_new_tokens=3084`) | 4 |
| EOS outputs | 380 |
| held same-category sub-threshold overlap candidates | 953 |
| frozen visual-review units | 15 |

`admitted=582` means admitted to this census only; it is not a negative label
bank or training authorization. The 953 held candidates remain unknown pending
visual review. GT mismatch alone is not hallucination evidence.

## Bounded visual review

The full-canvas cards are in
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-12-native-owner-scale-and-state/repeat/cards/`;
the frozen manifest is `review-manifest.json` (SHA-256
`3d9800af8a772ccfd0bf65fde0224b1d5f757faced32da55c7e251863b67c951`).
The separate visual-ruling sidecar is
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-12-native-owner-scale-and-state/repeat/review-results.json`;
it does not mutate the frozen manifest. The deterministic strata cover
strict versus sub-threshold, supported versus unknown seed, and near/mid/low
sub-threshold IoU where available.

All 15 cards were inspected as full-canvas images. The sidecar records six
`same_instance`, three `distinct_instance`, five
`group_or_invalid_extent`, and one `unresolved` ruling; none is a negative
label. The GT-matched/unknown seed field is retained only as support context,
never as a substitute for looking. The clearest physical drift miss is:

- `drift-supported_seed-near-00-coco2017_train_000000477605-2.png`: the two
  `person` boxes visibly revisit the same clipped left-edge person, but score
  IoU `0.908178`; the strict `>0.95` predicate misses the later row.

The sidecar also marks `114340` (same donut, medium confidence) and `274509`
(same narrow shelf/book item, medium confidence) as same-instance drift. The
`502725` cutlery card at IoU `0.847458` is explicitly `unresolved`, not forced
into a same-instance claim. Distinct-instance and invalid/group examples are
preserved as controls; unknown-seed rows remain `unknown_or_unmatched`, not
hallucinated by fiat.

## Exact source/token lineage

`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-12-native-owner-scale-and-state/repeat/lineage.json`
contains all 582 strict later-row pairs (1,164 seed/later spans), exact
literal `raw_span_text` and character offsets, exact generated token
`token_start`/`token_end_exclusive`, per-span token hashes, and the prompt-plus-
generated prefix boundary. It reports
`complete_exact_token_positions` (1,164/1,164) using the bound local
Qwen tokenizer; no model call was made. Sidecar SHA-256 is
`80219f472e064fba1141fe71a5caf251c2153411e0b85f6c396058a934cd2bf4`.
Replay command:

`python probes/native_owner_scale/repeat.py lineage --census /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-12-native-owner-scale-and-state/repeat/census.json`

## A FINAL-v2 raw128 identity projection

A's final v2 selection source is
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-12-native-owner-scale-and-state/scale/preparation/selection-v2-remainder.json`
(SHA-256
`30ec2bd91606df0bdb133ca3cfcbc285b999a4bc4b9345c288a0a54d4e4bbca5`).
It is schema `native_owner_scale.selection.v2` with 128 unique selected
identities inside the frozen 384 universe. The existing project consumer
command was:

`python probes/native_owner_scale/repeat.py project --source /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-12-native-owner-scale-and-state/scale/preparation/selection-v2-remainder.json --census /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-12-native-owner-scale-and-state/repeat/census.json --review-manifest /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-12-native-owner-scale-and-state/repeat/review-manifest.json --output /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-12-native-owner-scale-and-state/repeat/raw128-projection.json`

The existing project consumer produced:

`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-12-native-owner-scale-and-state/repeat/raw128-projection.json`
(SHA-256
`652680be3a4b9524d067767ba576dacaa789ef982670eb2a2b025894569690ad`).
The projection is
`existing_source384_census_filtered_by_A_selection_v2`, not a new model
rollout: it filters the already-computed Stable50 per-image census by A's
128 selected IDs. Counts are 128 source/eligible/executed images, 1,886 valid
rows, 582 strict repeats across 13 images, 539 held subthreshold candidates,
783 invalid-geometry rows, 4 malformed rows, 4 caps, and 124 EOS outputs.
All 15 frozen review units/cards are reused unchanged; zero new cards or
nominations were published. Because this is an identity projection over
existing greedy traces, it does not establish fresh A rollout quality or
negative-learning efficacy.

## Replay and acceptance

- Replay packet: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-12-native-owner-scale-and-state/repeat/replay-packet.json`
- Cold consumer receipt: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-12-native-owner-scale-and-state/repeat/consumer.json`
- Visual rulings: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-12-native-owner-scale-and-state/repeat/review-results.json`
- Strict-row lineage: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-12-native-owner-scale-and-state/repeat/lineage.json`
- Raw128 projection: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-12-native-owner-scale-and-state/repeat/raw128-projection.json`
- Consumer test: `python probes/native_owner_scale/repeat.py consume --census /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-12-native-owner-scale-and-state/repeat/census.json`
- The receipt passes the real exclusive-boundary invariant: `.95` and the
  next representable value below are not repeats; the next representable value
  above is a repeat.

This lane does not rerun the empty 0/768 raw-softmax comparison, claim
negative-learning efficacy, add a threshold arm, or expand A's candidate
population.
