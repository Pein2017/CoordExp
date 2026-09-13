# Cross-scene K-carrier transfer result

Status: **lead-accepted and closed**.
All16 frozen continuations are terminal, physical GPUs6/7 are released, and no
additional panel cell, tuning, replacement, or model call is requested.

## Decision-bearing answer

The useful A-region K-only coordinate-carrier effect **did not transfer beyond
the previously positive417044 donut history in this fixed four-scene panel**.
The exact A-K intervention preserved every annotated native owner at
IoU50/60/80, but it was token-identical to native on the normal9813 and finite
missing/mixed158044 scenes, and changed the non-donut477415 chair loop without
recovering an owner, terminating, or improving its joint burden.

The same-scene wrong-owner B K direction, globally L2-matched to A per scene,
changed every suffix but yielded zero IoU50/60 gains across all four scenes.
It remained capped on both loops and lost one9813 owner only at IoU80. Thus the
417044 gate gives bounded direction specificity at a fixed K amplitude, while
the cross-scene result rejects treating that A-K handle as a robust transferable
owner-repair mechanism.

This is conditional fixed-history evidence after a common supplied opener. It
does not test autonomous admission and does not establish or refute distributed
owner state outside these four coordinate-token K carriers.

## Complete scene-wise outcomes

All counts include the common supplied native history for retention. The
consumer separately stores free-suffix scores; every reported gained owner is
also free-suffix gained, so no supplied owner receives autonomous credit.

|Scene / role|Arm|Total tokens / stop|TP50/60/80|FP50|Strict repeats|Parser drops|Owner gains/losses|
|---|---|---:|---:|---:|---:|---:|---|
|9813 normal|native/self|100 / EOS|9/8/6|2|0|0|0/0 all thresholds|
||A K-only|100 / EOS, exact native|9/8/6|2|0|0|0/0|
||B K→A norm|100 / EOS|9/8/5|2|0|0|IoU80 0/1; IoU50/60 0/0|
|158044 finite missing/mixed|native/self|246 / EOS|4/4/2|23|5|0|0/0|
||A K-only|246 / EOS, exact native|4/4/2|23|5|0|0/0|
||B K→A norm|246 / EOS|4/4/2|23|4|0|0/0|
|417044 known donut loop|native/self|3084 / cap|1/1/1|307|291|1|0/0|
||A K-only|410 / EOS|11/11/9|30|1|0|10/0 at50/60; 8/0 at80|
||B K→A norm|3084 / cap|1/1/1|307|297|1|0/0|
|477415 non-donut chair loop|native/self|3084 / cap|2/2/2|9|4|332|0/0|
||A K-only|3084 / cap|2/2/2|12|5|329|0/0|
||B K→A norm|3084 / cap|2/2/2|11|7|330|0/0|

All four self suffixes are byte-identical to their fresh native suffixes. There
are no technical-invalid or missing cells. Aggregate A-K TP50 is26 versus16
native, but the entire +10 comes from the already known417044 scene. Its FP and
repeat reductions are likewise dominated by escaping that one catastrophic
loop and must not be interpreted as population transfer.

## Direction, amplitude, and first forks

The realized A/B K L2 pairs are norm-matched within float32 tolerance:

|Scene|A K L2|B K→A realized L2|B scale|First A/B free-token fork vs native|
|---|---:|---:|---:|---|
|9813|235.124120|235.124119|182.5192|none / 23|
|158044|189.925861|189.925860|142.6360|none / 5|
|417044|157.504251|157.504251|172.3426|15 / 15|
|477415|244.546859|244.546859|51.0119|4 / 4|

Large matched K magnitude is therefore insufficient: both normal/missing A-K
branches stay exact, both477415 directions remain failed loops, and only417044
A-K is useful. Conversely B changes all four trajectories, so its failures are
not no-actuation controls. The observations support a history- and
direction-dependent routing handle more strongly than a pure amplitude
threshold; they do not identify an abstract owner-specific direction.

## Direct-photo review

The shared renderer produced four native-vs-A and four native-vs-B cards from
the cold consumer. Bounded direct review found:

-9813 A-K is visually and tokenically unchanged. B makes only small person-box
coordinate shifts; the same visible people remain represented at IoU50/60,
while one localization drops below IoU80.
-158044 A-K is unchanged. B shifts the teddy/book coordinates slightly but
retains the same owner set and the overlapping right-side book/shelf revisits;
no missing visible owner is newly established.
-417044 A-K exactly reuses the archived positive output and its prior caveats,
including one strict duplicate and a below-threshold return to prefix A. B
continues narrow left-edge donut recurrence and then full-frame donut extents.
-477415 A and B both remain capped around lower-chair recurrence. B introduces
broad near-full-width lower-image `chair` boxes crossing multiple people and
chairs, not credible single-owner extents. Neither arm gains an owner.

These cards do not make unmatched boxes hallucinations or establish exhaustive
physical recall. Renderer duplicate hints are not the registered class-blind
later-row IoU>.95 metric.

## Technical evidence and measured cost

Raw root:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-12-native-owner-scale-and-state/state/`.

- The independently frozen scale raw128 selection overlaps this exposed state
  panel on158044,417044,477415;9813 is outside raw128. The overlap and the rule
  excluding the whole scale source universe from fresh evaluation are recorded
  in the final `scale/preparation/selection-v2-remainder.json`, SHA256
  `30ec2bd91606df0bdb133ca3cfcbc285b999a4bc4b9345c288a0a54d4e4bbca5`.
  State used no scale treatment output and nominated no scale candidate.
- Immutable gate packet-v1 SHA256
  `5a1208ed226a71a2d08a18194e84c6b1f1cc5a840cc9b8bbff5a7e4746762215`;
  versioned continuation packet-v2 SHA256
  `ce26a0b3f30b59d88ed060e445ce83b2068ea7c790a54011fbad869eca5b8a82`.
  V2 binds the completed gate, receipt, consumer and old executed runner rather
  than rerunning417044 after the EOS-stop correction.
- Final cold consumer: `panel-v1/consumer.json`, SHA256
  `2e99f43a47f5f6e94eab149224a7c3303209969a497d07fffe26cd707a84ccf5`.
  Launch SHA256
  `8c5af24e72c45fed9c28ae5d324e219fbc357a19a6704c33631edc1a9cf1850f`.
  Gate and both workers exit0; outer exit files also contain0.
- Raw donor tensors are retained per case. Every A/B mask consumed exactly
  layers0..27 once, with matched key counts24/4/2/25. Each branch records the
  exact MRoPE position hash, rope-delta hash and first cached opener position.
  All four 1214-parameter inventories and version counters are unchanged.
- Three model loads,16 continuations,23,074 decode tokens,23,086 total model
  forwards and16 vision forwards. Summed process-allocated wall time is
  1731.142s =0.480873 GPU-hours. Peak CUDA allocated10,272,426,496 bytes,
  reserved10,903,093,248 bytes; peak RSS11,922,042,880 bytes. No retry, OOM,
  training, optimizer, checkpoint or parameter mutation occurred.
- Terminal receipt: `panel-v1/terminal-receipt.json`, SHA256
  `ff86af1fa7149cc1a7b0d7bb014d291ffe78f16c01c55bc6174b3eb68ec93cc2`.
  Physical review: `panel-v1/physical-review.json`, SHA256
  `2a1853906acd8fd78d18461e8f95149676a1653bed3af8c3af8e2a73f31957ef`.
  Shared-renderer manifests are bound by `panel-v1/physical-review-v1/receipt.json`.

The pre-GPU EOS naming defect was exposed with a failing fixture: native
generation reports ordinary EOS as `im_end`, not local `eos`. The remaining
packet used explicit canonicalization, while early `length` and unknown stops
still fail closed. Ten focused tests pass. The capped gate was unaffected and
remains preserved byte-for-byte.

## Remaining explanations and stop

Still live, in descending proximity to this evidence: scene/history-specific
decision geometry; distributed cache state outside coordinate-token K;
boundary/opener dependence; different stability margins around a loop; and
mask-region geometry despite within-scene matched key counts. The panel does
not justify a layer, head, amplitude, boundary, scene, or donor search to
separate them. The fixed Lane C stop is reached.
