# Global-L2 controls: useful escape is not unique to the A-region delta

Status: **lead-accepted and closed**. Root independently verified all8
stage/output hashes, exact native token/text/parser/score reductions and both
reference suffix flags. Bounded physical review is complete. All8frozen
branches complete, exact reference gates pass, no model work remains queued.

## Decision

Norm-matching the wrong-owner B delta to A's global amplitude also produces a
useful closed continuation, with11new annotated owners and0lost atIoU50/60.
Downscaling A to B's original norm restores the exact native cap trajectory.
Therefore the first panel's difference between raw A and raw B is **not
evidence that the successful direction must originate from A's region**.

Amplitude alone is also insufficient: background at A's global norm still
caps, and the seeded Gaussian at that norm generates a long malformed span.
K-only A is locally sufficient for a useful escape even though its norm is
far smaller than V-only A, which does not escape. The strongest interpretation
is a magnitude- and direction-dependent cached-key routing handle at this
fixed history, not an abstract owner-specific ledger or a universal memory
mechanism. The K/V arms are not norm-matched and do not establish unique
component necessity.

## Fixed8branch complete-output results

Same417044 image, Stable50, original19-token history plus one common natural
opener, native MRoPE and four coordinate positions in all28layers. Maximum
3064free tokens preserves the original3084total limit. Global norm is one
CPUfloat64 L2 over all K/V scalar coordinates; no per-layer renormalization.

|Arm|Realized delta L2|Total tokens/stop|TP50/60/80|FP50|Strict repeats|Parser drops|
|---|---:|---|---|---:|---:|---:|
|native/self|0|3084/cap|1/1/1|307|291|1|
|original A|2045.545|340/EOS|11/11/8|23|0|0|
|B scaled to A|2045.545|340/EOS|12/12/10|22|0|0|
|background scaled to A|2045.545|3084/cap|1/1/1|303|285|5|
|A downscaled to B|8.552227|3084/cap|1/1/1|307|291|1|
|Gaussian to A, seed20260912|2045.545|3084/cap|1/1/1|1|0|1|
|A K-only|157.504251|410/EOS|11/11/9|30|1|0|
|A V-only|2039.472160|3084/cap|1/1/1|307|296|1|

Native/self and original A exactly reproduce their entire first-panel suffixes.
A-downscaled exactly reproduces native. B-scaled and K-only are distinct free
trajectories, not copies of original A. Every gained annotated owner is in the
free suffix; the common prefix person is retained. No losses atIoU50/60/80
occur in the three useful branches.

Raw row starts are309native,34originalA,34B-scaled,309background,309A-down,
3Gaussian,41K-only,309V-only. Gaussian's two parsed predictions are precisely
the forced prefix; the3064free tokens become one unclosed malformed third
span of13,915characters. Its low repeat/drop/FP counters are not a repair.

B-scaled gains atIoU50/60:
`1079494,1079910,1080038,1082111,1082918,1083042,1083135,1083295,1083564,1083599,1572342`.
K-only gains10atIoU50/60; it gains1572342relative to original A's set but does
not recover1083295. Original A gains10as previously recorded.

## Captured amplitude and component evidence

All229,376scalar carrier coordinates are saved in
`capture/donor_slices.safetensors`; global and per-layer K/V L2,RMS,max
diagnostics were frozen before any continuation.

|Donor delta|Global L2|RMS|Max abs|K L2|V L2|
|---|---:|---:|---:|---:|---:|
|A|2045.544984|4.271054|168.768131|157.504251|2039.472160|
|B|8.552227|0.017857|0.437622|0.913902|8.503257|
|background|0.575184|0.001201|0.024994|0.100863|0.566271|

Frozen scales:B×239.1827168293; background×3556.3319821371;
A×0.004180904094. The Gaussian uses one fixed CPUfloat64 generator at
seed20260912, then one global scaling. Realized float32 norms satisfy the
predeclared5e-5relative tolerance. Zero-norm fallbacks were not used.

After global scaling, B's K norm is about218.6 and background's about358.7;
background still fails despite larger K magnitude than the successful B and
A-K-only branches. This further weakens a pure magnitude-threshold account,
without identifying a unique semantic direction. Descriptive, posthoc K-delta
cosines areA/B0.1224andA/background0.0557; they are not selection criteria or
a representational identity test.

## Earliest free forks and physical review

Zero-based free-token fork offsets versus native:
originalA5(y1:206→269), B-scaled5(y1:206→266), background5(y1:206→0),
Gaussian0(description-token transition), K-only15(next-row y1:206→226),
V-only5(y1:206→269). K-only preserves the entire first free row and then
diverges. V-only takes the same initial y1 fork as successful original A but
does not recover, so crossing that single coordinate alone is insufficient.

The originalA physical caveats remain: A is an unannotated clipped donut,
free prediction3 reboxes it below strict.95, and prediction25 is a giant
multi-object extent. These do not disappear when comparing norm controls.

The standard comparison card for B-scaled versus K-only was personally viewed:

-Both outputs reach many separate visible donuts, including B(annotation
  1083135): B-scaled prediction10 IoU0.930; K-only prediction15 IoU0.911.
-Both avoid originalA's giant full-height donut extent; this does not make
  every small prediction a correct distinct object.
-B-scaled predictions2/3 revisit the same clipped neighboring-donut region
  with different extents, below.95. Strict-repeat0 does not establish physical
  uniqueness. Prediction4 spans across adjacent clipped-donut extents.
-K-only has one registered strict duplicate: predictions5/6 IoU0.962.
  Prediction7 also returns to the initial A region with a different extent
  (IoU0.714to prefixA). It is not a perfect coverage ledger.
-Several unmatched boxes correspond to real but unannotated donuts; FP22/30
  remain annotation-relative. Geometry ambiguity and partial overlap remain.

These controls distinguish a useful escape from indiscriminate corruption,
but they do not turn region labels into abstract owner-state semantics.

## Execution, provenance and stop

Raw root:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-12-parallel-owner-research/instance-state/amplitude-control/`.

-Frozen executed packet SHA256
  `90bbb8e7d246599c0f4563e1aa47499eb54985cb2c49ee05b7fab0bb5f3d6009`.
-`run-v1/launch.json`: all8stages completed. Capture/gate ran onGPU5, followed
  by six disjoint one-branch workers on0/1/4/5/6/7. NoDDP, retries or training.
-Exact native cached slice bytes and actual MRoPE position/delta hashes match
  every worker to capture. All donor masks consumed28layers;11total vision
  forwards and0vision recomputation during decode.8model loads,16,361total
  model forwards. All stage/source/output hashes are bound by cold consumer.
-Summed stage wall time1468.81seconds(24.48GPU-stage minutes); capture12.41s,
  gate260.24s, then parallel controls63.22..268.89s. End-to-end is about9minutes.
  Peak per-worker allocated10,255,289,344bytes; RSS11,954,331,648bytes.
-Exact outer log/exit:`run-v1.log` and`run-v1.exit`(0). Per-stage logs and
  exits are inside`run-v1/`. Raw donors and frozen scales are under`capture/`.
-`run-v1/consumer.json`: all8branches cold-reloaded through the original
  native parser/global matcher; raw/dropped spans retained.
-Visual manifest:`run-v1/visuals/manifest.json`; card:
  `0000_coco2017_train_000000417044_prediction_comparison.png`.
-Physical review:`run-v1/physical-review.json`, bound to the cold consumer
  SHA256`14f4dfff7e710b1c9f10c4e7514e47129a9e3dd1d1dceec422eece9c24b2663e`.

The finite8branch contrast is finished. No seed, alpha, norm-placement, layer,
head, history, image or architecture search follows automatically. The result
should update the portfolio away from a uniquely A-bound ledger explanation
and toward a bounded key-routing/direction mechanism, while retaining that
this is one exposed conditional history and not autonomous recovery.
