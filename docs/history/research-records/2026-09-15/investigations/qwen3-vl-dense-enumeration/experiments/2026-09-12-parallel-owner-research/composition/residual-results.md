# Residual-pair admission: both fail preservation

Scientific disposition: **both fixed residuals fail annotation-relative
preservation of the actual P+A owner set**, despite correctly assigning the
supplied new B row and meeting the IoU50 burden constraints. No fitting,
backfill, alternate order, or additional model call follows. This is not a
learned-composition or parameter-interference test.

Technical disposition: exit0, exactly two new continuations, exact accepted
loaded-model identity, and exact cold native reduction. Root independently
inspected the two raw/terminal/reduced results and replayed all four lost-owner
geometry checks, then **lead-accepted** this bounded result.

The [frozen unit](residual-unit.md) owns the contrast: literal residual B at
P+A versus the existing, hash-bound actual P+A rollout. Its denominator is six
owners for351017 and nine for417044, not Stable50's sole initial person. Both
accepted P+A baselines were reused without rerunning them. Original eight-branch
artifacts and the [redundant-pair conclusion](results.md) remain unchanged.

## Exact outcome

| Image / new B | TP50, P+A → new | FP50 | F1 at50 | Gained owner | Lost actual-A owners |
|---|---:|---:|---:|---|---|
|351017 / wine glass667769|6 → 4|0 → 0|.38710 → .27586|667769|666212,667754,667794|
|417044 / donut1083042|9 → 9|2 → 2|.69231 → .69231|1083042|1079910|

IoU60 gives the same owner changes and scores. At IoU80, TP/FP/F1 changes
from3/3/.19355 to1/3/.06897 for351017, and from6/5/.46154 to3/8/.23077
for417044. No new owner is gained at80. The losses at80 are666212,667794
and1079910,1080038,1572342 respectively. These higher-threshold movements are
secondary, not the IoU50 admission predicate.

Both continuations reach native EOS, with zero strict repeats, parser drops,
invalid geometry, or other malformed output. Total raw rows/tokens are4/38
and11/110. Thus the donut result is an owner exchange hidden by unchanged
aggregate TP/FP/F1 at50, not successful incremental completion. The sole failed
admission condition in both cases is actual-A owner preservation.

### Supplied versus freely generated owners

| Image | Supplied history P+A owners | Supplied new B | Free suffix matched owners at50 |
|---|---|---|---|
|351017|466970,478719|667769 (IoU .67888)|499060|
|417044|515293,1082918|1083042 (IoU .78812)|1083295,1083599,1079494,1080038,1082111,1572342|

The351017 suffix has one prediction and10 new tokens. The417044 suffix has
eight predictions (six matched, two unmatched) and81 new tokens. B credit is
literal intervention credit, not autonomous B discovery. Both A owners and the
original P owners remain present; the losses concern formerly free successors.

## Four-owner disappearance versus localization check

CPU-only nearest-same-category projection was restricted to the four IoU50
losses. Boxes below are native image pixels in xyxy order.

| Lost owner | GT box | Previous best same-category IoU | New best same-category box / IoU |
|---|---|---:|---|
|666212, wine glass|[696,314,774,522]|.81652|[919,302,971,472] / 0|
|667754, wine glass|[715,270,768,336]|.78197|[919,302,971,472] / 0|
|667794, wine glass|[824,298,887,468]|.82069|[919,302,971,472] / 0|
|1079910, donut|[855,416,962,467]|.89284|[870,346,996,434] / .11122|

The only new wine-glass row is supplied B667769, spatially disjoint from all
three lost glass annotations. The donut's nearest row is assigned to another
owner1080038. Corresponding same-category region rows are absent from the new
outputs: these are not near-.50 localization changes or assignment-only
switches. The outcome remains **annotation-relative owner loss**, not a claim
that internal visual representations were erased.

The donut B lies left of A; glass B lies right of A. This two-image contrast
does not isolate order as a cause. Forced-history effects and sequence changes
remain possible explanations; this closed admission does not distinguish them.
No extra experiment is authorized to do so.

## Verification, cost and evidence

One unchanged fp32/SDPA unmerged Stable50 load on physical GPU0; two native
greedy RP1 continuations; two image forwards;91 model forwards/new tokens.
Measured invocation wall time14.027988s (0.00389666 allocated GPU-hours), peak
CUDA allocation9,380,184,576bytes, peak RSS12,034,215,936bytes. Allocation is not
active-kernel utilization. GPU0 was released; GPU1 was unused. Training steps0.

Fifteen focused CPU tests and exact input verification passed before root's
grant. After execution, a separate CPU invocation exactly reproduced the saved
native reduction, checked exact loaded identity against the accepted first-wave
identity, and verified the exit0 receipt. No code or scientific input changed
after that validation.

Authoritative machine closeout:
[residual-closeout.json](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-12-parallel-owner-research/composition/residual-closeout.json),
SHA256 `e487cb753bb047e7fd1fd354c09a63f7f77ec4ce80a7b734227891f1c3f89477`.
It preserves the candidate-stage receipt; the lead-acceptance disposition is
recorded above. It includes full supplied/free owner partitions, score/burden
ledgers, the four-owner diagnostic, resource counts, and exact source hashes.

Raw root:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-12-parallel-owner-research/composition/`.

- `residual-preparation-v1/input.json`: frozen input SHA256
  `a16a26d28b8221ccbe1d9e4d84a16eed691229d55bd5fa9181425b9e0dc92ead`.
- `residual-acquisition-v1/rows.jsonl`: untouched literal/raw outputs, SHA256
  `f8a9faa7ec61affe0798c6e7d738b01a4aab223d1cd1c55d50735da600eff3f3`.
- `residual-acquisition-v1/reduction.json`: native reduction, SHA256
  `5756a82d95c76333416929e025ae0ec2ba27a0b60fb73990ab4abbb6b61fa30c`.
- `residual-acquisition-v1/loaded-model.json` and `terminal.json`: identity and
  resource/completion receipts.
- `residual-acquisition-v1.log` and `residual-acquisition-v1.exit-code`: full
  invocation evidence; [launch record](residual-launch.md) contains the command.

No architecture, GT, checkpoint, shared trainer, or protected transfer panel
changed. This composition lane stops at the scoped admission result.
