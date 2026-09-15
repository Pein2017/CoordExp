# Screened teacher did not compile into natural recovery

Date:2026-09-12. Status: **lead-accepted completed negative result; closed**.
Root independently verified training/cold receipts, all385endpoint rows,
all8terminal/exits, tokens/media/parser/matching, legacy aggregates and the
primary visual failure. All8GPUs are released; no further model work or
promotion is authorized by this record.

## Primary outcome

The visually screened source remains useful candidate data. The frozen
32-update objective did **not** learn its complete behavior into natural
Stable50 output.

| Image417044 | Stable50 | screened25 |
|---|---:|---:|
| Reviewed owners atIoU50 | 3 /25 | 3 /25 |
| Annotated stratum | 1 /11 | 1 /11 |
| Supplied-source unlabeled stratum | 1 /1 | 1 /1 |
| Teacher free-discovery unlabeled stratum | 1 /13 | 1 /13 |
| Annotated TP50 /60 /80 | 1 /1 /1 | 1 /1 /1 |
| Tokens /stop | 3084 /cap | 3084 /cap |
| Strict later-row repeats, IoU >.95 | 291 | 278 |
| Parser drops | 1 | 2 |

Both recover P0/P1/P2: zero owner gains or losses. P2 already appears later
in the original native loop; the13-row “free-discovery” stratum records
teacher provenance, not13novel owners relative to baseline. Fresh natural
Stable50 tokens exactly reproduce its retained baseline.

Worker and root personally viewed the primary comparison. Repeated left-edge
boxes and huge multi-owner/background rectangles remain. In the **new
endpoint's generated order**, rows6and7 are identical `[0,178,70,215]`;
row87labels `[0,0,1151,863]` as one donut. These are not original teacher
P-row identifiers. The physical gate fails, independently of failed25/25
coverage and termination.

## Retention regressed

| Disjoint partition | Images | TP50 baseline→trained | TP60 | TP80 |
|---|---:|---:|---:|---:|
| Trained image | 1 | 1→1 | 1→1 | 1→1 |
| Normal bank | 56 | 416→412 | 382→382 | 268→268 |
| Other exposed images | 327 | 1482→1462 | 1404→1389 | 1085→1078 |
| Union | 384 | 1899→1875 | 1787→1772 | 1354→1347 |

AtIoU50, normal56 gains3/loses7; other327 gains24/loses44. Union F1 decreases
0.606032→0.582660; valid predictions3421→3590, strict repeats582→689,
parser drops792→1069, caps4→5. Other327 geometry-invalid drops787→1062.
Normal KL and margin protection did not guarantee preservation for this
particular positive objective. OriginalGT and all match/denominator rules
are unchanged. Fresh transfer256 was not consumed.

## Teacher fitting and free-path mismatch are separate observations

Teacher-forced summed NLL decreases285.7570→204.9768; target argmax tokens
226/249→237/249. Twelve wrong target-argmax tokens remain across ten rows.
Under **exact cleaned teacher histories**, P2 remains9/10 and P6 remains8/10,
with minimum margins−0.1102and−2.0549. P6NLL14.1225→10.0216 is progress,
not a solved complete row. Saved records contain aggregate counts/margins
only: which two P6tokens are wrong, including next-owner entry, is unavailable.

Separately, natural-vs-cleaned first token mismatch occurs at zero-based27:
third-row y2 target bin312 /ID151982 versus actual bin313 /ID151983. Both
rasterize to pixel270, so pixel geometry agrees while text/KV history differs.
The next free row misses the intended teacher P6owner.

This **does not establish that the one-bin difference caused failure**.
P6is not fully argmax-correct even on the exact cleaned history. Incomplete
conditional fitting and deviation from trained histories remain unseparated.
No rescue inference or new probe was run. The result does not prove CE
impossibility, invalidate the screened data, or show that more updates/EOS
supervision/another objective would work. Those alternatives are untested,
outside this reached stop rule.

## Execution and cost

Single driver08:09:38–09:18:11UTC; all six phases exit0. Fixed32-update
two-rank fit ended08:59:17; cold25scores ended08:59:49 with all deltas exactly0;
all8endpoint shards ended09:17:59, then reduction and visualization completed.

- Training model/image forwards1802/1752;1696backwards and32synchronized
  updates perrank. Peak allocated11,837,361,664bytes, reserved13,748,928,512,
  RSS12,661,207,040; maximum rank2961.18seconds.
- Endpoint385natural calls,9model loads,47,328model forwards,385image forwards.
  Peak allocated10,261,089,280bytes, reserved11,018,436,608,
  RSS13,383,979,008; slowest shard1073.62seconds. All frozen guards pass.
- Process-allocated GPU time: training1.64508h, cold0.006825h,
  endpoint1.12192h; total2.77382GPUh. Not kernel-active or reservation time.
- No dose/label/geometry/description change, retry, adaptive checkpoint,
  supplied detection prefix, KV intervention, output filter or GT mutation.

## Source-bound evidence

Raw root:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-12-parallel-owner-research/data-flywheel/`.

- `training-v1/receipt.json`: SHA256
  `fcf9b67cf1f9f5a839e4d0140b68a0c0a772e8ce516f115808c599f4282db4e5`.
- `training-v1/cold-check.json`: SHA256
  `0b7a8905cbf73677db898fe11e3e057cc7e21d1b708887840cd35c11051e335a`.
- `endpoint-v1/result.json`: SHA256
  `bba7936d70e82168cf1d0df360ee94febb74932f104232f3c0dc5235c75e2742`.
- `closeout.json`: binds inputs, receipts, disjoint union aggregates, actual
  viewed card, physical failure and the exact teacher/path limitations.
  The immutable reducer retains its earlier visual-pending state; closeout
  adds the later root-confirmed failure.
- Actual viewed image:
  `endpoint-v1/visuals/0000_coco2017_train_000000417044_prediction_comparison.png`.
- Full phase logs/exits, `driver-v1.log`, and all eight shard raw/model/terminal
  artifacts are retained. Preparation-v1 was superseded before execution;
  accepted preparation-v2was executed exactly once.

The unit is closed at its frozen negative endpoint. No promotion or automatic
extension; useful candidate-data validity and this failed training microproof
are separate conclusions.
