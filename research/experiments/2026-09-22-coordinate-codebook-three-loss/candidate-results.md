# Candidate: three-loss epoch16 comparison

Status: candidate, lead acceptance pending. Fixed16-epoch fitting and all planned native cells are complete; all17 GPU producers exited0 and were joined. No successor or additional model work is launched. Authority is unit.md plus lead-ruling-01.md; epoch32 is not the matched control.

The restored three-loss recipe produces nearly the same training fit as CE-only at the matched16-epoch dose and still fails the prospective bad-output and annotation-owner recurrence limits. This does not establish that missing auxiliary losses caused the earlier failures, nor isolate either auxiliary or the codebook component. Validation coverage/CE is descriptive and is not the reason for the guardrail failure.

| Panel/metric | Matched source | CE-only16 | Three-loss16 |
|---|---:|---:|---:|
| Training IoU50 matches /9519 |5580|6505|6512|
| Training IoU80 matches /9519 |3515|4638|4635|
| Training clean images /1024 |320|452|457|
| Training plain token-weighted teacher CE |1.666406|1.173919|1.178286|
| Validation IoU50 matches /2033 |1224|1235|1242|
| Validation IoU80 matches /2033 |800|744|750|
| Validation clean images /256 |104|97|93|
| Validation plain token-weighted teacher CE |1.654581|1.838695|1.843367|

Teacher denominators are90934 training and19541 validation in all three conditions. Training-versus-CE16 IoU50 match counts improve on184 images, worsen on174 and tie on666. The corresponding known-annotation coverage sets gain490 and lose483 owners; these are annotation proxies, not new physical adjudication. Validation gains111 and loses104 known owners. Full per-image identities, both IoU thresholds, strata and target-count bands are retained.

Retained32: source/CE16/three-loss16 clean4/9/9, IoU50=264/337/322 of679, IoU80=158/196/190. Additions992: clean316/443/448, IoU50=5316/6168/6190 of8840, IoU80=3357/4442/4445. No group was dropped or replaced.

## Guardrails and redistributed failures

| New source-negative→final-positive images | Training count / limit | Validation count / limit |
|---|---:|---:|
| Bad output |57/51 — fail|21/12 — fail|
| Cap |1/10|0/2|
| Annotation-owner recurrence |73/51 — fail|18/12 — fail|
| Consecutive owner run≥5 |2/10|0/2|

Relative to CE16, training bad images repair35/persist40/new40; owner-recurrent images repair47/persist54/new58. Validation bad images repair7/persist16/new10; owner-recurrent images repair7/persist12/new15. These CE-relative counts do not replace the source-relative guardrails.

CE16→three-loss16 training invalid geometry323→325, parser drops368→356 but affected images75→80, exact-row revisits303→389 and owner-revisit images101→112 (total owner revisits192→185). Validation invalid geometry83→33 and drops92→43, but affected images23→26 and owner-revisit images19→27. Thus lower aggregate counts do not establish broadly improved behavior. Natural EOS is1022→1023/1024 training and256→256 validation; UNKNOWN is3442→3541 and923→983, respectively, and remains annotation-unmatched rather than physically false.

Among source-bad images, training dropped spans fall7946→84 and validation2635→7. Newly bad images contribute272/36 drops and32637/4366 dropped-span characters. Against CE16, previously bad training images improve368→289 drops, while40 newly bad images add67; validation previously bad92→31, with10 newly bad adding12. The saved summary includes each group's raw span burden and identity list.

Overlapping final dropped-span taxonomy: training has38 reversed-x,19 reversed-y,17 degenerate-x and5 degenerate-y affected images;15 have literal digits inside box text. Validation counts are10/4/5/3 and3 literal-digit images. These are parser/text diagnostics; one malformed span can contain several rows. They are not physical false-object counts.

Illustrations retained in `examples-v1.json`: image4555 has a reversed-x final row `(166,726,157,743)`; image37502 emits `<|coord_831|>9<|coord_999|><|coord_805|>` inside its box, after both controls had no parser drops. Retained training image14038 (original COCO val identity) has a same-annotation-owner run3→6→8 with natural EOS; final also has a parser drop. Full source/CE/final file pointers and severity are saved.

## Trend and objective evidence

On the exact96-image sentinel, source/three-loss8/three-loss16 clean counts are21/28/35; IoU50=640/724/759 and IoU80=396/464/505 of1294. Matched CE16 is37 clean,772/512 matches. Epoch8 is descriptive and did not select the endpoint; there is no CE8 evaluation claim.

All984 optimizer calls are finite/applied, with7872 global packs,16384 image presentations and1438560 supervised atoms. The first LR0 call can advance Adam state without changing parameters. Initial903 trainable hashes and frozen hashes match the prior CE-only run on every rank; the entire logged LR prefix matches. The effective losses are segment_balanced CE1, typegate0.2 with all four groups, axis0.01 margin1/999, gaussian0.

The first actual four-rank objective/gradient checks and independent known-violation/mutation tests pass; max total logit-gradient difference4.54e-9. Valid zero hinges remain valid. Mean logged weighted contributions across calls are CE1.163087, typegate0.00139723, axis0.0000120402; typegate is positive in984 calls and axis in982. These scalar contributions are not gradient-effect estimates or comparable evaluation teacher CE.

## Evidence and limitations

The new root is `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-coordinate-codebook-three-loss`. `reductions/final-v1.json` and `reductions/replay-v1.json` are byte-identical, SHA256399f0960c8c6e1ef921c43ef949f63936995f1bb968786ca51d6c473d2d4cd14. They bind exactly3936 distinct cells:2560 new and1376 reused (1280 source,96 CE16), with zero missing/mutated/HOLD cells. The source/three-loss trajectory alone has2656 cells. Old source reuse is not new replication.

`summary-v2.json` retains paired failures, common96, bands and per-image structural taxonomy; `owner-turnover-v1.json` separates identity-set turnover from image-level match-count deltas. Raw teacher CE remains separate from the composite training objective in `logging.jsonl` and `objective-telemetry-v1.json`. Source/qualification runtime boundaries and historical numerical-not-bitwise resume limits are unchanged; this run started fresh. Mature SFT includes959/1024 training and248/256 validation identities, so validation is not historically unseen. Incomplete annotations do not make UNKNOWN verified false.

The original packing receipt's stale descriptive metadata is preserved with its bound erratum; actual984-call config/schedule and executed prefix were independently accepted. Final CPU reducer amendments have separate captures and do not rebind executed model producers. See `delegation-notes.md` for observed Luna/high versus Luna/max routing, child local corrections, parent integration/metadata mistakes and the full-artifact replay lesson; no capability ranking is claimed.

Model execution cost:4757.983802080154 wall seconds (1.321662h),37007.938354730606 allocated GPU-seconds (10.279983h),17 terminal exit0 jobs. Fresh final readback/check receipts and exact changed-path bindings are listed in the candidate manifest. Exact replay commands are in `execution-notes.md`. Candidate status is not lead acceptance or promotion.
