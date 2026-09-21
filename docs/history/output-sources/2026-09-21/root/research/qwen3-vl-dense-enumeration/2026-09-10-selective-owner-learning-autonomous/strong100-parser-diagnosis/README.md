# Strong100 parser-drop diagnosis

Status: **candidate; named failure mode answered**. No model calls, candidate/code/GT edits, new labels, new arms or visual-matcher metrics.

## Conclusion

The aggregate improvement is **not solely parser censoring**: F1 gains remain on images with unchanged drop counts and on images with no drops under either model. However, the guard16 result is not clean preservation. Image59571 develops a zero-width bottle burst; its lower valid-FP count partly reflects more generated spans becoming invalid. Outside drop-improved images, the F1 benefit is a precision/volume tradeoff, not an owner-recall gain.

## Exact denominators and native global scores

| Paired group | Images | Source TP50/FP50 | Strong100 TP50/FP50 | Source F1 -> strong F1 | Drops | Strict repeats |
|---|---:|---:|---:|---:|---:|---:|
| All development |128|614/489|618/443|.615848 -> .633197|58 ->24|112 ->96|
| Guard16 |16|56/146|54/121|.384880 -> .409091|7 ->22|44 ->38|
| Dev112 |112|558/343|564/322|.655314 -> .668246|51 ->2|68 ->58|
| Equal drop counts |122|565/318|562/299|.677864 -> .683283|1 ->1|72 ->63|
| No drops in either output |121|561/254|558/240|.707440 -> .711281|0 ->0|35 ->27|
| Candidate drop count increases |2|13/119|12/99|.158537 -> .167832|7 ->22|36 ->33|
| Candidate drop count decreases |4|36/52|44/45|.441718 -> .536585|50 ->1|4 ->0|

All128 development outputs end naturally, without caps. Valid predictions1103 ->1061; dropped complete spans58 ->24; total emitted complete spans1161 ->1085. Aggregate improvement therefore does not arise from generating more malformed output overall.

The four drop-improved images contribute TP+8; the two worsened images contribute-1; the other122 contribute-3. Net+4 owner recovery is concentrated in repaired-image cases. These groups do not prove one-to-one conversion of former invalid spans into particular recovered owners.

## Six images change drop counts

| Image | Split | Drops | Delta TP50 | Delta FP50 | Delta valid predictions | Delta repeats | Delta tokens |
|---|---|---:|---:|---:|---:|---:|---:|
|59571|guard|1 ->15|-1|-21|-22|-1|-78|
|70033|guard|6 ->7|0|+1|+1|-2|+18|
|182967|dev112|1 ->0|+1|0|+1|0|-1|
|355210|dev112|1 ->0|+1|-11|-10|0|-111|
|360071|dev112|1 ->0|+1|+1|+2|0|+9|
|39654|dev112|47 ->1|+5|+3|+8|-4|-328|

Drop growth totals15:59571 supplies14 (93.3%). Drop reductions total49:39654 supplies46 (93.9%). Image226097 retains one drop under both models; excluding all images with any drop gives121 images, rather than122 with equal counts.

## Three decisive raw checks

All58 Source and24 candidate drops are `geometry_invalid`. These are completed object spans with invalid geometry, not missing logs, a changed parser policy, or valid annotation-unmatched predictions.

- **59571:** Source has one zero-width bottle at generated order15. Strong100 has15 zero-width bottles at orders16-29 and64. Example candidate bins: `[290,407,290,438]`. Valid predictions80 ->58, but valid+invalid spans81 ->73:22 fewer valid predictions decomposes into8 fewer emitted spans and14 additional invalid spans. Canonical owner2096209 is lost; there are no gains and no assignment-only loss flags. This is count accounting, not a matched-row causal claim.
- **39654:** Source has47 invalid banana spans: order0 and34-79; two have zero width and45 have reversed x bounds. Strong100 retains only the original zero-width span at order0, `[0,0,0,99]`. Valid predictions33 ->41 while total complete spans80 ->42. It gains1493317,1494971,1495237,1495538,1543410 and loses none. This is a genuine reduction of a malformed generation sequence, not exclusion of additional candidate predictions by the parser.
- **70033:** Source has six reversed-x book spans at orders20-25; Strong100 has seven at21-27. Example bins in both: `[757,0,756,86]`. The two matched owners33101 and50493 are retained; no matching-reassignment flags.

These six Source/candidate raw texts were re-parsed with `native_record` and re-scored with the canonical global owner scorer, reproducing saved outputs. No visual-render matching was used.

## Burden sensitivity, not a replacement metric

As an explicit sensitivity, charge one additional error unit per dropped complete span:

`2TP / (2TP + FP + FN + dropped_spans)`

This does not label underlying physical objects false or alter official scores.

| Group | Source | Strong100 |
|---|---:|---:|
| All128 |.598441|.625506|
| Guard16 |.375839|.377622|
| Two drop-worsened images |.152047|.145455|

The aggregate positive direction survives. On the two drop-worsened images it reverses; guard16 becomes nearly neutral and still loses two owners. Report the local malformed-bottle failure alongside the aggregate gain. No parser repair or metric rewrite is indicated by this evidence.

## Reproducible source boundary

- Candidate root: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-selective-owner-learning-autonomous/soft-preservation-dense48-strong100/evaluation/` (`manifest.json`, `consumer.json`, `reduction.json`).
- Source root: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-05-sft256-dev128-baseline/natural-eval-v1/qwen3-vl-2b-sft256-source-dev128-natural-v1/gt_vs_pred.jsonl`.
- `reproduce.md` contains a bounded CPU-only query for all denominators, changed-image deltas, drop positions and six selected parser/scorer replays.

Scientific acceptance remains lead-owned. The six prospective positive-support candidates remain unadmitted/unlaunched.
