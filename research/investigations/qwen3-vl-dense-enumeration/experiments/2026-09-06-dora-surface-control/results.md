# Completed DoRA surface and fixed-dose controls

Scientific disposition: neither magnitude-only course improves the registered
terminal dev128 outcome over Source. The exact-recipe lr1e-5 surface control is
near-null; lr0.003 fits the training panel strongly while development deteriorates.
Technical disposition: both256-step courses, checkpoint integrity, all six
natural reads and their complete aggregates are lead-accepted. No promotion.

| Panel / checkpoint | Source IoU50/60/80 owners | Old full DoRA, lr1e-5 | Magnitude-only, lr1e-5 | Magnitude-only, lr0.003 |
|---|---|---|---|---|
| Dev128 / step64 | 614 / 585 / 451 | 609 / 578 / 444 | 615 / 585 / 449 | 574 / 542 / 400 |
| Dev128 / step256 | 614 / 585 / 451 | 596 / 563 / 418 | 612 / 585 / 451 | 522 / 491 / 367 |
| Fixed train256 / step256 | 1259 / 1190 / 908 | 1394 / 1317 / 1046 | 1256 / 1191 / 911 | 1688 / 1639 / 1507 |

Denominators remain128 images/891 owners on development and256/1955 on training.
The lr1e-5 terminal dev difference from Source is-2/0/0 owners, with7 IoU50
gains and9 losses. It retains more owners than matched full DoRA (+16/+22/+33),
but also lacks its training gains (-138/-126/-135). Retaining near-Source behavior
without fitting is not evidence that the surface preserves learning transfer.

The lr0.003 terminal training difference from Source is+429/+449/+599 owners;
development is-92/-94/-84, with24 IoU50 gains and116 losses. IoU50 development
losses by density1-3/4-7/8-15/16+ are6/12/39/35 owners. Among498 common matched
dev owners, mean IoU changes-0.02301; among1214 common train owners it changes
+0.09574. Thus magnitude-only has substantial fitting capacity at this dose,
but does not avoid the observed train/development divergence. This does not
identify a unique mechanism or prove a universal surface limitation. The
lr0.003 arm jointly changes surface and numerical dose versus full lr1e-5;
it is not a pure surface comparison or a tuned optimum.

All dev decodes stop naturally in both profiles. Small-lr terminal training
retains4 length caps; lr0.003 training has none. High-lr terminal dev has one
all-spans-dropped behavioral output (image224468, a single coordinate followed
by prose), which remains in the128-image/891-owner denominator with zero
predictions. This is not the repaired scoring exception. All six reads have
zero score failures. High-lr terminal train has one evaluator-invalid prediction.

At terminal dev, comparison-retained predictions are1120 (small LR) and861
(large LR), versus1103 Source; strict physical duplicate candidates are12/0
versus9. Unmatched counts are508/339 versus489 and remain unknown, not false
positives or hallucinations. Small/high-lr dropped counts are8/40 versus58.
Fewer large-lr predictions and duplicates do not establish improved precision.

Both arms preserve all392 Source A/B tensors and change196 magnitudes; selected
embeddings remain byte-identical to Source. Native training retains world8,
EBS64, cosine256, seed19 and the16/64/128/256 save/forward-eval cadence. Natural
reads remain unmerged FP32/SDPA, world8/B2/max3084/greedy. This is single-seed,
historically used development evidence, not untouched-test generalization.
Both registered doses are closed; no extra search or checkpoint selection.

## Evidence and recovery

- Small-lr aggregate: /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-06-ce-controls-rloo-successor/native-ce/analysis-magnitude_small_lr-v1/aggregate.json
  SHA-256675c25c9e81bfc25a6970bab75ad77e8f9f17876cdadb1302eb6048ccbe94096.
- Prior-lr aggregate: /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-06-ce-controls-rloo-successor/native-ce/analysis-magnitude_prior_lr-v2/aggregate.json
  SHA-256a143ea90f1d941f2e7332c31d7d08ba26e1b9a879dc1d42b9e940a6c595e42a8.
- The lead revalidated all six run/config/adapter/embedding/cohort identities
  and every native artifact hash against the aggregates after completion.
- The high-lr failed dev v1 was preserved. Recovery reran only dev256 in a new
  v2 root, then the never-executed train256 v1; completed dev64 was reused.
  See [parser recovery (preserved from archive ref `archive/research-restructure-20260909/dora-prox-linear-n2` at `ba801de51`)](/data/CoordExp/outputs/research/restructure-research-probe-development/preservation-20260909/historical-links/dora-prox-linear-n2/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-06-dora-surface-control/parser-recovery-v2.md). On recovered image460339/span1,
  replaying its actual native trace with old lexical evidence reproduces
  selected11; role-bound evidence selects8 with score0.33112907584011214.
- Full per-threshold, density, gain/loss, geometry and decode diagnostics are
  retained in the aggregates. Runtime success alone did not determine closure.
