---
doc_id: progress.diagnostics.artifacts.et_rmp_rp_sample_bank_2026_04_29
layer: progress
doc_type: artifact-bank
status: frozen
domain: stage1-et-rmp-ce
summary: Fixed representative sample bank for pre-support-mass-enhancement ET-RMP repetition-penalty visual analysis.
tags: [stage1, et-rmp-ce, repetition-penalty, visualization, sample-bank]
updated: 2026-04-29
---

# ET-RMP RP Representative Sample Bank

This directory freezes six representative `val200` examples for targeted
analysis of repetition-penalty behavior on the old ET-RMP checkpoint, before
support-mass enhancement.

Parent diagnostic note:
[../../2026-04-29_et_rmp_rp_continuation_bias_hypothesis.md](../../2026-04-29_et_rmp_rp_continuation_bias_hypothesis.md)

## Scope

- Checkpoint:
  `output_remote/stage1_2b/set_continuation/coco1024_sota1332_setcont_et_rmp_ce_v1/setcont-coco1024-sota1332-et-rmp-ce-v1/v0-20260429-022918/checkpoint-300`
- Eval slice: `val200`
- Decode: greedy, `max_new_tokens=3084`
- Compared settings: `rp=1.10`, `rp=1.15`, `rp=1.18`
- Source scratch gallery:
  `temp/rp_compare_gallery_20260429`

## Contents

- [index.json](index.json): machine-readable case index copied from the
  generated gallery.
- [research_subset.json](research_subset.json): narrowed first-pass research
  subset. Use `core_6` by default and only add `optional_2` when a second
  positive or negative probe is needed.
- `<tag>/<tag>_comparison.png`: three-way visual comparison for one sample.
- `<tag>/rp110_canonical.jsonl`, `<tag>/rp115_canonical.jsonl`,
  `<tag>/rp118_canonical.jsonl`: canonical GT/pred/matching records for
  reproducible rerendering or follow-up scoring.

## Case Index

Default first-pass review set: `core_6` in
[research_subset.json](research_subset.json). Do not manually relabel all 200
`val200` images for the first pass.

| Tag | Image idx | Source image | Reason |
|---|---:|---|---|
| `benefit_121` | 121 | `images/val2017/000000012639.jpg` | `rp=1.18` gains TP and reduces FN. |
| `benefit_010` | 10 | `images/val2017/000000001268.jpg` | `rp=1.18` recovers extra objects cleanly. |
| `benefit_158` | 158 | `images/val2017/000000016010.jpg` | `rp=1.18` opens up and finds more objects. |
| `hurt_025` | 25 | `images/val2017/000000002157.jpg` | `rp=1.18` sharply under-predicts versus `rp=1.10/1.15`. |
| `hurt_061` | 61 | `images/val2017/000000006471.jpg` | `rp=1.18` loses high-quality matches. |
| `hurt_178` | 178 | `images/val2017/000000017959.jpg` | Dense kite/crowd case; `rp=1.18` shifts toward many tiny people and misses kite coverage. |

## Review Contract

Use the shared GT-vs-Pred review semantics:

- GT panel on the left.
- Pred panel on the right.
- GT boxes are green.
- FN boxes are orange.
- matched prediction boxes are green.
- FP boxes are red.

Do not treat red FP boxes in dense crowd regions as automatically false in
subjective review. The reviewer should classify them as plausible unlabeled,
duplicate, localization miss, wrong class, or background.
