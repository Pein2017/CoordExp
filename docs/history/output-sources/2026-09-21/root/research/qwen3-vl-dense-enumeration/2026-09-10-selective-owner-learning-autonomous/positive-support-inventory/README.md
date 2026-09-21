# Positive owner-support inventory

**Status: candidate inventory, not training-data admission. Zero additional already-admitted branches were found in the named current evidence chain. No inference, training, render, GT edit or repository change was performed.**

## Evidence tiers

- Source train256: 696 annotation-relative IoU50 misses across160 images.
- K4 contains matched sampled rows for217 distinct Source-missed owners across94 images and49 categories. The other479 misses have no matched K4 witness in this bank; GT alone is not a branch target.
-134 distinct gained owners occur in89 net-improving samples across54 images;110 occur in77 owner-preserving samples across49 images.
- Strong joint tier:65 owners across32 images/28 categories, represented by43 complete sampled trajectories and101 owner-gain occurrences. Strong means preserving Source IoU50 owners without extra FP, strict repeats, drops or cap—not visually certified correctness.
- The strict same-category/same-layout/insertion-interval census admits5 CPU candidates;4 images were selected; only368:2022537 and7116:181378 received visual admission plus verified useful Source/round1 greedy continuations.
- Six additional owners have an especially direct retained entrance: the first difference from Source is inside their first gained-object row. These are8 strong sampled witnesses, not8 independent owners. All six remain unadmitted.

## Six exact Source-native entrance candidates

Every variant below has an identical-to-Source prefix up to the listed action index, an existing complete sampled row, measured Source fork logits, and an EOS-complete strong sampled suffix. None has a recorded Source greedy partial/full-row release in this chain.

| Image:owner | Category | Source -> candidate fork | Row IoU | Existing sampled support |
|---|---|---|---:|---|
| 252411:1676022 | hair drier | description_subtoken@1: `sink` -> `hair` (seed2026090602) | 0.558 | 1/4 owner hits; 1 strong witnesses |
| 465695:1612233 | bed | description_subtoken@1: `suit` -> `bed` (seed2026090601) | 0.996 | 1/4 owner hits; 1 strong witnesses |
| 496747:1614837 | dining table | y1@7: `<|coord_615|>` -> `<|coord_610|>` (seed2026090602); y1@7: `<|coord_615|>` -> `<|coord_613|>` (seed2026090603); y1@7: `<|coord_615|>` -> `<|coord_631|>` (seed2026090604) | 0.775/0.550/0.701 | 4/4 owner hits; 3 strong witnesses |
| 529411:2147505 | teddy bear | description_subtoken@1: `dog` -> `ted` (seed2026090603) | 0.932 | 1/4 owner hits; 1 strong witnesses |
| 538814:1121312 | oven | x1@4: `<|coord_302|>` -> `<|coord_263|>` (seed2026090602) | 0.869 | 1/4 owner hits; 1 strong witnesses |
| 540567:1491504 | bottle | description_subtoken@1: `cat` -> `b` (seed2026090601) | 0.649 | 2/4 owner hits; 1 strong witnesses |

### Image paths and row identities

No existing matching PNG/comparison was found in the five named artifact roots. The original images below exist and match the retained image SHA256. The inventory JSON contains exact bbox bins/pixels, complete row text/token IDs/hash, literal Source prefix IDs/hash, target token ID, full route/witness IDs, raw acquisition paths/selectors, prompt identity and measured fork margin. No render path is fabricated.

- **252411:1676022** — [original image](/data/CoordExp/public_data/coco/rescale_32_1024_bbox/images/train2017/000000252411.jpg); GT category `hair drier`. Sample boxes: seed2026090602 `[47, 361, 76, 414]`; complete-row SHA256 `81f730623e3848f2502691e646c65adeabafd1a81009aadd2a0f30cc5fec42fa`.
- **465695:1612233** — [original image](/data/CoordExp/public_data/coco/rescale_32_1024_bbox/images/train2017/000000465695.jpg); GT category `bed`. Sample boxes: seed2026090601 `[0, 2, 999, 989]`; complete-row SHA256 `d22c330ac20dac2b812aba8faa851f9f8568bfe7da78f18bddbec2658fab194a`.
- **496747:1614837** — [original image](/data/CoordExp/public_data/coco/rescale_32_1024_bbox/images/train2017/000000496747.jpg); GT category `dining table`. Sample boxes: seed2026090602 `[0, 610, 423, 801]`; complete-row SHA256 `c30971f734cfdbf488050d60d8f1e22f96ab128494e52c835b1071a829e0ab6e`; seed2026090603 `[0, 613, 443, 891]`; complete-row SHA256 `9f82273dc78ea7a3333d6c1a1605304f48c8aa8dc75456f57b807f84615d1df5`; seed2026090604 `[0, 631, 496, 751]`; complete-row SHA256 `3141f0bda8a3b0f2b0f7b4855c4950b2227fdbd62110f75fe3bd5a85384a19e0`.
- **529411:2147505** — [original image](/data/CoordExp/public_data/coco/rescale_32_1024_bbox/images/train2017/000000529411.jpg); GT category `teddy bear`. Sample boxes: seed2026090603 `[16, 502, 327, 861]`; complete-row SHA256 `26ea7ddfbb22ec4961a8ce1871ae1b8cb00c4531ea8060759a31a2104da9169e`.
- **538814:1121312** — [original image](/data/CoordExp/public_data/coco/rescale_32_1024_bbox/images/train2017/000000538814.jpg); GT category `oven`. Sample boxes: seed2026090602 `[263, 0, 798, 923]`; complete-row SHA256 `6b3d59d83bea9149bf82b7181b2d4fe8815ce4d730fe7b183ab5ea572bea519c`.
- **540567:1491504** — [original image](/data/CoordExp/public_data/coco/rescale_32_1024_bbox/images/train2017/000000540567.jpg); GT category `bottle`. Sample boxes: seed2026090601 `[313, 240, 357, 302]`; complete-row SHA256 `f96f432e3466df691805adb665bdba5a555316094b2aaff2aaf48a7f4a353a09`.

Common exact witness/token source:

- [Fixed-witness inputs](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-fixed-witness-route-access/inputs.json) — `witnesses[witness_id]`, `routes[route_id].ids`, and `groups[example_id].prompt_token_ids`.
- [Measured fixed-route readout](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-fixed-witness-route-access/reduction.json) — `comparisons[witness_id].first_forks.source_greedy.source`.
- [Source GT/prediction rows](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-05-sft256-dev128-baseline/natural-eval-v1/qwen3-vl-2b-sft256-source-train256-natural-v1/gt_vs_pred.jsonl) — select the same `row_id`.

## Admission work still missing

- All six need visual entity/category/extent and prior-binding adjudication; none is declared physically correct here.
- Hair drier and bottle have zero Source any-category box overlap; teddy bear has best Source overlap0.022. These are potentially useful access cases, but category subtokens such as `b` or `ted` do not uniquely specify the intended full row.
- Bed has Source other-category overlap0.903: distinguish missing-owner recovery from category/extent correction before counting it as a new-instance treatment.
- Dining table and oven already have same-category Source overlap0.396 and0.377. Their coordinate forks may primarily repair geometry. The three table y1 alternatives must not be silently collapsed to one uniquely correct token.
- All six are first-object rows, unlike the existing mid-history person/boat entrances. They broaden positive categories and coordinate fields, but do not yet establish reliable remaining-owner transitions.
- The retained complete sampled suffix is an existence witness. Greedy force-and-release usefulness and prospective positive/reference-mask admission remain unexecuted.

## Other strict-insertion candidates are not admitted extras

-72583:1550401 (apple): selected and visually omitted for ambiguous overlapping painted-fruit instance/extent and prior bindings.
-575303:278521 (cow): selected in numeric-v2 and visually omitted because prior Source localization already substantially addresses the same cow.
-575303:1820447 (cow): CPU eligible but unselected/unreviewed; previous no-backfill rule supplied no release evidence. This is not the same owner as the omitted numeric-v2 cow.

Global matching caveat: among all217 sampled gains,3 owners already have Source same-category IoU>=0.5 support; among the65 strong owners this count is zero. This still does not certify physical novelty:12/217 and2/65 respectively have Source any-category IoU>=0.5 overlap.

`inventory.json` owns the full217-owner index and exact six-candidate token/path records. This inventory may remain unused; the lead owns any prospective visual admission or next training question.
