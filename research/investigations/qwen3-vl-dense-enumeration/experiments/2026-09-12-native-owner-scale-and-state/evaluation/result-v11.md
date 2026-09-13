# Frozen candidate evaluation v11

Status: lead-accepted, closed, including bounded physical review. No promotion.
See [parent results](../results.md) and the authoritative iteration receipt for
the final scientific disposition. The single regranted invocation completed on GPUs0–7,
then lossless merge and a cold consumer each exited0. No baseline rerun or extra
target-image inference was performed.

Output root:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-12-native-owner-scale-and-state/evaluation/`.
`completion-v11.json` is the compact identity/counter/metric receipt;
`candidate-natural-v11-consumer/result.json` retains full per-image evidence.

- Exact640 rows: train11, reference54, other exposed319, fresh256.
- 8 model loads, 640 image forwards, 45330 model forwards/new tokens;
  640 EOS stops, no capped outputs. Maximum worker elapsed569.23 seconds.
- Reviewed-c target geometric coverage: Stable50 0/16 at all thresholds;
  scaled terminal10/16 at50 and60,9/16 at80. This is GT-independent frozen
  target matching, not an exhaustive physical-owner verdict.
- Fresh256 GT proxy at50: TP1029→1080, FP1857→691,
  micro-F1 0.44144→0.60897; paired owners gained90/lost39.
- Full640 strict repeats1689→177 and parser drops1505→32.

Physical review handoff:

- `candidate-review-cards-v11/blind32-manifest.json`:32 source-blind mixed
  proposal cards; source queue is `candidate-natural-v11-consumer/blind-review-queue.jsonl`.
- `candidate-review-cards-v11/admitted11-manifest.json`:11 paired natural
  endpoints with16 reviewed-c target overlays; no extra inference.
- `candidate-natural-v11-consumer/blind-review-source-map.json` is separate;
  withhold it from the source-blind reviewer. Exact overlapping proposals can
  obscure on-image ordinal labels; full IDs and literal boxes remain in the
  queue and all IDs are listed in the card legend/manifest.

The renderer validated43 PNG files and exact card/target denominators; one card
was inspected for rendering legibility only. No physical labels were supplied
by the implementor. Matching is a proxy: global assignment prevents one
prediction matching two targets, but does not exclude duplicate/grouped boxes
as physical false positives. Unmatched proposals are not negative labels, and
blind32 proposal coverage is not exhaustive scene recall.

Completed review: Astra-low personally viewed all32 source-blind cards before
the separate source-map join. All419 proposals were assigned once:294 to154
physical-owner clusters,2 group-invalid and123 unresolved. Reviewed physical
presence is147 in each arm,140 retained,7 gained and7 lost; class and extent
caveats remain. Root verified bindings and join counts and spot-checked434996
and93437. This is not an equivalence result or exhaustive scene recall.
Root separately viewed all11 admitted paired cards; findings and exact natural
counters are in `candidate-review-cards-v11/root-admitted11-review.json`.
