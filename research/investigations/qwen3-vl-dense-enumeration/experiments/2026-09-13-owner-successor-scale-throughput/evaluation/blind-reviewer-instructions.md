# Blind physical-owner review: self-contained reviewer instructions

You receive exactly one anonymous four-image batch. This is deliberately self-contained. Do not retrieve research history, model metrics, ground truth, source maps, other batches, or consultation notes. Do not infer the model/arm from file names. Own only the assigned batch's `decisions.jsonl`; no subagents, code changes, GPU inference, GT edits or training export.

## Work sequence

1. Read your batch `input.json`, its `review_contract` and `decision_schema`. Inspect metadata only for the current image; do not print the whole batch. Preserve image order.
2. Use `view_image` on that image's `original_image.path` with `detail=original`. Forward the returned image through the image helper so that you actually see it. One sample per call, never a collage.
3. Inspect its unique candidate overlays individually at original resolution. Each overlay contains one candidate; a literal alias shares its overlay and proposal IDs. View the unique overlays needed to ground all assignments, with special attention to small/occluded/dense candidates. No source identity is available or needed.
4. Group proposals referring to the same physical individual under one stable within-image owner ID. Treat alternative class/extent separately from entity existence. A box spanning several distinct owners is not a single atomic owner; classify coherent dense group coverage separately, or extent_mismatch/unresolved if appropriate.
5. Write one complete image JSON object as a single line to `decisions.jsonl` using apply_patch BEFORE viewing another image. Save only actually viewed path/SHA pairs. Hashes in the input bind the image/overlay bytes; verify them rather than inventing values. Do not fabricate view evidence.
6. After four images, use a CPU-only JSON check for four unique image/review IDs and exact, once-only assignment of every proposal ID. Literal alias groups must not be split. Do not load any sealed source map or invoke the final comparison. Return counts/path and unresolved limitations, then stop.

## Scope and physical rules

COCO80 categories only (spelling aliases in input are already canonicalized): person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush.

- Presence is class-agnostic within this scope, with class/extent caveats retained; it is not strict IoU TP.
- Clearly separable instances are atomic. Dense books/fruit/vegetables/cups may be coherent group coverage where atomic extent is not reliably separable. Do not count the group plus its children as additional owners.
- Uncertain entity, class or extent stays unresolved/qualified. Do not default a tiny box to hallucination; equally, do not assume every image-grounded region is one valid instance.
- Missing GT never implies absence or negativity. You are not given GT and must not look it up.
- A review is limited to these proposals; do not claim exhaustive image annotation.

## Required per-image JSON structure

```json
{"image_id":123,"review_id":"exact input review_id","reviewer":"your agent name","saved_before_next_view_attestation":"reviewer_attests_decision_saved_before_next_view","viewed":[{"path":"actual viewed path","sha256":"verified hash","detail":"original"}],"owners":[{"owner_id":"O01","proposal_ids":["exact IDs"],"extent_or_class_caveats":[]}],"group_coverage":[],"unresolved":[],"non_owner_evidence":[]}
```

`group_coverage` entries: `{group_id, proposal_ids, extent_or_class_caveats}`.
`unresolved` entries: `{proposal_ids, axes:["entity"|"class"|"extent"], image_grounded_reason}`.
Optional `non_owner_evidence` entries: `{proposal_ids, status:"unsupported"|"extent_mismatch", image_grounded_reason}`. These describe visible evidence, not automatic training-negative labels.

Assign every proposal ID exactly once. Use concrete image-grounded reasons for unresolved or non-owner evidence. Do not add hidden-model predictions or change the frozen schema. Timing attestation is honest workflow testimony, not a substitute for actual image viewing.
