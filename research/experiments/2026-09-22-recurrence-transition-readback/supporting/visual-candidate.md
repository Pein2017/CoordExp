# Visual candidate: numerical exit versus physical localization

Status: candidate; lead acceptance outstanding. CPU-only, four frozen selected
images; no model call, annotation change, or owner matching. Row IDs below are
**zero-based**, matching census/transition row indices. All geometry is norm1000.

Evidence root:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-transition-readback/visuals/`.
`visual-evidence.json` binds the selection, raw batch/image hashes, source,
representative row coordinates in norm1000 and pixels, judgments, and PNG hashes.
`review.json` preserves the explicit crop/row selections and visual judgments.
Each case has an untouched full-image PNG and local overlays with an unmarked
crop alongside each individually marked row. Enlargements use nearest-neighbor;
they add no image detail. This is bounded visual interpretation, not new GT.

| Image | Immediate event | Visual reading | Later evidence |
|---|---|---|---|
| train:269858 | Exact r16–19; r20 x1 350→348; r21 returns | Only a ~2.5 px extent change in the same crowded platform patch. Unique owner HOLD. | r22 “person” lands on a bench. r23 credibly localizes another bench; r26–28 clearly localize three new individual people on the opposite platform. |
| train:196924 | Exact r19–20, then EOS | Same reflective tableware region as r18; bowl category and precise owner HOLD. | No later row and no demonstrated recovery. |
| train:351017 | Exact r58–341, then cap | [0,0,0,0] is invalid zero-area geometry; preceding r57 also has zero width. No imaginary physical owner is assigned. | Cap is not recovery. |
| val:7511 | Exact r27–88; r89 expands to [0,571,999,999] | Repeated ~43.8×3.5 px strip is water, despite formally valid geometry. First exit spans many owners/background; no individual is localized. | r93 clearly localizes the kite; r94–95 localize two separate people. Those people were incidentally contained in the huge r89–92 boxes, so first-ever coverage remains HOLD. |

The strongest supported distinction is **first changed numerical row versus
later credible physical localization**. The first numerical exit in neither
train:269858 nor val:7511 establishes immediate individual-owner progress.
Both do subsequently emit credible new localized objects. The long repetitions
are heterogeneous: a crowded plausible-person patch, ambiguous reflective
tableware, invalid degenerate geometry, and geometrically valid water-only
false persons. These images do not identify a shared circuit, hidden accumulator,
cause of escape, or population-level recovery frequency.

Decisive figures for lead inspection:

- `train-269858-panel-0.png`: repeat, first changed row, return.
- `train-269858-panel-1.png`: mislabeled bench and later actual bench.
- `train-269858-panel-2.png`: three clear later individuals.
- `val-7511-panel-0.png`: water-only repetition.
- `val-7511-panel-1.png`: huge first-exit box and later kite.
- `train-351017-panel-0.png`: invalid line and point geometry.

Reproduce from the research-probes checkout:

```bash
python -m probes.training_set_completion.recurrence_transition_visuals --self-check
python -m probes.training_set_completion.recurrence_transition_visuals \
  --selection /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-transition-readback/dynamics/selection.json \
  --review /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-transition-readback/visuals/review.json \
  --out /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-recurrence-transition-readback/visuals
```

The geometry self-check distinguishes /1000 from /999 on non-square images.
The production rendering command verifies source image/raw hashes and raw batch
image ID plus split-bearing row ID. The first attempt exposed that mature raw
rows lack a separate `split` field; verification now uses their existing exact
`coco2017_{split}_{image_id}` row IDs for both cohorts. Rerun succeeded for all
four cases. No source from outputs was executed. Existing scored-detection
renderers require a different artifact contract; this small consumer reuses the
maintained saved-row parser and installed Pillow without fabricating scored GT.
