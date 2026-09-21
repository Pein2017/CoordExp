# Representative Eight-Coordinate Visual Review

Source run:
`outputs/coordexp_swift/infer/val200/qwen3-vl-2b-desc-first-geo-sorted-xy-quad-clockwise-pure-ce-typegate-dora-r16a32-step5529-val200-hf-rp1p10-maxnew3084`

The renderer draws the prediction polygon that is used for COCO segmentation
mask-IoU matching. A cyan dashed outline marks the axis-aligned envelope when
the eight source coordinate bins are not a strict rectangle.

| Row | Review role | TP | FN | FP | F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| `coco2017_val_000000002299` | non-rectangular source coordinates, dense scene | 20 | 26 | 7 | 0.548 |
| `coco2017_val_000000010092` | non-rectangular source coordinates, sparse scene | 3 | 2 | 1 | 0.667 |
| `coco2017_val_000000006954` | clean high-quality sample | 7 | 0 | 0 | 1.000 |
| `coco2017_val_000000003934` | medium-quality sample | 9 | 5 | 4 | 0.667 |
| `coco2017_val_000000019109` | relatively strong dense sample | 18 | 9 | 5 | 0.720 |
| `coco2017_val_000000000632` | dense over-prediction failure | 6 | 11 | 41 | 0.187 |

## Non-rectangular predictions

Both cases are one-bin equality violations between repeated coordinate slots,
not materially free-form quadrilaterals.

1. `coco2017_val_000000002299`, prediction 2, `chair`:
   source bins are `[72, 698, 113, 697, 113, 921, 72, 921]`. The first top
   edge differs by one normalized bin and becomes a one-pixel slant. Its mask
   IoU with its own axis-aligned envelope is `0.997032`. This prediction is a
   false positive because the row has no chair GT, independent of snapping.
2. `coco2017_val_000000010092`, prediction 2, `chair`:
   source bins are `[906, 890, 998, 889, 998, 999, 906, 999]`. The same one-bin
   top-edge difference collapses to identical pixels during norm1000-to-pixel
   conversion, so its polygon/envelope mask IoU is `1.0`. It matches GT 4 at
   mask IoU `0.764656`.

Recommended benchmark treatment: preserve and evaluate the decoded polygon;
do not snap it to a rectangle. Snapping would hide an observable property of
the eight-token representation and can only inflate geometry metrics. Keep the
axis-aligned envelope as an explicitly secondary compatibility projection.
Report rectangle consistency separately: this slice has 1,405 strict source
rectangles out of 1,407 predictions, with maximum repeated-slot deviation of
one bin.
