## MODIFIED Requirements

### Requirement: Parsing and coordinate handling
- The evaluator SHALL reuse the shared coord-processing module (via `src/common/geometry`/`src/common/schemas`) used by inference/visualization, supporting coord tokens or ints in 0–999 with one geometry per object.
- Coordinates SHALL be converted according to `coord_mode`: when `norm1000`, denormalize using per-image width/height, clamp to bounds, and round; when `pixel`, only clamp/round applies to avoid double scaling. Missing width/height SHALL cause the object to be dropped and counted. If invalid/degenerate, the evaluator SHALL drop the object, increment a counter, and retain the raw geometry in per-image diagnostics.
- Polygons retain segmentation and also expose a bbox; bbox GT SHALL be given a minimal quadrilateral segmentation when segm evaluation is enabled so bbox–poly predictions can be paired; degenerate geometries are dropped and counted.

#### Scenario: Coord-token prediction is denormalized correctly
- GIVEN a prediction with `bbox_2d: ['<|coord_10|>', '<|coord_20|>', '<|coord_200|>', '<|coord_220|>']`, `coord_mode="norm1000"`, and width=1000, height=800
- WHEN parsed by the evaluator
- THEN it produces a pixel bbox [10, 16, 200, 176] (rounded) and uses that bbox in COCO artifacts.

#### Scenario: Polygon prediction pairs with bbox GT
- GIVEN a GT bbox and a prediction polygon overlapping the same region with `coord_mode="norm1000"`
- WHEN the evaluator derives a bbox/segmentation via the shared helper
- THEN the polygon prediction is eligible for IoU matching against the bbox GT instead of being discarded for geometry mismatch.

