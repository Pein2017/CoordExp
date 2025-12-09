## ADDED Requirements

- ### Requirement: Coord token and scaling utilities
- The system SHALL provide a shared module, wired through `src/common/geometry` and `src/common/schemas`, that decodes/encodes coord tokens `<|coord_N|>` (or bare ints) in the range 0–999 independent of image size, with vectorized helpers for bbox/poly/line lists and clamping of out-of-range values.
- The module SHALL convert between normalized `norm1000` coords and pixel coords given width/height, clamping to image bounds, rounding consistently, and dropping/flagging degenerate geometries (x2<=x1, y2<=y1, polygons with <3 points, lines with <2 points); it SHALL retain the raw geometry in the returned record when dropping for debugging purposes.
- The module SHALL expose an explicit `coord_mode` argument (`norm1000` or `pixel`) to prevent double-scaling and be reused by inference, visualization, and evaluation.

#### Scenario: Norm tokens scaled to pixels
- GIVEN bbox tokens `['<|coord_10|>','<|coord_20|>','<|coord_200|>','<|coord_220|>']` and width=1000, height=800
- WHEN processed in `coord_mode="norm1000"`
- THEN the module returns the pixel bbox [10, 16, 200, 176] (clamped/rounded) and marks it valid.

### Requirement: Mixed-geometry bridging
- The shared module SHALL include helpers to convert a polyline to its tight bbox and a bbox to a minimal quadrilateral segmentation so that bbox GT can be matched against poly/line predictions via IoU.
- Detection evaluator and visualization SHALL use these helpers to keep bbox–poly/line matching feasible without bespoke logic in each tool and SHALL import them via `src/common/geometry` to avoid parallel implementations.

#### Scenario: Polygon prediction matches bbox GT
- GIVEN a GT bbox and a prediction polygon covering the same region
- WHEN the evaluator uses the shared helper to derive a bbox/segmentation for IoU
- THEN the prediction can be paired and scored correctly instead of being dropped for geometry mismatch.
