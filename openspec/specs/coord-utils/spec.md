# coord-utils Specification

## Purpose
TBD - created by archiving change refactor-inference-viz-decouple. Update Purpose after archive.
## Requirements
### Requirement: Mixed-geometry bridging
- The shared module SHALL include helpers to convert a polygon to its tight bbox and a bbox to a minimal quadrilateral segmentation so that bbox GT can be matched against polygon predictions via IoU.
- Detection evaluator and visualization SHALL use these helpers to keep bbox–poly matching feasible without bespoke logic in each tool and SHALL import them via `src/common/geometry` to avoid parallel implementations.

#### Scenario: Polygon prediction matches bbox GT
- GIVEN a GT bbox and a prediction polygon covering the same region
- WHEN the evaluator uses the shared helper to derive a bbox/segmentation for IoU
- THEN the prediction can be paired and scored correctly instead of being dropped for geometry mismatch.
