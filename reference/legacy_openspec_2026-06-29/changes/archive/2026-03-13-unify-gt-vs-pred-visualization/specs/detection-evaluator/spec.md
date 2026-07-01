# detection-evaluator Specification (Delta)

## ADDED Requirements

### Requirement: Visualization-audit outputs reuse the canonical GT-vs-Pred resource contract
Detection-evaluator SHALL route evaluator overlays, Oracle-K audit surfaces, and
other visualization-audit outputs through the canonical GT-vs-Pred
visualization-resource contract rather than bespoke renderer-local schemas.

Normative behavior:

- evaluator-selected scenes intended for shared review rendering MUST materialize
  canonical visualization records
  with:
  - stable join keys (`record_idx`, `image_id`, `file_name` / `image` when
    available),
  - canonical bbox-only `gt` / `pred` objects in pixel space,
  - preserved prediction order,
  - additive `matching`, `stats`, and `provenance` metadata,
- evaluator visualization records intended for shared review rendering MUST
  include canonical `matching`,
- when current evaluator logic already computes matching for a selected scene, it
  MUST preserve and normalize that matching under the canonical `matching`
  namespace rather than forcing a later stage to derive it,
- when evaluator or Oracle-K workflows compose multiple runs for the same scene,
  candidate alignment MAY start from stable join keys, but composition MUST fail
  fast unless width, height, and canonical normalized GT content match exactly,
- overlay rendering MUST delegate to the shared GT-vs-Pred review semantics
  instead of maintaining a second unrelated box-drawing contract,
- Oracle-K visualization-facing outputs MUST preserve the join keys above so
  baseline and Oracle runs can be composed from aligned canonical resources.
- this change MUST NOT require the evaluator to rewrite or take ownership of
  `monitor_dumps` paths.

#### Scenario: Evaluator-selected error scene keeps precomputed matching
- **GIVEN** an evaluator-selected scene for a primary IoU threshold
- **WHEN** a canonical visualization record is materialized for overlay or audit
- **THEN** the record preserves the evaluator’s selected-scene join keys
- **AND** it carries precomputed matching when that matching already exists
- **AND** the renderer is not forced to recompute it from scratch.

#### Scenario: Oracle-K visualization composition rejects misaligned GT scenes
- **GIVEN** baseline and Oracle candidate records that share join keys
- **AND** their normalized GT scenes differ in width, height, or canonical `gt`
- **WHEN** a visualization-facing comparison scene is composed
- **THEN** composition fails fast rather than silently pairing the wrong scenes.

#### Scenario: Evaluator remaps GT indices into canonical GT order
- **GIVEN** an evaluator-selected scene whose source-local GT indices are defined
  over a post-validation GT array
- **WHEN** a canonical visualization record is materialized
- **THEN** the evaluator remaps those GT indices into the deterministic
  `canonical_gt_index` domain before exporting `matching`.
