## Why

CoordExp already has useful visualization infrastructure, but it is split across
multiple source-specific contracts:

- offline overlays read `gt_vs_pred.jsonl`,
- evaluator overlays render selected scenes during evaluation,
- Stage-2 monitor dumps serialize rollout-rich `gt_objects` / `pred_objects` /
  `match` / `stats` payloads,
- backend-compare and Oracle-K tools compose multiple runs using ad hoc join and
  rendering logic.

The main gap is no longer “do we have visualization tools.”
The gap is that the repo still lacks one explicit, reusable visualization
resource contract.

That missing contract shows up in four concrete ways:

- object-shape drift:
  - offline artifacts use `type` / `points` / `desc`,
  - monitor dumps use `bbox_2d`, `points_norm1000`, `geom_type`, and field-order
    variants,
- matching drift:
  - evaluator-selected scenes and monitor dumps carry different match payloads
    and index domains,
- comparison drift:
  - stable join keys exist, but there is no shared rule for verifying two aligned
    records actually represent the same GT scene,
- scaling drift:
  - some producers store `norm1000`-style coordinates or coord tokens in the
    `0..999` range and rely on renderer-local inverse scaling.

We need one low-redundancy visualization contract that cooperates with the
current infrastructure rather than replacing it:

- keep `gt_vs_pred.jsonl` as the base single-run artifact family,
- normalize all visualization scenes into one pixel-space bbox-only review
  contract,
- preserve prediction order,
- reuse shared geometry and inverse-scaling helpers,
- keep raw monitor dumps as telemetry and leave their paths unchanged,
- and let comparison workflows compose normalized scenes safely.

## What Changes

- Add a new capability, `gt-vs-pred-visualization`, that defines the canonical
  visualization-resource contract and the default GT-vs-Pred review semantics.
- Keep the visualizer’s public responsibility intentionally narrow:
  - input: canonical or canonicalized `gt_vs_pred.jsonl`
  - required explicit `output path`
  - no ownership of `monitor_dumps` path layout
- Keep the canonical single-view resource **gt-vs-pred compatible at the top
  level**:
  - `gt`, `pred`, `image`, `width`, `height`, `coord_mode`, `record_idx`,
    `source_kind`.
- Standardize one explicit object-level schema for visualization:
  - each `gt[*]` / `pred[*]` object uses:
    - `index`
    - `desc`
    - pixel-space `bbox_2d`
- Standardize canonical GT ordering independently of source-local GT order so
  matching and comparison use one reproducible `canonical_gt_index` domain.
- Standardize one explicit matching sub-schema for visualization:
  - canonical index domains,
  - `matched_pairs`,
  - `fn_gt_indices`,
  - `fp_pred_indices`,
- Require canonical visualization resources used by the shared renderer to carry
  canonical matching already materialized:
  - source-provided matching may be reused,
  - raw sources without canonical matching must go through an explicit
    normalization/materialization step,
  - the renderer itself must fail fast rather than attempting fallback matching.
  - and optional threshold/scope/debug fields.
- Require source adapters to normalize all supported source forms into that
  object schema:
  - offline `type` / `points`,
  - scored artifacts,
  - monitor `points_norm1000`,
  - coord-token / `norm1000` payloads in the `0..999` range,
  - and any bbox-derivable source geometry.
- Require inverse scaling from `norm1000` to use shared geometry helpers and
  per-record `width` / `height`, not renderer-local ad hoc logic.
- Preserve prediction order as a contract invariant.
- Keep raw monitor dumps as telemetry artifacts, but require a lossless
  normalization path into the canonical visualization resource outside the
  visualizer itself.
- Tighten multi-run comparison composition:
  - candidate alignment may start from stable join keys,
  - but composition must fail fast unless normalized GT scenes match exactly.
- Make materialized canonical visualization resources derived sidecars rather
  than path-overloading the raw inference artifact:
  - they MUST NOT overwrite the source `gt_vs_pred.jsonl`,
  - the default materialized location SHOULD be
    `<run_dir>/vis_resources/gt_vs_pred.jsonl`.
- Bring both relevant monitor-dump producer families into scope:
  - `stage2-ab-training`
  - `rollout-matching-sft`

## Capabilities

### Added Capabilities
- `gt-vs-pred-visualization`: canonical visualization-resource contract,
  normalized object/matching schemas, shared GT-vs-Pred renderer semantics, and
  safe comparison-scene composition rules.

### Modified Capabilities
- `inference-engine`: preserve prediction order in emitted `gt_vs_pred.jsonl`
  artifacts so downstream visualization remains faithful to parsed rollout order.
- `detection-evaluator`: route overlays and Oracle-K visualization-audit
  surfaces through the shared visualization contract, preserving normalized
  matching and safe comparison alignment.

## Impact

- No model or decoding behavior changes are required by this proposal.
- Existing `gt_vs_pred.jsonl` remains the canonical base single-run artifact
  family rather than being replaced by a new parallel artifact family.
- Materialized visualization resources become derived sidecars, so implementers
  do not have to guess whether a given `gt_vs_pred.jsonl` path uses inference
  schema or visualization schema.
- `monitor_dumps` path layout remains unchanged and outside the visualizer’s path
  ownership boundary.
- Canonical visualization resources intentionally narrow localization review to
  pixel-space `bbox_2d` plus `desc`, which makes offline, online, and
  comparison scenes share one explicit object schema.
- Legacy `vis_tools/` remain valid semantic references, but future reusable
  visualization behavior should converge on one shared contract and adapter
  stack.
- Comparison workflows such as backend-compare and Oracle-K become safer because
  candidate join keys are no longer treated as sufficient proof of scene
  identity.
