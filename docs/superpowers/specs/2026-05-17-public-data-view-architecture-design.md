# Public Data View Architecture Refactor Design

Status: design reviewed and gated; implementation has not started.

Date: 2026-05-17

Owner: CoordExp data and training infrastructure

Primary scope: `public_data/` layout, dataset view contracts, image-store reuse,
length-budget filtering, and latest compact detection training input resolution.

## Purpose

This design refactors public detection data from preset directories that mix
image resolution, annotation policy, coordinate storage, object-count filtering,
and model-facing tokenization into a smaller architecture with explicit image
stores and annotation views.

The motivating change is the move to the `compact_full` training template. The
new template shortens assistant sequences enough that object-count capping is no
longer the right admission policy. Future model-facing datasets should be
prepared offline with a total token-length budget, and training should not filter
or truncate object sequences at runtime.

## Current Problems

- `max_objects` still exists as a latest compact runtime field and causes
  `DetectionTrainingDataset` to reject rows after loading.
- Current public-data preset names encode too many independent concerns, for
  example `rescale_32_1024_bbox_lvis_proxy_len12000`.
- Same-resolution variants can duplicate or imply duplicated image roots.
- Existing JSONL layout still distinguishes `.jsonl`, `.norm.jsonl`, and
  `.coord.jsonl`, even though all model-facing Qwen3-VL grounding data should
  live on the norm1000 coordinate lattice.
- Current docs say JSONL image paths are relative to the JSONL directory, while
  the desired architecture needs image paths to be relative to a shared image
  store.
- LVIS-proxy variants need explicit annotation-view semantics so noisy proxy
  policies are not confused with clean COCO80 supervision.

## Design Decisions

### Decision 1: Deprecate `max_objects` As Runtime Policy

`max_objects` is past-tense artifact provenance, not a training runtime
admission policy.

New behavior:

- Latest compact training must not filter, truncate, or reject rows because of
  object count.
- Dataset filtering happens before training by generating a new JSONL view.
- Existing object-count-filtered artifacts are represented as legacy views such
  as `max-60`.
- Short-term config compatibility may warn and ignore old `data.max_objects`.
  After migration, `data.max_objects` can become an obsolete key.

### Decision 2: New JSONL Image Paths Are Image-Store-Relative

Canonical view JSONLs store safe relative image paths such as:

```json
{"images": ["images/train2017/000000123456.jpg"]}
```

The path is resolved against the view's declared image store, not against the
JSONL directory.

The view `meta.json` owns the image-store reference:

```json
{
  "image_store": "public_data/coco/images/res-1024",
  "image_path_semantics": "image_store_relative"
}
```

For committed artifacts, `image_store` is a repo-root-relative POSIX path.
Absolute image-store paths are allowed only for temporary tests or local debug
metadata. Loader and validator code must resolve the path from an explicit
`repo_root` or from the metadata/manifest location using a repo-root discovery
rule; it must not depend on the process current working directory.

### Decision 3: Use Short, Self-Contained Names

Do not encode invariant or repo-global facts in path names.

Naming rules:

- Use `res-1024`, not `rescale_32_1024`.
- Omit `32` because Qwen3-VL is the repo target and the 32-aligned image
  lattice is a global requirement.
- Omit `bbox` from image-store names because bbox/poly is an annotation-view
  concern, not an image-store concern.
- Do not use contextual `v1` or `v2` names. If a future image store differs,
  name the actual meaningful difference.
- Use `len-12000` for total-token-budget-filtered views.
- Use `max-60` only for legacy object-count-filtered views.

### Decision 4: Image Store Metadata Is Lightweight By Default

Image stores get a lightweight `meta.json` by default. Full image checksums are
optional and generated only when freezing or syncing image stores across nodes.

Default image-store metadata records:

- schema version;
- dataset id;
- image store id;
- max pixel / visual token budget;
- image path semantics;
- source or adoption path;
- split coverage.

JSONL checksums remain the cheap default sentinel for model-facing artifacts.

### Decision 5: Preserve Old Image Roots In Phase 1; Move Only Behind A Cleanup Gate

The long-term canonical image store is:

```text
public_data/coco/images/res-1024/
  images/
  meta.json
```

Phase 1 must not break the old runnable image root before the new view layout
has passed smoke and production has switched. The implementation must therefore
use a non-destructive image-store population strategy by default, or stop for an
explicit destructive-action checkpoint before any `mv`-style operation.

Acceptable Phase 1 outcomes:

- `public_data/coco/images/res-1024/` exists and new view JSONLs resolve images
  through that store;
- the old sibling roots remain available for rollback and comparison;
- any temporary duplicate image bytes are reclaimed only in the separate cleanup
  step after production training has passed an agreed acceptance signal and the
  user explicitly approves destructive cleanup.

The final architecture still has one canonical image store. Preserving the old
root during Phase 1 is rollback safety, not a second canonical compatibility
layer.

### Decision 6: Migrate Old Sibling Roots Into Views

Old sibling roots should not remain as a second canonical layout. Their JSONL
contents should be migrated or regenerated under `views/`, and historical
provenance should be recorded in view metadata.

Examples of old roots to retire as canonical disk roots:

```text
public_data/coco/rescale_32_1024_bbox/
public_data/coco/rescale_32_1024_bbox_len12000/
public_data/coco/rescale_32_1024_bbox_lvis_proxy_len12000/
public_data/coco/rescale_32_1024_bbox_max60/
public_data/coco/rescale_32_1024_bbox_max60_lvis_proxy/
```

### Decision 7: Canonical Training Configs Infer The Image Store From View Metadata

New canonical training configs should only need the view JSONLs:

```yaml
data:
  train_jsonl: public_data/coco/views/coco80/len-12000/train.jsonl
  val_jsonl: public_data/coco/views/coco80/len-12000/val.jsonl
```

The loader resolves `image_store` from the view `meta.json`.

Explicit `data.image_root` may remain as a short-term legacy or debug override,
but it must match the view metadata. Mismatches should fail fast.

### Decision 8: Canonical Views Store Norm1000 Integers

All JSONL files under `public_data/<dataset>/views/**` are prepared model/eval
views. Their geometry is stored as bare norm1000 integers:

```json
{"bbox_2d": [124, 93, 456, 812]}
```

They do not store coord-token strings:

```json
{"bbox_2d": ["<|coord_124|>", "<|coord_93|>", "<|coord_456|>", "<|coord_812|>"]}
```

Coord-token rendering is a training/template responsibility:

```text
prepared JSONL norm1000 ints
-> dataset parser
-> template renderer converts ints to Qwen coord tokens
-> assistant text and labels
```

Pixel-space coordinates are allowed in raw/source annotations or temporary
build products only, not in canonical `views/**/{train,val}.jsonl`.

The valid canonical coordinate range is exactly `0..999`, inclusive. The value
`1000` is invalid in canonical view JSONL even if a raw conversion produced it
before clamping. Raw pixel-space conversion code must round/clamp into the
norm1000 lattice before writing a view, and validators must reject out-of-range
coordinates in committed views.

### Decision 9: Regenerate Views From Aligned Upstream Sources, Not From Coord Tokens

The first migration should not rerun raw image resizing unless an alignment
check shows the current image store is unusable. Re-resizing images would mix an
image-byte change into a data-contract migration and make old-vs-new sample
comparisons harder to interpret.

Instead:

- prepare/adopt the current aligned 1024 prepared images as
  `public_data/coco/images/res-1024` without breaking the old image root during
  Phase 1;
- build canonical COCO80 views from the resized pixel-space JSONLs when
  available;
- build length-budget views by re-running the length-budget factory from the
  resized pixel-space source and the target tokenizer/template;
- build LVIS-proxy views by re-running the proxy projection or by consuming
  existing `.norm.jsonl` only when it is explicitly the best aligned source;
- use existing `.norm.jsonl` files for legacy views when the goal is to preserve
  historical sample membership such as `max-60`;
- treat `.coord.jsonl` reverse conversion as a last-resort legacy importer, not
  as the default migration source.

Every regenerated view should be compared against the old artifact it replaces
using row counts, object counts, image ids, and length-budget stats where
applicable.

### Decision 10: Keep All-Proxy In Phase 1; Defer Hard-Proxy To Proxy Confidence

The Phase 1 proxy migration creates the broad all-proxy research view only:

```text
public_data/coco/views/coco80-lvis-proxy/len-12000/
```

The `coco80-lvis-proxy/len-12000` view preserves the broad all-proxy research
surface so later experiments can train with different real/proxy/plausible
weights. It is not the default training-safe hard-bbox view.

Phase 1 must not create `coco80-lvis-proxy-hard/len-12000`, must not run a
hard-proxy train smoke, and must not switch direct training to proxy
supervision. The hard-proxy training view is a Phase 2 artifact that can be
materialized only after `proxy_confidence` produces a versioned
`policy_proxy_hard_bbox_v0.json` plus pair-metrics checksum.

Proxy metadata must be rich enough for template rendering and loss computation
to trace every supervised token span back to the source object and its
supervision policy.

The current parallel-array proxy metadata is useful but too fragile for the new
architecture. The new contract should key supervision by stable `object_id`
rather than by object index alone, because object order may change during
training.

Sketch for the extensible supervision metadata contract:

```json
{
  "objects": [
    {
      "object_id": "coco:ann:1666628",
      "bbox_2d": [699, 284, 722, 336],
      "category_id": 85,
      "category_name": "clock",
      "desc": "clock"
    },
    {
      "object_id": "lvis:ann:13415:as-coco:86",
      "bbox_2d": [623, 482, 636, 509],
      "category_id": 86,
      "category_name": "vase",
      "desc": "vase"
    }
  ],
  "metadata": {
    "supervision": {
      "schema_version": 1,
      "object_supervision": {
        "coco:ann:1666628": {
          "source": "coco",
          "target_role": "real",
          "target_relation": "ground_truth",
          "desc_ce_weight": 1.0,
          "coord_ce_weight": 1.0,
          "coord_reg_weight": 1.0
        },
        "lvis:ann:13415:as-coco:86": {
          "source": "lvis",
          "target_role": "proxy_candidate",
          "target_relation": "same_extent_exact_name",
          "mapped_coco_category_id": 86,
          "mapped_coco_category_name": "vase",
          "lvis_category_id": 1139,
          "lvis_category_name": "vase",
          "lvis_ann_id": 13415,
          "desc_ce_weight": 0.0,
          "coord_ce_weight": 0.0,
          "coord_reg_weight": 0.0,
          "eligible_for_direct_bbox_training": false,
          "evidence": {
            "mapping_kind": "exact_canonical",
            "why_recovered": "strict exact_canonical mapping; no COCO vase annotation at IoU >= 0.50"
          }
        }
      },
      "summary": {
        "real_count": 20,
        "proxy_candidate_count": 2,
        "proxy_plausible_count": 0,
        "support_only_count": 0
      }
    }
  }
}
```

Expected target roles:

- `real`: COCO80 annotation.
- `proxy_candidate`: LVIS-derived object rendered in the all-proxy research view
  for provenance and future weighting, but not yet approved for direct
  hard-bbox training. Phase 1 proxy candidates must not carry default direct
  coordinate/regression supervision weights.
- `proxy_hard`: Phase 2 LVIS-derived object safe for direct bbox/coord
  supervision under a calibrated policy.
- `proxy_plausible`: LVIS-derived object that may supervise text and/or weak
  object presence but should not necessarily supervise coordinates.
- `support_anchor`: evidence that suggests a missing COCO object but has extent
  mismatch risk, such as tablecloth/table-family relations.
- `support_cue`: related-object evidence that should not be treated as a COCO
  object box without an explicit weak-supervision objective.
- `ignore`: retained for audit/evidence only.

Expected relation labels should stay open-ended but versioned, for example:

- `ground_truth`;
- `same_extent_exact_name`;
- `same_extent_alias`;
- `hyponym_same_extent`;
- `same_extent_empirical_strong`;
- `table_family_extent_mismatch`;
- `part_of_person_anchor`;
- `container_or_product_cue`;
- `functional_overlap_anchor`;
- `related_object_substitution`.

Loss weights should be split by span family rather than using one overloaded
weight:

- `desc_ce_weight`;
- `structural_ce_weight`;
- `coord_ce_weight`;
- `coord_reg_weight`;
- optional future fields such as `presence_weight`, `eos_suppression_weight`,
  or `type_gate_weight`.

For Phase 1 all-proxy views, `proxy_candidate` records are provenance-bearing
and renderability-bearing, not direct hard-bbox training targets. Their default
coordinate and regression weights must be `0.0` or omitted unless a later
approved policy promotes the object to `proxy_hard`. Direct proxy training
weights belong to the Phase 2 hard-proxy policy or to a separate weighted-loss
objective, not to the Phase 1 layout migration.

The renderer should emit assistant-token span metadata keyed by `object_id` and
span family. The loss layer should join rendered spans to
`metadata.supervision.object_supervision[object_id]` to apply per-source and
per-relation weights.

Renderer-side span sketch:

```json
{
  "object_id": "lvis:ann:13415:as-coco:86",
  "span_family": "bbox_coord",
  "field": "bbox_2d",
  "token_start": 381,
  "token_end": 389,
  "source_role": "proxy_candidate",
  "target_relation": "same_extent_exact_name"
}
```

Token spans are runtime sidecars derived from rendering; they should not be
stored in the dataset JSONL because token positions depend on template,
tokenizer, object ordering, and chat-template wrapping.

### Decision 11: Keep Object Payloads Minimal; Store Supervision Roles Only In Metadata

The canonical object payload should avoid duplicated supervision truth.

Objects carry `object_id` plus fields needed by rendering, training targets, and
evaluation:

```json
{
  "object_id": "lvis:ann:13415:as-coco:86",
  "bbox_2d": [623, 482, 636, 509],
  "category_id": 86,
  "category_name": "vase",
  "desc": "vase"
}
```

Fields such as `target_role`, `target_relation`, `source`, loss weights, LVIS
mapping evidence, and proxy tier live only in:

```text
metadata.supervision.object_supervision[object_id]
```

The renderer and loss layers must join through `object_id`. Do not duplicate a
minimal `source_role` or `target_role` into each object for convenience; doing
so would create two sources of truth and increase drift risk when proxy policy
is reweighted or reclassified.

### Decision 12: Implement A Dedicated COCO View Factory First

The first implementation should not make the whole `public_data/run.sh` unified
runner view-aware. The blast radius is already large enough: image-store layout,
view metadata, norm1000 integer storage, length-budget filtering, proxy
metadata, latest compact parsing, and config resolution all change together.

Instead, implement a dedicated COCO view factory first:

```bash
PYTHONPATH=. conda run -n ms python public_data/scripts/build_coco_views.py \
  --model-path model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp \
  --image-store public_data/coco/images/res-1024 \
  --source-preset public_data/coco/rescale_32_1024_bbox \
  --output-root public_data/coco/views \
  --views coco80/full coco80/len-12000 coco80/max-60 coco80-lvis-proxy/len-12000 \
  --max-total-tokens 12000 \
  --splits train val
```

The implementation should still use reusable internal abstractions so it can be
folded into the unified runner later:

- `ImageStore`;
- `AnnotationView`;
- `ViewMetadataWriter`;
- `LengthBudgetPolicy`;
- `ProxyPolicy`;
- `SupervisionMetadataBuilder`.

After the COCO factory, metadata, validator, latest compact training, and smoke
path are working, a second-phase change can expose view concepts through
`public_data/run.sh`.

### Decision 13: Migrate Smoke Configs Before Production Configs

The first implementation should not immediately switch Stage-1 production
configs to the new `views/**/train.jsonl` paths.

Instead, add or update smoke configs first, for example:

```text
configs/stage1/recursive_detection_ce_latest/smoke/compact_full_coco80_len12000_tiny.yaml
configs/stage1/recursive_detection_ce_latest/smoke/compact_full_coco80_lvis_proxy_all_len12000_parse.yaml
```

The smoke path should prove:

- `coco80/len-12000` parses and trains;
- `coco80-lvis-proxy/len-12000` parses and produces source/role/span sidecars
  for future weighted proxy objectives;
- view metadata correctly resolves the image store;
- no runtime object-count filtering remains.

After these checks pass, a second phase can migrate production configs such as
`configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2.yaml`.

### Decision 14: Delay Deleting Old Roots Until Production Training Has Switched

Old sibling artifact roots should not be deleted in the first implementation.
They become non-canonical rollback and comparison evidence while new `views/`
artifacts are validated.

Cleanup gate:

1. new views are generated and validated;
2. smoke configs pass;
3. production configs are migrated;
4. Git-tracked provenance manifests and JSONL checksums are updated;
5. rollback/regeneration path is verified;
6. production training reaches an agreed acceptance signal, such as a usable
   checkpoint, validation point, or user-approved run milestone;
7. the user explicitly approves destructive cleanup;
8. only then remove old sibling roots in a separate cleanup step.

Until that gate is reached, old roots should not be used by new code or docs as
canonical paths, but they may remain on disk for comparison and rollback.

### Decision 15: Completion Requires A Real Latest-Compact Smoke Run

Targeted tests and structural validation are necessary but not sufficient for
this migration. The implementation is complete only after at least one real
latest-compact smoke run exercises the end-to-end path through:

```text
view JSONL
-> view metadata image-store resolution
-> norm1000 integer parser
-> compact template rendering to Qwen coord tokens
-> tokenizer/chat-template wrapping
-> target/span sidecars
-> loss computation
```

Minimum completion gate:

1. targeted unit tests;
2. public-data view structural validation;
3. length-budget stats validation;
4. latest compact config parse;
5. latest compact real smoke run on `coco80/len-12000`;
6. all-proxy parse plus source/role/span sidecar validation.

The smoke may be tiny, but it must use the real `src/sft.py` training entrypoint
for at least the base COCO80 length-budget view.

### Decision 16: Train On Hard Proxy Only After Proxy Confidence Exists

The first implementation should not wire all-proxy weighted losses into latest
compact training and should not train from hard-proxy supervision.

Training use after Phase 1:

- `coco80/len-12000` is the only required direct Stage-1 smoke/training surface
  for the layout migration.
- `coco80-lvis-proxy/len-12000` is preserved as a rich research view. It must
  parse, render, and expose object/span provenance, but it is not the first
  direct weighted-loss production path.
- `coco80-lvis-proxy-hard/len-12000` becomes the first proxy training candidate
  only in Phase 2, after the calibrated hard-bbox policy exists.

The Phase 1 all-proxy view may contain renderable proxy candidates so that
formatting, token spans, and future weighting can be validated, but default
latest-compact training must not consume those candidates as direct
coordinate/regression supervision.

Future work can add an explicit proxy-weighted objective config that consumes
all-proxy metadata:

```yaml
objective:
  proxy_weighting:
    enabled: true
    role_weights:
      real:
        desc_ce: 1.0
        coord_ce: 1.0
        coord_reg: 1.0
      proxy_hard:
        desc_ce: 1.0
        coord_ce: 1.0
        coord_reg: 1.0
      proxy_plausible:
        desc_ce: 0.25
        coord_ce: 0.0
        coord_reg: 0.0
```

That later objective should be treated as its own loss-contract change rather
than being folded into the data-layout migration.

### Decision 17: Derive Proxy-Hard Inclusion From Continuous Confidence Metrics

Proxy inclusion should not be governed primarily by ambiguous discrete labels
such as `strict`, `plausible`, `anchor`, or `cue`. Those labels are useful for
human explanation, but they mix semantic similarity, geometry agreement,
recovery value, noise risk, and training use.

The canonical model should be:

```text
raw LVIS/COCO overlap evidence
-> continuous proxy confidence metrics
-> versioned threshold policy
-> materialized proxy view
```

Create a separate, prerequisite workstream:

```text
lvis-coco-proxy-confidence-calibration
```

Expected artifacts:

```text
public_data/coco/proxy_confidence/
  lvis_coco80_pair_metrics.jsonl
  policy_proxy_hard_bbox_v0.json
  policy_proxy_all_v0.json
  reports/
    top_included.md
    top_excluded.md
    borderline_pairs.md
```

The pair metrics should expose continuous evidence, for example:

```json
{
  "lvis_category_name": "mug",
  "coco_category_name": "cup",
  "semantic": {
    "name_similarity": 0.82,
    "alias_match": false,
    "hyponym_score": 0.74,
    "definition_similarity": 0.79
  },
  "geometry": {
    "matched_pair_count": 1259,
    "precision_at_iou50": 0.978,
    "mean_iou": 0.901,
    "iou50_rate": 0.982,
    "iou75_rate": 0.932,
    "area_ratio_median": 0.96,
    "center_distance_norm_median": 0.03
  },
  "recovery": {
    "train_recovered_count": 339,
    "val_recovered_count": 4,
    "coverage_gain": 0.762
  },
  "risk": {
    "extent_mismatch_score": 0.04,
    "part_of_score": 0.02,
    "container_cue_score": 0.01,
    "substitution_score": 0.03
  },
  "confidence": {
    "bbox_supervision_score": 0.94,
    "desc_supervision_score": 0.96,
    "presence_score": 0.97,
    "eos_suppression_score": 0.97
  }
}
```

The hard-proxy policy should be a threshold artifact, not code comments:

```json
{
  "policy_id": "proxy_hard_bbox_v0",
  "include_if": {
    "bbox_supervision_score": {">=": 0.90},
    "precision_at_iou50": {">=": 0.90},
    "mean_iou": {">=": 0.75},
    "iou75_rate": {">=": 0.50},
    "matched_pair_count": {">=": 50},
    "extent_mismatch_score": {"<=": 0.20},
    "substitution_score": {"<=": 0.10}
  }
}
```

Discrete fields in view metadata, such as `target_role: proxy_hard` or
`target_relation: same_extent_empirical_strong`, are derived explanations of a
specific policy decision. They are not the primary source of confidence.

Proxy views must record the policy id and metrics checksum they consumed. Each
proxy object's supervision metadata should include the relevant score snapshot
and a stable reference to the pair metrics row used to include it.

### Decision 18: Proxy Confidence V0 Is Global Category-Pair Confidence

The first proxy confidence calibration should be global over LVIS category to
COCO80 category pairs, not per sample or per instance.

The calibrated unit is:

```text
(lvis_category_id, coco_category_id)
```

The pair-level metrics summarize evidence across the dataset and define a stable
mapping policy. Individual instances inherit the pair policy and then pass only
simple safety gates such as duplicate overlap, conflicting overlap, and LVIS
exhaustiveness checks.

This keeps the first version interpretable and reusable:

```text
global pair confidence + instance safety gates -> materialized proxy view
```

Per-instance confidence, image-context confidence, detector/verifier-assisted
confidence, or sample-specific weighting are future workstreams. They should not
be mixed into the v0 proxy-confidence artifact.

### Decision 19: Record Semantic Embedding Evidence, But Do Not Let It Dominate Hard-Bbox Inclusion

Proxy-confidence v0 may use local sentence-embedding weights to record semantic
similarity evidence. The repository already has a canonical semantic
description helper in `src/common/semantic_desc.py`, and local sentence
embedding weights are available under:

```text
model_cache/all-MiniLM-L6-v2-local
```

If implementation uses a different embedding checkpoint, the metrics artifact
must record the exact model path, revision if any, checksum policy, text
normalization function, pooling method, and embedding dimensionality.

Semantic scores should be stored in the pair metrics table, for example:

```json
{
  "semantic": {
    "canonical_name_match": false,
    "alias_match": true,
    "embedding_model": "model_cache/all-MiniLM-L6-v2-local",
    "embedding_cosine": 0.82,
    "definition_embedding_cosine": 0.79
  }
}
```

However, the first `proxy_hard_bbox_v0` inclusion policy should primarily rely
on geometry overlap, recovery, and risk metrics. Semantic embeddings are used
for:

- reporting and explanation;
- manual audit prioritization;
- low-support fallback evidence;
- relation naming.

They should not by themselves promote a pair into direct hard bbox supervision,
because many risky mappings are semantically close but geometrically wrong, such
as local parts, containers, surface anchors, or related-object substitutions.

### Decision 20: Derive Proxy Policies From Train Evidence; Use Val As Holdout Diagnostics

The first proxy-confidence policy should be derived from training-split evidence
only. Validation evidence is held out for diagnostics and sanity checks.

Pair metrics may report multiple scopes:

```json
{
  "metrics_train": {},
  "metrics_val": {},
  "metrics_all": {}
}
```

But the hard-bbox inclusion policy should explicitly declare:

```json
{
  "policy_id": "proxy_hard_bbox_v0",
  "decision_split": "train",
  "diagnostic_splits": ["val"]
}
```

If a pair passes train thresholds but fails validation diagnostics badly, the
report should mark it as `borderline` or `manual_review` rather than silently
including it. The validation split must not be used to tune default inclusion
thresholds for the first policy.

### Decision 21: Split Proxy Confidence Into A Prerequisite Phase Before Hard-Proxy View Generation

`proxy_confidence` is now a separate calibration work item, not a small detail
inside the layout migration.

The first implementation should be phased:

```text
Phase 1: public-data view architecture migration
  - create/adopt public_data/coco/images/res-1024
  - create views/coco80/full
  - create views/coco80/len-12000
  - create views/coco80/max-60
  - create views/coco80-lvis-proxy/len-12000 as all-proxy research view
  - update latest compact parser/rendering to consume norm1000 integer views
  - validate all-proxy source/role/span metadata
  - smoke latest compact training on coco80/len-12000

Phase 1.5: lvis-coco-proxy-confidence-calibration
  - build public_data/coco/proxy_confidence/lvis_coco80_pair_metrics.jsonl
  - build public_data/coco/proxy_confidence/policy_proxy_hard_bbox_v0.json
  - generate inclusion/exclusion/borderline reports
  - use train-derived decisions and val holdout diagnostics

Phase 2: hard-proxy training view
  - materialize views/coco80-lvis-proxy-hard/len-12000 from the policy artifact
  - validate length budget after hard-proxy inclusion
  - smoke/train hard-proxy view
```

This prevents a rushed hard-proxy inclusion policy from being hidden inside a
data-layout migration. The all-proxy view remains available after Phase 1 for
research and later weighted objectives, but the direct hard-proxy training view
waits for the calibrated policy artifact.

### Decision 22: Keep Weak/Support Proxy Evidence Out Of The Default Rendered Object List

Phase 1 all-proxy views should not put weak support evidence into the canonical
`objects` list when that evidence is not intended to be rendered as a detection
target by the current template.

Default rule:

- `objects` contains real COCO objects plus proxy candidates that the view
  policy intends to render as detection targets.
- weak/support/evidence-only candidates live under a sidecar such as
  `metadata.supervision.support_objects`.

This prevents support evidence such as tablecloth/table, clothing/person, or
container/product cues from silently becoming target objects in the assistant
sequence before a weighted objective exists.

Sketch:

```json
{
  "objects": [
    {
      "object_id": "coco:ann:1666628",
      "bbox_2d": [699, 284, 722, 336],
      "desc": "clock"
    },
    {
      "object_id": "lvis:ann:13415:as-coco:86",
      "bbox_2d": [623, 482, 636, 509],
      "desc": "vase"
    }
  ],
  "metadata": {
    "supervision": {
      "object_supervision": {
        "coco:ann:1666628": {"target_role": "real"},
        "lvis:ann:13415:as-coco:86": {"target_role": "proxy_candidate"}
      },
      "support_objects": {
        "lvis:ann:999:as-coco:67": {
          "target_role": "support_anchor",
          "target_relation": "table_family_extent_mismatch",
          "bbox_2d": [100, 200, 500, 700],
          "desc": "dining table",
          "source_desc": "tablecloth"
        }
      }
    }
  }
}
```

If a later objective wants to render support evidence, it must opt in with a
clear template and loss policy. The view metadata must declare whether the
length budget includes only `objects` or also one or more support sidecar
families.

### Decision 23: All-Proxy Phase 1 Objects Include Real COCO Plus Renderable Proxy Candidates

The Phase 1 all-proxy research view should preserve current renderable LVIS
proxy candidates in the canonical `objects` list, while keeping weak
support/cue/anchor evidence in sidecars.

Default all-proxy `objects`:

- real COCO objects;
- current renderable proxy candidates that were previously appended to
  `objects` by the proxy builder and can be represented as detection targets.

Default all-proxy sidecars:

- `support_anchor`;
- `support_cue`;
- `ignore`;
- other evidence-only objects that should not be rendered by the current
  compact detection template.

These renderable proxy candidates are not yet the first weighted-loss
production path. They exist so Phase 1 can validate template rendering, object
span provenance, and future weighted objective readiness. Direct training still
uses only the base COCO view until `proxy_confidence` and the hard-proxy view
are ready.

### Decision 24: Length Budget Counts Only Rendered Training Objects In Phase 1

Phase 1 `len-12000` accounting should include only the object families rendered
by the current training template.

For the all-proxy Phase 1 view, the training/rendered sequence is:

```text
real COCO objects + renderable proxy candidates
```

Support sidecars do not count toward the 12k budget because they are not
rendered into the assistant response and do not consume sequence tokens.

View metadata should make this explicit:

```json
{
  "length_budget_scope": {
    "rendered_families": ["objects"],
    "excluded_sidecars": ["metadata.supervision.support_objects"]
  }
}
```

If a future objective/template renders support evidence, it should use a
separate view/policy, for example a support-rendered view, and recompute length
budget under that rendered surface. Current training views should remain
`coco + proxy candidates`, not hidden support-object training.

### Decision 25: Generate `coco80/full` As The Canonical Parent View

Phase 1 should materialize `public_data/coco/views/coco80/full/` even though it
is not the immediate training target.

`coco80/full` is the canonical unfiltered prepared COCO80 view. It provides a
stable parent for derived views such as:

```text
coco80/len-12000
coco80/max-60
future coco80/len-16000
future coco80/len-20000
```

This makes provenance clearer:

- `coco80/full` aligns with the current resized COCO 1024 base artifact, but
  stores canonical norm1000 integer coordinates and image-store-relative paths;
- `coco80/len-12000` is a length-budget-filtered child view;
- `coco80/max-60` is a legacy object-count-filtered child view;
- future length budgets can be rebuilt from `coco80/full` without returning to
  old preset roots.

Phase 1 minimum views:

```text
public_data/coco/views/coco80/full/
public_data/coco/views/coco80/len-12000/
public_data/coco/views/coco80/max-60/
public_data/coco/views/coco80-lvis-proxy/len-12000/
```

### Decision 26: Proxy Length Views Derive From `coco80/full` And Filter After Annotation Policy

`coco80-lvis-proxy/len-12000` should derive from `coco80/full`, not from
`coco80/len-12000`.

The order is:

```text
coco80/full
-> apply LVIS proxy annotation policy
-> render the final object sequence
-> filter by the 12k total-token budget
-> write coco80-lvis-proxy/len-12000
```

This makes the view name mean exactly:

```text
the final proxy-augmented rendered sequence fits in 12k tokens
```

The view metadata should record:

```json
{
  "parent_view": "public_data/coco/views/coco80/full",
  "annotation_policy": "append_lvis_proxy_candidates",
  "sample_policy": {
    "type": "length_budget",
    "max_total_tokens": 12000,
    "applied_after_annotation_policy": true
  }
}
```

The base COCO `len-12000` view remains a sibling child of `coco80/full`, not the
parent of proxy views.

### Decision 27: Phase 1 Completion Includes Git-Tracked Provenance Manifests

Artifact-local `meta.json` files are not enough. Phase 1 must also update
Git-tracked provenance manifests under `manifests/public_data_provenance/`.

The two layers have different roles:

- view/image-store `meta.json`: runtime artifact contract used by loaders,
  validators, and local tools;
- Git-tracked provenance manifest: cross-node reproducibility contract and
  cheap checksum source of truth.

Recommended manifest layout:

```text
manifests/public_data_provenance/coco/images/res-1024.json
manifests/public_data_provenance/coco/views/coco80/full.json
manifests/public_data_provenance/coco/views/coco80/len-12000.json
manifests/public_data_provenance/coco/views/coco80/max-60.json
manifests/public_data_provenance/coco/views/coco80-lvis-proxy/len-12000.json
```

Image-store manifests stay lightweight by default. View manifests must include
JSONL checksums with the existing cheap scope:

```text
scope = jsonl_training_samples_only
algorithm = sha256
```

View manifests should also record records, object counts, source view, policy id,
tokenizer/template where relevant, source-comparison artifact checksums, and
length-stat artifact checksums when a length budget is applied.

## Target Layout

The first target dataset is COCO at the 1024 visual-token image budget:

```text
public_data/coco/
  raw/
    annotations/
    images/

  images/
    res-1024/
      images/
        train2017/...
        val2017/...
      meta.json

  views/
    coco80/
      full/
        train.jsonl
        val.jsonl
        meta.json

      len-12000/
        train.jsonl
        val.jsonl
        train.length_stats.json
        val.length_stats.json
        meta.json

      max-60/
        train.jsonl
        val.jsonl
        meta.json

    coco80-lvis-proxy/
      len-12000/
        train.jsonl
        val.jsonl
        train.length_stats.json
        val.length_stats.json
        meta.json
```

Future Phase 2 or optional compatibility views may add:

```text
public_data/coco/views/
  coco80-lvis-proxy/
    max-60/
      train.jsonl
      val.jsonl
      meta.json

  coco80-lvis-proxy-hard/
    len-12000/
      train.jsonl
      val.jsonl
      meta.json
```

## View Metadata Sketch

Each view root owns a `meta.json`:

```json
{
  "schema_version": 1,
  "kind": "annotation_view",
  "dataset": "coco",
  "view": "coco80/len-12000",
  "image_store": "public_data/coco/images/res-1024",
  "image_path_semantics": "image_store_relative",
  "coordinate_space": "norm1000",
  "coordinate_storage": "integer",
  "coordinate_range": [0, 999],
  "coordinate_chart": "xyxy",
  "assistant_coordinate_rendering": "qwen_coord_tokens",
  "primary_jsonl": {
    "train": "train.jsonl",
    "val": "val.jsonl"
  },
  "sample_policy": {
    "type": "length_budget",
    "max_total_tokens": 12000,
    "budget_includes": [
      "image_patch_tokens",
      "system_prompt_tokens",
      "user_prompt_tokens",
      "assistant_response_tokens"
    ]
  },
  "length_budget_scope": {
    "rendered_families": ["objects"],
    "excluded_sidecars": ["metadata.supervision.support_objects"]
  },
  "summary": {
    "records": 1000,
    "rendered_object_count": 42000,
    "support_sidecar_count": 0
  }
}
```

## Required Migration Surfaces

This is a contract-level change. The implementation must update all relevant
preprocessing and training surfaces, not only latest compact parsing.

Known surfaces:

- `public_data/scripts/convert_to_coord_tokens.py`: stop presenting coord-token
  export as the canonical public-data view writer. In Phase 1, the dedicated
  COCO view factory is the canonical norm1000 writer; coord-token export may
  remain legacy/debug-only.
- `public_data/pipeline/stages.py`: do not change the unified runner's default
  behavior in Phase 1. Add only compatibility hooks or documentation needed to
  keep it from advertising coord-token JSONL as the new canonical architecture.
  A full default pipeline migration is a later approved phase after the
  dedicated COCO factory has passed smoke.
- `public_data/scripts/build_coco_length_budget_artifacts.py`: emit
  `train.jsonl` and `val.jsonl` with norm1000 ints. Estimate length by rendering
  coord tokens internally.
- `src/detection/data.py`: parse norm1000 integer boxes for canonical views while
  preserving legacy coord-token parsing for old `.coord.jsonl` paths until
  production configs are migrated.
- `src/detection/template.py`: convert norm1000 ints into Qwen coord tokens
  during assistant rendering.
- `src/detection/dataset.py`: remove runtime object-count rejection and resolve
  image stores from view metadata.
- `src/sft.py`: resolve the latest compact image store before setting
  `ROOT_IMAGE_DIR`, or stop setting that environment variable once the latest
  compact path passes absolute image paths into chat messages. It must never
  derive `ROOT_IMAGE_DIR` from `None` or stale config state.
- `public_data/scripts/validate_jsonl.py`: add a canonical view mode that
  rejects pixel-space and coord-token geometry, and validates image-store
  relative paths.
- `docs/data/CONTRACT.md`, `docs/data/PREPARATION.md`, and
  `public_data/README.md`: update the public contract and examples.
- `manifests/public_data_provenance/`: update provenance schema/examples for
  image stores and views.
- Stage-1 latest compact configs/tests: switch from `.coord.jsonl` paths to
  canonical view `train.jsonl` / `val.jsonl`.

## Length-Budget Semantics

`len-12000` is a view-level sample policy. The budget includes:

- post-merge image patch tokens;
- system prompt tokens;
- user prompt and chat-template tokens;
- rendered assistant response tokens;
- all detection object sequences after the chosen annotation policy.

The budget is computed from rendered assistant text, not from the JSONL storage
format. Therefore moving canonical storage from coord-token strings to norm1000
integers must not change the admission decision if rendering is equivalent.

## Proxy-View Semantics

LVIS-proxy data is an annotation-view policy, not an image-store policy.

The all-proxy view is for research and flexible weighting. The hard-proxy view
is the recommended direct Stage-1 candidate only after the calibrated
proxy-confidence policy exists and has been materialized into a separate
Phase 2 view.

Anchor/cue relations such as tablecloth-to-dining-table should not be trained
as hard COCO boxes unless a later policy explicitly represents their weak
supervision semantics.

## Compatibility Position

This migration intentionally updates codebase references instead of preserving
old roots as canonical paths or hiding the old layout behind symlinks. Legacy
artifacts may be imported during migration, but the new canonical surface is
`images/` plus `views/`.

Phase 1 preserves old roots as rollback/comparison evidence. Physical deletion
or a destructive image move is a separate user-approved cleanup decision after
the new layout has passed smoke, provenance has been updated, rollback has been
verified, and production training has reached an agreed acceptance signal.

Short-term compatibility exceptions:

- old configs with `data.max_objects` may warn and ignore;
- explicit `data.image_root` may be accepted only if consistent with view
  metadata;
- legacy `.coord.jsonl` may be supported by dedicated import/migration tooling,
  not by the canonical public-data contract.

## Verification Expectations

Implementation must include narrow checks for:

- image-store-relative path resolution;
- view JSONL coordinate storage as norm1000 ints only;
- no runtime object-count rejection in latest compact training;
- length-budget validation after rendering norm1000 ints into coord tokens;
- manifest JSONL checksums for migrated views;
- recursive provenance-manifest discovery for nested `images/` and `views/`
  manifests;
- config parsing with inferred image store;
- rendered all-proxy span sidecars that join by stable `object_id`, not by
  object index or generated legacy ids;
- a Stage-1 latest compact smoke run on at least the COCO80 `len-12000` view.

## Phase Approval Gates

Approval to implement Phase 1 does not approve Phase 1.5 or Phase 2.

- Phase 1 approval covers the image-store/view architecture migration, base
  COCO length-budget smoke, and all-proxy parse/span provenance validation.
- Phase 1.5 approval is required before building proxy-confidence metrics and
  threshold policy artifacts.
- Phase 2 approval is required before materializing
  `coco80-lvis-proxy-hard/len-12000` or running hard-proxy training smoke.

## Execution Gate

This document records the approved design direction. It is not permission to
start implementation.

Before implementation starts:

- review the matching Superpowers implementation plan:
  `docs/superpowers/plans/2026-05-17-public-data-view-architecture.md`;
- review this design and the plan with read-only subagents;
- incorporate accepted review feedback into documentation;
- wait for explicit user instruction to begin implementation.

Implementation must not start until the user explicitly asks for it.
