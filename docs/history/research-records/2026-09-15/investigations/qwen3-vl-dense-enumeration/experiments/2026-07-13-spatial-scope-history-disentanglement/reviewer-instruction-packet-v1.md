---
packet_id: dense-union-51-image-only-review-v1
packet_full_name: Dense-Union-51 Image-Only Independent Review Packet, version 1
review_tool_id: coordexp-image-only-review-jsonl
review_tool_full_name: CoordExp Image-Only Review JavaScript Object Notation Lines Tool
review_tool_version: 1.0.0
language: English
---

# Dense-Union-51 Image-Only Independent Review Packet, Version 1

## Operational purpose

This packet governs two independent, blinded visible-object reviews of
**Dense-Union-51 - Annotation-Derived Dense Union of 51 Images**. Each reviewer
must enumerate every visually separable instance belonging to the **Common
Objects in Context 80-category ontology (`COCO-80`)** from the unmodified
processed source canvas. The review is reference construction, not model-output
assessment.

The immutable packet identifier is
`dense-union-51-image-only-review-v1`. The required review tool is
`coordexp-image-only-review-jsonl`, version `1.0.0`. A tool implementation may
change only after this packet, its digest, and the review queue are versioned
again. The tool must fail closed on unknown fields, unknown states, unknown
reason codes, invalid categories, invalid coordinates, duplicate identifiers,
wrong reviewer roles, image hash drift, packet digest drift, or ontology digest
drift.

## Reviewer roles, independence, and blinding

The only reviewer role identifiers are:

- `reviewer-one`: Independent Reviewer One;
- `reviewer-two`: Independent Reviewer Two.

Each role runs in an independent context and produces a separate JavaScript
Object Notation Lines (`JSONL`) artifact. A reviewer must not receive, inspect,
infer, or communicate the other reviewer's labels. The reviewer must not act as
the adjudicator. The two passes must be completed before either pass is exposed
to adjudication.

For each queue record, the reviewer may receive only:

1. the unmodified source-canvas image named by the queue record;
2. that image's immutable identifier, dimensions, and Secure Hash Algorithm
   256-bit (`SHA-256`) digest;
3. the immutable review identifier and assigned reviewer role slot;
4. this packet and its digest; and
5. `coco-80-review-ontology-v1.json` and its digest.

## Forbidden evidence

The reviewer must not receive or use:

- official individual boxes, official crowd regions, official annotation
  identifiers, or per-image official category/count summaries;
- historical or newly generated model predictions;
- experiment arm names, scores, confidence values, selected-token traces,
  failure labels, or result summaries;
- crops, tiles, masks, outlines, painted hints, resized images, enhanced images,
  or any other derived visual input;
- another reviewer's labels, notes, decisions, or intermediate state; or
- adjudication records or audit-augmented reference objects.

Any exposure to forbidden evidence invalidates the affected role pass. Record
the failure externally; do not continue labeling the affected queue.

## Source-canvas-only image policy

The tool must verify the queue image path, byte digest, width, and height before
display. It must present the complete, unmodified source canvas without crop,
resize, interpolation, masking, marking, color transformation, or region-only
view. Every coordinate is an integer pixel edge in this source-canvas frame.
The reviewer must inspect the complete canvas, including borders and dense
overlap regions, before closing an image.

## Review procedure

For each assigned queue entry:

1. Verify the review identifier, role identifier, image identifier, image
   digest, packet digest, ontology digest, and source dimensions.
2. Inspect the complete source canvas systematically. A raster-style scan is
   recommended, followed by a second full-canvas omission check.
3. Enumerate every visually separable `COCO-80` instance. There is no minimum
   object size or visible-percentage threshold.
4. Assign exactly one reviewer state and one allowed reviewer reason code.
5. Use only canonical normalized category names and identifiers from the frozen
   ontology. Free-text categories and aliases are forbidden.
6. Draw the smallest reproducible axis-aligned rectangle containing the visible
   evidence when the state requires a box.
7. Before closing the image, check for duplicate labels, missed small objects,
   overlapping distinct instances, image-edge truncation, and unresolved
   groups.
8. Canonicalize and validate all labels before writing the role artifact.

## Instance and visible-extent rule

An instance is individually reportable when its category and instance identity
are visually distinguishable and a reviewer can draw a reproducible tight box
around its visible extent. Occlusion and image-edge truncation do not prevent
acceptance when visible evidence still supports the category, identity, and a
stable visible-extent box. Never extrapolate a fully occluded extent.

Overlapping objects remain separate when distinct visible evidence supports
separate identities. When multiple same-category instances cannot be reliably
individuated, emit one `crowd` group region instead of speculative individual
boxes.

## Reviewer states

- `accepted`: one distinguishable `COCO-80` instance, one uniquely selected
  category, and one reproducible visible-extent box.
- `ambiguous`: object evidence exists, but category or instance identity is not
  uniquely resolvable. Record every plausible canonical `COCO-80` category.
- `partial`: a reportable category or instance is plausible or partially
  visible, but occlusion, truncation, or indistinct boundary prevents a
  reproducible individual box.
- `crowd`: multiple instances of one category are visible but cannot be
  individuated reliably. Record one category and one group region.
- `out-of-scope`: visible content is not a `COCO-80` category and cannot enter a
  reportable-object denominator.

Reviewers must exhaustively enumerate visually separable `COCO-80` instances.
`out-of-scope` is used only when the reviewer explicitly considers a salient,
object-like candidate during that scan and rejects it from `COCO-80`. Reviewers
must not inventory every background region, texture, scene element, or other
non-`COCO-80` content. Consequently, `out-of-scope` records are audit traces,
not an exhaustive non-ontology denominator.

## Reason codes

The complete packet reason-code registry is:

- `none`: no exceptional qualification;
- `category_not_unique`: multiple ontology categories remain plausible;
- `instance_not_separable`: individual identity cannot be separated reliably;
- `boundary_not_reproducible`: no reproducible individual visible-extent box;
- `occluded_but_boxable`: occluded instance remains individually reportable;
- `truncated_but_boxable`: image-edge-truncated instance remains individually
  reportable;
- `non_coco80`: content is outside the frozen ontology;
- `official_duplicate`: adjudication-only code for a proposal already owned by
  an official individual;
- `reviewer_disagreement`: adjudication-only code for conflicting reviewer
  labels; and
- `adjudicator_override`: adjudication-only code requiring links to superseded
  labels and a retained audit rationale.

Reviewers may use only `none`, `category_not_unique`,
`instance_not_separable`, `boundary_not_reproducible`,
`occluded_but_boxable`, `truncated_but_boxable`, and `non_coco80`.
`official_duplicate`, `reviewer_disagreement`, and `adjudicator_override` are
forbidden in reviewer-label artifacts.

State/reason constraints are:

| Reviewer state | Required or allowed reviewer reason codes |
|---|---|
| `accepted` | `none`, `occluded_but_boxable`, or `truncated_but_boxable` |
| `ambiguous` | `category_not_unique` or `instance_not_separable` |
| `partial` | `boundary_not_reproducible` or `instance_not_separable` |
| `crowd` | `instance_not_separable` |
| `out-of-scope` | `non_coco80` |

## Coordinate contract

For every reviewer record, provide the smallest axis-aligned integer pixel-edge
corner bounding box (`xyxy`) judged to contain the visible evidence. For
`out-of-scope`, this box is retained only for audit and never enters a
reportable-object denominator:

```text
0 <= x1 < x2 <= source_image_width
0 <= y1 < y2 <= source_image_height
```

A null box is permitted only for `partial` with reason
`boundary_not_reproducible`. Integer coordinates are source-canvas annotation
coordinates; they are not converted
from normalized coordinate tokens. Do not round or copy coordinates from any
forbidden source.

## Reviewer-label record schema

Every reviewer-label record must contain exactly these fields:

| Field | Type and operational meaning |
|---|---|
| `schema_version` | String equal to `dense-union-51.reviewer-label.v1` |
| `packet_id` | String equal to `dense-union-51-image-only-review-v1` |
| `review_identifier` | Immutable identifier copied from the assigned queue row |
| `reviewer_role_identifier` | `reviewer-one` or `reviewer-two`, matching the queue role slot |
| `reviewer_local_object_identifier` | Canonical immutable identifier assigned after within-image sorting |
| `image_id` | Integer image identifier copied from the queue |
| `normalized_category_name` | One canonical ontology name for `accepted`, `partial`, or `crowd`; otherwise null unless `ambiguous` has exactly one plausible category |
| `official_coco_category_id` | Official gapped ontology identifier paired with `normalized_category_name`, or null under the same rule |
| `candidate_categories` | Sorted list of objects with `normalized_category_name` and `official_coco_category_id`; nonempty for `ambiguous`, empty otherwise |
| `source_canvas_box_xyxy` | Four source-canvas integers, or the single allowed `partial` null value |
| `reviewer_state` | One frozen reviewer state |
| `reason_code` | One reviewer-allowed reason code compatible with the state |

For `ambiguous`, `candidate_categories` contains every plausible category,
sorted first by `official_coco_category_id` and then by
`normalized_category_name`. For `out-of-scope`, all category fields are null or
empty. A record may not contain comments, explanations, scores, uncertainty
numbers, aliases, or additional fields.

## Deterministic identifiers, ordering, and output

The reviewer must first complete the candidate set for one image, then
canonicalize it. Within an image, sort candidates by:

1. boxed records before null-box records;
2. `y1`, `x1`, `y2`, and `x2`, using positive infinity for a null box;
3. reviewer-state order `accepted`, `ambiguous`, `partial`, `crowd`,
   `out-of-scope`;
4. `official_coco_category_id`, using positive infinity for null;
5. the canonical serialized `candidate_categories` list.

Two candidates with an identical complete sort key are a duplicate-label error
and must be resolved before output. After sorting, assign zero-padded one-based
identifiers:

```text
<reviewer-role-identifier>:<image-id>:<four-digit-ordinal>
```

Example: `reviewer-one:139:0001`.

Each role output uses Unicode Transformation Format 8-bit (`UTF-8`) and
`JSONL`, one object per line, with lexicographically sorted keys, compact
separators, and a final newline. Records follow frozen image order and then
`reviewer_local_object_identifier`. Unknown fields and duplicate identifiers
are fatal. The role artifact must be written once; a correction requires a new
packet and artifact version rather than in-place editing.

## Completion boundary

A reviewer role is complete only when all 51 assigned source canvases have a
validated queue disposition and its canonical role-label artifact has been
sealed by digest. Reviewer completion does not accept additions. Only later,
blinded adjudication may join the two role artifacts with official individuals
and official crowd regions. Reviewers must not perform that join themselves.
