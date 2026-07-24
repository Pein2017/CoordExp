---
packet_id: trajectory-owner-set-image-only-review-v1
packet_full_name: Trajectory Owner-Set Image-Only Independent Review Packet, version 1
review_tool_id: coordexp-trajectory-owner-set-review-jsonl
review_tool_full_name: CoordExp Trajectory Owner-Set Image-Only Review JSONL Tool
review_tool_version: 1.0.0
language: English
---

# Trajectory Owner-Set Image-Only Independent Review Packet, Version 1

## Operational purpose

This packet governs two independent, route-blind reviews of source-canvas
images selected by the frozen-panel owner-ledger salvage gate. Each reviewer
must enumerate every visually separable instance in the frozen Common Objects
in Context 80-category ontology (`COCO-80`). The work constructs an image-first
physical-owner reference. It is not model-output assessment.

The immutable packet identifier is
`trajectory-owner-set-image-only-review-v1`. The review tool identifier is
`coordexp-trajectory-owner-set-review-jsonl`, version `1.0.0`. Any semantic
change requires a new packet, schema registry, review queue, and artifact
version. The implementation must fail closed on unknown fields, unknown
states or reasons, invalid categories or coordinates, noncanonical ordering,
wrong role, missing image dispositions, source-image drift, packet or ontology
drift, queue drift, or forbidden evidence.

## Reviewer roles, independence, and route blindness

The only role identifiers are:

- `reviewer-one`: Independent Reviewer One;
- `reviewer-two`: Independent Reviewer Two.

Each role works in an independent context and writes a separate JavaScript
Object Notation Lines (`JSONL`) artifact. Neither role may inspect or receive
the other role's labels. Neither role may adjudicate. Both role artifacts must
be complete and sealed before either is exposed to adjudication.

For one assigned image, a reviewer may receive only:

1. the complete, unmodified source-canvas image;
2. image identifier, byte SHA-256 digest, width, and height;
3. assigned reviewer role and review identifier;
4. this packet and its SHA-256 digest; and
5. the frozen `COCO-80` ontology and its SHA-256 digest.

## Forbidden evidence

A reviewer must not receive or use:

- official boxes, official counts, official annotation identifiers, or an
  official per-image category summary;
- model predictions, candidate or route identifiers, decode modes, token data,
  parser evidence, owner sets, edges, frontier labels, census labels, or
  category-capacity witnesses;
- any selection, ordering, grouping, statistical-analysis, prior-work,
  future-work, downstream-processing, or decision information;
- crops, tiles, overlays, masks, outlines, resized or enhanced images, or any
  derived visual input;
- the other reviewer's work; or
- adjudication, owner-ledger, replay, or outcome artifacts.

Exposure invalidates that role artifact. Record the failure outside the role
artifact and stop that pass.

## Source-canvas-only policy

The tool must verify the queue image path, byte digest, width, and height before
display. Display the complete source canvas without crop, resize,
interpolation, masking, marking, or color transformation. Coordinates are
integer pixel edges in this exact canvas:

```text
0 <= x1 < x2 <= source_image_width
0 <= y1 < y2 <= source_image_height
```

Review the full canvas, including image borders and overlap regions, before
closing its disposition.

## Review procedure

For each assigned queue row:

1. Verify the review identifier, role, image identifier and digest, dimensions,
   packet digest, ontology digest, and queue digest.
2. Inspect the full canvas systematically, followed by a second omission pass.
3. Enumerate every visually separable `COCO-80` instance. There is no minimum
   size or visible-percentage threshold.
4. Assign exactly one state and one compatible reason to every label.
5. Use only canonical ontology category names and official category identifiers.
6. Draw the smallest reproducible source-canvas pixel-edge box containing the
   visible evidence when a box is required.
7. Check for duplicated labels, missed small objects, overlapping distinct
   instances, edge truncation, and unresolved groups.
8. Canonicalize the label list, set the image disposition to `complete`, and
   validate the whole role artifact before sealing.

An image with no reportable or salient out-of-scope label still requires one
explicit `complete` image-disposition row with an empty label list. A missing
row is never an empty-image disposition.

## Instance and visible-extent rule

An instance is individually reportable when category and physical identity are
visually distinguishable and the reviewer can draw a reproducible tight box
around visible evidence. Occlusion and image-edge truncation do not prevent
acceptance when those conditions remain satisfied. Never extrapolate a fully
occluded extent.

Overlapping objects remain separate when distinct visible evidence supports
separate identities. When multiple same-category instances cannot be
individuated reliably, emit one `crowd` group region instead of speculative
individual boxes.

## Reviewer states and reasons

- `accepted`: one distinguishable `COCO-80` instance, one unique category, and
  one reproducible visible-extent box. Reasons: `none`,
  `occluded_but_boxable`, or `truncated_but_boxable`.
- `ambiguous`: object evidence exists but category or instance identity is not
  unique. Reasons: `category_not_unique` or `instance_not_separable`.
- `partial`: a category or instance is plausible, but a reproducible individual
  box is unavailable. Reasons: `boundary_not_reproducible` or
  `instance_not_separable`.
- `crowd`: multiple same-category instances are visible but cannot be
  individuated. Reason: `instance_not_separable`.
- `out-of-scope`: salient object-like content considered during the scan is
  outside `COCO-80`. Reason: `non_coco80`.

`out-of-scope` is an audit trace, not an exhaustive inventory of backgrounds,
textures, or scene elements.

Every label has a four-integer source-canvas `xyxy` box except `partial` with
`boundary_not_reproducible`, for which the box may be null. `accepted`,
`partial`, and `crowd` have one canonical category. `ambiguous` has a sorted,
nonempty list of every plausible canonical category; its scalar category is
present only when the candidate list has length one. `out-of-scope` has no
category. Free text, scores, rationales, aliases, and extra fields are
forbidden.

## Role-artifact schema and ordering

Each role artifact contains exactly one row per assigned image. Reviewer One
uses schema `trajectory_owner_set_review.reviewer-one.v1`; Reviewer Two uses
`trajectory_owner_set_review.reviewer-two.v1`. A row contains exactly:

- `schema_version`, `packet_id`, `packet_sha256`, `ontology_sha256`, and
  `review_queue_sha256`;
- `review_identifier`, `reviewer_role_identifier`, `image_id`, `image_sha256`,
  `source_image_width`, and `source_image_height` copied from the assigned row;
- `image_disposition`, exactly `complete`; and
- `labels`, the canonical list of zero or more label objects.

Each label contains exactly:

- `reviewer_local_object_identifier`;
- `normalized_category_name` and `official_coco_category_id`;
- `candidate_categories`, a sorted list of exact category-name and official-ID
  pairs;
- `source_canvas_box_xyxy`;
- `reviewer_state`; and
- `reason_code`.

Within an image, sort labels by boxed before null-boxed; `y1`, `x1`, `y2`,
`x2`; state order `accepted`, `ambiguous`, `partial`, `crowd`,
`out-of-scope`; official category identifier with null last; and canonical
candidate-category serialization. Duplicate complete sort keys are invalid.
Assign one-based identifiers after sorting:

```text
<reviewer-role-identifier>:<image-id>:<four-digit-ordinal>
```

Role rows follow the reviewer queue's canonical numeric image order. JSONL uses
UTF-8, canonical lexicographically sorted object keys, compact separators, and
a final newline.

## Seal, adjudication, and owner-ledger boundary

A role is sealed only when its artifact has exactly one valid disposition for
every queue image and its SHA-256 digest is bound to the queue, packet,
ontology, role, image count, and label count. Adjudication requires both
independent seals and exact role artifacts.

Only the separate route-blind adjudicator may then receive both role artifacts
and the official owner ledger. Official owner identifiers, categories, and
boxes are immutable. Reviewer proposals already owned by an official
individual are `official_duplicate`, not new owners. Multiple reviewer labels
for one proposal may be reconciled, but every label must receive exactly one
disposition. The only addable state is an accepted missing physical owner with
one canonical category and one source-canvas box. The sealed owner ledger is
therefore add-only.

Adjudication must retain `ambiguous`, `partial`, `crowd`,
`unresolved_disagreement`, and any uncertainty that owner enumeration is
complete. An unmatched official annotation is never automatically a
hallucination. Missing dispositions, invalid geometry, disagreement, or
adjudication failure cannot establish ledger completeness.

The adjudicator is blind to the private queue manifest and to all selection,
ordering, grouping, statistical-analysis, prior-work, future-work,
downstream-processing, or decision information. Adjudication operates on the
assigned image set only. Both role artifacts must be sealed before its queue is
assembled, and the assigned owner ledger and uncertainty ledger must be sealed
before any downstream processing begins.
