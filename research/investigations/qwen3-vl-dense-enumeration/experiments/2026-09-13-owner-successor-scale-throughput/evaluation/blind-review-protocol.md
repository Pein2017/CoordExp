# Frozen blind32 physical-owner review protocol

Current status,2026-09-14: **all32images/607proposals reviewed, validated and
source-joined; root independently reproduced the physical accounting.**
The [completed result](../../2026-09-14-label-vs-compilation/physical/result.md)
owns outcomes and uncertainty. The original preparation remains immutable,
including its historical0-label status before dispatch.

## Fixed population and source boundary

This route consumes only the paired evaluator's already frozen confirmation
blind32 IDs. It may not select, backfill, reorder, or relabel an image role.
The input is the future paired consumer's anonymous mixed N16/A/B queue and its
separate sealed source map.

Preparation reads only the anonymous queue. It does **not** read the source
map. Each reviewer batch omits source arm, source prediction chronology, source
map, model outcome, and GT-negative fields. The source map is joined only after
all exact proposal decisions have been saved.

## Rendering and batching

`blind_review.py prepare` verifies 32 unique image IDs and partitions them in
queue order into exactly eight batches of four. For every image it provides:

- the literal original frame path, SHA256, and native pixel dimensions;
- one native-resolution overlay per exact candidate;
- at most one rectangle on each overlay;
- one shared overlay only when proposals have literal-identical `desc`, bbox,
  and bbox format. This is a transport alias, never an IoU-based physical
  identity inference.

There is no resize and no multi-sample collage. A reviewer uses one image per
`view_image` call with `detail=original`; inspect the original and individual
candidate overlays needed for small or crowded instances. Record only paths
actually viewed and their hashes. Availability in a batch does not prove it
was viewed. Comparison accepts a viewed path/SHA pair only when the frozen
preparation manifest and its hash-bound batch assign that exact original or
overlay to the decision's image; a view from another image is rejected.

Root may assign at most eight Luna workers, one four-image batch each. Within a
batch, save one image decision line to `decisions.jsonl` **before** viewing the
next image. No worker owns another batch or the final source join.

```bash
env PYTHONPATH=/data/CoordExp/.worktrees/research-probes${PYTHONPATH:+:$PYTHONPATH} \
  python research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-13-owner-successor-scale-throughput/evaluation/blind_review.py prepare \
  --queue <PAIRED_CONSUMER>/blind-review-queue.jsonl \
  --output <PAIRED_CONSUMER>/blind32-review-preparation-v1
```

## Per-image decision schema

Each JSONL line contains the exact `image_id`, `review_id`, reviewer identity,
and nonempty `viewed` list with `{path, sha256, detail:"original"}`. It assigns
every proposal ID exactly once among:

```json
{
  "image_id": 123,
  "review_id": "owner-successor-paired:123",
  "reviewer": "Luna worker name",
  "saved_before_next_view_attestation": "reviewer_attests_decision_saved_before_next_view",
  "viewed": [{"path": "/actual/viewed/path.png", "sha256": "...", "detail": "original"}],
  "owners": [
    {"owner_id": "O01", "proposal_ids": ["..."], "extent_or_class_caveats": []}
  ],
  "group_coverage": [
    {"group_id": "G01", "proposal_ids": ["..."], "extent_or_class_caveats": ["dense coherent group"]}
  ],
  "unresolved": [
    {"proposal_ids": ["..."], "axes": ["entity", "extent"], "image_grounded_reason": "specific visible ambiguity"}
  ],
  "non_owner_evidence": [
    {"proposal_ids": ["..."], "status": "unsupported", "image_grounded_reason": "specific evidence from the viewed image"}
  ]
}
```

`saved_before_next_view_attestation` is the reviewer's workflow attestation.
It is not machine proof of `view_image` call timing; the comparator verifies
the saved value and exact image-bound viewed artifacts, not an external tool
timeline.

`non_owner_evidence` may be empty. If used, status is exactly `unsupported` or
`extent_mismatch`; an image-grounded reason is mandatory. It is not an
automatic negative or hallucination label.

## Physical decision policy

- Scope is canonical COCO80-compatible objects. Each anonymous proposal has
  exactly one category, but reviewed physical-owner presence is class-agnostic;
  class and extent doubts remain explicit caveats. It is not strict box TP.
- Clearly separable physical instances are atomic owners. One within-image
  owner ID may bind multiple boxes/classes judged to be aliases of that owner.
- A coherent dense group may be valid `group_coverage`, but it is never an
  atomic gain and is never double-counted with visible children.
- Entity grounding, class, and visible extent are separate uncertainty axes.
  Unresolved proposals stay neutral.
- Missing raw GT does not imply absence, hallucination, or negative evidence.
  Raw GT and confirmation/training roles remain unchanged.

Literal-identical transport aliases must be assigned together; the validator
rejects splitting them across owners/groups/unresolved. Visually judged aliases
with different literal boxes remain separate proposal IDs under one reviewed
owner entry, preserving their exact source-row denominators.

## Post-review exact join

After all eight decision files are frozen, run:

```bash
env PYTHONPATH=/data/CoordExp/.worktrees/research-probes${PYTHONPATH:+:$PYTHONPATH} \
  python research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-13-owner-successor-scale-throughput/evaluation/blind_review.py compare \
  --queue <PAIRED_CONSUMER>/blind-review-queue.jsonl \
  --preparation <PAIRED_CONSUMER>/blind32-review-preparation-v1/manifest.json \
  --source-map <PAIRED_CONSUMER>/blind-review-source-map.json \
  --review <BATCH01>/decisions.jsonl --review <BATCH02>/decisions.jsonl \
  --review <BATCH03>/decisions.jsonl --review <BATCH04>/decisions.jsonl \
  --review <BATCH05>/decisions.jsonl --review <BATCH06>/decisions.jsonl \
  --review <BATCH07>/decisions.jsonl --review <BATCH08>/decisions.jsonl \
  --output <PAIRED_CONSUMER>/blind-review-comparison-v1.json
```

The join rejects image/proposal omission, duplication, literal-alias splitting,
unknown source IDs, duplicate source rows, changed viewed-file hashes,
cross-image or foreign viewed artifacts, or unsupported uncertainty/negative
statuses. It reports exact queue proposal,
source-row, per-arm proposal, atomic-owner, group, unresolved, and non-owner
denominators. Atomic physical-owner gained/retained/lost sets are computed for
N16→A, N16→B, and A→B across all 32 images. Group presence and uncertain burden
remain separate and cannot change atomic net gains.

No comparison automatically promotes a checkpoint, changes raw annotations,
or establishes exhaustive recall.

## CPU acceptance

```text
pytest -q research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-13-owner-successor-scale-throughput/evaluation/test_blind_review.py
5 passed
```

Synthetic fixtures exercise only transport invariants; they are not fabricated
model endpoints or scientific outcomes. The real paired queue and rendering
now exist; do not rerender or repeat inference when collecting these decisions.
