---
title: Annotation Review Provenance and Interpretation Correction
description: Corrects the executed provenance and scientific interpretation of the first image-only review pass without modifying the sealed artifacts.
type: investigation
role: evidence-correction
authority: non_normative_research
unit_id: 2026-07-13-spatial-scope-history-disentanglement
status: active_correction
updated: 2026-07-18
---

# Annotation Review Provenance and Interpretation Correction

## Why This Correction Exists

The pre-execution protocol described two independent image-only reviewers and a
local annotation tool. A later reconstruction of the executed first review pass
established a narrower provenance:

- Independent Reviewer One was `gpt-5.6-sol` with `xhigh` reasoning effort.
- It inspected the original images directly through image viewing.
- It did not use a detector, an annotation user interface, a ground-truth
  overlay, or an external annotation model.
- The model proposed the categories and bounding-box coordinates. Local code
  only serialized, saved, and validated those proposals.

The executed artifact is:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-13-spatial-scope-history-disentanglement/readiness-v1/reviewer-one-labels.jsonl
```

It contains 479 records over 51 images and has Secure Hash Algorithm 256-bit
digest:

```text
d9149f023308e1e407ed54a6ff6693ba889e413abc753234044cb1dd38d90a52
```

The matching `readiness-v2` copy is byte-identical. The artifact itself records
reviewer-role and packet identifiers, but it does not encode the annotator model,
reasoning effort, or executed viewing mechanism. Those details therefore come
from the reconstructed execution record rather than from the artifact schema.

## Correct Evidentiary Status

Reviewer One's output is a model-generated, image-only, high-recall candidate
annotation pass. It is not human-confirmed ground truth. The sealed ledger and
all derived metrics remain mechanically reproducible relative to that ledger,
but their absolute semantic validity depends on later human confirmation.

The existing artifacts must not be edited in place. Any human correction or
replacement reference set must receive a new ledger version and preserve the
old ledger as provenance.

The serialized metric key `manual_precision` is retained only for artifact
compatibility. In current interpretation it means **audit-ledger precision**,
not precision against a fully human-produced reference set.

## Required Two-Axis Review

An official Common Objects in Context ground-truth mismatch is not sufficient
evidence of hallucination. Every candidate must be judged on two separate axes.

### Entity and category axis

1. **Unlabeled true positive**: a real Common Objects in Context 80-category
   entity omitted by the accepted reference set.
2. **Duplicate**: the same physical entity is already represented by an
   accepted reference object.
3. **Semantic error**: a real entity is assigned the wrong category.
4. **Entity hallucination**: no corresponding physical entity exists.
5. **Uncertain**: the image does not support a reliable decision.

### Geometry axis

1. **Acceptable geometry**: the box is a usable estimate of the entity's visible
   physical extent.
2. **Localization error**: the box is shifted, oversized, undersized, or
   incomplete.
3. **Instance-binding or neighbor-contamination error**: the box mixes adjacent
   entities, falls between instances, or assigns boundaries from different
   physical owners.
4. **Uncertain**: the visible extent cannot be judged reliably.

An imperfect unlabeled true positive is represented by the cross-product
`unlabeled true positive` plus `localization error` or
`instance-binding or neighbor-contamination error`. Entity discovery and box
quality must not be collapsed into one pass/fail label.

Cross-category overlap is a trigger for crop-enlarged visual inspection, not an
automatic rejection. Dense scenes contain real occlusion and overlapping
objects.

## Example and Limitation

Image `139` illustrates why official mismatch is insufficient: Reviewer One
proposed multiple potted plants while the official reference set contains fewer
plant instances, and later image inspection supports real omitted plant
entities. Image `12120` similarly contains many real people for which broad
crowd or imperfect individual regions may be semantically meaningful while
remaining unsuitable as exact instance boxes.

A side review further reported an image-`139` plant proposal whose box was
displaced or oversized and contaminated by a nearby person. That statement is
scientifically plausible and consistent with the two-axis taxonomy, but this
correction does not promote it as an artifact-level fact until it is linked to
an exact prediction or candidate identifier.

## Consequence for Mechanism Analysis

The working interpretation must distinguish:

```text
object not discovered
```

from:

```text
object discovered, but category or physical extent recovered imperfectly
```

The latter is evidence for a semantic-to-geometry or instance-binding failure,
not a pure object-discovery failure. Analyses should therefore report at least:

- whether each physical entity was discovered at least once;
- category consistency across rollouts;
- error for each of `x1`, `y1`, `x2`, and `y2`;
- center and box-size error;
- contamination by neighboring instances; and
- agreement between semantic identity and geometry ownership.

Reviewer-generated annotations remain useful for finding high-recall candidate
objects. They must not be promoted to ground truth without human confirmation.

## Relationship to Historical Files

The [readiness amendment](readiness-amendment.md),
[reviewer instruction packet](reviewer-instruction-packet-v1.md), and historical
independent reviews preserve what the experiment intended and what reviewers
believed at the time. This correction records the later provenance
reconstruction and supersedes their interpretation of Reviewer One as a human
annotation source; it does not rewrite the frozen pre-execution protocol.
