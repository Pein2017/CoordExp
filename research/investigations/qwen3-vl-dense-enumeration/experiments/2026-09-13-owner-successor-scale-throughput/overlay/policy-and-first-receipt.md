# COCO-80 reviewed positive overlay

## User rulings

The user permits dense small-instance ambiguity to be downgraded, particularly
books, fruit, vegetables and cup-like objects. This is conditional on difficult
visual separation, not a blanket class exemption. A coherent same-class cluster
can be valid group coverage; clearly distinguishable individual instances retain
their own discovery obligation. Group and child instances must not be added as
independent atomic gains. A child can refine a group; emitting the same group
again is not exempt from the unchanged strict duplicate rule.

Only COCO-80 categories are in scope. Generic fruit/vegetable and unsupported
nearby categories are not added. The current supply user prompt already names
exactly the canonical80 in the canonical order, with no missing/extra classes.
Root verified against src/eval/detection_categories.py; prompt SHA256:
9af434aca876a872b8fc80e884480d9fcd135f301d3bb67a36bd0b754cbc3df8.
No generation prompt was changed during the running experiment.

Group policy is supplemental annotation scope, not a retroactive change to the
frozen singleton c/w training bank or raw COCO benchmark. The reviewed39654
banana box is useful group evidence, not an invalid object simply because it
contains several bananas. It is already GT-supported and therefore is not
counted as a newly discovered unlabeled owner.

## First actual root review and export

Root inspected every exact OVR-C01 through OVR-C10 context/crop with view_image.
These are AI visual reviews, not human annotations. Decisions are preserved in
root-review-v1.json and a generated append-only event file in the output root.

-5 distinct unlabeled donuts on417044 accepted as singleton positive records.
-5 proposals retain uncertainty: one partial donut box, one pastry subtype/
 extent, one plush rabbit-like toy not automatically mapped to teddy bear, and
 two dense book-spine regions without reliable singleton/group boundaries.
-0 negative labels,0 unreviewed proposals,0 accepted groups in this small
 candidate slice. Resolved object existence can survive unresolved class/extent.
- All10 literal coordinate-bin lists matched their original generated rows.
 Root checked16 distinct image/source/checkpoint/output/card file bindings.

Output root:
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-13-owner-successor-scale-throughput/overlay/

The immutable positive-owner-patch-v1.json SHA256 is
17970413fdefd381bfed8f6ac4ebadbd3cf76ab8c5eb5f2d7379b89d39c65745.
The root-review-events-v1.jsonl SHA256 is
f69ce19652a626eb4f04424f928e670bc4e3c86bfb6d350b0904e4e927cfb576.
This is a positive-only, selection-conditioned supplemental layer, not an
exhaustive benchmark or an authorized training export. Original raw labels and
image-level train/evaluation roles are unchanged.

Root found and reproduced an explicit-alias double-counting path and an absent
confirmation-selection guard. The same Luna owner repaired both and added the
COCO-80 export gate. Root reran11 focused tests and the real five-positive
export. No worker-only review became a training target.
