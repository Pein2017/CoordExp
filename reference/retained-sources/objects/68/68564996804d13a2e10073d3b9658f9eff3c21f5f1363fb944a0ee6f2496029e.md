# Stage01 support seed evidence brief

Status: candidate extraction complete; lead visual bbox+crop review is still required. No model, GPU, new image, new annotation, or training action was performed.

The fixed N16 original-positive cohort selected by the lead from the existing positive bank is covered in the declared order: 11 images. Six rows come from the historical transformed train partition and five from its historical transformed dev partition; all are COCO2017 train images and existing N16 learning cases, not a new validation split or validation role. The bound transformed inputs contain 169 atomic GT rows. The original COCO instances census contains 175 owners: 169 atomic and 6 crowd/group. All six omitted transformed rows are retained with `is_crowd=true`; their boxes have explicit original-pixel provenance and a labeled derived 1000-canvas normalization, without asserting training targets.

The already accepted package/review records bind 16 candidate supports across all 11 images: 15 GT owners and 1 reviewed non-GT physical owner (417044:review:P6). Every candidate carries its acquisition row hash, exact record ID, review decision hash, evidence paths, description, and 1000-canvas bbox. Candidate support is not final target-set admission. Existing review observations retain clipping, overlap, extent, and class/identity uncertainty; no uncertain owner was invented or merged.

Checks: exact ordered 11-image coverage; all source paths exist; GT source IDs are unique; candidate owner IDs are unique; counts are deterministic. The ledger preserves atomic/group distinction and records class errors separately from owner coverage. Next action is root-owned single-sample bbox+crop review, followed by target-set admission only for supports that pass that review.
