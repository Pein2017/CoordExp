# Fixed visual FP audit

## Superseding user ruling

The user subsequently ruled that previous-prediction IoU>.95 is sufficient
to classify repetition without visual review. Therefore all484 strict-repeat
FPs are automatically classified and excluded from the active visual task.
The10 originally sampled strict repeats are not visual-distribution evidence.
Continue only the existing54 non-repeat samples from610 non-repeat FPs,
without resampling. Extra capped-image-only viewing is cancelled. Original
packet/sample files remain unchanged provenance; `active-review-scope.json`
owns the amended IDs and denominator. The three worker batches are unchanged;
root skips the10 strict-repeat entries in its original batch.

Question: which types of visual evidence underlie annotation-relative FP50
after Source256 RLOO round1? No training, relabeling of the official dataset,
model calls, source/config changes or broad FP-negative supervision follows.

Population: all1094 globally unmatched predictions on256 train images /1955 GT.
The deterministic structural strata are sampling devices, not visual labels:
strict previous-prediction pixel IoU>.95; otherwise same-category GT IoU>=.1;
otherwise any-category GT IoU>=.5; otherwise weak GT relation. Uniform seeded
sampling without replacement gives10/20/14/20 examples, total64; the14-example
other-category high-overlap stratum is a census. Samples and image ownership
are frozen in `sample.jsonl` and `batches/*.jsonl`, subject to the superseding
strict-repeat exclusion above. Do not drop, replace or reweight ambiguous
non-repeat cases.

For each assigned image, actually view its original `image_path` or native
overview before the target crop. View every assigned `crop_path`; if needed,
make an additional deterministic pixel crop inside your own reviewer directory
and view it. No generated/redrawn image can serve as evidence. The provided
crop shows untouched context at left and target red / GT blue at right;
magenta denotes a previous highly overlapping prediction. Native overview
matching colors are reference-only: global matching owns sample membership.
Read category and bbox metadata, but do not infer visual truth from matching
or overlap alone. Context plus zoom must distinguish a different nearby instance,
an object part, a box spanning multiple instances and missing annotation.

Write one JSON object per sampled prediction, with these fields:

- `case_id`, `reviewer`.
- `primary_reason`: `duplicate_prediction`, `unlabeled_real_instance`,
  `localization_error`, `category_error`, `multi_instance_box`,
  `annotation_extent_or_grouping`, `hallucinated_entity`, or `uncertain`.
- `entity`: `real_single`, `real_multiple`, `no_supported_entity`, or `uncertain`.
- `category`: `correct`, `incorrect`, or `uncertain`.
- `geometry`: `acceptable`, `too_tight`, `too_loose`, `shifted`,
  `multi_instance`, or `uncertain`.
- `confidence`: `high`, `medium`, or `low`.
- `evidence`: one or two concise sentences describing visible facts, and why
  the primary reason was selected. State ambiguity rather than inventing detail.
- `viewed_paths`: absolute paths of images actually inspected.

`hallucinated_entity` requires positive visual grounds for no corresponding
entity in that region, not just absence of a GT match. `unlabeled_real_instance`
is a visual-review candidate, not a new certified GT. A wrong category on a
real object is different from hallucinating an entity. If a clear repeated
prediction has an additional underlying localization/category problem, use
`duplicate_prediction` as its primary reason and retain the other problem on
the independent axes; do not count one item twice in the primary distribution.
When a single box groups several visible instances, record that distinctly
from an unsupported object. Uncertainty must remain in the denominator.

Each worker owns only its assigned review JSONL and optional extra crops, not
another worker's rows. Root integrates once, checks complete IDs and evidence
paths, and visually tests decision-changing uncertain/hallucination claims.
All judgments remain model-generated, bounded review evidence; neither the
FP count nor official labels are rewritten. Stop after the assigned non-repeat
rows. No additional visual check is required to accept the strict-repeat rule.
