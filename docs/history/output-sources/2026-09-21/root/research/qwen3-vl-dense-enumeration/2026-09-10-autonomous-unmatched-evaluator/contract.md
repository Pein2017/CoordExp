# Autonomous unmatched evaluator: frozen development / holdout boundary

2026-09-10. User authorized autonomous exploration and available GPUs; no CLI
product, external paid API, GT overwrite, detector training, or runtime restart.
The current 54 visually reviewed candidates are development, never validation.

Question: can an automatic image-based method separate supported entity/category,
acceptable single-instance localization, and uncertainty on real unmatched
detector candidates from new images, without routine Codex image inspection?
Strongest alternative is candidate/prompt agreement bias rather than visual
discrimination. Independent re-grounding tests this without candidate box edges
or category in the model prompt; geometry-only agreement is not semantic proof.

Holdout v1 is frozen before method results are read: seed 2026091001, sample
64 distinct images uniformly from the 91 source images outside the 39 development
images, then one uniformly selected non-strict-repeat unmatched candidate per
image. No protected confirmation512 is opened. Original source predictions and
GT remain immutable. Scope is this source checkpoint/cohort, not all COCO.

One-time blinded visual reference collection is allowed for measuring errors;
it is not the deployed evaluator. Reviewers see source pixels, candidate box
and category, but no GT or proposed evaluator answers. Reference distinguishes
entity support, category support, single-instance visible extent and uncertainty.
These provisional visual judgments are not new GT. Uncertain is not negative.

Freeze all algorithm/prompt/threshold choices on development before applying
them to holdout. Do not retune from holdout errors or use a failing holdout as
development while retaining a test claim. Report raw confusion counts, accepted
uncertain cases, clean-candidate retention, total decision coverage, latency,
allocated GPU-hours and exact artifact identities.

Rough-use promotion criterion (chosen before holdout labeling or predictions):
selective full-row acceptance precision >= 90% among definite reference labels,
at least 10 definite accepts, retention >= 25% of definite clean candidates,
total acceptance coverage >= 15%, and no more than 20% of accepted cases with
uncertain reference. Report conservative precision treating uncertain accepted
cases as unverified too; do not advertise a 90% population guarantee from 64
cases. Hot amortized latency <= 5 seconds per candidate at batch >= 8 and cold
initialization <= 180 seconds on one local GPU. Rejection/repair is separately
measured and never silently promoted from acceptance results. Entity-only
support may be reported separately but does not establish good localization.

Invalid schema, unresolved objects, category synonym disagreement, crop-boundary
truncation, and geometric disagreement remain unknown, not accepted negatives.
Strict IoU > .95 repetition is an existing automatic rule and needs no visual
review; weaker same-instance duplication is outside re-grounding's evidence
unless an explicit previous-prediction comparison is provided. No automatic GT
mutation, hard reward labels, or keeping a GPU service resident after a run.

Each local method batch is bounded by 30 minutes / 200 responses, with raw output
and process exit code preserved. Stop a branch when its fixed development rule
fails meaningful precision or coverage, rather than tuning indefinitely. Current
branch initially: two-view 8B point re-grounding, 108 development responses,
fixed IoU .75. That branch failed development precision/coverage and is closed.

## Co-DETR continuation selected after user offer

The user offered an already-working Co-DETR checkpoint and inference script
and explicitly delegated the choice. The lead selects that shortest existing
path: stop new Grounding DINO downloads and stop Source2B crop regeneration,
preserving their partial artifacts. Do not open further parallel model routes.

Use `/data/CoordExp/external/Co-DETR`, ViT-L COCO checkpoint
`models/co_dino_5scale_vit_large_coco.pth`, matching official config
`projects/configs/co_dino_vit/co_dino_5scale_vit_large_coco.py`, and existing
`tools/codetr_infer_human_refined12.py` model and prediction-collection functions.
The named runtime is Conda `mmdet`, verified torch1.11.0/mmcv1.5.0/mmdet2.25.3;
do not change the shared environment or external checkout.

First fixed rule, before development outputs: same-category Co-DETR score
>=.50 and candidate IoU>=.70 -> supported full-row candidate; otherwise unknown.
Separately measure entity/category support at IoU>=.25 and localization
agreement. Category conflicts remain visible. No candidate class, coordinates,
GT, or reference labels are supplied to Co-DETR; full-image predictions are
shared across candidates on the same image. Co-DETR was trained for COCO and
can reproduce annotation conventions or omissions; AP and non-detection do
not establish the truth of unmatched proposals. Freeze the final choice on
development before holdout predictions or reference comparison. Original
promotion criteria and 64-image holdout remain unchanged.
