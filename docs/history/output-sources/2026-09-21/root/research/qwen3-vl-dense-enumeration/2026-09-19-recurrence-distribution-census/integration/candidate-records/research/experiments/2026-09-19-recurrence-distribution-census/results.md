# Lane A candidate results: recurrence distribution census

Status: `candidate`. Root owns scientific acceptance and any successor decision.

The estimand is an image unit descriptive distribution in two declared strata:
the accepted 145-image mature package (580 outputs, four tied/untied and
original/normalized conditions) and a deterministic prospective 128-image
seed-19 cohort (256 original-policy outputs, 128 images per model). The new
cohort was sampled from the actual processed length-12000 manifest after
split-qualified prior exclusions. It estimates that eligible processed
population, not COCO generally. Tied versus untied remains a package
comparison, and the original and normalized policies are kept separate.

## Source, denominators and accounting

The processed source contains 122,218 records. Excluding 525 prior identities
left 121,693 eligible records; the frozen sample has 122 train and 6 val
images. Identity is `(metadata.split, image_id)` with file name, canonical
path and image SHA-256 cross-checks. The audit found no aliases or conflicts.
The final mechanism panel is a separate fixed sample: 25 carried boundaries
and 20 deterministic prospective additions, with 21 failure and 24
non-recurrent proxy boundaries. It is not a census of recurrence.

The primary recurrence predicates are literal exact repeat rows and existing
`<=8`-bin same-description near-repeat rows. A repeat-row count and an
all-pairs count answer different questions: a run of length `r` contributes
`r choose 2` pair edges. Invalid geometry, malformed rows, cap/EOS and
endpoint outcomes remain numerical outcomes. The fixed all-complete-row
indexing binds category, region and size attribution to the same parsed row
indices as recurrence accounting; invalid rows are counted in their own
category fields rather than dropped or shifted into another category.

| condition | images | complete rows | valid / invalid / malformed | exact exposure | exact rows / pair edges | near exposure | near rows / pair edges |
|---|---:|---:|---:|---:|---:|---:|---:|
| mature tied-original | 145 | 2,969 | 2,291 / 678 / 5 | 9 (6.2%, CI 2.8–10.3%) | 1,676 / 154,328 | 14 (9.7%, CI 4.8–14.5%) | 1,759 / 157,686 |
| mature tied-normalized | 145 | 2,218 | 1,627 / 591 / 3 | 6 (4.1%, CI 1.4–7.6%) | 1,019 / 101,948 | 10 (6.9%, CI 2.8–11.0%) | 1,057 / 120,060 |
| mature untied-original | 145 | 2,607 | 1,786 / 821 / 4 | 9 (6.2%, CI 2.8–10.3%) | 1,358 / 139,487 | 11 (7.6%, CI 3.4–12.4%) | 1,431 / 140,117 |
| mature untied-normalized | 145 | 1,313 | 1,302 / 11 / 0 | 4 (2.8%, CI 0.7–5.5%) | 40 / 109 | 10 (6.9%, CI 3.4–11.0%) | 72 / 305 |
| seed19 tied-original | 128 | 886 | 883 / 3 / 1 | 3 (2.3%, CI 0–5.5%) | 9 / 18 | 7 (5.5%, CI 1.6–9.4%) | 38 / 82 |
| seed19 untied-original | 128 | 860 | 857 / 3 / 0 | 2 (1.6%, CI 0–3.9%) | 5 / 11 | 10 (7.8%, CI 3.1–12.5%) | 35 / 57 |

The mature original-policy row and pair totals are dominated by a few long
runs. For example, the longest exact run among mature tied-original images is
330 rows and the longest among untied-original images is 284; among images
with any exact run longer than one, the median run is 147 and 62 rows,
respectively. In the seed-19 cohort the corresponding maxima are only 5 and 4
rows, with three tied and two untied images having a run longer than one.
Thus the low image exposure and very large mature pair totals are compatible:
they describe concentrated long-run behavior, not broad recurrence across
most images.

## Sequence phase, coordinate region and size

In the mature original policy, exact recurrence starts at median row 6 for
both model packages; near recurrence starts at median row 6 for tied and row
5 for untied. In the prospective cohort, the exact onset medians are rows 17
and 18.5 (tied and untied), while near onset remains row 8 for both. These
onset summaries are conditional on the small number of exposed images (3/2
exact in the new cohort), so they are descriptive phase summaries rather than
evidence for a changed mechanism. The normalized mature conditions have fewer
exact exposures (6 and 4) and short exact runs (maxima 9 and 9).

Repeat rows are concentrated in the left-top and left-bottom coordinate-region
bins in the mature original outputs. Tied-original exact rows are
left-top 1,004, left-bottom 653 and right-bottom 19; untied-original exact
rows are left-top 965, left-bottom 387 and right-bottom 6. Near rows have the
same broad pattern. In the seed-19 cohort, tied exact rows are left-top 7,
right-bottom 1 and right-top 1; untied exact rows are left-top 4 and right-top
1. Seed-19 near rows are more distributed (tied: left-top 27, right-bottom 8,
right-top 2, left-bottom 1; untied: left-top 16, right-bottom 9, right-top 7,
left-bottom 3). The region labels describe numerical coordinate bins. They do
not identify moved visual content or establish a physical spatial cause; the
left-top concentration can reflect fixed coordinate/canvas preferences.

The area-bin result is similarly concentrated. Mature tied-original exact
rows are tiny 1,155, large 519 and small 2; untied-original exact rows are
tiny 1,087 and large 271. New exact rows are all tiny for tied (9) and
untied (5), with one additional small near row in each model. Because invalid
rows are retained and the size is decoded from the output box, this is a
numerical box-size distribution, not a claim that physical small objects cause
recurrence.

## Categories and exposure denominators

The full category table, including emitting-image denominators, valid and
invalid complete rows, exact and near image exposure, repeat rows and image
unit bootstrap intervals, is in `scientific-synthesis.json`. The most visible
mature original near-repeat exposures were book 4/5 emitting images, person
2/76 for tied and 3/77 for untied, apple 2/4 for tied, chair 1/16, bottle
1/14, bowl 1/10, bird 1/3 and carrot 1/3. In the seed-19 tied outputs the
near-exposure images were person 1/68, bottle 1/9, bowl 1/8, book 1/5,
knife 1/3, cow 1/2 and bicycle 1/1. In seed-19 untied outputs they were book
3/5, person 2/68, car 2/16, bottle 1/9, bowl 1/9, knife 1/3 and cow 1/2.
The small denominators for rare categories give broad image-unit intervals;
these tables do not support a claim that only a few semantic classes recur.

Category exposure and repeated-row volume also separate sharply. A category
can have one or a few exposed images but hundreds of rows because one image
enters a long run. The mature tied book near-repeat example contributes 440
rows from four exposed images; this is why category conclusions use exposure
rates and retain row totals as a separate quantity.

## Density, duplicate-description and small-object proxies

The saved pre-burst annotation fields are imperfect context proxies. They are
not interventions, and density computed from duplicated model outputs would be
endogenous; this report uses the saved input-side metadata only.

The high-density bin is enriched for recurrence in every original-policy
stratum, with wide image-unit uncertainty. Mature tied-original high density
has exact exposure 8/30 (26.7%, CI 13.3–43.3%) and near exposure 10/30
(33.3%, CI 16.7–50.0%), while its low bin is 0/68 for both. Mature
untied-original high density is 9/30 exact and 9/30 near (both 30.0%, CI
13.3–46.7%), while low is 0/68. In seed19, tied high density is 3/16 exact
(18.8%, CI 0–37.5%) and 4/16 near (25.0%, CI 6.3–43.8%); untied is 2/16
exact (12.5%, CI 0–31.3%) and 6/16 near (37.5%, CI 12.5–62.5%). Low-density
seed19 images have zero exact and near exposure in both models. The medium
bin is lower and uncertain: new near exposure is 3/47 tied and 4/47 untied.

Output opportunity is strongly confounded with this proxy. Mature
tied-original complete rows per image average about 76.6 in high density,
11.0 in medium and 2.2 in low; the corresponding untied averages are 67.7,
9.0 and 2.2. Seed19 averages are 22.5, 8.3 and 2.1 for tied, and 20.6, 8.3
and 2.1 for untied. The enrichment therefore supports a concentration
description, while leaving output length and annotation context unresolved.

The input duplicate-description proxy shows the same pattern. In mature
tied-original, the `6+` bin has exact exposure 9/49 (18.4%, CI 8.2–30.6%)
and near exposure 13/49 (26.5%, CI 14.3–38.8%), compared with zero exact and
near exposure in the 52-image `none` bin. Untied has 9/49 exact and 11/49
near. In seed19, tied `6+` is 3/29 exact and 5/29 near; untied is 2/29 exact
and 8/29 near. The `none` bins again have zero exposure. The `6+` groups also
have many more complete rows per image, so this is an association with a
high-opportunity context, not an owner-loss estimate.

The saved small-object-fraction proxy is weaker but directionally similar:
new tied high-fraction images have 2/18 exact and 3/18 near exposure, and new
untied high-fraction images have 1/18 exact and 5/18 near; low-fraction images
have zero exact exposure and at most one near-exposed image. The intervals are
wide and the proxy is correlated with output context, so it does not isolate
physical object size.

## Execution and qualification record

The two native producers each completed 32 groups. The valid work remained on
GPU0 (tied) and GPU1 (untied) after launch; GPUs2–7 were not assigned because
the parent directed preservation of in-flight groups rather than a kill or
duplicate rerun. Qualification was executed once per model, not once per
group: tied selected-row reconstruction maximum `5.7220458984375e-06`, untied
`7.62939453125e-06`; both passed no-op token identity, coefficient identity
and temporary-value restoration. There were 8,971 model forwards, 70 vision
forwards, 1,131.023867 GPU-seconds (`0.3141733 GPU-hours`) and 22,181,553
runtime bytes. All 64 accepted groups have `candidate_complete` receipts and
no owned process remains.

Two earlier preflight attempts are preserved as technical exits. Attempt 1
failed before model calls because `image_plan` was not bound (0.003137 s across
the tied and untied receipts); attempt 2 failed before accepted output because
prompt reconstruction width mismatched (0.094623 s across the two receipts).
They consumed setup time but zero model calls and are not scientific negative
results.

The final panel has a deterministic allocation imbalance that must remain
visible: its 20 prospective additions are 10 tied and 10 untied, while the
prospective proxy additions are 10 tied and 0 untied because sorted metadata
strata filled the remaining proxy slots. Across the full panel, failures are
9 tied and 12 untied, proxies 17 tied and 7 untied, with 23 unique image
identities. This panel is suitable for the downstream bounded diagnostics with
model/source strata retained; it is not a balanced model comparison and must
not be pooled as one.

## Reproduction and artifact integrity

The CPU synthesis was generated from saved reductions with:

```bash
PYTHONPATH=. python probes/training_set_completion/recurrence_census/synthesize.py
```

The independent raw-output replay reads `panel.json` and saved runtime raw
files, calls the existing pure cell scorer/aggregator, and writes a separate
file. It does not run selection and refuses the two frozen shared artifact
paths:

```bash
PYTHONPATH=. python probes/training_set_completion/recurrence_census/replay.py \
  --source-root /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-distribution-census \
  --output /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-distribution-census/replay/replayed-new-census.json
```

The replay matched `new-census.json` for both conditions on complete, valid,
invalid and malformed rows, exact/near repeat rows and pair edges, invalid
repeat rows, and image exposure (128 images and 256 outputs). Lane acceptance
also passes:

```bash
PYTHONPATH=. python probes/training_set_completion/recurrence_census/accept.py
```

At closeout, `shared-panel.json` is 2,006,111 bytes with SHA-256
`005bc6deff209bc96ce469e8cbc20d3dbd30a7e424a7ff18a0f7f8942a1bdb49`, and
`shared-sources.json` is 145,147 bytes with SHA-256
`8f7b43d2d9f4014bb42024324543eea27bad222b6c8432d3e44c93b45743fe9e`. These
bytes were preserved while producing this synthesis and replay.

The machine-readable result is
`scientific-synthesis.json`; source and denominator records are
`eligible-manifest.json`, `identity-audit.json`, `exclusions.json` and
`new128.runtime.jsonl`; mature and prospective reductions are
`mature-census.json` and `new-census.json`; the fixed panel is
`shared-panel.json`. The evidence supports a candidate descriptive result:
recurrence is concentrated in a small number of images and long runs, with
enrichment in high-density and duplicate-description contexts and numerical
coordinate/size bins, but the image-level denominators, output-length
confounding, invalid rows and model/source imbalance prevent a causal or
unique-category claim.

## Archival binding reconciliation

The accepted receipts bind `panel.json` SHA-256 `6b7df228...` and the
`natural.py` producer SHA-256 `5076d095...`; exact copies and the source
revision manifest are preserved under the lane output
`archival/` directory. The source history is real: the archived producer
revisions are `ff8bdbcc...` (first preflight), `27f4249b...` (second
preflight) and `5076d095...` (accepted run), so this was not a no-op source
transition.

The launched `panel.json` embeds an earlier `prelaunch-panel.json` binding
(`b9d32fa4...`, 825,706 bytes) and `shared-sources.json` binding
(`22234490...`, 144,619 bytes). Those exact original bytes, the original
1,526-byte prelaunch receipt and the original 1,482-byte selection rule were
not recoverable from the output tree or session snapshots. The live files
were later rewritten to `8c474528...` (826,796 bytes), `8f7b43d2...`
(145,147 bytes), and `1abb2edf...` (2,010 bytes), respectively. The
archival receipt records this as a provenance gap and does not claim a
byte-exact original binding pass. It does not alter the cohort, saved image
identities, 256 outputs, or accepted runtime receipts.

The superseding receipt is
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-distribution-census/archival/archival-binding-reconciliation.json`;
`state.json` keeps the repo-relative result document and the explicit
candidate boundary.
