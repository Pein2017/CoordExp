# Next-stage scale preparation — CPU candidate v2

Propose one fresh four-epoch trajectory on **1,024 training / 256 validation
images**, saving fixed checkpoints at epochs 1, 2 and 4. The primary question is
larger-panel natural training completion and coordinate fidelity, subject to
format/duplication guardrails. Validation coverage or CE degradation alone does
not select settings or veto the training result. No next-stage model call,
GPU allocation, training, evaluation or packing-cache pass has occurred.

The source is the same mature untied+axis001 step2444 under the qualified live
PEFT runtime, with the unchanged architecture, full-response segment-balanced
loss and nominal optimizer. Do not initialize from either overfit checkpoint.
The machine-generated config proposal reuses the maintained entry and strict
runtime projections. Exact epoch checkpoint step indices await the actual
packing plan before any launch; the proposed config is not launch-ready.

## Frozen proposed data

The canonical owner is [research/assets.md](../../assets.md), asset
`coco12k-geo-sorted-xy-v1`. SHA-ordered metadata selection retains all corrected
32 records unchanged by identity, and adds 992: 400 ordinary, 400 dense
same-class, 96 dense other and 96 middle-density images. Validation contains
128 ordinary and 128 dense same-class images. Ordinary means 1–4 positives;
dense means at least 10; same-class requires dominant-class share at least .75.
The seeds and exact source-line bindings are in `selection.json`.

Parent readback verified all 2,560 original/processed image bindings, canonical
identity and original-content disjointness, exclusion of the current fit32 and
monitor64 from validation, and exact preservation of each corrected regression
record. Strict-loader projections strip only admission metadata and rebase
image paths without changing resolved bytes. All 1,280 targets fit cap3084.

| Panel | Images | Known positives | Response tokens | Maximum target tokens |
| --- | ---: | ---: | ---: | ---: |
| Training | 1024 | 9519 | 90934 | 630 |
| Validation | 256 | 2033 | 19541 | 170 |

Density histograms, class-positive counts and length quantiles are in
`distributions-v2.json`. Annotation precedence, owner IDs and `geo_sorted_xy`
serialization remain intact. UNKNOWN is not a verified false positive.

**Known source exposure is substantial:** 959/1024 training identities (5 of
the corrected32, 954 additions) and 248/256 validation identities occur in the
actual mature SFT training source. This is checked against its bound training
config and canonical train JSONL, not the refined18 inference panel. It does
not establish identical refined annotations or absence of other pretraining/
SFT exposure. Validation is excluded from this next training phase and the
current package, but is not an untouched test set.

## Exposure, evaluation and proposed guardrails

Epochs 1/2/4 expose exactly 1024/2048/4096 image presentations and
90934/181868/363736 canonical response tokens before packing effects. The
current packing ratio estimates 480/960/1920 pack presentations and
60/120/240 updates; these are not frozen actual pack counts. Four-epoch
training estimates range heuristically from 1501 to 3392 GPU-seconds. Different
visual grids, padding and packing can invalidate either estimate.

Evaluate every training and validation image at source and all three fixed
checkpoints: **5120 native cells**, empty prefix, greedy, RP1, cap3084. Saved
throughput projects about 142105–147612 GPU-seconds for decoding, dominating
training. Ideal eight-worker wall time is roughly 5–5.3 hours before setup and
tails. Propose an **8-hour / 64-GPU-hour containment envelope for lead review**;
this is not execution authorization or a completion guarantee. Cap-heavy
outcomes may leave explicit HOLD cells; do not shorten the cap or denominator.

Proposed nonzero guardrails apply separately to each panel, relative to its
matched source: at most 10 percentage points more bad images and at most 10%
newly bad images; at most 2 points more caps and 2% newly capped images; at
most 5 points more annotation-owner recurrence incidence and 5% newly recurrent
images; at most 2 points more severe-run incidence and 2% newly severe images
(proxy run length at least five). Integer allowances use floor(fraction*N).
Bad means malformed/parser drop, invalid geometry, cap or non-natural EOS;
UNKNOWN is excluded. These thresholds require lead review before launch.
Unlike the superseded v1 union rule, source failures alone do not veto a model.

Report paired gains/losses, exact versus annotation-owner recurrence, run
lengths, malformed span characters and image incidence, generated lengths,
EOS/cap, IoU50/80 and coordinate error, with ordinary/dense strata and the
corrected32 regression group separate. Lower aggregate invalid/repeat counts
alone cannot qualify an outcome. Rank eligible checkpoints by training clean
completion, IoU50 coverage, IoU80 coverage, then coordinate error; break ties
by earlier exposure. Distinguish insufficient fit, fitting with guardrail
failure, and fitting within guardrails with validation regression. No LR or
architecture search is proposed.

## Bindings and remaining launch decisions

All JSON/JSONL evidence is under
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-coordinate-codebook-scale-preparation/`.
Use `admission-v2.json`, `trajectory-proposal-v2.json`,
`exposure-cost-supplement-v2.json`, `parent-readback-v1.json` and
`distributions-v2.json`; original v1 files remain preserved and superseded.
The lead must freeze the proposed tolerances, actual packing/epoch boundaries,
operational allocation and new envelope before launch. No current result,
selection, state, frontier or catalog was changed by this preparation.
