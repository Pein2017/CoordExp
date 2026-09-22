# Mixed Old-Prefix and Refreshed-Prefix Coordinate Correction Results

## Verdict

Refreshing the complete rollout-derived correction event package changes what
the coordinate objective learns, but the current comparison does not isolate
prefix state from changed owners, coordinate targets, and candidate rows. The
resulting local correction is not a safe complete-rollout treatment.

The experiment supports only the operational claim that refreshing the whole
event distribution is slightly better than repeating stale events under this
budget. It does not prove that prefix-state distribution itself caused the
difference. Refreshed-event training improved matched-event margins and
slightly improved aggregate clean-rollout geometry, while failing to preserve
the intended row owner and increasing invalid geometry relative to the shared
intermediate.

## Executed Comparison

All final branches started from the same shared intermediate and used:

- the same 192 image identities and one trusted event per image;
- one epoch, six optimizer steps, learning rate `3e-6`, and seed `29`;
- eight data-parallel ranks and effective batch size 32;
- language-tower Weight-Decomposed Low-Rank Adaptation only;
- first-wrong-coordinate preference with token-type stabilization;
- normalized integer coordinates in `[0, 999]`.

The old and refreshed StateBanks were rebuilt under the same current builder.
The historical/current builder mismatch reduced the exact paired cohort from
the planned 224 images to a conservative 192-image intersection.

## Exact-Prefix 32-Bit Floating-Point Replay

| Prefix bank | Shared intermediate margin | Old-prefix control margin | Refreshed-prefix treatment margin |
|---|---:|---:|---:|
| Old | 0.50531 | **0.61695** | 0.59057 |
| Refreshed | 0.47954 | 0.56147 | **0.58713** |

Paired branch effects:

| Comparison | Mean margin delta | Improved events | Bootstrap 95% interval |
|---|---:|---:|---:|
| Old control minus refresh on old prefixes | +0.02637 | 113 / 192 | [0.01433, 0.04327] |
| Refresh minus old control on refreshed prefixes | +0.02566 | 121 / 192 | [0.01371, 0.04021] |
| Prefix-matching difference of differences | +0.05203 | 114 / 192 positive | [0.03273, 0.07498] |

The full-cohort interaction is real for the two compared event packages. Each
branch is best on the package it trained on. Both branches also improve over
the shared intermediate on both banks, so the result contains additional
optimization and package-specific adaptation.

### Why this is not yet a prefix-only result

The banks share image identities, not complete supervised decisions:

| Matched property | Images |
|---|---:|
| Same physical owner | 150 / 192 |
| Same selected coordinate axis | 152 / 192 |
| Same owner, axis, and acceptable coordinate set | 139 / 192 |
| Same owner, axis, acceptable set, and complete candidate token row | 69 / 192 |
| Same complete decision including prefix | 34 / 192 |

After requiring the same owner, selected axis, acceptable set, and candidate
row, and retaining only the 35 cases whose prefix actually changes, the paired
difference of differences is `0.00261`, with bootstrap 95% interval
`[-0.000001, 0.00755]`. This subset is small and the interval touches zero.
Accordingly, the experiment does not establish a causal prefix-state effect.
It compares stale versus refreshed event packages, not prefix alone.

## Clean Greedy Rollout

All three models ran on the same 256-image cohort with repetition penalty 1.0,
temperature 0, and a 512-token limit.

| Model | Mean Average Precision | Average Precision at 0.75 | Mean recall | Predictions | Dropped | Truncated |
|---|---:|---:|---:|---:|---:|---:|
| Shared intermediate | 0.38825 | 0.41194 | 0.45152 | 3,099 | 8 | 6 |
| Old-prefix control | 0.38666 | 0.41071 | 0.45000 | 3,098 | 18 | 5 |
| Refreshed-prefix treatment | **0.38894** | **0.42116** | **0.45310** | 3,090 | 16 | 5 |

Refreshed-prefix treatment versus the shared intermediate:

- Mean Average Precision: `+0.00070`;
- Average Precision at 0.75: `+0.00922`;
- mean recall: `+0.00157`;
- dropped predictions: `+8`, consisting of 11 invalid geometries and five
  malformed object spans, versus two and six in the shared intermediate.

Refreshed-prefix treatment versus old-prefix control:

- Mean Average Precision: `+0.00228`;
- Average Precision at 0.75: `+0.01045`;
- mean recall: `+0.00310`;
- dropped predictions: `-2`.

The branch comparison favors refreshed states, especially for tighter boxes,
but the gain is small, in-sample, and not clean relative to the shared
intermediate.

## Owner-Level Check

The refreshed StateBank was created only when the shared intermediate produced
the trusted physical owner at the recorded row, so it retained category at all
192 intended rows by construction. After either branch update, only 175 of 192
owners retained the same category at that row.

| Model | Intended-row category retained | Mean intended-row Intersection over Union when retained | Mean best same-category Intersection over Union anywhere | Owners found at Intersection over Union at least 0.5 |
|---|---:|---:|---:|---:|
| Shared intermediate | 192 / 192 | 0.76342 | **0.77478** | **175 / 192** |
| Old-prefix control | 175 / 192 | **0.70880** | 0.76413 | 172 / 192 |
| Refreshed-prefix treatment | 175 / 192 | 0.69817 | 0.76456 | 172 / 192 |

Best same-category Intersection over Union for refresh minus shared was
`-0.01022`, with bootstrap 95% interval `[-0.02772, 0.00380]`. This check is
not a perfect physical-owner metric in crowded same-class scenes, but it rules
out interpreting the aggregate Average Precision at 0.75 gain as uniform
repair of the exact trained owners.

A diagnostic duplicate proxy also worsened: exact repeated rows were 307, 320,
and 311 for shared, old control, and refresh respectively. Same-category
high-overlap pairs increased more strongly under refresh, though that proxy can
confound distinct crowded instances and is not treated as an official metric.

## Interpretation

The evidence supports two bounded observations:

1. **Event-package specialization is real.** A coordinate preference becomes
   most effective on the combination of prefix, owner, candidate row, and
   target distribution used to train it.
2. **The treatment is not enough.** Updating language-tower parameters for one
   selected coordinate can change row ownership, ordering, other boundaries,
   and format behavior during free generation.

The observed clean Average Precision at 0.75 improvement is encouraging but
does not pass the predeclared safety gate. This unit therefore stops without a
1,024-image or full-dataset promotion.

The next training proposal should supervise a coherent row-level object state
or complete-box correction while retaining the valuable own-prefix refresh.
It should not merely repeat more first-wrong-coordinate updates.

## Artifacts

- Paired cohort: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-21-mixed-old-and-refreshed-prefix-coordinate-correction/paired-cohort/image-ids-192.json`
- Old-prefix StateBank: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-21-mixed-old-and-refreshed-prefix-coordinate-correction/old-prefix-paired-192/state-bank/manifest.json`
- Refreshed-prefix StateBank: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-21-mixed-old-and-refreshed-prefix-coordinate-correction/refreshed-prefix-paired-192/state-bank/manifest.json`
- Exact-prefix evaluations: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-21-mixed-old-and-refreshed-prefix-coordinate-correction/evaluations/`
- Paired statistics and owner analysis: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-21-mixed-old-and-refreshed-prefix-coordinate-correction/analysis/paired-statistics-and-clean-owner-analysis.json`
- Duplicate proxies: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-21-mixed-old-and-refreshed-prefix-coordinate-correction/analysis/clean-rollout-duplicate-proxies.json`
- Prefix-only matched-subset analysis: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-21-mixed-old-and-refreshed-prefix-coordinate-correction/analysis/prefix-only-matched-subset-analysis.json`
- Shared-intermediate rollout: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-21-mixed-old-and-refreshed-prefix-coordinate-correction/clean-rollouts/qwen3-vl-2b-mixed-prefix-shared-intermediate-train-256-hf/`
- Old-control rollout: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-21-mixed-old-and-refreshed-prefix-coordinate-correction/clean-rollouts/qwen3-vl-2b-mixed-prefix-old-control-train-256-hf/`
- Refresh-treatment rollout: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-21-mixed-old-and-refreshed-prefix-coordinate-correction/clean-rollouts/qwen3-vl-2b-mixed-prefix-refreshed-treatment-train-256-hf/`

## Limitations

- The clean-rollout cohort is the same 256-image training-screen cohort, not a
  held-out generalization set.
- The paired branches use only 192 images admitted by both builders.
- The full paired comparison changes supervised owners, coordinate targets,
  and candidate rows in addition to prefixes; the prefix-only matched subset
  contains only 35 changed-prefix cases.
- Best same-category matching can credit the wrong same-class instance in
  crowded scenes.
- One first failed parallel Old-Prefix Repeat run is retained only as runtime
  provenance; the reported control is the later verified checkpoint.
