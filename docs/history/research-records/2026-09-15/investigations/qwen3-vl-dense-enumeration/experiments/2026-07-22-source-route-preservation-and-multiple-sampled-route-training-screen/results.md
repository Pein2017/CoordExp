# Results: Source-Route Preservation and Multiple-Sampled-Route Training Screen

## Verdict

The treatment learns the selected physical owners, improves box quality and
standard detection metrics, and often reduces redundant output. It does not
reliably expand the final greedy physical-owner set.

The stable failure pattern is owner exchange:

- owners selected by sampled-route supervision become substantially easier to
  recover;
- admitted training images sometimes gain owners;
- non-admitted images lose owners at every evaluated milestone;
- the only positive full-cohort owner delta is sensitive to the matching
  threshold and does not survive the lower-overlap entity-discovery view.

This is not evidence that the gradient is too weak to learn. It is evidence
that the current 118-image supervision is narrow relative to the behavior that
must be preserved across the full greedy rollout distribution.

## Executed Scope

The Source checkpoint and both treatment arms were evaluated under ordinary
greedy decoding. Each treatment arm contained 992 unique events from 118
images, split evenly between 496 sampled-route rows and 496 exact Source-route
preservation rows. The single-route and multiple-route arms ran for 31 optimizer
updates. The multiple-route arm was additionally evaluated at steps 10, 20,
30, and 31 on all 256 training-screen images.

All full-cohort inference artifacts are benchmark-eligible. Parser and scoring
failures are zero. At most one request per milestone reached the generation
length limit, and one to five parsed predictions were dropped depending on the
checkpoint.

## Full 256-Image Multiple-Route Curve

The Source checkpoint finds 1,684 unique annotated owners at intersection over
union 0.50 and 1,856 at intersection over union 0.30.

| Checkpoint | Owner delta at 0.50 | Owner delta at 0.30 | Mean Average Precision | Mean Recall | Prediction count | Duplicate-candidate delta at 0.50 |
|---|---:|---:|---:|---:|---:|---:|
| Source | 0 | 0 | 0.3864 | 0.4529 | 3,290 | 0 |
| Multiple-route step 10 | -2 | -6 | 0.3971 | 0.4595 | 2,819 | -50 |
| Multiple-route step 20 | -6 | -12 | 0.3932 | 0.4553 | 2,973 | +239 |
| Multiple-route step 30 | +4 | -25 | 0.3968 | 0.4588 | 2,656 | -86 |
| Multiple-route step 31 | -7 | -27 | 0.3941 | 0.4563 | 2,575 | -64 |

The step-30 `+4` result at intersection over union 0.50 is not broad entity
discovery. At intersection over union 0.30 it becomes `-25`. Standard mean
Average Precision improves by roughly 0.7 to 1.1 points at every intermediate
milestone, which is consistent with better geometry, confidence ordering, or
duplicate suppression rather than more unique physical objects.

Step 20 also shows that checkpoint behavior is not a simple monotonic dose
curve. It retains fewer owners while producing many more duplicate candidates
than the neighboring checkpoints. Greedy trajectories can bifurcate after a
small parameter change, so no single milestone should be interpreted without
the owner ledger and lower-overlap view.

## Admitted and Non-Admitted Transfer

At intersection over union 0.50, the owner deltas are:

| Scope | Step 10 | Step 20 | Step 30 | Step 31 |
|---|---:|---:|---:|---:|
| Admitted 118 images | +13 | +8 | +16 | +5 |
| Non-admitted 138 images | -15 | -14 | -12 | -12 |
| Full 256 images | -2 | -6 | +4 | -7 |

At intersection over union 0.30, the owner deltas are:

| Scope | Step 10 | Step 20 | Step 30 | Step 31 |
|---|---:|---:|---:|---:|
| Admitted 118 images | +11 | +3 | -2 | -1 |
| Non-admitted 138 images | -17 | -15 | -23 | -26 |
| Full 256 images | -6 | -12 | -25 | -27 |

The transfer sign is therefore stable: every milestone loses unique owners on
images that supplied no gradient. More optimization does not repair this.

## Selected-Owner Learning Is Real

The multiple-route bank contains 472 unique selected physical owners, all of
which join exactly to evaluation annotations. Across milestones, Source-missed
selected owners are recovered approximately four to six times as often as
Source-missed non-selected owners.

- Step 10, intersection over union 0.50: selected-owner recovery is about 30
  percent, compared with about 5 percent for non-selected owners.
- Step 20: 22 of 83 missed selected owners are recovered at intersection over
  union 0.50, compared with 50 of 764 non-selected owners.
- Step 30: 29 of 83 missed selected owners are recovered, compared with 47 of
  764 non-selected owners.

The treatment therefore has a precise training effect. The problem is not that
noise completely hides the supervision. The problem is that the effect is
concentrated and is paid for by changes elsewhere in the greedy trajectory.

## Single Route versus Multiple Routes

At final step 31, the single-route arm finds 1,668 unique owners, sixteen fewer
than Source, and increases duplicate candidates by 246. The multiple-route arm
finds 1,677 owners, seven fewer than Source, while reducing duplicate
candidates by 64. Multiple complementary routes are therefore safer than one
route, but Source-route replay still does not preserve the complete functional
rollout distribution.

## Human-Refined Development Panel

On the twelve human-refined images, Source finds 138 owners at intersection
over union 0.50. The multiple-route milestones find 142, 135, 134, and 133
owners at steps 10, 20, 30, and 31. The apparent step-10 gain is entirely
contributed by image 4134 under this threshold, so the panel does not establish
a general validation gain.

## Interpretation

The evidence rejects two simple explanations:

1. **The gradient is too weak to do anything.** Selected-owner recovery is
   strongly enriched at every milestone.
2. **Training simply needs more updates.** Non-admitted transfer is negative at
   every milestone and lower-overlap coverage worsens with exposure.

One plausible remaining explanation is insufficient image breadth: 496 sampled
events are concentrated in only 118 images, so the update learns image- and
route-specific modes instead of a broad rule for reallocating probability to
uncovered owners. A competing explanation is intrinsic owner exchange under
positive-only row imitation: broadening the images will merely change which
owners are gained and lost.

The smallest discriminator is a constant-dose breadth experiment. Keep the
same 496 sampled events, 496 Source-preservation events, 31 optimizer updates,
learning rate, effective batch size, and decode policy, while comparing a
newly matched 118-image arm with a 496-image arm. This is a bounded test of the
sample-scarcity explanation, not promotion of the current objective to a
longer epoch.

Stop this treatment family if the broader arm still gains selected owners while
losing a comparable number of never-trained owners on disjoint evaluation
slices. Promote only if broader image coverage produces net held-out owner
growth without unacceptable geometry, duplicate, malformed-output, or semantic
regression.

## Evidence Handles

- StateBanks and receipts:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-22-source-route-preservation-and-multiple-sampled-route-training-screen/state-banks-v2/`
- Training runs:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-22-source-route-preservation-and-multiple-sampled-route-training-screen/train-992-events-v1/`
- Clean rollout artifacts:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-22-source-route-preservation-and-multiple-sampled-route-training-screen/clean-rollouts-v1/`
- Pairwise owner analyses:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-22-source-route-preservation-and-multiple-sampled-route-training-screen/analysis-v1/`
- Strict final comparator:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-22-source-route-preservation-and-multiple-sampled-route-training-screen/analysis-v2/train-256-source-vs-single-vs-multi-step31-strict.json`
