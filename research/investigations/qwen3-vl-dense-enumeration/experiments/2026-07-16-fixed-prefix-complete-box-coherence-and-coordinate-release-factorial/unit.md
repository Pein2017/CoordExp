---
title: Fixed-Prefix Complete-Box Coherence and Progressive Coordinate-Release Factorial
description: Small causal study separating boundary-wise composition, shared instance ownership with extent uncertainty, object-part shortcuts, and same-category region aggregation.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: authorized_within_goal
unit_id: 2026-07-16-fixed-prefix-complete-box-coherence-and-coordinate-release-factorial
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-07-17
---

# Fixed-Prefix Complete-Box Coherence and Progressive Coordinate-Release Factorial

Execution and bounded interpretation are complete. See [the verified two-
target results and stop-rule verdict](results.md). No training or architecture
was promoted.

## Terminology

- **Qwen3 Vision-Language (`Qwen3-VL`)**: the pretrained multimodal model
  family under investigation.
- **Common Objects in Context (`COCO`) visible extent**: the primary detection
  reference. It follows the accepted `COCO`-style visible or modal box used by
  the active detection task.
- **Plausible complete physical extent**: a secondary, explicitly
  counterfactual box for an occluded object. It asks whether the model can
  complete a physical instance beyond its directly visible support. It is not
  used as the primary detection accuracy target.
- **Exact pre-`x1` prefix**: the frozen image, prompt, adapter, tokenizer,
  complete prior self-rollout, current object description, geometry opener,
  token identifiers, and Key-Value cache immediately before the left
  horizontal coordinate.
- **Coherent box**: four boundaries assigned in advance to one declared
  reference interpretation: visible target, complete physical target,
  adjacent instance, or multi-instance union.
- **Boundary hybrid**: a valid four-coordinate sequence assembled from the
  boundaries of two coherent parent boxes.
- **Progressive coordinate release**: force an exact reference path through a
  declared coordinate slot, then allow all later coordinates and closure to be
  generated natively.

## Authorization Boundary

The user authorized this complete inference research unit and a long-running
goal through its bounded conclusion. The unit may implement the minimum new
scoring and release seams, run small graphics-processing-unit experiments,
and adapt sample count within the declared targets. It does not authorize
model training, a final architecture, an explicit object-slot system, or a
population metric campaign.

## Question

At one exact pre-`x1` state where the object phrase is already fixed, do the
four autoregressive coordinate decisions behave primarily as:

1. separately competitive boundaries that can compose high-probability hybrid
   boxes;
2. a shared physical-instance owner with a still-uncertain visible-versus-
   complete extent mode;
3. a shortcut from one discriminative visible part to category and part-sized
   geometry; or
4. a same-category region aggregate that never resolves one owner?

The unit also asks a treatment-relevant question:

> If an earlier coordinate is placed on a known object-consistent path, does
> that causal choice make the remaining coordinates more coherent, or does it
> merely constrain a sequential language continuation without binding an
> object?

## Why This Is the Next Discriminator

The preceding 119-candidate human review found `108` semantically exact rows
but only `28` acceptable boxes. Localization errors include utensil heads,
laptop screens, chair backs, adjacent-instance unions, and horizontal-versus-
vertical owner mixtures. For directionally codable bottom-boundary errors,
`35/39` end too early. Yet selected rows reconstruct coherent occluded chairs,
so neither pure visual inability nor a universal object-part-only account is
sufficient.

The four coordinates are always autoregressive; literal token independence is
therefore impossible. The empirical question is whether high-probability
complete boxes behave like combinations of locally plausible edges or like a
small number of coherent object-and-extent modes.

## Competing Hypotheses

### Hypothesis 1: boundary-wise composition

The state before each coordinate retains several spatial alternatives. An
early boundary narrows the legal continuation but does not establish one
stable physical owner.

Expected mechanism:

```text
pre-x1 state
  -> choose a plausible left edge
  -> choose a plausible top edge under that text history
  -> choose a plausible right edge
  -> choose a plausible bottom edge
```

Predictions:

- valid boundary hybrids score near coherent parent boxes;
- a forced early edge does not make all remaining boundaries converge to one
  owner more than a matched wrong-owner edge;
- released suffixes frequently remain hybrid or multi-instance;
- coordinate-slot correctness correlations are weak after validity and scale
  constraints are accounted for.

Falsification:

- both coherent endpoints consistently outrank every nontrivial hybrid;
- an owner-disambiguating coordinate causes the later coordinates to remain
  with that owner across clean and crowded targets.

### Hypothesis 2: shared instance owner with extent-mode uncertainty

A latent current owner exists, but the model alternates between visible support
and completed physical extent, especially under occlusion.

Expected mechanism:

```text
physical instance owner
  -> visible-evidence extent or completed-physical extent
  -> four mutually compatible boundaries
```

Predictions:

- coherent visible and complete boxes both outrank arbitrary hybrids;
- adjacent-owner forcing switches later boundaries coherently to the adjacent
  object;
- target-owner forcing preserves owner while bottom or occluded-side extent
  remains bimodal;
- repeated free suffixes cluster around a few coherent owner-and-extent modes.

This hypothesis has two separately adjudicated estimands:

```text
owner coherence:
do later coordinates remain attributable to one physical instance?

extent mode:
conditional on the same owner, is support concentrated on visible extent,
one of several plausible completion extents, or a discriminative part?
```

Falsification of owner coherence requires owner cues to fail to change later
owner attribution or boundary hybrids to remain as probable as coherent owner
paths. Failure to support one reviewer-chosen completion box rejects that
completion reference, not shared instance ownership.

### Hypothesis 3: discriminative object-part shortcut

The model recognizes category from a salient part, then localizes the same part
instead of the reportable object extent.

Predictions:

- utensil-head, laptop-screen, chair-back, or person-head boxes consistently
  outrank complete-object boxes;
- forcing target-consistent early boundaries does not recover the missing
  shaft, body, keyboard, legs, or lower extent;
- part preference survives when adjacent-owner controls are eliminated;
- full-object completion remains weak in the clean, visibly complete target.

Falsification:

- a target-owner cue reproducibly recovers coherent complete extent on clean
  and occluded cases;
- visible and complete modes both recur with similar support.

### Hypothesis 4: same-category region aggregation

The current phrase selects a category-support field rather than a physical
instance. In dense repeated-class scenes, the four boundaries summarize
several activations or a row of parts.

Predictions:

- multi-chair unions and between-chair boxes score competitively;
- forcing one chair's early boundaries still yields neighboring or cross-row
  later boundaries;
- the clean single-object cases are substantially more coherent than the dense
  chair case;
- errors follow repeated-category density more than coordinate slot.

Falsification:

- a same-class owner cue makes later boundaries reliably remain on one chair;
- dense-case coherent target boxes dominate union and hybrid controls.

## Frozen Target Panel

The final target fixture must freeze exact candidate identifiers, source call
bundles, source row indices, prefix hashes, image paths, and all reference
boxes before model scoring. Three targets are permitted; the first smoke uses
only the first and third.

The fixture is an admission gate, not a documentation follow-up. Before any
model execution it must materialize:

```text
source request and exact call-bundle path
source row or span index
exact pre-x1 token identifiers and SHA-256 hash
image path, digest, width, and height
visible reference in source pixels and normalized coordinate bins
part reference
adjacent-owner reference when applicable
multi-instance union reference when applicable
optional two or three plausible completion references or one declared envelope
manual adjudication note and admissible claim type
```

Every source candidate must resolve to one call and one row. Pixel-to-bin and
bin-to-pixel round trips must remain within one coordinate bin. No reference
may be changed after the first score is observed; a corrected fixture creates
a new immutable run identifier.

### Target A: visibly complete utensil with a part prediction

```text
image identifier: 12576
source candidate:
spatial-scope-history-request:
18404482ab0b688758cca6759b5f83bbe5e5959d3689641fdc53bf857977d8a2:
span-11
description: fork
reviewed part box in source-canvas pixels: [569, 518, 674, 561]
primary COCO annotation: coco-ann:686666
primary normalized coordinate bins: [660, 448, 999, 488]
```

The reviewed prediction covers the fork head while the shaft is visible. This
is the cleanest object-part discriminator and differs mainly at the right
boundary. It is a teacher-forced part-versus-full likelihood discriminator.
Forcing its full-object `x2` directly supplies the missing shaft extent, so the
resulting release is only a late extent-and-closure test and must not be called
instance-owner binding. Its frozen owner-discriminative slot set is empty and
its extent-discriminative slot set is only `x2`.

### Target B: laptop screen versus laptop body

```text
image identifier: 9400
source candidate:
spatial-scope-history-request:
1903886302de47f2ecad67ae88766ce2ef73737c27c43ac0d06c6872643f5a04:
span-18
description: laptop
reviewed screen-like box in source-canvas pixels: [121, 550, 475, 863]
accepted complete-laptop reference in source-canvas pixels:
[129, 554, 814, 864]
```

This is a second clean part-versus-complete case. It is confirmation only and
does not enter the first smoke if Target A already provides a decisive clean
result.

### Target C: dense, partially occluded same-class chair field

```text
image identifier: 19432
source request:
spatial-scope-history-request:
ff5660928aa3d9401fb9679d0112b3a5477f1bf3b13a25b9339c2911e02fa0d9
description: chair
reviewed source spans: 0, 2, 3, 4, 5, and 6
```

This one trajectory contains visible-part, coherent individual-chair,
horizontal-neighbor, and cross-row union phenotypes. The first owner test uses
the exact pre-`x1` state of generated row `13`, where the source row closely
matches one accepted individual chair and the nearest same-row chair differs
at `x1` while leaving several informative coordinates to release:

```text
recipient generated row: 13
target owner: coco-ann:378536
target source-canvas box:
[619.3619995117188, 104.86799621582031,
 749.2859497070312, 300.5099792480469]

adjacent same-row owner: coco-ann:387701
adjacent source-canvas box:
[741.1319580078125, 108.73799896240234,
 840.5999755859375, 301.8059997558594]

overlapping upper-row owner: coco-ann:385840
upper-row source-canvas box:
[642.2760009765625, 26.118000030517578,
 755.4600219726562, 115.32599639892578]

multi-row union control: exact natural source span 4
union normalized coordinate bins: [519, 33, 657, 347]
union source-canvas box: [598, 29, 757, 300]
```

These are visible-reference owner and region-aggregation controls; they make no
amodal completion claim. If enlarged-image adjudication or exact source-prefix
resolution contradicts the declared target identities, Target C is held and a
new fixture is required before any owner claim.

The semantic slot contracts are frozen as:

```text
target versus adjacent same-row chair:
  owner-discriminative slots = x1, x2

target versus overlapping upper-row chair:
  owner-discriminative slots = y1, y2

target versus multi-row union:
  owner-discriminative slots = none
  region-discriminative slots = x1, y1
```

Small annotation differences on a shared row, such as the one-bin `y2`
difference between the target and adjacent chairs, remain raw extent evidence
and cannot vote on owner crossover.

## Panel 1: Complete-Box Teacher-Forced Factorial

At the exact pre-`x1` prefix, score the four selected coordinate tokens and
the natural box-close token for every declared coherent box. Preserve, for
each coordinate slot:

```text
selected full-vocabulary log probability
selected coordinate-only normalized log probability
complete 1,000-bin coordinate logit vector as float32 data
top-20 coordinate bins and margins
```

Also score the exact source-native coordinate row as a calibration path. It is
not a manual owner or extent reference and does not enter a coherence contrast;
it reveals whether apparent hybrid preference is merely movement back toward
the model's own realized coordinate mode.

The primary complete-box score is frozen as:

```text
sum of the four selected full-vocabulary coordinate-token log probabilities
```

The box-close-token log probability is reported separately. Coordinate-only
normalization is diagnostic and cannot replace the primary score after results
are observed.

For each pair of coherent parent boxes, score all unique, valid binary
boundary combinations:

```text
discriminative part versus reportable visible whole object
target owner versus adjacent owner
target owner versus overlapping upper-row owner
target owner versus multi-instance union
```

Invalid boxes remain counted as invalid combinations but do not enter the
primary coherence contrast.

For coherent parents `a` and `b`, define:

```text
coherence contrast
= mean log probability of coherent endpoints a and b
- mean log probability of valid non-endpoint boundary hybrids
```

Also report:

```text
visible-versus-complete preference
target-versus-adjacent owner margin
target-versus-union margin
per-slot conditional contribution to every full-box margin
```

No statistical independence claim follows from a zero interaction: all slots
share an autoregressive history. The contrast only measures whether complete
coherent configurations receive probability beyond their component edges.
Report every hybrid's width, height, area, aspect ratio, center, and
geometry-sort rank displacement. A positive coherence contrast is a
configuration preference, not latent-owner proof; the progressive owner
crossover owns the causal interpretation.

## Panel 2: Progressive Coordinate Release

For each coherent pair, find the earliest coordinate slot at which the two
paths differ. Run these arms from one identical exact pre-`x1` state:

1. free coordinate generation;
2. force the first coordinate and release the remaining coordinates;
3. force the first two coordinates and release the remaining coordinates;
4. force the first three coordinates and release the bottom boundary;
5. force all four coordinates and release only box closure as an exact-path
   validity control.

If two reference paths share an early coordinate, begin their causal contrast
at the earliest differing slot rather than pretending that a shared token is
an owner intervention.

Every arm first runs greedy decoding. The sampled owner gate runs `16` paired
temperature-`0.4`, top-p nucleus-threshold-`0.95` suffixes with repetition
penalty `1.0` only for arms that leave at least one frozen
owner-discriminative coordinate to release. The free arm is sampled once,
because it is identical across the two reference paths. Force-three and
force-four remain greedy validity controls when all owner-discriminative slots
are already forced. Extend an admitted sampled cell to `32` suffixes only when
the leading outcome differs by fewer than four counts or a mechanism verdict
changes at that boundary.

Report two separate attribution axes for every complete released box. The
**boundary-configuration attribution** is descriptive and uses the full
endpoint-plus-hybrid lattice:

```text
target visible extent
target plausible complete extent
adjacent owner
overlapping upper-row owner
multi-instance union
boundary hybrid
other same-category region
invalid or no closure
```

The **endpoint-family attribution** asks whether the complete geometry is
target-like or alternate-owner-like. It compares only the two coherent
endpoints in the selected panel. Boundary hybrids do not compete in this
owner-compatible margin because the factorial lattice intentionally contains
one-bin variants arbitrarily close to each endpoint.

Boundary-configuration assignment is frozen before scientific execution and
mutually exclusive:

1. parser-invalid or missing natural box closure is `invalid or no closure`;
2. an exact coherent-endpoint match is assigned that endpoint before any
   margin test; an exact non-endpoint factorial match is assigned `boundary
   hybrid`;
3. otherwise compute scale-normalized edge distance from prediction `p` to each frozen
   reference `r`:

```text
one quarter times (
  absolute(p.x1-r.x1)/r.width
  + absolute(p.x2-r.x2)/r.width
  + absolute(p.y1-r.y1)/r.height
  + absolute(p.y2-r.y2)/r.height
)
```

4. assign the unique nearest reference family only when its distance is at
   most `0.35` and its margin over the second-nearest family is at least
   `0.05`;
5. assign an accepted boundary-hybrid reference to `boundary hybrid` and a
   union reference to `multi-instance union` under the same rule;
6. ties or overlapping families without the required margin become
   `boundary hybrid` only when a frozen hybrid is nearest, otherwise
   `other same-category region`;
7. everything outside the radius is `other same-category region`.

Endpoint-family attribution uses the same distance, radius `0.35`, and margin
`0.05`, but compares only the selected pair's two coherent endpoints. It
reports the nearest endpoint, second endpoint, distances, margin, accepted
flag, and one of `target-like`, `alternate-like`, `ambiguous`, or
`outside-radius`. This axis is owner-compatible geometry evidence, not proof
of a latent owner state.

For every coordinate slot, also preserve:

```text
raw absolute error to both endpoints
which endpoint the coordinate is closer to, or tie
endpoint separation at that slot
whether the coordinate was forced or released
```

A one-bin vertical difference cannot own an instance-level conclusion merely
because it is fractionally nearer one endpoint. Owner preservation requires
endpoint-family attribution plus no crossover on the materially separated
released coordinates.

All raw distances, per-slot errors, and the two nearest families on both axes
remain in the artifact so the classification can be audited without rerunning
the model. The first greedy smoke discovered that one full-lattice margin
cannot serve both configuration and owner estimands; this split is the bounded
contract repair made before sampled scientific execution.

Primary causal observations are:

- probability that later coordinates remain with the cued owner;
- probability that a target-owner cue changes visible extent into complete
  extent;
- probability that a complete-extent cue remains target-consistent;
- probability that a dense-chair cue still collapses to a union or hybrid;
- phrase and geometry owner consistency when a row continuation is available.

## Confounds and Required Controls

### Autoregressive validity constraints

Forcing `x1`, `y1`, or `x2` changes the support of later legal boxes even
without object binding. Every target cue therefore has a matched adjacent-
owner or union cue at approximately similar scale and coordinate rank.

### Generic box-shape prior

Hybrid boxes may have unusual area, aspect ratio, center, or traversal rank.
These attributes are reported for every scored path. Configuration preference
alone cannot establish owner coherence; the owner-release crossover must also
succeed.

### Geometry-sorted traversal prior

The adapter was trained with geometry-sorted rows. A coordinate cue may act as
a traversal-rank signal rather than an instance owner. Report whether released
boxes move to the canonical next spatial region even when they do not preserve
the cued object.

### Visible versus complete annotation semantics

Primary detection conclusions use only the visible `COCO`-style reference.
Secondary complete-physical conclusions are mechanism observations and must be
reported separately.

### Incomplete or crowd annotations

Image `19432` contains a `COCO` crowd region and overlapping individual chair
annotations. Do not classify an extra chair-like box as unsupported merely
because it lacks one individual annotation. Owner claims require manual visual
adjudication; otherwise use the region-aggregation label.

### Numerical execution branch

All stored logits and aggregate score arithmetic use 32-bit floating point.
Discovery may use the standard `bfloat16` model runtime, but every
conclusion-changing near tie or owner switch must reproduce at physical batch
size one. If `bfloat16` and full-model 32-bit floating point disagree on the
mechanism classification, the 32-bit run owns the conclusion and the numeric
branch is reported explicitly.

### Candidate-selection bias

Targets come from an unmatched, bagging-enriched cohort. Results identify
possible mechanisms and do not estimate their population prevalence.

## Outcome-to-Belief Map

| Observation | Belief update |
|---|---|
| Valid hybrids score near coherent endpoints and staged forcing does not preserve owner | Favor boundary-wise composition. |
| Coherent part and reportable-whole boxes beat hybrids; owner cue switches all later boundaries while optional completion extent stays multimodal | Favor shared owner with extent-mode uncertainty. |
| Part box dominates complete box in clean targets and target-consistent forcing cannot restore full visible extent | Favor discriminative object-part shortcut. |
| Dense-chair unions remain competitive and owner cues do not prevent unions, while clean targets are coherent | Favor same-category region aggregation in crowded scenes. |
| `x1` or `x1,y1` establishes owner, but `y2` remains part-sized | Support phase-separated owner and extent decisions rather than one static box state. |
| Owner coherence succeeds but one exact complete-physical reference has low support | Preserve the owner result and reject only that completion reference. |
| Full-model 32-bit floating point removes a conclusion-changing split | Record a numerical branch, not a semantic mechanism. |
| All coherent and hybrid boxes are uniformly low probability | The selected description state does not contain a stable geometry mode; stop this operator. |

Mechanisms may differ by target. A clean-part result and a crowded-chair result
must not be forced into one universal diagnosis.

## Stop Rules

- Stop after Target A plus Target C if they already separate the clean and
  crowded mechanisms; use Target B only for ambiguous clean evidence.
- Stop boundary-wise composition as the leading clean-target explanation if
  coherent endpoints beat every valid hybrid and an owner cue preserves the
  remaining boundaries.
- Stop pure shared-owner interpretation if owner cues repeatedly produce
  unions or between-instance boxes.
- Stop pure object-part interpretation if clean and occluded target cues
  reproducibly recover complete extent.
- Stop architecture inference even after a positive result. This unit may
  identify where an intervention could act, not what final carrier it needs.
- Do not train from this unit. A later proposal must state the shortest
  conclusion-supported objective and its preservation control.

## Reused Surfaces and Minimal Implementation

Frozen execution inputs:

```text
worktree:
/data/CoordExp/.worktrees/research-probes

inference configuration:
configs/coordexp_swift/infer/
qwen3_vl_2b_desc_first_geo_sorted_gaussian_rps_dora_r16a32_step4887_val200.yaml

source data:
/data/CoordExp/.worktrees/CoordExp-swift/outputs/coordexp_swift/infer/
val200_inputs/coco_val200_len12000.rebased_images.coord.jsonl
```

Reuse the current Qwen3-VL model loader, canonical assistant-continuation
renderer, detection-row parser, request-scoped sampler, coordinate-token
constants, fixed-prefix hash checks, and prior full-coordinate logit capture.

Add only:

1. one small immutable target fixture;
2. one complete-box scorer that emits per-slot and joint coordinate evidence;
3. one progressive coordinate-release loop;
4. one outcome classifier against the frozen references.

The first implementation may remain experiment-local under
`scripts/research/`. Promote a reusable `src/analysis` module only if a second
unit needs the same seam. No OpenSpec change is required.

## Artifact Root

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-07-16-fixed-prefix-complete-box-coherence-and-coordinate-release-factorial/
<immutable-run-identifier>/
```

Each run must preserve the target fixture, runtime identity, exact prefixes,
reference semantics, per-sequence scores, sampled release bundles, summary,
and a compact receipt.

## First Smoke and Rough Cost

The first smoke uses Target A and one adjudicable Target C state:

```text
teacher-forced scores for coherent parents and valid hybrids
one greedy staged-release suffix per arm
no more than two targets
physical batch size one
```

The first smoke validates fixture resolution, scoring, classification,
progressive release, parser closure, and receipts only. One greedy suffix per
arm cannot support or falsify a distributional mechanism hypothesis.

If the smoke is exact, the first scientific gate uses Target A and one
adjudicated Target C state with the declared `16` paired suffixes per admitted
release arm. It may extend to `32` only under the frozen close-count rule.
Target B is added only when the clean part-versus-full result remains
ambiguous. This is an inference case study, not a dataset-scale evaluation.
