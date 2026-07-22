# Comprehensive Retrospective: Qwen3-VL Dense Enumeration Main Thread

Source session: 019f4a19-d81c-75a2-84b0-2c20379e686e

Source snapshot: /data/CoordExp/.codex/sessions/2026/07/10/rollout-2026-07-10T03-37-15-019f4a19-d81c-75a2-84b0-2c20379e686e.jsonl

Reconstructed: 2026-07-22

## What this note is and is not

This is a historical continuity record for the long main thread. It is not a
replacement for executed artifacts, source code, research result records,
current documentation, or stable specifications. Research unit results and
their immutable output roots remain the authority for numerical claims.

The session was a fork and began by referring to earlier painted-visual-control
work. Claims about that earlier work are reported only to the extent that the
session itself restated them. The transcript was an appendable file while this
reconstruction was being made, so the snapshot boundary is the 128,587 records
and approximately 833.9 megabytes present at bootstrap start. Later appends
were intentionally excluded rather than silently widening the historical
scope.

The source was traversed chronologically in bounded streams, first through
compact event and response metadata and then through selected raw evidence
where a metric, receipt, path, or conclusion needed confirmation. Coverage was:

| Source records | Approximate time | Main historical transition |
| --- | --- | --- |
| 1 to 32,146 | Jul 10 to Jul 12 09:28 UTC | Painted causal intervention, native commit probes, proposal bridge, and the first rollout closeout |
| 32,147 to 64,293 | Jul 12 to Jul 15 11:05 UTC | Worktree/archive reset, Dense-Union spatial-policy study, sampled rescue, phrase-geometry and batch-precision probes |
| 64,294 to 96,440 | Jul 15 to Jul 19 11:17 UTC | Spatial-routing and residual seams, natural row-transition studies, human review, ordering and causal-value work |
| 96,441 to 128,587 | Jul 19 to Jul 22 03:42 UTC | Prefix-order closure, calibration/training screens, trajectory training, and the new completion study |

No raw transcript, large tool output, credentials, generated index, or cache
has been copied into memories.

## Executive reconstruction

The program's north-star objective became: determine whether the pretrained
Qwen3 Vision-Language model can greedily enumerate the complete supported
Common Objects in Context 80-category object set while preserving physical
instance binding, usable geometry, low duplication and invalid output, and
natural termination. Sampling can reveal support, but it is not an acceptable
final enumeration policy by itself.

The central belief at the end of the session is deliberately narrower than
several earlier hypotheses:

1. The model has real object support and its prefixes are causally active.
   Repeated low-temperature rollouts can expose valid physical objects that a
   greedy route misses, and coherent native rows can redirect later behavior.

2. That state is not yet a stable order-free covered-object ledger. Earlier
   prefix order can change later owners even when the covered physical set and
   final row are held fixed. Some local covered-owner suppression exists, but
   path-sensitive competition and route dependence coexist with it.

3. Several seemingly promising control surfaces are real but bounded. A
   post-vision spatial-routing seam and a layer-23 first-coordinate geometry
   basin transfer were demonstrated in selected cases. Neither establishes a
   reusable object pointer, a complete object state, a deployable attention
   policy, or an architecture choice.

4. Local learnability did not transfer automatically into healthy free rollout.
   Coordinate objectives improved fixed-prefix margins but did not improve
   clean rollout. Single-route sampled-trajectory imitation moved greedy
   behavior toward a selected route family, but exchanged ordinary owners and
   did not increase final coverage.

5. The active next question is no longer “which architecture should be built?”
   It is: under what least controlled, ground-truth-assisted prefix conditions
   can frozen native greedy decoding complete the remaining trusted object set,
   and is failure first due to termination, accessibility and selection,
   trajectory route state, description-to-instance binding, or geometry?

The last historical unit is therefore a diagnostic completion surface, not
training and not a claim of success. It was authored and at research-unit
status ready at the transcript boundary. The transcript contains no execution
of the unit; post-boundary receipts belong to the live evidence surface in
`memories/current.md` and must be reviewed from their immutable output root.

## Original problem and the first methodological correction

The thread inherited painted-image and post-scatter residual work. The useful
causal chain was:

    painted image perturbation
      -> visual embedding change
      -> exact post-scatter intervention
      -> late-middle language-model residual state
      -> immediate next-token effect

This was meaningful evidence on selected strict replay events, but not
evidence of full-row control, reliable geometry, termination, duplicate
handling, detection quality, or a production interface. A crucial cache
semantics concern was exposed immediately: an intervention demonstrated by
recomputing prefixes with cache disabled cannot simply be treated as a
per-emitted-row visual update inside a normal cached decoder.

The user corrected an initial tendency to frame a final cursor or visual bridge
architecture too early. The durable operating rule became:

    architecture is an outcome of discriminating experiments,
    not a premise that experiments are asked to justify.

Requirements, hypotheses, experimental handles, candidate implementations,
and outcome-to-belief maps must remain separate. The research lead should
choose small, high-information tests, keep promising interventions as causal
teachers, and delay architecture promotion until the mechanism and treatment
gates pass.

A row-trajectory persistence experiment reinforced this boundary. A one-time
residual write could help phrase generation in two of three interpretable
events, but matched neither a complete row nor all four coordinates in any of
the three. The appropriate conclusion was phase-specific immediate influence,
not a persistent object-row controller.

## Phase one: native commit, coordinate-local state, and the rejected proposal bridge

### From cursor mechanics to a native transition question

The research focus shifted to whether an emitted row naturally changes later
object selection. The priority order was:

    commit-caused redirection
      > remaining-object selection
      > exhaustion and STOP
      >> cursor synthesis.

The first commit-to-uncovered studies used the long pure-cross-entropy,
token-type-gated step-4887 checkpoint and pre-existing validation artifacts.
They repaired a serious ledger issue: a repeat of an earlier covered annotation
must not be mislabelled unknown. The resulting taxonomy distinguished new
uncovered owners, current-row repeats, earlier-covered repeats, ambiguous
matches, unknowns, terminal output, and malformed output.

An early smoke made a permanent distinction: a teacher-forced candidate score
can favor a remaining owner while free generation emits an unmatched or
low-overlap row. Teacher-forced redistribution is therefore explanatory only;
it does not establish usable free selection or enumeration.

Generated-row versus canonical-row comparisons subsequently showed robust
object-relative suppression for attributable rows, including same-class
controls. Source swaps ruled out the simplistic claim that any coordinates
would work. The supported version was a trajectory-gated, coordinate-local
occupied or recently-emitted field in text and key-value state. It was not an
object-indexed semantic ledger or a reliable target-specific selection gate:
later prefixes could increase readability for both the target and stable
non-selected controls.

### A, B, and C proposal bridge

The user proposed a carefully bounded explicit intermediate bridge:

* A: cross-entropy continuation baseline;
* B: proposal supervision without feeding the proposal back;
* C: the same proposal supervision plus a proposal-conditioned bridge into row
  generation.

Coverage repulsion, slots, strong EOS tricks, and an additional arm were
deliberately deferred. Repetition penalty 1.10 was treated as inherited decode
engineering, not as a model of object semantics. Prefix corruption and jitter
were allowed for robustness, but historically corrupted tokens were not
promoted to supervision targets.

The bridge was implemented only after extensive real-path validation:
no-op parity, causal isolation, packed-segment isolation, cache growth,
rotation behavior, and gradient behavior under Scaled Dot Product Attention
and Flash Attention. Several implementation lessons matter beyond this one
bridge:

* a zero index-add that changes the backward graph is not a genuine C-off
  control;
* ownership of a parameter in float32 does not prevent autocast computation in
  bfloat16;
* a probe is insufficient unless it traverses the actual Qwen forward and
  cache path;
* losses must normalize over nonempty positive bags;
* cache identity must remain arm-neutral even if live arm labels are rebound
  later.

All three arms completed 512 matched eight-GPU steps. B and C passed the
predeclared held-out representation gate; C was selected by priority rather
than demonstrated dominance. That finding remained representation-level.

The own-prefix behavioral panel was decisive. C-on generated many more rows,
but the behavior was reproduced by another-image input and token permutation,
was partly sensitive to write position, and was not reproduced by
norm-matched random input. At repetition penalty 1.00 it became even more
duplicate-prone. The evaluator returned a safety non-inferiority hold. The
bridge was trainable and received gradients, but learned a structured global
continuation or row-prior effect rather than object-specific visual-to-text
pairing. It was closed, not rewritten into a speculative production design.

This outcome also triggered a process correction. An independent retrospective
found that the panel had been overbuilt before its smallest causal discriminator
was run. The durable workflow changed toward one owner, one smallest real
smoke, one focused repair, and escalation only when a semantic stall or missing
receipt could alter the conclusion.

## Phase two: worktree reset and the dense-enumeration investigation

The user designated main as canonical. Older worktrees became historical
evidence and mechanism donors, not branches to merge wholesale. Durable
research documents and selected receipts were retained; temporary
implementation, configuration, and test surfaces were intentionally not
carried forward. The continuing worktree was named research-probes.

The project established a research graph:

    executed artifact
      -> research unit
      -> investigation synthesis
      -> research decision
      -> next discriminator.

Research holds scientific interpretation and links; source and immutable
artifacts retain runtime and numerical authority. Stable OpenSpec contracts
are reserved for compatibility-sensitive reusable seams, not for hypotheses
or pilot results. Abbreviations, arm names, metric symbols, and legacy run
tokens must be expanded and operationally explained. Gaussian RPS was retained
only as historical provenance; the canonical name is Gaussian Soft-Target
Coordinate Cross-Entropy with Ordered Cumulative-Distribution Penalty.

### Spatial scope, history policy, and bagging

The first dense-enumeration unit separated several confounded claims:

* equal call count is not equal per-object opportunity;
* full-canvas masked reset versus masked cumulative is a total prefix-policy
  contrast, not a pure token-length experiment;
* native-scale tiles, full-canvas masking, and full-image bagging change
  different parts of the system;
* seed allocation, stop behavior, mask fill, diversity, ownership, and merge
  semantics need explicit receipts.

The Dense-Union-51 execution was difficult because the initial repository did
not have genuine request-scoped sampled decoding. The eventual work built and
validated that narrow execution seam. Several real defects mattered:

* a monkeypatched sampler could be bypassed by Transformer dispatch;
* official COCO image identifiers and evaluator-local identifiers could be
  conflated;
* data-parallel resume/rebind behavior could relabel requests incorrectly;
* native tiles could be visually mismatched with prompt image plans;
* a rounded zero-area box should be parsed and invalid, not cause a run-wide
  abort;
* per-wave batch-three tails had a different partition contract;
* global object matching is not the union of per-call matchings.

After two execution roots and an all-batch-four sensitivity check, the
scientific verdict was stable. A single masked owning opportunity improved
locally by approximately 0.194 and 0.207 in the two roots, but the final masked
policy did not beat equal-invocation full-image bagging, by approximately
minus 0.058 and minus 0.030. Cumulative accepted-row policy was harmful.
Native-scale tiles were worse. Masking failed retention, precision, and
prediction-inflation gates.

Bagging found roughly 73 percent of 948 audit-accepted objects versus
approximately 65 percent for masked full canvas, but its low precision mixed
duplication, fragmentation, true hallucination, and residual annotation
ambiguity. The conclusion was neither “masking is useless” nor “bagging is the
solution.” A local input-level spatial restriction effect exists; a safe final
masked policy did not emerge. Bagging is a support probe and a source of
candidate trajectories, not an inference policy.

The user then explicitly reset the research flow. Exploratory work should
start from an outline and a first real observation, not from a frozen API or
heavy contract. Only conclusion-changing defects block an exploratory pilot.
Use curated four-to-eight-image case studies, let visual review lead where
annotations are incomplete, retain compact receipts, and defer elegance,
future consumers, and broad infrastructure until evidence justifies them.

## Phase three: fixed-prefix modes, phrase-geometry gates, and numerical boundaries

### Sampled rescue and phrase-geometry compatibility

The next question was why objects seen in sampled trajectories disappear from a
greedy route. On image 12576, the greedy trajectory emitted 14 rows and
accepted 13 objects but missed pizza. At exact prefix state 56, low-temperature
sampling exposed left cup in 21 of 32 first rows and pizza in 11 of 32; at a
later greedy-terminal state pizza disappeared. A second chair case supplied
independent state-entry versus terminal evidence.

A forced description token could produce a phrase-geometry-bound complete row,
so once selection was made, transcription was often not the first bottleneck.
However, a complete-row factorial showed a strict compatibility condition:
only the coherent cup phrase plus left-cup geometry advanced to right cup.
Phrase alone, geometry alone, no-row, and covered duplicate controls did not
produce that advancement. This supports a bounded phrase-and-geometry
transition gate; it does not decide between visual object commitment and
geometry-sorted text serialization.

Two counterfactual branches narrowed one tempting explanation. Swapping the
selected local post-vision left-cup support and later replacing the raw
left-cup bounding-box pixels before visual encoding left the right-cup
successor unchanged in paired samples and greedy decoding. This closes the
tested claim that ongoing local committed-object support is behaviorally
necessary at that state. It does not prove visual independence, a pure textual
transducer, or a ledger.

### Batch precision and a spatial-routing seam

Image 7574 exposed a more basic confound. The same coherent white-bowl prompt
repeated white bowl in physical batch one and advanced to orange bowl in
physical batch four. Under bfloat16, homogeneous copies produced a broad
coordinate-distribution switch. Full-model float32 made cached and direct
layouts practically invariant for that recipient.

Feature replay showed that primary visual features and all DeepStack streams
were byte-identical across those layouts; replaying batch-one features into
downstream batch four did not recover the batch-one logits. The causal locus
for that exact split was after get_image_features. A six-case prevalence panel
then found first-action bfloat16 differences in three cases, but they were
one-to-two source-pixel same-instance shifts removed by float32. The isolated
broad bowl shift remained a counterexample, not a license to call every exact
token mismatch a different object transition.

The program froze a same-encoding post-vision spatial-key eligibility
intervention. It showed a real region-conditioned routing seam:

* one of five valid anchors completed a semantic-plus-geometry owner switch;
* one switched geometry without semantic ownership;
* others were one-sided or destructive.

Finite positive soft spatial bias strongly changed likelihoods but produced no
owner switch across the frozen dose panel. Hard eligibility effects relied
mainly on nonlinear competing-row collapse. Query-time factorials showed that
direct row reads carried much of geometry routing, while matched early and
row-query restriction produced one semantic outcome mostly through non-owner
suppression. A crossed-region hybrid was asymmetric. Count-balanced soft
cross-bias failed its matched controls, so crossed arms were not interpreted.

The attention-bias family was retired. It offered a useful causal seam, but no
support for a safe attention policy, a recurrent instance pointer, or a
general competition-release mechanism.

### Residual portability and boundary-coherence limits

Hard routing created an opportunity to ask whether its effect compiled into a
portable residual state. At image 139, layer 23 restored roughly 98.4 percent
of a donor semantic-confidence path. At image 632, strong coordinate-path
actuation did not produce a donor-owned tight box, so this was an eligibility
failure rather than a negative transport result.

After donor eligibility was repaired, an image 7818 paired wine-glass state at
layer 23 switched the unrestricted continuation to the paired instance with
donor intersection-over-union 0.800259; layer 13 did not. Nearly all measured
effect was at x1. This is bounded one-way, owner-specific first-coordinate
geometry-basin portability, not a transferable four-coordinate object file or
an architecture choice.

Human review of 119 comments provided a complementary geometry lesson: y2
often ended too early, and coordinate correctness was neither all-or-none nor
fully independent. COCO modal or visible geometry became primary, with
plausible amodal extent a secondary reference. Off-support coordinate transport
and part-versus-whole tests found real grammar compatibility but failed
common-support admission. Their later waves, training screen, and architecture
promotion correctly did not run.

## Phase four: natural row transitions, order sensitivity, and what they do not prove

Natural-prefix work replaced increasingly off-manifold coordinate surgery.

At prefix state 56, the model showed directional row transitions: a pizza child
could lead to left cup and a left-cup child to right cup. Image 15254 showed a
strict bowl-carrot reciprocal crossover across 231 paired variants under
float32, but their boxes were highly nested. Image 2299 showed that appending
one person B lowered B's frozen row by about 6.25 natural-log units and raised
four other person variants by 1.39 to 2.44, primarily at x1. Yet natural
sibling sampling was concentrated: one near successor appeared 153 of 160
times, leaving insufficient support for the intended layer trace or training
claim.

The correct conclusion was that native rows can create executable,
geometry-sensitive successor state. It was not that the model maintains a
general physical-object covered set.

Historical random-order and geometry-sorted checkpoint-3668 adapters then
provided a useful but non-identical comparator. Random order sent every tested
image 2299 state toward one person, a habitual attractor. Geometry-sorted
training made next owner and x1 respond to prior geometry. The historical pair
was not a strict comparator to modern step-4887 DoRA, but it did reject the
claim that random row order automatically creates order-free coverage.

The later modern pure-cross-entropy random-versus-sorted screen confirmed the
same direction. Random order changed routes but did not show symmetric
covered-set behavior; on image 2299 it collapsed to a habitual person-rank-18
successor. Sorted pure cross-entropy remained path-sensitive but kept more
route diversity and became the practical mechanism baseline.

The same-covered-set experiment supplied the most important behavioral
constraint. With physical covered set, row count, final row, prompt, model,
and paired seeds held fixed, changing earlier order altered the future
four-row physical-object set in 99 of 216 pairs; adjacent swaps changed 10 of
36. Image 19109 showed order-dependent recurrence of a covered motorcycle.
Only eight of 216 pairs changed completion or termination. When both arms
selected the same physical instance, their boxes were normally stable. This
means many apparent same-description coordinate changes are different
same-class instance selections, not generic box noise.

Candidate-row scoring made the training consequence sharper: full-row average
log-probability can be misleading because easy later tokens obscure the
earliest branch choice. The earliest distinguishing token, often category or
x1 and sometimes y1, is the operative local decision. STOP was not a universal
bottleneck. Complete-row scores remain useful explanation, not the primary
free-rollout objective.

Subsequent sampled-history causal-value work found three same-owner
coordinate-history effects. Only image 7816 gave a safe final unique-owner
gain, from nine to ten. There, sampled x1 or y1 in row four could unlock the
later route; y2 alone could not. Image 12576 required the joint history of two
earlier rows. Other positive-looking changes exchanged owners. This rules out
one universal latest-row carrier, general sampled-coordinate labels, and a
simple recurrent defect.

## Phase five: treatment screens and why their apparent positives were not promoted

### Coordinate boundary treatment

The program deliberately compared the existing Gaussian coordinate objective
against pure cross-entropy before layering another loss on top. Under modern
matched validation, pure cross-entropy had stronger aggregate metrics and a
cleaner current-sampler coordinate signature. Pure cross-entropy became the
mechanism baseline; Gaussian smoothing and the ordered cumulative-distribution
penalty remained an ablation.

An eight-event Smoke B then showed that coordinate-preference training could
improve held-out exact-prefix coordinate margins. It did not improve free
rollout, matched-box geometry, or health relative to matched controls, and it
increased duplicate candidates. A 256-image one-epoch screen was nevertheless
run after the user explicitly judged two steps insufficient and authorized a
larger treatment screen. It improved fixed-prefix margins in 242 of 256
events at learning rate 1e-5 and 241 of 256 at 3e-6, but clean rollout did not
improve at either rate. The lower rate reduced disturbance but still lowered
mean average precision.

The belief update is not “coordinate supervision cannot work.” It is:
selected-coordinate correction can be locally learnable while failing to
preserve owner identity, whole-box coherence, and self-generated prefix state.
Do not scale that objective unchanged. An old-and-refreshed-prefix follow-up
also failed to isolate a convincing prefix-only improvement; its strict
prefix-only subset was small and its interval crossed zero.

### Trajectory support, forced paths, and single-route imitation

The human-refined twelve images were reclassified as development and validation
cases, not training data and not a final blind cohort. They contain 346 refined
objects and are valuable because physical-entity discovery and geometry can be
reviewed separately.

Twelve images times one greedy plus 16 sampled trajectories produced 204
trajectories. The audit found both phenomena:

* some images had a best individual sampled trajectory whose conservative
  coverage lower bound beat greedy's upper bound without more harm;
* other useful instances were distributed across complementary trajectories.

This ruled out a false choice between “learn one route” and “only unions
matter.” The valid unit of treatment might be a trajectory or route family,
but any positive credit must retain ordinary owners and avoid turning unknown
annotations into negatives.

Forced-path tests on four representative cases passed three of four
promotion-style criteria. They showed that some sampled row fragments can
unlock a real missed owner from the exact greedy prefix, but the useful
intervention depth differed by case. Beam search was deferred: low-temperature
sampling already exposed diverse owner routes, whereas beam would introduce
shared-prefix, length, and early-stop confounds.

The final single-route positive-row imitation screen used 8 GPUs, 512 events
from 118 images, one epoch, and checkpoints at steps 5, 10, 15, and 16.
Step 15 was the best diagnostic checkpoint:

| Measure | Source | Step 15 | Meaning |
| --- | --- | --- | --- |
| Train-256 mean average precision | 0.3880 | 0.3913 | Small metric lift, not a coverage win |
| Unique annotated owners | 1,677 | 1,658 | Lost 19 owners |
| Predictions | 2,978 | 2,689 | Fewer rows, not better set completion |
| Fixed route-added owners | 93 | 109 | Recovered 16 selected-route owners |
| Ordinary route owners | 712 | 697 | Lost 15 ordinary owners |
| Total admitted-image coverage | 907 | 907 | No net expansion |
| Non-admitted-image owners | 770 | 751 | Transfer regression |

Only 118 of 238 route-added owners were direct event targets. Direct targets
moved from 61 to 63, while same-route but non-direct owners moved from 32 to
46. The treatment therefore changed a route-family selection distribution,
not owner-by-owner memorization or final-set expansion. The final step 16
regressed more sharply. This exact objective is closed and should not scale to
1,024 images or deployment.

The next treatment hypothesis is a matched-budget comparison:

    frozen Source
      -> single-route only
      -> single-route plus Source-route preservation
      -> multi-route plus the same preservation anchor.

It must require route-added gain, ordinary-owner retention, and no
non-admitted regression. It remains a hypothesis, not an authorized launch.

## Data, annotation, and evaluation lessons

Several distinctions became non-negotiable:

* an unmatched Common Objects in Context prediction is not automatically an
  entity hallucination;
* entity/category discovery and geometry/physical extent are separate axes;
* a true object can have shifted, partial, oversized, or
  neighbor-contaminated geometry;
* automatic matching must use global one-to-one physical-owner assignment;
* ambiguous matches create a conservative lower bound and a review-expanded
  upper bound, never double credit;
* crop-enlarged human review owns conclusion-changing ambiguous cases;
* forced context rows are never autonomous discoveries;
* visible/modal extent is primary for this work, while plausible amodal extent
  is explanatory secondary evidence.

This matters scientifically and for training. High-confidence positives,
duplicates, invalid rows, and known harmful rows can train a treatment. Unknown
or annotation-ambiguous rows should remain neutral rather than become false
negatives. In dense scenes, a low official precision can mix missing labels,
valid fragments, duplicates, semantic errors, real hallucinations, and
geometry problems.

## Mathematical and statistical framing that survived the session

The aspirational enumeration process resembles weighted selection without
replacement. If U_t is the set of reportable but uncommitted objects at step t,
an ideal transition selects an object in U_t, binds phrase and geometry,
commits the row, redistributes probability toward U_(t+1), and stops only when
no supported object remains.

This is a useful idealization, not an established internal model. The current
model lacks a proven explicit U_t. Text prefix and cached state jointly encode
some traversal, recent-row, geometry, and stopping information.

The July 22 “health bar” discussion rejected a literal conserved object-count
budget. A more useful diagnostic model is a candidate-specific continuation
margin:

    margin_t(owner) =
        score(owner row | image, prefix_t)
        - score(STOP | image, prefix_t).

The margin can change with global stop tendency, local same-class competition,
geometry, and route-specific prefix state. It should be probed at the earliest
actual branch token. It is not a scalar resource whose decrease proves
displacement or an endogenous count ledger.

Teacher-forced likelihood, full-row likelihood, token ranks, and margins
remain explanatory tools. The conclusion authority remains free native greedy
rollout, owner coverage, duplicate and invalid behavior, termination reason,
and visual review.

## Durable research and workflow philosophy

The user repeatedly established the following principles:

* Preserve Qwen3-VL capability; do not escape immediately to an external
  detector, a DETR-style query bank, slots, or a separate stronger backbone.
* Treat visual designation and hard interventions as causal teachers, not
  final interfaces.
* Do not infer a coverage ledger from a human-annotator analogy or from one
  reciprocal transition.
* Use small, falsifiable, high-information research units before expensive
  scale, but use the available GPUs when a coherent treatment has crossed its
  smoke gate.
* A mechanism study should identify a treatment or a discriminating decision;
  it should not merely accumulate yes-or-no findings.
* Keep major semantic changes and material cost decisions visible to the user,
  but allow autonomous bounded work when the objective is already authorized.
* Compare native versus forced behavior, bfloat16 source behavior versus
  float32 confirmation, current versus historical checkpoints, distinct
  decoding policies, and on-support versus off-support interventions without
  collapsing their meaning.
* Favor visual review and exact artifacts over aggregate scores when working
  on small dense cohorts.
* Do not make unknown labels negative, do not call every mismatch a
  hallucination, and do not use a forced row as model-discovered coverage.
* Low-temperature sampling is currently a donor-discovery method. Beam search
  is deferred, not rejected forever.
* Worktree boundaries matter: main is canonical; old worktrees are evidence
  donors only; preserve durable research before dropping temporary code.

The team also learned to route agents empirically. Bounded mechanical
discovery and artifact lookup can use less expensive workers. Code-contract
reviews often benefit from an independent model family. Causal and
conclusion-critical interpretation needs the strongest available reviewer.
This is a provisional operational observation, not an immutable
role-to-model policy.

## Rejected, deferred, or superseded paths

The following should not be quietly reopened without new evidence:

* A mutable post-scatter visual cursor is not a demonstrated production
  interface under cached decoding.
* The A/B/C proposal bridge is a generic continuation carrier, not
  object-specific visual-text pairing.
* Full-canvas masked policy and native tiles are not accepted final
  enumeration policies under the executed protocol.
* Uniform positive spatial-key bias and the count-balanced phase-restricted
  soft operator are closed. Hard routing is not an inference policy.
* One late-middle residual patch or one layer-23 x1 transfer is not a complete
  object state or an architecture selection.
* A strict covered-set carrier, persistent ledger, cursor renderer, slots, and
  external detector are not selected.
* Random complete-row order did not establish order-free coverage.
* Off-support coordinate transport and a single early coordinate do not prove
  a complete object owner.
* Sampled-row novelty and a single natural sibling do not define a safe
  preference target.
* First-wrong-coordinate loss, generic coordinate consistency, and the
  executed single-route imitation objective must not be scaled unchanged.
* Global terminal suppression is diagnostic only and cannot be substituted for
  correct next-object selection.

## Terminal active state and continuation

At the historical snapshot boundary, the user had authorized a new
human-refined greedy set-completion study on the twelve refined validation
images. Its question is:

    What least ground-truth-derived prefix assistance lets frozen native greedy
    decoding enumerate the remaining trusted physical objects, and where does
    it first fail when it cannot?

Its planned controls are intentionally diagnostic:

* ground-truth prefix schedules in geometry-sorted, reverse,
  category-grouped, and fixed-random orders;
* separate same-remaining-set order controls;
* remaining-object depths N, 16, 8, 4, 2, and 1;
* native greedy suffix;
* one-time and repeated STOP-suppression diagnostics, recorded separately;
* a forced complete trusted row followed by native release, with exact native
  row no-op replay before interpretation;
* strict and relaxed complete-row budgets;
* a non-binding token ceiling with zero truncation required for evidence;
* full-model float32, physical batch one per independent process, and
  repetition penalty 1.0.

The study is exploratory: it should collect a completion surface, mine
partial-success and failure families, and then run only selected causal
replays. It must not substitute teacher-forced likelihood for rollout,
introduce training, beam search, slots, a coverage carrier, an external
detector, or visual masking.

The transcript ended while the minimal runner was still being prepared. In the
post-boundary live state, thirteen JSON receipts were then written for all
twelve designated images. They record 380 valid arms with no token-limit
invalid evidence, but there is no aggregate/report and some parser outcomes
contain dropped spans. Treat the protocol as executed evidence awaiting
receipt-level review and scientific interpretation, not as a success claim or
an authorization to train.

## Important source map

Primary formal reading path:

1. research/index.md
2. research/investigations/qwen3-vl-dense-enumeration/compass.md
3. research/investigations/qwen3-vl-dense-enumeration/overview.md
4. research/investigations/qwen3-vl-dense-enumeration/2026-07-13-to-2026-07-16-weekly-research-report.md
5. relevant experiment unit.md and results.md records under
   research/investigations/qwen3-vl-dense-enumeration/experiments/

High-value result records include:

* 2026-07-13-spatial-scope-history-disentanglement
* 2026-07-14-sampled-rescue-object-transition-causal-replay
* 2026-07-15-prefix-state-phrase-geometry-factorial
* 2026-07-15-pre-vision-raw-bounding-box-visual-support-counterfactual-commit
* 2026-07-15-object-centered spatial eligibility and downstream routing units
* 2026-07-16-fixed-encoding-persistent-hard-routing-geometry-state-portability-image7818
* 2026-07-17-native-sibling-row-branch-value-and-commit-crossover
* 2026-07-19-same-covered-set-prefix-order-equivalence
* 2026-07-19-common-object-prefix-permutation-short-horizon
* 2026-07-19-sampled-history-target-reachability-and-complete-row-value
* 2026-07-20-matched-random-sorted-prefix-order-screen
* 2026-07-20-own-prefix-entity-transition-and-coordinate-boundary-calibration-training-screen
* 2026-07-21-256-image-coordinate-boundary-training-screen
* 2026-07-21-greedy-prefix-forced-owner-path-intervention
* 2026-07-21-individual-trajectory-versus-union-support-audit
* 2026-07-21-best-sampled-trajectory-positive-row-imitation-screen
* 2026-07-22-human-refined-greedy-set-completion-conditions

Important model and output handles:

* current mechanism-source checkpoint:
  /data/CoordExp/.worktrees/CoordExp-swift/outputs/prod/coordexp_swift/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_typegate_dora_r16a32_llm_12000_acceler8_ebs24_8epoch_warmup0p1/checkpoints/step-4887/checkpoint.json
* coordinate-screen outputs:
  /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-21-256-image-coordinate-boundary-training-screen/
* forced-owner-path outputs:
  /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-21-greedy-prefix-forced-owner-path-intervention/
* positive-imitation outputs:
  /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-21-best-sampled-trajectory-positive-row-imitation-screen/
* reviewed trajectory-support audit:
  /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-21-individual-trajectory-versus-union-support-audit/support-audit-step4887-20260721a/analysis/reviewed-support-v3.json

## Reconstruction limits

This recap records the evolution instead of pretending the final view was
obvious from the beginning. It intentionally preserves corrected claims:

* local committed-object pixel erasure is a null for one revalidation gate,
  not proof of visual independence;
* batch-four behavior is not intrinsic native commitment when batch-one
  differs and float32 removes the selected numeric split;
* a score change is not a rollout change;
* an apparent coordinate change can be a same-class owner switch;
* a sampled rescue is not automatically a safe final-set improvement;
* a training gain on a selected route is not direct owner imitation or net
  coverage gain.

For any future claim, re-open the linked unit, result record, resolved
artifact, and live code path. Do not treat this memory as a substitute for
that verification.
