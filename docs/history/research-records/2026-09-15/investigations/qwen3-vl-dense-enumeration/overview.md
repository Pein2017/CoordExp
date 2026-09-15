---
title: Qwen3 Vision-Language Dense Enumeration Bottleneck
description: Detailed hypothesis and evidence atlas for object-mode fragmentation, executable prefix state, post-vision spatial routing, and late first-coordinate owner selection in dense autoregressive detection.
type: investigation
status: active
topic: qwen3-vl-dense-enumeration
updated: 2026-09-09
---

# Qwen3 Vision-Language Dense Enumeration Bottleneck

The [program research compass](compass.md) owns the north-star objective,
current belief register, discriminator queue, architecture non-commitment, and
candidate paper thesis. This overview remains the detailed hypothesis and
evidence atlas. The [weekly integrated research
report](2026-07-13-to-2026-07-16-weekly-research-report.md) is the compact
supervisor handoff for the executed 2026-07-13 through 2026-07-16 sequence.

## Current Research Route

The [current compass](compass.md) owns the current question-oriented synthesis
and evidence boundaries. The atlas below preserves historical hypotheses and
observations; the [experiment router](experiments/index.md) leads to original
protocols and results.

## Terminology and Name Registry

- **Qwen3-VL — Qwen3 Vision-Language**: the pretrained multimodal model family
  under investigation.
- **COCO-80 — Common Objects in Context 80-category ontology**: the closed set
  of reportable object categories used by this investigation.
- **STOP — terminal no-more-objects decision**: the model action that ends
  object enumeration; it is not assumed to be a calibrated coverage verifier.
- **Full-Image Single Rollout (`FULL_SINGLE`)**: one sampled autoregressive
  rollout over the complete image using an independent baseline seed that is
  not reused in the multi-call arms.
- **Full-Image K-Rollout Independent Bagging (`FULL_BAG_K`)**: `K` independent
  full-image rollouts aggregated with the same merge policy as the spatial
  arms; `K` is the number of spatial cells. It matches total call count, not
  sampled opportunities per object or total computation.
- **Full-Canvas Masked Region with Per-Region Reset (`MASK_RESET`)**: `K`
  full-size canvases, each exposing one spatial region, decoded from a fresh
  prompt.
- **Full-Canvas Masked Region with Cumulative Accepted-Row Prefix
  (`MASK_CUMULATIVE`)**: the same masked canvases decoded while retaining rows
  admitted and serialized by a frozen prompt-state transition.
- **Native-Scale Tile with Per-Tile Reset (`TILE_RESET`)**: unresized image
  tiles decoded independently.
- **Owning-seed per-object comparison**: compare an object's raw owning-cell
  outcome with the raw full-image outcome using the same seed, thereby matching
  one sampled opportunity without claiming equal input.

## Question

Why does object-level capability remain strong while a single full-image
autoregressive rollout becomes short, conservative, duplicate-prone, or
invalid as scene density and sequence length grow?

The working decomposition is:

```text
static image evidence
  -> choose a spatial scope
  -> identify one reportable object
  -> bind phrase and geometry
  -> emit and commit the row
  -> recover the next uncovered object
  -> stop only when no supported object remains
```

The present investigation does not assume that a ledger, slot, cursor, or
specialized detector is required. It asks which part of this loop first needs
an external intervention.

## First-Principles Mechanism Candidates and Invariants

### Masked-Input Spatial Restriction Utility (`HYPOTHESIS_MASKED_SPATIAL_POLICY`)

Exposing one core-plus-halo region on a full-size masked canvas may recover
objects more reliably than one seed-matched full-image opportunity and may add
final utility beyond matched-call full-image bagging. This is a policy-level
hypothesis: masking changes pixels and visual activations before the vision
tower.

The deeper **Post-Vision Spatial Candidate Competition
(`HYPOTHESIS_POST_VISION_COMPETITION`)** mechanism proposes that simultaneous
object candidates in one fixed full-image representation suppress valid
continuations. The initial masked-policy unit did not test it directly. Later
fixed-encoding units established a causal spatial-routing seam but did not show
that harmful recurrent competition generally owns dense-scene low recall.

### Accepted-Row Prefix Policy Interference (`HYPOTHESIS_ACCEPTED_ROW_PREFIX_POLICY`)

The declared cumulative accepted-row prompt policy may harm later rescue
relative to per-region reset. This comparison bundles prompt length, semantic
content, correctness, order, visibility consistency, and state reconstruction;
it estimates a total policy effect rather than pure memory horizon.

The deeper **History-Horizon Instability (`HYPOTHESIS_HISTORY_HORIZON`)**
mechanism claims that length alone harms retrieval under matched content,
correctness, order, visual input, and decode processing. It requires a later
controlled prefix panel.

### Tile-Local Computation Explanation (`HYPOTHESIS_TILE_LOCAL_COMPUTATION`)

Native-scale tiles may help because they contain fewer image tokens or because
the vision tower encodes a local frame differently, not because the language
decoder received a better search scope. Full-size masked-canvas arms are needed
to preserve the nominal full image-token grid while restricting visible
content, but they do not preserve the encoded visual features.

### No-Resize Scale Invariant (`INVARIANT_NO_RESIZE_SCALE`)

Conventional resized tiles improve small-object recognition by changing scale.
This is a protocol invariant rather than a causal hypothesis: the investigation
forbids resizing so that any tile benefit cannot be attributed to object
magnification.

### Partition and Boundary Context (`HYPOTHESIS_CONTEXT_BOUNDARY`)

Hard tiles can cut objects or remove needed context. Core-plus-halo inputs with
unique core ownership separate useful local context from naive union and
non-maximum-suppression gain.

### Geometry-Sorted Traversal Prior (`HYPOTHESIS_GEOMETRY_ORDER`)

The adapter was trained with geometry-sorted rows. Raster cell order can align
with that prior, especially when outputs accumulate in the prefix. Multiple
counterbalanced cumulative orders plus reset-order controls can establish order
sensitivity; causal attribution to training order still requires a
random-order-trained checkpoint.

### Dense-Scene Annotation Gaps (`HYPOTHESIS_ANNOTATION_GAPS`)

COCO annotations are incomplete in dense scenes. Structured decomposition may
recover real COCO-80 objects that official evaluation counts as false
positives. A blinded, manually audited subset is required before interpreting
precision loss as hallucination.

### Residual Recognition or Control-State Limit (`HYPOTHESIS_RESIDUAL_LIMIT`)

Objects that remain missed under matched-call full-image bagging, no-resize tiles,
masked scope, and short history raise the posterior for a genuine recognition
limit or an inability to compile object-specific visual evidence into decoder
control. This unit cannot distinguish those two residual explanations.

### Fixed-Prefix Object-Mode Fragmentation (`HYPOTHESIS_FIXED_PREFIX_OBJECT_MODES`)

Repeated full-image bagging may recover different objects because several valid
next-object modes coexist at one identical image and prefix boundary. The
strongest alternative is earlier trajectory divergence: different early rows
may create different later prefix states, while each fixed state remains
nearly unimodal. Existing bagging does not distinguish these explanations.

### Phase-Specific Late-Middle Control (`HYPOTHESIS_PHASE_SPECIFIC_LATE_MIDDLE_CONTROL`)

Historical painted-to-clean probes found that stop-versus-continue control
became strongly readable and causally patchable mainly around language-model
layers 17 through 21, while description identity became strong mainly around
layers 20 through 23. A one-time patch did not recover an exact full row, so
the supported hypothesis is phase-specific control in late-middle residual
states, not one universal decision layer. The [research compass](compass.md)
records the exact evidence handles and claim boundary.

### Bagging-to-Greedy Learnability (`HYPOTHESIS_BAGGING_TO_GREEDY_LEARNABILITY`)

Verified missed objects now have stable probability support in one bounded
fixed-prefix case. Whether a training objective can concentrate that
stochastic support into greedy uncovered-object transitions remains
speculative because no nontrivial endogenous state target, safe 256-image
cohort, or mechanism-level training arm has yet passed its gate.

### Phrase-Geometry Compatibility-Gated Transition (`HYPOTHESIS_PHRASE_GEOMETRY_TRANSITION`)

An appended row may update the successor state only when its phrase and
geometry form a compatible object event. At exact image-`12576` prefix state
56, only `cup` plus left-cup geometry advances to the right cup; either factor
alone returns the left cup. This rejects independent phrase and geometry main
effects in that case. Replacing the selected local post-vision left-cup support
and, in a separate complete visual re-encoding, the raw left-cup bounding-box
pixels did not alter that successor in greedy decoding or any of eight paired
samples. A text- or prefix-mediated transaction followed by visual selection
of the still-visible right cup is now the leading bounded explanation, while
visual independence remains unproven.

## Belief-Update Rules

| Observation | Belief update |
|---|---|
| Owning-seed raw superiority and safe post-merge masked superiority both pass | Supports masked-input spatial-policy utility; raises but does not establish post-vision competition. |
| Owning-seed raw masked advantage is positive, but bagging catches up post-merge | Local restriction helps one opportunity, while repeated full-image opportunities recover comparable final utility. |
| Post-merge masked advantage appears without an owning-seed raw advantage | Aggregation, merge, ownership, or opportunity allocation remains the leading explanation. |
| `MASK_RESET` safely exceeds `MASK_CUMULATIVE` under neutral repetition penalty | Supports harm from the declared cumulative accepted-row prefix policy; does not isolate pure history length or a missing ledger. |
| `TILE_RESET` exceeds `MASK_RESET` while `MASK_RESET` matches `FULL_BAG_K` | Raises fewer-image-token, local-frame, or vision-side explanations. |
| Matched-call bagging matches structured arms, owning-seed differences are equivalent, and mask retention passes | No incremental masked-input utility is detected under this protocol; generic sampling is a sufficient policy comparator but is not thereby proven to be the mechanism. |
| Mask retention fails | A null masked comparison is inconclusive because intervention harm may cancel scope benefit. |
| Official precision falls but blinded manual precision holds | Supports annotation incompleteness rather than hallucination. |
| The same visible objects remain missed by every arm | Raises residual recognition/control-synthesis explanations. |

## Claim Boundary

No tile or masked-canvas result by itself proves prior full-image recognition,
a pure language-side disease, post-vision candidate competition, history-length
failure, or a missing ledger. Both input-level spatial interventions alter
visual computation. Matched call count does not match per-object opportunities.
Reset versus cumulative estimates a total accepted-row prefix-policy effect.
The later same-feature spatial-eligibility crossover establishes a usable
region-conditioned routing seam but does not establish recurrent complete-row
switching or general harmful full-image competition. The follow-up finite
positive-bias panel establishes graded regional reweighting but no semantic or
geometric owner reversal, so a mild context-preserving attention preference is
not sufficient on the frozen anchors.

## Historical Evidence Update — retained atlas

This section preserves the earlier evidence narrative and its original route
language. The current question-oriented synthesis above is authoritative for
today's cross-direction summary; these older paragraphs remain source-linked
context, not a second current frontier.

The completed [Masked Spatial Policy and Accepted-Row Prefix Policy
Disentanglement unit](experiments/2026-07-13-spatial-scope-history-disentanglement/unit.md)
executed two independently derived seed roots on the sealed Dense-Union-51
cohort. [Its results](experiments/2026-07-13-spatial-scope-history-disentanglement/results.md)
support a reproducible local input-level spatial-restriction effect in one
seed-matched owning opportunity. They do not show a final masked-policy benefit
over equal-call full-image bagging, and the masked policy fails retention,
mask-harm, manual-precision, and prediction-count safety gates.

The same evidence shows that the complete cumulative accepted-row prefix policy
is harmful relative to reset and that native-scale tiling is worse than
full-canvas masking. Because the cumulative intervention bundles length,
content, correctness, order, and visibility consistency, pure history-horizon
instability remains unresolved. Post-vision competition, a language-only root
cause, and the need for a ledger or architecture change also remain unresolved.

The bagging-trajectory re-analysis, fixed-prefix one-row distribution, and
sampled-rescue causal replay sequence is now complete. The [verified
results](experiments/2026-07-14-sampled-rescue-object-transition-causal-replay/results.md)
show two valid object modes at one exact prefix and state-dependent rescue in
two cases. They also localize strong within-row description-conditioned
binding while ruling out strict object-specific
commit-to-uncovered redistribution for the primary case.

The completed [Prefix-State Phrase-Geometry
Factorial](experiments/2026-07-15-prefix-state-phrase-geometry-factorial/results.md)
rejects standalone phrase, standalone geometry, and uniform complete-row
advancement explanations at one exact state. It supports a left-cup-specific
phrase-and-geometry transition gate while leaving visual grounding versus
textual geometry-sorted serialization unresolved.

The completed [Visual-Support Counterfactual Commit Test](experiments/2026-07-15-visual-support-counterfactual-commit/results.md)
finds no detectable dependence on the selected local post-vision left-cup
support under an exact donor-substitution operator. The `8/8`, `8/8`, `8/8`
successor result raises a text-mediated commit explanation without proving a
purely textual transition because visual evidence may already be globally
contextualized.

The completed [Pre-Vision Raw-Bounding-Box Visual-Support Counterfactual Commit
Test](experiments/2026-07-15-pre-vision-raw-bounding-box-visual-support-counterfactual-commit/results.md)
replaced the complete raw left-cup bounding-box pixels before contextual visual
encoding. Clean, target, and equal-shaped control conditions all produced the
right-cup first action in eight of eight paired samples and greedy decoding;
the Pixel-Support Effect was `0.0`. This strong null closes the committed-
object-support branch. No architecture or 256-image training screen is
promoted. Its proposed own-rollout-row-correction follow-up is retained as a
historical branch idea; the subsequently executed fixed-encoding and residual-
portability chain now owns the current discriminator boundary.

The completed [Native Coherent-Row Commit-to-Uncovered Redistribution
Factorial](experiments/2026-07-15-native-coherent-row-commit-to-uncovered-redistribution/results.md)
shows that prefix geometry can act as a visual address: crossed description-
geometry rows cause the model to re-identify and repair the physical object at
the supplied region. The batch-four panel also produced perfect factual-row
advancement to distinct same-category successors. However, the exact coherent
white-bowl prompt repeats the original bowl at physical batch size one and
advances to an orange bowl at physical batch size four, independently of a
`16`- versus `512`-token maximum generation horizon. Stable native commit is
therefore held.

The completed [Mixed-Length, Homogeneous, and Equal-Length Batch Coordinate-
Logit Invariance Probe](experiments/2026-07-15-homogeneous-mixed-batch-coordinate-logit-invariance/results.md)
locates the immediate confound. Under `bfloat16`, four identical target
recipients are sufficient to move the complete coordinate distribution from
the white-bowl basin to the orange-bowl basin; equal-length semantic companions
and target batch position add no effect. The shift is broad rather than a
single near tie. Full-model `float32` makes cached and direct execution
practically invariant across every tested layout and selects the same orange-
bowl mode. This closes semantic cross-request contamination for the exact
recipient, but holds the subsystem origin and population relevance. The next
locus is the first vision-to-decoder seam where the `bfloat16` batch-shape
difference becomes material, not a ledger or training intervention.

The completed [Single-Target Visual-Feature Replay into Homogeneous Batch
Four](experiments/2026-07-15-single-target-visual-feature-replay-into-homogeneous-batch-four/results.md)
closes the vision-output branch. Batch-one and batch-four primary image
features and all three DeepStack streams were elementwise identical. Replaying
the exact batch-one feature bundle into downstream batch four left both cached
and direct coordinate vectors exactly equal to their natural batch-four
references, for a recovery fraction of `0.0`. The causal locus is therefore
after `get_image_features`, while scatter, position preparation, DeepStack
consumption, and language-decoder execution remain unresolved.

The completed [Selected-Transition Batch-Precision Prevalence
Screen](experiments/2026-07-15-selected-transition-batch-precision-prevalence-screen/results.md)
separates exact numerical recurrence from material behavioral recurrence.
Physical batch one and four homogeneous copies differed at the first free
action in three of six selected `bfloat16` transitions, but all three retained
the same description and physical instance and changed only one or two
source-image pixels. Full-model `float32` removed the differences throughout
the complete `64`-token continuation. The result does not replicate or refute
the earlier cell-three partial-row white-to-orange bowl switch because the
screen used cell-zero complete-row transitions. One bounded complete-logit and
same-layout-repeat panel remains before the numeric branch is closed or
promoted.

The completed [Repeated First-Differing-Slot Full-Coordinate-Logit
Panel](experiments/2026-07-15-repeated-first-differing-slot-full-coordinate-logit-panel/results.md)
reproduced all three source `bfloat16` splits twice with zero same-layout noise
and exact homogeneous-copy equality. The perturbations were distributed across
the coordinate vector but low amplitude: root-mean-square shifts were only
`4.17%` to `6.22%` of the image-`7574` broad anchor, and all selected bins
remained in one frozen object neighborhood. Matched exact-recipient `float32`
attenuated the physical-batch shift by `5,342` to `11,278` times. This closes
layer escalation for these three micro-splits without claiming global numeric
invariance or invalidating the isolated image-`7574` broad shift.

The completed [Fixed-Encoding Object-Centered Spatial-Eligibility
Crossover](experiments/2026-07-15-fixed-encoding-object-centered-spatial-eligibility-crossover/results.md)
holds one full-image visual encoding fixed and changes only post-vision
language-decoder image-token key eligibility. Five cases pass the no-op gate.
Image `139` shows a complete regional semantic-plus-geometry owner switch;
image `12120` switches geometry without switching description ownership; the
three same-category cases remain one-sided or destructive. The panel therefore
supports a spatially usable decoder routing seam but not a recurrent instance
pointer or general competition-release mechanism. The windows contain multiple
official object centers in the dense cases, further limiting the result to
region-conditioned routing.

The completed [Fixed-Encoding Soft Spatial-Key Bias Dose
Response](experiments/2026-07-15-fixed-encoding-soft-spatial-key-bias-dose-response/results.md)
preserves all causal keys and adds finite positive bias to one frozen regional
key set. Every case passes float32 no-op and actuation gates, and the later
phrase and geometry likelihoods move strongly and usually monotonically. Yet
none of four anchors at bias `0.5`, `1.0`, or `2.0` reverses complete-row or
geometry ownership, and no phase-specific reversal recurs. Comparison with the
parent hard endpoint shows that its successful reversals are driven mainly by
non-owner collapse under exclusion, not by a smooth increase in target-row
support. Uniform decoder-wide positive soft bias is therefore closed without
closing selective redistribution or query-scoped routing as distinct future
hypotheses.

The completed [Fixed-Encoding Row-Scoring-Query-Only Spatial-Key Eligibility
Crossover](experiments/2026-07-15-fixed-encoding-row-scoring-query-only-spatial-key-eligibility-crossover/results.md)
shows that direct response-row reads are potent and can switch geometry and
same-category spatial ownership. They do not reproduce image `139`'s
cross-category semantic-plus-geometry parent phenotype. The frozen result
therefore supports dependence on some earlier-query computation without
choosing among image-token recompilation, earlier prefix-text computation, or
another nonlinear all-query trajectory effect. The next bounded discriminator
is one image-`139` earlier-query-only arm completing the baseline, row-only,
earlier-only, and all-query two-by-two factorial.

The completed [Fixed-Encoding Earlier-Query-Only Spatial-Key Eligibility
Factorial](experiments/2026-07-15-fixed-encoding-earlier-query-only-spatial-key-eligibility-factorial/results.md)
closes that discriminator. Earlier-only restriction barely distinguishes the
`vase` and `clock` regions: description, geometry, and complete-row crossovers
are `+0.343`, `-0.043`, and `+0.024`. Direct row reads carry the strong geometry
and most complete-row routing. The large all-query semantic crossover appears
only when earlier and row-scoring restrictions are region-matched, and it is
driven mainly by suppressing the competing row rather than releasing the
target row. This supports a phase-separated compatibility or competitive-
exclusion mechanism on one anchor, not an earlier-only object compiler or a
localized mediator.

The completed [Fixed-Encoding Cross-Region Earlier-Query and Row-Scoring-Query
Spatial-Key Eligibility Hybrid](experiments/2026-07-15-fixed-encoding-cross-region-earlier-and-row-query-spatial-key-eligibility-hybrid/results.md)
closes the crossed-region discriminator. Both query intervals alter the first
semantic decision, and geometry-margin sign follows the row-scoring region.
The interaction is asymmetric: Clock-Earlier and Vase-Row produces a pairwise
clock-phrase and vase-geometry chimera, while Vase-Earlier and Clock-Row has
`person` as its top first description token and therefore does not produce the
symmetric chimera. Both complete-row margins contract to near zero. Combined
with the suppression-heavy matched endpoints, this supports directional but
incomplete phase interaction under hard routing; it does not establish native
binding or separate it from an abrupt hard phase-switch artifact.

The completed [Fixed-Encoding Count-Balanced Soft Cross-Region Earlier-Query
and Row-Scoring-Query Spatial Bias](experiments/2026-07-15-fixed-encoding-count-balanced-soft-cross-region-earlier-and-row-query-spatial-bias/results.md)
does not adjudicate that artifact alternative because its matched controls
fail first. Vase-Earlier and Vase-Row predicts `clock` as the first semantic
token and favors clock geometry. Clock-Earlier and Clock-Row predicts `clock`
and has clock-favoring geometry, but its preferred clock geometry score is
suppressed by `1.023263` natural-log units per token relative to unrestricted
scoring. The execution is valid negative evidence; the crossed arms remain
uninterpreted, and this single count-balanced finite soft operator is closed.
The historical lower-dose panel used uniform bias over every causally visible
query row, whereas this operator used separate earlier-query and row-scoring-
query slices; their points cannot be pooled into a dose trajectory.

The completed [Fixed-Encoding Conditional Downstream Layer-Output Residual-
State Portability Gate](experiments/2026-07-15-fixed-encoding-downstream-residual-state-portability-gate/results.md)
supports bounded one-sided layer-`23` portability of the eligible image-`139`
clock description path. Image `632` does not establish geometry portability:
both persistent hard-routing paths close and gain `+2.329` to `+2.544`
natural-log units per coordinate, yet their donor Intersection over Union
(`IoU`) values are only `0.000` and `0.018`. The correct classification is
`no_eligible_donor_control_only`. Strong coordinate actuation is therefore not
equivalent to tight instance geometry, and the image cannot adjudicate whether
a valid donor-owned geometry state would be portable.

The completed [Fixed-Encoding Persistent Hard-Routing Geometry-Donor
Eligibility Screen](experiments/2026-07-15-fixed-encoding-persistent-hard-routing-geometry-donor-eligibility-screen/results.md)
repairs that missing prerequisite. Images `7818`, `12576`, and `2157` each
produce two eligible same-description tight geometry paths under persistent
row-scoring-query hard routing. This falsifies support-envelope-only actuation
on the clean resolution-qualified cases without promoting hard routing as an
inference policy.

The completed [Fixed-Encoding Persistent Hard-Routing Geometry-State
Portability on Image 7818](experiments/2026-07-16-fixed-encoding-persistent-hard-routing-geometry-state-portability-image7818/results.md)
then finds one-way owner-specific portability. With image, phrase, prefix, and
recipient execution fixed, the paired `wine glass` donor state after decoder
block `23` switches the unrestricted geometry owner from annotation `664730`
to `661523` with donor Intersection over Union (`IoU`) `0.800259`; the trusted
block-`13` control does not switch. Almost all measured teacher-forced release
is at `x1`, so the supported mechanism is a late first-coordinate and spatial-
owner basin switch followed by native autoregressive box completion, not a
portable four-coordinate object file.

The completed [Fixed-Prefix Complete-Box Coherence and Progressive Coordinate-
Release Factorial](experiments/2026-07-16-fixed-prefix-complete-box-coherence-and-coordinate-release-factorial/results.md)
then directly tests whether early coordinates organize the rest of a box. In
one dense-chair state, target-versus-adjacent `x1` forcing moves released `x2`
to the cued endpoint on `63/64` paired suffixes. This is strong coordinate-
conditioned geometry transport, but it remains confounded with width, box-
validity, and geometry-sorted continuation. In a visible fork state, the
part-sized row outranks the reportable whole by `2.443562` natural-log units,
and part-versus-whole `x1,y1` forcing yields identical released `x2` on `14/16`
paired seeds. The fork therefore supplies strong case-level evidence for a
part-sized or late-extent basin, while a discriminative-part cause remains
unisolated from trajectory, edge, and width priors. This rejects a universal
early complete-owner account. The chair's forced current geometry also changes the next-row category,
showing executable cross-row state without proving a correct covered set.

The completed [Object-Specific Geometry Transport, Decision Phase, and
Cross-Row Influence Horizon](experiments/2026-07-17-object-specific-geometry-transport-and-cross-row-influence-horizon/results.md)
stops the next hard-clamp route at Wave One. On image `7818`, neither frozen
`x1,y1` history is supported under both donor states. On image `19432`, forced
real and synthetic `x1` histories produce monotonic, valid, naturally closing
`x2` behavior with much more stable translated-width than absolute-coordinate
distributions, but only the target arm passes the frozen 10-percent history-
support floor. This is descriptive off-support translation-grammar
compatibility, not an admitted donor, grammar, owner, phase, or coverage
handle. Later waves, training, and architecture promotion do not run.

The completed [Native Sibling-Row Branch Value and Commit Crossover
results](experiments/2026-07-17-native-sibling-row-branch-value-and-commit-crossover/results.md)
replace forced coordinates with complete rows naturally emitted from exact
shared parent prefixes. No released state produces a positive,
safety-preserving Greedy Branch-Value Gap at Horizon Four. One row-zero
bowl-carrot state has a strict reciprocal immediate-successor effect across all
231 exact-row pairs and reproduces under full-model 32-bit floating point. The
two owners occupy nearly identical spatial support, two other states reject
reciprocity, and one dense-person state is refused. The bounded conclusion is
executable native row-conditioned successor state, not a general
physical-object covered set.

The subsequent exact-prefix transition sequence is also complete. The
completed [Next-Row Likelihood Change and Causal Source Trace](experiments/2026-07-17-next-row-probability-transition-and-causal-source-trace/results.md)
finds strong own-owner suppression and selective later-owner increases,
concentrated at `x1`, in a same-category native image-`2299` chain, but stops
at its natural-sibling admission gate (153 of 160 samples select one
successor), leaving physical commit versus geometry-sorted traversal
unresolved. The earlier [random-versus-geometry-sorted common-prompt
draft](experiments/2026-07-17-random-versus-geometry-sorted-common-prompt-prefix-comparison/unit.md)
was superseded before execution by that unit; its earlier-row order-swap
panel therefore remains unexecuted. The completed [Image 2299 Near-Complete
Human Relabel Successor Transition](experiments/2026-07-18-image2299-near-complete-human-relabel-successor-transition/results.md)
supports an immediate physical-owner-sensitive transition: the just-emitted
person is excluded from all 96 paired successors, redistribution is
owner-specific, and 53 of 93 matched successors move backward in geometry
order, while multi-row covered-set state versus last-row spatial routing
stays unresolved. The completed [Historical Random versus Geometry-Sorted
Image 2299 Screen](experiments/2026-07-18-historical-random-versus-geometry-sorted-image2299-screen/results.md)
shows the two historical checkpoint-3668 adapters implement qualitatively
different next-row rules under one identical forced state, and the completed
[Person 25 Dominant-Owner Commit and Persistence Closeout](experiments/2026-07-18-person25-dominant-owner-commit-and-persistence-closeout/results.md)
closes the missing dominant-owner treatment on the random-order adapter:
strong latest-row redirection without a reliable covered-object ledger, with
a fixed-prefix `y2` state where a large valid-boundary probability cluster
loses greedy argmax to one isolated extreme coordinate token. The completed
[Human-Resolved Dense Branch Value and Calibration Screen](experiments/2026-07-18-human-resolved-dense-branch-value-and-calibration-screen/results.md)
finds no positive safety-preserving sampled-branch value gap over four rows,
closing sampled-row novelty as a direct preference-training target. The
completed [Fixed-Prompt Clean-versus-Degraded Coordinate Branch
Replication](experiments/2026-07-18-fixed-prompt-clean-versus-degraded-coordinate-branch-replication/results.md)
replicates distributed raw physical-boundary mass in all three same-object
focus cases but fails its predeclared policy-survival gate in two, separating
independent boundary error, autoregressive geometry drift, and tail sampling
while rejecting `x1` as a universal owner lock. Finally, the completed
[Matched-Objective Coordinate-Branch Signature Comparison](experiments/2026-07-18-matched-objective-coordinate-branch-signature-comparison/results.md)
shows that the scoring checkpoint, not prefix source, owns the bowl,
umbrella, and bus coordinate modes and that pure cross-entropy wins the
matched validation-200 comparison; pure cross-entropy is therefore the next
mechanism baseline and the proposed coordinate-consistency training screen is
not promoted. The later [Matched Random-Order-Trained versus Geometry-Sorted-
Trained Prefix-Order Screen](experiments/2026-07-20-matched-random-sorted-prefix-order-screen/results.md)
compares the step-`4,887` pure-cross-entropy checkpoints under identical
literal prompts and frozen prefixes. Random complete-row training changes the
transition policy but does not create useful covered-set invariance: on image
`2299` every order and leave-one-out arm routes to one habitual person, and no
tuple passes symmetric coverage. Geometry-sorted training remains path-
sensitive but preserves more route diversity and supplies the cleanest local
category-plus-`x1` intervention. This supports designing a small geometry-
sorted-base local transition-calibration screen, with execution separately
authorized and random ordering retained as an ablation. The [research
compass](compass.md) owns the resulting discriminator boundary.

## Historical Completed Evidence Units

This table remains the detailed unit inventory. Unit records own their original
protocols, populations, metrics, and stop boundaries; the current compass and
the intake rows above own cross-direction routing.

| Unit | Status | Purpose |
|---|---|---|
| [Masked Spatial Policy and Accepted-Row Prefix Policy Disentanglement](experiments/2026-07-13-spatial-scope-history-disentanglement/unit.md) | complete; dual-root evidence verified; [results](experiments/2026-07-13-spatial-scope-history-disentanglement/results.md); architecture not promoted | Localized a reproducible one-opportunity input-mask effect and a harmful cumulative prompt-policy effect while rejecting safe final-policy promotion. |
| [Sampled-Rescue Object Transition Distribution and Causal Replay](experiments/2026-07-14-sampled-rescue-object-transition-causal-replay/unit.md) | complete; evidence verified; [results](experiments/2026-07-14-sampled-rescue-object-transition-causal-replay/results.md); architecture and training not promoted | Supports bounded fixed-state object-mode fragmentation, trajectory-state dependence, and description-conditioned row binding; rules out strict object-specific commit-to-uncovered redistribution in the primary case. |
| [Prefix-State Phrase-Geometry Factorial](experiments/2026-07-15-prefix-state-phrase-geometry-factorial/unit.md) | complete; evidence verified; [results](experiments/2026-07-15-prefix-state-phrase-geometry-factorial/results.md); architecture and training not promoted | Supports a bounded phrase-and-geometry compatibility gate and rules out three standalone explanations; visual grounding remains unresolved. |
| [Visual-Support Counterfactual Commit Test](experiments/2026-07-15-visual-support-counterfactual-commit/unit.md) | complete; evidence verified; [results](experiments/2026-07-15-visual-support-counterfactual-commit/results.md); architecture and training not promoted | Finds no detectable dependence on selected local post-vision committed-object support under exact substitution; raises text-mediated commit while retaining globally distributed visual evidence as the strongest alternative. |
| [Pre-Vision Raw-Bounding-Box Visual-Support Counterfactual Commit Test](experiments/2026-07-15-pre-vision-raw-bounding-box-visual-support-counterfactual-commit/unit.md) | complete; evidence verified; [results](experiments/2026-07-15-pre-vision-raw-bounding-box-visual-support-counterfactual-commit/results.md); architecture and training not promoted | Strong null under full pixel-space re-encoding: clean, target, and equal-shaped control each produced the right-cup successor in all eight paired samples; closes committed-object-support probing. |
| [Native Coherent-Row Commit-to-Uncovered Redistribution Factorial](experiments/2026-07-15-native-coherent-row-commit-to-uncovered-redistribution/unit.md) | complete; evidence verified; primary execution-invariance gate failed; [results](experiments/2026-07-15-native-coherent-row-commit-to-uncovered-redistribution/results.md); architecture and training not promoted | Supports geometry-conditioned visual object repair, but same-class coordinate selection changes with physical batch execution; stable native commit remains unestablished. |
| [Mixed-Length, Homogeneous, and Equal-Length Batch Coordinate-Logit Invariance Probe](experiments/2026-07-15-homogeneous-mixed-batch-coordinate-logit-invariance/unit.md) | complete; evidence verified; [results](experiments/2026-07-15-homogeneous-mixed-batch-coordinate-logit-invariance/results.md); architecture and training not promoted | Isolates a broad `bfloat16` batch-shape-dependent coordinate-distribution shift for one exact recipient; semantic neighbors and position are inactive, and full-model `float32` restores practical invariance. |
| [Single-Target Visual-Feature Replay into Homogeneous Batch Four](experiments/2026-07-15-single-target-visual-feature-replay-into-homogeneous-batch-four/unit.md) | complete; evidence verified; [results](experiments/2026-07-15-single-target-visual-feature-replay-into-homogeneous-batch-four/results.md); architecture and training not promoted | Finds exact primary and DeepStack equality across batch shapes and zero recovery from replaying batch-one visual features into downstream batch four, localizing the split after `get_image_features`. |
| [Selected-Transition Batch-Precision Prevalence Screen](experiments/2026-07-15-selected-transition-batch-precision-prevalence-screen/unit.md) | complete; evidence verified; [results](experiments/2026-07-15-selected-transition-batch-precision-prevalence-screen/results.md); architecture and training not promoted | Finds recurrent exact low-precision trajectory sensitivity but no material first-action object, stop, description, or coordinate-basin switch in six selected cases; routes one final full-logit repeat panel. |
| [Repeated First-Differing-Slot Full-Coordinate-Logit Panel](experiments/2026-07-15-repeated-first-differing-slot-full-coordinate-logit-panel/unit.md) | complete; evidence verified; [results](experiments/2026-07-15-repeated-first-differing-slot-full-coordinate-logit-panel/results.md); architecture and training not promoted | Confirms deterministic low-amplitude batch-shape perturbations around neighboring coordinate ties and closes a language-layer sweep for the three selected micro-splits. |
| [Fixed-Encoding Object-Centered Spatial-Eligibility Crossover](experiments/2026-07-15-fixed-encoding-object-centered-spatial-eligibility-crossover/unit.md) | complete; evidence verified; [results](experiments/2026-07-15-fixed-encoding-object-centered-spatial-eligibility-crossover/results.md); architecture and training not promoted | Establishes a causally potent regional post-vision routing seam in selected cases, but only one complete owner switch and no recurrent general competition release. |
| [Fixed-Encoding Soft Spatial-Key Bias Dose Response](experiments/2026-07-15-fixed-encoding-soft-spatial-key-bias-dose-response/unit.md) | complete; evidence verified; [results](experiments/2026-07-15-fixed-encoding-soft-spatial-key-bias-dose-response/results.md); uniform operator closed; architecture and training not promoted | Establishes graded regional actuation below owner-basin crossover; no finite dose produces a complete-row or geometry owner reversal, while hard eligibility relies mainly on nonlinear non-owner suppression. |
| [Fixed-Encoding Row-Scoring-Query-Only Spatial-Key Eligibility Crossover](experiments/2026-07-15-fixed-encoding-row-scoring-query-only-spatial-key-eligibility-crossover/unit.md) | complete; evidence verified; [results](experiments/2026-07-15-fixed-encoding-row-scoring-query-only-spatial-key-eligibility-crossover/results.md); architecture and training not promoted | Direct row reads switch geometry and same-category spatial ownership in selected cases but do not recover image `139`'s cross-category semantic-plus-geometry parent phenotype; earlier-query dependence is supported but not localized. |
| [Fixed-Encoding Earlier-Query-Only Spatial-Key Eligibility Factorial](experiments/2026-07-15-fixed-encoding-earlier-query-only-spatial-key-eligibility-factorial/unit.md) | complete; evidence verified; [results](experiments/2026-07-15-fixed-encoding-earlier-query-only-spatial-key-eligibility-factorial/results.md); architecture and training not promoted | Rules out earlier-only region-specific identity compilation on image `139`; direct row reads dominate geometry, while matched earlier-state and row-read restriction produces cross-category semantic exclusion mainly through non-owner suppression. |
| [Fixed-Encoding Cross-Region Earlier-Query and Row-Scoring-Query Spatial-Key Eligibility Hybrid](experiments/2026-07-15-fixed-encoding-cross-region-earlier-and-row-query-spatial-key-eligibility-hybrid/unit.md) | complete; evidence verified; [results](experiments/2026-07-15-fixed-encoding-cross-region-earlier-and-row-query-spatial-key-eligibility-hybrid/results.md); architecture and training not promoted | Establishes asymmetric crossed phase interaction: both query intervals affect the first semantic token, row-region sign controls aggregate geometry, only Clock-Earlier and Vase-Row is a pairwise phrase-geometry chimera, and hard-switch artifact remains unresolved. |
| [Fixed-Encoding Count-Balanced Soft Cross-Region Earlier-Query and Row-Scoring-Query Spatial Bias](experiments/2026-07-15-fixed-encoding-count-balanced-soft-cross-region-earlier-and-row-query-spatial-bias/unit.md) | complete; evidence verified; [results](experiments/2026-07-15-fixed-encoding-count-balanced-soft-cross-region-earlier-and-row-query-spatial-bias/results.md); operator closed; architecture and training not promoted | The matched vase arm selects `clock` and clock geometry, while the matched clock arm destructively suppresses clock geometry; matched adjudication fails, so crossed-arm interpretation is prohibited. |
| [Fixed-Encoding Conditional Downstream Layer-Output Residual-State Portability Gate](experiments/2026-07-15-fixed-encoding-downstream-residual-state-portability-gate/unit.md) | complete; evidence verified; [results](experiments/2026-07-15-fixed-encoding-downstream-residual-state-portability-gate/results.md); architecture and training not promoted | Supports bounded layer-`23` clock-description confidence portability; image `632` yields strong coordinate actuation but no eligible donor-owned box, so geometry and combined phase-specific portability are not established. |
| [Fixed-Encoding Persistent Hard-Routing Geometry-Donor Eligibility Screen](experiments/2026-07-15-fixed-encoding-persistent-hard-routing-geometry-donor-eligibility-screen/unit.md) | complete; evidence verified; [results](experiments/2026-07-15-fixed-encoding-persistent-hard-routing-geometry-donor-eligibility-screen/results.md); architecture and training not promoted | Finds two eligible same-description tight geometry paths on each of three clean resolution-qualified images and opens only the frozen image-`7818` portability test. |
| [Fixed-Encoding Persistent Hard-Routing Geometry-State Portability on Image 7818](experiments/2026-07-16-fixed-encoding-persistent-hard-routing-geometry-state-portability-image7818/unit.md) | complete; evidence verified; [results](experiments/2026-07-16-fixed-encoding-persistent-hard-routing-geometry-state-portability-image7818/results.md); architecture and training not promoted | Establishes a bounded one-way block-`23` pre-`x1` geometry-owner basin switch with a valid block-`13` negative control; the effect is concentrated at the first coordinate and does not establish a complete object state. |
| [Human-Audited Rare-Object Trajectory Genealogy and Causal Branch Replay](experiments/2026-07-16-human-audited-rare-object-trajectory-genealogy/unit.md) | redirected after partial verified evidence; manual review complete; [results](experiments/2026-07-16-human-audited-rare-object-trajectory-genealogy/results.md) | Finds real Common Objects in Context 80-category support for all 119 purposive unmatched candidates, but no frozen unique-entity ledger; semantic descriptions are usually sound while geometry frequently covers parts, multiple instances, or axis-wise mixtures. |
| [Fixed-Prefix Complete-Box Coherence and Progressive Coordinate-Release Factorial](experiments/2026-07-16-fixed-prefix-complete-box-coherence-and-coordinate-release-factorial/unit.md) | complete; evidence verified; [results](experiments/2026-07-16-fixed-prefix-complete-box-coherence-and-coordinate-release-factorial/results.md); architecture and training not promoted | Establishes strong `x1`-to-`x2` geometry transport in one dense-chair state, a strong case-level fork part-sized or late-extent basin without isolating its cause, and cross-row transition sensitivity; physical owner state and correct uncovered redistribution remain unproved. |
| [Object-Specific Geometry Transport, Decision Phase, and Cross-Row Influence Horizon](experiments/2026-07-17-object-specific-geometry-transport-and-cross-row-influence-horizon/unit.md) | complete; evidence verified; stopped after Wave One common-support failure; [results](experiments/2026-07-17-object-specific-geometry-transport-and-cross-row-influence-horizon/results.md); architecture and training not promoted | Finds sharp donor-specific early-coordinate basins and coherent off-support coordinate-translation behavior, but no admitted common-support contrast; later phase and cross-row waves do not run. |
| [Native Sibling-Row Branch Value and Commit Crossover](experiments/2026-07-17-native-sibling-row-branch-value-and-commit-crossover/unit.md) | complete; evidence verified; [results](experiments/2026-07-17-native-sibling-row-branch-value-and-commit-crossover/results.md); architecture and training not promoted | Finds no safe greedy branch-value handle and one numerically robust reciprocal successor switch between spatially nested bowl and carrot owners; establishes native cross-row state but not a general physical-instance ledger. |
| [Next-Row Likelihood Change and Causal Source Trace](experiments/2026-07-17-next-row-probability-transition-and-causal-source-trace/unit.md) | complete; evidence verified; stopped at the natural-sibling admission gate; [results](experiments/2026-07-17-next-row-probability-transition-and-causal-source-trace/results.md); architecture and training not promoted | Establishes strong exact self-owner suppression with selective later-owner increases concentrated at `x1` in a same-category chain; the equal-depth reciprocal discriminator is unidentified because no alternative owner passed admission, so physical commit versus traversal stays unresolved. |
| [Image 2299 Near-Complete Human Relabel Successor Transition](experiments/2026-07-18-image2299-near-complete-human-relabel-successor-transition/unit.md) | complete; evidence verified; [results](experiments/2026-07-18-image2299-near-complete-human-relabel-successor-transition/results.md); architecture and training not promoted | Finds zero immediate self-owner repeats across 96 paired successors, owner-specific redistribution, and frequent backward geometry-order transitions; supports an immediate physical-owner-sensitive transition while leaving multi-row covered-set state versus last-row spatial routing unresolved. |
| [Historical Random versus Geometry-Sorted Image 2299 Next-Row Screen](experiments/2026-07-18-historical-random-versus-geometry-sorted-image2299-screen/unit.md) | complete; evidence verified; [results](experiments/2026-07-18-historical-random-versus-geometry-sorted-image2299-screen/results.md); architecture and training not promoted | The random-order adapter sends all 96 samples to one prefix-insensitive person while the geometry-sorted adapter moves generated `x1` and physical owner with the latest row; supports a learned spatial transition habit, not a coverage ledger, from one historical seed pair. |
| [Human-Resolved Dense Branch Value and Calibration Screen](experiments/2026-07-18-human-resolved-dense-branch-value-and-calibration-screen/unit.md) | complete; evidence verified; [results](experiments/2026-07-18-human-resolved-dense-branch-value-and-calibration-screen/results.md); sampled-branch preference training not promoted | No sampled-only third-row branch beats the greedy branch over four rows; alternative routes more often enter weakly localized `tie` output, so sampled novelty alone is not a safe preference-training target. |
| [Person 25 Dominant-Owner Commit and Persistence Closeout](experiments/2026-07-18-person25-dominant-owner-commit-and-persistence-closeout/unit.md) | complete; evidence verified; [results](experiments/2026-07-18-person25-dominant-owner-commit-and-persistence-closeout/results.md); architecture and training not promoted | A person-25 row strongly redirects sampling toward overlapping person 18 on the random-order adapter, but earlier history gives only partial exclusion (`55/96` recurrence) and greedy picks a merged extent because one isolated `y2=999` token beats a much larger valid-boundary cluster. |
| [Fixed-Prompt Clean-versus-Degraded Coordinate Branch Replication](experiments/2026-07-18-fixed-prompt-clean-versus-degraded-coordinate-branch-replication/unit.md) | complete; evidence verified; [results](experiments/2026-07-18-fixed-prompt-clean-versus-degraded-coordinate-branch-replication/results.md); architecture and training not promoted | Replicates distributed raw physical-boundary mass in all three same-object cases, but the predeclared combined gate fails because top-p removes the tight cluster in two; separates independent boundary error, branch amplification, and tail sampling and rejects `x1` as a universal owner lock. |
| [Matched-Objective Coordinate-Branch Signature Comparison](experiments/2026-07-18-matched-objective-coordinate-branch-signature-comparison/unit.md) | complete; bounded case-level evidence verified; [results](experiments/2026-07-18-matched-objective-coordinate-branch-signature-comparison/results.md); proposed training screen not promoted | Exact current-sampler replication shows the scoring checkpoint, not prefix source, owns the bowl, umbrella, and bus coordinate modes; pure cross-entropy wins the matched validation-200 comparison and becomes the next mechanism baseline. |
| [Same Covered Physical-Object Set under Different Earlier Prefix Orders](experiments/2026-07-19-same-covered-set-prefix-order-equivalence/unit.md) | complete; bounded evidence verified; [results](experiments/2026-07-19-same-covered-set-prefix-order-equivalence/results.md); training and architecture not promoted | Establishes both a local order-robust emitted-owner suppression case and physical next-owner switches under the same covered set and final row, motivating a short future-horizon value test. |
| [Common Physical Objects under Different Prefix Permutations and a Short Future Horizon](experiments/2026-07-19-common-object-prefix-permutation-short-horizon/unit.md) | complete for the geometry-sorted pure-cross-entropy checkpoint; bounded evidence verified; [results](experiments/2026-07-19-common-object-prefix-permutation-short-horizon/results.md); matched random-order replication pending; training and architecture not promoted | Finds that same-set prefix order can alter later strict owner sets without generic decoding collapse. Some routes reconverge, while dense image 19109 exposes order-sensitive recurrence of already covered motorcycles; routes next to complete-row covered-versus-uncovered scoring. |
| [Matched Random-Order-Trained versus Geometry-Sorted-Trained Prefix-Order Screen](experiments/2026-07-20-matched-random-sorted-prefix-order-screen/unit.md) | complete; bounded checkpoint-conditional evidence verified; [results](experiments/2026-07-20-matched-random-sorted-prefix-order-screen/results.md); architecture not promoted; small training-screen design supported | Random complete-row training changes path sensitivity but does not create symmetric covered-set behavior; its image-2299 invariance is one habitual-successor collapse. Geometry-sorted pure cross-entropy remains the pragmatic base for a local transition-calibration screen. |
