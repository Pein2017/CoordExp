---
title: Qwen3 Vision-Language Autoregressive Detection Research Compass
description: Program-level north star, belief register, discriminator queue, and architecture-promotion boundary for dense autoregressive detection research.
type: investigation
role: research-compass
authority: non_normative_research
architecture_promotion_status: not_promoted
topic: qwen3-vl-dense-enumeration
status: active
updated: 2026-07-17
---

# Qwen3 Vision-Language Autoregressive Detection Research Compass

This page is the mutable program-level compass for the Qwen3 Vision-Language
(`Qwen3-VL`) dense-enumeration investigation. It records the north star,
current beliefs, strongest alternatives, next discriminators, and candidate
paper thesis. It does not own executed facts, runtime behavior, a final
architecture, or an implementation contract.

The [weekly integrated research
report](2026-07-13-to-2026-07-16-weekly-research-report.md) is the compact
supervisor snapshot of the executed sequence summarized by this compass.

Executed facts remain in experiment `unit.md` and `results.md` records. The
[investigation overview](overview.md) owns the detailed evidence atlas and
hypothesis decomposition. [Research decisions](../../decisions/) own current
route choices. A mechanism is not promoted to [Mechanisms](../../mechanisms/)
until independent units support it and a novel prediction succeeds.

## North-Star Objective

Determine whether the pretrained Qwen3-VL model can be trained to enumerate
the complete supported Common Objects in Context 80-category (`COCO-80`) object
set through deterministic greedy autoregressive generation, while preserving:

- object phrase and geometry binding;
- precision and low unsupported-hallucination behavior;
- unique-object coverage rather than repeated hits;
- valid structured rows and natural termination;
- the model's pretrained visual, language, and one-shot detection capability.

The immediate research target is to convert object support that is currently
dispersed across stochastic rollouts and prefix-conditioned trajectories into
a safe greedy traversal. This is an objective, not an established capability.

## Working First-Principles Model

Ideal enumeration resembles weighted selection without replacement. If
`U_t` is the set of reportable objects not yet committed at object step `t`, a
successful transition must:

```text
choose one object from U_t
  -> bind one phrase and one geometry to that object
  -> commit the emitted object
  -> remove or suppress that object
  -> redistribute probability toward U_(t+1)
  -> stop only when no supported object remains
```

The current model has no proven explicit `U_t`. Its text prefix and cached
language-model states must jointly approximate traversal, commitment, and
remaining-object state. Repeated full-image sampling may therefore expose
object support that exists in the model distribution without proving that one
greedy trajectory can cover it safely.

## Research Graph

```mermaid
flowchart LR
    A["North-star objective"] --> B["Competing mechanism hypotheses"]
    B --> C["Small discriminating research unit"]
    C --> D["Evidence and bounded verdict"]
    D --> E["Belief and decision update"]
    E --> F["Bounded learnability screen"]
    F --> G["Candidate paper thesis update"]
    D --> H["Demoted or rejected claims"]
    E --> B
```

Markdown links, rather than this diagram, are the durable graph edges.

## Non-Negotiable Research Commitments

1. Measure selection, row transcription, commit or coverage, and termination
   separately before combining them into one explanation.
2. Require correct-object interventions to outperform wrong-object,
   source-swapped, token-permuted, position-only, and norm-matched controls.
3. Use same-prefix controls before attributing a result to next-object
   competition, and same-feature controls before attributing a result to the
   language-model side of the visual-language boundary.
4. Treat unmatched predictions under incomplete annotations as unresolved
   until classified as an unlabeled real object, duplicate, localization or
   category mismatch, unsupported hallucination, or uncertain.
5. A training improvement must preserve native one-shot capability and cannot
   be promoted solely because it produces more rows or slightly more recall.
6. Architecture follows passed causal gates. Slots, a persistent ledger, a
   cursor renderer, a fixed bridge layer, and a final forward pass remain
   unselected.
7. A 256-image training screen is evidence about learnability, not a final
   quality estimate or scale claim.
8. A mechanism claim must survive semantically equivalent physical batch
   execution before it is treated as an intrinsic model capability.

## Current Evidence Boundary

### Verified or bounded-supported

- Pixel-level full-canvas spatial restriction changes which object is rescued
  in one seed-matched opportunity, but the final masked policy does not safely
  outperform equal-call full-image bagging.
- Full-image bagging exposes additional unique object support, but also produces
  highly correlated repetition, fragmentation, category disagreement, and
  some true hallucination. It is a diagnostic and candidate source, not an
  accepted final policy.
- The complete cumulative accepted-row prefix policy is harmful relative to
  reset. That intervention bundles length, content, errors, order, and visual
  consistency, so a token-length-only mechanism is not established.
- Painted and post-scatter visual designation can causally control a selected
  object row in bounded probes.
- The tested split-layer proposal bridge behaves mainly as a non-specific
  continuation pulse rather than source-specific object control.
- Native prefix state produces a content-sensitive short-term transition, but
  no stable order-free object ledger has been demonstrated.
- At exact prefix state 56 on image `12576`, repeated one-row sampling exposes
  two valid object modes: the target pizza and a competing left cup. The same
  target disappears at the greedy-terminal state; an independent chair case
  reproduces the rescue-entry-versus-terminal state contrast.
- One source-specific first description token can carry a matching phrase and
  box through a complete row. This establishes description-conditioned
  within-row binding, not a visual instance pointer.
- The exact phrase-geometry factorial rejects phrase-only, geometry-only, and
  uniform complete-row advancement explanations at Prefix State 56 (`P56`).
  Only a coherent `cup` plus left-cup geometry row advances to the right cup.
  This supports a bounded phrase-and-geometry compatibility gate, not yet a
  visually grounded commit or order-free ledger.
- The right-cup successor at that exact state survives both selected local
  post-vision left-cup support replacement and complete pre-vision raw
  left-cup bounding-box donor replacement in greedy decoding and all eight
  paired samples. This disfavors a behaviorally necessary local
  committed-object visual revalidation gate without proving visual
  independence.
- On image `7574`, crossed phrase-geometry rows cause the decoder to
  re-identify the physical object at the supplied geometry. However, the exact
  coherent white-bowl prompt repeats the white bowl at physical batch size one
  and advances to an orange bowl at physical batch size four. The result is
  independent of a `16`- versus `512`-token maximum generation horizon and
  localizes the instability to same-class first-coordinate selection.
- Across six selected complete-row transitions, physical batch shape changes
  exact `bfloat16` continuation tokens in every case and the first action in
  three cases. The three first-action changes remain on the same object and
  differ by only one or two source-image pixels; full-model `float32` removes
  every promoted difference. Recurrent numeric sensitivity is therefore
  supported, while a recurrent material object- or coordinate-basin switch is
  not.
- At the exact first differing slots of those three cases, both repeats
  reproduce the `bfloat16` split with zero same-layout noise, but complete-
  vector shifts remain only `4.17%` to `6.22%` of the earlier broad anchor and
  every selected bin stays inside one object neighborhood. Exact-recipient
  `float32` attenuates the batch-layout shift by `5,342` to `11,278` times.
  Layer escalation is closed for these micro-splits without claiming global
  numerical invariance.
- With one exact full-image visual encoding and fixed base prefix, hard
  post-vision image-token key eligibility causally changes later description
  and coordinate likelihoods. One of five valid cases completes a regional
  semantic-plus-geometry owner switch; one switches geometry without semantic
  ownership; three remain one-sided or destructive. The seam is real, but a
  recurrent whole-row instance pointer and general competition release are not
  established.
- Finite decoder-wide positive spatial-key bias at `0.5`, `1.0`, and `2.0`
  strongly and dose-dependently changes later row likelihoods on four retained
  anchors, but yields zero complete-row or geometry owner reversals. Hard
  eligibility's positive cases arise mainly from nonlinear non-owner collapse,
  so mild context-preserving attention preference is not a sufficient owner-
  binding operator on this panel.
- Row-scoring-query-only hard eligibility is sufficient for geometry and
  same-category spatial-owner switching in selected cases, but it does not
  reproduce image `139`'s cross-category semantic-plus-geometry all-query
  phenotype. Some earlier-query computation is required for that phenotype;
  its locus and interaction with the direct row read remain unresolved.
- In the image-`139` earlier-query-by-row-query factorial, earlier-only
  restriction has almost no regional crossover, while row-only restriction
  carries geometry routing. The large all-query cross-category description
  crossover appears only for matched earlier and row regions and is driven
  mainly by collapse of the competing row. This supports phase-separated
  compatibility or competitive exclusion, not an earlier-only object compiler
  or a localized causal mediator.
- In the image-`139` crossed-region hybrid, both earlier and row-scoring regions
  affect the first semantic token, while aggregate geometry-margin sign follows
  the row-scoring region. Clock-Earlier and Vase-Row forms a pairwise clock-
  phrase and vase-geometry chimera; the reverse hybrid's top first description
  token is `person`, so symmetric phrase-geometry handoff is falsified. Both
  complete-row margins become nearly neutral, and constructive geometry
  activation occurs only in the vase-row direction.
- At the cardinality-derived finite soft bias `3.9060049`, the matched vase arm
  predicts `clock` and favors clock geometry, while the matched clock arm
  materially suppresses its preferred clock geometry owner. The execution is
  valid, but the matched-control gate fails; the crossed arms are therefore not
  interpreted, and this operator is closed as an adjudicator of the hard
  crossed-region result.
- One eligible image-`139` clock-description path retains approximately `98.4%`
  of its persistent hard-routing release after one layer-`23` returned-state
  replacement. The image-`632` geometry anchor is control-only: hard routing
  adds `+2.329` to `+2.544` natural-log units per coordinate, but donor
  Intersection over Union (`IoU`) remains `0.000` to `0.018`. This supports
  bounded semantic confidence portability and coarse coordinate actuation,
  not donor-owned geometry or a combined phase-specific controller.
- A follow-up eligibility screen finds two tight, instance-owned persistent-
  hard geometry paths on each of three clean resolution-qualified images. On
  the frozen image-`7818` successor, the paired `wine glass` donor state after
  decoder block `23` switches the unrestricted owner from annotation `664730`
  to `661523` with donor Intersection over Union (`IoU`) `0.800259`; the trusted
  block-`13` control does not switch. The teacher-forced effect is almost
  entirely at `x1`, supporting bounded one-way first-coordinate geometry-basin
  portability rather than a portable full-box object representation.

### Still speculative or unresolved

- How frequently the same exact prefix contains multiple stable valid
  next-object modes across the validation population.
- Whether the phrase-and-geometry-compatible row is interpreted as a true
  object commit or as a canonical geometry-sorted serialization event; the
  executed local post-vision and raw-bounding-box counterfactuals do not
  distinguish those remaining explanations.
- Object support revealed by bagging can be distilled into safe greedy
  enumeration.
- Pixel-mask rescue being primarily caused by post-vision candidate
  competition; the hard-key panel provides one complete case and the finite-
  bias panel produces graded but sub-threshold routing, not recurrent support.
- Whether the asymmetric hard crossed-region result and matched non-owner
  suppression reflect useful compatibility-sensitive binding or an artifact
  of hard key exclusion plus an abrupt phase switch. The count-balanced finite
  soft cross operator does not resolve this because it fails its matched-owner
  controls before crossed-arm adjudication.
- Whether the image-`7818` one-way `x1` basin switch generalizes across images,
  prefixes, and both donor directions, and whether any causal information
  persists beyond the first coordinate rather than acting only through the
  emitted `x1` token.
- Long prefix length alone causes the cumulative-policy failure.
- Incomplete dense labels are the dominant cause of conservative termination.
- Whether physical batch sensitivity comes from low-margin Brain Floating
  Point 16-bit (`bfloat16`) kernel
  numerics, padding or attention-mask semantics, multimodal position handling,
  request collation, or unintended cross-request interaction.
- An explicit slot, ledger, cursor, or new forward architecture is necessary.

## Historical Late-Middle Language-Layer Evidence

The layer evidence is phase-specific and must not be reduced to a claim that
"layer 17 or layer 21 controls detection."

- A divergence-conditioned logit-lens study found that the painted-to-clean
  stop-versus-continue margin became strongly readable mainly at language-model
  layers 17 through 21, while first-description-token identity became strongly
  readable mainly at layers 20 through 23. This was linear-readability evidence
  on 20 selected validation events, not causal full-row evidence. See
  [Post-Scatter Layer Onset](../../ideas/qwen3-vl-painted-gt-transcription-probe/experiments/2026-07-09-pvci-post-scatter-layer-onset/unit.md).
- Residual replacement through the real language-model head recovered the
  immediate stop-versus-continue decision strongly from layer 17 onward and
  description identity from layer 20 onward on 18 strict replay events. This
  establishes bounded immediate-token causal sufficiency, not a universal
  layer or complete-row controller. See
  [Residual Patch Causal Sufficiency](../../ideas/qwen3-vl-painted-gt-transcription-probe/experiments/2026-07-09-pvci-residual-patch-causal-sufficiency/unit.md).
- A one-time late-middle patch recovered the immediate divergent token on all
  ten fresh replay-eligible events but recovered the exact full row and exact
  four-coordinate sequence on zero events. Continuation, identity, and geometry
  therefore require distinct timing or state persistence. See
  [Row-Trajectory Causal Persistence](../../ideas/qwen3-vl-painted-gt-transcription-probe/experiments/2026-07-10-pvci-row-trajectory-causal-persistence/unit.md).
- Historical `ms-swift` provenance also localized selected duplication and
  coordinate-basin transitions near layer 17 or 18, with later layers carrying
  or sharpening ownership. The synthesis is routed through the
  [Autoregressive Binding Template Study](../autoregressive-binding-template-study/).

Current belief: late-middle residual states are promising phase-specific
observation and intervention surfaces. They are not yet evidence for one fixed
"heart-bypass" layer, autonomous object selection, complete row binding, or
cross-row coverage.

## Current Belief Register

| Mechanism hypothesis | Current state | Strongest alternative | Next belief-changing observation |
|---|---|---|---|
| Input-level spatial restriction changes one object opportunity | Bounded support under hard eligibility; the count-balanced all-keys-readable soft cross operator is now closed because both matched-owner controls fail, so its crossed arms cannot adjudicate the hard asymmetry | Hard exclusion and its abrupt phase switch may manufacture the asymmetry; the historical uniform-dose operator and current query-phase-restricted operator cannot be pooled into one trajectory | No immediate successor is active. A historical-uniform-operator `3.9060049` endpoint would require a fresh separately authorized execution; artifact-only cross-operator pooling is rejected. |
| Safe final masked policy improves enumeration | Ruled out under the executed protocol | Equal-call full-image sampling is at least as useful after aggregation | Revisit only after a new masking mechanism avoids retention and output-expansion failures. |
| Fixed-prefix next-object probability is fragmented across several real objects | Supported at one exact state; not population-estimated | Some other bagging rescues may still arise only from earlier trajectory divergence | Replicate only when a future route decision requires prevalence, not as the immediate discriminator. |
| Prefix state is an executable but fragile traversal state | Phrase-geometry coherence controls bounded repair behavior, but the dramatic same-class successor is confounded by one broad `bfloat16` execution branch; the three recurrent micro-splits are now classified as low-amplitude same-object coordinate rank changes | The isolated predecessor remains a real broad numerical counterexample, while the underlying semantic commit question remains unresolved | No active successor. The later fixed-encoding and residual-portability chain supersedes the earlier instruction to return to same-encoding visual eligibility; if separately authorized, use the `x1` causal-mediation discriminator below. |
| Late-middle residual states implement phase-specific decisions | Bounded one-sided support: an eligible clock-description path is conditionally portable, and one same-description paired-object state after decoder block `23` switches the unrestricted geometry owner through an `x1` basin change where the trusted block-`13` control does not | The portable state may select only the first coordinate, with the emitted `x1` and native autoregressive computation recovering the rest of the box; one image and one direction do not establish a general object state | No successor is active. If separately authorized, mediate `x1` by forcing paired `x1` without replacement and forcing baseline `x1` under paired-state replacement, then compare later coordinates and final owner. |
| Early coordinate choice establishes a stable complete-object owner | Mixed and target-dependent: one dense-chair state shows `63/64` released `x2` edges follow target-versus-adjacent `x1` cues, while one visible fork remains part-like on `27/32` suffixes after part-versus-whole `x1,y1` forcing | The chair effect may be autoregressive width, box-validity, or geometry-sorted rank rather than physical owner state; the fork may reflect prefix-local extent bias rather than universal part recognition | If resumed, compare a real-object `x1` cue with a matched synthetic or object-free `x1` cue, then adjudicate the next row against a frozen unique-entity ledger. Do not increase same-arm sample count. |
| Incomplete dense annotations teach conservative omission and premature termination | Plausible | Sequence-mode concentration or weak visual evidence is the dominant cause | Compare exhaustive labels with original labels and controlled thinning on the same images. |
| Bagging support can be concentrated into deterministic greedy traversal | Speculative; a fixed-prefix signature exists but no endogenous training target is identified | Bagging may expose modes that remain trajectory-specific and cannot be safely concentrated by the current supervision | Resolve the cross-row prefix factor before proposing any 256-image training screen. |
| A persistent ledger, object slot, or external detector is necessary | Unsupported and not authorized | Native prefix state plus better data and transition training may suffice | Consider only after transition shaping fails despite reliable object support and complete labels. |

The completed [Fixed-Prefix Complete-Box Coherence and Progressive Coordinate-
Release Factorial](experiments/2026-07-16-fixed-prefix-complete-box-coherence-and-coordinate-release-factorial/results.md)
adds two durable constraints. First, coordinate generation is not literally
independent: a dense-chair `x1` cue strongly transports to released `x2`.
Second, that transport is not sufficient evidence for a complete-object file:
the visible fork remains in a part-sized extent basin even after its whole-
object left and top boundaries are forced. The emitted chair geometry also
changes the next-row category, proving cross-row state sensitivity while
failing to establish correct uncovered-object redistribution.

## Current Discriminator Queue

The completed [Mixed-Length, Homogeneous, and Equal-Length Batch Coordinate-
Logit Invariance Probe](experiments/2026-07-15-homogeneous-mixed-batch-coordinate-logit-invariance/results.md)
resolves the immediate execution surface. Four byte-identical `bfloat16`
recipients produce a broad white-to-orange coordinate-distribution shift;
equal-length semantic neighbors and target batch position add no effect. The
exact predecessor mixed-length batch adds a smaller shift. Full-model
`float32` makes all cached and direct layouts practically invariant and selects
the orange-bowl mode.

The completed [Single-Target Visual-Feature Replay into Homogeneous Batch
Four](experiments/2026-07-15-single-target-visual-feature-replay-into-homogeneous-batch-four/results.md)
causally localizes the exact split after `get_image_features`: batch-one and
batch-four primary plus DeepStack outputs are elementwise equal, and replaying
batch-one features into downstream batch four recovers zero batch-one state.

The completed [Selected-Transition Batch-Precision Prevalence
Screen](experiments/2026-07-15-selected-transition-batch-precision-prevalence-screen/results.md)
found exact `bfloat16` first-action differences in three of six selected
recipients, but every difference remained on the same object and moved a box
edge by only one or two source-image pixels. Full-model `float32` made all three
complete `64`-token continuations invariant. The literal exact-token recurrence
gate fired; the material object- or coordinate-basin recurrence gate did not.

The completed [Repeated First-Differing-Slot Full-Coordinate-Logit
Panel](experiments/2026-07-15-repeated-first-differing-slot-full-coordinate-logit-panel/results.md)
reproduced every source split with zero same-layout repeat noise, but all
complete-vector shifts were low amplitude and remained inside one physical-
object neighborhood. Matched exact-recipient `float32` suppressed the
physical-batch shift by more than three orders of magnitude. The panel closes
language-layer escalation for these micro-splits. It does not erase the
isolated image-`7574` broad-shift counterexample or establish production
numeric invariance.

The completed [Fixed-Encoding Object-Centered Spatial-Eligibility
Crossover](experiments/2026-07-15-fixed-encoding-object-centered-spatial-eligibility-crossover/results.md)
closes the active hard-key discriminator. Five cases pass no-op parity, but
only image `139` completes semantic-plus-geometry owner reversal. Image
`12120` is phase-specific and image `12639` is a strong one-sided near-rescue.
The preregistered recurrence gate does not fire, so no free-row replay,
phase-specific unit, training screen, or architecture is automatically active.

The completed [Fixed-Encoding Soft Spatial-Key Bias Dose
Response](experiments/2026-07-15-fixed-encoding-soft-spatial-key-bias-dose-response/results.md)
closes the uniform positive-bias follow-up. All four anchors actuate strongly,
but every complete-row and geometry target gamma remains negative at every
frozen dose. The hard endpoint is not a smooth continuation of the finite-
bias curve: in its positive cases, target-row likelihood changes little while
the competing row collapses. No free-row replay, phase-specific unit, training
screen, larger cohort, or additional dose is active.

The completed [Fixed-Encoding Row-Scoring-Query-Only Spatial-Key Eligibility
Crossover](experiments/2026-07-15-fixed-encoding-row-scoring-query-only-spatial-key-eligibility-crossover/results.md)
preserves unrestricted image-key access for every image-token and unscored-
prefix query while applying hard regional eligibility only where logits predict
the canonical row. It produces geometry and same-category owner switches but
loses image `139`'s cross-category semantic switch. The frozen verdict supports
dependence on earlier-query computation without separating image-token
recompilation, earlier prefix-text computation, and other all-query trajectory
effects. The next discriminator is one image-`139` earlier-query-only arm that
completes a two-by-two factorial with the existing baseline, row-only, and all-
query conditions. Balanced target-plus/competitor-minus bias remains held.

The completed [Fixed-Encoding Earlier-Query-Only Spatial-Key Eligibility
Factorial](experiments/2026-07-15-fixed-encoding-earlier-query-only-spatial-key-eligibility-factorial/results.md)
shows that earlier-only regional restriction is not a region-specific identity
compiler. Direct row reads dominate geometry routing, while matched all-query
restriction produces the selected cross-category semantic effect mainly by
competitive non-owner suppression.

The completed [Fixed-Encoding Cross-Region Earlier-Query and Row-Scoring-Query
Spatial-Key Eligibility Hybrid](experiments/2026-07-15-fixed-encoding-cross-region-earlier-and-row-query-spatial-key-eligibility-hybrid/results.md)
closes the hard crossed-region discriminator. Both intervals affect the first
semantic token, and geometry sign follows the row region, but only Clock-
Earlier and Vase-Row yields the pairwise chimera. The reverse hybrid's top
semantic token is `person`, constructive activation is asymmetric, and both
complete-row margins become nearly neutral.

The completed [Fixed-Encoding Count-Balanced Soft Cross-Region Earlier-Query
and Row-Scoring-Query Spatial Bias](experiments/2026-07-15-fixed-encoding-count-balanced-soft-cross-region-earlier-and-row-query-spatial-bias/results.md)
executes that all-keys-readable discriminator at the one cardinality-derived
dose. Its runtime and structural gates pass, but both matched owner controls
fail: the matched vase arm selects `clock` and clock geometry, and the matched
clock arm destructively suppresses clock geometry. The frozen classification
is `no_adjudication_close_count_balanced_soft_cross_operator`; crossed arms are
not interpreted.

No continuation remains inside the attention-logit bias family. The historical
bias values `0`, `0.5`, `1`, and `2` use uniform regional bias over every
causally visible query row, while the current `3.9060049` operator uses phase-
restricted earlier-query and row-scoring-query slices. They cannot be pooled
into one dose trajectory. A fresh historical-uniform-operator `3.9060049`
endpoint would be a new separately authorized execution.

The completed [Fixed-Encoding Conditional Downstream Layer-Output Residual-
State Portability Gate](experiments/2026-07-15-fixed-encoding-downstream-residual-state-portability-gate/results.md)
finds bounded one-sided layer-`23` clock-description confidence portability.
Its image-`632` geometry discriminator is execution-valid but control-only:
neither persistent hard donor owns its generated box under the absolute
`IoU >= 0.30` rule. The unit closes without a layer, dose, or transplant sweep,
and no combined phase-specific claim is allowed.

The completed [Fixed-Encoding Persistent Hard-Routing Geometry-Donor
Eligibility Screen](experiments/2026-07-15-fixed-encoding-persistent-hard-routing-geometry-donor-eligibility-screen/results.md)
finds valid tight donors in three clean resolution-qualified images. Its frozen
image-`7818` successor then finds that one paired-object state after decoder
block `23` switches the unrestricted geometry owner, while the corresponding
block-`13` state does not. The effect is almost entirely an `x1` release and is
therefore classified as one-way first-coordinate geometry-basin portability,
not a complete object-state controller.

That successor is closed. No layer sweep, additional image, training bridge,
or architecture is active. If separately authorized later, the smallest next
discriminator is `x1` causal mediation: force paired `x1` without replacement,
then force baseline `x1` under paired-state replacement, and compare the later
coordinates and final owner.

## Training-Screen Gate

A 256-image training screen may start when all of the following are true:

1. A missed object has stable, auditable support under a controlled state, or a
   specific earlier trajectory transition is shown to unlock it.
2. A correct-object intervention identifies a trainable decision stage and
   outperforms wrong-object or generic controls.
3. Dense-image labels distinguish verified positives, known covered objects,
   uncertain or incomplete annotations, and unsupported outputs well enough to
   supervise termination and coverage safely.
4. The proposed training arm predicts a mechanism-level change, such as more
   probability on uncovered objects, less probability on committed objects, or
   a smaller greedy-versus-sampled coverage gap.

The screen compares a few matched-budget arms on 256 images. It is a stop-or-
promote test for learnability, not a final model, broad hyperparameter search,
or paper result.

Current gate decision: **not authorized**. Controlled object support and tight
persistent-hard geometry donors pass the first gate, and one late residual
state causally switches a same-description spatial owner. The positive state is
still privileged, one-way, and almost entirely mediated at `x1`; no clean-
feature synthesis target or preservation-safe training arm is frozen. A safe
label cohort also remains absent.

## Demoted or Rejected Claims

- A local masked-input rescue is not evidence for a safe masked inference
  policy or a language-only root cause.
- A decodable representation is not evidence that the decoder causally uses
  its object identity.
- More continuation or lower termination probability is not improved
  enumeration when duplicates, invalid rows, and unsupported output expand.
- Attention magnitude is not causal object binding.
- One late-middle residual patch is not a complete object-row controller.
- One owner-specific `x1` basin switch is not a portable four-coordinate object
  representation or a general instance-binding mechanism.
- A stable native order-free coverage ledger has not been established.
- A null under raw committed-object pixel erasure is not proof of visual
  independence, a purely textual transducer, or a native commit ledger.
- A physical batch-four successor is not an intrinsic native commit capability
  when the exact same prompt repeats the source object at batch size one.
- A `bfloat16` coordinate argmax is not a stable object-transition label when
  full-model `float32` removes the batch dependence and tied batch-four logits
  choose different bins inside one object mode.
- One-to-two-pixel coordinate jitter is not evidence for a distinct object or
  coordinate basin merely because exact token identifiers differ.
- One positive fixed-encoding hard-key crossover is not a recurrent instance
  pointer or proof that unrestricted visual competition dominates low recall.
- Strong dose-dependent regional likelihood movement without owner reversal is
  not evidence that a uniform soft attention bias solves object binding.
- Slots, a persistent ledger, a cursor renderer, and a final architecture are
  not selected.

## Candidate Paper Thesis

Candidate, not current claim:

> Dense autoregressive detection is a state-transition and policy-concentration
> problem, not merely an object-recognition problem: repeated sampling can
> expose competing valid object modes at one exact state, while earlier
> trajectory and complete-row prefix transitions determine whether that support
> remains available to a greedy trajectory.

The stronger claim that sampled support can be distilled into greedy coverage
remains conditional on identifying an endogenous cross-row training target and
passing a later training screen.

## Update Rule

After each closed research unit:

1. update only belief rows changed by the evidence;
2. link the result instead of copying its full metric table;
3. record the strongest remaining alternative and next discriminator;
4. move disproven claims to the demoted section rather than deleting them;
5. strengthen the paper thesis only after its causal and preservation gates
   pass;
6. keep implementation and architecture authorization separate from the belief
   update.
