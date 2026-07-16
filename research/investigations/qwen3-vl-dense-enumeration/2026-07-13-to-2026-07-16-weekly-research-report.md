---
title: Qwen3 Vision-Language Dense Enumeration Weekly Research Report
description: Integrated evidence, mechanism update, implementation retrospective, and external research-principal handoff for work executed from 2026-07-13 through 2026-07-16.
type: investigation
role: supervisor-report
authority: non_normative_research
architecture_promotion_status: not_promoted
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified_at_linked_units
updated: 2026-07-16
---

# Qwen3 Vision-Language Dense Enumeration Weekly Research Report

## Purpose And Scope

This report summarizes the dense-enumeration research executed between
2026-07-13 and 2026-07-16 in:

```text
/data/CoordExp/.worktrees/research-probes/
research/investigations/qwen3-vl-dense-enumeration/
```

It is written as a self-contained handoff for GPT-Pro, the external Generative
Pre-trained Transformer Pro research reviewer named by the user, or another
research principal investigator. Executed facts are owned by the linked experiment
`results.md` files and immutable runtime receipts. Cross-unit mechanism models
are explicitly labeled as synthesis or hypothesis.

This report does not promote a final architecture, training launch, stable
software contract, or production inference policy.

## Executive Verdict

The week's most important result is not a new detector architecture. It is a
substantial narrowing of the bottleneck.

The evidence no longer supports a simple explanation such as:

```text
Qwen3-VL cannot see the missing objects.
```

The stronger integrated picture is:

```text
fixed visual evidence contains multiple object/location modes
  -> the current prefix establishes a fragile executable trajectory state
  -> earlier decoder computation shapes semantic compatibility
  -> current-row visual reads strongly route geometry and spatial ownership
  -> late pre-x1 residual computation collapses toward one owner basin
  -> emitting x1 launches native autoregressive completion of the box
```

Repeated sampling proves that multiple valid next-object modes can coexist at
one exact image and prefix. Hard visual routing proves that fixed post-vision
features can causally change spatial ownership. A final one-time decoder-block-
`23` residual replacement proves that a privileged hard-routed state can switch
between two same-description object geometries after hard routing is removed.
Approximately `99.9%` of its measured positive teacher-forced effect occurs at
`x1`, the first coordinate.

The most plausible current bottleneck is therefore:

> Qwen3 Vision-Language (`Qwen3-VL`) contains usable multi-instance visual
> evidence and strong row transcription, but it does not reliably synthesize
> and update the object-specific, phase-appropriate control trajectory that
> selects one owner, commits it, and redistributes the next decision toward
> remaining objects.

This conclusion is still local. Selection, commit, coverage, stopping, and
clean-image endogenous synthesis remain unsolved.

## Original Research Question

The program asks why a geometry-sorted Qwen3-VL detector can identify and
localize many individual Common Objects in Context 80-category (`COCO-80`)
objects, while one greedy autoregressive rollout remains conservative, short,
and incomplete in dense scenes.

The desired behavior is:

```text
locate one valid object
  -> describe it
  -> emit its geometry
  -> commit it
  -> find another uncovered object
  -> stop only when no reportable object remains
```

The research posture was experiment-first. Each unit attempted to isolate one
changed factor and close it before designing an architecture.

## Phase 1: Spatial Restriction, Bagging, And Prefix Policy

### Question

Does showing the model only part of the full-resolution image improve object
coverage because the global visual or language decision contains competing
objects? Does cumulative accepted-row history help the model track coverage?

### Compared policies

- Full-Image Single Rollout: one ordinary full-image generation.
- Full-Image K-Rollout Independent Bagging: repeated full-image generations,
  aggregated only for object-support analysis.
- Full-Canvas Masked Region with Per-Region Reset: preserve canvas size and
  pixel scale while revealing one spatial region per independent call.
- Full-Canvas Masked Region with Cumulative Accepted-Row Prefix: use the same
  regional masking while feeding prior accepted rows back into later calls.
- Native-Scale Tile with Per-Tile Reset: physically tile the image without
  resize.

### Executed evidence

- Masked reset rescued more missed-object opportunities than one matched
  full-image call by approximately `+0.19` to `+0.21` across two independent
  roots.
- After aggregating the same number of calls, masked reset did not outperform
  full-image bagging and was slightly worse.
- Cumulative accepted-row prompting was approximately `0.19` to `0.20` worse
  than reset.
- Native-scale tiling was substantially worse than full-canvas masking.
- Masking produced retention loss, fragmented objects, prediction inflation,
  and low manual precision. The owning result does not isolate a
  hallucination-specific rate or prove that lost context caused every error.
- Full-image bagging produced nonzero Prediction-Set Diversity (`0.179404`), so
  it exposed different model modes rather than merely replaying one output.

### Belief update

- Pixel-level spatial restriction has a real local causal effect on which
  objects become accessible.
- It is not a safe final inference policy.
- Full-image bagging is the cleaner probe of latent object support.
- Cumulative prompting is harmful as a complete policy, but that arm confounds
  prefix length, content, errors, ordering, and visual consistency. It does not
  isolate token length alone.
- Input masking changes the vision computation and therefore cannot locate the
  competition specifically after the vision tower.

Primary result:
[Spatial Scope and History Disentanglement](experiments/2026-07-13-spatial-scope-history-disentanglement/results.md).

## Phase 2: Fixed-Prefix Object-Mode Fragmentation

### Question

Does bagging rescue objects because each rollout diverges much earlier, or can
multiple valid next objects coexist at one identical image and prefix state?

### Executed evidence

At exact Prefix State 56 on image `12576`, with image and prefix held fixed,
32 next-row samples produced:

- target pizza: `11/32`;
- competing left cup: `21/32`;
- other or invalid rows: `0/32`.

Greedy selected the left cup. The target pizza was therefore present in the
same fixed-state distribution but lost the greedy competition.

At a greedy-terminal prefix from the same image:

- `19/32` samples stopped immediately;
- `13/32` produced an already-covered knife duplicate or localization
  mismatch;
- no new object was recovered.

An independent chair case reproduced the contrast between a rescue-entry state
with remaining object support and a terminal state that always stopped.

### Within-row native capability

Forcing only one source-specific first description token was enough for the
native decoder to complete the matching phrase and geometry:

- forcing `pizza` completed the pizza row;
- forcing `cup` completed the cup row.

This is strong evidence that row transcription is not the primary limitation
once the current object mode is selected.

### Cross-row commit result

Forcing complete rows did not produce a strict order-free commit signature.
Target pizza, repeated pizza, and unsupported chair histories all mainly routed
back to the left cup. A valid left-cup row routed to the right cup, consistent
with a learned successor or geometry-sorted traversal transition.

### Belief update

- Several valid next-object modes can coexist at one exact state.
- Greedy low recall can arise from mode concentration rather than missing
  visual support.
- Earlier trajectory and prefix state determine whether support remains
  available.
- Strict object-specific commit-to-uncovered redistribution is falsified in the
  primary case.

Primary result:
[Sampled-Rescue Object Transition Distribution and Causal Replay](experiments/2026-07-14-sampled-rescue-object-transition-causal-replay/results.md).

## Phase 3: Prefix Phrase-Geometry Transaction State

### Question

What component of an emitted row changes the next-object transition: phrase,
geometry, row length, or coherent phrase-geometry binding?

### Executed evidence

An equal-length two-by-two factorial crossed:

```text
description in {pizza, cup}
geometry in {pizza box, left-cup box}
```

Only the coherent `cup` phrase plus left-cup geometry advanced from the left
cup to the right cup. The other three combinations returned to the left cup.
Covered-pizza and no-row controls were nearly identical. An unsupported-chair
row concentrated probability on the left cup but did not advance to the right
cup.

### Belief update

- Phrase alone is insufficient.
- Geometry alone is insufficient.
- Any legal equal-length row does not uniformly advance the traversal.
- The prefix behaves as content-sensitive executable state.
- A coherent phrase-geometry row acts like a transaction-consistency gate.

The remaining ambiguity was whether this gate represents true visual-object
commit or only validation of a geometry-sorted serialization transition.

Primary result:
[Prefix-State Phrase-Geometry Factorial](experiments/2026-07-15-prefix-state-phrase-geometry-factorial/results.md).

## Phase 3A: Does Commit Require The Emitted Object's Local Visual Support?

Two counterfactual experiments replaced the selected committed object's local
support while keeping the successor decision fixed:

1. selected post-vision left-cup visual support replacement;
2. complete pre-vision raw left-cup bounding-box pixel replacement followed by
   full re-encoding.

For clean, target replacement, and unrelated control conditions, greedy and all
eight paired samples still selected the right cup.

### Belief update

- The tested successor transition does not require local visual revalidation of
  the previously committed object.
- A coherent textual row can carry enough transaction state for the decoder to
  select another still-visible object.
- The result does not prove visual independence because the successor and
  global contextual evidence remain in the image.

Primary results:

- [Post-Vision Committed-Support Counterfactual](experiments/2026-07-15-visual-support-counterfactual-commit/results.md)
- [Pre-Vision Raw-Bounding-Box Counterfactual](experiments/2026-07-15-pre-vision-raw-bounding-box-visual-support-counterfactual-commit/results.md)

## Phase 4: Runtime Numerical Confound And Float32 Control

### Initial apparent mechanism

An image-`7574` coherent-row factorial initially appeared to show native
same-category commit and repair. Crossed phrase-geometry rows caused the model
to re-identify the object at the supplied geometry.

However, the identical coherent white-bowl prompt produced:

- physical batch size one: repeated white bowl;
- physical batch size four: a real but COCO-unlabeled orange bowl.

The difference began at the first coordinate after an identical phrase.

### Executed diagnosis

- Brain Floating Point 16-bit (`bfloat16`) physical batch shape shifted the
  complete coordinate distribution from the white-bowl basin toward the orange
  bowl.
- Semantic neighbors and request position were not necessary causes.
- Full-model Institute of Electrical and Electronics Engineers 754 32-bit
  floating-point (`float32`) execution made all layouts practically identical
  and selected the orange-bowl distribution.
- Primary merged visual features and all three DeepStack streams were bitwise
  equal across batch shapes.
- Replaying batch-one visual features into downstream batch four did not
  recover batch-one behavior, localizing the divergence after
  `get_image_features`.
- Across six selected transitions, `bfloat16` eventually changed coordinate
  tokens in all six, but the three first-action differences were only one- or
  two-pixel changes on the same object. Exact `float32` removed every promoted
  difference.
- Repeated full-coordinate analysis found these recurrent shifts were only
  `4.17%` to `6.22%` of the broad predecessor anchor and did not establish new
  object basins.

### Belief update

- `bfloat16` batch layout is a conclusion-critical confound for delicate
  mechanism probes.
- Exact token inequality is not automatically meaningful model behavior.
- The large bowl branch is real and deterministic but currently isolated.
- Ordinary batch-sensitive micro-shifts do not explain dense low recall or
  coverage failure.
- Later conclusion-critical probes use full-model `float32`, batch size one.

Primary results:

- [Mixed-Length, Homogeneous, and Equal-Length Batch Coordinate-Logit Invariance](experiments/2026-07-15-homogeneous-mixed-batch-coordinate-logit-invariance/results.md)
- [Single-Target Visual-Feature Replay](experiments/2026-07-15-single-target-visual-feature-replay-into-homogeneous-batch-four/results.md)
- [Selected-Transition Batch-Precision Prevalence](experiments/2026-07-15-selected-transition-batch-precision-prevalence-screen/results.md)
- [Repeated First-Differing-Slot Coordinate Panel](experiments/2026-07-15-repeated-first-differing-slot-full-coordinate-logit-panel/results.md)

## Phase 5: Fixed-Encoding Post-Vision Spatial Routing

### Question

Can spatial competition be changed after the full image has already been
encoded, without changing pixels, resolution, visual-token values, prefix,
positions, or model weights?

### Hard spatial eligibility

The image was encoded once. Only decoder access to selected already-computed
image-token keys changed.

Five of six cases passed execution trust:

- image `139`: one complete semantic-plus-geometry regional owner switch;
- image `12120`: geometry switched while description remained person-like;
- images `632`, `2299`, and `12639`: one-sided redistribution or destructive
  suppression;
- image `9400`: excluded for no-operation drift.

This establishes a real post-vision causal routing seam, but not a recurrent
instance pointer.

### Finite positive spatial bias

Biases `0.5`, `1.0`, and `2.0` produced clear and often monotonic likelihood
movement but zero complete-row or geometry owner reversal across four anchors.
Hard eligibility's positive endpoints were mainly driven by nonlinear collapse
of competing rows, not smooth constructive release of the target.

### Query-phase decomposition

- Restricting current row-scoring queries was sufficient for strong geometry
  routing and selected same-category owner switches.
- Earlier-query-only restriction barely distinguished regions and was not an
  object-identity compiler.
- Image `139`'s cross-category semantic effect required matched interaction
  between earlier-query and row-scoring-query computation.
- In a crossed-region hybrid, geometry sign followed the row-scoring region,
  while both intervals affected the first semantic token.
- Clock-Earlier and Vase-Row formed a clock-phrase and vase-geometry chimera;
  the reverse arm did not form the symmetric chimera.
- A count-balanced finite soft cross operator failed its matched owner controls,
  so its crossed arms were not interpreted.

### Belief update

The executed evidence supports a phase-separated but incomplete picture:

```text
earlier decoder computation
  -> shapes semantic compatibility and competing-owner state

current-row visual reads
  -> strongly route geometry and spatial ownership
```

Hard exclusion remains a privileged and often destructive intervention. Mild
uniform attention reweighting is not sufficient object binding.

Primary results:

- [Hard Spatial Eligibility Crossover](experiments/2026-07-15-fixed-encoding-object-centered-spatial-eligibility-crossover/results.md)
- [Finite Soft Spatial-Key Bias Dose Response](experiments/2026-07-15-fixed-encoding-soft-spatial-key-bias-dose-response/results.md)
- [Row-Scoring-Query-Only Eligibility](experiments/2026-07-15-fixed-encoding-row-scoring-query-only-spatial-key-eligibility-crossover/results.md)
- [Earlier-Query-Only Factorial](experiments/2026-07-15-fixed-encoding-earlier-query-only-spatial-key-eligibility-factorial/results.md)
- [Cross-Region Earlier-And-Row Hybrid](experiments/2026-07-15-fixed-encoding-cross-region-earlier-and-row-query-spatial-key-eligibility-hybrid/results.md)
- [Count-Balanced Soft Cross-Region Control](experiments/2026-07-15-fixed-encoding-count-balanced-soft-cross-region-earlier-and-row-query-spatial-bias/results.md)

## Phase 6: Late Residual-State Portability

### Semantic portability gate

On image `139`, a hard-routed clock-description state after decoder block `23`
recovered approximately `98.4%` of the persistent semantic release when
transplanted once into an unrestricted recipient. Decoder block `13` remained
below the preregistered threshold.

The unrestricted recipient already chose clock, so this is semantic-path
confidence portability, not an owner switch.

On image `632`, hard routing added `+2.329` to `+2.544` natural-log units per
coordinate, but donor Intersection over Union (`IoU`) was only `0.000` to
`0.018`. Coordinate release alone is therefore not evidence of instance-owned
geometry.

### Geometry-donor eligibility screen

Three clean resolution-qualified images each supplied two valid same-
description hard-routing geometry paths:

| Image | Target donor `IoU` | Paired donor `IoU` |
|---:|---:|---:|
| `7818` | `0.840` | `0.807` |
| `12576` | `0.935` | `0.847` |
| `2157` | `0.960` | `0.944` |

These boxes were substantially tighter and closer to the designated object
than the coarse regional support envelope. This falsifies the broad claim that
persistent hard routing can only copy its support envelope.

### Image-7818 geometry-state portability

Target and paired donors shared the exact prefix:

```text
<|object_ref_start|>wine glass<|object_ref_end|><|box_start|>
```

The unrestricted recipient selected annotation `664730`. One replacement of
the pre-`x1` returned residual after decoder block `23` with the paired donor
state switched the generated owner to annotation `661523` with donor `IoU`
`0.800259`. The trusted decoder-block-`13` control did not switch.

The teacher-forced per-coordinate release was:

| Coordinate | Replacement minus unrestricted log-probability |
|---|---:|
| `x1` | `+4.335076` |
| `y1` | `+0.002125` |
| `x2` | `+0.001132` |
| `y2` | `-0.000065` |

Approximately `99.9%` of the positive summed effect was at `x1`.

### Belief update

The strongest supported mechanism is:

```text
hard-routed visual support
  -> late pre-x1 residual state
  -> first-coordinate and spatial-owner basin switch
  -> native autoregressive completion of the remaining box
```

This is not evidence for a portable four-coordinate object file, general
instance identity, autonomous selection, cross-row commit, or a final bridge
layer.

Primary results:

- [Conditional Downstream Residual-State Portability](experiments/2026-07-15-fixed-encoding-downstream-residual-state-portability-gate/results.md)
- [Persistent Hard-Routing Geometry-Donor Eligibility](experiments/2026-07-15-fixed-encoding-persistent-hard-routing-geometry-donor-eligibility-screen/results.md)
- [Image-7818 Geometry-State Portability](experiments/2026-07-16-fixed-encoding-persistent-hard-routing-geometry-state-portability-image7818/results.md)

## Strongest Executed Facts

1. Multiple valid next-object modes coexist at one exact image and prefix.
2. Once a source-specific description mode is selected, native phrase and
   geometry row completion is strong.
3. Prefix state is content-sensitive executable state, not merely a token or
   row counter.
4. A coherent phrase-geometry row can alter the successor, but no stable order-
   free object ledger is established.
5. The tested successor does not require re-reading the committed object's local
   visual support.
6. Fixed post-vision features retain spatial evidence that the decoder can use
   causally.
7. Current-row visual reads strongly route geometry and spatial ownership.
8. Earlier computation and row reads interact asymmetrically in semantic
   compatibility.
9. Hard exclusion frequently works through competing-owner collapse; finite
   positive bias does not reproduce its owner switches.
10. At sufficient resolution, hard routing can compile a tight instance-owned
    geometry path rather than copy a coarse support envelope.
11. One late pre-`x1` state can causally switch the geometry owner between two
    same-description instances.
12. That positive state currently looks like a first-coordinate basin selector,
    not a complete object vector.
13. `bfloat16` batch shape is a mechanism-probe confound; exact `float32`
    control is required for low-margin token comparisons.

## Falsified Or Strongly Demoted Explanations

### Falsified under the executed protocols

- Masked reset is a safe final policy superior to equal-call full-image
  bagging.
- Native-scale tiling is better than full-canvas masking.
- Any valid complete row uniformly advances the traversal.
- Phrase alone or geometry alone explains the primary successor transition.
- Strict object-specific commit automatically redistributes probability to
  uncovered objects.
- The selected committed object's local visual support must be revalidated for
  the tested successor.
- Mild uniform regional attention bias is sufficient for owner binding.
- Earlier-query-only regional restriction is an object-identity compiler.
- Row-query-only restriction explains the complete cross-category semantic and
  geometry phenotype.
- The crossed-region semantic and geometry handoff is symmetric.
- Large coordinate-token release automatically implies tight donor geometry.
- One portable residual state has been shown to contain all four box
  coordinates.

### Demoted but unresolved

- Global post-vision competition is the dominant cause of low recall.
- A native order-free ledger exists in the prefix.
- Long token history alone explains cumulative-policy failure.
- An explicit slot, ledger, cursor, or detector-like query is necessary.
- Decoder block `23` is a unique or universal geometry layer.
- Incomplete COCO annotations are the dominant cause of conservative stopping.
- Attention normalization is the single competition mechanism.

## Integrated Mechanism Hypothesis

The following is a cross-unit synthesis, not a single executed fact:

```text
Full-image visual representation
  retains evidence for several objects
        |
        v
Prefix-conditioned trajectory state
  exposes or suppresses several valid next-object modes
        |
        v
Earlier decoder computation
  shapes semantic compatibility and competing-owner state
        |
        v
Current-row visual reads
  route geometry and spatial ownership
        |
        v
Late pre-x1 residual computation
  collapses toward one first-coordinate/owner basin
        |
        v
Emitted x1
  acts as an autoregressive route into y1/x2/y2 completion
```

This model explains why:

- bagging can recover an object that greedy misses;
- prefix perturbations alter later availability;
- forcing one description token can complete a coherent row;
- uniform positive attention bias moves likelihoods without reliably changing
  owners;
- hard eligibility can create abrupt owner switches;
- one late residual replacement can transfer a paired-object geometry path;
- no stable cross-row coverage behavior follows automatically.

The unresolved gap is endogenous state synthesis:

```text
clean image + current prefix
  -> produce the same useful late object-specific state
     that privileged hard visual routing can already compile
```

## Highest-Information Open Questions

### 1. First-coordinate causal mediation

Does the transported residual state persistently control downstream geometry,
or does it mainly select `x1`, after which ordinary autoregression recovers the
box?

Minimal discriminator:

1. force paired `x1` without residual replacement;
2. perform paired-state replacement while forcing baseline `x1`;
3. compare `y1`, `x2`, `y2`, closure, and final owner.

### 2. Generality of the owner switch

The positive portability result is one image, one same-description pair, and
one direction. Images `12576` and `2157` have qualified donors but have not been
tested for portability.

### 3. Endogenous synthesis

Can clean visual and prefix states predict or compile the privileged donor-like
late state without continued hard routing?

### 4. Commit versus canonical serialization

Does phrase-geometry coherence mean that the model commits a physical object,
or only that the row satisfies a learned geometry-sorted successor grammar?

### 5. Training learnability

Can a bounded objective concentrate sampled object support into greedy
enumeration while preserving precision, valid closure, one-shot capability,
and low unsupported hallucination?

### 6. Annotation completeness

Would exhaustive labels and controlled label thinning change object coverage,
stopping supervision, or the conservative policy?

No successor or 256-image training screen is currently authorized.

## Implementation And Research-Flow Retrospective

The scientific unit sequence became much faster after the initial spatial-
scope implementation, but the first unit was overbuilt.

### Concrete evidence

- The 2026-07-13 initial implementation commit added `51` files and
  approximately `33,567` lines.
- Fourteen sequential runtime or contract fixes followed before the unit
  closed that day.
- Later units generally used one-image or four-to-six-case panels and stopped
  after decisive evidence.
- Several later first smokes still revealed assumptions that source reading and
  tests had missed:
  - request identity was mistakenly included in one semantic comparator;
  - a predecessor batch was incorrectly assumed to be homogeneous;
  - Ground-Truth canonical rows were conflated with parent-realized donor rows;
  - conclusion receipts required bounded supersession or repair.
- These failures were caught before final interpretation, and failed artifacts
  were retained rather than silently overwritten.

### What worked

- Small fixed panels and explicit stop rules prevented unbounded Graphics
  Processing Unit (`GPU`) expansion.
- Full-model `float32` and exact no-operation controls protected delicate causal
  claims.
- Independent scientific audits caught conclusion-changing issues.
- Later experiment-local runners scaled reasonably once the intervention seam
  was known.
- Negative or ineligible results closed branches instead of triggering post hoc
  searches for a passing case.

### What was slow

- Real model smoke often occurred after too much contract and infrastructure
  work.
- The first unit generalized before a second consumer established a stable
  seam.
- Repeated worker retries were less efficient than escalating by failure type.
- Some review rounds protected promotion-quality completeness rather than the
  first pilot's conclusion validity.

### Revised minimal implementation loop

1. Write a compact decision card: question, changed factor, comparison, three
   to five conclusion-critical invariants, primary observation, and stop rule.
2. Search for the existing repository owner, native platform capability, and
   installed dependency before adding code. Reuse only when research semantics,
   validation, ordering, and provenance remain equivalent.
3. Assign one implementation owner and a bounded file surface.
4. Build the smallest skeleton that can run one real image and one arm.
5. Run that real smoke immediately, before broad manifests, exhaustive guards,
   resume support, or future-facing interfaces.
6. Classify each failure as scientific-design, runtime integration, artifact
   attribution, or ordinary mechanical implementation.
7. Fix only the blocker that can invalidate the primary observation.
8. Expand to the four-to-eight-case panel only after the smoke establishes the
   seam.
9. Run one focused result or contract audit; reuse that reviewer for the fixed-
   point recheck.
10. Close, narrow, or promote. Extract a reusable module only after another
    real unit demonstrates the same semantic owner.

## Recommended Delegated-Agent Routing

- GPT-5.6 Luna with medium reasoning: repository discovery, artifact lookup,
  and small executable preflight.
- GPT-5.6 Luna with high reasoning: thin deterministic implementation with an
  explicit mechanical test.
- GPT-5.6 Luna with maximum reasoning: bounded implementation only when the
  acceptance criterion is executable and scientific judgment is not delegated.
- GPT-5.6 Sol with medium reasoning: scientific protocol review, research
  synthesis, and contract audit.
- GPT-5.6 Sol with high reasoning: novel hook, cache, row-state, precision, or
  other conclusion-critical integration and repair.
- GPT-5.6 Sol with extra-high reasoning: only an unresolved Priority zero or
  Priority one causal contradiction after focused evidence collection.

Use one implementation owner. Allow one focused follow-up for a local defect;
then escalate by failure type rather than spawning another duplicate worker.
Use `fork_turns=1` for discovery or worker lanes and `fork_turns=2` for a review
that needs the recent scientific decision. Full-history inheritance is an
exception.

## GPT-Pro Package: Key Files

### Tier 1: program routers

1. [Research compass](compass.md): north star, belief register, demoted claims,
   and next-discriminator boundary.
2. [Investigation overview](overview.md): detailed hypothesis and evidence map.
3. [Experiment index](experiments/index.md): one-line verdict for each unit.
4. [Investigation index](index.md): complete reading path and upstream
   decisions.
5. This weekly report.

### Tier 2: core state-transition chain

1. [Spatial scope and bagging](experiments/2026-07-13-spatial-scope-history-disentanglement/results.md)
2. [Fixed-prefix sampled rescue](experiments/2026-07-14-sampled-rescue-object-transition-causal-replay/results.md)
3. [Phrase-geometry factorial](experiments/2026-07-15-prefix-state-phrase-geometry-factorial/results.md)
4. [Post-vision committed-support counterfactual](experiments/2026-07-15-visual-support-counterfactual-commit/results.md)
5. [Pre-vision raw-box counterfactual](experiments/2026-07-15-pre-vision-raw-bounding-box-visual-support-counterfactual-commit/results.md)
6. [Native coherent-row commit and execution-invariance failure](experiments/2026-07-15-native-coherent-row-commit-to-uncovered-redistribution/results.md)

### Tier 3: fixed-encoding spatial mechanism

1. [Hard spatial eligibility](experiments/2026-07-15-fixed-encoding-object-centered-spatial-eligibility-crossover/results.md)
2. [Finite soft-bias response](experiments/2026-07-15-fixed-encoding-soft-spatial-key-bias-dose-response/results.md)
3. [Row-scoring-query-only eligibility](experiments/2026-07-15-fixed-encoding-row-scoring-query-only-spatial-key-eligibility-crossover/results.md)
4. [Earlier-query factorial](experiments/2026-07-15-fixed-encoding-earlier-query-only-spatial-key-eligibility-factorial/results.md)
5. [Cross-region hybrid](experiments/2026-07-15-fixed-encoding-cross-region-earlier-and-row-query-spatial-key-eligibility-hybrid/results.md)
6. [Count-balanced soft cross-region matched-control failure](experiments/2026-07-15-fixed-encoding-count-balanced-soft-cross-region-earlier-and-row-query-spatial-bias/results.md)

### Tier 4: late state and runtime trust

1. [Conditional residual portability](experiments/2026-07-15-fixed-encoding-downstream-residual-state-portability-gate/results.md)
2. [Geometry-donor eligibility](experiments/2026-07-15-fixed-encoding-persistent-hard-routing-geometry-donor-eligibility-screen/results.md)
3. [Image-7818 geometry-state portability](experiments/2026-07-16-fixed-encoding-persistent-hard-routing-geometry-state-portability-image7818/results.md)
4. [Batch-coordinate invariance](experiments/2026-07-15-homogeneous-mixed-batch-coordinate-logit-invariance/results.md)
5. [Visual-feature replay](experiments/2026-07-15-single-target-visual-feature-replay-into-homogeneous-batch-four/results.md)
6. [Batch-precision prevalence](experiments/2026-07-15-selected-transition-batch-precision-prevalence-screen/results.md)
7. [Repeated coordinate panel](experiments/2026-07-15-repeated-first-differing-slot-full-coordinate-logit-panel/results.md)

## Conclusion-Owning Artifact Receipts

### Hard spatial eligibility

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-07-15-fixed-encoding-object-centered-spatial-eligibility-crossover/
cohort-six-float32-20260715b/receipt.json

Secure Hash Algorithm 256-bit (`SHA-256`):
6148fa75998385eeab0ac0118d3fe540acaa6c7dfece85ee6116907ed3a703ee
```

### Geometry-donor eligibility

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-07-15-fixed-encoding-persistent-hard-routing-geometry-donor-eligibility-screen/
cohort-six-float32-20260715a/merged/receipt.json

SHA-256:
2046d5784030f255bc039b9252b2e63770e9f9b069219cfa9478a6e94c8f59b2
```

### Final image-7818 geometry portability

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-07-16-fixed-encoding-persistent-hard-routing-geometry-state-portability-image7818/
image7818-layer23-13-float32-20260716a/receipt.json

SHA-256:
73805bc237276de1595c6d0ea071db047a6498a97b87d1f0d2cdcee9e56e0ec4
```

## Packaging Warning

At report creation time, substantial parts of the current research tree are
modified or untracked. A package based only on `git ls-files` can omit the
research compass and multiple 2026-07-14 through 2026-07-16 units. Package from
the actual filesystem paths above, or first perform an explicit reviewed Git
collection step. Do not assume tracked-file status is evidence status.

## Questions For GPT-Pro

1. Does the first-coordinate mediation model best explain the image-`7818`
   owner switch, or is there a stronger alternative consistent with all four
   per-coordinate effects?
2. What is the smallest training scaffold that could teach clean Qwen3-VL to
   synthesize the privileged late state without replacing its pretrained visual
   and language knowledge?
3. Can phrase-geometry transaction consistency be turned into an early,
   object-contrastive credit signal without imposing detector slots or an
   external backbone?
4. How should commit and coverage be modeled if local committed-object visual
   revalidation is unnecessary but prefix state is fragile?
5. Which next experiment most sharply distinguishes a first-coordinate route,
   a persistent geometry state, and a geometry-sorted serialization policy?
6. What evidence threshold should precede a 256-image learnability screen?

The immediate requested task for GPT-Pro is mechanism synthesis and experiment
selection, not final architecture design.
