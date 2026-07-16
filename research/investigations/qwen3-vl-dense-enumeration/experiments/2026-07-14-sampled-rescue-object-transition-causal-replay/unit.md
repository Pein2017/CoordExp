---
title: Sampled-Rescue Object Transition Distribution and Causal Replay
description: Tests whether greedy-missed objects are stable next-row modes under an identical prefix and identifies which row phase or committed transition blocks greedy enumeration.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: complete_for_unit
unit_id: 2026-07-14-sampled-rescue-object-transition-causal-replay
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-07-14
---

# Sampled-Rescue Object Transition Distribution and Causal Replay

## Terminology

- **Qwen3 Vision-Language (`Qwen3-VL`)**: the pretrained multimodal model
  family under investigation.
- **Fixed-prefix one-row transition distribution**: the empirical distribution
  of the next parsed object row or terminal action produced from an identical
  image, prompt, legal prior-row prefix, and object-row boundary.
- **Sampled-rescue object**: a verified reportable object omitted by greedy
  decoding but emitted by at least one sampled continuation under the declared
  comparison state.
- **Causal replay**: release greedy decoding after forcing only one declared
  transition component, such as row continuation, description, first
  coordinate, or a complete committed row.
- **Common pre-row boundary**: the last token-identical, legal object-row
  boundary shared by a greedy trajectory and a sampled trajectory before their
  first different row-level action.
- **Phase-divergence boundary**: a token-identical prefix inside an object row
  immediately before greedy and sampled trajectories first differ in row
  continuation, description, geometry, or closure. It is analyzed separately
  from every pre-row boundary.
- **Rescue-entry boundary**: the exact sampled prefix immediately before a
  sampled-rescue object is emitted.
- **Greedy-terminal boundary**: the exact greedy prefix immediately before the
  model emits its terminal no-more-objects action.
- **Candidate-path score**: a teacher-forced diagnostic for one declared row
  phase or complete candidate row under one fixed prefix. First-differing-token
  log odds, conditional phase negative log likelihood, raw sums, token counts,
  and phase means remain separate; terminal and full-row scores are never
  directly compared as one headline margin.
- **Complete-row-conditioned transition redistribution**: the observed change
  in next-row candidate scores or empirical frequencies after a complete donor
  row is appended.
- **Object-specific commit evidence**: a stricter crossover signature in which
  each valid donor row suppresses its own object more than another object's
  row does and redistributes support toward a predeclared uncovered object,
  while duplicate and irrelevant complete-row controls do not reproduce the
  effect.
- **Object-transition outcome**: one of new verified object, already committed
  object, unsupported output, invalid row, or terminal no-more-objects action.
- **Common Objects in Context 80-category ontology (`COCO-80`)**: the closed
  reportable category set used by this unit.
- **Weight-Decomposed Low-Rank Adaptation (`DoRA`)**: the adapter method used by
  the frozen checkpoint inherited from the preceding dense-enumeration unit.
- **No-resize image processing (`do_resize=false`)**: the processor preserves
  the input image scale rather than resizing it before model encoding.

## Question

At an identical image and prefix state, do objects missed by greedy decoding
appear as valid next-row modes, or do they require an earlier divergent
trajectory? If an object row is successfully emitted, does the native prefix
state commit that object and redistribute the next transition toward a valid
uncovered object?

## Testable Hypotheses

| Hypothesis | Expected mechanism | Supporting prediction | Falsifying observation |
|---|---|---|---|
| **Fixed-state object-mode fragmentation** | Several real objects retain non-negligible next-row support at one identical state, but greedy decoding selects only the strongest sequence branch. | A greedy-missed object recurs under repeated one-row sampling at the common pre-row or greedy-terminal boundary and has a competitive phase-local diagnostic rather than only parser noise. | Under an adequately powered declared screen, the object remains materially weaker at both unchanged states but becomes stable under its successful sampled prefix. |
| **Earlier-trajectory state dependence** | Early row choices change the executable prefix state, making later objects available or unavailable. | The object is stable at the rescue-entry boundary but materially weaker at the common pre-row and greedy-terminal boundaries; transplanting the successful sampled prefix restores it. | The object appears reliably from the unchanged greedy state. |
| **Phase-localized object transition barrier** | Row continuation, description ownership, geometry ownership, and row closure are separable decisions; one phase blocks an otherwise supported object. | Forcing one phase produces a source-specific gain that survives into the complete row, while shorter forcing and wrong-object controls do not. | Correct-object and wrong-object forcing have the same effect, or only generic continuation increases. |
| **Object-specific commit hypothesis** | A completed object row acts as an object-indexed executable state transition rather than only text advancement or geometry-sorted successor prompting. | A complete-row crossover shows donor-specific self-suppression and redistribution to a predeclared uncovered object; duplicate and irrelevant controls fail to reproduce it. | Every plausible complete row produces the same successor advance, or probability moves only to terminal, invalid, repeated, or unsupported rows. |
| **Generic continuation alternative** | The intervention merely makes the model write another row without conveying object identity or coverage. | Row count or continuation rises equally for correct, wrong-object, and irrelevant controls. | Only the correct-object intervention changes phrase and geometry ownership and the following uncovered-object transition. |

## Strongest Competing Explanation

Bagging may recover objects only because different early rollout choices create
different later prefix states. Under an identical prefix, the missed object may
have no stable support. Alternatively, pixel masking may change vision-tower
representations rather than expose a language-side transition mode.

The fixed-prefix one-row comparison separates current-boundary probability from
earlier trajectory divergence. Wrong-object and irrelevant replay controls
separate source-specific object control from generic continuation. A later
fixed-encoding spatial unit remains responsible for vision-side localization.

## Primary Observation

For each selected legal boundary, report candidate-path scores and descriptive
empirical frequencies of:

```text
terminal no-more-objects action
new verified object
already committed object
unsupported output
invalid row
```

The initial eight-seed panel is a mode-discovery screen, not a precise
probability estimate. Zero hits means only "not observed at the screening
budget." The primary temperature-`0.4`, top-p nucleus-threshold-`0.95` state
expands to 32 seeds when it has a valid hit, when the source bagging trajectory
proves the mode occurred at that temperature, or when candidate diagnostics
keep the mode plausible. With zero hits in 32 independent draws, the one-sided
95-percent binomial upper bound is reported; the mode is not called impossible.

For each sampled-rescue object, report whether forcing row continuation,
description, first coordinate, or a complete row causes source-specific
full-row recovery. After complete-row forcing, measure complete-row-conditioned
transition redistribution. Promote it to object-specific commit evidence only
after the declared crossover controls succeed.

## Exploratory Outline

### Phase 0: Existing trajectory re-analysis

- Reuse completed full-image bagging outputs before launching new inference.
- Verify the actual decode policy of every inherited arm. The inherited
  `FULL_SINGLE` arm is a single temperature-`0.4` sampled rollout rather than
  greedy decoding, so generate one exact repetition-penalty-`1.0` greedy
  anchor for each retained image before labeling any object as greedy-missed.
- On four to eight visually reviewed dense images, cluster boxes by
  class-agnostic geometry first, then estimate category disagreement within
  each geometry mode.
- Record per-object inclusion frequency, first-hit row, first trajectory
  divergence, repeated-hit count, and pairwise inclusion association.
- Reconstruct the common pre-row, phase-divergence, rescue-entry, and greedy-
  terminal boundaries for each retained rescue trajectory. Record exact token
  identifiers, grammar phase, row index, and the two-trajectory equality check.
- Select boundaries where one interpretation can be visually and
  annotation-audited.
- Before causal replay, freeze a per-boundary object ledger containing verified
  reportable objects, prefix-covered objects, verified uncovered objects,
  uncertain objects, and unsupported regions. New outputs must map to one
  frozen object identifier or to uncertain, unsupported, or invalid.

Candidate images from the prior review are `2299`, `19432`, `12576`, `17959`,
`15254`, and `7816`. Final selection remains bounded to four through eight and
must state why each case is mechanistically useful.

Each retained case must contribute a distinct mechanistic role: dense
same-class competition, small-object rescue, category disagreement at stable
geometry, late rescue after divergent history, or apparent unlabeled-object
rescue. Do not select cases merely because they have the largest metric delta.

### Phase 1: Fixed-prefix candidate scoring and one-row distribution

- Hold image, prompt, legal prefix, object-row boundary, checkpoint, tokenizer,
  processor, and parser fixed.
- Score the greedy candidate, sampled-rescue candidate, terminal action, and a
  same-image wrong-object candidate under teacher forcing. Make empirical
  one-row outcomes primary. Use first-differing-token log odds and conditional
  phase negative log likelihood for localization; retain full-row joint log
  probability only as a matched-candidate diagnostic with explicit token
  counts.
- At each available common pre-row, rescue-entry, and greedy-terminal
  boundary, compare greedy decoding with sampled one-row continuations at
  temperatures `0.2`, `0.4`, and `0.6`, top-p nucleus threshold `0.95`, and
  initially eight seeds per sampled condition. Greedy controls disable
  sampling.
- Use neutral repetition penalty `1.0` for the primary interpretation.
- Expand the primary temperature-`0.4` state according to the declared
  32-seed rule above. Temperatures `0.2` and `0.6` are separate sensitivity
  distributions and are never pooled to tighten a frequency bound.
- Support earlier-trajectory dependence only when rescue-entry or transplanted-
  prefix behavior repeats and the unchanged state remains materially weaker
  under its own candidate diagnostics and powered temperature-`0.4` panel.
- Do not create a population metric from the selected cases.

### Phase 2: Sampled-rescue causal replay

Use the exact verified row from a successful sampled trajectory as the primary
donor. Do not replace it with a ground-truth row in the primary causal claim.
Construct nested donor prefixes from the same fixed boundary and release greedy
continuation after, separately:

1. no intervention;
2. forcing the row-entry wrapper through the object-reference opener;
3. extending that donor prefix through the first description token that
   distinguishes the rescued object;
4. extending through the complete description and description closer;
5. extending through the geometry opener and first horizontal coordinate;
6. extending through the complete rescued-object row, then scoring the next
   transition;
7. transplanting the complete successful sampled prefix;
8. repeating every nested depth with a same-image wrong-object donor;
9. repeating the relevant depth with an irrelevant but syntax-matched control.

Adjacent nested arms must differ only by their declared incremental donor span.
Interpret phase effects as the incremental change from the preceding arm, not
as independent interventions.

Phrase owner, geometry owner, row validity, closure, donor-object suppression,
predeclared-uncovered-object redistribution, and terminal score remain separate
observations.

At complete-row depth, compare four controls: the rescued-object row, a
complete same-image wrong-object row, a duplicate row for an already covered
object, and a syntax/length-matched irrelevant row. The primary observable is
complete-row-conditioned transition redistribution. Object-specific commit is
supported only by a donor-object crossover; if geometry-sorted successor
prompting remains sufficient, phrase/geometry swap controls become a later
discriminator rather than an automatic extension of this unit.

Historical late-middle residual evidence may select a small secondary patch
panel after token-level replay localizes a phase. It does not preselect one
bridge layer. Candidate starting bands are layers 17 through 21 for
stop-versus-continue and layers 20 through 23 for description identity, with
the exact historical evidence linked from the [research compass](../../compass.md).

### Phase 3: Conditional late-middle residual replay

Run this phase only when Phase 2 identifies a source-specific token phase and
the corresponding full-row outcome remains incomplete. Compare correct-donor,
wrong-object-donor, clean no-op, and source-swapped patches in the smallest
historically supported layer band. Do not perform a broad layer or attention-
head sweep inside this unit.

## Scientific Invariants and Conclusion-Threatening Checks

| Surface | Required invariant or check | Why it matters |
|---|---|---|
| Model | Exact checkpoint, adapter payload, tokenizer, processor, special-token surface, and prompt renderer remain fixed. | Prevents a loading or template difference from masquerading as a transition effect. |
| Image | Full-image pixels and `do_resize=false` processing remain identical across fixed-prefix arms. | Keeps this unit separate from the later fixed-encoding spatial-localization unit. |
| Prefix | Token identifiers must be exactly equal within a fixed-prefix comparison. | The central claim is invalid if different histories are compared as one state. |
| Decode | First-row projection rule, repetition penalty, temperature, sampling flag, maximum token budget, and parser remain explicit. Full generation may be retained after the first complete row because later causal tokens cannot alter the already generated first-row transition. | Prevents length or parser behavior from becoming the hidden changed factor without inventing a new backend stopping interface. |
| Sampling | Request seeds derive from an immutable run root and remain unique within each estimated image, boundary, and temperature distribution. Any cross-state seed reuse is intentional paired randomness and remains explicit in the receipts. | Prevents accidental repeated draws without misrepresenting paired cross-state comparisons as independent samples. |
| Numeric reduction | Accumulate token log probabilities, margins, hidden-state differences, and causal-effect reductions in 32-bit floating point. | Avoids low-precision cancellation in close candidate comparisons without requiring full-model 32-bit execution. |
| Object identity | Geometry modes are clustered before category labels are compared, and every headline rescue is visually audited. | Separates one object with category uncertainty from several independent discoveries. |
| Annotation | Unmatched outputs are classified as unlabeled real object, repeated or fragmented object, localization/category mismatch, unsupported hallucination, or uncertain. | Incomplete dense labels cannot safely define false positive or termination truth alone. |
| Boundary ledger | Verified, prefix-covered, uncovered, uncertain, and unsupported object states are frozen before causal replay outputs are inspected. | Prevents post-hoc promotion of a replay output into a successful uncovered object. |

Only a failure that can change the scientific conclusion blocks the first
smoke. Formatting, broad reproducibility hardening, future resume behavior, and
unobserved edge cases are deferred.

## Scope and Reused Surfaces

- Worktree: `/data/CoordExp/.worktrees/research-probes`.
- Primary inference configuration:
  `/data/CoordExp/.worktrees/research-probes/configs/coordexp_swift/infer/qwen3_vl_2b_desc_first_geo_sorted_gaussian_rps_dora_r16a32_step4887_val200.yaml`
  with Secure Hash Algorithm 256-bit (`SHA-256`) digest
  `f3000588accbcf1d9ada3b2f3e0b3324d660b4810b75d8f5d050d9f184f9ca80`.
- Primary checkpoint:
  `/data/CoordExp/.worktrees/research-probes/outputs/prod/coordexp_swift/qwen3_vl_2b_desc_first_geo_sorted_gaussian_rps_dora_r16a32_llm_12000_accelerate8_ebs24_8epoch_warmup0p1/checkpoints/step-4887/checkpoint.json`
  with SHA-256 digest
  `613e5d97f4a7a53d6325b5c1909813d6bb82e72622942b225df9724b5556a536`.
- The historical path token `gaussian_rps` means **Gaussian Soft-Target
  Coordinate Cross-Entropy with Ordered Cumulative-Distribution Penalty**. It
  is retained only for provenance and is not a canonical abbreviation in this
  unit.
- Existing evidence source: the two completed seed roots and review artifacts
  from the [spatial-scope and history unit](../2026-07-13-spatial-scope-history-disentanglement/results.md).
- Existing manually reviewed comparison manifest:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-13-spatial-scope-history-disentanglement/review/mask-reset-vs-full-bag-k-root-2026071301/manifest.json`.
- Existing geometry-first unique-object review:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-13-spatial-scope-history-disentanglement/review/mask-reset-vs-full-bag-k-root-2026071301/unique-object-coverage.json`.
- Reuse current CoordExp-Swift model loading, sampled batch inference, parsing,
  accepted reference ledger, and visualization surfaces.
- Add only the missing trajectory analyzer and bounded one-row or forced-prefix
  execution seams. Do not freeze a reusable code interface before the first
  real smoke.
- The [execution plan](execution-plan.md) owns staged graphics-processing-unit
  use, expected source surfaces, adaptive subagent routing, and the explicit
  user approval gate. It is not an implementation contract.

## Non-Goals

- no model training;
- no architecture, slot, ledger, cursor, or detector implementation;
- no broad layer or attention-head sweep;
- no claim that temperature or bagging is the final inference policy;
- no population mean average precision estimate from the selected case panel;
- no fixed-encoding spatial intervention inside this unit.

## Representative Smoke and Stop Rules

Start with one image, one legal prefix, greedy decoding with sampling disabled,
and eight sampled one-row continuations at temperature `0.4` and top-p nucleus
threshold `0.95`. Inspect raw rows and parser status before adding temperatures
or causal replay.

Stop or narrow when:

- no sampled-rescue object appears under the identical prefix, which redirects
  the next unit toward earlier trajectory-state dependence only after the
  declared powered comparison; zero of eight alone does not trigger this;
- candidate scoring and generation disagree because of an implementation or
  token-boundary error, which blocks interpretation until the narrow mismatch
  is fixed;
- the correct-object replay is not stronger than wrong-object or irrelevant
  controls, which rejects that replay seam as generic continuation;
- description changes without source-consistent geometry, which narrows the
  next probe to instance binding rather than full-row recovery;
- forcing a complete row does not change committed or uncovered candidate
  scores, which weakens a native commit-transition explanation;
- one fixed-prefix case yields a clear directional signature, in which case
  replicate on the remaining selected cases before adding more machinery.

## Rough Cost

The first observation is an analysis-only pass over existing artifacts. The
first graphics-processing-unit smoke is one-row generation on one image. The
full exploratory panel remains four through eight images with short
continuations; it is intentionally much smaller than the previous policy
matrix.

## Artifact Handle

Logical root:

```text
outputs/research/qwen3-vl-dense-enumeration/2026-07-14-sampled-rescue-object-transition-causal-replay/<immutable-run-id>/
```

Resolved durable root:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-14-sampled-rescue-object-transition-causal-replay/<immutable-run-id>/
```

Each run must retain checkpoint and config identity, image and prefix identity,
decode condition and seed, raw generated row, parser status, request identity,
and the primary object-transition classification. A path alone is not evidence.

## Outcome-to-Route Map

| Observation | Route update |
|---|---|
| Stable valid rescued modes at one identical prefix | Authorizes a bounded uncovered-object transition training candidate. |
| Rescues require earlier divergent prefixes | Prioritize on-policy prefix and commit-state training rather than fixed-state margin shaping. |
| Continue forcing is sufficient | Study remaining-object mass versus terminal action; do not merely suppress termination. |
| Description forcing is sufficient | Study semantic next-object competition with full-row source-specific controls. |
| First-coordinate forcing is required | Test early spatial commitment or geometry-specific binding. |
| Complete-row forcing changes the next object without a donor crossover | Supports a prefix-conditioned successor effect, not object-specific commit. |
| Complete-row donor crossover passes | Supports an object-specific commit-transition training target; does not yet prove a persistent ledger. |
| Only pixel-level masking rescues the object | Prioritize fixed-encoding spatial localization before language-side training. |

## Executed Evidence and Bounded Verdict

Execution is closed. The verified [results](results.md) support fixed-state
object-mode fragmentation at one exact prefix, earlier-trajectory state
dependence in two cases, and strong within-row description-conditioned
phrase-and-geometry binding. The complete-row crossover required for strict
object-specific commit-to-uncovered redistribution failed in the primary case.
The observed cross-row transition is instead compatible with a
geometry/order/frontier-like successor process plus generic complete-row
advancement, while its exact causal factor remains unresolved.

Wave 4 was not entered because first-description-token forcing already
completed valid phrase-and-box rows; the remaining question concerns cross-row
prefix state rather than residual persistence. A 256-image training screen is
not authorized. The only next discriminator is the bounded **Prefix-State
Phrase-Geometry Factorial**, defined operationally in the results record.

## Closure Contract

Close the unit with separate `Observed`, `Supported`, `Ruled out`, `Unresolved`,
and `Not claimed` sections. Name one next discriminator and state explicitly
whether a 256-image training screen is authorized by the evidence. Do not create
an empty `results.md` before execution.

## Approval State

The user authorized bounded implementation and graphics-processing-unit
execution for this research goal on 2026-07-14. Waves 0 through 3 executed and
closed; Wave 4 stopped at its declared gate. Training, architecture changes,
OpenSpec work, and a 256-image training screen were not authorized and did not
run.
