---
title: Human-Audited Rare-Object Trajectory Genealogy and Causal Branch Replay
description: Artifact-first study of why intermittently retrieved objects become reachable or disappear along native Qwen3 Vision-Language rollout branches.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: complete_for_manual_review_only
unit_id: 2026-07-16-human-audited-rare-object-trajectory-genealogy
topic: qwen3-vl-dense-enumeration
status: redirected
evidence_status: partial_verified
updated: 2026-07-16
---

# Human-Audited Rare-Object Trajectory Genealogy and Causal Branch Replay

The human-review phase is complete, but the physical-entity consolidation gate
did not close. The review exposed a more immediate coordinate-composition
question, so the trajectory-genealogy waves are held rather than silently
continued. See [the bounded results and redirection verdict](results.md).

## Terminology

- **Qwen3 Vision-Language (`Qwen3-VL`)**: the pretrained multimodal model
  family under investigation.
- **Common Objects in Context 80-category ontology (`COCO-80`)**: the closed
  reportable category universe for this unit. Visible objects outside these 80
  categories are out of scope rather than model hallucinations.
- **Full-Image K-Rollout Independent Bagging (`FULL_BAG_K`)**: independent
  stochastic full-image rollouts under the frozen decode policy; `K` is the
  number of calls inherited from the spatial-scope comparison.
- **Ground Truth (`GT`)**: the accepted reference ledger available before this
  unit. The ledger contains official annotations and previously audited
  additions, but is not assumed to exhaust every visible object.
- **Exact prefix state**: the fixed image, complete token prefix, Key-Value
  cache construction, prompt, model, and decode/runtime policy immediately
  before one legal object-row boundary.
- **Native row action**: one complete valid description-first object row
  naturally emitted from an exact prefix state. Directly forcing a target row
  is an oracle upper bound, not a native retrieval.
- **Future-recovery probability**: for target object `X`, the probability that
  `X` is emitted before the terminal no-more-objects action within a declared
  additional-row horizon when decoding resumes from an exact prefix state.
- **Candidate-conditioned unique retrieval**: unique physical entities in the
  frozen review universe that appear in a trajectory. It is not complete scene
  recall.

## Authorization Boundary

The user authorized:

1. this research specification;
2. the static unmatched-prediction review page;
3. artifact-only cohort construction and re-analysis needed to prepare manual
   review.

New graphics-processing-unit inference, model training, architecture changes,
and a 256-image training screen remain unauthorized. The review tool is an
experiment-local scaffold, not a stable product or OpenSpec contract.

## Question

For a real `COCO-80` object retrieved by only some stochastic trajectories:

> Is the object already a lower-ranked mode at one identical prefix, or does a
> prior naturally emitted row selectively unlock, preserve, or destroy its
> future recoverability; and can the natural transition be replayed strongly
> enough to change greedy decoding without reducing later unique-object
> utility?

This unit does not assume that Qwen3-VL requires or lacks an explicit
covered-set carrier. The native Attention plus Multilayer Perceptron computation,
an information-equivalent recoding of model outputs, a learned compressed
state, and an explicit runtime carrier remain competing implementation
families.

## Competing Explanations

### Fixed-state object-mode competition

Several valid rows have nonzero probability at the same exact prefix. Sampling
occasionally chooses a lower-ranked target row; before the first differing
token, the successful and failed draws have identical model state.

### Earlier-trajectory unlocking or killing

A naturally emitted earlier row changes the executable prefix state so that a
target gains or loses future-recovery probability.

### Generic row progression

Any syntactically valid row advances a learned serialization process. A
seemingly successful predecessor is not target-specific and is reproduced by
irrelevant, covered-duplicate, or wrong-object rows.

### Recognition or localization limit

The target never acquires reproducible support at a controlled state. Bagging
may expose unstable fragments, category errors, or annotation artifacts rather
than an executable object mode.

### Runtime numeric branch

Close coordinate or object decisions change with physical batch shape or
reduced precision. Conclusion-critical branch changes must reproduce at
physical batch size one or in 32-bit floating-point scoring before they are
interpreted as a semantic mechanism.

## Primary Observation

Let `s` be an exact prefix state and `a` a complete native row. For a manually
audited target object `X`, estimate:

```text
future-recovery-probability(X, s)
future-recovery-probability(X, s plus a)
```

and the action-conditioned downstream utility:

```text
future unique supported entities
- duplicate rows
- unsupported rows
- invalid rows
```

The primary discriminator is whether one **natural** positive sibling row
changes target recovery on fresh suffix seeds more than all generic, wrong,
and covered-object controls. Direct target-token or target-row forcing is only
an accessibility upper bound.

## Frozen Five-Image Pilot Cohort

The pilot cohort is selected before the new per-prediction verdicts are known
and must not be silently replaced after review.

| Image identifier | Mechanistic role | `FULL_BAG_K` post-merge unmatched candidates |
|---|---|---:|
| `12576` | Fixed-state pizza versus cup mode fragmentation and same-scene food-instance competition | 15 |
| `19432` | Earlier-state chair rescue plus dense repeated-class and duplicate-expansion control | 85 |
| `9400` | Category and localization disagreement on visible people, laptops, bottles, and nearby objects | 15 |
| `7816` | Low-unmatched, relatively clean bagging negative case | 4 |
| `15254` | Retained null case with no `FULL_BAG_K` unmatched candidate | 0 |

The total first-pass annotation queue is 119 candidates. The panel is a
purposive mechanism cohort, not a prevalence or population-precision sample.

## Bounded Review Universe

For this unit, the review universe is:

```text
accepted reference-ledger entities
union
physical entities proposed by the included FULL_BAG_K predictions
```

Objects missed by both the accepted reference ledger and every included model
trajectory are intentionally outside scope. Consequently, this unit must not
claim:

- complete scene coverage;
- true absolute recall;
- correct exhaustive termination;
- population hallucination rate;
- absence of still-undiscovered objects.

Permitted claims are candidate-conditioned unique retrieval, audited proposed-
object support, and trajectory-conditioned rescue.

## Human Review Contract

### First pass: blinded candidate review

The static reviewer presents exactly one highlighted unmatched prediction at a
time over the original image, with the accepted reference ledger available as
an overlay. It hides arm name, seed, frequency, score, greedy-versus-sampled
identity, and later causal outcomes. Hidden provenance remains in the export
for later joins.

The three headline verdicts are:

- **Approve**: a real, reportable `COCO-80` physical entity supports the
  prediction. Approval does not imply that category and geometry are both
  correct.
- **Reject**: no reportable entity supports the prediction, or the visible
  entity is outside the declared `COCO-80` ontology.
- **Unknown**: the image evidence is insufficient for a reliable decision.

To keep object retrieval separate from row correctness, every decision also
retains:

```text
entity reference
semantic status: exact, wrong category, ambiguous, or not applicable
geometry status: acceptable, localization error, fragment, multiple entities,
                 or ambiguous
free-text comment
```

Approval is not complete until one canonical entity reference is assigned.
Several predictions may share the same entity reference. A new visible but
previously unlabeled entity receives a stable identifier of the form:

```text
human:<image-id>:<four-digit-local-entity-index>
```

### Second pass: entity consolidation

After every candidate has a headline verdict, approved candidates are grouped
by physical entity. Category-disagreeing, fragmented, or repeated boxes may
share one entity reference. Unique-object counts are computed from entity
references, never from prediction count.

Unknown candidates remain outside both positive and negative denominators and
cannot become causal targets.

### Freeze gate

The reviewed ledger is exported and hashed before branch discovery or new
causal replay. The frozen record must contain:

- review-set identifier;
- source manifest, queue, and accepted-ledger digests;
- reviewer identity;
- selected image identifiers and ontology;
- candidate and hidden trajectory provenance;
- verdict, entity reference, semantic status, geometry status, comment, and
  timestamp;
- annotation-gate summary.

### Candidate-to-trajectory join gate

Human annotation may begin before this gate closes. Wave 0 trajectory
genealogy and every later causal claim may not begin until each reviewed
candidate is joined back to its native source evidence.

For all 119 candidate identifiers, the join receipt must resolve:

```text
exact source request or call
raw complete rollout artifact
parsed object row and row index corresponding to the prediction span
sampling seed and decode-policy identity
```

The required compact receipt is:

```text
119 resolved
0 missing
0 ambiguous
0 row or span mismatches
```

If a post-merge candidate cannot be resolved, preserve its human verdict for
visual-validity analysis but exclude it explicitly from trajectory genealogy.

## Experimental Outline

### Wave 0: artifact-only trajectory atlas

1. Pass the candidate-to-trajectory join gate.
2. Reuse existing `FULL_BAG_K` terminal call bundles and the frozen reviewed
   entity ledger.
3. Parse complete suffixes, not only first actions.
4. For every accepted entity, record inclusion frequency, first-hit row,
   category variants, geometry variants, repeated hits, and trajectory
   co-occurrence.
5. Separately report:

```text
bagged-union candidate-conditioned coverage
best executable single-trajectory coverage
greedy candidate-conditioned coverage
```

The union-minus-best-path gap estimates path incompatibility. The best-path-
minus-greedy gap estimates policy concentration or optimization failure.

### Wave 1: exact-prefix sibling branch atlas

For each eligible intermittent target:

1. identify a legal pre-row exact prefix shared by successful and failed
   branches;
2. sample natural complete sibling rows at that state;
3. from each sibling row, run a fixed additional-row horizon with paired fresh
   suffix seeds;
4. estimate target future-recovery probability and the complete future entity
   inclusion vector;
5. classify each sibling as direct hit, unlock, preserve, kill, absorption, or
   generic progression.

Object-name sequences are a visualization only. Statistical comparisons use
the exact prefix state and complete native row action.

### Wave 2: causal natural-row replay

At no more than three critical branch states per target, compare:

1. no appended row;
2. positive natural sibling row;
3. negative natural sibling row;
4. syntax- and approximately length-matched irrelevant row;
5. covered-duplicate row;
6. wrong-object row;
7. direct target-row forcing as an oracle upper bound.

Discovery trajectories select the branch. Fresh suffix seeds estimate the
effect. Use a fixed additional-row horizon rather than only a token cap.

### Wave 3: greedy reproducibility

Only after a natural row has a selective causal effect:

1. append that row and release greedy decoding;
2. require the target to become top-ranked with positive margin;
3. verify phrase and geometry ownership;
4. measure total downstream unique-object utility;
5. reproduce conclusion-critical near ties at physical batch size one or with
   32-bit floating-point score accumulation.

### Wave 4: conditional internal-state analysis

Enter only if Wave 2 establishes a robust natural unlock or kill transition.
Compare good-branch and bad-branch next-row boundary states, then use the
smallest source-specific residual or Key-Value-cache intervention. The causal
readout is recovery of the full target future-recovery probability, not a
single coordinate token.

## Target Eligibility

An object may enter causal analysis only when:

- the human verdict is approved;
- one canonical entity reference is frozen;
- at least one natural positive and one natural negative trajectory exist;
- either two positive and two negative discovery trajectories exist, or fresh
  exact-prefix sampling demonstrates recurrent support;
- the target is not explained only by a duplicate, fragment, invalid row, or
  unsupported category.

Objects with one isolated hit remain leads rather than headline evidence.

## Controls and Invariants

- Keep checkpoint, adapter payload, tokenizer, processor, prompt renderer,
  image, `do_resize=false`, parser, repetition penalty `1.0`, and decode policy
  fixed within a comparison.
- Freeze exact token identifiers and prefix hash for every sibling-state
  comparison.
- Use fresh suffix seeds for confirmation; do not discover and confirm an
  effect with the same continuations.
- Treat one trajectory, not each emitted object, as the resampling unit.
- Accumulate close log probabilities, margins, and aggregate effects in
  32-bit floating point. Full-model 32-bit execution is reserved for a
  conclusion-changing near tie or batch-dependent branch.
- Keep entity retrieval, category correctness, geometry correctness, row
  validity, duplicate behavior, unsupported output, and terminal action as
  separate observations.
- Preserve the five selected images even when review finds no eligible rescue.

## Outcome-to-Route Map

| Observation | Route update |
|---|---|
| Target recurs at one exact state, but no earlier row selectively changes its future recovery | Prioritize same-state branch probability or row-level preference shaping; no carrier conclusion. |
| A natural predecessor selectively unlocks or preserves the target | Authorize a bounded shared-prefix branch-value training proposal. |
| Only direct target-token or target-row forcing works | Target basin is accessible; endogenous trigger remains unresolved; do not claim native unlocking. |
| Bagged union is high, but no high-coverage executable path exists | Objective-only greedy distillation has a low ceiling; compare runtime search or state scaffolds without selecting their form. |
| A high-coverage native path exists, but greedy is much worse | Prioritize latent-route or branch-value distillation before architecture changes. |
| Same entity set with different histories has irreducibly different future distributions | A bare covered set is insufficient, while a richer implicit or explicit state remains possible. |
| Correct information-equivalent recoding beats shuffled and wrong recodings | Supports an interface or inductive-bias gap, not missing visual information. |

## Stop Rules

- Stop population generalization if fixed-prefix target support does not
  reproduce on at least three independently audited intermittent targets.
- Stop local unlocking surgery after two natural predecessor rows fail to
  selectively change target future recovery relative to all generic controls.
- Stop the branch if only oracle target forcing works; record accessible mode
  but unresolved endogenous trigger.
- Do not count success when one rare object merely replaces another and total
  unique-object utility is unchanged.
- Reject apparent recall gain explained by longer output, duplicates, invalid
  rows, or unsupported predictions.
- Do not enter representation analysis before the behavioral causal gate.
- Do not start a 256-image training screen from this unit. A separate decision
  must identify a trainable branch target and preservation-safe objective.

## Reused Surfaces and Minimal Implementation

Reuse:

- `src/analysis/sampled_rescue_transition/` for call bundles, geometry modes,
  prefix equality, and trajectory chronology;
- `scripts/research/run_sampled_rescue_transition.py` for bounded exact-prefix
  and forced-row execution;
- the current `src/vis/` normalization, matching, and review semantics;
- the existing post-merge unmatched-prediction queue and accepted reference
  ledger.

Add only:

- one experiment-local static HTML review builder;
- the artifact-only full-suffix and sibling-branch analysis needed by this
  unit after the human ledger freezes.

No OpenSpec change is needed because no stable runtime, schema, or compatibility
contract is being introduced.

## Artifact Handles

Research artifact root:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-07-16-human-audited-rare-object-trajectory-genealogy/<immutable-run-id>/
```

Initial human-review artifact identifier:

```text
human-review-candidate-ledger-20260716a
```

Materialized static reviewer:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-07-16-human-audited-rare-object-trajectory-genealogy/
human-review-candidate-ledger-20260716a/unmatched-prediction-review.html
```

The reviewer embeds five images, 119 `FULL_BAG_K` unmatched candidates, and
81 accepted-ledger boxes. Its frozen source digests are:

```text
manifest:
ac188ea1c09526660a86710fb8c99f632e99cdc53b5816cc870cf67c81dd0dc7

unmatched queue:
0bdfe3a1cbbf31d0953dd71d64c4cc08d1ae036d793de423ce1a12e9ade17e78

accepted ledger:
52e9f21eb32f7c3793d1356125931cb4c2a1fa5a12647d0d437dec217668d8df
```

The initial reviewer consumes:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-07-13-spatial-scope-history-disentanglement/review/
mask-reset-vs-full-bag-k-root-2026071301/manifest.json

/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-07-13-spatial-scope-history-disentanglement/review/
mask-reset-vs-full-bag-k-root-2026071301/
unmatched-prediction-review-queue.jsonl

/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-07-13-spatial-scope-history-disentanglement/readiness-v2/
audit-augmented-ledger.jsonl
```

## Representative Smoke and Rough Cost

The first smoke uses one synthetic two-image fixture, including one image with
zero unmatched candidates, followed by generation of the five-image static
review page. No model or graphics-processing-unit execution is required.

The annotation workload is 119 candidates. Wave 0 is artifact-only. Later
inference remains bounded to the smallest exact-prefix sibling panel justified
by the frozen human ledger.
