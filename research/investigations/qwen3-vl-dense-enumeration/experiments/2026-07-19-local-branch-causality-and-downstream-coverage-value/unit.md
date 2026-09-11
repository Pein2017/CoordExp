---
title: Local Branch Causality and Downstream Unique-Object Value
description: A frozen-cohort experiment that asks whether a naturally sampled, verified uncovered object can be made causally reachable from the same model prefix, and whether that local branch improves later unique-object coverage under greedy decoding.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
training_promotion_status: not_promoted
implementation_status: complete_for_unit
unit_id: 2026-07-19-local-branch-causality-and-downstream-coverage-value
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified_insufficient_cohort
updated: 2026-07-19
---

# Local Branch Causality and Downstream Unique-Object Value

## Status and decision boundary

Discovery and case freezing are complete. The preregistered primary phenotype
was not found on enough images: zero cases were admitted against a minimum of
four. The intervention arms therefore did not run, and no training objective
was selected. See [results.md](results.md).

The unit asks whether a low-probability but valid sampled object is only a
harmless alternate traversal route, or whether it identifies a concrete local
decision boundary where a small training correction could improve greedy
set enumeration. A positive answer is necessary, but not sufficient, before
opening a training implementation proposal.

The experiment is intentionally limited to the frozen eight-image discovery
pool in [discovery-pool.json](discovery-pool.json). Discovery must finish for
all eight images before any downstream suffix is evaluated. At most one
primary causal pair is selected per image, so the image-level denominator
cannot depend on execution order or on how many correlated prefixes one image
produces.

## Question

At an **exact natural own-prefix** of the primary model, suppose greedy
continuation emits a bad next row while sampling from that same prefix can emit
a verified, previously uncovered physical object. Does changing only the
**earliest candidate-distinguishing token** causally steer the current row toward
the sampled object, and does that branch safely improve unique-object coverage
in the next four rows?

An exact natural own-prefix is the original prompt and image followed by raw
token identifiers that the primary checkpoint generated in one of its own
rollouts. It is never reconstructed from annotations, another checkpoint, a
permuted history, or decoded text that is tokenized again.

The earliest candidate-distinguishing token is the first token at which the
natural greedy row and a natural sampled row differ after their common row
prefix. It is deliberately not called an "object token": it can be a phrase,
coordinate, structural, or other token. The experiment determines what role it
actually plays.

## Why this question is useful

Previous studies support two compatible facts:

1. Prefix route can change which object or geometry is generated next.
2. Some objects appear in sampled trajectories but not in the paired greedy
   trajectory.

Neither fact shows that a sampled branch is worth training toward. At the same
exact prefix, the greedy token necessarily has at least as much local greedy
preference as the sampled token, so a simple token-rank comparison would be
tautological. The useful tests are instead:

- **causal sufficiency:** whether the earliest differing token can redirect the
  current row when decoding is released immediately afterwards; and
- **downstream value:** whether the redirected route yields more safe, unique
  uncovered objects rather than merely a different order, a duplicate, a
  malformed row, or unsupported output.

## Primary checkpoint and numerical contract

The only primary model is the description-first, geometry-sorted,
pure-cross-entropy-plus-token-type-gate, Weight-Decomposed Low-Rank Adaptation
checkpoint at step 4,887:

```text
/data/CoordExp/.worktrees/coordexp-infras/outputs/prod/coordexp_swift/
qwen3_vl_2b_desc_first_geo_sorted_pure_ce_typegate_dora_r16a32_llm_12000_accelerate8_ebs24_8epoch_warmup0p1/
checkpoints/step-4887/checkpoint.json
```

Use Hugging Face inference in full 32-bit floating point, physical batch size
one, repetition penalty `1.0`, the checkpoint-native prompt and tokenizer, and
identical source image bytes for every arm. The Gaussian-smoothed-coordinate
checkpoint and historical random-order checkpoints are contextual evidence only;
they are excluded from this experiment.

The primary continuation is greedy decoding. A secondary, paired sampled
suffix uses temperature `0.4`, top-p threshold `0.95`, and the eight fixed
seeds `101` through `108`.
No terminal-token suppression, grammar change, coordinate smoothing, image
crop, image resize, or external detector is allowed.

## Frozen discovery pool

The cohort and its rationale are frozen in
[discovery-pool.json](discovery-pool.json): images `2299`, `7816`, `12576`,
`19432`, `9400`, `9590`, `18380`, and `19109`.

Image `2299` has a near-complete human relabel and is the preferred source of
object-versus-object cases. On every other image, official Common Objects in
Context (`COCO`) labels may
establish a positive match but do not prove that an unmatched prediction is a
hallucination. Crop-assisted review must separate entity existence, category,
geometry, duplicate ownership, instance-binding error, and unresolved evidence.

## Competing explanations

### Local branch-ranking explanation

The model can represent a real uncovered object at the parent prefix, but
native greedy decoding chooses a nearby habitual successor, duplicate, or
terminal route. A local preference objective at the proven branch boundary
could make the valid route win without changing the model architecture.

**Prediction:** forcing the earliest candidate-distinguishing token produces
the sampled object as the current row and, together with the complete sampled
row, improves later unique-object coverage without worsening safety.

### Whole-row state explanation

The sampled object is viable only after several coordinated phrase and geometry
tokens. One early token does not carry enough state, but an exact complete row
can place the model in a better own-prefix state.

**Prediction:** token-only forcing fails, while an exact complete sampled-row
append has positive downstream value.

### Within-row object-binding explanation

The model can continue and may name the intended category, but cannot bind the
phrase and all four boundaries to one physical object.

**Prediction:** token-only or complete-row interventions produce ambiguous,
wrong, or mixed geometry even when object-category evidence is present.

### Generic continuation or route-state explanation

Any row-like perturbation changes continuation, regardless of physical owner.

**Prediction:** a covered-object or irrelevant-route control produces the same
apparent benefit as the valid sampled branch.

### No training value explanation

The sampled route is valid but does not improve the later set, or its apparent
benefit is offset by duplicates, malformed rows, unsupported predictions, or
loss of other objects.

**Prediction:** causal steering is possible, but the total unique-object gain
and other-owner preservation do not improve.

## Assumptions and case admission

Discovery is separated from downstream evaluation. No case is admitted because
its later suffix looks favorable.

The fixed discovery budget is one native greedy root trajectory and eight
sampled root trajectories with seeds `11` through `18` for every image. Each
trajectory is generated for at most eight complete rows. The naturally emitted
sampled row at a state is the only rescue candidate from that state; no extra
same-prefix sampling is allowed during discovery. Exact prefixes are
deduplicated before native greedy replay.

For each frozen image, generate the fixed natural root trajectories from the
primary checkpoint. At every deduplicated exact natural prefix, replay the
native greedy next row and compare it only with naturally emitted sampled rows
already attached to that exact prefix.

A primary case requires all of the following before any force-and-release arm:

1. The prefix is exact and naturally generated by the primary checkpoint.
2. For a primary object-versus-object case, the native greedy next row is a
   verified covered duplicate and the sampled row is a verified uncovered
   physical owner of the same category and the same tokenized row length.
   Terminal, malformed, unsupported, benign-uncovered, and unresolved outcomes
   are separate diagnostic or refusal strata.
3. At least one same-prefix sampled row maps, under crop-assisted review, to
   one verified uncovered physical owner.
4. The natural greedy row, sampled row, common row prefix, and first
   candidate-distinguishing token are all resolved from raw token identifiers.
5. The covered-owner set, sampled-owner identity, and geometry are not
   ambiguous under the frozen review rule.

The primary promotion stratum is object-versus-object: a verified covered
duplicate versus a sampled, verified uncovered object. Terminal, malformed,
unsupported, benign-uncovered, and unresolved greedy outcomes are diagnostic
or refusal strata and are structurally excluded from the promotion
denominator.

A different verified uncovered object is a benign alternate route, not a
negative control. Unmatched predictions without sufficient visual evidence are
unresolved, not automatically hallucinations.

If multiple primary pairs survive on one image, select exactly one before any
suffix generation by this outcome-blind order: lowest natural row index, then
lowest discovery seed, then lexicographically smallest exact-prefix hash. Keep
all other pairs as non-independent diagnostics. Freeze the resulting case
manifest and its hash before running any force-and-release arm.

Primary prefixes may come from sampled root trajectories. Such a case tests
**conditional own-prefix learnability** only: it does not prove that native
greedy decoding can reach that prefix. Any later training screen must establish
fresh end-to-end greedy improvement. Cases on the native root-greedy trajectory
are reported as a separately stronger reachability stratum.

## Frozen owner and review contract

The selected validation JSONL supplies a positive-only entity ledger. Every
annotated Common Objects in Context 80-category object, including the
human-added negative annotation identifiers on image `2299`, is a verified
positive owner. Annotation absence is never negative evidence.

Automatic provisional ownership requires exact normalized category agreement,
intersection over union at least `0.50`, and a margin greater than `0.05` over
the second-best same-category owner. Every earlier prefix row must be one clean
complete row with exactly one unambiguous owner before its owner can enter the
covered set. Any unmatched, ambiguous, mixed, malformed, or multiply owned
earlier row makes that prefix an `unresolved_refusal` until crop review resolves
it.

Crop review records entity existence/category and geometry/physical ownership
separately. It may override the provisional geometry threshold only through an
explicit review record containing the image, enlarged crop, owner identifier,
reviewer, and reason. If plausible assignments can reverse a comparison, the
case is refused. Before interventions, write and hash the per-image positive
ledger, review overrides, and one selected primary pair.

The only executable outcome labels are:

- `verified_covered_duplicate`;
- `verified_uncovered_owner`;
- `verified_benign_uncovered_owner`;
- `verified_unsupported` after explicit crop review;
- `terminal_diagnostic`;
- `malformed_diagnostic`; and
- `unresolved_refusal`.

## Intervention method

The intervention is explicit token-prefix append, not a hidden in-call
"forced decoding" application programming interface. The runner prepares the
same multimodal model state for the image and exact parent prefix, appends the
declared raw token identifiers, then resumes ordinary generation. This preserves
byte-identical token history and avoids decode-and-retokenize drift.

For a branch pair, let `P` be the exact parent prefix, let `c` be their shared
row-token prefix, let `g` be the native greedy differing token, and let `s` be
the sampled differing token. Let `G` and `S` be the full natural greedy and
sampled rows, respectively.

## Exposure-matched arms

All arms use the same image, model, parent prefix, maximum row horizon, parser,
owner-review rule, and decoding parameters. The primary horizon contains the
branch row plus four released successor rows. One- and two-successor-row
horizons are secondary diagnostics.

| Arm | Intervention | Purpose | Required interpretation check |
|---|---|---|---|
| Native greedy continuation | Start at `P` and greedily generate the branch row plus four successor rows. | Baseline behavior. | Raw branch row and released suffix must parse or be reported as failure. |
| Greedy-token no-op control | Append `c` and `g`, then greedily release through the rest of the branch row and four successors. | Tests that token-prefix append itself does not change the native route. | Must reproduce exact raw branch-row and successor token identifiers, parser status, row count, and terminal position. |
| Sampled-token force and release | Append `c` and `s`, then greedily release through the rest of the current row and four successors. | Tests causal sufficiency of the earliest candidate-distinguishing token. | Current-row owner, geometry, and suffix are scored separately. |
| Greedy-row no-op control | Append exact full row `G`, then greedily generate four successors. | Tests that full-row append has no mechanical effect beyond the native greedy row. | Must reproduce exact successor token identifiers, parser status, row count, and terminal position. |
| Sampled-row append and release | Append exact full row `S`, then greedily generate four successors. | Tests downstream value of the complete sampled branch. | Must improve coverage without safety reversal to support a training route. |
| Covered-owner sibling control, when naturally available | Append a natural complete row emitted from the same exact prefix for an already covered physical owner, then release. | Tests whether any owner-like row creates generic continuation. | A generic benefit weakens the owner-specific explanation. The primary native greedy duplicate already supplies the required wrong-owner control. |
| Verified-uncovered sibling control, when naturally available | Append a different verified uncovered natural row, then release. | Measures benign route freedom rather than treating another valid object as a negative. | Different valid routes may remain equally acceptable. |

The first and fourth arms are separate no-op checks because a correct local
token intervention can still be invalidated by an append implementation that
changes the decode trajectory. Under deterministic full-float32, batch-one
greedy decoding, parity must be exact at raw token level. Any no-op mismatch
refuses the affected case; owner-level similarity is insufficient.

The complete-row pair is a distinct intervention that sets a downstream
mediator. It is not a conservative substitute for the token intervention. A
primary promotion case already requires same-category, equal-token-length rows,
which removes the simplest description-length and position confound. If only
the complete-row pair succeeds, the result routes only to whole-row own-prefix
training.

The secondary stochastic sensitivity check applies to the two complete-row
arms. For each image, release four sampled successor rows from both full-row
prefixes with paired seeds `101` through `108`. Seeds are repeated measurements
within one image, never independent cases. A stochastic result that reverses
the deterministic direction holds promotion.

## Outcomes

### Current-row causal steering

For the token force-and-release arm, classify the current row as:

- the sampled physical owner;
- the native greedy physical owner;
- another verified uncovered physical owner;
- a covered duplicate;
- terminal;
- malformed;
- unresolved; or
- unsupported after crop review.

Phrase/category evidence and geometry/owner evidence are recorded separately.
A correct category with mixed or implausible boundaries is not a clean sampled
owner recovery.

### Total unique-object gain

For each arm, count unique verified physical owners emitted in the branch row
and its four released successors that were not already covered in `P`.

```text
total unique-object gain = unique verified owners after the parent prefix
                           minus owners already covered in the parent prefix
```

The comparison of interest is sampled-token versus greedy-token no-op, and
sampled-row append versus greedy-row no-op. The complete-row comparison
estimates branch-conditioned value after setting the complete row. It is a
different intervention, not a conservative substitute for token-only release.

### Other-owner preservation

Count unique verified physical owners only in the four successor rows,
excluding:

1. every owner already covered in `P`;
2. the native greedy branch owner; and
3. the sampled branch owner; and
4. the actual current-row owner produced by that arm.

This prevents an apparent win that merely swaps the two branch owners while
losing all other future discovery. It is reported alongside, not substituted
for, total unique-object gain.

### Safety and failure measures

Record row-level duplicate ownership, terminal output, malformed syntax,
unsupported entity, unresolved review, phrase/geometry owner mismatch, and
instance-binding error. A reported count always includes the denominator and
the review source.

## Expected outcomes and routing

| Observation after valid no-op controls | Interpretation | Next route |
|---|---|---|
| The sampled-token arm reaches the sampled owner and safely improves total unique-object gain and other-owner preservation. | A local branch boundary is both causally usable and valuable. | Propose a small own-prefix local branch-ranking training screen. |
| The sampled-token arm fails, but sampled-row append safely improves downstream coverage. | The useful signal is distributed across a complete row or its state transition. | Test own-prefix complete-row continuation training before local token ranking. |
| Current rows are category-plausible but phrase/geometry ownership is unstable. | The limitation is within-row object-to-box binding. | Design a within-row binding treatment, not a prefix preference loss. |
| Covered-owner or generic row controls reproduce the apparent benefit. | The effect is generic continuation or route-state change, not valid-object preference. | Close this local branch-ranking route and revisit state/visual interventions. |
| A valid sampled branch has no coverage value or worsens safety. | Sampling shows route diversity, not a useful greedy correction. | Close the preference-training route for this evidence scope. |
| Fewer than four interpretable images are admitted. | The cohort cannot support the planned causal generalization. | Record insufficient cohort; do not train from the cases. |

## Promotion and stop rules

A local branch-ranking training proposal is allowed only when all conditions
hold:

1. At least four interpretable images contribute primary object-versus-object
   cases.
2. At least `ceil(0.75 × N)` images pass the two-part gate, where `N` is the
   number of interpretable images; the required passing count is never fewer
   than three.
3. In every passing image, sampled-token versus greedy-token no-op has at least
   one more verified total unique owner, non-lower other-owner preservation,
   and no new definite unsupported, malformed, duplicate, or phrase/geometry
   binding failure.
4. Sampled-row versus greedy-row no-op has the same coverage direction and no
   safety reversal. The native greedy covered duplicate supplies a matched
   wrong-owner control that must not show the rescue benefit.
5. No-op controls have exact raw-token parity, unresolved assignments cannot
   reverse the sign, and the eight paired stochastic suffixes do not reverse
   the deterministic conclusion.
6. No fully resolved image shows a severe reversal. Either losing at least one
   verified total owner **or** adding one definite unsupported, malformed,
   duplicate, or binding failure is a severe reversal and vetoes promotion.

Complete the fixed discovery budget on all eight frozen images. Then freeze at
most one primary pair per image and all diagnostic strata before any suffix
generation. Do not expand the image pool, retrospectively alter owner rules,
add sampling seeds, or search for favorable suffixes. If fewer than four
images remain interpretable, record `insufficient_cohort`. If the promotion
threshold is unmet after all selected primary cases run, close or narrow the
route according to the routing table rather than adding cases or a larger
architecture.

## Artifact and receipt contract

Logical output root:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-07-19-local-branch-causality-and-downstream-coverage-value/<immutable-run-identifier>/
```

Each run receipt must contain the source commit or dirty-diff identity,
checkpoint and configuration identity, source image identifier, exact parent
prefix token hash, arm, token additions, seed, raw generated token identifiers,
raw text, parser status, owner-review status, and horizon. It must also record
whether the case was discovered, admitted, rejected, or blocked and why.

## Non-goals

This unit does not:

- estimate dataset-wide detection accuracy;
- claim that all prefix dependence is harmful;
- prove that the model has or lacks an explicit covered-set memory;
- modify the model architecture, visual features, attention pattern, or
  coordinate representation;
- use a detector, a powerful external model, terminal-token suppression, or
  bagging aggregation as a treatment; or
- authorize an OpenSpec change before the causal and value gates pass.

## Reused implementation surfaces and smoke

The first consumer remains experiment-local. Reuse the exact-prefix row
materialization, token append, row parser, and bounded horizon generation
surfaces from:

```text
scripts/research/run_same_covered_set_prefix_order_probe.py
```

The only new capability is a small force-and-release wrapper that can append a
declared token or full row, then call ordinary greedy generation. It must first
pass one full-model float32 smoke on one frozen image and validate both no-op
controls before the eight-image discovery panel begins.

## Known confounds and protections

| Confound | Protection |
|---|---|
| Official COCO annotations omit real objects. | Treat unmatched output as unresolved until crop-assisted review; do not call it hallucination by default. |
| A sampled row is only a geometry perturbation of a covered owner. | Require physical-owner review and separate phrase/category from geometry. |
| Forcing an appended token changes mechanics rather than model state. | Require greedy-token and greedy-row no-op controls. |
| Sampling gains come from more compute rather than a useful branch. | Compare same-parent arms at the same horizon; discovery is frozen before suffix scoring. |
| Fixed or reconstructed prefixes do not reflect inference. | Admit exact natural own-prefix states only. |
| A different valid next object is mistaken for failure. | Record it as a benign alternate route, never as a negative control. |
| The terminal row or malformed output has no physical owner. | Keep terminal, malformed, and unresolved outcomes in separate diagnostic strata. |
| Image resize, crop, or decoding heuristics change visual evidence. | Keep original image bytes and use repetition penalty 1.0 with no visual intervention. |

## Verification before execution

1. Parse [discovery-pool.json](discovery-pool.json).
2. Resolve its local link from this file.
3. Confirm the eight frozen image identifiers and primary checkpoint identity.
4. Run the one-image full-model float32 no-op smoke.
5. Record any failure that changes the intervention semantics before launching
the cohort.
