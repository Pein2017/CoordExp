---
title: Sampled-History Target Reachability and Complete-Row Causal Value
description: A frozen-target experiment that separates delayed greedy discovery from sampling-dependent access, then tests whether one natural sampled row safely makes a later physical object greedily reachable.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
training_promotion_status: not_promoted
implementation_status: complete
unit_id: 2026-07-19-sampled-history-target-reachability-and-complete-row-value
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: stage_7_complete
updated: 2026-07-19
---

# Sampled-History Target Reachability and Complete-Row Causal Value

## Status and decision boundary

This unit follows the closed [Local Branch Causality and Downstream
Unique-Object Value](../2026-07-19-local-branch-causality-and-downstream-coverage-value/results.md)
unit. That unit found no usable same-prefix covered-duplicate versus
sampled-uncovered cohort. It did find a stronger pattern: a physical owner that
is absent from the first eight root-greedy rows is often easy for greedy
decoding after a sampled history reaches the corresponding prefix.

The present unit asks where that accessibility comes from. It does not assume
that sampling found a better trajectory, that one row created a covered set,
or that an owner absent from eight greedy rows is absent from a longer greedy
rollout.

The first stage is mandatory because every earlier root trajectory hit its
eight-row cap. No causal branch arm may run until the extended greedy screen
classifies the frozen targets.

This unit is now complete. Three same-parent complete-row interventions prove
that natural same-owner coordinate history can redirect later physical-owner
access. Only image `7816` yields a safe final unique-owner gain; images `12576`
and `18380` exchange routes. Follow-up controls show one image-local `x1` or
`y1` route switch and one separate image requiring a joint two-row history.
The promotion threshold is not met. No training objective, object ledger,
architecture change, or further arm inside this unit is authorized. The final
evidence and claim boundary are recorded in [results.md](results.md).

## Main question

For a verified physical object that appears in a sampled trajectory but not in
the first eight rows of the paired root-greedy trajectory:

1. Does a longer root-greedy trajectory find it later?
2. If not, at which exact sampled-prefix state does it first become reachable
   by greedy decoding within the original absolute eight-row horizon?
3. At an adjacent unreachable-to-reachable transition, is appending the
   natural sampled complete row rather than the native greedy complete row
   sufficient to retrieve the target later and improve safe unique-object
   coverage?

The target must occur after the intervened row. Appending the target row itself
is excluded because it would make target retrieval trivial.

## Frozen model and numerical contract

Use only the description-first, geometry-sorted,
pure-cross-entropy-plus-token-type-gate Weight-Decomposed Low-Rank Adaptation
checkpoint at step `4,887`:

```text
/data/CoordExp/.worktrees/CoordExp-swift/outputs/prod/coordexp_swift/
qwen3_vl_2b_desc_first_geo_sorted_pure_ce_typegate_dora_r16a32_llm_12000_accelerate8_ebs24_8epoch_warmup0p1/
checkpoints/step-4887/checkpoint.json
```

Every primary arm uses:

- Hugging Face generation;
- all model parameters in 32-bit floating point;
- physical batch size one;
- greedy decoding;
- repetition penalty `1.0`;
- no top-p filtering, terminal suppression, grammar change, crop, resize, or
  external detector; and
- exact source image bytes, prompt, tokenizer, and raw prefix token identifiers.

Existing validation-200 artifacts cannot replace Stage 1 because they use
physical batch size four and repetition penalty `1.1`.

## Frozen provisional targets

[provisional-targets.json](provisional-targets.json) freezes target selection
before any extended-greedy result is observed. It uses only the immutable
eight-row discovery artifacts from the preceding unit.

At most one target is selected per image by:

1. verified physical owner;
2. absent from the root-greedy owner set within eight rows;
3. present in a natural sampled trajectory;
4. first sampled occurrence after row zero, so earlier sampled history exists;
5. lowest first-occurrence row, then lowest sampled seed, then owner identifier.

Image `19109` owner `1680320` occurs at sampled row zero and is retained only
as a root-decision diagnostic. Images `9400` and `9590` contain no eligible
eight-row sampled-only owner and are negative discovery controls.

## Stage 1: extended root-greedy screen

For all eight frozen images, extend the exact root-greedy trajectory from the
empty assistant prefix until either:

- the model naturally emits the terminal action; or
- the trajectory consumes `512` generated token identifiers.

Malformed output is recorded and does not silently become a terminal action.
Every target receives exactly one label:

- `terminally_omitted`: greedy terminates before emitting the target;
- `delayed`: greedy emits the target after row eight;
- `right_censored`: the 512-token cap is reached without target or natural
  terminal;
- `unresolved`: geometry or physical ownership prevents a reliable target
  decision.

Only `terminally_omitted` supports a final-recall interpretation. `Delayed`
targets may still support a route-efficiency study. `Right_censored` targets
may support a bounded prefix-mechanism study but cannot establish final recall.

This stage may remove targets but may not add images, owners, seeds, or
selection rules.

## Stage 2: prefix sufficiency ladder

The post-Stage-1 cohort and allowed interpretation are frozen in
[stage2-admission.json](stage2-admission.json). Diagnostic cases may be run for
comparison but cannot enter the primary target-access denominator.

Let `P_k` be the exact sampled prefix after the first `k` complete sampled
rows. Let the frozen target first occur in the sampled trajectory at row `j`.
For every `k` from zero through `j`, start from exact `P_k` and greedily
generate until the complete trajectory reaches the absolute eight-row horizon.

Assign one reachability state to every tested prefix:

- `hit`: a complete, single-owner, phrase-and-geometry-consistent target row
  appears in the greedy continuation;
- `clean_miss`: the continuation ends at the absolute eight-row horizon or a
  natural terminal, all target-relevant rows are interpretable, and the target
  does not appear. Unrelated unmatched rows remain explicit crop-review
  warnings but do not automatically erase the target-specific judgment;
- `right_censored`: the available generation budget ends before the requested
  horizon or a natural terminal; and
- `unresolved`: malformed output, mixed ownership, or ambiguous geometry
  prevents a trustworthy hit-or-miss judgment.

The four-valued state is required because absence from strict owner matches is
not negative evidence when the continuation itself is ambiguous.
When the token budget ends before the requested horizon or a natural terminal,
`right_censored` takes precedence over an incomplete final row; the incomplete
row evidence is still retained.

Also record:

- target first-hit row;
- strict verified unique owners;
- duplicate owners;
- malformed rows;
- unresolved entity or geometry;
- phrase-and-geometry owner disagreement; and
- natural terminal position.

Interpretation is deliberately narrow:

- root prefix `clean_miss` and sampled target prefix `hit`: accumulated sampled
  history is sufficient for greedy target retrieval;
- sampled target prefix `clean_miss`: even the sampled target prefix does not
  make the target greedy;
  its natural occurrence remains a same-prefix stochastic branch;
- a non-monotonic reachability sequence: route state is unstable, and there is no single
  monotonic memory-acquisition point.

The smallest `k` whose reachability state is `hit` is called the earliest
tested sufficient prefix. It is not called a causal onset because each prefix
contains a different accumulated history.

## Stage 3: same-parent complete-row intervention

Only adjacent transitions with:

```text
clean_miss at P_k and hit at P_(k+1)
```

enter this stage.

The exact post-Stage-2 candidate set and permitted claims are frozen in
[stage3-admission.json](stage3-admission.json). The three admitted sampled
rows and their native greedy counterparts name the same physical object and
have the same nine-token row length. They differ only in several coordinate
tokens. Stage 3 therefore tests same-owner coordinate-history sensitivity; it
cannot by itself establish a covered-set carrier or a different-object commit
effect.

At exact parent `P_k`, let:

- `S_k` be the natural sampled complete row that produced `P_(k+1)`; and
- `G_k` be the native greedy complete row replayed from the same `P_k`.

Run these exposure-matched arms:

| Arm | Execution | Purpose |
|---|---|---|
| Direct native greedy | Greedily complete from `P_k` to the absolute eight-row horizon. | Native reference. |
| Greedy-row append no-op | Append exact raw row `G_k`, then greedily complete to the same horizon. | Must exactly reproduce the direct native row and suffix. |
| Sampled-row intervention | Append exact raw row `S_k`, then greedily complete to the same horizon. | Tests whether this complete row causes later target access. |

The target must not be the owner of `S_k`; it must appear in a later suffix
row. The two branch rows must be complete and have one resolved physical owner.
They must also add exactly the same number of raw token identifiers. Exact
no-op raw-token parity is mandatory. A parity or length mismatch refuses the
primary causal case.

This is the primary causal contrast:

```text
same image and same exact parent prefix
  + sampled complete row
versus
same image and same exact parent prefix
  + native greedy complete row
```

It changes one natural complete row while holding the earlier state fixed.

## Primary outcomes

### Later target retrieval

The frozen target must appear after the branch row in the sampled-row arm and
not in the matched greedy-row arm.

### Safe unique-object change

Count strict verified physical owners in the complete trajectory through the
absolute eight-row horizon. Report:

```text
unique-owner difference
  = sampled-row-arm unique owners
    minus greedy-row-arm unique owners
```

Unresolved outputs are reported separately and cannot be converted into
negative evidence from missing Common Objects in Context annotations.

### Other-owner preservation

Exclude the two branch owners and the frozen target, then compare the remaining
strict verified owner sets. A target rescue that loses another verified owner
is a route swap, not a unique-coverage gain.

### Safety

Record definite duplicates, malformed rows, definite unsupported entities,
phrase-and-geometry owner disagreement, and instance-binding error. Entity
existence/category and geometry are always judged separately.

## Same-owner geometry control

When `S_k` and `G_k` refer to the same physical owner but contain different raw
coordinate tokens, retain that pair as a strong diagnostic control.

If same-owner geometry variation is sufficient to unlock the later target,
the evidence favors spatial route perturbation carried by row geometry over a
simple emitted-owner commit update.

## Long-horizon confirmation

Any positive eight-row pair is repeated under the same two complete-row arms
until natural terminal or the fixed 512-token total budget.

- If both arms converge to the same final verified owner set, the effect is
  acceleration or reordering, not final recall improvement.
- If the sampled-row arm retains a larger safe final set, it supports a real
  detection-capability improvement.
- If either arm reaches the cap without terminal, final recall remains
  right-censored.

## Promotion and stop rules

1. Require at least four fully interpretable image-level targets after Stage 1
   and physical-owner checks for a general training proposal.
2. Require a positive, safe complete-row causal value on at least
   `ceil(0.75 × N)` images, never fewer than three.
3. A severe resolved reversal blocks a universal complete-row preference
   objective. A severe reversal is loss of a verified unique owner or addition
   of one definite unsupported, malformed, duplicate, or binding failure.
4. If fewer than four targets survive, complete a bounded case study only; do
   not expand the image pool, seeds, or owner rules.
5. If most sampled target prefixes remain `clean_miss`, stop the
   sampled-history route and return to direct same-prefix stochastic target
   choice.
6. If all positive effects disappear at the 512-token horizon, classify them
   as route acceleration and do not claim recall repair.
7. If same-owner geometry controls reproduce the effect, do not call it
   covered-set commitment.
8. Token-level intervention is forbidden until a complete-row pair has positive
   causal value. Full-row causality alone does not identify the responsible
   token.
9. Training remains blocked until a later fresh end-to-end greedy screen can
   improve unique-object coverage without safety loss.

## Training routes implied by possible outcomes

| Result | Allowed next training question |
|---|---|
| One complete sampled row yields safe final unique-owner gain. | Compare complete rows at the same natural own-prefix using downstream set value. |
| Complete-row gain exists, and a later token split reproduces it. | Test a local branch preference at the proven token boundary. |
| Several sampled rows must accumulate before target access. | Consider trajectory-level reward or a compact state treatment; do not pretend one row owns the effect. |
| Only order or speed changes and final sets converge. | No recall-treatment proposal from this route. |
| Same-owner coordinate changes reproduce target access. | Study row geometry as route state before any covered-set carrier. |

## Non-goals

This unit does not:

- estimate validation-set mean average precision or recall;
- treat sample union as one feasible trajectory;
- infer hallucination from annotation absence;
- claim that all prefix sensitivity is harmful;
- add an object slot, query bank, detector, visual cursor, explicit ledger, or
  terminal suppression;
- choose a reinforcement-learning method before causal value is established;
  or
- authorize an OpenSpec training implementation.

## Minimal implementation path

Reuse exact raw-prefix append, positive-only owner matching, full-float32 model
loading, and per-row generation from:

```text
scripts/research/run_local_branch_causal_value.py
```

Add only:

1. a greedy extension mode with a 512-token total budget;
2. deterministic freezing of the six provisional target traces;
3. greedy continuation from each exact sampled prefix; and
4. complete-row append-and-release for admitted `0→1` transitions.

Do not add a generic experiment framework, training code, hidden-state hook,
or token-level intervention in this unit.

## Verification before GPU execution

1. Parse and hash [provisional-targets.json](provisional-targets.json).
2. Confirm every target occurrence, owner, seed, row index, prefix hash, and
   source discovery digest.
3. Confirm that existing long-rollout artifacts cannot satisfy the frozen
   repetition-penalty and physical-batch contract.
4. Pass deterministic target-freeze tests.
5. Reuse the already passed full-model float32 token-append no-op smoke.
6. Run Stage 1 on all eight images before admitting any Stage 2 case.

## Executed follow-up amendments

Stages 1 through 3 followed the frozen protocol above. The following
follow-ups were admitted only after the preceding result opened a narrower
question.

### Stage 4: final-horizon confirmation

Only the safe positive image-`7816` pair was continued to natural terminal.
The sampled-coordinate arm ended with ten strict unique owners, including
target owner `211764`, with no verified loss or duplicate. The native arm ended
with nine strict unique owners and duplicate owner `205108`.

### Stage 5: row-0 by row-4 crossover

Image `7816` contained two earlier native-versus-sampled coordinate variants.
A four-arm crossover held the common suffix fixed. Row 4, not row 0, was
sufficient for the later `205108` versus `211764` owner switch in this image.

### Stage 6: row-4 coordinate factorial

The natural native or sampled values of `x1`, `y1`, and `y2` were crossed while
identical `x2` stayed fixed. Either sampled `x1` or sampled `y1` was sufficient
for the rescued route; sampled `y2` alone was not. Reverse-direction and
orthogonal-coordinate controls produced unmatched hybrid boxes rather than the
target. Candidate-row scoring confirmed a coordinate-stage change rather than
a terminal or rollout-length effect.

### Stage 7: image-`12576` two-row crossover

An independent route-exchange case crossed native or sampled row 3 with native
or sampled row 4 while holding later rows 5 and 6 fixed. Only the joint sampled
row-3 and sampled row-4 endpoint produced cup target owner `678023`; both
crossed prefixes produced native pizza owner `1571077`. This rejects a
universal latest-row carrier. Because the crossed prefixes are off-policy, the
result is classified as joint interaction or endpoint dependence, not formal
hidden-state mediation.

### Final stop

The preregistered multi-image safety gate fails: one safe final gain is below
the required minimum of three, and the other two causal cases are route
exchanges. No extra factorial arm, larger cohort, training screen, or OpenSpec
implementation is active from this unit.
