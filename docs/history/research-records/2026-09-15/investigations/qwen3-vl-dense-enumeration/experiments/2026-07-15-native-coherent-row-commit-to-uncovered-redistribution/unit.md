---
title: Native Coherent-Row Commit-to-Uncovered Redistribution Factorial
description: Exact own-rollout test of whether one coherent object row suppresses that physical object and redistributes the immediate next action toward still-uncovered objects.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: complete_for_unit
unit_id: 2026-07-15-native-coherent-row-commit-to-uncovered-redistribution
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
conclusion_status: narrowed_after_failed_execution_invariance_gate
updated: 2026-07-15
---

# Native Coherent-Row Commit-to-Uncovered Redistribution Factorial

## Question

At one exact native duplicate-onset state, does writing a correct object row:

1. selectively reduce the probability of emitting that same physical object
   again; and
2. transfer the immediate next-action probability to visible objects that have
   not yet been emitted?

The experiment distinguishes useful **commit-to-uncovered redistribution**
from lexical repetition effects, geometry-sorted frontier movement, generic
row advancement, and simple anti-repeat behavior.

This unit does not assume that a ledger, object slot, or new architecture is
required. It first asks whether the unmodified decoder already performs the
required state transition at a real failure onset.

## Candidate Qualification

The sole case is Common Objects in Context validation image `7574`.

- The target physical object is the standalone white bowl on the upper-left
  cabinet, Common Objects in Context annotation `1535235`.
- In canonical sampling cell `3`, the model first emitted the coherent row
  `bowl [187, 125, 274, 162]` and immediately emitted the same physical bowl
  again as `bowl [195, 131, 270, 162]`.
- Direct image review confirms that the two boxes belong to the same bowl, not
  to two same-class instances, a nested object, or an unlabeled container.
- The first row has Intersection over Union `0.6942` with annotation `1535235`;
  the repeated row has Intersection over Union `0.7214` with the same
  annotation. Intersection over Union is the intersection area divided by the
  union area of two boxes.
- Clearly visible, not-yet-emitted objects remain after the first bowl row,
  including a microwave, refrigerator, oven, bottles, wine glasses, and sink.
- The source run used repetition penalty `1.0`, temperature `0.4`, and top-p
  nucleus threshold `0.95`.

The exact source bundle is:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-07-13-spatial-scope-history-disentanglement/executions/
  dense-union-51-primary-after-wave-local-tail-contract/artifacts/calls/
  6ea680bc021555440810a7eb43dfec3903993f0532d8c6d8d17daffc326e4b93/
  terminal-output-bundle.json
```

The causal recipient state is the exact original prompt token sequence with
zero generated-prefix tokens. No earlier generated-row error is present.

## Object Factors

The **target bowl row** is source row index `0`, generated-token span `[0, 10)`.
It contains:

- two model-native lexical description tokens for `bowl`;
- the four model-native coordinate tokens from the first bowl box; and
- the standard object-reference and box wrappers.

The **other bottle row** is source row index `6`, generated-token span
`[60, 70)`. It refers to the clear or gray bottle on the right-side counter,
Common Objects in Context annotation `90913`. The distinct blue bottle farther
right is annotation `88785`. The source row also contains two lexical
description tokens and four coordinate tokens.

The other bottle is deliberately spatially later under the trained
geometry-sorted ordering policy. It is therefore a strong control for a
geometry-frontier explanation, not an assumption that every row should leave
the target bowl equally available.

## Five Conditions

The four appended-row conditions form a two-by-two description-by-geometry
factorial. Each appended row contains exactly ten tokens and occupies identical
prefix positions.

| Condition | Description tokens | Geometry tokens | Operational meaning |
|---|---|---|---|
| No Appended Row Baseline | none | none | Reuses the exact prompt-only state. It is intentionally outside the equal-length factorial. |
| Target Bowl Description with Target Bowl Geometry | target bowl | target bowl | Exact self-generated coherent row; primary factual commit treatment. |
| Other Bottle Description with Other Bottle Geometry | other bottle | other bottle | Exact self-generated coherent other-object row; controls generic coherent-row progression and later geometry frontier. |
| Target Bowl Description with Other Bottle Geometry | target bowl | other bottle | Crossed row testing target lexical history without target geometry. |
| Other Bottle Description with Target Bowl Geometry | other bottle | target bowl | Crossed row testing target geometry without target lexical history. |

No ground-truth token is injected. Every lexical and coordinate factor comes
from the same model-native trajectory.

## Competing Hypotheses and Predictions

### Hypothesis 1: Native object-specific commit-to-uncovered redistribution

The coherent target row creates an executable state transition for the target
physical object.

Predictions:

- the no-appended-row baseline selects the target bowl frequently;
- the coherent target row sharply reduces the target-bowl repeat count;
- probability moves to manually verified not-yet-emitted objects rather than
  to termination, invalid rows, unsupported objects, or another target-bowl
  fragment;
- the crossed row with target geometry but other description does not reproduce
  the full coherent-target effect.

Falsification:

- the coherent target row retains the target-bowl basin; or
- the target bowl decreases but the probability does not move to supported
  uncovered objects.

### Hypothesis 2: Description-history or category inhibition

The lexical history of `bowl`, rather than a physical-object commit, suppresses
later bowl outputs.

Prediction:

- both conditions containing the target-bowl description suppress bowl outputs
  similarly despite different geometry.

### Hypothesis 3: Geometry-sorted frontier movement

The coordinate span is the active traversal carrier.

Prediction:

- conditions sharing target-bowl geometry behave similarly;
- conditions sharing later bottle geometry behave similarly;
- description coherence adds little selective effect.

### Hypothesis 4: Generic complete-row advancement

Any grammar-valid ten-token row changes a shared progression state.

Prediction:

- all four equal-length appended rows converge on approximately the same first
  action and suppress the target bowl similarly.

### Hypothesis 5: Anti-repeat without useful redistribution

The target row inhibits a repeated token, phrase, or region, but does not make
another supported object more likely.

Prediction:

- the target-bowl repeat count falls while terminal, invalid, unsupported, or
  other duplicate mass rises.

### Hypothesis 6: No detectable native commit

The exact coherent target row does not selectively suppress its physical
object.

Prediction:

- the target bowl remains a dominant immediate next action under the coherent
  target condition, with no selective uncovered-object gain.

## Primary Observations

For each first complete free action, record the raw phrase and box before
assigning orthogonal attributes:

- predicted physical-object identity, including the target bowl, the other
  bottle, another named visible object, or unresolved identity;
- visually supported, visually unsupported, or uncertain support;
- valid complete row, immediate terminal action, or invalid or incomplete row;
- before, near, or after the geometry-sorted position implied by the appended
  geometry; and
- for each factual coherent-row condition only, whether the predicted physical
  object was emitted by that condition's appended row.

`Repeat` and `uncovered` are derived relative to each factual coherent-row
condition. The target bowl is uncovered after the coherent other-bottle row,
and the other bottle is uncovered after the coherent target-bowl row. Crossed
rows have no declared committed physical-object owner, so they receive no
repeat or uncovered label.

The Common Objects in Context 80-class ontology is the closed class list used
by the active prompt. Direct image review, not annotation matching alone, owns
physical-object identity and visual support.

The primary quantities are:

- **target suppression**: no-appended-row target-bowl frequency minus coherent
  target-row target-bowl frequency;
- **useful redistribution gain**: coherent target-row uncovered-object
  frequency minus no-appended-row uncovered-object frequency;
- **same-target-geometry coherence contrast**: coherent target-row outcome
  minus the other-description-with-target-geometry outcome.

A lower target-bowl frequency without positive useful redistribution is not a
successful commit-to-uncovered result.

## Execution Contract

1. Reuse the existing model loader, image materialization, prompt builder,
   request-scoped sampled decoder, parser, and compact artifact writer.
2. Use the immutable no-appended-row sampled calls for canonical sampling cells
   `0` through `7` to qualify the case; all eight selected the target bowl as
   their first action.
3. Execute one fresh greedy request and exactly eight fresh paired sampled
   requests for every condition, including the no-appended-row baseline, using
   the same canonical cell `0` through `7` seed vector. The fresh baseline is a
   reproducibility control; the equal-length four-condition factorial owns the
   causal comparison.
4. Use temperature `0.4`, top-p nucleus threshold `0.95`, repetition penalty
   `1.0`, and physical inference batch size four.
5. All requests retain the source runtime's `512` maximum-new-token horizon
   because the verified request-scoped sampler is attested only for that policy.
   Interpretation is restricted to the first complete free action; later
   generated rows are ignored.
6. Store exact prompt identifiers, source-bundle hash, source row spans,
   description and geometry token vectors, composed-row hashes, paired seeds,
   generated first actions, parser outcomes, and runtime identity.

The fixed inference configuration is:

```text
/data/CoordExp/.worktrees/research-probes/configs/coordexp_infras/infer/
  qwen3_vl_2b_desc_first_geo_sorted_gaussian_rps_dora_r16a32_step4887_val200.yaml
```

The historical filename token `gaussian_rps` expands to Gaussian soft-target
coordinate supervision; it is not the name of this experiment.

The source data is:

```text
/data/CoordExp/.worktrees/coordexp-infras/outputs/coordexp_swift/infer/
  val200_inputs/coco_val200_len12000.rebased_images.coord.jsonl
```

## Trust Gates

Stop before interpretation if any of the following occurs:

- the fresh base prompt does not exactly equal the source prompt, or an
  appended prompt does not equal that base prompt plus its declared ten-token
  suffix;
- current model, tokenizer, adapter, embedding delta, source-image hash,
  installed runtime, or base generation-configuration identity does not match
  the source bundle;
- sampled requests do not retain source temperature `0.4`, top-p nucleus
  threshold `0.95`, repetition penalty `1.0`, maximum-new-token horizon `512`,
  and request-scoped sampler identity;
- a description or geometry factor is not extracted from the declared native
  source row;
- any factorial row is not a valid ten-token complete row;
- paired seeds differ across conditions;
- repetition penalty is not exactly `1.0`;
- the fresh target-row greedy replay cannot reproduce a target-bowl repeat;
- parser or runtime failure is asymmetric across conditions; or
- manual review no longer supports the single-object duplicate assignment.

## Decision Rules

- A target-row condition with at most `2/8` target-bowl repeats and at least
  `6/8` verified objects uncovered relative to the coherent target-bowl row is
  a large suppression-plus-redistribution effect for this case only if it also
  differs from both crossed controls that share its description or geometry.
- At least `5/8` paired seeds must change selectively from target-bowl baseline
  to a verified uncovered object before calling the result a paired rescue.
- If outcomes group by description, close with a lexical-history result.
- If outcomes group by geometry, close with a geometry-frontier result.
- If all equal-length conditions converge, close with a generic-progression
  result.
- If target repeats fall without uncovered-object gain, close with anti-repeat
  without useful redistribution.
- If the coherent target row retains the duplicate basin, close with no
  detectable native commit at this state.
- Mixed low-margin outcomes are inconclusive. Do not add temperatures, geometry
  jitter, long rollouts, hidden-state probes, or post-hoc conditions in this
  unit.

The strongest conclusion available from this event is that a native coherent
target row does or does not cause selective successor redistribution at this
exact state. Because the other factual row is spatially later and the crossed
rows are semantically incoherent, this factorial cannot by itself uniquely
identify a physical-object ledger or commit mechanism. One event cannot
authorize architecture or training. A physical-object commit claim or a
conclusion that would change the training direction requires one independently
prequalified event.

## Artifact Handle

The logical artifact root is:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-07-15-native-coherent-row-commit-to-uncovered-redistribution/
  <immutable-run-identifier>/
```

## Non-Goals

- no training, final architecture, object slots, explicit ledger, reward
  optimization, OpenSpec change, or population metric;
- no claim that all low recall or duplicate bursts arise from missing commit;
- no ground-truth row forcing;
- no reinterpretation of unsupported predictions from annotations alone;
- no expansion beyond the first free action and one qualified image.

## Closeout

Execution is complete. The batch-four factorial produced a perfect
phrase-geometry coherence interaction, but the coherent target-row successor
failed the unit's execution-invariance gate: the same exact prompt repeated the
white target bowl at physical batch size one and advanced to an orange bowl at
physical batch size four, independently of whether the maximum generation
horizon was `16` or `512` tokens.

The durable conclusion is therefore narrowed to geometry-conditioned visual
object repair plus batch-sensitive same-class coordinate selection. Stable
native physical-object commit, an order-free ledger, architecture promotion,
and training promotion remain unsupported. See the [executed results and
bounded verdict](results.md).
