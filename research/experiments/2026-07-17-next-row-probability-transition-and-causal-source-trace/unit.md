---
title: Next-Row Likelihood Change and Causal Source Trace
description: Object-resolved study of how one natural complete row changes exact candidate-row likelihoods and where any object-specific change is computed.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: complete_for_unit
unit_id: 2026-07-17-next-row-probability-transition-and-causal-source-trace
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
conclusion_status: stopped_at_natural_sibling_admission_gate
updated: 2026-07-17
---

# Next-Row Likelihood Change and Causal Source Trace

## Status and authority

This unit records the user-approved scientific contract and its completed
bounded execution. The [results](results.md) own the executed evidence and
interpretation. The natural equal-depth sibling discriminator did not pass its
admission gate, so layer and source tracing were not opened. Training remains
closed. Architecture promotion and OpenSpec work remain unauthorized.

The earlier [random-versus-geometry-sorted common-prompt
draft](../2026-07-17-random-versus-geometry-sorted-common-prompt-prefix-comparison/unit.md)
is superseded. Its common-prompt comparison remains a control here rather than
the main question.

## Question

For the same image and the same already emitted object set, how does appending
one naturally supported complete object row change the exact-row likelihood of:

1. the object just emitted;
2. other verified not-yet-emitted objects;
3. previously emitted objects;
4. the separate row-entry-versus-terminal margin; and
5. unsupported or wrong-object controls?

When that change is object-specific, at which generation phase, language layer,
and visual-or-prefix source is it causally computed?

The unit does not ask merely whether random-order and geometry-sorted training
produce different outputs. It asks what state-transition rule each model
implements and how that rule is produced.

## Competing explanations

### Physical-object commit

Appending object row `r` suppresses natural row variants owned by `r` itself and
selectively increases natural row variants owned by other physical objects that
remain, largely independently of the textual order of the same emitted set.

### Geometry-sorted traversal frontier

Appending a row moves a one-way spatial frontier. Objects later in geometric
rank become easier, while an omitted earlier object may remain difficult even
though it is physically uncovered.

### Semantic complement or category anti-repetition

The next-row change follows class, phrase, or common co-occurrence rather than a
spatially indexed physical instance.

### Conflicting random-order supervision

Random-order hard cross-entropy supervision diffuses probability among several
valid objects without producing stable self-suppression, remaining-object
preservation, or order robustness.

### Prompt specialization or generic continuation

The main effect follows prompt wording, row count, or a broad continue-another-
row signal. Correct-object, wrong-object, and unsupported controls then change
similarly.

## Primary observation

For prefix `P` and one frozen natural complete-row variant `R_v(o)` owned by
physical object `o`, record the raw exact-row log-likelihood:

```text
L_P(R_v(o)) = sum_j log p(R_v,j(o) | image, P, R_v,<j(o))
```

The primary comparison is the paired before-and-after change caused by appending
one exact, contiguous, naturally emitted complete row `r`:

```text
Delta_v(r -> o) = L_(P+r)(R_v(o)) - L_P(R_v(o))
```

This is a **paired exact-row log-likelihood change**. It says whether the same
frozen path became easier or harder after one row was appended. It is not a
closed probability distribution and does not prove where probability mass went.
Rows index the appended natural object; columns index exact natural variants of
emitted, remaining, and control objects. Free rollout, attention maps,
hidden-state similarity, and aggregate detection metrics cannot replace this
paired primitive.

For every variant, report the raw sum, token count, per-token mean, and each
phase-local sum and mean. Raw paired changes own the sign conclusion. Per-token
means are a sensitivity analysis only. Do not tune a length-normalization
exponent on the cohort and do not compare different-length rows as if they were
mutually exhaustive outcomes.

When an owner has multiple frozen natural variants, report all variant-level
changes. An owner-level diagnostic may additionally use a declared log-sum-exp
over exactly those frozen variants, but it must be named a **frozen-variant owner
score**, not the probability of that physical object. A primary owner claim
requires either consistent signs across at least two natural variants or a
predeclared owner aggregation reported beside every constituent variant.

Claims about a next-row outcome distribution require repeated one-row sampling
from the exact same boundary and owner-level adjudication. The empirical
frequency table remains separate from teacher-forced path likelihoods.

Terminal behavior is also separate. At each exact row boundary, report the
single-token row-entry-versus-terminal log odds:

```text
M_continue(P) = log p(object_ref_start | image, P)
                - log p(im_end | image, P)
```

Report `M_continue(P)` and `M_continue(P+r)` independently. Never compare the
magnitude of this one-token margin directly with a full-row likelihood change.

The full-row score must also be decomposed into:

1. description tokens;
2. `x1`, the left bounding-box coordinate;
3. `y1`, the top bounding-box coordinate;
4. `x2`, the right bounding-box coordinate;
5. `y2`, the bottom bounding-box coordinate;
6. row closure; and
7. terminal-versus-continue behavior.

## Cohort contract

Use four to eight deliberately selected images. Primary mechanism claims are
restricted to physical owners that are manually resolved.

The cohort must include, if a qualifying image exists:

1. at least one image with three or more same-category, spatially non-overlapping
   objects;
2. one object recovered by low-temperature sampling but missed by paired greedy
   decoding;
3. one dense repeated-category scene; and
4. one previously studied state with a known natural row-conditioned successor
   effect.

The accepted object ledger is the union of:

1. trusted dataset annotations;
2. human-approved unmatched predictions with a physical entity reference; and
3. other manually confirmed Common Objects in Context 80-category objects.

Human-review `unknown` entries are excluded from conclusion-owning candidate
sets. A suspicious false positive may be accepted only after crop-level review
on the original image, using interpolation for inspection when useful. Any
remaining ambiguity is recorded for later user review and is not silently
resolved by nearest-box matching.

The first high-value target is a fully resolved three-owner, same-category,
non-overlapping scene. Nested container-content pairs cannot own the physical-
commit conclusion.

### Executed cohort status

The initial discovery map was refined during execution without relaxing the
physical-owner rule.

| Image | Current role | Admission status and reason |
|---|---|---|
| `12576` | Natural one-way traversal control; sampled pizza rescue | Not scored. The historical frozen boundary uses a prompt that differs from the active processor prompt, and the fail-fast prompt-identity check rejected it. No repair or splice was attempted. |
| `7574` | Natural trajectory-change control | Scored at row zero under configuration precision and full-model 32-bit floating point, using only the resolved bowl and microwave owners. The result supports own-owner suppression but rejects uniform redistribution. |
| `15254` | Clean row-zero numerical and semantic-complement control | Scored at row zero under both numerical modes. The nested bowl-carrot relation remains unsuitable for a physical-instance commit claim. |
| `2299` | Preferred three-or-more same-category physical-instance case | Admitted through a fresh exact native chain `P0 -> PA -> PAB` whose rows have unique person owners. Exact-row scoring completed. Repeated sampling from `PA` remained concentrated on person B, so the required reciprocal equal-depth sibling was not admitted. |
| `19432` | Dense-chair and greedy-stop-versus-sampled-rescue control | Not used in the conclusion-owning execution because no shorter clean parent was admitted. |

The frozen owner review for image `2299` is:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-07-17-native-sibling-row-branch-value-and-commit-crossover/
  analysis-image2299-p54-v1/blind-review-frozen.json
```

The verified image-`19432` sampled-rescue receipt is:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-07-14-sampled-rescue-object-transition-causal-replay/
  wave2-fixed-prefix-distribution/image-19432/
  rescue-entry-chair-383277/receipt.json
```

The conclusion-owning panel uses row-zero scores from images `7574` and `15254`
plus the fresh image-`2299` chain. Image `12576` remains a documented blocked
case, not missing data. Image `19432` remains historical context only. No row
was deleted, corrected, reordered, or spliced to manufacture a parent.

A valid parent is either the prompt-only row-zero state or a contiguous prefix
of one recorded native rollout. Its receipt must bind the source rollout, exact
row interval, prompt tokens, prefix token IDs, token hash, and byte-for-byte
reconstruction equality.

## Checkpoint roles

### Primary mechanism checkpoint

Use the current step-4,887 geometry-sorted Qwen3 Vision-Language adapter already
used by the recent mechanism units. It owns within-model transition, layer, and
source tracing.

The current inference configuration is:

```text
/data/CoordExp/.worktrees/research-probes/configs/coordexp_infras/infer/
  qwen3_vl_2b_desc_first_geo_sorted_gaussian_rps_dora_r16a32_step4887_val200.yaml
```

The historical path token `gaussian_rps` expands to Gaussian soft-target
coordinate cross-entropy plus a cumulative-distribution regularizer. It is
retained only for artifact provenance and is not the name of a promoted
mechanism.

## Historical no-repeat constraints

The historical experiment-knowledge handoff is provenance, not current
authority. It changes this unit in four concrete ways:

1. do not repeat raw boundary-vector patches that previously changed hidden
   readouts without establishing a physical-object state transition;
2. distinguish a direct hidden-state language-model-head readout, a final-
   normalization readout, actual final logits, and generated behavior;
3. require prefix-position accounting that includes the true forced or emitted
   tail, because an older route map was invalidated by ending at the wrong
   prefix position; and
4. treat coordinate-rank movement, target geometry, instance binding,
   continuation, and termination as separate outcomes.

The changed discriminator is therefore a natural complete-row before-and-after
change over exact candidate-row likelihood, followed by a bounded causal
trace only when that transition is object-specific. This is not another raw
boundary-patch or isolated coordinate-rank experiment.

### Historical comparison checkpoints

The checkpoint-3,668 random-order and geometry-sorted Low-Rank Adaptation pair
may be used to discover contrasting states and compare common-prompt behavior.
Their training changed both target-row order and prompt wording, used one seed,
and predates the current Weight-Decomposed Low-Rank Adaptation route. They cannot
support a causal claim about ordering.

No matched current random-order Weight-Decomposed Low-Rank Adaptation checkpoint
was located during contract drafting. The primary within-model study does not
wait for one. A matched future checkpoint may be added as a replication arm,
not retroactively treated as part of the first execution.

## Analysis order

### Execution outcome

Stage 1 completed. The strongest native image-`2299` transition is concentrated
in geometry, especially `x1`: after appending person B after person A, B's own
frozen row falls by `-6.2517` summed natural-log units in full-model 32-bit
floating point, while four frozen rows belonging to two other verified people
rise by `+1.3919` to `+2.4343`. All descriptions are `person`, and the
row-entry-versus-terminal margin remains above `11.0`, so category repetition
and a broad terminal switch are insufficient explanations.

The necessary equal-depth natural crossover did not become available. Across
160 independent samples from `PA`, person B occurred 153 times, person owner
`0006` occurred twice, person owner `0004` occurred twice, and three rows were
unmatched. The rare alternative owners failed the predeclared requirement of
three distinct exact natural variants. The sampling cap and stop rule were
therefore applied. Stages 2 through 5 and the training screen did not run.

### Stage 1: object-resolved transition matrix

At each selected natural parent state:

1. freeze all natural row variants and physical owners before scoring;
2. score every verified remaining candidate-row variant;
3. score one or more already emitted natural variants;
4. record row-entry-versus-terminal log odds separately;
5. use same-parent natural owner A versus natural owner B as the strongest
   crossover control, matching category, wrapper, and token length where
   possible;
6. retain duplicate-covered and irrelevant rows as descriptive controls, but
   do not let an off-support irrelevant row own the conclusion;
7. append each naturally supported sibling row in turn; and
8. recompute the same exact-row and terminal diagnostics.

Inspect the matrix before deeper hooks. Stop a case if the only effect is broad
continuation without object specificity.

The first artifact must expose, for every donor and scored row: physical owner,
natural-support source, exact token hash, token length, category, description,
geometry, and whether it is emitted, remaining, duplicate-covered, terminal, or
descriptive control. It must also expose pairwise donor crossover, not only each
arm versus the unmodified parent.

### Stage 2: prefix-state controls

This stage stayed closed because Stage 1 did not admit the equal-depth natural
sibling contrast needed to distinguish physical-object commit from a
geometry-sorted traversal frontier.

For the same emitted object set, compare:

1. geometry-sorted row order;
2. reverse order;
3. one fixed random order;
4. earlier rows reordered while the last row remains fixed; and
5. the last row changed while earlier rows remain fixed.

An order may own a native-state conclusion only when that exact contiguous
history occurred naturally. Reordered histories are otherwise labeled forced-
prefix robustness probes and cannot support a native-state claim. Compare
phrase, geometry, and coherent full-row changes without silently treating a
sequence assembled from individually natural rows as a natural history.

### Stage 3: tied coordinate read-versus-write role

The Qwen3 Vision-Language 2-billion-parameter model shares its input token
embeddings and output language-model-head weights. CoordExp also applies one
shared trainable selected-token delta on both sides.

For naturally supported coordinate alternatives with a common-support check:

1. measure the coordinate output-logit contribution;
2. hold the teacher-forced token identity fixed;
3. toggle only its next-step input delta contribution; and
4. measure later coordinates, row closure, and the next-row transition matrix.

This stage tests whether coordinate tokens merely report geometry or also write
task state into the following computation. It does not authorize untying the
base embedding and output head.

### Stage 4: layerwise readability and causal replacement

Only on two to four states with a stable object-specific transition:

1. read candidate margins after every language layer at the exact divergent
   generation phases using the real final normalization and tied output head;
2. treat those readouts as descriptive;
3. within the same checkpoint, replace natural recipient states with natural
   donor states at coarse layers and then narrow the range; and
4. require the final object-specific margin or row owner to follow the donor
   before calling a layer causal.

Do not patch hidden states across different checkpoints. Do not interpret a
one-token rescue as complete-row recovery.

If current-query replacement fails but a whole-prefix condition changes the
decision, test a bounded cached key/value source window rather than declaring
that no state exists.

### Stage 5: causal source trace

Only after Stage 4 identifies a causal phase and layer:

1. separate last-row, earlier-row, and image-token cached sources;
2. split attention output, multilayer-perceptron output, and residual addition;
3. treat raw attention weights as route suggestions, not causal evidence;
4. test primary merged image embeddings and the three DeepStack streams; and
5. enter multimodal merger or vision-tower blocks only if the downstream source
   test requires it.

Vision block indexes 5, 11, and 17 that source DeepStack are not language-layer
indexes 5, 11, and 17. Historical language-layer 17-through-23 effects must not
be conflated with those vision depths.

## Invariants and validity controls

- batch size one for conclusion-owning model calls;
- repetition penalty `1.0` for newly generated discovery trajectories;
- no repetition-penalty post-processing in raw teacher-forced scoring;
- identical image bytes, tokenizer, output wrapper, and prompt tokens within a
  comparison;
- identical candidate row tokens before score comparison;
- no hidden duplicate filtering before raw-output inspection;
- no arbitrary coordinate clamp without an admitted common-support history;
- no cross-checkpoint hidden-state replacement;
- full-model 32-bit floating point replay for a low-margin or numerically
  conclusion-changing effect;
- Brain Floating Point 16-bit discovery is allowed but cannot own a fragile
  token-level conclusion; and
- Scaled Dot-Product Attention versus FlashAttention 2 implementation remains
  explicit in every receipt.

The experiment-local runner must fail before scoring unless its receipt binds
the exact base model, adapter, selected-token embedding delta, prompt hash,
wrapper, tokenizer, image bytes, attention implementation, model dtype, score
accumulation dtype, cache policy, effective batch size, and generation policy.
Existing natural rows discovered under a different repetition penalty retain
that policy in provenance; they are not relabeled as `1.0` trajectories.

## Training gate

A 256-image training screen remains closed. Its three opening conditions were:

1. on-support natural rows produce stable, object-specific paired likelihood
   changes and the empirical one-row outcome table supports the same owner-level
   direction;
2. a phase, layer, or source is causally implicated and correct-object
   intervention outperforms wrong-object or generic controls; and
3. the candidate ledger safely distinguishes verified positives, emitted
   objects, uncertain objects, and unsupported predictions.

If the gate opens, compare exactly one mechanism-matched calibration with the
current training baseline.

### Conditional calibration choices

- If failure is row-to-row selective likelihood change, test a before-versus-
  after complete-row transition loss. This differs from current-boundary multiple-trie or
  object-marginal training because it supervises how adding an emitted row
  changes later full-row probabilities.
- If a correct privileged visual condition creates a causal object-specific
  late state that clean computation does not synthesize, use that condition as
  a training-only teacher. Match the verified object margin or a bounded causal
  projection, not the entire hidden state.
- If source tracing reaches a multimodal merger or DeepStack stream, permit only
  the smallest implicated visual-side trainable surface.
- If input-versus-output coordinate-delta toggling reveals a harmful role
  conflict, consider separating only the selected coordinate deltas while
  retaining the pretrained tied base weights.

An explicit covered-set carrier, object slot, detector query, or persistent
external memory remains unpromoted. Reconsider it only after a native transition
calibration fails despite reliable object support and complete labels.

## Stop rules

- If common prompts explain the historical checkpoint difference, classify it
  as prompt specialization and stop that comparison.
- If only broad continuation changes, stop before layer and visual tracing.
- If a candidate transition depends on unresolved ownership, remove the case
  from primary evidence.
- If a proposed donor/intervention lacks common support, do not inspect or
  interpret downstream outcomes.
- If a layer is readable but causal replacement does not change the final
  object-specific decision, do not design a loss around that layer.
- If coordinate input and output roles do not conflict, do not split their
  selected-token deltas.
- If a 256-image screen changes its teacher-forced objective but not the
  preregistered transition primitive, stop rather than scale.

## Reused surfaces and implementation boundary

The execution reused existing batch inference, native sibling-row fixtures,
human-review ownership, and runtime attestation. It added only two experiment-
local scorers: exact next-row likelihood change and the bounded prefix-row
factorial. The coordinate input-versus-output toggle, residual replacement, and
cached-source paths were not opened because the natural-sibling gate failed.

Do not start an OpenSpec change before a second real consumer establishes a
stable reusable interface.

## Artifact handle and rough cost

Logical root:

```text
outputs/research/qwen3-vl-dense-enumeration/
  2026-07-17-next-row-probability-transition-and-causal-source-trace/<run-id>/
```

The completed pass used three scored images, one prompt-mismatch blocked case,
and one held historical control. It used existing trajectories where possible
and capped additional natural-branch discovery at 128 samples. No broad
validation benchmark, all-layer survey, hyperparameter search, or training run
was performed.

## Non-goals

- proving that a perfect internal ledger exists;
- selecting a final architecture;
- treating attention maps as causal explanations;
- attributing the historical checkpoint difference to ordering alone;
- using incomplete Common Objects in Context labels as exhaustive negatives;
- suppressing terminal logits without identifying a valid next object; or
- explaining every language or vision layer globally.
