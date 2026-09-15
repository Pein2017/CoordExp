---
title: Sorted Full-Canvas Visual-Token-Budget Intervention
description: Paired input-side test of whether denser full-canvas Qwen visual tokenization restores calibrated localization support for owners that lacked support in the frozen Sorted step-4887 census.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: authorized_separately
unit_id: 2026-08-03-sorted-full-canvas-visual-token-budget-intervention
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-08-03
---

# Sorted Full-Canvas Visual-Token-Budget Intervention

## Closure

Execution is complete under run `20260803T161241Z`. The evidence chain is
verified, but the treatment arm is scientifically invalid because resolved
retention was `86/114 = 0.754`, below the frozen `0.90` gate. The nominal
`14/63` persistent-owner recovery is therefore void under this unit's declared
rule. The bounded verdict, artifact handles, diagnostics, and next
discriminator are owned by [results.md](results.md). The protocol below remains
unchanged as the record of the prospective design.

## Decision and outcome

This unit decides whether the strongest unexplained cohort from the completed
[Sorted owner accessibility phenotype census](../2026-08-03-sorted-owner-accessibility-phenotype-census/results.md)
should be treated as a full-canvas visual-token-density limitation or whether
that route should be closed.

The behavioral unit is one frozen physical owner from the predecessor ledger.
The primary observation is whether a native false-negative owner previously
classified as `persistent_no_tested_localization_support` gains calibrated
category-conditioned local support when only the full canvas receives a denser
Qwen visual-token grid.

This unit does not train a model and does not evaluate natural greedy owner
coverage. A successful likelihood intervention may authorize a later training
probe; it cannot itself promote a training objective or architecture.

## Question

Holding the Sorted step-`4887` checkpoint, prompt text, category query,
assistant history, owner ledger, coordinate tokens, candidate bank, and scoring
functional fixed, does approximately doubling the number of merged full-canvas
visual tokens restore native-true-positive-calibrated localization support for
a material, cross-image subset of persistent owners?

## Competing explanations

The working explanation is a **full-canvas patch-sampling limit**: the same
optical raster information is present, but small owners occupy too few visual
patches for the category-to-coordinate readout to form a usable local peak.

The strongest alternative is that apparent size is only a proxy for occlusion,
annotation-extent ambiguity, same-category competition, or a category-to-
coordinate interface failure. Under that explanation, a denser token grid does
not restore support above the identity-control noise floor while native-TP and
resolved controls remain stable.

The intervention cannot establish whether additional sensor detail would help.
Both image pools are deterministic resamples of the same raw COCO JPEGs. It
tests denser patch/token sampling over the same optical information.

## Source boundary

| Source | Frozen role |
| --- | --- |
| Predecessor run | `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-03-sorted-owner-accessibility-phenotype-census/20260803T065743Z/` |
| Predecessor capture manifest | `baa4081fdc072f22ba6a4cf705574a957f88f492f370cca4ddf2214c5b8a8df5` |
| Predecessor calibration | `9dd6d7646fc55db6155124dc4bbfa46642b32b006d39758bd2e24d1ca97058c5` |
| Checkpoint/runtime | Geometry-sorted pure-cross-entropy/type-gate step `4887`, HF fp32, repetition penalty `1.0` |
| Panel | Frozen human-refined twelve-image panel, `346` physical owners |
| Raw pixels | `/data/CoordExp/public_data/coco/raw/images/val2017/` |

The predecessor plan remains the authority for owner identity, native
generated-prefix tokens, category tokens, candidate identity, normalized
coordinate tokens, ambiguity assignment, and one-times pixel-frame geometry.
The successor overlay owns only the changed media/prompt identities and the
pre-score query-group subset.

## Arms and changed factor

### Current canvas token budget

The current image pool was generated with factor `32` and
`max_pixels = 32 * 32 * 1024`. It yields approximately `1024` merged visual
tokens per image. The predecessor census is the decision-bearing current-arm
evidence. The successor pipeline performs only a representative real-HF parity
smoke; it does not spend GPU time re-running the complete current arm.

### Doubled canvas token budget

The treatment pool is generated directly from the same raw COCO JPEGs with the
same factor-`32`, maximize-under-cap dimension policy and PIL RGB/LANCZOS/JPEG
materialization, changing only:

```text
max_pixels: 32 * 32 * 1024
          -> 32 * 32 * 2048
```

This produces about twice as many merged visual tokens and about `sqrt(2)`
times the linear canvas extent. Processor-side resize remains disabled. A
processor `min_pixels` or `max_pixels` setting is not an arm because it is
inert when `do_resize=false`.

No third scale is authorized by this unit. A partial result may motivate a
separate successor, but this run stops after the doubled-budget arm.

## Invariants

The following must be byte-identical to the sealed predecessor plan:

- checkpoint, adapter, selected-token embedding delta, tokenizer, precision,
  attention implementation, and repetition-penalty stratum;
- system and user prompt text outside the visual pad run;
- every native generated-prefix token sequence;
- category descriptions and canonical category-query suffix tokens;
- candidate identifiers, normalized coordinate bins and coordinate token IDs;
- owner identifiers, cohort labels, ambiguity assignments, and strict
  one-times pixel-frame geometry;
- the `17` score-independent owner-centred candidate roles and exact-GT anchor;
- full-vocabulary log-softmax coordinate scoring and the predecessor support
  functional.

The following must change and be sealed per treatment image:

- raster bytes, declared/decoded dimensions, media digest;
- `image_grid_thw` and the number of image-pad tokens;
- executed prompt identity, observed/query prefix digests, and exact-prefix
  admission identifiers;
- explicit intervention unit, arm, overlay, and predecessor-plan identities.

The treatment plan must fail if the current `1024`-budget image pool cannot be
reproduced byte-for-byte from raw COCO under the declared resampler.

## Frozen cohorts

All cohorts are selected from the predecessor presentation ledger before any
treatment score exists.

| Role | Frozen population | Treatment contexts |
| --- | --- | --- |
| Primary persistent cohort | `63` persistent owners outside image `4134` | Every predecessor-tested non-loop native context for each owner's category |
| Person-only primary safeguard | `51/63` primary owners whose normalized description is `person` | Same as primary |
| Scale-flat secondary cohort | `9` persistent owners in loop-degenerate image `4134` | Root plus the frozen upper-bound frontier and upper-bound best-diagnostic context when available |
| Treatment calibration | `70` discovery native-TP controls | Deterministic due boundary |
| Native-TP retention | `71` confirmation native-TP controls | Due and frozen known-support contexts |
| Resolved retention | `114` resolved native-FN owners | Frozen usable-support contexts plus due/frontier when available |

Image `4134` is not discarded. It is a declared secondary stratum because its
person cohort was scale-flat in the predecessor, its native route was loop-
degenerate, and its `148` contexts dominated predecessor wall time. It cannot
decide the primary scale mechanism.

The frozen query-group selection is the union needed by these roles. Shared
category/context groups are executed once. Candidate membership is never
filtered or expanded using treatment scores.

## Primary observation and calibration

Raw log probabilities are never compared across visual-token-budget arms.

Within the doubled-budget arm, recompute the discovery native-TP lower-decile
thresholds for:

- `peak_lift`; and
- `local_concentration`.

The treatment support criterion is the same conjunction as the predecessor:
both statistics must exceed the treatment-specific discovery-TP `q10` plus
epsilon `0.002`, under ambiguity bound `U`, at one eligible tested context.
Rank and margin remain diagnostics and never define support.

The primary estimand is the owner-level proportion of the `63` primary
persistent owners that gain treatment-calibrated support. It is reported
overall and for the `51` person owners, with Wilson `95%` intervals, recovered
image count, per-image counts, and leave-one-image-out sensitivity.

## Controls and gates

1. **CPU identity gate:** all twelve current-pool images reproduce the sealed
   predecessor media digests from raw COCO under the declared transform.
2. **Real-HF parity gate:** one representative current-arm shard reproduces
   predecessor selected log probabilities within `1e-3`, prompt/media/grid
   identity, and cache-versus-uncached admission.
3. **Treatment mechanics gate:** one real doubled-budget case executes the
   changed grid, new prompt-pad count, sealed media, scalar parity, and finite
   coordinate scores before the full capture starts.
4. **Native-TP retention:** at least `90%` of the `71` confirmation native-TP
   controls retain calibrated support.
5. **Resolved retention:** at least `90%` of the `114` resolved owners retain
   support at one declared control context.
6. **Cross-image safeguard:** recovered primary owners must span at least three
   images. A result owned only by image `7511` is image-conditional.

If either retention gate fails, the treatment arm is out of calibration. Any
persistent recovery is scientifically void rather than a positive result.

## Minimum decision-bearing effect and stop rule

The doubled-budget route is promoted as a bounded explanation only if all are
true:

- at least `13/63` primary persistent owners recover support;
- at least `10/51` person-only primary owners recover support;
- recovered owners span at least three images; and
- both retention gates pass.

The thresholds are mechanism-screen thresholds, not population-effect
estimates. Below approximately twenty-percent recovery, the intervention does
not explain enough of the persistent cohort to justify a scale-targeted
training route.

Stop immediately with no scientific result if the CPU identity or real-HF
parity gate fails. Stop with an invalid-arm verdict if calibration/retention
fails. If recovery is below the minimum or disappears in the person-only or
leave-one-image-out sensitivity, close the full-canvas token-density route and
move to the scale-flat `4134` route/repetition discriminator. Do not widen the
bank, broaden the query, add crops, retune thresholds, or launch training from
this unit.

## Alignment

| Element | This unit |
| --- | --- |
| Intervention unit | Whole-image visual token budget |
| Final outcome represented | Category-conditioned owner-local likelihood support, not natural rollout coverage |
| Transfer assumption | Support restored without weight changes identifies a subset whose one-times miss is compatible with patch-sampling accessibility |
| Preservation risk | Denser visual grids may globally perturb MRoPE, DeepStack, and the category-coordinate interface |
| Preservation control | Per-treatment TP calibration plus native-TP and resolved retention |
| Neutral evidence | Ambiguity-bound flips, unsupported unmatched rows, raw cross-arm log-probability shifts |
| Signal supply | Frozen exact anchors and local banks for every eligible owner; discovery TP controls define the treatment operating point |

## Execution outline

1. Build and seal the treatment image pool, derived panel, query-group subset,
   and compact visual-prompt overlay under a fresh run identifier.
2. Run CPU contract checks and byte-exact current-pool reproduction.
3. Run a current-arm and doubled-budget real-HF smoke on image `13348`.
4. Choose the largest candidate batch among `1`, `8`, and `16` that passes the
   live scalar/argmax parity and memory gate; never silently fall back after a
   launched shard.
5. Capture the selected doubled-budget shards across available GPUs, largest
   estimated shard first. This successor is score-only and emits no free-decode
   sidecars.
6. Analyze only after every selected shard is complete and identity-consistent.
7. Run independent Fable audit over the fixed tree and executed artifacts
   before closing the result.

The rough full-run target is `6–10` GPU-hours total and roughly `2–3` hours
wall time under four or more free GPUs. The representative smoke, not this
estimate, owns launch sizing.

## Artifact handle

Every attempt uses a fresh immutable root:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-08-03-sorted-full-canvas-visual-token-budget-intervention/<run-id>/
```

The compact receipt must bind source code/diff identity, predecessor plan and
capture identities, raw/current/treatment media identities, transform policy,
derived panel, visual prompt overlay, selected query groups, resolved runtime,
per-shard arm identity, and analyzer identity. A path alone is not evidence.

## Permitted claims

On success, the strongest permitted claim is:

> Under the frozen Sorted step-4887 category-conditioned scoring interface, a
> deterministic denser full-canvas visual-token grid restored calibrated local
> support for a material cross-image subset of previously persistent owners
> while native-TP and resolved controls retained support. For that subset, the
> one-times failure is compatible with a full-canvas patch-sampling limit.

On failure, the permitted claim is only that this one doubled-token-budget
full-canvas intervention did not restore support above the declared threshold
under the fixed interface. Failure does not establish absent visual
information, crop failure, architecture necessity, or owner impossibility.

## Not claimed

- No new optical or sensor information is introduced.
- No owner crop or localized zoom is tested.
- No natural greedy recall, final owner-set utility, or stopping policy is
  improved by this unit.
- No raw cross-arm likelihood is directly comparable.
- No treatment, objective, explicit state interface, or architecture is
  promoted.
- No unrecovered owner is declared visually impossible.

## Originating-intent alignment

| Condition | Source | Class | Decision effect | Disposition |
| --- | --- | --- | --- | --- |
| Explain data-driven FN heterogeneity over the frozen owner census rather than return to one anecdotal owner | User direction and predecessor result | scientific invariant | cohort and claim | inherited |
| Exclude owner crop-and-upscale | User direction and predecessor result | scientific invariant | intervention and claim | inherited |
| Change only full-canvas token density; keep owner/query/bank/history fixed | Predecessor next discriminator | scientific invariant | estimand | inherited |
| Use doubled rather than quadrupled visual-token budget | Independent runtime audit and lead synthesis | conservative design choice | dose, cost, stop rule | approved within the current probe request |
| Exclude `4134` from the primary but retain it as a restricted secondary stratum | Predecessor scale-flat residue and runtime tail | conservative design choice | primary cohort and cost | approved within the current probe request |
| Do not automatically escalate to a third scale | Lead scope control | conservative design choice | stop rule and cost | approved within the current probe request |
