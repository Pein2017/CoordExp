---
title: Sorted Crossing-Boundary Owner Release and Realization
description: Prospective score-only decomposition of description release, target-conditioned coordinate realization, and owner displacement at the native boundary where the sorted route crosses a supported false-negative owner.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: authorized_within_goal
unit_id: 2026-08-03-sorted-crossing-boundary-owner-release-realization
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-08-04
---

# Sorted Crossing-Boundary Owner Release and Realization

## Decision and outcome

The completed [native-prefix reachability-prevalence
result](../2026-08-03-sorted-supported-fn-native-prefix-reachability-prevalence/results.md)
shows that owner-local support frequently exists before native omission, while
STOP and a pure covered-owner ledger failure are not the leading simple
explanations. It also shows that a generic favorable prefrontier context is not
necessarily a due context.

This unit therefore asks the smallest still-decision-bearing question at the
exact **crossing boundary**:

> When the native sorted trajectory moves past a supported false-negative
> owner, does the target fail because its natural description path is not
> released, because its target-conditioned coordinate landscape cannot
> realize it, or because another physical owner displaces it?

The outcome is an owner-level case series over all 26 U-bound crossing owners.
It may route one simpler training hypothesis, or close the favorable-surface
route as a weak descriptor. It cannot promote an owner-commit token, typed
binding, contrastive loss, slot, detector, RL policy, or other architecture.

## Exact state pair

For target physical owner `C`:

- `P` is the exact native self-prefix at the last root/ahead boundary before
  the next complete native row moves `C` to `passed_by_frontier`;
- `E` is that exact next native emitted row, including its realized
  description and coordinates; and
- `P+E` is the exact native boundary immediately after `E`.

The primary cohort is every owner that satisfies the frozen U-bound favorable
top-three surface at `P`. The current sealed census gives exactly `26` owners
across all twelve images. The L-bound crossing cohort (`25`) and the owners
favorable under U and L at the same crossing context (`24`) are sensitivities,
never replacement primary denominators.

The cohort retains both preregistered native-row strata:

- `12/26` have an `E` that strict-matches a physical owner; and
- `14/26` have an unmatched `E`.

Do not run only the twelve matched rows. That would select away the other half
of the cohort and structurally bias the result toward valid-owner displacement.

Before any GPU work, a CPU-only plan builder must emit and seal one cohort
registry row per owner. Each row binds:

- target owner and image ID;
- `P` boundary/context ID and complete prefix token digest;
- `E` sidecar row ID/index, coordinate token IDs/digest, raw-span digest, and
  strict match status, plus the separately derived full-row suffix identity;
- `P+E` boundary/context ID and complete prefix token digest;
- U, L, and same-context U/L crossing flags; and
- the next complete native row `F` and its post-`E` boundary when present.

The predecessor boundary convention is authoritative: boundary index `b`
contains native rows `< b`, so `E` is sidecar row index `b`, and `P+E` is
boundary `b+1`. The plan must independently re-derive U=`26`, L=`25`, exact
same-context U/L=`24`, matched-E=`12`, and unmatched-E=`14`, or fail before
request construction.

## Exact token ownership

No conclusion-bearing string is retokenized.

- `P` and `P+E` use the literal generated-prefix token IDs and digests from
  the sealed context registry.
- Full-row `E` tokens and their digest are owned only by the literal suffix
  `tokens(P+E) - tokens(P)`. The sidecar does not contain a full-row token
  sequence: validate only E's coordinate-token subsequence and coordinate
  digest against it, together with row ID/index, description, strict-match
  fields, and raw-span digest where the raw span bytes are available.
- `F`, when used, is the analogous literal suffix between consecutive native
  boundaries after `E`.
- `D_C` uses the sealed category `query_suffix_token_ids` through
  `<|box_start|>`; its wrappers, description tokens, and digest must match the
  predecessor request identity.
- Inserted `C` uses the owner's sealed exact-GT-anchor coordinate token IDs
  plus frozen natural wrappers; it is never reconstructed from floating-point
  coordinates or prose.

Coordinate-only greedy has a fixed grammar: exactly four tokens drawn from the
1,000 coordinate-token IDs, followed by the exact `<|box_end|>` token. Any
other token, arity, premature termination, extra coordinate, or invalid box is
`malformed` and never silently repaired.

## Two separate measurement ladders

Let `D_C` be the target's exact natural description path:

```text
<|object_ref_start|>{description}<|object_ref_end|><|box_start|>
```

The two ladders are never pooled as one raw score.

### 1. Natural description release

At both `P` and `P+E`, teacher-force the target path and the exact native next
action. Record:

- gate margin where the native alternative is STOP;
- first-divergent-token target-versus-native margin;
- complete description-path sum and token mean; and
- whether deterministic argmax follows the target through every observable
  description token.

If target and native descriptions are identical, description release is not
observable: the first informative divergence is a coordinate token. Such
cases are coordinate-only observations, not a separate factorial level.

### 2. Target-conditioned coordinate realization

At both `P+D_C` and `P+E+D_C`:

- score the frozen target-local and same-category physical-owner candidate
  families using the predecessor calibration and identities;
- record target owner rank, best competing owner, local-peak location,
  target-versus-competitor margin, and support disposition; and
- deterministically greedily decode only the coordinate row through
  `<|box_end|>`, then owner-match the resulting box.

No free row or trajectory decode is part of the primary pass. Low-temperature
in-row coordinate sampling may be added only after the deterministic pass if
likelihood and greedy coordinates leave the failure branch unresolved. The
optional diagnostic is frozen to temperature `0.2`, `K=16`, the same
coordinate grammar, no repetition penalty, and base seed `17`. Per-owner seeds
are derived deterministically from the sealed `(unit_id, image_id,
gt_owner_id, context_id, base_seed)` tuple. RNG/runtime identity and derived
seeds are sealed in the sampling receipt. Sampling remains disabled unless this
receipt contract is implemented. It cannot alter deterministic primary branch
assignments or the primary routing denominator.

## Primary classification

Classification is owner-level and uses only within-owner, within-context
comparisons. Raw log probabilities are never pooled across images.

U owns primary support, rank, competitor, and branch fields. L is a sensitivity
reported beside U and never changes the U branch. Exact margin ties, or signs
that differ between admitted cached and uncached scoring, are ambiguous.
Owner matching uses the predecessor's one-row strict physical-owner matcher;
same-description semantic drift and overlapping-but-unmatched boxes remain
explicit unmatched or ambiguous outcomes.

At `P+E`, assign at most one primary branch in this exhaustive order:

1. **displaced**: either:
   - `likelihood_displaced`: the U best candidate belongs to another physical
     owner with a strictly negative target-minus-competitor margin; or
   - `greedy_displaced`: the coordinate-only greedy box strict-matches another
     physical owner.
   Both sub-tags are retained. Candidate/greedy disagreement still enters the
   displaced branch if either sub-tag is uniquely true;
2. **release_lost**: branch 1 does not apply, the natural target description
   loses at its first observable description divergence with a strictly
   negative target-minus-native margin, and forced `D_C` retains U-calibrated
   target-local coordinate support. Because the native trajectory is greedy,
   the negative sign is construction-expected for a different-description
   target; only its calibrated magnitude and the retained coordinate support
   are informative;
3. **realization_fail**: branches 1 and 2 do not apply, forced `D_C` lacks
   U-calibrated target-local support, and the greedy box is target-missed,
   unmatched, or malformed without a unique other-owner displacement; or
4. **ambiguous**: every remaining case, including exact ties, nonunique owner
   matches, missing score fields, and branch predicates that cannot be
   determined without optional sampling.

Same-description cases skip release classification. At `P`, their
`P+D_C` coordinate readout is construction-determined because `D_C` is already
the exact prefix of native row `E`; record it for replay only and never treat it
as displacement evidence. Classification at `P+E` remains decision-bearing.
Optional sampling may describe an ambiguous deterministic case but cannot
reassign it.

Separately tag the paired change from `P` to `P+E` as target access opened,
retained, or suppressed for each ladder. This tag describes local prefix
interaction; it is not a counterfactual claim that the owner would otherwise
have been emitted.

Valid-owner starvation is not identified by one boundary. Join the existing
trajectory features as a descriptive tag only.

## Controls

### Native replay alignment

Bind the exact checkpoint, tokenizer, prompt, image, precision, attention,
position-ID, and repetition-penalty identity used by the predecessor census.
Before reading a probe result, re-score the native action and require argmax
replay through every token that precedes the tested divergence. Quarantine a
case on mismatch. More than two quarantined primary owners stops the unit
before interpretation.

Use explicit position IDs, no `generate()`, and fresh cache state per logical
context group. Cached execution is admitted only if a real matched-E,
unmatched-E, and same-description smoke each have maximum selected-logit
absolute difference at most `1e-3` versus uncached scoring and preserve every
compared argmax, margin sign, owner rank, support disposition, owner match, and
primary branch. Otherwise use uncached evidence for the affected surface; if
uncached replay also fails, quarantine the case. The parity receipt is
conclusion-bearing.

### Timing controls

- Derive a disjoint descriptive timing-control registry by excluding all 26
  primary owners, then selecting owners with at least one U-favorable,
  noncrossing boundary whose immediate next sidecar row strict-matches a
  physical owner. For each owner choose the latest qualifying native boundary;
  break any remaining tie by context ID. The audit reconstruction expects
  about 14 owners, but the sealed CPU registry owns the exact count. A mismatch
  is reported, not patched by allowing primary/control overlap. These controls
  remain descriptive because timing, description identity, and route tier are
  entangled.
- Select one exact due-boundary native true positive per image for coordinate
  replay calibration, preferring a due-supported non-singleton owner and then
  breaking ties by native row index and owner ID. This must yield twelve fixed
  owners before scoring or stop for redesign.

## Secondary downstream compatibility

The user also asked whether catching `C` after it was skipped damages later
owners. This is a secondary local teacher-forced readout, not a primary branch
or launch gate.

Seal all primary branch assignments before reading any secondary field. The
primary secondary readout exists for all 26 owners: append exact clean `C` to
`P`, then compare exact `E` under `P+C` versus native `P`. The optional late-
catch-up extension appends `C` to `P+E`, then compares exact `F` under
`P+E+C` versus native `P+E` when `F` exists. Report description, coordinate,
and complete-row paired deltas separately.

The twelve TP replay controls also supply a benign-substitution reference:
replace a natively emitted TP row with its exact clean GT twin, then score the
following exact native row. Secondary deltas are interpreted only relative to
that reference distribution and never pooled across images as raw values.
These readouts measure local compatibility of exact row sequences; they are
not final-set retention, eventual recovery, or free-rollout results.

## Primary evidence products

Emit one sealed owner record per primary case with:

- target owner and image identity;
- `P`, `E`, and `P+E` context identities;
- matched/unmatched `E` stratum and same/different-description observability;
- replay-alignment receipt;
- both release and realization ladders at both contexts;
- pre/post access tag and primary branch;
- existing starvation/competition/frontier tags; and
- secondary compatibility fields when eligible.

Produce a compact matrix over all 26 owners and owner-local paired plots for
the two ladders. Colors encode only discrete branch/status fields or within-
owner deltas, never cross-image raw likelihood.

The immutable artifact family must include:

- CPU plan receipt, primary/control cohort registry, request registry, and all
  exact input/source digests;
- scorer source identity, complete runtime/model/tokenizer identity, raw shard
  score rows, per-shard receipts, and merge receipt;
- cached-versus-uncached parity receipt and quarantine ledger;
- primary branch registry sealed before secondary analysis;
- secondary compatibility rows and a separate optional-sampling receipt when
  either is executed;
- analyzer source, JSON/Markdown report, owner records, and exact hashes; and
- visualizer source, visual specs/products, and self-sealed manifest.

Every conclusion-bearing file and source identity receives path, byte size,
and SHA-256 lineage. Missing, tampered, duplicated, or unknown artifacts fail
closed.

## Strongest alternatives

- The favorable crossing cohort is selected on existing coordinate support,
  so survival of forced-description geometry is partly selected-for and weak
  evidence; failure to survive is more informative.
- The exact GT target row is an oracle intervention. Its downstream
  compatibility does not imply the model could naturally generate it.
- An unmatched `E` can mix unsupported output, annotation incompleteness,
  duplicate/extent drift, and malformed ownership. Preserve this stratum; do
  not give it one causal label.
- A single crossing boundary cannot prove trajectory-level starvation or a
  stable set ledger.

## Stop rule

Stop after one deterministic scoring pass, bounded optional in-row sampling if
needed, visualization, and independent audit.

- A deterministic case is **interpretable** only if replay is admitted, every
  required primary score exists, no tie or nonunique match is present, and the
  truth table assigns exactly one of the first three branches without optional
  sampling. At least `20/26` owners, including at least six matched-E and seven
  unmatched-E owners, must be interpretable or no successor is routed.
- If at least two thirds of those deterministic interpretable owners land in
  one branch,
  route exactly one mechanism-matched successor at that branch.
- Report separately any case where `likelihood_displaced` is true while the
  coordinate-only greedy box strict-matches the target. If excluding this
  decoding-contradicted likelihood cell would change whether a branch reaches
  the two-thirds threshold, treat the result as split and route no successor.
- If branch shares split, or if pre/post changes do not cohere with the branch
  ladder, close the favorable-surface route as a weak local descriptor and
  return to a due-anchored analysis of all 114 owners.
- If replay quarantines exceed two, stop without interpretation and repair the
  runtime alignment rather than changing thresholds.
- Do not add a commit token, binding head/loss, slot, detector, or RL objective
  merely because the parallel training design exists. It is a complexity
  ceiling until this simpler probe supplies mechanism-matched evidence.

## Artifact handle

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-08-03-sorted-crossing-boundary-owner-release-realization/
  <run-id>/
```

## Not claimed

- No natural probability for the complete target row.
- No population prevalence beyond the frozen twelve images.
- No causal effect of emitting `E` or inserting the GT row for `C`.
- No distinction of trajectory-level starvation.
- No final-set, natural-stop, or long-horizon preservation result.
- No training or architecture promotion.

## Originating-intent alignment

| Condition | Source | Class | Decision effect | Disposition |
| --- | --- | --- | --- | --- |
| Explain why supported false-negative owners are omitted | User direction | scientific invariant | primary question | inherited |
| Use all owners data-first rather than one anecdotal owner | User direction | scientific invariant | 26-owner cohort | inherited |
| Compare the self-prefix before and after the row that skips the target | User direction | scientific invariant | exact `P`/`P+E` pair | inherited |
| Separate description release from coordinate landscape shape | User direction and scientific audit | scientific invariant | two measurement ladders | inherited |
| Retain unmatched native rows | Scientific audit | conclusion-protection control | primary strata | inherited |
| Prefer greedy and likelihood; add sampling only if unresolved | User delegation | conservative design choice | staged execution | inherited |
| Measure downstream effect of catching the skipped owner | User direction | secondary scientific question | compatibility readout | inherited |
| Keep owner-commit training only as a complexity ceiling | User direction | architecture boundary | stop rule | inherited |
