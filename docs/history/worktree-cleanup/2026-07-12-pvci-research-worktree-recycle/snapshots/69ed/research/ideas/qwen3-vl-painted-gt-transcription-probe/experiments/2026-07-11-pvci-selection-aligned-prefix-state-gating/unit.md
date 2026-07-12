---
title: PVCI Selection-Aligned Prefix-State Commit Gating
description: Tests whether the same object row writes a stronger native local commit field only when the frozen prefix state is aligned with that object, using a stable nonselected-object difference-in-differences control.
type: idea
role: research-unit
authority: non_normative_research
promotion_status: not_promoted
unit_id: 2026-07-11-pvci-selection-aligned-prefix-state-gating
topic: qwen3-vl-painted-gt-transcription-probe
status: complete
tags:
  - coordexp-swift
  - research-unit
  - qwen3-vl
  - native-commit
  - prefix-state
  - trajectory
updated: 2026-07-11
---

# PVCI Selection-Aligned Prefix-State Commit Gating

## Question

For the identical image, object identity, phrase, canonical coordinates, row
syntax, and candidate universe, does completing an object row write a stronger
non-literal local suppression field when the frozen native prefix is aligned
with that object than when the object is merely annotated-uncovered but not the
native next-row selection?

This unit changes no architecture and trains no model. It is not a causal proof
that self-selection alone creates the field: the paired prefixes also differ
in depth, preceding rows, remaining-set size, coordinate history, and
same-description history. The strongest permitted positive claim is
`selection_aligned_prefix_state_gating_supported` on this frozen four-image
panel. A stable nonselected same-class object L is written at both prefixes to
subtract generic depth/context responsiveness.

## Prior Evidence and Decision Relevance

The completed endogenous write/read bridge showed that self-generated valid
coordinates and their canonical snap both write a one-step object-relative
spatial suppression field. The completed source-swap factorial then falsified
generic any-coordinate occupancy on its panel:

```text
AddressRead_J = 7/8
AddressRead_K = 0/8
AddressRead_E = 1/8
joint Occupancy = 0/8
```

J was the object selected by the source free continuation; arbitrary same-class
annotated K and annotation-empty same-size E were not sufficient. The remaining
confound is whether J's effect reflects an already active prefix trajectory or
some generic object-row commit that the K/E factorial failed to expose.

One existing line-16 pilot already writes the same canonical GT4 car under C0
as off-trajectory K and under C2 as selected J. An ad hoc, non-authoritative
reanalysis with common same-description exact controls suggested:

```text
C0: LocalNear = +4.2097, CommonAddressExactControl = -4.7130, Read = false
C2: LocalNear = -4.7195, CommonAddressExactControl = -8.7226, Read = true
InteractionLocal = -8.9292
InteractionAddress = -4.0095
```

This illustrates why the unit is worth running, but it is not an eligibility
input or frozen result: it has no materialized reanalysis receipt and does not
include the stable-L difference-in-differences control. The new gate must
reproduce the target branch under the new exact contract.

## Frozen Four-Pair Panel

Source:

`/data/CoordExp/outputs/painted_gt/pvci_endogenous_commit_write_read/plan_step4887/write_read_attempts.jsonl`

Self-selection means exactly: the source native RP1.0 free continuation was a
complete valid row with one strict normalized-description plus IoU>=0.5 match
to the annotated-uncovered target T. Nonselection means the earlier source
continuation had one strict match to a different annotated-uncovered object.
No candidate-rank reinterpretation may change these labels. T's exact/near
rank and score under the new common universe are recorded as mediators.

Freeze these four independent image/object pairs:

| Pair | Earlier nonselected PRE | Later selected PRE | T |
|---|---|---|---|
| `line-000009-row-000/J4` | C0, source selects GT1 | C1, source selects GT4 | GT4 `person` |
| `line-000012-row-000/J4` | C1, source selects GT2 | C2, source selects GT4 | GT4 `person` |
| `line-000016-row-000/J4` | C0, source selects GT1 | C2, source selects GT4 | GT4 `car` |
| `line-000017-row-000/J3` | C0, source selects GT1 | C1, source selects GT3 | GT3 `person` |

T must remain annotated-uncovered at both prefixes. The two prefix histories
are the exact frozen canonical `prefix_row_texts`; no generated row is inserted
or replaced by the planner.

For each pair, form the intersection of annotated-uncovered GT indices at both
prefixes and retain only objects with T's normalized description. Candidate L
must differ from T, remain uncovered at both prefixes, never be the strict
source-selected object at any frozen C0/C1/C2 depth for that image, and admit a
valid deterministic near perturbation.

Choose L before inspecting any new write branch by minimizing the maximum
absolute mismatch to T's exact canonical-row summed log probability at the two
PRE boundaries. Use the completed bridge scorer's already-frozen raw PRE
scores; tie-break by the sum of the two mismatches, then minimum GT index.
Freeze and hash those source scores and the resulting choices:

```text
line 9:  T=GT4, L=GT3
line 12: T=GT4, L=GT3
line 16: T=GT4, L=GT6
line 17: T=GT3, L=GT2
```

At each PRE, the exact canonical T/L mean-logprob mismatch must be at most
`0.30` nat/token and their rank among common same-description exact candidates
must differ by at most two. These PRE-only eligibility checks reduce baseline
floor/preference confounding; they do not prove its absence.

## Frozen Candidate Universe

Materialize:

- `T_exact` and `L_exact`, their canonical compact rows;
- `T_near` and `L_near`, existing deterministic near perturbations with valid
  bounds, IoU in `[0.65, 0.95]`, and zero shared coordinate token IDs with their
  own exact row;
- `B`, the canonical exact rows for every common same-description object other
  than T and L.

The identical ordered candidate universe is scored under all six histories in
a pair. Candidate row text, token IDs, coordinate IDs, GT identity, and order
must be byte/token identical across histories. STOP is scored separately and
never enters the commit contrasts.

Use the identical external address-control identities for both writes:

```text
B = {W_exact for every common same-description W not in {T, L}}
```

`B` must contain at least one unique candidate; its frozen sizes are `8, 1, 2,
8` for the four pairs. Line 12 therefore has a fragile single external address
control, which must remain visible in reporting. Neither T nor L exact/near may
enter B.

## Prefix-by-Write Factorial

Score these histories:

```text
EARLY_PRE
EARLY_WRITE_T = EARLY_PRE + canonical_row(T)
EARLY_WRITE_L = EARLY_PRE + canonical_row(L)

LATE_PRE
LATE_WRITE_T = LATE_PRE + canonical_row(T)
LATE_WRITE_L = LATE_PRE + canonical_row(L)
```

The T row is exactly identical at early and late prefixes; the L row is exactly
identical at early and late prefixes. T and L share normalized description and
row structure. Within each prefix, PRE subtraction removes the direct baseline
likelihood difference before any cross-prefix comparison.

No free continuation is required for the primary result. Optional one-row
continuations are descriptive only and cannot change the label.

## Primary Contrasts

Freeze numerical sign tolerance `epsilon_num = 1e-3` nat and substantive
interaction margin `m = 1.0` nat before execution. For `W in {T, L}` and
`p in {early, late}`:

```text
Local(W,p) =
  log P(W_near | PRE_p + WRITE_W)
  - log P(W_near | PRE_p)

Exact(W,p) =
  log P(W_exact | PRE_p + WRITE_W)
  - log P(W_exact | PRE_p)

ExternalAddress(W,p) =
  Local(W,p)
  - median_{c in B}(
      log P(c | PRE_p + WRITE_W)
      - log P(c | PRE_p)
    )

Read(W,p) =
  [Local(W,p) < -epsilon_num]
  AND [Exact(W,p) < -epsilon_num]
  AND [ExternalAddress(W,p) < -epsilon_num]

InteractionLocal(W) = Local(W,late) - Local(W,early)
InteractionExact(W) = Exact(W,late) - Exact(W,early)
InteractionAddress(W) =
  ExternalAddress(W,late) - ExternalAddress(W,early)

SelectionSpecificLocalDiD =
  InteractionLocal(T) - InteractionLocal(L)

SelectionSpecificAddressDiD =
  InteractionAddress(T) - InteractionAddress(L)
```

Negative T interactions mean the same T write becomes more locally
suppressive in the later selection-aligned prefix. Negative
`SelectionSpecificLocalDiD` and `SelectionSpecificAddressDiD` mean that
strengthening is greater for T than for the stable nonselected L,
reducing—but not eliminating—the generic prefix-depth confound. The same B
identities normalize both writes, so the DiD cannot move merely because T and
L enter one another's control sets.

Raw direct-forward scores are primary. Native forward remains bfloat16;
logit readout/log-softmax is float32; aggregation is Python double. RP1.1 is
secondary only. Report phrase, structure, coordinate, and sequence spans; the
sequence contrast is the frozen gate and the coordinate span is mechanistic
localization evidence.

Record at both PREs:

- T/L exact and near summed log probability;
- their rank among the identical common candidate universe;
- margin to the top canonical candidate;
- STOP probability;
- source native-generation attribution and strict-match receipt.

These diagnose baseline preference/floor mediation; they do not redefine the
source selection labels.

## Frozen Gates

The four pairs are four independent images and four unique T identities. There
is no attempt/cluster pseudo-replication. Because `n=4` is small, a strong
terminal panel direction requires all four images. Three of four is exploratory
and returns an inconclusive label.

Define:

```text
TrajectorySwitch =
  Read(T,late)
  AND NOT Read(T,early)
  AND InteractionLocal(T) <= -m
  AND InteractionAddress(T) <= -m

SelectionSpecificSwitch =
  TrajectorySwitch
  AND NOT (Read(L,late) AND NOT Read(L,early))
  AND SelectionSpecificLocalDiD <= -m
  AND SelectionSpecificAddressDiD <= -m
  AND InteractionExact(T) < -epsilon_num
```

### H1: selection-aligned prefix-state gating supported

Require all of:

- `Read(T,late)` in `4/4`;
- `Read(T,early)` in `0/4`;
- `SelectionSpecificSwitch` in `4/4`.

Primary label: `selection_aligned_prefix_state_gating_supported`.

This supports a bounded prefix-state interaction, not a causal self-selection
mechanism and not a semantic ledger.

### H2: commit readable across both prefix states

If `Read(T,early)` and `Read(T,late)` both hold in `4/4`, return
`commit_readable_across_both_prefix_states`. This keeps a generic object-row
commit viable; it is not an equivalence claim.

### H3: selected-context read not replicated

If `Read(T,late)` holds in at most `2/4`, return
`selected_context_read_not_replicated`.

### H4: global prefix responsiveness

If `Read(T,late)` holds in at least `3/4`, both `InteractionLocal(T) <= -m`
and `InteractionLocal(L) <= -m` hold in `4/4`, both address interactions are
also `<= -m` in `4/4`, but the two selection-specific DiDs do not both reach
`<= -m` in `4/4`, return `global_prefix_responsiveness_supported`. This says
the later prefix generally strengthens same-class row writes rather than
uniquely privileging T.

### Otherwise

Return `mixed_cross_prefix_result_inconclusive`. Report all four raw pairs,
medians, minima/maxima, sign counts, and exact failed gate clauses. No
population prevalence or architecture claim is permitted.

Evaluate H1, H2, H3, H4, then the fallback in exactly that order and stop at
the first match. Analyzer overlap fixtures must prove that one and only one
label is emitted.

## Evidence and Eligibility Gates

- Reconstruct every source attempt and pair from the frozen bridge artifacts;
  no hand-authored prefix or target row may enter execution.
- Hash the source plan/manifest/hash receipt, completed source-swap artifacts,
  this unit, planner, scorer, analyzer, canonical/perturbation/scoring helpers,
  input JSONL, image bytes, config, live checkpoint, Git state, and argv.
- Require equivalence to the completed native bridge source receipt for model
  family/base weights, adapter and special-token delta hashes, checkpoint step,
  tokenizer/chat template, image preprocessing, prompt-prefix construction,
  native dtype, and direct-forward/log-softmax semantics. New scorer argv,
  branch count, and analyzer code are expected to differ and receive new hashes.
- Require native bfloat16 forward, float32 log-softmax/readout, and Python-double
  aggregation. Float32 model execution is allowed only as a separately labeled
  numerical diagnostic and cannot replace the native result.
- Require exact source native-generation attribution for earlier-other and
  later-T, with no ambiguity or parser failure.
- Require T and L annotated-uncovered at both prefixes and L distinct from both
  source-selected objects.
- Require exact candidate-universe and control-set equality across all six
  histories in each pair.
- Require designated-candidate determinism for the raw primary view. RP1.1 is
  optional robustness evidence; missing RP1.1 cannot invalidate a complete raw
  result.
- Freeze the two-pair GPU gate as `line-000016-row-000/J4` and
  `line-000012-row-000/J4`. The first spans the existing two-row depth case;
  the second stress-tests `|B|=1`.
- Gate GO is contract/runtime health only: all twelve histories score finite;
  exact T/L write-token, ordered candidate-universe, B-set, source-attribution,
  checkpoint/config/image/tokenizer/precision, and raw determinism receipts
  pass with zero scorer/analyzer failures. Causal sign never gates GO. The old
  line-16 source-swap event cannot substitute because it lacks the stable-L
  factorial.
- Partial runs, any failed pair, prior-runtime-only checkpoint attestation, or
  any source/hash/precision/determinism drift are ineligible for terminal labels.

## Stop Condition

Stop after one passing two-pair runtime gate and the exact frozen four-pair
factorial, or immediately on contract failure. Return one frozen primary label,
the per-pair T/L interactions, and the exact bounded interpretation. No
architecture is promoted.

## Planned Artifacts

- plan: `/data/CoordExp/outputs/painted_gt/pvci_selection_aligned_prefix_state_gating/plan_step4887/`;
- gate scorer: `/data/CoordExp/outputs/painted_gt/pvci_selection_aligned_prefix_state_gating/gate2_v1/`;
- full scorer: `/data/CoordExp/outputs/painted_gt/pvci_selection_aligned_prefix_state_gating/full4_v1/`;
- analysis: scorer root plus `/analysis_v1/`.

## Research Unit Closeout

### Execution and eligibility

The exact two-pair runtime gate passed independently of causal sign, followed by
the exact frozen four-pair panel. The full scorer completed all four pairs and
all six histories per pair with zero failures under the live step-4887
checkpoint. Native model execution was bfloat16; candidate logit readout and
log-softmax were float32; sequence aggregation used Python double. The analyzer
accepted the run as `complete_contract_eligible` and emitted exactly one frozen
primary label:

```text
mixed_cross_prefix_result_inconclusive
```

The full artifacts are:

- scorer: `/data/CoordExp/outputs/painted_gt/pvci_selection_aligned_prefix_state_gating/full4_v1/`;
- analysis: `/data/CoordExp/outputs/painted_gt/pvci_selection_aligned_prefix_state_gating/full4_v1/analysis_v1/`.

The execution receipt binds the pre-execution research contract at SHA-256
`a52c99aac3f9adfbba1f5ad84e17303338f94945e7999bd3f92e5dc7e6986bd1`.
This closeout necessarily changes the live unit-file hash and is not presented
as the exact executed input.

Artifact SHA-256 receipts:

```text
summary.json           63f2422a6a70c6ac1af73475302b4eb241243518ca99a3f5af35d8de53a71487
events.jsonl           3411824bf85aff88d55979781b97d19ed7d5995e54d9c2975960a5de18a998ec
execution_receipt.json b3c994625792537ae37a23224156d8dd6403631f39ef4fcda2980ce2afb09410
analysis.json          c878e0891d6ece2d2070e9d30962843e91c3c0a367fa8c522a4ba5a50538bb85
pair_analysis.jsonl    ef080bfed9e04dcabffa0043487454294a88b9c813c7e00cd8eb0be464c9cf48
```

`gate_go: false` in the full analysis is expected because the analyzer reserves
that field for the exact two-pair runtime gate identities. The full run is
nevertheless terminal-eligible; its scientific result is carried by
`eligible: true`, `analysis_scope: complete_contract_eligible`, and the frozen
verdict label.

### Observed primary result

For target T, the same canonical write became more readable under every later
prefix:

```text
Read(T,late)  = 4/4
Read(T,early) = 1/4

InteractionLocal(T):
  negative 4/4; <= -1 nat 4/4; median -4.4800

InteractionAddress(T):
  negative 4/4; <= -1 nat 3/4; median -3.0134
```

However, the stable nonselected same-description control L strengthened at
least as consistently and generally more strongly:

```text
InteractionLocal(L):
  negative 4/4; <= -1 nat 4/4; median -6.4611

InteractionAddress(L):
  negative 4/4; <= -1 nat 4/4; median -3.7736
```

Consequently, the target-minus-control difference-in-differences did not show
a replicated selection-specific advantage:

```text
SelectionSpecificLocalDiD:
  negative 2/4; <= -1 nat 2/4; median -0.0258

SelectionSpecificAddressDiD:
  negative 2/4; <= -1 nat 1/4; median +1.1276

SelectionSpecificSwitch = 0/4
```

The RP1.1 secondary view agrees with the absence of a selection-specific
switch and cannot change the primary raw verdict.

### Bounded interpretation

Supported on this frozen panel:

- the same later native prefix state makes a canonical coordinate-row write
  more locally readable for T;
- this responsiveness is not unique to the native next-row-selected object;
- a stable nonselected same-class object L is strengthened at least as
  consistently, so the target effect is better explained by generic
  prefix-history, depth, trajectory phase, or another shared context variable
  than by a current-object-specific selection gate.

The formal result remains the frozen fallback rather than H4 because line 12's
T address interaction was `-0.8616` nat and therefore missed the predeclared
`-1.0` margin. This prevents promotion of a universal global-prefix claim, but
does not rescue H1: the selection-specific DiDs are mixed and
`SelectionSpecificSwitch` is `0/4`.

The result is intentionally bounded by the frozen `n=4` panel. Line 12 has only
one external B address control. Both limitations restrict generalization but do
not invalidate the contract-eligible descriptive result.

Not supported:

- `selection_aligned_prefix_state_gating_supported`;
- a semantic object ledger or object-specific current-selection memory in the
  native text prefix/KV state;
- durable multi-object coverage, commit composition, autonomous next-object
  selection, calibrated STOP, or any architecture promotion.

### Decision consequence

The result narrows the architecture posterior without selecting an
architecture. Qwen3-VL exposes a real one-step spatial commit primitive, but
the native prefix state examined here does not supply the object-specific
coverage routing needed to redistribute selection toward uncovered instances.
Merely reading or amplifying the later decoder state is therefore not yet a
credible substitute for an explicit object-specific coverage intervention.

The next route-deciding question, if work resumes, should be whether the
smallest explicit object-specific commit/coverage intervention can causally
redistribute native next-object probability away from a committed object and
toward still-uncovered objects. It should separately control generic prefix
depth/row-position, use no slots or external detector initially, and stop after
a bounded oracle intervention before any learned architecture is proposed.

Promotion decision: `not_promoted`. The unit is complete and no architecture
is authorized by this result.

Observed: later prefixes strengthen the target write, but they strengthen the
stable nonselected same-class control at least as consistently; the frozen
label is `mixed_cross_prefix_result_inconclusive`.

Supported: a generic prefix-history, depth, or trajectory-phase modulation of
the native one-step spatial commit primitive is more consistent with this panel
than a target-specific current-selection gate.

Next decider: before any architecture proposal, use a bounded oracle
object-specific coverage intervention to test whether native next-object
probability redistributes from a committed object toward still-uncovered
objects while controlling generic prefix depth and row position.
