---
title: PVCI Endogenous Commit Write-Read Bridge
description: Compares raw generated-row commits with canonicalized same-object commits to locate whether free-rollout failure is in writing or reading commit state.
type: idea
role: research-unit
authority: non_normative_research
promotion_status: not_promoted
unit_id: 2026-07-11-pvci-endogenous-commit-write-read-bridge
topic: qwen3-vl-painted-gt-transcription-probe
status: complete
tags:
  - coordexp-swift
  - research-unit
  - qwen3-vl
  - endogenous-prefix
  - commit-write-read
  - causal-counterfactual
updated: 2026-07-11
---

# PVCI Endogenous Commit Write-Read Bridge

## Question

When Qwen3-VL generates a valid, strictly attributable row for an
annotated-uncovered object, are its self-generated coordinates sufficient to
write a readable same-object commit, or does that commit appear only after the
coordinates are replaced by the canonical GT coordinates for the same object?

This is the smallest bridge between the completed teacher-forced canonical
result and native rollout. It does not ask the model to choose a new
architecture. It asks whether the existing decoder can read its own valid
geometry write, once object selection is held fixed by hindsight attribution.

This is deliberately narrower than a general parser-normalization test. In the
frozen ten matched rows, the generated class phrase and compact-row structure
already equal the canonical row; RAW and SNAP differ only in the four coordinate
tokens. The planner must re-attest that factor collapse and abort if it does not
hold. The experiment therefore localizes a geometry-write/read bottleneck, not
arbitrary phrase or serialization noise.

## Decision Relevance

- If raw and canonicalized coordinates create the same object-specific
  suppression, the geometry write/read path is already present; the primary
  bottleneck moves upstream to selection, phrase-coordinate binding, or
  localization quality.
- If canonicalized coordinates suppress but raw coordinates do not, the
  highest-leverage intervention is training the native prefix on realistic
  coordinate deviations or writing a minimal geometry-normalized commit, not
  slots or a new memory architecture.
- If even canonicalized same-object rows do not suppress, the native prefix
  read/ledger hypothesis fails for this off-canonical panel; order-policy and
  valid-position controls then decide whether an explicit state is justified.

## Frozen Source Panel

Source artifacts:

`/data/CoordExp/outputs/painted_gt/pvci_native_canonical_commit_depth/full5_v1/events.jsonl`

Freeze all RP1.0 one-row continuations after C0, C1, and C2: exactly 15 source
attempts (`5 images x 3 depths`). Do not select examples by downstream scores.

The existing strict attribution contract yields the following immutable map.
Planning must recompute it and fail on any disagreement rather than silently
redefining the matched subset:

| Source event | Depth | RP1.0 attribution | J |
|---|---:|---|---:|
| `line-000009-row-000` | C0 | strict annotated-uncovered | 1 |
| `line-000009-row-000` | C1 | strict annotated-uncovered | 4 |
| `line-000009-row-000` | C2 | strict annotated-uncovered | 4 |
| `line-000010-row-000` | C0 | unmatched unknown | - |
| `line-000010-row-000` | C1 | unmatched unknown | - |
| `line-000010-row-000` | C2 | unmatched unknown | - |
| `line-000012-row-000` | C0 | unmatched unknown | - |
| `line-000012-row-000` | C1 | strict annotated-uncovered | 2 |
| `line-000012-row-000` | C2 | strict annotated-uncovered | 4 |
| `line-000016-row-000` | C0 | strict annotated-uncovered | 1 |
| `line-000016-row-000` | C1 | unmatched unknown | - |
| `line-000016-row-000` | C2 | strict annotated-uncovered | 4 |
| `line-000017-row-000` | C0 | strict annotated-uncovered | 1 |
| `line-000017-row-000` | C1 | strict annotated-uncovered | 3 |
| `line-000017-row-000` | C2 | strict annotated-uncovered | 3 |

Therefore the panel contains:

- 10 strict unique matches to annotated-uncovered objects;
- 5 `unmatched_unknown` rows.

The 5 unknown rows remain explicit pipeline failures and are never treated as
background, hallucination, or successful avoidance. The conditional write/read
factorial is defined only for the 10 strict matches because only there is the
committed object identity observationally identified.

The ten matched attempts come from four images and eight unique `(image, J)`
clusters. Attempt counts are descriptive, not independent trials. Every terminal
label must also pass the frozen cluster- and image-robustness gates below.

## Prefix Factorial

For each strict match `J` from canonical pre-prefix `P_t`:

```text
PRE  = P_t
RAW  = P_t + exact source-generated token IDs
SNAP = P_t + canonical GT row tokens for the same strict-matched object J
```

`RAW` and `SNAP` commit the same attributed object and occupy the same generated
slot. In this frozen panel their phrase and structure tokens are identical; only
their four coordinate tokens differ. Thus SNAP does not restore canonical
object order: J may be off-trajectory, and SNAP inserts that same J in the same
slot with canonical geometry. The source generated row, token IDs, decoded text,
and parse are immutable artifacts. Decoding and re-tokenizing may be recorded as
a receipt but must never replace the source token IDs. `SNAP` is a privileged
causal control, not a proposed inference dependency.

For each strict match define:

```text
U_pre  = annotated objects uncovered at PRE
J      = the sole strict match of the source-generated RAW row
U_post = U_pre ∖ {J}
S_J    = same-description objects in U_post
```

All ten frozen strict matches must have at least one member in `S_J`; otherwise
the object-specificity gate is ineligible and planning aborts.

At every branch score:

- `J_exact`, the canonical strict-matched object row;
- deterministic zero-coordinate-ID-overlap `J_near` when feasible;
- the exact raw generated row;
- canonical rows for every object in `U_pre`, with J singled out and all
  same-description controls in `S_J` identified;
- STOP as a separate one-token diagnostic.

Generate one additional row from RAW and SNAP under RP1.0 only. For this second
generation, the covered ledger is the canonical prefix plus J and the remaining
ledger is exactly `U_post`. Attribute into mutually exclusive outcomes:
`repeat_J`, `previously_covered_match`, `remaining_strict_match`, `STOP`,
`unmatched_unknown`, or `invalid`. Unknown, STOP, invalid, and repeat J are not
successful avoidance and J can never count as remaining.

## Primary Contrasts

```text
Commit_raw(J)  = log P(J | RAW)  - log P(J | PRE)
Commit_snap(J) = log P(J | SNAP) - log P(J | PRE)
Write_gap(J)   = Commit_raw(J) - Commit_snap(J)

Specificity_b(J) =
  Delta_b(J_exact) - median_{K in S_J} Delta_b(K_exact)
```

`Specificity_b(J) < 0` means the written row suppresses J more than other
still-uncovered objects with the same class phrase. This relative contrast is
required for an object-specific interpretation; absolute J suppression alone
is only generic list/position evidence.

Define an attempt-level readable commit as:

```text
Read_b(J) = [Commit_b(J) < 0] AND [Specificity_b(J) < 0]
ReadPair(J) = Read_raw(J) AND Read_snap(J)

Norm(J) =
  Read_snap(J)
  AND NOT Read_raw(J)
  AND Commit_snap(J) < Commit_raw(J)
  AND Specificity_snap(J) < Specificity_raw(J)
```

Compute exact and near sequence/span deltas. Raw direct-forward score is
primary. RP1.1 scoring may be reported secondarily but does not change the
frozen decision. Native model/generation stays bfloat16; readout/log-softmax is
float32; stable aggregation is Python double.

## Competing Hypotheses and Frozen Gates

All conditional counts use the 10 strict-matched source attempts as denominator;
the top-level report also retains the full `10/15` matchability rate. In
addition, repeated attempts are collapsed by median within the eight unique
`(image, J)` clusters and then by median within the four contributing images.
For every cluster, compute the median Commit and median Specificity separately;
`ClusterRead_b` is true only when both medians are negative. For every image,
take the median of its cluster-level Commit values and separately the median of
its cluster-level Specificity values; `ImageRead_b` is true only when both are
negative. Define `ClusterReadPair = ClusterRead_raw AND ClusterRead_snap` and
`ImageReadPair = ImageRead_raw AND ImageRead_snap`. Define `ClusterNorm` and
`ImageNorm` by applying the four conjuncts of `Norm` to those corresponding
aggregate values. A primary terminal label requires its executable aggregate
predicate below. Seven passing attempts concentrated in only two images are
ineligible for a terminal mechanism label.

### H1: Raw endogenous geometry commit is readable

- `Commit_raw(J) < 0` and `Commit_snap(J) < 0` for exact J in at least `7/10`;
- `Specificity_raw(J) < 0` and `Specificity_snap(J) < 0` in at least `6/10`;
- `ReadPair(J)` holds in at least `6/10` attempts;
- both raw and snap near-J suppression in at least `6/10`, with at least `8/10`
  paired near availability;
- `ClusterReadPair` passes at least `6/8` unique clusters and `ImageReadPair`
  passes at least `3/4` contributing images;
- RAW and SNAP free continuations each repeat J in at most `2/10` and each
  strictly select an object in `U_post` in at least `5/10`.

Outcome: preserve native prefix/KV geometry commit as the working prior;
intervene first in object selection, row binding, or localization, not memory.

### H2: Geometry-write normalization bottleneck

- SNAP exact suppression in at least `7/10`;
- SNAP object-specificity in at least `6/10`;
- `Read_snap(J)` in at least `6/10`;
- RAW exact suppression in at most `4/10`;
- RAW object-specificity in at most `4/10`;
- `Read_raw(J)` in at most `4/10`;
- `Norm(J)` in at least `6/10` paired attempts;
- `ClusterNorm` in at least `6/8` unique clusters and `ImageNorm` in at least
  `3/4` contributing images.

Outcome: train self-prefix coordinate deviations or a minimal
geometry-normalizing commit write; do not add a separate object ledger yet.

### H3: SNAP off-trajectory read failure

`Read_snap(J)` occurs in at most `4/10`, `ClusterRead_snap` is false in at
least `6/8` unique clusters, and `ImageRead_snap` is false in at least `3/4`
contributing images.
The canonical-prefix effect does not generalize to a hindsight-selected
off-trajectory identity even when its coordinates are canonicalized.

Outcome: preserve three live explanations: failure to read an off-trajectory
commit, rejection by the geo-sorted order prior, or insufficient same-slot
serialization. An order-diversified/valid-position control is required before
explicit memory receives a strong posterior increase. Architecture remains
unpromoted.

### Locality modifier

Locality is orthogonal to the primary mechanism label:

- `near_geometry_generalized`: corresponding RAW/SNAP near-J suppression reaches
  `6/10` with at least `8/10` paired availability;
- `coordinate_local`: a primary exact+specificity mechanism gate passes but the
  corresponding near-J suppression is below `6/10` despite at least `8/10`
  availability;
- `near_unavailable`: fewer than `8/10` paired near controls are feasible.
- `near_inconclusive`: paired near controls are available, but neither the
  generalized-near gate nor a primary-mechanism-plus-near-failure gate passes.

`coordinate_local` prioritizes spatial commit-field/cursor fidelity and blocks
any broad object-state wording, but it does not shadow the RAW-vs-SNAP primary
diagnosis.

Primary-label precedence is `H3 > H2 > H1 > mixed`; the locality modifier is
reported separately.

## Evidence Gates

- Materialize all 15 source attempts before model loading; preserve all unknown
  and failed rows.
- Recompute strict attribution from the source continuation receipt and source
  ledger; never trust a shallow `matched` label.
- Hash source events, unit, planner, scorer, analyzer, helpers, config,
  checkpoint, Git state, and argv.
- Require exact RAW source token IDs, decoded-text and re-encoding receipts,
  canonical SNAP tokenization receipts, and attestation that phrase/structure
  tokens are identical while only coordinate tokens differ.
- Freeze and attest the exact 15-attempt attribution map, including every J;
  recomputation disagreement is a hard contract failure.
- Repeat candidate scoring at every PRE/RAW/SNAP boundary and require exact
  per-token determinism.
- Require the same candidate universe at PRE/RAW/SNAP and fail if any frozen
  same-description control is missing.
- Two-source-attempt GPU gate first; full 15 only after contract GO.
- Any partial run is ineligible for a final mechanism label.

## Stop Condition

Stop after one passing two-attempt runtime gate and the exact frozen 15-attempt
panel, or immediately on contract failure. Return one bounded primary label:
`raw_geometry_commit_readable`, `geometry_write_normalization`,
`snap_offtrajectory_read_failure`, or `mixed`, plus one locality modifier:
`near_geometry_generalized`, `coordinate_local`, `near_unavailable`, or
`near_inconclusive`. Every label is
conditional on the frozen 10/15 identity-attributable subset. No architecture
is promoted.

## Planned Artifacts

- plan: `/data/CoordExp/outputs/painted_gt/pvci_endogenous_commit_write_read/plan_step4887/`;
- scorer: `/data/CoordExp/outputs/painted_gt/pvci_endogenous_commit_write_read/full15/`;
- analysis: scorer root plus `/analysis/`.

## Research Unit Closeout

Observed:

- the frozen full panel executed `15/15` attempts with zero scorer or analyzer
  failures: `10` strictly attributable/scored rows plus `5` retained
  `unmatched_unknown` source rows;
- RAW and SNAP each suppress `J_exact` in `10/10`, are more suppressive for J
  than the median still-uncovered same-description controls in `10/10`, and
  satisfy `ReadPair` in `10/10`;
- paired near controls are available in `10/10`; RAW and SNAP both suppress the
  zero-coordinate-ID-overlap `J_near` candidate in `9/10`;
- the paired readable direction holds in `8/8` unique `(image, J)` clusters and
  `4/4` contributing images;
- `Norm` is `0/10`: canonical coordinates do not repair a RAW failure because
  RAW is already readable throughout the conditional panel;
- RAW and SNAP each produce `7/10` strict `U_post` continuations, `3/10`
  unmatched continuations, and `0/10` strict repeats of J. Their coarse outcome
  class agrees in all ten attempts;
- the raw-view median exact J delta is `-5.5174` for RAW and `-4.7441` for SNAP;
  median same-class specificity is `-8.2185` and `-8.3700`; median near-J delta
  is `-5.6449` and `-4.5919`. Phrase and structure contributions are tiny; the
  effect is coordinate-dominated;
- the frozen primary verdict is `raw_geometry_commit_readable` with locality
  modifier `near_geometry_generalized`; H1 passes, H2/H3 do not.

Supported: under canonical pre-prefixes, for the frozen `10/15` source
continuations that already produce a syntactically valid row uniquely
attributable to an annotated-uncovered object, the model's own generated
coordinates are sufficient to write a readable, object-relative, one-step
spatial commit into the native prefix/KV route. Exact coordinate-token
repetition is not the whole explanation because a nearby candidate with zero
shared coordinate IDs is also suppressed in `9/10`.

Not supported: a semantic object-identity ledger independent of coordinates;
behavior for the five unidentifiable source rows; long-horizon endogenous
composition; order-free coverage; complete enumeration; phrase-coordinate
binding under a wrong phrase; STOP calibration; dense-scene generalization; or
detection-metric improvement. The bounded mechanism can still be a generic
coordinate/list anti-repetition field rather than visually grounded object
memory.

Next decider: a same-image source-swap factorial that writes (a) true J
geometry, (b) J-near displaced geometry, (c) verified background geometry, and
(d) another uncovered same-class K geometry, then measures whether suppression
and continuation follow arbitrary written coordinates or are strengthened by
real visual object support.

Evidence scope:

- plan: `/data/CoordExp/outputs/painted_gt/pvci_endogenous_commit_write_read/plan_step4887/`;
- runtime gate: `/data/CoordExp/outputs/painted_gt/pvci_endogenous_commit_write_read/gate2_v1/`;
- full scorer: `/data/CoordExp/outputs/painted_gt/pvci_endogenous_commit_write_read/full15_v1/`;
- canonical analysis: `/data/CoordExp/outputs/painted_gt/pvci_endogenous_commit_write_read/full15_v1/analysis_v1/`;
- scorer events SHA256:
  `d8fb93bf3847e71585a8cc6b0de144c1ab117306c2a1522e2e88829882910bd1`;
- execution receipt SHA256:
  `24509bce7ebb8b51db7838b411b46e9d9032a38e9175083bb940a798917fe39f`;
- scorer summary SHA256:
  `5cc58ff1d0f014e7f6e09c0ec5927ff0d863b4de5ce3f218b66efd2b6d381361`.

Promotion decision: `not_promoted`. The unit closes with an architecture
posterior update—reuse native prefix/KV commit as the current simplest prior and
defer slots or an explicit ledger—but no final architecture is selected.
