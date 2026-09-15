---
title: Paired Natural-Terminal Native versus Forced-Opener One-Row Release Results
description: Completed current-runtime causal comparison showing 44 verified new-owner recoveries from opener forcing at 200 Source natural stops, alongside dominant unmatched-row failure.
type: investigation
role: research-result
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: complete
unit_id: 2026-07-25-paired-natural-terminal-forced-opener-release
topic: qwen3-vl-dense-enumeration
status: complete_ready_for_user_discussion
evidence_status: verified_paired_causal_release
updated: 2026-07-25
---

# Paired Natural-Terminal Native versus Forced-Opener One-Row Release Results

## Decision

Forcing only the canonical new-row opener reveals a real and nontrivial pool of
true positive objects behind Source's natural stopping decision.

On the fixed 200-boundary panel, the current-runtime native arm recovers a
verified uncovered physical owner on 1 boundary. The forced-opener arm recovers
one on 45 boundaries. The paired primary outcome is therefore:

- `44/200` boundaries with a verified uncovered owner produced only after
  forcing the opener;
- `44` causally gained physical owners and zero paired lost owners; and
- a descriptive Wilson interval of `[16.82%, 28.24%]` for the `22.0%` panel
  fraction.

This answers the motivating question positively: when Source is forced to
continue after a real stop, it finds a previously missed true positive with
substantial frequency in this panel. The effect is not merely repetition or
malformed output.

The same experiment also shows why indiscriminate continuation is not a final
solution. Forced continuation produces:

| Forced-opener one-row outcome | Boundaries |
| --- | ---: |
| Verified uncovered physical owner | 45 |
| Covered-owner repeat | 19 |
| Valid row without a strict owner match | 134 |
| Invalid geometry | 2 |

Thus `198/200` forced arms realize a syntactically complete valid row, but only
`45/200` realize a verified uncovered owner. The largest remaining failure is
conditional owner grounding, not row grammar. Continuation is a real causal
bottleneck, but it is only one stage of set completion.

No training, prompt change, architecture change, or longer forced trajectory
was run.

## Exact Causal Contrast

Every pair uses the same Source checkpoint, image, original `list all objects`
prompt, literal historical assistant prefix, full-model 32-bit floating point
HF runtime, scaled dot-product attention, physical batch size one, greedy
decode, parser, entity ledger, and owner matcher.

The native arm generates at most one row from the boundary. The treatment arm
adds exactly token `151646` (`<|object_ref_start|>`) to the assistant
continuation and releases the rest of at most one row. No owner description,
coordinate, covered-set text, or alternate user prompt is supplied.

For boundary `i`, the primary success is a member of the frozen remaining-owner
set produced by the forced arm but not the paired native arm. This avoids
selecting an arbitrary intended owner at the 145 boundaries with multiple
remaining owners.

## Runtime-Control Finding

The paired native control is conclusion-bearing. It reproduces historical
terminal behavior on 198/200 boundaries. The two exceptions are exactly the
two boundaries that the preceding FP32 diagnostic had identified as having a
small positive opener-minus-terminal margin despite a historical stop:

- image `235809`: native and forced both emit the identical covered-owner
  repeat;
- image `484369`: native and forced both emit the identical verified uncovered
  owner.

In both cases, raw row token IDs are exactly equal across native and forced
arms. Neither counts as a causal gain. Comparing only a new forced run with the
historical EOS would have falsely attributed both current-runtime continuations
to forcing.

## Paired Owner Outcomes

| Native any uncovered owner | Forced any uncovered owner | Boundaries |
| --- | --- | ---: |
| No | No | 155 |
| No | Yes | 44 |
| Yes | Yes | 1 |
| Yes | No | 0 |

Each successful one-row release contains one strict owner match, so the 44
primary boundary successes correspond to 44 distinct causal owner gains. The
per-boundary owner-net mean is `+0.22`, median `0`, and range `0--1`; its
within-panel bootstrap mean interval is `[+0.165, +0.280]`.

Zero paired losses do not establish preservation safety. Native terminates on
198 boundaries and therefore offers no later row to lose. This experiment owns
only the immediate one-row causal effect, not downstream trajectory utility or
final set coverage.

## Match-Quality Audit

All 44 causal gains were reconstructed from raw entity-match evidence and
verified against the frozen remaining-owner IDs. Their geometry is not
concentrated at the strict threshold:

| Match statistic | Value |
| --- | ---: |
| Mean Intersection over Union | 0.813 |
| Median Intersection over Union | 0.848 |
| Minimum Intersection over Union | 0.525 |
| Mean normalized center distance | 0.0185 |
| Median normalized center distance | 0.0054 |
| Gains retained at Intersection over Union at least 0.55 | 43/44 |
| Gains retained at Intersection over Union at least 0.60 | 41/44 |

The gains cover 30 categories. The most frequent are `person` 5, `book` 3,
`chair` 3, then several categories with one or two gains. The result is not a
single-category artifact.

Direct inspection of the six lowest-Intersection-over-Union or largest-center-
distance examples found category-consistent visible objects and geometry
consistent with their frozen annotations. These include a baseball glove,
potted plant, carrot, person, handbag, and baseball bat. The shared standard
visualization API was not used because it accepts canonical scored-inference
artifacts rather than these causal receipts; no unsupported renderer fallback
was added.

## What the 134 Unmatched Rows Mean

All 134 are accepted canonical rows, and their entity-match status is
`unmatched`, not `ambiguous`. Eighty-seven have at least one same-category
candidate in the trusted ledger. Only 13 of those reach same-category
Intersection over Union `0.30`, and none reaches `0.50`.

Therefore the unmatched majority is not a collection of strict true positives
barely below the declared threshold. It mixes localization failures, wrong
instances or categories, and possibly visually valid objects outside the
trusted frozen ledger. It must not be called either verified recovery or pure
hallucination without a separate entity-and-geometry audit.

The two invalid cases are complete textual rows rejected for degenerate box
ordering:

- image `154435`: carrot box with `x1 = x2 = 0`;
- image `46997`: dining-table box with `y1 = y2 = 999`.

## Diagnostic Margin and Recovery Are Related but Distinct

The preceding checkpoint scores create perfectly nested positive-margin sets.
Partitioning the Source forced-release result by the first checkpoint whose
diagnostic margin is positive gives:

| First positive diagnostic band | Boundaries | Causal gains | Rate |
| --- | ---: | ---: | ---: |
| Source already positive | 2 | 0 | 0.0% |
| Transition positive beyond Source | 43 | 16 | 37.2% |
| Pairwise positive beyond transition | 49 | 11 | 22.4% |
| Owner-conditioned positive beyond pairwise | 51 | 11 | 21.6% |
| All four checkpoints nonpositive | 55 | 6 | 10.9% |

Transition-positive states are enriched for Source forced recovery, but the
relationship is neither necessary nor sufficient. Twenty-seven of the 43
transition-only crossings do not yield a new owner after forcing, while six
boundaries remain diagnostic-negative under every checkpoint yet do yield a
verified owner when Source is forced to continue.

This separates two questions:

1. whether the policy chooses to start another row; and
2. which physical owner, if any, the conditional row policy realizes after
   that choice is fixed.

Training can move the first decision without guaranteeing the second. Larger
continuation margins are not a monotonic proxy for conditional owner quality.

## Panel Strata

The causal success rate decreases strongly with prefix depth:

| Prefix depth | Boundaries | Causal gains | Rate |
| --- | ---: | ---: | ---: |
| 1--2 | 52 | 21 | 40.4% |
| 3--5 | 53 | 12 | 22.6% |
| 6--9 | 51 | 6 | 11.8% |
| 10 or more | 44 | 5 | 11.4% |

Sparse images yield `16/41` gains, medium and dense images each yield `14/74`,
and the 11 very-dense images yield none. These are descriptive associations in
a depth-first deterministic panel; prefix depth, scene density, stop strength,
and remaining-owner accessibility are confounded and cannot be assigned a
causal role here.

Recovery is not monotonic in the number of remaining owners: `16/55` with one,
`11/67` with two or three, `13/48` with four to seven, and `4/30` with eight or
more. This remains inconsistent with a simple count-like unfinished-set signal.

## Mechanism Judgment

The evidence supports a two-stage bottleneck:

1. **Premature stopping is causally real.** At 44 boundaries, Source already
   contains a conditional row trajectory to a verified missed owner, but the
   native next action prevents that trajectory from being entered.
2. **Conditional owner grounding remains the larger failure.** Once entry is
   forced, most boundaries emit a well-formed row that does not strictly match
   a trusted owner. Lowering stop probability alone would therefore trade some
   real gains for many unmatched rows and repeats.

This reconciles the earlier observations. Step 36 can improve final owner
coverage by moving a useful subset of premature stops, while stronger full-row
arms can fail through broad continuation and output expansion. The result does
not show that transition training improves post-continue owner selection; only
Source was causally released in this unit.

## Supported, Unresolved, and Not Claimed

### Supported

- Forced continuation reveals a verified owner missed by the paired native arm
  at 44/200 actual Source stops.
- The effect spans categories and survives stricter geometry thresholds.
- Current-runtime native replay is necessary for correct attribution.
- Continue/stop susceptibility and conditional owner realization are related
  but distinct.
- Broad continuation without owner selection would produce mostly unmatched
  rows, repeats, or invalid geometry rather than uniform set expansion.

### Unresolved

- Whether transition step 36 improves conditional owner realization after the
  opener is held fixed on the same 200 boundaries.
- Which of the 134 unmatched rows are localization failures, wrong owners,
  unsupported objects, or valid objects absent from the trusted ledger.
- Whether a recovered first row preserves or improves later free-rollout owner
  coverage and natural termination.
- Whether the deterministic 200-boundary panel represents other Source stops.

### Not claimed

- No production forced-continuation policy, gate-only training objective,
  architecture, explicit covered-set state, prompt change, or final free-
  rollout set gain is promoted.

## Verified Artifacts

The immutable artifact root is:

`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-25-paired-natural-terminal-forced-opener-release/`

Key products are:

- smoke `smoke-v1/source.json`, SHA-256
  `12e50b7b1c3718a7292d6a4149d30b529cf1cd600748a12322cc159f35947296`;
- deterministic repeat `smoke-v2/source-repeat.json`, SHA-256
  `4e696841a99a82239a7612ba670740890cc64b0e6d3ddb83bc30b6c5f2f06d5c`;
- eight production receipts under `production-v1/`, with exactly 200 unique
  boundary IDs; and
- final `reduction-v4/summary.json`, SHA-256
  `e14bb7b16bcc01f6fc7a3c48bbdb79c868e57afa77fe217dcbc71c38085f7550`,
  plus `reduction-v4/cases.jsonl`, SHA-256
  `135c7ddfd8414cc8e121efa07e304b6dc742257456d0e91f29d6e655cfbad8d3`.

## Stop

The paired Source causal release and audit are complete. Stop for user
discussion before any checkpoint comparison, longer trajectory, training, or
architecture work.
